#include "Dialect/Tla/IR/TlaAttrs.h"
#include "PassesCommon.h"
#include "PassesInternal.h"
#include "Passes/TlaTensorToMemref.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"

// tla-cube-region: lowers the cube (AIC) compute ops (tla.copy / tla.mmad) via
// the shared !tla.tensor->memref lowering (TlaTensorMemrefLowering), then flattens
// the tla.cube region. Runs after tla-vector-region, before tla-finalize-memref.
//
// Flow: collect materialized descriptors, then lower tla.copy (descriptor +
// payload driven) and tla.mmad, both reconstructing tile memrefs directly from
// tla.tensor_desc. Descriptors are available for every copy, so copy lowering is
// a single descriptor-driven path. Finalize DCEs dead scaffolding and unrealized
// casts.

namespace tla {
namespace {

  struct LowerTlaMmadPattern : public OpRewritePattern<::tla::MmadOp> {
    LowerTlaMmadPattern(MLIRContext *ctx,
                        DenseMap<Value, TensorDescriptor> &tensorDescriptorByValue,
                        SmallVectorImpl<Operation *> &toErase,
                        DenseMap<Value, Value> &loweredMemrefByValue)
        : OpRewritePattern<::tla::MmadOp>(ctx),
          tensorDescriptorByValue(tensorDescriptorByValue), toErase(toErase),
          loweredMemrefByValue(loweredMemrefByValue) {}

    LogicalResult matchAndRewrite(::tla::MmadOp op, PatternRewriter &rewriter) const override {
      if (op->getNumOperands() < 3)
        return success();

      Value acc = op->getOperand(0);
      Value lhs = op->getOperand(1);
      Value rhs = op->getOperand(2);
      Type accType = acc.getType();
      Type lhsType = lhs.getType();
      Type rhsType = rhs.getType();

      Value initC = op->getOperand(3);
      Value unitFlag = op->getOperand(4);

      auto i1Type = rewriter.getI1Type();
      auto i64Type = rewriter.getI64Type();
      auto i8Type = rewriter.getI8Type();

      Value initCVal = initC;
      Value unitFlagVal = rewriter.create<arith::TruncIOp>(op.getLoc(), i8Type, unitFlag);

      auto lhsInfo = ::tla::decodeTileTypeInfo(lhsType);
      auto rhsInfo = ::tla::decodeTileTypeInfo(rhsType);
      auto accInfo = ::tla::decodeTileTypeInfo(accType);
      if (failed(lhsInfo) || failed(rhsInfo) || failed(accInfo)) {
        op.emitError() << "tla.mmad currently requires structured tla.tensor operand types";
        return failure();
      }
      if (lhsInfo->rank != 2 || rhsInfo->rank != 2 || accInfo->rank != 2) {
        op.emitError() << "tla.mmad currently supports rank-2 tiles only";
        return failure();
      }
      if (lhsInfo->addressSpace != "l0a" || rhsInfo->addressSpace != "l0b" ||
          accInfo->addressSpace != "l0c") {
        op.emitError()
            << "unsupported tla.mmad tile addrspaces; expected acc in l0c, lhs in l0a, rhs in l0b";
        return failure();
      }
      bool supportedF16Route = lhsInfo->elementType == "f16" && rhsInfo->elementType == "f16" &&
                               accInfo->elementType == "f32";
      bool supportedBf16Route = lhsInfo->elementType == "bf16" && rhsInfo->elementType == "bf16" &&
                                accInfo->elementType == "f32";
      bool supportedF32Route = lhsInfo->elementType == "f32" && rhsInfo->elementType == "f32" &&
                               accInfo->elementType == "f32";
      if (!supportedF16Route && !supportedBf16Route && !supportedF32Route) {
        op.emitError() << "unsupported tla.mmad element types; expected f16,f16 -> f32, bf16,bf16 "
                          "-> f32, or f32,f32 -> f32 (L0C accumulator is fp32)";
        return failure();
      }

      auto maybeStaticShapeCheck = [&](int64_t lhsM, int64_t lhsK, int64_t rhsK, int64_t rhsN,
                                       int64_t accM, int64_t accN) -> LogicalResult {
        if (lhsM == ShapedType::kDynamic || lhsK == ShapedType::kDynamic ||
            rhsK == ShapedType::kDynamic || rhsN == ShapedType::kDynamic ||
            accM == ShapedType::kDynamic || accN == ShapedType::kDynamic) {
          return success();
        }
        if (lhsK != rhsK || lhsM != accM || rhsN != accN) {
          op.emitError() << "unsupported tla.mmad tile shape contract; expected lhs(MxK), "
                            "rhs(KxN), acc(MxN)";
          return failure();
        }
        return success();
      };
      if (failed(maybeStaticShapeCheck(lhsInfo->originShapeDims[0], lhsInfo->originShapeDims[1],
                                       rhsInfo->originShapeDims[0], rhsInfo->originShapeDims[1],
                                       accInfo->originShapeDims[0], accInfo->originShapeDims[1])))
        return failure();
      if (accInfo->layoutTag != TensorLayoutTag::L0C || lhsInfo->layoutTag != TensorLayoutTag::zN ||
          rhsInfo->layoutTag != TensorLayoutTag::nZ) {
        op.emitError()
            << "unsupported tla.mmad operand layout; expected acc L0Clayout, lhs zN, rhs nZ";
        return failure();
      }

      // Materialize each tile operand's memref directly from its tla.tensor_desc
      // descriptor (shared with the tla.copy path).
      auto materializeTensorOperand = [&](Value tensor) -> FailureOr<Value> {
        auto it = tensorDescriptorByValue.find(tensor);
        if (it == tensorDescriptorByValue.end()) {
          op.emitError() << "missing descriptor for tla.mmad tile operand";
          return failure();
        }
        return ::tla::materializeTileMemrefFromDescriptor(
            rewriter, op.getLoc(), it->second, op.getOperation(), loweredMemrefByValue);
      };

      FailureOr<Value> lhsMemref = materializeTensorOperand(lhs);
      FailureOr<Value> rhsMemref = materializeTensorOperand(rhs);
      FailureOr<Value> accMemref = materializeTensorOperand(acc);
      if (failed(lhsMemref) || failed(rhsMemref) || failed(accMemref)) {
        op.emitError() << "failed to bridge tla.mmad operands to memref values";
        return failure();
      }

      // Match the tla.copy runtime ABI: pass dynamic strided memrefs to the C stub
      // (same as buildRuntimeMemref in LowerTlaCopyPattern).
      auto toRuntimeMemref = [&](Value v) -> FailureOr<Value> {
        auto baseType = dyn_cast<MemRefType>(v.getType());
        if (!baseType) {
          op.emitError() << "tla.mmad memref operand must have memref type";
          return failure();
        }
        MemRefType runtimeType = ::tla::getDynamicStridedMemrefType(baseType);
        return ::tla::castMemrefToType(rewriter, op.getLoc(), v, runtimeType);
      };
      FailureOr<Value> lhsRuntime = toRuntimeMemref(*lhsMemref);
      FailureOr<Value> rhsRuntime = toRuntimeMemref(*rhsMemref);
      FailureOr<Value> accRuntime = toRuntimeMemref(*accMemref);
      if (failed(lhsRuntime) || failed(rhsRuntime) || failed(accRuntime))
        return failure();

      auto materializeIndexDim = [&](Value tensor, int64_t staticOriginDim, StringRef fieldName,
                                     bool takeSecondDim) -> FailureOr<Value> {
        auto it = tensorDescriptorByValue.find(tensor);
        if (it != tensorDescriptorByValue.end()) {
          Value dim = takeSecondDim ? it->second.originShape1 : it->second.originShape0;
          if (dim && dim.getType().isIndex())
            return dim;
        }
        if (staticOriginDim == ShapedType::kDynamic) {
          op.emitError() << "tla.mmad requires " << fieldName
                         << " from tensor descriptor SSA when type origin_shape is dynamic";
          return failure();
        }
        return rewriter.create<arith::ConstantIndexOp>(op.getLoc(), staticOriginDim).getResult();
      };
      FailureOr<Value> mIndex = materializeIndexDim(lhs, lhsInfo->originShapeDims[0], "M", false);
      FailureOr<Value> kIndex = materializeIndexDim(lhs, lhsInfo->originShapeDims[1], "K", true);
      FailureOr<Value> nIndex = materializeIndexDim(rhs, rhsInfo->originShapeDims[1], "N", true);
      if (failed(mIndex) || failed(kIndex) || failed(nIndex))
        return failure();

      auto castIndexToI64 = [&](Value v) -> Value {
        return rewriter.create<arith::IndexCastOp>(op.getLoc(), i64Type, v).getResult();
      };
      Value mI64 = castIndexToI64(*mIndex);
      Value kI64 = castIndexToI64(*kIndex);
      Value nI64 = castIndexToI64(*nIndex);

      SmallVector<Type, 8> operandTypes = {(*lhsRuntime).getType(),
                                           (*rhsRuntime).getType(),
                                           (*accRuntime).getType(),
                                           i64Type,
                                           i64Type,
                                           i64Type,
                                           i1Type,
                                           i8Type};
      StringRef calleeName = supportedF16Route    ? "mmad_half_half_float"
                             : supportedBf16Route ? "mmad_bf16_bf16_float"
                                                  : "mmad_float_float_float";
      auto callee =
          ::tla::getOrCreateRuntimeCall(op->getParentOfType<ModuleOp>(), calleeName, operandTypes);
      SmallVector<Value, 8> operands = {
          *lhsRuntime, *rhsRuntime, *accRuntime,           mI64,
          nI64,        kI64,        initCVal,              unitFlagVal
      };
      rewriter.create<func::CallOp>(op.getLoc(), callee, operands);
      toErase.push_back(op.getOperation());
      return success();
    }

  private:
    DenseMap<Value, TensorDescriptor> &tensorDescriptorByValue;
    SmallVectorImpl<Operation *> &toErase;
    DenseMap<Value, Value> &loweredMemrefByValue;
  };

  struct LowerTlaCopyPattern : public OpRewritePattern<::tla::CopyOp> {
    LowerTlaCopyPattern(MLIRContext *ctx,
                        DenseMap<Value, TensorDescriptor> &tensorDescriptorByValue,
                        SmallVectorImpl<Operation *> &toErase,
                        DenseMap<Value, Value> &loweredMemrefByValue)
        : OpRewritePattern<::tla::CopyOp>(ctx),
          tensorDescriptorByValue(tensorDescriptorByValue), toErase(toErase),
          loweredMemrefByValue(loweredMemrefByValue) {}

    LogicalResult matchAndRewrite(::tla::CopyOp op, PatternRewriter &rewriter) const override {
      if ((op->getNumOperands() != 2 && op->getNumOperands() != 3) || op->getNumResults() != 0) {
        op.emitError() << "expected tla.copy to have 2 or 3 operands and 0 results";
        return failure();
      }

      Value dstTile = op->getOperand(0);
      Value srcTile = op->getOperand(1);
      auto dstIt = tensorDescriptorByValue.find(dstTile);
      auto srcIt = tensorDescriptorByValue.find(srcTile);
      if (dstIt == tensorDescriptorByValue.end()) {
        op.emitError() << "missing descriptor for tla.copy dst tile; expected a tla.tensor_desc "
                          "operand materialized by tla-lower-tensor-desc";
        return failure();
      }
      if (srcIt == tensorDescriptorByValue.end()) {
        op.emitError() << "missing descriptor for tla.copy src tile; expected a tla.tensor_desc "
                          "operand materialized by tla-lower-tensor-desc";
        return failure();
      }

      const TensorDescriptor &dstDesc = dstIt->second;
      const TensorDescriptor &srcDesc = srcIt->second;
      if (!::tla::validateTensorDescriptorV1(
              op, dstDesc, "malformed descriptor for tla.copy dst tile operand",
              /*requireShapeOperands=*/true)) {
        return failure();
      }
      if (!::tla::validateTensorDescriptorV1(
              op, srcDesc, "malformed descriptor for tla.copy src tile operand",
              /*requireShapeOperands=*/true)) {
        return failure();
      }
      StringRef srcAddrspace = srcDesc.addrspace;
      StringRef dstAddrspace = dstDesc.addrspace;
      std::string src2Dst = std::string(srcDesc.addrspace) + "2" + std::string(dstAddrspace);
      if (srcAddrspace == "l0c") {
        if (op->getNumOperands() != 3) {
          op.emitError() << "expected tla.copy " << src2Dst << " has 3 operands";
          return failure();
        }
      } else if (op->getNumOperands() != 2) {
        op.emitError() << "expected tla.copy " << src2Dst << " has 2 operands";
        return failure();
      }
      bool rankOk = dstDesc.rank == srcDesc.rank;
      bool sameElem = dstDesc.elementType == srcDesc.elementType;
      auto buildRuntimeMemref = [&](const TensorDescriptor &desc) -> FailureOr<Value> {
        FailureOr<Value> baseMemref = ::tla::getOrMaterializeDescriptorBaseMemref(
            rewriter, op.getLoc(), desc, op.getOperation(), loweredMemrefByValue);
        if (failed(baseMemref))
          return failure();
        auto baseType = dyn_cast<MemRefType>((*baseMemref).getType());
        if (!baseType)
          return failure();
        MemRefType runtimeType = ::tla::getDynamicStridedMemrefType(baseType);
        return ::tla::castMemrefToType(rewriter, op.getLoc(), *baseMemref,
                                                   runtimeType);
      };

      StringRef extraDesc = "";
      struct L0C2DstInfo {
        uint8_t unitFlag = 0;
        bool relu_enable = false;
        QuantMode quantMode = QuantMode::NO_QUANT;
        L0C2UBMode l0c2UbMode = L0C2UBMode::NO_SPLIT_VEC_0;
        uint8_t subBlockId = 0;
      } l0c2DstInfo;
      if (srcAddrspace == "l0c") {
        auto params = op->getOperand(2);
        auto l0c2DstParamsOp = dyn_cast<::tla::CopyL0C2DstParamsOp>(params.getDefiningOp());
        if (!l0c2DstParamsOp) {
          op.emitError() << "expected tla.CopyL0C2DstParams as third operand";
          return failure();
        }
        l0c2DstInfo.unitFlag = static_cast<uint8_t>(l0c2DstParamsOp.getUnitFlag());
        l0c2DstInfo.relu_enable = l0c2DstParamsOp.getReluEnable();
        l0c2DstInfo.quantMode = l0c2DstParamsOp.getQuantMode().getQuantMode();
        if (dstAddrspace == "ub") {
          l0c2DstInfo.l0c2UbMode = l0c2DstParamsOp.getL0c2ubMode().getL0c2ubMode();
          StringRef splitMode = "nosplit";
          switch (l0c2DstInfo.l0c2UbMode) {
            case L0C2UBMode::NO_SPLIT_VEC_0:
              break;
            case L0C2UBMode::NO_SPLIT_VEC_1:
              l0c2DstInfo.subBlockId = 1;
              splitMode = "nosplit";
              break;
            case L0C2UBMode::SPLIT_M:
              splitMode = "splitm";
              break;
            case L0C2UBMode::SPLIT_N:
              splitMode = "splitn";
              break;
          }
          if ((l0c2DstInfo.l0c2UbMode==L0C2UBMode::SPLIT_M || l0c2DstInfo.l0c2UbMode==L0C2UBMode::SPLIT_N)
              && (srcDesc.elementType != dstDesc.elementType)) {
              op->emitError("When copy l0c to ub with split mode, src and dst type must be same");
              return failure();
          }
          extraDesc = splitMode;
        }
      }

      std::string calleeName = ::tla::getCopyRouteCallee(
          op.getContext(), srcAddrspace, dstAddrspace, srcDesc.layoutTag, dstDesc.layoutTag,
          srcDesc.elementType, dstDesc.elementType, extraDesc);
      if (!calleeName.empty()) {
        bool l0c2DstNarrow = srcAddrspace == "l0c" && (dstAddrspace == "gm" || dstAddrspace == "ub") &&
                           srcDesc.layoutTag == TensorLayoutTag::L0C &&
                           dstDesc.layoutTag == TensorLayoutTag::RowMajor &&
                           srcDesc.elementType == "f32" &&
                           (dstDesc.elementType == "f16" || dstDesc.elementType == "bf16");
        if (!rankOk || (!sameElem && !l0c2DstNarrow)) {
          op.emitError() << "tla.copy supported route has src/dst descriptor metadata mismatch "
                            "(rank/element type)";
          return failure();
        }

        FailureOr<Value> dstRuntimeMemref = buildRuntimeMemref(dstDesc);
        FailureOr<Value> srcRuntimeMemref = buildRuntimeMemref(srcDesc);
        if (failed(dstRuntimeMemref) || failed(srcRuntimeMemref))
          return failure();
        SmallVector<Value, 20> payload =
            ::tla::buildCopyPayloadForRoute(rewriter, op.getLoc(), srcDesc, dstDesc);
        SmallVector<Type, 22> operandTypes = {(*srcRuntimeMemref).getType(),
                                              (*dstRuntimeMemref).getType()};
        operandTypes.reserve(2 + payload.size());
        for (Value payloadValue : payload)
          operandTypes.push_back(payloadValue.getType());
        SmallVector<Value, 22> operands = {*srcRuntimeMemref, *dstRuntimeMemref};
        operands.append(payload.begin(), payload.end());
        if (srcAddrspace == "l0c") {
          auto i8Type = rewriter.getI8Type();
          auto unitFlagVal = rewriter.create<arith::ConstantIntOp>(op.getLoc(), l0c2DstInfo.unitFlag, 8);
          operandTypes.push_back(i8Type);
          operands.push_back(unitFlagVal);
          if (dstAddrspace == "ub") {
            auto subBlockIdVal = rewriter.create<arith::ConstantIntOp>(op.getLoc(), l0c2DstInfo.subBlockId, 8);
            operandTypes.push_back(i8Type);
            operands.push_back(subBlockIdVal);
          }
        }
        auto callee =
            ::tla::getOrCreateRuntimeCall(op->getParentOfType<ModuleOp>(), calleeName, operandTypes);
        
        // Enclose `copy` with atomic add and atomic none
        auto getAtomicKind = [](AtomicMode mode) -> hivm::AtomicKind {
          switch (mode) {
          case AtomicMode::add:
            return hivm::AtomicKind::ADD;
          // For further extension, add other atomic mode case
          default:
            return hivm::AtomicKind::NONE;
          }
        };

        auto atomicModeAttr = op->getAttrOfType<::tla::AtomicModeAttr>("atomic_mode");
        Type dstType = cast<MemRefType>((*dstRuntimeMemref).getType()).getElementType();
        bool _enable_atomic = atomicModeAttr && atomicModeAttr.getAtomicMode() != AtomicMode::none;
        if (_enable_atomic) {
          if (atomicModeAttr.getAtomicMode() != AtomicMode::add) {
            op.emitError() << "currently only atomic add is supported";
            return failure();
          }
          auto modeAttr = hivm::AtomicKindAttr::get(rewriter.getContext(), getAtomicKind(atomicModeAttr.getAtomicMode()));
          rewriter.create<hivm::SetAtomicOp>(op.getLoc(), modeAttr,
                                             mlir::TypeAttr::get(dstType));
        }
        rewriter.create<func::CallOp>(op.getLoc(), callee, operands);
        if (_enable_atomic) {
          auto modeAttr = hivm::AtomicKindAttr::get(rewriter.getContext(), hivm::AtomicKind::NONE);
          rewriter.create<hivm::SetAtomicOp>(op.getLoc(), modeAttr,
                                             mlir::TypeAttr::get(dstType));
        }
        toErase.push_back(op.getOperation());
        return success();
      }

      // GM<->UB row-major copies are owned by tla-vector-region's LowerCopyPattern
      // (UB is vector-core memory); the cube pass only lowers the cube-side routes
      // above (L1 / L0A / L0B / L0C).

      op.emitError() << "tla.copy descriptor/layout combination is unsupported: " << srcAddrspace
                     << "(" << ::tla::stringifyTensorLayoutTag(srcDesc.layoutTag)
                     << ") -> " << dstAddrspace << "("
                     << ::tla::stringifyTensorLayoutTag(dstDesc.layoutTag) << ")";
      return failure();
    }

  private:
    DenseMap<Value, TensorDescriptor> &tensorDescriptorByValue;
    SmallVectorImpl<Operation *> &toErase;
    DenseMap<Value, Value> &loweredMemrefByValue;
  };

// Flatten a tla.cube region by splicing its body into the parent block.
struct LowerTlaCubePattern : public OpRewritePattern<::tla::CubeOp> {
  using OpRewritePattern<::tla::CubeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(::tla::CubeOp op, PatternRewriter &rewriter) const override {
    if (op->getNumRegions() == 0 || op->getRegion(0).empty()) {
      rewriter.eraseOp(op);
      return success();
    }
    Block &body = op->getRegion(0).front();
    Block *parentBlock = op->getBlock();
    parentBlock->getOperations().splice(op->getIterator(), body.getOperations(), body.begin(),
                                        body.end());
    rewriter.eraseOp(op);
    return success();
  }
};


class TlaCubeRegionPass : public PassWrapper<TlaCubeRegionPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TlaCubeRegionPass)

  StringRef getArgument() const override { return "tla-cube-region"; }
  StringRef getName() const override { return "TlaCubeRegionPass"; }
  StringRef getDescription() const override {
    return "Lower tla.cube compute ops (tla.copy / tla.mmad) and flatten the cube region.";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect, mlir::memref::MemRefDialect,
                    scf::SCFDialect, hivm::HIVMDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Drive one function at a time so the cube lowering only touches the cube
    // (AIC) function's ops, not a sibling vector (AIV) function in a mixed kernel.
    // Mirrors tla-vector-region, which iterates the module's functions and filters
    // by core kind. Snapshot the functions up front: lowering appends runtime-call
    // declarations to the module, which must not be fed back through the loop.
    SmallVector<func::FuncOp, 4> funcOps(module.getOps<func::FuncOp>());
    for (func::FuncOp funcOp : funcOps) {
      if (funcOp.isDeclaration())
        continue;
      // Skip the generated vector_region helpers: they hold lowered AVE ops and
      // no cube work.
      if (funcOp->hasAttr(kHivmVectorFunctionAttrName))
        continue;
      // Only AIC (and not-yet-split MIX) functions hold cube work. Pure vector
      // (AIV) functions have no tla.cube / tla.copy / tla.mmad to lower.
      std::optional<HivmCoreKind> coreKind = getExpectedFunctionCoreKind(funcOp.getOperation());
      if (coreKind != HivmCoreKind::AIC && coreKind != HivmCoreKind::MIX)
        continue;
      if (failed(runOnCubeFunction(funcOp))) {
        signalPassFailure();
        return;
      }
    }
  }

  // Lower all cube compute ops within a single cube (AIC/MIX) function. The
  // lowering state (descriptors, memref cache, staged erases) is fresh per
  // function, matching tla-vector-region's per-function handoff. Only `root` (the
  // function) is threaded; the ModuleOp needed for runtime-symbol insertion is
  // derived on demand from the op being rewritten (getParentOfType<ModuleOp>()).
  LogicalResult runOnCubeFunction(func::FuncOp funcOp) {
    Operation *root = funcOp.getOperation();
    SmallVector<Operation *, 8> toErase;
    ::tla::TlaTensorMemrefLowering lowering;
    auto &tensorDescriptorByValue = lowering.descriptorByValue;
    // Set on a lowering failure that does not abort the remaining work; reported
    // once at the end.
    bool passFailed = false;

    // Read the descriptors materialized by tla-lower-tensor-desc. Cube lowering
    // must not reconstruct metadata from raw tensor producer chains.
    if (failed(::tla::collectMaterializedTensorDescriptors(funcOp, tensorDescriptorByValue)))
      return failure();

    // tla.tensor_ptr / tla.ptr_add were already folded into the inttoptr byte
    // address by tla-lower-ptr (run before tla-lower-tensor-desc), so each
    // tensor_desc.base here is the raw inttoptr boundary and the copy / subview
    // materialization resolves it straight to a memref.

    // Descriptor-driven tla.copy lowering (supported v1 routes -> runtime calls;
    // unsupported combinations stay as tla.copy and fail legalization later).
    LowerTlaCopyPattern lowerCopy(&getContext(), tensorDescriptorByValue, toErase,
                                  lowering.loweredMemrefByValue);
    SmallVector<::tla::CopyOp, 16> copyOps;
    root->walk([&](::tla::CopyOp op) { copyOps.push_back(op); });
    bool copyLoweringFailed = false;
    for (::tla::CopyOp op : copyOps) {
      if (!op || !op->getBlock())
        continue;
      PatternRewriter rewriter(op.getContext());
      rewriter.setInsertionPoint(op);
      if (failed(lowerCopy.matchAndRewrite(op, rewriter)))
        copyLoweringFailed = true;
    }
    if (copyLoweringFailed)
      passFailed = true;

    LowerTlaCubePattern lowerCube(&getContext());
    SmallVector<::tla::CubeOp, 4> cubeOps;
    root->walk<WalkOrder::PostOrder>([&](::tla::CubeOp op) { cubeOps.push_back(op); });
    for (::tla::CubeOp op : cubeOps) {
      if (!op || !op->getBlock())
        continue;
      PatternRewriter rewriter(op.getContext());
      rewriter.setInsertionPoint(op);
      if (failed(lowerCube.matchAndRewrite(op, rewriter)))
        return failure();
    }

    // Lower tla.mmad.
    LowerTlaMmadPattern lowerMmad(&getContext(), tensorDescriptorByValue, toErase,
                                  lowering.loweredMemrefByValue);
    SmallVector<Operation *, 16> mmadOps;
    root->walk([&](Operation *op) {
      if (llvm::isa<::tla::MmadOp>(op))
        mmadOps.push_back(op);
    });
    for (Operation *op : mmadOps) {
      if (!op->getBlock())
        continue;
      PatternRewriter rewriter(op->getContext());
      rewriter.setInsertionPoint(op);
      if (auto mmadOp = llvm::dyn_cast<::tla::MmadOp>(op)) {
        if (failed(lowerMmad.matchAndRewrite(mmadOp, rewriter))) {
          return failure();
        }
      }
    }

    // The tla.tensor_desc ops are dead (tla.copy / tla.mmad were lowered off their
    // descriptors, not their values), but they still hold their `base` scaffolding
    // (the inttoptr boundary) which the base-memref materialization staged for
    // erasure. Stage the dead tensor_descs so the flush erases them first; that
    // scaffolding cannot be erased while a live tensor_desc still references it.
    ::tla::stageDeadTensorDescriptors(root, toErase);

    // Flush staged erases: the lowered tla.copy / tla.mmad ops (which
    // tla-finalize-memref marks illegal, so they must be erased here), the dead
    // tla.tensor_desc ops, and the ptr bridges / scaffolding consumed while
    // materializing their tile memrefs.
    DenseSet<Operation *> pendingErase;
    for (Operation *op : toErase)
      if (op && op->getBlock())
        pendingErase.insert(op);
    bool progress = true;
    while (progress && !pendingErase.empty()) {
      progress = false;
      for (Operation *op : toErase) {
        if (!op || !pendingErase.contains(op) || !op->getBlock())
          continue;
        bool hasLiveResultUses = false;
        for (Value result : op->getResults())
          if (!result.use_empty()) {
            hasLiveResultUses = true;
            break;
          }
        if (hasLiveResultUses)
          continue;
        pendingErase.erase(op);
        op->erase();
        progress = true;
      }
    }
    if (!pendingErase.empty()) {
      for (Operation *op : pendingErase)
        op->emitError() << "staged erase failed for '" << op->getName().getStringRef()
                        << "' in tla-cube-region: operation still has live result users";
      return failure();
    }
    return passFailed ? failure() : success();
  }
};

} // namespace

std::unique_ptr<Pass> createTlaCubeRegionPass() { return std::make_unique<TlaCubeRegionPass>(); }

void registerTlaCubeRegionPass() { PassRegistration<TlaCubeRegionPass>(); }

} // namespace tla
