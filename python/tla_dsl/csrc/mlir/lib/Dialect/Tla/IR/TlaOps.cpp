#include "Dialect/Tla/IR/TlaOps.h"
#include "Dialect/Tla/IR/TlaAttrs.h"
#include "Dialect/Tla/IR/TlaTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/StringSwitch.h"

#define GET_OP_CLASSES
#include "tla/Ops.cpp.inc"

namespace tla {

template <typename TreeType>
static mlir::LogicalResult getIndexTreeLeavesForVerify(
    mlir::Operation *op, TreeType treeType,
    llvm::SmallVectorImpl<int64_t> &leaves, llvm::StringRef name) {
  if (failed(getTlaIndexTreeLeaves(treeType.getTree(), leaves)))
    return op->emitOpError() << "failed to decode " << name;
  return mlir::success();
}

static bool isSupportedCmpElementType(mlir::Type elementType) {
  if (elementType.isF16() || elementType.isF32())
    return true;
  auto intType = mlir::dyn_cast<mlir::IntegerType>(elementType);
  return intType && (intType.isSignless() || intType.isUnsigned()) &&
         intType.getWidth() == 32;
}

static bool isSupportedCmpMode(llvm::StringRef mode) {
  return llvm::StringSwitch<bool>(mode)
      .Cases("lt", "le", "gt", "ge", true)
      .Cases("eq", "ne", true)
      .Default(false);
}

mlir::LogicalResult AllocPtrOp::verify() {
  auto resTy = llvm::dyn_cast<PtrType>(getResult().getType());
  if (!resTy)
    return emitOpError("result must be !tla.ptr");
  if (resTy.getAlignment() == 0)
    return emitOpError("result pointer alignment must be positive");
  auto ms = resTy.getAddrspace();
  if (ms == AddressSpace::generic || ms == AddressSpace::gm)
    return emitOpError("alloc_ptr requires on-chip !tla.ptr (l1, l0a, l0b, l0c, ub)");
  int64_t sizeBytes = getSizeBytesAttr().getInt();
  if (sizeBytes <= 0)
    return emitOpError("size_bytes must be positive");
  int64_t elemBytes = getByteSizeOfFixedWidthScalarType(resTy.getPointee());
  if (elemBytes <= 0)
    return emitOpError("alloc_ptr pointee must be a fixed-width scalar type");
  if (sizeBytes % elemBytes != 0)
    return emitOpError("size_bytes must be a multiple of result pointee type size");
  return mlir::success();
}

mlir::LogicalResult TensorPtrOp::verify() {
  auto resTy = mlir::dyn_cast<PtrType>(getPtr().getType());
  if (!resTy)
    return emitOpError("result must be !tla.ptr");
  if (auto tensorTy = mlir::dyn_cast<TlaTensorType>(getSrc().getType())) {
    if (tensorTy.getPtr() != resTy)
      return emitOpError("result ptr type must match the tensor's embedded pointer type");
  }
  return mlir::success();
}

mlir::LogicalResult PtrAddOp::verify() {
  auto srcTy = mlir::dyn_cast<PtrType>(getPtr().getType());
  auto resTy = mlir::dyn_cast<PtrType>(getResult().getType());
  if (!srcTy || !resTy)
    return emitOpError("operands and result must be !tla.ptr");
  if (srcTy.getPointee() != resTy.getPointee())
    return emitOpError("result pointee type must match the source pointer's pointee");
  if (srcTy.getAddrspace() != resTy.getAddrspace())
    return emitOpError("result address space must match the source pointer's address space");
  return mlir::success();
}

mlir::LogicalResult TensorDescOp::verify() {
  // `packed` is empty for a linear (row-major/column-major) layout, or exactly
  // the 8 zN/nZ leaves [packed_shape0..3, packed_stride0..3] for a packed
  // layout. tla.tensor_desc is the single tile-descriptor op all tile producers
  // lower into.
  size_t n = getPacked().size();
  if (n != 0 && n != 8)
    return emitOpError("packed operands must be 0 (linear layout) or 8 "
                       "[packed_shape0..3, packed_stride0..3]");
  return mlir::success();
}

// Walk the enclosing ops looking for an ancestor of type AncestorOp. The
// required region may be several levels up (e.g. a compute op nested inside a
// scf.for loop inside a tla.vec.func), so this checks all transitive parents
// rather than just the immediate one.
template <typename AncestorOp> static bool hasEnclosing(mlir::Operation *op) {
  for (mlir::Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp())
    if (mlir::isa<AncestorOp>(parent))
      return true;
  return false;
}

// The region-wrapper requirement is a frontend/authoring constraint, enforced
// while ops still live in the tla.func container. Once tla-lower-func lowers
// tla.func to func.func -- and tla-vector-region / TlaSplitMixedFuncPass
// inline the frontend tla.cube / tla.vector wrappers into the resulting
// func.func (carrying the AIC/AIV/MIX core context on function attributes such
// as hivm.func_core_type / hivm.part_of_mix / hacc.entry rather than a lexical
// region) -- the lexical wrapper is legitimately gone. Ops already inside a
// lowered func.func are therefore exempt; the constraint is fully enforced at
// parse time and in the frontend, where ops are still under tla.func.
static bool isInLoweredFunc(mlir::Operation *op) {
  return op->getParentOfType<mlir::func::FuncOp>() != nullptr;
}

template <typename AncestorOp> static bool hasEnclosingRegion(mlir::Operation *op) {
  return hasEnclosing<AncestorOp>(op) || isInLoweredFunc(op);
}

mlir::LogicalResult MmadOp::verify() {
  if (!hasEnclosingRegion<CubeOp>(getOperation()))
    return emitOpError("must be nested inside a tla.cube region");
  return mlir::success();
}

mlir::LogicalResult VecFuncOp::verify() {
  if (!hasEnclosingRegion<VectorOp>(getOperation()))
    return emitOpError("must be nested inside a tla.vector region");
  return mlir::success();
}

static mlir::LogicalResult
verifyInterleaveLikeElementTypes(mlir::Operation *op, VectorSSAType src0Type,
                                 VectorSSAType src1Type, VectorSSAType dst0Type,
                                 VectorSSAType dst1Type) {
  mlir::Type src0ElementType = src0Type.getElementType();
  mlir::Type src1ElementType = src1Type.getElementType();
  mlir::Type dst0ElementType = dst0Type.getElementType();
  mlir::Type dst1ElementType = dst1Type.getElementType();
  if (src0ElementType != src1ElementType ||
      src0ElementType != dst0ElementType || src0ElementType != dst1ElementType)
    return op->emitOpError()
           << "requires all operands and results to have the same element "
              "type, got "
           << "src0=" << src0ElementType << ", src1=" << src1ElementType
           << ", dst0=" << dst0ElementType << ", dst1=" << dst1ElementType;

  return mlir::success();
}

mlir::LogicalResult InterleaveOp::verify() {
  if (!hasEnclosingRegion<VecFuncOp>(getOperation()))
    return emitOpError("must be nested inside a tla.vec.func region");

  return verifyInterleaveLikeElementTypes(
      getOperation(), getSrc0().getType(), getSrc1().getType(),
      getDst0().getType(), getDst1().getType());
}

mlir::LogicalResult DeInterleaveOp::verify() {
  if (!hasEnclosingRegion<VecFuncOp>(getOperation()))
    return emitOpError("must be nested inside a tla.vec.func region");

  return verifyInterleaveLikeElementTypes(
      getOperation(), getSrc0().getType(), getSrc1().getType(),
      getDst0().getType(), getDst1().getType());
}

// Register predicates describe the physical lane width of a 256-byte data
// register. Valid data lanes may be smaller, but predicate compatibility is
// determined by element width rather than VectorSSA valid_lanes.
static mlir::FailureOr<int64_t> getPhysicalVectorLanes(
    VectorSSAType vectorType) {
  constexpr int64_t kVectorRegisterBytes = 256;
  int64_t elementBytes =
      getByteSizeOfFixedWidthScalarType(vectorType.getElementType());
  if (elementBytes <= 0 || kVectorRegisterBytes % elementBytes != 0)
    return mlir::failure();
  return kVectorRegisterBytes / elementBytes;
}

static mlir::LogicalResult verifyMaskMatchesVector(mlir::Operation *op,
                                                   mlir::Value mask,
                                                   VectorSSAType vectorType) {
  if (!mask) return mlir::success();
  auto maskType = mlir::dyn_cast<MaskSSAType>(mask.getType());
  if (!maskType) return op->emitOpError("expected a !tla.mask<N> predicate");
  auto expectedPhysicalLanes = getPhysicalVectorLanes(vectorType);
  if (mlir::failed(expectedPhysicalLanes))
    return op->emitOpError()
           << "cannot derive predicate lanes for " << vectorType;
  if (maskType.getPhysicalLanes() != *expectedPhysicalLanes)
    return op->emitOpError()
           << "mask has " << maskType.getPhysicalLanes()
           << " predicate lanes, expected " << *expectedPhysicalLanes << " for "
           << vectorType.getElementType() << " VectorSSA";
  return mlir::success();
}

// Vector compute ops must live inside a tla.vec.func region.
#define TLA_VERIFY_IN_VEC_FUNC(OpTy)                                           \
  mlir::LogicalResult OpTy::verify() {                                         \
    if (!hasEnclosingRegion<VecFuncOp>(getOperation()))                        \
      return emitOpError("must be nested inside a tla.vec.func region");       \
    return mlir::success();                                                    \
  }

TLA_VERIFY_IN_VEC_FUNC(FullOp)

#undef TLA_VERIFY_IN_VEC_FUNC

#define TLA_VERIFY_MASKED_VECTOR_LHS(OpTy)                                     \
  mlir::LogicalResult OpTy::verify() {                                         \
    if (!hasEnclosingRegion<VecFuncOp>(getOperation()))                        \
      return emitOpError("must be nested inside a tla.vec.func region");       \
    return verifyMaskMatchesVector(getOperation(), getMask(),                  \
                                   getLhs().getType());                        \
  }

TLA_VERIFY_MASKED_VECTOR_LHS(AddOp)
TLA_VERIFY_MASKED_VECTOR_LHS(SubOp)
TLA_VERIFY_MASKED_VECTOR_LHS(MulOp)
TLA_VERIFY_MASKED_VECTOR_LHS(DivOp)
TLA_VERIFY_MASKED_VECTOR_LHS(MaxOp)
TLA_VERIFY_MASKED_VECTOR_LHS(MinOp)
TLA_VERIFY_MASKED_VECTOR_LHS(AddsOp)
TLA_VERIFY_MASKED_VECTOR_LHS(SubsOp)
TLA_VERIFY_MASKED_VECTOR_LHS(MulsOp)
TLA_VERIFY_MASKED_VECTOR_LHS(DivsOp)
TLA_VERIFY_MASKED_VECTOR_LHS(MaxsOp)
TLA_VERIFY_MASKED_VECTOR_LHS(MinsOp)

#undef TLA_VERIFY_MASKED_VECTOR_LHS

#define TLA_VERIFY_MASKED_VECTOR_OPERAND(OpTy)                                 \
  mlir::LogicalResult OpTy::verify() {                                         \
    if (!hasEnclosingRegion<VecFuncOp>(getOperation()))                        \
      return emitOpError("must be nested inside a tla.vec.func region");       \
    return verifyMaskMatchesVector(getOperation(), getMask(),                  \
                                   getOperand().getType());                    \
  }

TLA_VERIFY_MASKED_VECTOR_OPERAND(ExpOp)
TLA_VERIFY_MASKED_VECTOR_OPERAND(LogOp)
TLA_VERIFY_MASKED_VECTOR_OPERAND(SqrtOp)
TLA_VERIFY_MASKED_VECTOR_OPERAND(AbsOp)
TLA_VERIFY_MASKED_VECTOR_OPERAND(NegOp)

#undef TLA_VERIFY_MASKED_VECTOR_OPERAND

mlir::LogicalResult StoreOp::verify() {
  if (!hasEnclosingRegion<VecFuncOp>(getOperation()))
    return emitOpError("must be nested inside a tla.vec.func region");
  auto destType = mlir::dyn_cast<TlaTensorType>(getDest().getType());
  if (!destType)
    return emitOpError("dest must be !tla.tensor");
  if (destType.getPtr().getAddrspace() != AddressSpace::ub)
    return emitOpError("dest !tla.tensor must be in ub address space");
  return verifyMaskMatchesVector(getOperation(), getMask(),
                                 getSource().getType());
}

static mlir::LogicalResult verifyMaskProducerType(mlir::Operation *op,
                                                  MaskSSAType maskType,
                                                  mlir::Type elementType) {
  int64_t elementBytes = getByteSizeOfFixedWidthScalarType(elementType);
  constexpr int64_t kVectorRegisterBytes = 256;
  if (elementBytes <= 0 || kVectorRegisterBytes % elementBytes != 0)
    return op->emitOpError() << "unsupported predicate dtype " << elementType;
  int64_t expectedPhysicalLanes = kVectorRegisterBytes / elementBytes;
  if (maskType.getPhysicalLanes() != expectedPhysicalLanes)
    return op->emitOpError() << "result mask has " << maskType.getPhysicalLanes()
                             << " predicate lanes, expected " << expectedPhysicalLanes
                             << " for dtype " << elementType;
  return mlir::success();
}

mlir::LogicalResult CreateMaskOp::verify() {
  if (!hasEnclosingRegion<VecFuncOp>(getOperation()))
    return emitOpError("must be nested inside a tla.vec.func region");
  return verifyMaskProducerType(getOperation(), getResult().getType(),
                                getDtype());
}

mlir::LogicalResult UpdateMaskOp::verify() {
  if (!hasEnclosingRegion<VecFuncOp>(getOperation()))
    return emitOpError("must be nested inside a tla.vec.func region");
  return verifyMaskProducerType(getOperation(), getMask().getType(),
                                getDtype());
}

mlir::LogicalResult WhereOp::verify() {
  if (!hasEnclosingRegion<VecFuncOp>(getOperation()))
    return emitOpError("must be nested inside a tla.vec.func region");
  return verifyMaskMatchesVector(getOperation(), getMask(), getX().getType());
}

mlir::LogicalResult GatherOp::verify() {
  if (!hasEnclosingRegion<VecFuncOp>(getOperation()))
    return emitOpError("must be nested inside a tla.vec.func region");
  return verifyMaskMatchesVector(getOperation(), getMask(),
                                 getResult().getType());
}

mlir::LogicalResult SqueezeOp::verify() {
  if (!hasEnclosingRegion<VecFuncOp>(getOperation()))
    return emitOpError("must be nested inside a tla.vec.func region");
  return verifyMaskMatchesVector(getOperation(), getMask(), getSrc().getType());
}

mlir::LogicalResult CastOp::verify() {
  if (!hasEnclosingRegion<VecFuncOp>(getOperation()))
    return emitOpError("must be nested inside a tla.vec.func region");
  return verifyMaskMatchesVector(getOperation(), getMask(),
                                 getSource().getType());
}

mlir::LogicalResult ReduceOp::verify() {
  if (!hasEnclosingRegion<VecFuncOp>(getOperation()))
    return emitOpError("must be nested inside a tla.vec.func region");
  return verifyMaskMatchesVector(getOperation(), getMask(),
                                 getOperand().getType());
}

mlir::LogicalResult LoadOp::verify() {
  if (!hasEnclosingRegion<VecFuncOp>(getOperation()))
    return emitOpError("must be nested inside a tla.vec.func region");

  auto sourceType = mlir::dyn_cast<TlaTensorType>(getSource().getType());
  if (!sourceType)
    return emitOpError("source must be !tla.tensor");
  if (sourceType.getPtr().getAddrspace() != AddressSpace::ub)
    return emitOpError("source !tla.tensor must be in ub address space");

  bool isDintlv = false;
  if (auto loadDistAttr = getLoadDist())
    isDintlv = loadDistAttr->getLoadDist() == ::LoadDist::dintlv_b32;

  if (isDintlv) {
    if (!getResult2())
      return emitOpError(
          "load_dist dintlv_b32 requires a second result (dual-destination load)");
  } else if (getResult2()) {
    return emitOpError(
        "second result is only valid with load_dist dintlv_b32");
  }
  return mlir::success();
}

mlir::LogicalResult BitwiseNotOp::verify() {
  if (!hasEnclosingRegion<VecFuncOp>(getOperation()))
    return emitOpError("must be nested inside a tla.vec.func region");

  auto operandVector = mlir::dyn_cast<VectorSSAType>(getOperand().getType());
  auto resultVector = mlir::dyn_cast<VectorSSAType>(getResult().getType());
  bool operandIsVector = static_cast<bool>(operandVector);
  bool resultIsVector = static_cast<bool>(resultVector);
  if (operandIsVector != resultIsVector)
    return emitOpError(
        "requires operand and result to have the same !tla.vector or !tla.mask category");

  if (operandIsVector) {
    if (operandVector.getElementType() != resultVector.getElementType())
      return emitOpError(
          "requires !tla.vector operand and result to have identical element types");
    return verifyMaskMatchesVector(getOperation(), getMask(), operandVector);
  }

  auto operandMask = mlir::cast<MaskSSAType>(getOperand().getType());
  auto resultMask = mlir::cast<MaskSSAType>(getResult().getType());
  if (resultMask != operandMask)
    return emitOpError(
        "requires MaskSSA operand and result to have identical types");
  if (getMask() && getMask().getType() != operandMask)
    return emitOpError(
        "requires optional mask to have the same MaskSSA type as operand");
  return mlir::success();
}

template <typename OpTy>
static mlir::LogicalResult verifyBitwiseBinaryOp(OpTy op) {
  if (!hasEnclosingRegion<VecFuncOp>(op.getOperation()))
    return op.emitOpError("must be nested inside a tla.vec.func region");

  auto lhsVector = mlir::dyn_cast<VectorSSAType>(op.getLhs().getType());
  auto rhsVector = mlir::dyn_cast<VectorSSAType>(op.getRhs().getType());
  auto resultVector = mlir::dyn_cast<VectorSSAType>(op.getResult().getType());
  bool lhsIsVector = static_cast<bool>(lhsVector);
  bool rhsIsVector = static_cast<bool>(rhsVector);
  bool resultIsVector = static_cast<bool>(resultVector);
  if (lhsIsVector != rhsIsVector || lhsIsVector != resultIsVector)
    return op.emitOpError(
        "requires lhs, rhs, and result to have the same !tla.vector or !tla.mask category");

  if (lhsIsVector) {
    mlir::Type lhsElementType = lhsVector.getElementType();
    if (rhsVector.getElementType() != lhsElementType ||
        resultVector.getElementType() != lhsElementType)
      return op.emitOpError(
          "requires !tla.vector lhs, rhs, and result to have identical element types");
    return verifyMaskMatchesVector(op.getOperation(), op.getMask(), lhsVector);
  }

  auto lhsMask = mlir::cast<MaskSSAType>(op.getLhs().getType());
  auto rhsMask = mlir::cast<MaskSSAType>(op.getRhs().getType());
  auto resultMask = mlir::cast<MaskSSAType>(op.getResult().getType());
  if (rhsMask != lhsMask || resultMask != lhsMask)
    return op.emitOpError(
        "requires MaskSSA lhs, rhs, and result to have identical types");
  if (op.getMask() && op.getMask().getType() != lhsMask)
    return op.emitOpError(
        "requires optional mask to have the same MaskSSA type as operands");
  return mlir::success();
}

mlir::LogicalResult BitwiseAndOp::verify() {
  return verifyBitwiseBinaryOp(*this);
}

mlir::LogicalResult BitwiseOrOp::verify() {
  return verifyBitwiseBinaryOp(*this);
}

mlir::LogicalResult BitwiseXorOp::verify() {
  return verifyBitwiseBinaryOp(*this);
}

mlir::LogicalResult ArangeOp::verify() {
  if (!hasEnclosingRegion<VecFuncOp>(getOperation()))
    return emitOpError("must be nested inside a tla.vec.func region");
  auto order = getOrderAttr().getValue();
  if (order != "increase" && order != "decrease")
    return emitOpError("unsupported arange order '")
           << order << "'; expected 'increase' or 'decrease'";
  return mlir::success();
}

// Synchronization/mutex/barrier ops must live inside a tla.cube or tla.vector
// region (either core-kind region; not the func-level scope).
#define TLA_VERIFY_IN_CUBE_OR_VECTOR(OpTy)                                     \
  mlir::LogicalResult OpTy::verify() {                                          \
    if (!hasEnclosingRegion<CubeOp>(getOperation()) &&                         \
        !hasEnclosingRegion<VectorOp>(getOperation()))                         \
      return emitOpError(                                                       \
          "must be nested inside a tla.cube or tla.vector region");            \
    return mlir::success();                                                     \
  }

TLA_VERIFY_IN_CUBE_OR_VECTOR(SetFlagOp)
TLA_VERIFY_IN_CUBE_OR_VECTOR(WaitFlagOp)
TLA_VERIFY_IN_CUBE_OR_VECTOR(MutexLockOp)
TLA_VERIFY_IN_CUBE_OR_VECTOR(MutexUnlockOp)
TLA_VERIFY_IN_CUBE_OR_VECTOR(PipeBarrierOp)

#undef TLA_VERIFY_IN_CUBE_OR_VECTOR

template <typename OpTy>
static mlir::LogicalResult verifyCrossCoreFlagOp(OpTy op)
{
    if (!hasEnclosingRegion<CubeOp>(op.getOperation()) && !hasEnclosingRegion<VectorOp>(op.getOperation()))
        return op.emitOpError("must be nested inside a tla.cube or tla.vector region");
    auto flagType = mlir::cast<CrossFlagType>(op.getFlag().getType());
    auto aivIdAttr = op->template getAttrOfType<mlir::IntegerAttr>("aiv_id");
    if (flagType.getMode() == 4) {
        if (!aivIdAttr || (aivIdAttr.getInt() != 0 && aivIdAttr.getInt() != 1))
            return op.emitOpError("mode 4 requires aiv_id to be the compile-time integer 0 or 1");
    } else if (aivIdAttr) {
        return op.emitOpError("aiv_id is only valid for mode 4 cross flags");
    }
    return mlir::success();
}

mlir::LogicalResult CrossCoreSetFlagOp::verify()
{
    return verifyCrossCoreFlagOp(*this);
}

mlir::LogicalResult CrossCoreWaitFlagOp::verify()
{
    return verifyCrossCoreFlagOp(*this);
}

mlir::LogicalResult LocalMemBarOp::verify() {
  auto kind = getBarrierKind();
  if (kind < 0 || kind > 11)
    return emitOpError("barrier_kind ") << kind << " is out of range [0, 11]";
  if (!hasEnclosingRegion<CubeOp>(getOperation()) &&
      !hasEnclosingRegion<VectorOp>(getOperation()))
    return emitOpError(
        "must be nested inside a tla.cube or tla.vector region");
  return mlir::success();
}

mlir::LogicalResult CopyOp::verify() {
  auto srcTy = mlir::dyn_cast<TlaTensorType>(getSrc().getType());
  auto dstTy = mlir::dyn_cast<TlaTensorType>(getDst().getType());
  if (!srcTy || !dstTy)
    return mlir::success(); // Operand type verifier handles malformed tensors.
  AddressSpace src = srcTy.getPtr().getAddrspace();
  AddressSpace dst = dstTy.getPtr().getAddrspace();

  // Cube data-path copies: GM->L1, L1->L0A, L1->L0B, L0C->GM, L0C->UB, L1->UB.
  bool cubeRoute = (src == AddressSpace::gm && dst == AddressSpace::l1) ||
                   (src == AddressSpace::l1 && dst == AddressSpace::l0a) ||
                   (src == AddressSpace::l1 && dst == AddressSpace::l0b) ||
                   (src == AddressSpace::l0c && dst == AddressSpace::gm) ||
                   (src == AddressSpace::l0c && dst == AddressSpace::ub) ||
                   (src == AddressSpace::l1 && dst == AddressSpace::ub);
  // Vector staging copies: GM->UB, UB->GM, UB->L1.
  bool vectorRoute = (src == AddressSpace::gm && dst == AddressSpace::ub) ||
                     (src == AddressSpace::ub && dst == AddressSpace::gm) ||
                     (src == AddressSpace::ub && dst == AddressSpace::l1);

  if (cubeRoute && !hasEnclosingRegion<CubeOp>(getOperation()))
    return emitOpError("copy between GM/L1/L0A/L0B/L0C/UB must be nested inside "
                       "a tla.cube region");
  if (vectorRoute && !hasEnclosingRegion<VectorOp>(getOperation()))
    return emitOpError(
        "copy between GM/UB/L1 must be nested inside a tla.vector region");

  // Validate atomic_mode attribute when present.
  auto atomicModeAttr = getAtomicModeAttr();
  if (atomicModeAttr && atomicModeAttr.getAtomicMode() != AtomicMode::none) {
    if (atomicModeAttr.getAtomicMode() != AtomicMode::add)
      return emitOpError("unsupported atomic_mode; currently only 'add' is supported");

    if (dst != AddressSpace::gm)
      return emitOpError("atomic operation requires dst to be in GM address space");

    auto elemType = dstTy.getPtr().getPointee();
    if (!elemType.isF32() && !elemType.isF16() && !elemType.isBF16() &&
        !elemType.isInteger(32) && !elemType.isInteger(16) && !elemType.isInteger(8))
      return emitOpError("atomic operation requires dst element type to be one of "
                         "f32, f16, bf16, i32, i16, i8");
  }

  return mlir::success();
}

mlir::LogicalResult CmpOp::verify() {
  if (!hasEnclosingRegion<VecFuncOp>(getOperation()))
    return emitOpError("must be nested inside a tla.vec.func region");
  if (!isSupportedCmpMode(getMode()))
    return emitOpError()
           << "mode must be one of lt, le, gt, ge, eq, ne, got \""
           << getMode() << "\"";

  auto lhsType = getLhs().getType();
  mlir::Type lhsElementType = lhsType.getElementType();
  if (!isSupportedCmpElementType(lhsElementType))
    return emitOpError() << "unsupported compare element type "
                         << lhsElementType;
  if (failed(verifyMaskMatchesVector(getOperation(), getResult(), lhsType)) ||
      failed(verifyMaskMatchesVector(getOperation(), getMask(), lhsType)))
    return mlir::failure();

  auto rhsType = getRhs().getType();
  auto rhsVectorType = mlir::dyn_cast<::tla::VectorSSAType>(rhsType);
  if (!rhsVectorType) {
    if (rhsType != lhsElementType)
      return emitOpError() << "scalar operand must have element type "
                           << lhsElementType << ", got " << rhsType;
    return mlir::success();
  }

  mlir::Type rhsElementType = rhsVectorType.getElementType();
  if (lhsElementType != rhsElementType)
    return emitOpError() << "operands must have the same element type, got "
                         << lhsElementType << " and " << rhsElementType;

  return mlir::success();
}

mlir::LogicalResult DebugPrintOp::verify() {
  auto type = getValue().getType();
  auto integerType = mlir::dyn_cast<mlir::IntegerType>(type);
  bool isSignlessI32 = integerType && integerType.isSignless() &&
                       integerType.getWidth() == 32;
  if (!isSignlessI32 && !type.isF32())
    return emitOpError("expected a signless i32 or f32 scalar, got ") << type;
  if (!hasEnclosingRegion<CubeOp>(getOperation()) &&
      !hasEnclosingRegion<VectorOp>(getOperation()))
    return emitOpError("must be nested inside a tla.cube or tla.vector region");
  return mlir::success();
}

static mlir::ParseResult parseIndexTreeValueOp(mlir::OpAsmParser &parser,
                                               mlir::OperationState &result) {
  llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand, 4> dynElemsOperands;
  llvm::SMLoc dynElemsOperandsLoc = parser.getCurrentLocation();
  mlir::Type resultType;

  if (parser.parseOperandList(dynElemsOperands))
    return mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return mlir::failure();
  if (parser.parseArrow())
    return mlir::failure();
  if (parser.parseType(resultType))
    return mlir::failure();

  result.addTypes(resultType);
  if (parser.resolveOperands(dynElemsOperands, parser.getBuilder().getIndexType(),
                             dynElemsOperandsLoc, result.operands))
    return mlir::failure();
  return mlir::success();
}

template <typename OpTy> static void printIndexTreeValueOp(OpTy op, mlir::OpAsmPrinter &printer) {
  if (!op.getDynElems().empty()) {
    printer << ' ';
    printer << op.getDynElems();
  }
  printer.printOptionalAttrDict(op->getAttrs());
  printer << " -> ";
  printer.printType(op.getResult().getType());
}

mlir::ParseResult MakeShapeOp::parse(mlir::OpAsmParser &parser, mlir::OperationState &result) {
  return parseIndexTreeValueOp(parser, result);
}

void MakeShapeOp::print(mlir::OpAsmPrinter &printer) { printIndexTreeValueOp(*this, printer); }

mlir::ParseResult MakeCoordOp::parse(mlir::OpAsmParser &parser, mlir::OperationState &result) {
  return parseIndexTreeValueOp(parser, result);
}

void MakeCoordOp::print(mlir::OpAsmPrinter &printer) { printIndexTreeValueOp(*this, printer); }

mlir::ParseResult MakeStrideOp::parse(mlir::OpAsmParser &parser, mlir::OperationState &result) {
  return parseIndexTreeValueOp(parser, result);
}

void MakeStrideOp::print(mlir::OpAsmPrinter &printer) { printIndexTreeValueOp(*this, printer); }

mlir::ParseResult FuncOp::parse(mlir::OpAsmParser &parser, mlir::OperationState &result) {
  mlir::StringAttr symNameAttr;
  if (parser.parseSymbolName(symNameAttr, "sym_name", result.attributes))
    return llvm::failure();

  llvm::SmallVector<mlir::OpAsmParser::Argument, 4> arguments;
  if (parser.parseArgumentList(arguments, mlir::OpAsmParser::Delimiter::Paren,
                               /*allowType=*/true))
    return llvm::failure();

  llvm::SmallVector<mlir::Type, 4> inputTypes;
  inputTypes.reserve(arguments.size());
  for (const auto &arg : arguments) {
    if (!arg.type)
      return parser.emitError(arg.ssaName.location) << "expected type for function argument";
    inputTypes.push_back(arg.type);
  }

  llvm::SmallVector<mlir::Type, 2> resultTypes;
  if (parser.parseOptionalArrowTypeList(resultTypes))
    return llvm::failure();

  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return llvm::failure();

  auto functionType = parser.getBuilder().getFunctionType(inputTypes, resultTypes);
  result.attributes.set("function_type", mlir::TypeAttr::get(functionType));

  mlir::Region *body = result.addRegion();
  if (parser.parseRegion(*body, arguments))
    return llvm::failure();

  return llvm::success();
}

void FuncOp::print(mlir::OpAsmPrinter &printer) {
  printer << " @" << getSymName() << "(";

  auto fnType = llvm::dyn_cast<mlir::FunctionType>(getFunctionType());
  llvm::ArrayRef<mlir::Type> inputs = fnType ? fnType.getInputs() : llvm::ArrayRef<mlir::Type>{};
  mlir::Block *entry = getBody().empty() ? nullptr : &getBody().front();

  for (size_t i = 0; i < inputs.size(); ++i) {
    if (i)
      printer << ", ";
    if (entry && i < entry->getNumArguments()) {
      printer << entry->getArgument(i);
    } else {
      printer << "%arg" << i;
    }
    printer << ": " << inputs[i];
  }
  printer << ")";

  if (fnType && !fnType.getResults().empty()) {
    printer << " -> (";
    for (size_t i = 0; i < fnType.getResults().size(); ++i) {
      if (i)
        printer << ", ";
      printer.printType(fnType.getResults()[i]);
    }
    printer << ")";
  }

  printer.printOptionalAttrDictWithKeyword((*this)->getAttrs(), {"sym_name", "function_type"});
  printer << " ";
  printer.printRegion(getBody(), /*printEntryBlockArgs=*/false);
}

mlir::LogicalResult ScalarLoadOp::verify() {
  auto srcTy = mlir::dyn_cast<TlaTensorType>(getSource().getType());
  if (!srcTy)
    return emitOpError("source must be !tla.tensor");
  if (srcTy.getPtr().getAddrspace() != AddressSpace::gm)
    return emitOpError("source !tla.tensor must be in gm address space");
  auto layoutTag = srcTy.getLayout().getLayoutTag();
  if (layoutTag != LayoutTag::row_major && layoutTag != LayoutTag::column_major)
    return emitOpError("source !tla.tensor layout must be row_major or column_major");
  mlir::Type expected = srcTy.getPtr().getPointee();
  llvm::SmallVector<int64_t, 4> shapeLeaves;
  if (failed(getIndexTreeLeavesForVerify(getOperation(), srcTy.getLayout().getShape(), shapeLeaves,
                                         "shape")))
    return mlir::failure();
  size_t rank = shapeLeaves.size();

  if (getResult().getType() != expected)
    return emitOpError("result type must match tensor element type, expected ")
           << expected << ", got " << getResult().getType();

  auto indexCount = getIndices().size();
  // Indices must match the logical rank: no row-omitted shorthand for rank-2.
  if (!((rank == 1 && indexCount == 1) || (rank == 2 && indexCount == 2)))
    return emitOpError(
        "scalar_load expects rank-1/2 source with matching indices (rank-1: 1; rank-2: 2)");
  for (mlir::Value idx : getIndices()) {
    if (!idx.getType().isIndex())
      return emitOpError("indices must be index-typed");
  }
  return mlir::success();
}

mlir::LogicalResult ScalarStoreOp::verify() {
  auto destTy = mlir::dyn_cast<TlaTensorType>(getDest().getType());
  if (!destTy)
    return emitOpError("dest must be !tla.tensor");
  if (destTy.getPtr().getAddrspace() != AddressSpace::gm)
    return emitOpError("dest !tla.tensor must be in gm address space");
  auto layoutTag = destTy.getLayout().getLayoutTag();
  if (layoutTag != LayoutTag::row_major && layoutTag != LayoutTag::column_major)
    return emitOpError("dest !tla.tensor layout must be row_major or column_major");
  mlir::Type expected = destTy.getPtr().getPointee();
  llvm::SmallVector<int64_t, 4> shapeLeaves;
  if (failed(getIndexTreeLeavesForVerify(getOperation(), destTy.getLayout().getShape(), shapeLeaves,
                                         "shape")))
    return mlir::failure();
  size_t rank = shapeLeaves.size();

  if (getValue().getType() != expected)
    return emitOpError("value type must match tensor element type, expected ")
           << expected << ", got " << getValue().getType();

  auto indexCount = getIndices().size();
  if (!((rank == 1 && indexCount == 1) || (rank == 2 && indexCount == 2)))
    return emitOpError(
        "scalar_store expects rank-1/2 dest with matching indices (rank-1: 1; rank-2: 2)");
  for (mlir::Value idx : getIndices()) {
    if (!idx.getType().isIndex())
      return emitOpError("indices must be index-typed");
  }
  return mlir::success();
}

} // namespace tla
