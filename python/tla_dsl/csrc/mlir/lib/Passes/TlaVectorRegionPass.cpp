#include "PassesCommon.h"
#include "PassesInternal.h"
#include "Passes/TlaTensorToMemref.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVMAVE/IR/HIVMAVE.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "llvm/ADT/StringSwitch.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace tla {
namespace {
// ParsedTensorInfo + parseTensorInfo live in the shared header
// Passes/TlaTensorToMemref.h (raw, non-normalized decode). Unqualified uses below
// resolve to ::tla:: via namespace lookup.

static hivmave::VFPgeOp createAvePgeMask(OpBuilder &b, Location loc, VectorType maskType,
                                         hivmave::PgePattern pattern) {
  return b.create<hivmave::VFPgeOp>(loc, maskType, pattern);
}

static hivmave::VFPltOp createAvePltMask(OpBuilder &b, Location loc, VectorType maskType,
                                         Value trueShape) {
  return b.create<hivmave::VFPltOp>(loc, maskType, b.getIndexType(), trueShape);
}

static LogicalResult lowerLocalMemBar(OpBuilder &b, ::tla::LocalMemBarOp op) {
  int64_t barrierKind = op.getBarrierKind();
  if (barrierKind < 0 || barrierKind > 11)
    return op.emitError("barrier_kind ") << barrierKind << " is out of range [0, 11]";
  Value encoded = b.create<arith::ConstantIntOp>(op.getLoc(), barrierKind, 32);
  b.create<hivmave::VFMemBarOp>(op.getLoc(), encoded);
  return success();
}

static hivmave::LoadDist mapTlaLoadDistToAve(::LoadDist dist) {
  switch (dist) {
  case ::LoadDist::norm:
    return hivmave::LoadDist::NORM;
  case ::LoadDist::brc_b32:
    return hivmave::LoadDist::BRC_B32;
  case ::LoadDist::dintlv_b32:
    return hivmave::LoadDist::DINTLV_B32;
  }
  llvm_unreachable("unsupported tla.load load_dist");
}

static bool isDualDestLoadDist(hivmave::LoadDist pattern) {
  return pattern == hivmave::LoadDist::DINTLV_B8 ||
         pattern == hivmave::LoadDist::DINTLV_B16 ||
         pattern == hivmave::LoadDist::DINTLV_B32;
}

static hivmave::VFLoadOp createVFLoad(OpBuilder &b, Location loc, VectorType vecType,
                                      Value memref, Value index, hivmave::LoadDist pattern,
                                      bool unaligned) {
  // Always pass ``pattern`` at create time. Dual-destination dists need two
  // result types up front (AVE's convenience builder only emits a single NORM
  // result), and single-destination non-NORM (e.g. BRC_B32) can use the same
  // pattern-in-create overload instead of create-then-setPattern.
  SmallVector<Type, 2> resultTypes;
  resultTypes.push_back(vecType);
  if (isDualDestLoadDist(pattern))
    resultTypes.push_back(vecType);

  auto load = b.create<hivmave::VFLoadOp>(loc, resultTypes, pattern, memref,
                                          ValueRange{index});
  if (unaligned)
    load->setAttr(hivmave::UnalignedAttr::name,
                  hivmave::UnalignedAttr::get(b.getContext()));
  return load;
}

static std::string buildUniqueVectorHelperName(ModuleOp module, int &nextVectorRegionId) {
  std::string helperName;
  do {
    helperName = "vector_region_" + std::to_string(nextVectorRegionId++);
  } while (module.lookupSymbol<func::FuncOp>(helperName));
  return helperName;
}

enum class VectorBinaryKind { Add, Sub, Mul, Div, Max, Min, And, Or, Xor };
enum class VectorRhsKind { Vector, Scalar };

static FailureOr<hivmave::CombiningKind>
getAveReductionCombiningKind(::tla::ReduceOp reduceOp, Type elementType) {
  auto kindAttr = reduceOp->getAttrOfType<StringAttr>("kind");
  if (!kindAttr)
    return reduceOp.emitError("tla.reduce requires string kind attribute"), failure();
  StringRef kind = kindAttr.getValue();
  if (kind == "add")
    return hivmave::CombiningKind::ADD;
  if (kind == "max") {
    if (auto intType = dyn_cast<IntegerType>(elementType))
      return intType.getSignedness() == IntegerType::Unsigned ? hivmave::CombiningKind::UMAX
                                                              : hivmave::CombiningKind::MAX;
    if (isa<FloatType>(elementType))
      return hivmave::CombiningKind::MAX;
  }
  if (kind == "min") {
    if (auto intType = dyn_cast<IntegerType>(elementType))
      return intType.getSignedness() == IntegerType::Unsigned ? hivmave::CombiningKind::UMIN
                                                              : hivmave::CombiningKind::MIN;
    if (isa<FloatType>(elementType))
      return hivmave::CombiningKind::MIN;
  }
  return reduceOp.emitError()
             << "tla.reduce supports only add, max, and min reductions, got \""
             << kind << "\"",
         failure();
}

static bool isSupportedVectorReductionElementType(Type elementType) {
  if (isa<Float16Type, Float32Type>(elementType))
    return true;
  auto intType = dyn_cast<IntegerType>(elementType);
  if (!intType)
    return false;
  switch (intType.getWidth()) {
  case 16:
  case 32:
    return true;
  default:
    return false;
  }
}

// A store destination tile is directly a tla.tensor_desc result after
// tla-lower-tensor-desc (the sole descriptor producer). Only look one level;
// the caller diagnoses when the value is not a descriptor result.
static ::tla::TensorDescOp findTensorDescProducer(Value tensorValue) {
  return tensorValue ? tensorValue.getDefiningOp<::tla::TensorDescOp>() : nullptr;
}

static LogicalResult validateVectorReduction(::tla::ReduceOp reduceOp, Type elementType) {
  if (!isSupportedVectorReductionElementType(elementType))
    return reduceOp.emitError()
           << "tla.reduce unsupported reduction element type " << elementType;
  return success();
}

enum class VectorUnaryKind { Exp, Log, Sqrt, Abs, Neg, Not };

struct TlaUnaryOperands {
  Value operand;
  Value mask;
};

struct VectorUnaryInfo {
  VectorUnaryKind kind;
  StringRef name;
  TlaUnaryOperands operands;
};

template <typename OpTy> static TlaUnaryOperands getTlaUnaryOperands(OpTy op) {
  return TlaUnaryOperands{op.getOperand(), op.getMask()};
}

static std::optional<VectorUnaryInfo> getVectorUnaryInfo(Operation *op) {
  if (!op)
    return std::nullopt;
  if (auto o = dyn_cast<::tla::ExpOp>(op))
    return VectorUnaryInfo{VectorUnaryKind::Exp, "exp", getTlaUnaryOperands(o)};
  if (auto o = dyn_cast<::tla::LogOp>(op))
    return VectorUnaryInfo{VectorUnaryKind::Log, "log", getTlaUnaryOperands(o)};
  if (auto o = dyn_cast<::tla::SqrtOp>(op))
    return VectorUnaryInfo{VectorUnaryKind::Sqrt, "sqrt", getTlaUnaryOperands(o)};
  if (auto o = dyn_cast<::tla::AbsOp>(op))
    return VectorUnaryInfo{VectorUnaryKind::Abs, "abs", getTlaUnaryOperands(o)};
  if (auto o = dyn_cast<::tla::NegOp>(op))
    return VectorUnaryInfo{VectorUnaryKind::Neg, "neg", getTlaUnaryOperands(o)};
  if (auto o = dyn_cast<::tla::BitwiseNotOp>(op))
    return VectorUnaryInfo{VectorUnaryKind::Not, "bitwise_not", getTlaUnaryOperands(o)};
  return std::nullopt;
}

static LogicalResult validateVectorUnaryElementType(Operation *op, VectorUnaryInfo info,
                                                    Type elementType) {
  switch (info.kind) {
  case VectorUnaryKind::Exp:
  case VectorUnaryKind::Log:
  case VectorUnaryKind::Sqrt:
    if (!isa<FloatType>(elementType))
      return op->emitError() << "tla." << info.name
                             << " requires floating-point element type, got "
                             << elementType;
    if (isa<BFloat16Type>(elementType))
      return op->emitError() << "tla." << info.name
                             << " does not support bf16 element type yet";
    return success();
  case VectorUnaryKind::Abs:
  case VectorUnaryKind::Neg:
  case VectorUnaryKind::Not:
    if (auto floatType = dyn_cast<FloatType>(elementType)) {
      if (isa<BFloat16Type>(floatType))
        return op->emitError() << "tla." << info.name
                               << " does not support bf16 element type yet";
      if (floatType.isF16() || floatType.isF32())
        return success();
      return op->emitError()
             << "tla." << info.name
             << " requires f16 or f32 floating-point element type, got "
             << elementType;
    }
    if (auto intType = dyn_cast<IntegerType>(elementType)) {
      unsigned width = intType.getWidth();
      if (width == 8 || width == 16 || width == 32)
        return success();
      return op->emitError()
             << "tla." << info.name
             << " requires i8, i16, or i32 element type, got "
             << elementType;
    }
    return op->emitError() << "tla." << info.name
           << " requires f16/f32 or i8/i16/i32 element type, got "
           << elementType;
  }
  return failure();
}

// The lhs/rhs/mask operands of vector binary ops (mask may be null).
struct TlaBinaryOperands {
  Value lhs;
  Value rhs;
  Value mask;
};

static TlaBinaryOperands getTlaBinaryOperands(Operation *op) {
  TlaBinaryOperands r{};
  if (auto o = dyn_cast<::tla::AddOp>(op)) {
    r.lhs = o.getLhs(); r.rhs = o.getRhs(); r.mask = o.getMask();
  } else if (auto o = dyn_cast<::tla::SubOp>(op)) {
    r.lhs = o.getLhs(); r.rhs = o.getRhs(); r.mask = o.getMask();
  } else if (auto o = dyn_cast<::tla::MulOp>(op)) {
    r.lhs = o.getLhs(); r.rhs = o.getRhs(); r.mask = o.getMask();
  } else if (auto o = dyn_cast<::tla::DivOp>(op)) {
    r.lhs = o.getLhs(); r.rhs = o.getRhs(); r.mask = o.getMask();
  } else if (auto o = dyn_cast<::tla::MaxOp>(op)) {
    r.lhs = o.getLhs(); r.rhs = o.getRhs(); r.mask = o.getMask();
  } else if (auto o = dyn_cast<::tla::MinOp>(op)) {
    r.lhs = o.getLhs(); r.rhs = o.getRhs(); r.mask = o.getMask();
  } else if (auto o = dyn_cast<::tla::AddsOp>(op)) {
    r.lhs = o.getLhs(); r.rhs = o.getRhs(); r.mask = o.getMask();
  } else if (auto o = dyn_cast<::tla::SubsOp>(op)) {
    r.lhs = o.getLhs(); r.rhs = o.getRhs(); r.mask = o.getMask();
  } else if (auto o = dyn_cast<::tla::MulsOp>(op)) {
    r.lhs = o.getLhs(); r.rhs = o.getRhs(); r.mask = o.getMask();
  } else if (auto o = dyn_cast<::tla::MaxsOp>(op)) {
    r.lhs = o.getLhs(); r.rhs = o.getRhs(); r.mask = o.getMask();
  } else if (auto o = dyn_cast<::tla::MinsOp>(op)) {
    r.lhs = o.getLhs(); r.rhs = o.getRhs(); r.mask = o.getMask();
  } else if (auto o = dyn_cast<::tla::DivsOp>(op)) {
    r.lhs = o.getLhs(); r.rhs = o.getRhs(); r.mask = o.getMask();
  } else if (auto o = dyn_cast<::tla::BitwiseAndOp>(op)) {
    r.lhs = o.getLhs(); r.rhs = o.getRhs(); r.mask = o.getMask();
  } else if (auto o = dyn_cast<::tla::BitwiseOrOp>(op)) {
    r.lhs = o.getLhs(); r.rhs = o.getRhs(); r.mask = o.getMask();
  } else if (auto o = dyn_cast<::tla::BitwiseXorOp>(op)) {
    r.lhs = o.getLhs(); r.rhs = o.getRhs(); r.mask = o.getMask();
  }
  return r;
}

struct VectorOpInfo {
  VectorBinaryKind kind;
  VectorRhsKind rhsKind;
  StringRef mnemonic;
  TlaBinaryOperands operands;
};

struct AnyVectorOperationInfo {
  std::optional<VectorOpInfo> binary;
  std::optional<VectorUnaryInfo> unary;
};

static std::optional<VectorOpInfo> getVectorBinaryInfo(Operation *op) {
  if (!op)
    return std::nullopt;
  if (isa<::tla::AddOp>(op))
    return VectorOpInfo{VectorBinaryKind::Add, VectorRhsKind::Vector, "add",
                        getTlaBinaryOperands(op)};
  if (isa<::tla::SubOp>(op))
    return VectorOpInfo{VectorBinaryKind::Sub, VectorRhsKind::Vector, "sub",
                        getTlaBinaryOperands(op)};
  if (isa<::tla::MulOp>(op))
    return VectorOpInfo{VectorBinaryKind::Mul, VectorRhsKind::Vector, "mul",
                        getTlaBinaryOperands(op)};
  if (isa<::tla::DivOp>(op))
    return VectorOpInfo{VectorBinaryKind::Div, VectorRhsKind::Vector, "div",
                        getTlaBinaryOperands(op)};
  if (isa<::tla::MaxOp>(op))
    return VectorOpInfo{VectorBinaryKind::Max, VectorRhsKind::Vector, "max",
                        getTlaBinaryOperands(op)};
  if (isa<::tla::MinOp>(op))
    return VectorOpInfo{VectorBinaryKind::Min, VectorRhsKind::Vector, "min",
                        getTlaBinaryOperands(op)};
  if (isa<::tla::BitwiseAndOp>(op))
    return VectorOpInfo{VectorBinaryKind::And, VectorRhsKind::Vector, "bitwise_and",
                        getTlaBinaryOperands(op)};
  if (isa<::tla::BitwiseOrOp>(op))
    return VectorOpInfo{VectorBinaryKind::Or, VectorRhsKind::Vector, "bitwise_or",
                        getTlaBinaryOperands(op)};
  if (isa<::tla::BitwiseXorOp>(op))
    return VectorOpInfo{VectorBinaryKind::Xor, VectorRhsKind::Vector, "bitwise_xor",
                        getTlaBinaryOperands(op)};
  return std::nullopt;
}

static std::optional<VectorOpInfo> getVectorScalarBinaryInfo(Operation *op) {
  if (!op)
    return std::nullopt;
  if (isa<::tla::AddsOp>(op))
    return VectorOpInfo{VectorBinaryKind::Add, VectorRhsKind::Scalar, "adds",
                        getTlaBinaryOperands(op)};
  if (isa<::tla::SubsOp>(op))
    return VectorOpInfo{VectorBinaryKind::Sub, VectorRhsKind::Scalar, "subs",
                        getTlaBinaryOperands(op)};
  if (isa<::tla::MulsOp>(op))
    return VectorOpInfo{VectorBinaryKind::Mul, VectorRhsKind::Scalar, "muls",
                        getTlaBinaryOperands(op)};
  if (isa<::tla::MaxsOp>(op))
    return VectorOpInfo{VectorBinaryKind::Max, VectorRhsKind::Scalar, "maxs",
                        getTlaBinaryOperands(op)};
  if (isa<::tla::MinsOp>(op))
    return VectorOpInfo{VectorBinaryKind::Min, VectorRhsKind::Scalar, "mins",
                        getTlaBinaryOperands(op)};
  if (isa<::tla::DivsOp>(op))
    return VectorOpInfo{VectorBinaryKind::Div, VectorRhsKind::Scalar, "divs",
                        getTlaBinaryOperands(op)};
  return std::nullopt;
}

static std::optional<AnyVectorOperationInfo> getAnyVectorOperationInfo(Operation *op) {
  if (auto info = getVectorBinaryInfo(op))
    return AnyVectorOperationInfo{*info, std::nullopt};
  if (auto info = getVectorScalarBinaryInfo(op))
    return AnyVectorOperationInfo{*info, std::nullopt};
  if (auto info = getVectorUnaryInfo(op))
    return AnyVectorOperationInfo{std::nullopt, *info};
  return std::nullopt;
}

// The mask-register width (b8/b16/b32) matching the element type.
static hivmave::MaskWidth maskWidthForElement(Type elementType) {
  unsigned bits = elementType.getIntOrFloatBitWidth();
  if (bits <= 8)
    return hivmave::MaskWidth::B8;
  if (bits <= 16)
    return hivmave::MaskWidth::B16;
  return hivmave::MaskWidth::B32;
}

static std::optional<hivmave::CmpType> mapCmpMode(StringRef mode) {
  return llvm::StringSwitch<std::optional<hivmave::CmpType>>(mode)
      .Case("lt", hivmave::CmpType::LT)
      .Case("le", hivmave::CmpType::LE)
      .Case("gt", hivmave::CmpType::GT)
      .Case("ge", hivmave::CmpType::GE)
      .Case("eq", hivmave::CmpType::EQ)
      .Case("ne", hivmave::CmpType::NE)
      .Default(std::nullopt);
}

// True for the tla ops that produce a vector compute result inside a vec.func
// region: element-wise binary/unary ops, bitwise ops, where/select,
// reductions, and gather.
static bool isVectorComputeOp(Operation *op) {
  return getAnyVectorOperationInfo(op).has_value() ||
         isa_and_nonnull<::tla::CmpOp>(op) ||
         isa_and_nonnull<::tla::WhereOp>(op) ||
         isa_and_nonnull<::tla::SqueezeOp>(op) ||
         isa_and_nonnull<::tla::ReduceOp>(op) ||
         isa_and_nonnull<::tla::GatherOp>(op) ||
         isa_and_nonnull<::tla::CastOp>(op) ||
         isa_and_nonnull<::tla::InterleaveOp>(op) ||
         isa_and_nonnull<::tla::DeInterleaveOp>(op);
}

static std::string getSqueezeLibraryCallName(Type elementType) {
  if (elementType.isF32())
    return "vsqueeze_float";
  if (elementType.isF16())
    return "vsqueeze_half";
  if (auto intType = dyn_cast<IntegerType>(elementType))
    if (intType.getWidth() == 32)
      return "vsqueeze_int32_t";
  return {};
}

static func::FuncOp getOrCreateSqueezeLibraryCall(ModuleOp module, Location loc,
                                                  VectorType vecType, VectorType pregType,
                                                  StringRef calleeName) {
  if (auto existing = module.lookupSymbol<func::FuncOp>(calleeName))
    return existing;
  OpBuilder moduleBuilder(module.getBodyRegion());
  auto fnType = FunctionType::get(module.getContext(), {vecType, pregType}, {vecType});
  auto callee = moduleBuilder.create<func::FuncOp>(loc, calleeName, fnType);
  callee.setPrivate();
  callee->setAttr("llvm.emit_c_interface", UnitAttr::get(module.getContext()));
  return callee;
}

static Value castMaskToPregType(OpBuilder &b, Location loc, Value mask,
                                VectorType pregVecType) {
  if (mask.getType() == pregVecType)
    return mask;
  return b.create<UnrealizedConversionCastOp>(loc, pregVecType, mask).getResult(0);
}

static VectorType fullPregVecType(MLIRContext *ctx) {
  return VectorType::get({256}, IntegerType::get(ctx, 1));
}

static hivmave::MaskWidthAttr maskWidthAttrForElement(OpBuilder &b, Type elementType) {
  return hivmave::MaskWidthAttr::get(b.getContext(), maskWidthForElement(elementType));
}

// The semantic width of a MaskSSA is carried by !tla.mask<N>, independently
// of its lowered predicate-register container. A carrier crossing SCF uses the
// backend-native vector<256xi1> container, but !tla.mask<64> must still select
// B32 rather than being misclassified as B8 from that container width.
static hivmave::MaskWidth maskWidthForMaskType(::tla::MaskSSAType maskType) {
  int64_t lanes = maskType.getPhysicalLanes();
  if (lanes <= 0)
    return hivmave::MaskWidth::B32;
  int64_t bytesPerLane = 256 / lanes;
  if (bytesPerLane <= 1)
    return hivmave::MaskWidth::B8;
  if (bytesPerLane <= 2)
    return hivmave::MaskWidth::B16;
  return hivmave::MaskWidth::B32;
}

static hivmave::MaskWidthAttr maskWidthAttrForMaskType(
    OpBuilder &b, ::tla::MaskSSAType maskType) {
  return hivmave::MaskWidthAttr::get(b.getContext(),
                                     maskWidthForMaskType(maskType));
}

// Build the AVE vector op for a tla binary op. The mask controls active lanes.
// For div the signedness is carried as the TypeFn cast attribute (cast_unsigned
// for unsigned integer element types, cast_signed otherwise).
static Value createVectorBinaryResult(OpBuilder &b, Location loc, VectorBinaryKind kind,
                                      Type tlaOperandType, Type elementType,
                                      VectorType vecType, Value lhs, Value rhs,
                                      Value mask) {
  switch (kind) {
  case VectorBinaryKind::Add:
    return b.create<hivmave::VFAddOp>(loc, vecType, lhs, rhs, mask, Value()).getResult();
  case VectorBinaryKind::Sub:
    return b.create<hivmave::VFSubOp>(loc, vecType, lhs, rhs, mask, Value()).getResult();
  case VectorBinaryKind::Mul:
    return b.create<hivmave::VFMulOp>(loc, vecType, lhs, rhs, mask, Value()).getResult();
  case VectorBinaryKind::Div: {
    auto cast = hivm::TypeFn::cast_signed;
    if (auto intType = dyn_cast<IntegerType>(elementType))
      if (intType.getSignedness() == IntegerType::Unsigned)
        cast = hivm::TypeFn::cast_unsigned;
    return b.create<hivmave::VFDivOp>(loc, vecType, lhs, rhs, mask,
                                      hivm::TypeFnAttr::get(b.getContext(), cast), Value())
        .getResult();
  }
  case VectorBinaryKind::Max:
    return b.create<hivmave::VFMaxOp>(loc, vecType, lhs, rhs, mask, Value()).getResult();
  case VectorBinaryKind::Min:
    return b.create<hivmave::VFMinOp>(loc, vecType, lhs, rhs, mask, Value()).getResult();
  case VectorBinaryKind::And:
    if (isa<::tla::VectorSSAType>(tlaOperandType))
      return b.create<hivmave::VFAndOp>(loc, vecType, lhs, rhs, mask, Value())
          .getResult();
    if (isa<::tla::MaskSSAType>(tlaOperandType))
      return b.create<hivmave::PregAndOp>(
                   loc, vecType,
                   maskWidthAttrForMaskType(
                       b, cast<::tla::MaskSSAType>(tlaOperandType)),
                   lhs, rhs, mask)
          .getRes();
    return nullptr;
  case VectorBinaryKind::Or:
    if (isa<::tla::VectorSSAType>(tlaOperandType))
      return b.create<hivmave::VFOrOp>(loc, vecType, lhs, rhs, mask, Value()).getResult();
    if (isa<::tla::MaskSSAType>(tlaOperandType))
      return b.create<hivmave::PregOrOp>(
                   loc, vecType,
                   maskWidthAttrForMaskType(
                       b, cast<::tla::MaskSSAType>(tlaOperandType)),
                   lhs, rhs, mask)
          .getRes();
    return nullptr;
  case VectorBinaryKind::Xor:
    if (isa<::tla::VectorSSAType>(tlaOperandType))
      return b.create<hivmave::VFXorOp>(loc, vecType, lhs, rhs, mask, Value())
          .getResult();
    if (isa<::tla::MaskSSAType>(tlaOperandType))
      return b.create<hivmave::PregXorOp>(
                   loc, vecType,
                   maskWidthAttrForMaskType(
                       b, cast<::tla::MaskSSAType>(tlaOperandType)),
                   lhs, rhs, mask)
          .getRes();
    return nullptr;
  }
  return nullptr;
}

static int64_t maskElementBitWidthForLanes(int64_t lanes) {
  if (lanes >= 256)
    return 8;
  if (lanes >= 128)
    return 16;
  return 32;
}

// AVE represents a semantic vector<Nxi1> predicate as a hardware
// vector<256xi1>. When those types differ, annotate the AVE producer with the
// semantic element width so HIVMAVE lowering still selects pge/plt.b8/b16/b32
// from N instead of the full container width.
static void annotateFullPregWidth(OpBuilder &b, Operation *op,
                                  VectorType resultType,
                                  int64_t semanticLanes) {
  if (resultType.getNumElements() == semanticLanes)
    return;
  op->setAttr(mlir::utils::elementAlignmentBitWidth,
              b.getI32IntegerAttr(maskElementBitWidthForLanes(semanticLanes)));
}

static Value createPredicatePge(OpBuilder &b, Location loc,
                                VectorType resultType,
                                int64_t semanticLanes,
                                hivmave::PgePattern pattern) {
  auto pge = createAvePgeMask(b, loc, resultType, pattern);
  annotateFullPregWidth(b, pge, resultType, semanticLanes);
  return pge.getRes();
}

static hivmave::VFPltOp createPredicatePlt(OpBuilder &b, Location loc,
                                           VectorType resultType,
                                           int64_t semanticLanes,
                                           Value trueShape) {
  auto plt = createAvePltMask(b, loc, resultType, trueShape);
  annotateFullPregWidth(b, plt, resultType, semanticLanes);
  return plt;
}

// An all-lanes-active predicate for a data or MaskSSA vector. MaskSSA keeps its
// semantic lane count in tlaOperandType even when its mapped value is the full
// predicate-register container.
static Value allTrueMaskFor(OpBuilder &b, Location loc, VectorType vecType,
                            Type tlaOperandType, bool useFullPreg) {
  int64_t semanticLanes = vecType.getNumElements();
  if (auto maskType = dyn_cast<::tla::MaskSSAType>(tlaOperandType))
    semanticLanes = maskType.getPhysicalLanes();
  VectorType maskType =
      useFullPreg ? fullPregVecType(b.getContext())
                  : VectorType::get({semanticLanes}, b.getI1Type());
  return createPredicatePge(b, loc, maskType, semanticLanes,
                            hivmave::PgePattern::ALL);
}

// Map the tla.cast round mode onto the HIVM round_mode attribute.
static hivm::RoundModeAttr mapCastRoundMode(OpBuilder &b, ::RoundMode mode) {
  hivm::RoundMode hv = hivm::RoundMode::ROUND;
  switch (mode) {
  case ::RoundMode::cast_round: hv = hivm::RoundMode::ROUND; break;
  case ::RoundMode::cast_floor: hv = hivm::RoundMode::FLOOR; break;
  case ::RoundMode::cast_ceil: hv = hivm::RoundMode::CEIL; break;
  case ::RoundMode::cast_trunc: hv = hivm::RoundMode::TRUNC; break;
  }
  return hivm::RoundModeAttr::get(b.getContext(), hv);
}

// Map the tla.cast register layout onto the AVE VCVT part (even/odd) attribute.
static hivmave::VCVT_PartTypeAttr mapCastPart(OpBuilder &b, ::RegSlot layout) {
  auto part = layout == ::RegSlot::one ? hivmave::VCVT_PartType::PART_ODD
                                              : hivmave::VCVT_PartType::PART_EVEN;
  return hivmave::VCVT_PartTypeAttr::get(b.getContext(), part);
}

// Map the tla.cast register layout onto the AVE pack pattern (pp0..pp3) used by
// 4x-width int casts (i32<->i8). reg_slot zero/one/two/three -> pp0/pp1/pp2/pp3.
static hivmave::VCVT_PPTypeAttr mapCastPP(OpBuilder &b, ::RegSlot layout) {
  hivmave::VCVT_PPType pp;
  switch (layout) {
  case ::RegSlot::one: pp = hivmave::VCVT_PPType::PP1; break;
  case ::RegSlot::two: pp = hivmave::VCVT_PPType::PP2; break;
  case ::RegSlot::three: pp = hivmave::VCVT_PPType::PP3; break;
  case ::RegSlot::zero:
  default: pp = hivmave::VCVT_PPType::PP0; break;
  }
  return hivmave::VCVT_PPTypeAttr::get(b.getContext(), pp);
}

// Element types the tla.cast lowering can emit AVE ops for: signed/signless
// integers i8/i16/i32/i64 and floats f16/bf16/f32. Unsigned integers, i1 (bool)
// and f64 have no AVE cast path and are rejected (the front-end rejects them too;
// this guards hand-written / non-front-end IR).
static bool isSupportedCastElementType(Type t) {
  if (auto f = dyn_cast<FloatType>(t))
    return f.getWidth() == 16 || f.getWidth() == 32;  // f16/bf16/f32, not f64
  if (auto i = dyn_cast<IntegerType>(t)) {
    if (i.isUnsigned() || i.getWidth() == 1)  // unsigned / bool
      return false;
    unsigned w = i.getWidth();
    return w == 8 || w == 16 || w == 32 || w == 64;
  }
  return false;
}

// Build the AVE cast op for a tla.cast, dispatching by (src, dst) element kind.
// The trait supplies rounding, saturation and register layout; the mask (source
// width) predicates active lanes.
static FailureOr<Value> createVectorCastResult(OpBuilder &b, Location loc,
                                               VectorType srcVecType,
                                               VectorType dstVecType,
                                               ArrayRef<int32_t> trait, Value src,
                                               Value mask) {
  // trait codes: [0] reg_slot, [1] sat_mode, [2] round_mode.
  Type s = srcVecType.getElementType();
  Type d = dstVecType.getElementType();
  auto rnd = mapCastRoundMode(b, static_cast<::RoundMode>(trait[2]));
  BoolAttr sat = b.getBoolAttr(static_cast<::SatMode>(trait[1]) == ::SatMode::sat);
  auto part = mapCastPart(b, static_cast<::RegSlot>(trait[0]));

  bool sFloat = isa<FloatType>(s);
  bool dFloat = isa<FloatType>(d);
  unsigned sb = s.getIntOrFloatBitWidth();
  unsigned db = d.getIntOrFloatBitWidth();
  // For same-width float<->int conversions the packed even/odd part does not
  // apply (src and dst occupy the full register); pass a null part attribute,
  // matching the arith->AVE lowering.
  hivmave::VCVT_PartTypeAttr partOrNull = (sb == db) ? hivmave::VCVT_PartTypeAttr() : part;

  if (sFloat && dFloat) {
    if (db < sb)
      return b.create<hivmave::VFTruncFOp>(loc, dstVecType, src, mask, rnd, sat, part)
          .getResult();
    // Widening float cast (e.g. f16 -> f32) takes no rounding/saturation.
    return b.create<hivmave::VFExtFOp>(loc, dstVecType, src, mask, part).getResult();
  }
  if (sFloat && !dFloat)
    return b.create<hivmave::VFFpToSIntOp>(loc, dstVecType, src, mask, rnd, sat, partOrNull)
        .getResult();
  if (!sFloat && dFloat) {
    // int -> float: the ISA does not allow #rnd and #part together. A same-width
    // source carries the round mode (rounding may be needed, e.g. i32->f32); a
    // width-changing widen/narrow carries the even/odd part with no round mode
    // (i16->f32 is exact). i64 sources carry both, matching the arith lowering.
    if (sb == db)
      return b.create<hivmave::VFSIntToFpOp>(loc, dstVecType, src, mask, rnd,
                                             hivmave::VCVT_PartTypeAttr())
          .getResult();
    if (sb == 64)
      return b.create<hivmave::VFSIntToFpOp>(loc, dstVecType, src, mask, rnd, part)
          .getResult();
    return b.create<hivmave::VFSIntToFpOp>(loc, dstVecType, src, mask,
                                           hivm::RoundModeAttr(), part)
        .getResult();
  }
  // int -> int (signed). A 2x width step (e.g. i32<->i16, i16<->i8) uses the
  // even/odd `part`; a 4x step (i32<->i8) uses the pack-pattern `pp` (PP0)
  // instead, matching the arith->AVE lowering. Integer casts do not round.
  auto uni = hivm::UnsignedModeAttr::get(b.getContext(), hivm::UnsignedMode::SI2SI);
  auto pp = mapCastPP(b, static_cast<::RegSlot>(trait[0]));
  if (db < sb) {
    if (sb / db >= 4)
      return b.create<hivmave::VFTruncIOp>(loc, dstVecType, src, mask, sat,
                                           hivmave::VCVT_PartTypeAttr(), pp,
                                           hivm::UnsignedModeAttr())
          .getResult();
    return b.create<hivmave::VFTruncIOp>(loc, dstVecType, src, mask, sat, part,
                                         hivmave::VCVT_PPTypeAttr(), uni)
        .getResult();
  }
  if (db / sb >= 4)
    return b.create<hivmave::VFExtSIOp>(loc, dstVecType, src, mask,
                                        hivmave::VCVT_PartTypeAttr(), pp)
        .getResult();
  return b.create<hivmave::VFExtSIOp>(loc, dstVecType, src, mask, part,
                                      hivmave::VCVT_PPTypeAttr())
      .getResult();
}

static FailureOr<Value> createVectorReductionResult(OpBuilder &b, Location loc,
                                                    ::tla::ReduceOp reduceOp,
                                                    Type elementType,
                                                    VectorType vecType,
                                                    Value operand,
                                                    Value explicitMask) {
  if (failed(validateVectorReduction(reduceOp, elementType)))
    return failure();
  auto aveKind = getAveReductionCombiningKind(reduceOp, elementType);
  if (failed(aveKind))
    return failure();
  // The active mask is supplied explicitly by the frontend; tla.reduce no longer
  // derives one from the operand's originShape.
  if (!explicitMask)
    return reduceOp.emitError("tla.reduce requires an explicit mask"), failure();

  return b.create<hivmave::ReductionOp>(loc, vecType, *aveKind, operand, explicitMask).getResult();
}

static Value createVectorUnaryResult(OpBuilder &b, Location loc, VectorUnaryKind kind,
                                     Type tlaOperandType, VectorType vecType,
                                     Value operand, Value mask) {
  switch (kind) {
  case VectorUnaryKind::Exp:
    return b.create<hivmave::VFExpOp>(loc, vecType, operand, mask, Value()).getResult();
  case VectorUnaryKind::Log:
    return b.create<hivmave::VFLnOp>(loc, vecType, operand, mask, Value()).getResult();
  case VectorUnaryKind::Sqrt:
    return b.create<hivmave::VFSqrtOp>(loc, vecType, operand, mask, Value()).getResult();
  case VectorUnaryKind::Abs:
    return b.create<hivmave::VFAbsOp>(loc, vecType, operand, mask, Value()).getResult();
  case VectorUnaryKind::Neg:
    return b.create<hivmave::VFNegOp>(loc, vecType, operand, mask, Value()).getResult();
  case VectorUnaryKind::Not:
    if (isa<::tla::VectorSSAType>(tlaOperandType))
      return b.create<hivmave::VFNotOp>(loc, vecType, operand, mask, Value()).getResult();
    if (isa<::tla::MaskSSAType>(tlaOperandType))
      return b.create<hivmave::PregNotOp>(
                   loc, vecType,
                   maskWidthAttrForMaskType(
                       b, cast<::tla::MaskSSAType>(tlaOperandType)),
                   operand, mask)
          .getRes();
    return nullptr;
  }
  return nullptr;
}


// The per-op vector width bundle (one 256-byte register's worth of a given
// element type). Derived fresh for each op from its own operands/result rather
// than shared across the region, so a single vec.func body can mix element
// widths (as tla.cast requires).
struct VecLowerCtx {
  int64_t lanes;
  Type elementType;
  VectorType vecType;
  VectorType maskVecType;
};

// Build the per-op {lanes, elementType, vecType, maskVecType} for a given
// element type. Each op derives its own types this way rather than reusing a
// region-global width: a tla.cast may have produced operands whose element
// width (hence lane count, at a fixed 256-byte register) differs from the
// region's, and same-256-byte register can hold f32 (64), f16 (128) or i8
// (256) lanes.
static FailureOr<VecLowerCtx> deriveVecCtxForElement(Type elementType,
                                                     bool useFullPreg) {
  auto lanesOr = getVectorLaneCount(elementType);
  if (failed(lanesOr) || *lanesOr <= 0)
    return failure();
  int64_t lanes = *lanesOr;
  auto i1Type = IntegerType::get(elementType.getContext(), 1);
  auto maskVecType = useFullPreg ? fullPregVecType(elementType.getContext())
                                 : VectorType::get({lanes}, i1Type);
  return VecLowerCtx{lanes, elementType,
                     VectorType::get({lanes}, elementType), maskVecType};
}

// Return the value already mapped into the helper, or clone an arith.constant
// on demand (loop bounds / index math constants are pulled in lazily this way).
static Value lookupOrCloneScalarValue(OpBuilder &b, Value value,
                                      DenseMap<Value, Value> &valueMap) {
  if (Value mapped = valueMap.lookup(value))
    return mapped;
  Operation *def = value.getDefiningOp();
  if (!def || def->getNumResults() != 1 || !isa<arith::ConstantOp>(def))
    return nullptr;
  Operation *cloned = b.clone(*def);
  valueMap[value] = cloned->getResult(0);
  return cloned->getResult(0);
}

// Materialize the valid-lane count of a tla.tensor as an index SSA value for the
// active mask. Falls back to the producing tla.tensor_desc's origin_shape0*origin_shape1,
// mapped into the helper via lookupOrCloneScalarValue (vec.func-external scalars are
// helper args; in-region index arithmetic is cloned ahead of the descriptor).
static FailureOr<Value> getTlaTensorValidLaneCount(OpBuilder &b, Location loc,
                                                           Value tensorValue,
                                                           DenseMap<Value, Value> &valueMap) {
  if (auto descOp = findTensorDescProducer(tensorValue)) {
    Value origin0 = lookupOrCloneScalarValue(b, descOp.getOriginShape0(), valueMap);
    Value origin1 = lookupOrCloneScalarValue(b, descOp.getOriginShape1(), valueMap);
    if (!origin0 || !origin1)
      return failure();
    return b.create<arith::MulIOp>(loc, origin0, origin1).getResult();
  }
  return failure();
}

static FailureOr<Value> castScalarForVectorElement(Value scalar, Type elementType) {
  if (scalar.getType() == elementType)
    return scalar;
  return failure();
}

static FailureOr<Value> materializeVectorScalarValue(OpBuilder &b, TlaBinaryOperands operands,
                                                     DenseMap<Value, Value> &valueMap,
                                                     VecLowerCtx &ctx) {
  Value scalar = lookupOrCloneScalarValue(b, operands.rhs, valueMap);
  if (!scalar)
    return failure();
  auto castScalar = castScalarForVectorElement(scalar, ctx.elementType);
  if (failed(castScalar))
    return failure();
  return *castScalar;
}

static FailureOr<Value> createVectorScalarBinaryResult(OpBuilder &b, Location loc,
                                                       VectorOpInfo info,
                                                       VecLowerCtx &ctx, Value lhs,
                                                       Value scalar, Value mask) {
  if (info.kind == VectorBinaryKind::Add || info.kind == VectorBinaryKind::Mul ||
      info.kind == VectorBinaryKind::Max || info.kind == VectorBinaryKind::Min) {
    if (info.kind == VectorBinaryKind::Add)
      return b.create<hivmave::VFAddsOp>(loc, ctx.vecType, lhs, scalar, mask, Value())
          .getResult();
    if (info.kind == VectorBinaryKind::Mul)
      return b.create<hivmave::VFMulsOp>(loc, ctx.vecType, lhs, scalar, mask, Value())
          .getResult();
    if (info.kind == VectorBinaryKind::Max)
      return b.create<hivmave::VFMaxsOp>(loc, ctx.vecType, lhs, scalar, mask, Value())
          .getResult();
    return b.create<hivmave::VFMinsOp>(loc, ctx.vecType, lhs, scalar, mask, Value())
        .getResult();
  }

  Value rhs =
      b.create<hivmave::VFBroadcastScalarOp>(loc, ctx.vecType, scalar).getRes();
  return createVectorBinaryResult(b, loc, info.kind, info.operands.lhs.getType(),
                                  ctx.elementType, ctx.vecType, lhs, rhs, mask);
}

static FailureOr<Type> lowerSCFCarrierType(Type type) {
  if (auto vectorType = dyn_cast<::tla::VectorSSAType>(type)) {
    auto ctx = deriveVecCtxForElement(vectorType.getElementType(), false);
    if (failed(ctx)) return failure();
    return Type(ctx->vecType);
  }
  if (isa<::tla::MaskSSAType>(type))
    return Type(fullPregVecType(type.getContext()));
  return type;
}

static LogicalResult lowerNestedVectorBlock(Block *sourceBlock, OpBuilder &b,
                                            ModuleOp module,
                                            DenseMap<Value, Value> &valueMap,
                                            bool useFullPreg);

// Re-create one vec.func body op inside the helper: tla ops become AVE vector
// ops; scf control flow and index arithmetic are carried verbatim. Each op
// derives its own vector/mask width from its operands or result element type,
// so a single region may mix element widths (e.g. across tla.cast).
static LogicalResult lowerNestedVectorOp(Operation &op, OpBuilder &b, ModuleOp module,
                                         DenseMap<Value, Value> &valueMap,
                                         bool useFullPreg) {
  Location loc = op.getLoc();

  // make_shape / make_coord are dead after tla-lower-tensor-desc (their leaves
  // were folded into tensor_desc operands); skip them. (tla-finalize-memref
  // erases them.)
  if (isa<::tla::MakeShapeOp, ::tla::MakeCoordOp>(op))
    return success();

  if (auto constant = dyn_cast<arith::ConstantOp>(op)) {
    valueMap[constant.getResult()] = b.clone(op)->getResult(0);
    return success();
  }

  // tla.tensor_desc (produced by tla-lower-tensor-desc, the sole descriptor
  // producer): consume it directly. The descriptor's row_offset / col_offset /
  // stride0 already encode the tile's position and pitch, so there is no
  // tile_view -> make_tensor producer chain to re-walk. Carve a lanes-wide
  // (256-byte) flat subview of the helper's base-memref arg at the flat offset
  // row_offset * stride0 + col_offset.
  if (auto descOp = dyn_cast<::tla::TensorDescOp>(op)) {
    Value baseMemref = valueMap.lookup(descOp.getResult());
    if (!baseMemref)
      return descOp.emitError("failed to map tla.tensor_desc base in vector helper"),
             failure();
    auto sourceType = dyn_cast<MemRefType>(baseMemref.getType());
    if (!sourceType || sourceType.getRank() != 1)
      return descOp.emitError("expected rank-1 base memref for vector tensor_desc"),
             failure();
    auto lanesOr = getVectorLaneCount(sourceType.getElementType());
    if (failed(lanesOr))
      return descOp.emitError("unsupported element type for vector tensor_desc"),
             failure();
    Value rowOff = lookupOrCloneScalarValue(b, descOp.getRowOffset(), valueMap);
    Value colOff = lookupOrCloneScalarValue(b, descOp.getColOffset(), valueMap);
    Value stride0 = lookupOrCloneScalarValue(b, descOp.getStride0(), valueMap);
    if (!rowOff || !colOff || !stride0)
      return failure();
    Value flatOffset = b.create<arith::AddIOp>(
        loc, b.create<arith::MulIOp>(loc, rowOff, stride0), colOff);
    valueMap[descOp.getResult()] =
        ::tla::materializeFlatReinterpretSubview(b, loc, baseMemref, flatOffset, *lanesOr);
    return success();
  }

  if (auto loadOp = dyn_cast<::tla::LoadOp>(op)) {
    Value source = valueMap.lookup(loadOp.getSource());
    if (!source)
      return failure();
    // The loaded vector's element type comes from the tile memref, not the
    // region-global width: a load feeding a differently-typed op keeps its own
    // dtype.
    auto sourceType = dyn_cast<MemRefType>(source.getType());
    if (!sourceType)
      return failure();
    auto opCtx = deriveVecCtxForElement(sourceType.getElementType(), useFullPreg);
    if (failed(opCtx))
      return failure();
    Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
    hivmave::LoadDist pattern = hivmave::LoadDist::NORM;
    if (auto loadDistAttr = loadOp.getLoadDist())
      pattern = mapTlaLoadDistToAve(loadDistAttr->getLoadDist());
    bool dual = isDualDestLoadDist(pattern);
    if (dual != static_cast<bool>(loadOp.getResult2()))
      return loadOp.emitError(
                 "dintlv load_dist requires exactly two results; other "
                 "load_dist values require one"),
             failure();
    // DINTLV_* still takes a VL-wide tile view (same as AVE ProcessVsstb /
    // HIVM2VLLoadOpLowering): the distribution pattern reads 2*VL from that
    // base address. Do not widen the memref to 2*VL — that breaks hivmc ABI.
    auto vfLoad = createVFLoad(b, loc, opCtx->vecType, source, zero, pattern,
                               loadOp.getUnalignedUbAccess().value_or(false));
    valueMap[loadOp.getResult()] = vfLoad.getRes();
    if (dual)
      valueMap[loadOp.getResult2()] = vfLoad.getRes1();
    return success();
  }

  // Preserve local-memory ordering while outlining a vec.func into its AVE
  // helper. Every operation in the source region must be recreated explicitly;
  // otherwise helper construction fails and leaves the whole vec.func behind.
  if (auto localMemBarOp = dyn_cast<::tla::LocalMemBarOp>(op)) {
    return lowerLocalMemBar(b, localMemBarOp);
  }

  // tla.cast: element-type conversion. The source vector already carries its
  // width; the destination width is one full 256-byte register's worth of the
  // target element type. The cast op picks the AVE cast (vtruncf / vfptosi /
  // vsitofp / vtrunci / ...) from the (src,dst) element kinds.
  if (auto castOp = dyn_cast<::tla::CastOp>(op)) {
    Value src = valueMap.lookup(castOp.getSource());
    if (!src)
      return failure();
    auto srcVecType = dyn_cast<VectorType>(src.getType());
    if (!srcVecType)
      return castOp.emitError("tla.cast source is not a vector value"), failure();
    auto dstType = dyn_cast<::tla::VectorSSAType>(castOp.getResult().getType());
    if (!dstType)
      return castOp.emitError("expected !tla.vector result type"), failure();
    auto dstLanesOr = getVectorLaneCount(dstType.getElementType());
    if (failed(dstLanesOr))
      return castOp.emitError("unsupported tla.cast destination element type"), failure();
    auto dstVecType = VectorType::get({*dstLanesOr}, dstType.getElementType());
    // Reject casts whose source or destination element type has no AVE cast path
    // (unsigned integers, i1/bool, f64) rather than emitting invalid AVE IR.
    if (!isSupportedCastElementType(srcVecType.getElementType()) ||
        !isSupportedCastElementType(dstVecType.getElementType()))
      return castOp.emitError("unsupported tla.cast element type: only signed "
                              "integers (i8/i16/i32/i64) and floats (f16/bf16/f32) "
                              "are supported; unsigned, bool and f64 are not"),
             failure();
    ArrayRef<int32_t> trait = castOp.getTrait();
    if (trait.size() != 3)
      return castOp.emitError("tla.cast trait must have 3 codes"), failure();
    // An optional mask predicates the source lanes of the AVE cast; all-true when
    // none is given.
    Value mask;
    if (castOp.getMask()) {
      mask = valueMap.lookup(castOp.getMask());
      if (!mask)
        return failure();
    } else {
      mask = allTrueMaskFor(b, loc, srcVecType,
                            castOp.getSource().getType(), useFullPreg);
    }
    auto result =
        createVectorCastResult(b, loc, srcVecType, dstVecType, trait, src, mask);
    if (failed(result))
      return castOp.emitError("unsupported tla.cast element type conversion"), failure();
    valueMap[castOp.getResult()] = *result;
    return success();
  }

  if (auto fullOp = dyn_cast<::tla::FullOp>(op)) {
    Value source = lookupOrCloneScalarValue(b, fullOp.getValue(), valueMap);
    if (!source)
      return failure();
    // No vector operand to key off: the broadcast width comes from the result
    // VectorSSA element type.
    auto resultType = dyn_cast<::tla::VectorSSAType>(fullOp.getResult().getType());
    if (!resultType)
      return fullOp.emitError("expected !tla.vector result type"), failure();
    auto opCtx = deriveVecCtxForElement(resultType.getElementType(), useFullPreg);
    if (failed(opCtx))
      return fullOp.emitError("unsupported tla.full result element type"), failure();
    if (source.getType() != opCtx->elementType)
      return fullOp.emitError("tla.full scalar type ")
                 << source.getType() << " does not match vector element type "
                 << opCtx->elementType,
             failure();
    valueMap[fullOp.getResult()] =
        b.create<hivmave::VFBroadcastScalarOp>(loc, opCtx->vecType, source).getRes();
    return success();
  }

  if (auto arangeOp = dyn_cast<::tla::ArangeOp>(op)) {
    Value start = lookupOrCloneScalarValue(b, arangeOp.getStart(), valueMap);
    if (!start)
      return failure();
    // Width comes from the result VectorSSA element type.
    auto resultType = dyn_cast<::tla::VectorSSAType>(arangeOp.getResult().getType());
    if (!resultType)
      return arangeOp.emitError("expected !tla.vector result type"), failure();
    auto opCtx = deriveVecCtxForElement(resultType.getElementType(), useFullPreg);
    if (failed(opCtx))
      return arangeOp.emitError("unsupported tla.arange result element type"), failure();
    if (isa<FloatType>(opCtx->elementType))
      return arangeOp.emitError("tla.arange does not support floating-point element types"),
             failure();
    if (start.getType() != opCtx->elementType)
      return arangeOp.emitError("tla.arange start type ")
                 << start.getType() << " does not match vector element type "
                 << opCtx->elementType,
             failure();
    auto vciType = hivmave::VCIType::INCREASE;
    if (arangeOp.getOrder() == "decrease")
      vciType = hivmave::VCIType::DECREASE;
    else if (arangeOp.getOrder() != "increase")
      return arangeOp.emitError("unsupported tla.arange order: ")
                 << arangeOp.getOrder(),
             failure();
    valueMap[arangeOp.getResult()] =
        b.create<hivmave::VFVCIOp>(
             loc, opCtx->vecType, start,
             hivmave::VCITypeAttr::get(b.getContext(), vciType))
            .getRes();
    return success();
  }

  if (auto info = getVectorBinaryInfo(&op)) {
    if (op.getNumResults() != 1)
      return failure();
    TlaBinaryOperands operands = info->operands;
    Value lhs = valueMap.lookup(operands.lhs);
    if (!lhs)
      return failure();
    Value rhs = valueMap.lookup(operands.rhs);
    if (!rhs)
      return failure();
    // Derive the vector width from the operands: a cast may have produced a
    // vector of a different lane width than the enclosing region's element type.
    auto opVecType = dyn_cast<VectorType>(lhs.getType());
    if (!opVecType)
      return failure();
    Type opElemType = opVecType.getElementType();
    Value mask;
    if (operands.mask) {
      mask = valueMap.lookup(operands.mask);
      if (!mask)
        return failure();
    } else {
      mask = allTrueMaskFor(b, loc, opVecType, operands.lhs.getType(),
                            useFullPreg);
    }
    Value result = createVectorBinaryResult(b, loc, info->kind, operands.lhs.getType(),
                                            opElemType, opVecType, lhs, rhs, mask);
    if (!result)
      return failure();
    valueMap[op.getResult(0)] = result;
    return success();
  }

  if (auto info = getVectorScalarBinaryInfo(&op)) {
    if (op.getNumResults() != 1)
      return failure();
    TlaBinaryOperands operands = info->operands;
    Value lhs = valueMap.lookup(operands.lhs);
    if (!lhs)
      return failure();
    // Element type follows the lhs vector operand, not the region width.
    auto lhsTy = dyn_cast<VectorType>(lhs.getType());
    if (!lhsTy)
      return failure();
    auto opCtx = deriveVecCtxForElement(lhsTy.getElementType(), useFullPreg);
    if (failed(opCtx))
      return failure();
    auto scalarOr = materializeVectorScalarValue(b, operands, valueMap, *opCtx);
    if (failed(scalarOr))
      return failure();
    Value mask;
    if (operands.mask) {
      mask = valueMap.lookup(operands.mask);
      if (!mask)
        return failure();
    } else {
      mask = createPredicatePge(b, loc, opCtx->maskVecType, opCtx->lanes,
                                 hivmave::PgePattern::ALL);
    }
    auto result = createVectorScalarBinaryResult(b, loc, *info, *opCtx, lhs, *scalarOr, mask);
    if (failed(result))
      return failure();
    valueMap[op.getResult(0)] = *result;
    return success();
  }

  // tla.where: per-lane select. The mask controls which lanes take `x`; the
  // remaining lanes take `y`. Lowers to ave.hir.vsel(mask, x, y).
  if (auto whereOp = dyn_cast<::tla::WhereOp>(op)) {
    Value mask = valueMap.lookup(whereOp.getMask());
    Value x = valueMap.lookup(whereOp.getX());
    Value y = valueMap.lookup(whereOp.getY());
    if (!mask || !x || !y)
      return failure();
    // Result width follows the selected vectors' element type.
    auto xTy = dyn_cast<VectorType>(x.getType());
    if (!xTy)
      return failure();
    auto opCtx = deriveVecCtxForElement(xTy.getElementType(), useFullPreg);
    if (failed(opCtx))
      return failure();
    valueMap[whereOp.getResult()] =
        b.create<hivmave::VFSelectOp>(loc, opCtx->vecType, mask, x, y);
    return success();
  }

  // tla.squeeze: mask-compress src lanes via linked bitcode (vsqz). Uses
  // NO_STORE_REG; STORE_REG + StoreUnAlign streaming writeback is not exposed
  // until unaligned store (StoreUnAlign/StoreUnAlignPost) is available in TLA.
  if (auto squeezeOp = dyn_cast<::tla::SqueezeOp>(op)) {
    Value src = valueMap.lookup(squeezeOp.getSrc());
    Value mask = valueMap.lookup(squeezeOp.getMask());
    if (!src || !mask)
      return failure();
    auto srcTy = dyn_cast<VectorType>(src.getType());
    if (!srcTy)
      return failure();
    auto opCtx = deriveVecCtxForElement(srcTy.getElementType(), useFullPreg);
    if (failed(opCtx))
      return failure();
    std::string calleeName = getSqueezeLibraryCallName(srcTy.getElementType());
    if (calleeName.empty())
      return squeezeOp.emitError("unsupported element type for tla.squeeze: ")
             << srcTy.getElementType(), failure();
    VectorType pregVecType = fullPregVecType(b.getContext());
    Value preg = castMaskToPregType(b, loc, mask, pregVecType);
    auto callee = getOrCreateSqueezeLibraryCall(module, loc, opCtx->vecType, pregVecType,
                                                calleeName);
    Value result = b.create<func::CallOp>(loc, callee, ValueRange{src, preg}).getResult(0);
    valueMap[squeezeOp.getResult()] = result;
    return success();
  }

  if (auto reduceOp = dyn_cast<::tla::ReduceOp>(op)) {
    if (op.getNumResults() != 1)
      return failure();
    Value operand = valueMap.lookup(reduceOp->getOperand(0));
    if (!operand)
      return failure();
    // Reduction width follows the operand vector's element type.
    auto operandTy = dyn_cast<VectorType>(operand.getType());
    if (!operandTy)
      return failure();
    auto opCtx = deriveVecCtxForElement(operandTy.getElementType(), useFullPreg);
    if (failed(opCtx))
      return failure();
    Value mask;
    if (reduceOp.getMask()) {
      mask = valueMap.lookup(reduceOp.getMask());
      if (!mask)
        return failure();
    }
    auto result = createVectorReductionResult(b, loc, reduceOp, opCtx->elementType,
                                              opCtx->vecType, operand, mask);
    if (failed(result))
      return failure();
    valueMap[op.getResult(0)] = *result;
    return success();
  }

  if (auto interleaveOp = dyn_cast<::tla::InterleaveOp>(op)) {
    if (op.getNumResults() != 2)
      return failure();

    Value src0 = valueMap.lookup(interleaveOp.getSrc0());
    Value src1 = valueMap.lookup(interleaveOp.getSrc1());
    if (!src0 || !src1)
      return failure();

    auto src0Type = dyn_cast<VectorType>(src0.getType());
    auto src1Type = dyn_cast<VectorType>(src1.getType());
    if (!src0Type || !src1Type || src0Type != src1Type)
      return failure();

    auto aveOp = b.create<hivmave::VFInterleaveOp>(
      loc,
      TypeRange{src0Type, src1Type},
      ValueRange{src0, src1});

    valueMap[op.getResult(0)] = aveOp->getResult(0);
    valueMap[op.getResult(1)] = aveOp->getResult(1);
    return success();
  }

  if (auto deInterleaveOp = dyn_cast<::tla::DeInterleaveOp>(op)) {
    if (op.getNumResults() != 2)
      return failure();

    Value src0 = valueMap.lookup(deInterleaveOp.getSrc0());
    Value src1 = valueMap.lookup(deInterleaveOp.getSrc1());
    if (!src0 || !src1)
      return failure();

    auto src0Type = dyn_cast<VectorType>(src0.getType());
    auto src1Type = dyn_cast<VectorType>(src1.getType());
    if (!src0Type || !src1Type || src0Type != src1Type)
      return failure();

    auto aveOp = b.create<hivmave::VFDeInterleaveOp>(
      loc,
      TypeRange{src0Type, src1Type},
      ValueRange{src0, src1});

    valueMap[op.getResult(0)] = aveOp->getResult(0);
    valueMap[op.getResult(1)] = aveOp->getResult(1);
    return success();
  }

  // tla.gather: per-lane indexed load from a UB tile.
  //   x (tile_view → rank-1 memref) → VFGatherOp base
  //   y (loaded index vector)        → index_vec
  //   mask (optional)                → mask (all-true if absent)
  if (auto gatherOp = dyn_cast<::tla::GatherOp>(op)) {
    Value base = valueMap.lookup(gatherOp.getX());
    Value indexVec = valueMap.lookup(gatherOp.getY());
    if (!base || !indexVec)
      return failure();
    auto baseType = dyn_cast<MemRefType>(base.getType());
    if (!baseType || baseType.getRank() != 1)
      return failure();
    auto elemByteWidth = getElementByteWidth(baseType.getElementType());
    if (failed(elemByteWidth))
      return failure();
    int64_t numElems = 256 / *elemByteWidth;
    auto resultVecType = VectorType::get(numElems, baseType.getElementType());
    Value mask;
    if (gatherOp.getMask()) {
      mask = valueMap.lookup(gatherOp.getMask());
      if (!mask)
        return failure();
    } else {
      // Predicate follows the gathered vector semantic lane count.
      auto maskVecType = useFullPreg ? fullPregVecType(b.getContext())
                                     : VectorType::get({numElems}, b.getI1Type());
      mask = createPredicatePge(b, loc, maskVecType, numElems,
                                 hivmave::PgePattern::ALL);
    }
    Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
    valueMap[gatherOp.getResult()] =
        b.create<hivmave::VFGatherOp>(loc, resultVecType, base, ValueRange(zero), indexVec, mask);
    return success();
  }

  if (auto info = getVectorUnaryInfo(&op)) {
    if (op.getNumResults() != 1)
      return failure();
    TlaUnaryOperands operands = info->operands;
    Value operand = valueMap.lookup(operands.operand);
    if (!operand)
      return failure();
    auto operandVecType = dyn_cast<VectorType>(operand.getType());
    if (!operandVecType)
      return failure();

    Type tlaOperandType = operands.operand.getType();
    if (isa<::tla::VectorSSAType>(tlaOperandType)) {
      if (failed(validateVectorUnaryElementType(
              &op, *info, operandVecType.getElementType())))
        return failure();
    } else if (!isa<::tla::MaskSSAType>(tlaOperandType)) {
      return op.emitError("expected !tla.vector<NxT> or !tla.mask<N> operand");
    }

    Value mask;
    if (operands.mask) {
      mask = valueMap.lookup(operands.mask);
      if (!mask)
        return failure();
    } else {
      mask = allTrueMaskFor(b, loc, operandVecType, tlaOperandType,
                            useFullPreg);
    }
    Value result = createVectorUnaryResult(b, loc, info->kind, tlaOperandType,
                                           operandVecType, operand, mask);
    if (!result)
      return failure();
    valueMap[op.getResult(0)] = result;
    return success();
  }

  // tla.create_mask: build a mask vector from a fixed pattern ->
  // ave.hir.pge<PATTERN>. The op's own dtype attr fixes the lane count
  // (256 bytes / element size) and hence the i1 mask width.
  if (auto maskOp = dyn_cast<::tla::CreateMaskOp>(op)) {
    auto pattern = hivmave::symbolizePgePattern(maskOp.getPattern());
    if (!pattern)
      return maskOp.emitError("unknown tla.create_mask pattern: ") << maskOp.getPattern(),
             failure();
    auto opCtx = deriveVecCtxForElement(maskOp.getDtype(), useFullPreg);
    if (failed(opCtx))
      return maskOp.emitError("unsupported tla.create_mask dtype: ") << maskOp.getDtype(),
             failure();
    valueMap[maskOp.getResult()] = createPredicatePge(
        b, loc, opCtx->maskVecType, opCtx->lanes, *pattern);
    return success();
  }

  // tla.update_mask: tail mask + remaining count. Lowers to ave.hir.plt,
  // whose mask result drives masked stores and whose second result
  // (true_shape - lanes) is threaded back as the loop-carried tail counter.
  // The op's own dtype attr fixes the lane count (256 bytes / element size)
  // and hence the i1 mask width and the tail decrement.
  if (auto updateMaskOp = dyn_cast<::tla::UpdateMaskOp>(op)) {
    // The true-shape operand may be a vec.func-external index constant (e.g.
    // tla.update_mask(1) building a single-lane mask): such constants are not
    // collected as helper arguments, so clone them inline like other scalar
    // operands instead of a bare valueMap lookup (which would miss them).
    Value trueShape =
        lookupOrCloneScalarValue(b, updateMaskOp.getTrueShape(), valueMap);
    if (!trueShape)
      return failure();
    auto opCtx = deriveVecCtxForElement(updateMaskOp.getDtype(), useFullPreg);
    if (failed(opCtx))
      return updateMaskOp.emitError("unsupported tla.update_mask dtype: ")
             << updateMaskOp.getDtype(), failure();
    auto plt = createPredicatePlt(b, loc, opCtx->maskVecType, opCtx->lanes,
                                   trueShape);
    valueMap[updateMaskOp.getMask()] = plt.getRes();
    // new_true_shape = true_shape - lanes, which is exactly what plt computes.
    // We materialize it with index arithmetic rather than consuming plt's second
    // result: that result is i32 in hardware but typed index, so carrying it
    // through the loop would leave an unfoldable i32<->index unrealized cast.
    Value lanesValue = b.create<arith::ConstantIndexOp>(loc, opCtx->lanes);
    valueMap[updateMaskOp.getNewTrueShape()] =
        b.create<arith::SubIOp>(loc, trueShape, lanesValue);
    return success();
  }

  if (auto cmpOp = dyn_cast<::tla::CmpOp>(op)) {
    Value lhs = valueMap.lookup(cmpOp.getLhs());
    if (!lhs)
      return failure();
    // The compare's operand width fixes both the input vectors and the i1 mask
    // result width.
    auto lhsTy = dyn_cast<VectorType>(lhs.getType());
    if (!lhsTy)
      return failure();
    auto opCtx = deriveVecCtxForElement(lhsTy.getElementType(), useFullPreg);
    if (failed(opCtx))
      return failure();
    auto cmpType = mapCmpMode(cmpOp.getMode());
    if (!cmpType)
      return cmpOp.emitError("unknown tla.cmp mode: ") << cmpOp.getMode(),
             failure();
    Value mask;
    if (cmpOp.getMask()) {
      mask = valueMap.lookup(cmpOp.getMask());
      if (!mask)
        return failure();
    } else {
      mask = createPredicatePge(b, loc, opCtx->maskVecType, opCtx->lanes,
                                 hivmave::PgePattern::ALL);
    }
    if (isa<::tla::VectorSSAType>(cmpOp.getRhs().getType())) {
      Value rhs = valueMap.lookup(cmpOp.getRhs());
      if (!rhs)
        return failure();
      valueMap[cmpOp.getResult()] =
          b.create<hivmave::VFCmpOp>(loc, opCtx->maskVecType, *cmpType, lhs, rhs, mask);
    } else {
      Value rhs = lookupOrCloneScalarValue(b, cmpOp.getRhs(), valueMap);
      if (!rhs)
        return failure();
      auto scalarOr = castScalarForVectorElement(rhs, opCtx->elementType);
      if (failed(scalarOr))
        return failure();
      valueMap[cmpOp.getResult()] =
          b.create<hivmave::VFCmpS>(loc, opCtx->maskVecType, *cmpType, lhs, *scalarOr, mask);
    }
    return success();
  }

  if (auto storeOp = dyn_cast<::tla::StoreOp>(op)) {
    Value dest = valueMap.lookup(storeOp.getDest());
    Value source = valueMap.lookup(storeOp.getSource());
    if (!dest || !source)
      return failure();
    Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
    Value mask;
    auto sourceTy = dyn_cast<VectorType>(source.getType());
    if (!sourceTy)
      return failure();
    auto opCtx = deriveVecCtxForElement(sourceTy.getElementType(), useFullPreg);
    if (failed(opCtx))
      return failure();
    if (storeOp.getMask()) {
      mask = valueMap.lookup(storeOp.getMask());
      if (!mask)
        return failure();
    } else {
      auto validLanes =
          getTlaTensorValidLaneCount(b, loc, storeOp.getDest(), valueMap);
      if (failed(validLanes))
        return storeOp.emitError("failed to determine tla.store dest valid lanes"),
               failure();
      mask = createPredicatePlt(b, loc, opCtx->maskVecType, opCtx->lanes,
                                *validLanes)
                 .getRes();
    }
    // store_unalign (this PR): mark AVE masked-store as unaligned UB access.
    if (storeOp.getUnalignedUbAccess().value_or(false)) {
      auto store = b.create<hivmave::VFMaskedStoreOp>(loc, dest, ValueRange{zero}, mask, source);
      store->setAttr(hivmave::UnalignedAttr::name,
                     hivmave::UnalignedAttr::get(b.getContext()));
    } else {
      b.create<hivmave::VFMaskedStoreOp>(loc, dest, ValueRange{zero}, mask, source);
    }
    return success();
  }

  // scf.for: rebuild the loop, including loop-carried iter_args, and lower its
  // body. Init args and the scf.yield operands may be register, index, or
  // scalar SSA values threaded through the helper (e.g. the tail counter produced by tla.update_mask).
  if (auto forOp = dyn_cast<scf::ForOp>(op)) {
    Value lb = lookupOrCloneScalarValue(b, forOp.getLowerBound(), valueMap);
    Value ub = lookupOrCloneScalarValue(b, forOp.getUpperBound(), valueMap);
    Value step = lookupOrCloneScalarValue(b, forOp.getStep(), valueMap);
    if (!lb || !ub || !step)
      return failure();
    // Loop-carried `index` values (e.g. the tla.update_mask tail counter) are
    // carried across the loop as i64 instead: after scf->cf lowering the
    // downstream index->iN conversion only rewrites the induction variable, so
    // an index iter_arg would leave dangling index<->iN unrealized casts on the
    // carried value that ReconcileUnrealizedCasts cannot fold across the cf
    // block boundary. Casting at the boundaries with arith.index_cast keeps the
    // carried value a plain integer that lowers cleanly.
    Type i64Ty = b.getIntegerType(64);
    auto regionIterArgs = forOp.getRegionIterArgs();
    SmallVector<bool> wasIndex(regionIterArgs.size(), false);
    SmallVector<Value> initArgs;
    for (auto [idx, init] : llvm::enumerate(forOp.getInitArgs())) {
      Value mapped = lookupOrCloneScalarValue(b, init, valueMap);
      if (!mapped)
        return failure();
      if (isa<IndexType>(mapped.getType())) {
        wasIndex[idx] = true;
        mapped = b.create<arith::IndexCastOp>(loc, i64Ty, mapped);
      }
      initArgs.push_back(mapped);
    }
    LogicalResult bodyStatus = success();
    auto newFor = b.create<scf::ForOp>(
        loc, lb, ub, step, initArgs,
        [&](OpBuilder &nb, Location nloc, Value iv, ValueRange iterArgs) {
          DenseMap<Value, Value> nestedMap = valueMap;
          nestedMap[forOp.getInductionVar()] = iv;
          for (size_t i = 0; i < regionIterArgs.size(); ++i) {
            Value newArg = iterArgs[i];
            if (wasIndex[i])
              newArg = nb.create<arith::IndexCastOp>(nloc, nb.getIndexType(), newArg);
            nestedMap[regionIterArgs[i]] = newArg;
          }
          if (failed(lowerNestedVectorBlock(forOp.getBody(), nb, module, nestedMap,
                                       useFullPreg))) {
            bodyStatus = failure();
            nb.create<scf::YieldOp>(nloc, iterArgs);
            return;
          }
          auto oldYield = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
          SmallVector<Value> yielded;
          for (auto [i, v] : llvm::enumerate(oldYield.getOperands())) {
            Value mapped = lookupOrCloneScalarValue(nb, v, nestedMap);
            if (!mapped) {
              bodyStatus = failure();
              break;
            }
            if (wasIndex[i] && isa<IndexType>(mapped.getType()))
              mapped = nb.create<arith::IndexCastOp>(nloc, i64Ty, mapped);
            yielded.push_back(mapped);
          }
          if (failed(bodyStatus)) {
            nb.create<scf::YieldOp>(nloc, iterArgs);
            return;
          }
          nb.create<scf::YieldOp>(nloc, yielded);
        });
    if (failed(bodyStatus))
      return failure();
    for (auto [i, oldRes] : llvm::enumerate(forOp.getResults())) {
      Value newRes = newFor.getResult(i);
      if (wasIndex[i] && !oldRes.use_empty())
        newRes = b.create<arith::IndexCastOp>(loc, b.getIndexType(), newRes);
      valueMap[oldRes] = newRes;
    }
    return success();
  }

  // scf.if: rebuild result-bearing conditionals after converting register
  // carrier types to physical builtin vectors. Each branch is lowered with an
  // independent value map, and its old scf.yield operands feed the new op.
  if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
    Value cond = lookupOrCloneScalarValue(b, ifOp.getCondition(), valueMap);
    if (!cond) return failure();

    SmallVector<Type> resultTypes;
    for (Value result : ifOp.getResults()) {
      auto loweredType = lowerSCFCarrierType(result.getType());
      if (failed(loweredType)) return failure();
      resultTypes.push_back(*loweredType);
    }

    bool hasElse = !ifOp.getElseRegion().empty();
    auto newIf = b.create<scf::IfOp>(loc, resultTypes, cond, hasElse);
    auto lowerBranch = [&](Block *oldBlock, Block *newBlock) -> LogicalResult {
      DenseMap<Value, Value> branchMap = valueMap;
      if (!newBlock->empty() &&
          newBlock->back().hasTrait<OpTrait::IsTerminator>())
        newBlock->back().erase();
      OpBuilder branchBuilder = OpBuilder::atBlockEnd(newBlock);
      if (failed(lowerNestedVectorBlock(oldBlock, branchBuilder, module,
                                        branchMap, useFullPreg)))
        return failure();

      auto oldYield = dyn_cast<scf::YieldOp>(oldBlock->getTerminator());
      if (!oldYield) return failure();
      SmallVector<Value> yielded;
      for (Value operand : oldYield.getOperands()) {
        Value mapped =
            lookupOrCloneScalarValue(branchBuilder, operand, branchMap);
        if (!mapped) return failure();
        yielded.push_back(mapped);
      }
      branchBuilder.create<scf::YieldOp>(oldYield.getLoc(), yielded);
      return success();
    };

    if (failed(lowerBranch(ifOp.thenBlock(), newIf.thenBlock())))
      return failure();
    if (hasElse && failed(lowerBranch(ifOp.elseBlock(), newIf.elseBlock())))
      return failure();

    for (auto [oldResult, newResult] :
         llvm::zip(ifOp.getResults(), newIf.getResults()))
      valueMap[oldResult] = newResult;
    return success();
  }

  // Index/scalar arithmetic (arith.*) feeding offsets/conditions: clone with
  // mapped operands.
  if (op.getDialect()->getNamespace() == arith::ArithDialect::getDialectNamespace()) {
    IRMapping mapper;
    for (Value operand : op.getOperands()) {
      Value mapped = lookupOrCloneScalarValue(b, operand, valueMap);
      if (!mapped)
        return failure();
      mapper.map(operand, mapped);
    }
    Operation *cloned = b.clone(op, mapper);
    for (auto [oldResult, newResult] : llvm::zip(op.getResults(), cloned->getResults()))
      valueMap[oldResult] = newResult;
    return success();
  }

  // GM scalar accesses lowered by tla-lower-scalar-access before outlining.
  if (auto memLoad = dyn_cast<mlir::memref::LoadOp>(op)) {
    Value mem = valueMap.lookup(memLoad.getMemRef());
    if (!mem)
      return failure();
    SmallVector<Value, 2> indices;
    for (Value idx : memLoad.getIndices()) {
      Value mapped = lookupOrCloneScalarValue(b, idx, valueMap);
      if (!mapped)
        return failure();
      indices.push_back(mapped);
    }
    valueMap[memLoad.getResult()] =
        b.create<mlir::memref::LoadOp>(loc, mem, indices).getResult();
    return success();
  }
  if (auto memStore = dyn_cast<mlir::memref::StoreOp>(op)) {
    Value mem = valueMap.lookup(memStore.getMemRef());
    Value val = valueMap.lookup(memStore.getValue());
    if (!val)
      val = lookupOrCloneScalarValue(b, memStore.getValue(), valueMap);
    if (!mem || !val)
      return failure();
    SmallVector<Value, 2> indices;
    for (Value idx : memStore.getIndices()) {
      Value mapped = lookupOrCloneScalarValue(b, idx, valueMap);
      if (!mapped)
        return failure();
      indices.push_back(mapped);
    }
    b.create<mlir::memref::StoreOp>(loc, val, mem, indices);
    return success();
  }

  // Index/scalar arithmetic (arith.*) feeding tensor_desc offset/stride operands:
  // clone with operands remapped. tla-lower-tensor-desc emits the abs-coord/origin
  // arithmetic (addi/subi/minsi) for dynamic-coord tile_views; static cases fold
  // to constants (createOrFold) and are handled by the arith::ConstantOp arm above.
  if (op.getDialect()->getNamespace() ==
      arith::ArithDialect::getDialectNamespace()) {
    IRMapping mapper;
    for (Value operand : op.getOperands()) {
      Value mapped = lookupOrCloneScalarValue(b, operand, valueMap);
      if (!mapped)
        return failure();
      mapper.map(operand, mapped);
    }
    Operation *cloned = b.clone(op, mapper);
    for (auto [oldResult, newResult] :
         llvm::zip(op.getResults(), cloned->getResults()))
      valueMap[oldResult] = newResult;
    return success();
  }

  if (op.hasTrait<OpTrait::IsTerminator>())
    return success();

  return failure();
}

static LogicalResult lowerNestedVectorBlock(Block *sourceBlock, OpBuilder &b, ModuleOp module,
                                            DenseMap<Value, Value> &valueMap,
                                            bool useFullPreg) {
  for (Operation &op : sourceBlock->getOperations()) {
    // Terminators are reproduced by the enclosing op (scf.for/scf.if) or by
    // buildHelperFunc's func.return.
    if (op.hasTrait<OpTrait::IsTerminator>())
      continue;
    if (failed(lowerNestedVectorOp(op, b, module, valueMap, useFullPreg)))
      return failure();
  }
  return success();
}

// Add `source` as a helper operand unless one already covers it. A tla.tensor_desc
// tile is covered by another tensor_desc sharing the same base (different per-lane
// tiles of one root share a single helper memref arg); a raw memref is covered by
// itself. tla-lower-tensor-desc is the sole descriptor producer, so every tile
// source is a tensor_desc (or a raw memref for bridged GM scalar accesses).
static void addVectorHelperOperand(Value source, SmallVectorImpl<Value> &operands) {
  if (auto descOp = source.getDefiningOp<::tla::TensorDescOp>()) {
    Value base = descOp.getBase();
    for (Value v : operands)
      if (auto d = v.getDefiningOp<::tla::TensorDescOp>(); d && d.getBase() == base)
        return;
    operands.push_back(source);
    return;
  }
  if (!llvm::is_contained(operands, source))
    operands.push_back(source);
}

// Collect, in body order, the unique base memrefs that tla.load/tla.store/
// tla.gather chunks and bridged GM scalar accesses reference. These become the
// helper's arguments.
static void collectVectorHelperOperands(Block *block, SmallVectorImpl<Value> &operands) {
  for (Operation &op : block->getOperations()) {
    if (auto loadOp = dyn_cast<::tla::LoadOp>(op)) {
      addVectorHelperOperand(loadOp.getSource(), operands);
      continue;
    }
    if (auto storeOp = dyn_cast<::tla::StoreOp>(op)) {
      addVectorHelperOperand(storeOp.getDest(), operands);
      continue;
    }
    if (auto gatherOp = dyn_cast<::tla::GatherOp>(op)) {
      addVectorHelperOperand(gatherOp.getX(), operands);
      continue;
    }
    // Bridged GM scalar accesses (after tla-lower-scalar-access) appear as memref.load/store.
    if (auto memLoad = dyn_cast<mlir::memref::LoadOp>(op)) {
      addVectorHelperOperand(memLoad.getMemRef(), operands);
      continue;
    }
    if (auto memStore = dyn_cast<mlir::memref::StoreOp>(op)) {
      addVectorHelperOperand(memStore.getMemRef(), operands);
      continue;
    }
    for (Region &region : op.getRegions())
      for (Block &nested : region)
        collectVectorHelperOperands(&nested, operands);
  }
}

// Collect unique scalar values used inside the region but defined outside it
// (e.g. a sub_block_idx/block_idx computed at the top of the kernel, or a
// vector-scalar RHS constant). Passing them into the helper avoids cloning float
// constants into vector helpers where vector.broadcast can fold to illegal
// vector arith.constant ops before the HIVMAVE conversion pipeline.
// They are passed in as trailing scalar arguments rather than recomputed inside
// the outlined vector function.
static void collectVectorHelperScalarOperands(::tla::VecFuncOp vecFuncOp,
                                              SmallVectorImpl<Value> &scalars) {
  vecFuncOp.walk([&](Operation *op) {
    for (Value operand : op->getOperands()) {
      Type operandType = operand.getType();
      if (!operandType.isIntOrIndex() && !isa<FloatType>(operandType))
        continue;
      // Index/integer constants defined outside vec.func are cloned inline by
      // lookupOrCloneScalarValue rather than passed as helper args. This keeps
      // static tile offsets/strides (now tensor_desc operands, materialized at
      // function scope by tla-lower-tensor-desc) as constants inside the helper
      // instead of bloating the helper signature. Float constants are still
      // passed as args (see the comment above re: vector.broadcast folding).
      if (operandType.isIntOrIndex() &&
          operand.getDefiningOp<arith::ConstantOp>())
        continue;
      Region *defRegion = operand.getParentRegion();
      if (defRegion && !vecFuncOp.getBody().isAncestor(defRegion) &&
          !llvm::is_contained(scalars, operand))
        scalars.push_back(operand);
    }
  });
  // tla-lower-tensor-desc is the sole descriptor producer: every tile inside
  // vec.func is a tensor_desc whose stride/offset operands -- and the arith
  // (addi/subi/minsi) that computes them from the parent's leaves -- are already
  // captured by the generic walk above. No dedicated tile_view / make_stride
  // scalar collection is needed.
}

// Compatibility lowering for the current NPUIR AVE-to-RegBase pipeline: AVE
// predicate ops lower to the backend-native vector<256xi1> container, but SCF
// signatures are not structurally type-converted. Conservatively put every
// predicate in an outlined helper on the full-preg representation whenever the
// helper contains both SCF and MaskSSA. This is intentionally a helper-level
// over-approximation, not a dataflow proof that a particular MaskSSA crosses an
// SCF edge; one uniform representation also keeps AVE predicate producers and
// consumers type-consistent.
//
// TODO: Remove this compatibility mode once NPUIR AVE-to-RegBase conversion
// structurally converts predicate-bearing scf.if/scf.for signatures and maps
// their block arguments, yields, and results to vector<256xi1>.
static bool requiresFullPregForControlFlow(::tla::VecFuncOp vecFuncOp) {
  bool hasControlFlow = false;
  bool hasMaskSSA = false;
  vecFuncOp.walk([&](Operation *op) {
    hasControlFlow |= isa<scf::ForOp, scf::IfOp>(op);
    auto isMaskSSA = [](Value value) {
      return isa<::tla::MaskSSAType>(value.getType());
    };
    hasMaskSSA |= llvm::any_of(op->getOperands(), isMaskSSA) ||
                  llvm::any_of(op->getResults(), isMaskSSA);
    return hasControlFlow && hasMaskSSA ? WalkResult::interrupt()
                                        : WalkResult::advance();
  });
  return hasControlFlow && hasMaskSSA;
}

// Build a vector_region helper for a tla.vec.func body. The helper receives one
// full-size UB memref per referenced tensor; the for/if control flow is carried
// inside the helper, where each tla.load/store is lowered to an AVE
// vload/masked-store over a 256-byte tile carved from the full memref at the
// per-iteration offset.
static FailureOr<func::FuncOp> buildHelperFunc(ModuleOp module, func::FuncOp parentFunc,
                                               ::tla::VecFuncOp vecFuncOp,
                                               ArrayRef<Value> helperOperands,
                                               ArrayRef<Value> scalarOperands,
                                               int &nextVectorRegionId,
                                               DenseMap<Value, Value> &loweredMemrefByValue) {
  MLIRContext *ctx = module.getContext();
  Operation *vectorOp = vecFuncOp.getOperation();
  OpBuilder moduleBuilder(module.getBodyRegion());
  moduleBuilder.setInsertionPointAfter(parentFunc);

  Block *body = vecFuncOp.getBody().empty() ? nullptr : &vecFuncOp.getBody().front();
  if (!body || helperOperands.empty())
    return failure();

  SmallVector<Type> functionInputs;
  functionInputs.reserve(helperOperands.size());
  for (Value operand : helperOperands) {
    // Already-bridged GM memrefs (scalar_load/store) keep their concrete type.
    if (auto mt = dyn_cast<MemRefType>(operand.getType())) {
      functionInputs.push_back(mt);
      continue;
    }
    auto operandType = getVectorHelperArgMemrefType(operand);
    if (failed(operandType))
      return failure();
    functionInputs.push_back(*operandType);
  }
  // Trailing scalar args: scalars captured from outside the region.
  for (Value scalar : scalarOperands)
    functionInputs.push_back(scalar.getType());
  auto funcType = moduleBuilder.getFunctionType(functionInputs, TypeRange{});

  // The per-iteration vector tile is one 256-byte register's worth of elements.
  // Each op inside the helper derives its own width from its operands/result,
  // so tiles may carry different element types within one region (e.g. a f32
  // load feeding a tla.cast to f16). Validate only that each tile operand is a
  // supported int/float type (the trailing scalar args are index/int and are
  // handled separately). This runs before the helper is created so a validation
  // failure leaks no partial IR.
  for (size_t i = 0; i < helperOperands.size(); ++i) {
    Type tileElementType = cast<MemRefType>(functionInputs[i]).getElementType();
    if (!isa<IntegerType>(tileElementType) && !isa<FloatType>(tileElementType))
      return vectorOp->emitError("unsupported element type for vector binary helper: ")
             << tileElementType;
    if (failed(getVectorLaneCount(tileElementType)))
      return vectorOp->emitError("unsupported element width for vector helper tile: ")
             << tileElementType;
  }

  std::string helperName = buildUniqueVectorHelperName(module, nextVectorRegionId);
  auto helper = moduleBuilder.create<func::FuncOp>(vectorOp->getLoc(), helperName, funcType);
  helper.setPrivate();
  helper->setAttr(hivm::TFuncCoreTypeAttr::name,
                  hivm::TFuncCoreTypeAttr::get(ctx, hivm::TFuncCoreType::AIV));
  helper->setAttr("hivm.vector_function", UnitAttr::get(ctx));
  helper->setAttr("no_inline", UnitAttr::get(ctx));

  Block *entry = helper.addEntryBlock();
  OpBuilder b = OpBuilder::atBlockBegin(entry);

  DenseMap<Value, Value> valueMap;
  for (auto [i, operand] : llvm::enumerate(helperOperands))
    valueMap[operand] = entry->getArgument(i);
  // Captured scalars map to their trailing block arguments.
  for (auto [j, scalar] : llvm::enumerate(scalarOperands))
    valueMap[scalar] = entry->getArgument(helperOperands.size() + j);
  // helperOperands holds one representative tensor_desc per unique base (see
  // addVectorHelperOperand). Every other vec.func-internal tensor_desc sharing
  // that base must resolve to the same helper memref arg so the TensorDescOp arm
  // in lowerNestedVectorOp can map it.
  DenseMap<Value, Value> baseToArg;
  for (Value operand : helperOperands)
    if (auto d = operand.getDefiningOp<::tla::TensorDescOp>())
      baseToArg[d.getBase()] = valueMap[operand];
  vecFuncOp.walk([&](::tla::TensorDescOp desc) {
    if (auto it = baseToArg.find(desc.getBase()); it != baseToArg.end())
      valueMap[desc.getResult()] = it->second;
  });
  bool useFullPreg = requiresFullPregForControlFlow(vecFuncOp);
  if (failed(lowerNestedVectorBlock(body, b, module, valueMap, useFullPreg))) {
    // Discard the partially-built helper so an unsupported construct fails
    // cleanly (the vec.func is left intact) instead of leaking malformed IR.
    helper.erase();
    return failure();
  }
  b.create<func::ReturnOp>(vectorOp->getLoc());
  return helper;
}

class LowerVecFuncRegionPattern : public OpRewritePattern<::tla::VecFuncOp> {
public:
  LowerVecFuncRegionPattern(MLIRContext *context, ModuleOp module, int &nextVectorRegionId,
                            DenseMap<Value, Value> &loweredMemrefByValue)
      : OpRewritePattern<::tla::VecFuncOp>(context, /*benefit=*/2), module(module),
        nextVectorRegionId(nextVectorRegionId), loweredMemrefByValue(loweredMemrefByValue) {}

  LogicalResult matchAndRewrite(::tla::VecFuncOp vecFuncOp,
                                PatternRewriter &rewriter) const override {
    auto *body = vecFuncOp.getBody().empty() ? nullptr : &vecFuncOp.getBody().front();
    if (!body)
      return rewriter.notifyMatchFailure(vecFuncOp, "expected tla.vec.func body");

    // Collect the load / binary compute / store ops (used for arg dedup and
    // graph validation); the helper builder walks the region itself to carry
    // the control flow structure.
    SmallVector<::tla::LoadOp, 4> loads;
    SmallVector<::tla::FullOp, 4> fulls;
    SmallVector<::tla::CreateMaskOp, 4> createMasks;
    SmallVector<::tla::UpdateMaskOp, 4> updateMasks;
    SmallVector<::tla::ArangeOp, 4> aranges;
    SmallVector<Operation *, 4> computeOps;
    SmallVector<::tla::StoreOp, 2> stores;
    vecFuncOp->walk([&](Operation *op) {
      if (auto load = dyn_cast<::tla::LoadOp>(op)) {
        loads.push_back(load);
      } else if (auto full = dyn_cast<::tla::FullOp>(op)) {
        fulls.push_back(full);
      } else if (auto createMask = dyn_cast<::tla::CreateMaskOp>(op)) {
        createMasks.push_back(createMask);
      } else if (auto updateMask = dyn_cast<::tla::UpdateMaskOp>(op)) {
        updateMasks.push_back(updateMask);
      } else if (auto arange = dyn_cast<::tla::ArangeOp>(op)) {
        aranges.push_back(arange);
      } else if (auto store = dyn_cast<::tla::StoreOp>(op)) {
        stores.push_back(store);
      } else if (isVectorComputeOp(op)) {
        computeOps.push_back(op);
      }
      return WalkResult::advance();
    });
    if (stores.empty()) {
      // Scalar-only (or empty) VF cannot be outlined as a BiSheng helper — that
      // path requires tla.store. If there is also no tile load/compute, inline
      // the body into the parent (same as tla.vector flattening) so GM
      // scalar_load/store + scf stay legal for later convert-scf-to-cf.
      if (!loads.empty() || !fulls.empty() || !createMasks.empty() ||
          !updateMasks.empty() || !aranges.empty() || !computeOps.empty())
        return rewriter.notifyMatchFailure(
            vecFuncOp, "expected tla.vec.func body with a tla.store");
      rewriter.inlineBlockBefore(body, vecFuncOp->getBlock(),
                                 vecFuncOp->getIterator());
      rewriter.eraseOp(vecFuncOp);
      return success();
    }

    // Validate the graph: every compute operand and store source must come from
    // a tla.load result or a prior compute result inside this region.
    DenseSet<Value> producedValues;
    for (::tla::LoadOp load : loads) {
      producedValues.insert(load.getResult());
      if (Value result2 = load.getResult2())
        producedValues.insert(result2);
    }
    for (::tla::FullOp full : fulls)
      producedValues.insert(full.getResult());
    for (::tla::CreateMaskOp createMask : createMasks)
      producedValues.insert(createMask.getResult());
    for (::tla::UpdateMaskOp updateMask : updateMasks)
      producedValues.insert(updateMask.getMask());
    for (::tla::ArangeOp arange : aranges)
      producedValues.insert(arange.getResult());
    auto isRegisterCarrier = [](Value value) {
      Type type = value.getType();
      return isa<::tla::VectorSSAType, ::tla::MaskSSAType>(type);
    };
    vecFuncOp.walk([&](scf::ForOp forOp) {
      for (BlockArgument arg : forOp.getRegionIterArgs())
        if (isRegisterCarrier(arg)) producedValues.insert(arg);
      for (Value result : forOp.getResults())
        if (isRegisterCarrier(result)) producedValues.insert(result);
    });
    vecFuncOp.walk([&](scf::IfOp ifOp) {
      for (Value result : ifOp.getResults())
        if (isRegisterCarrier(result)) producedValues.insert(result);
    });

    for (Operation *computeOp : computeOps) {
      if (isa<::tla::InterleaveOp>(computeOp) ||
          isa<::tla::DeInterleaveOp>(computeOp)) {
        if (computeOp->getNumResults() != 2)
          return rewriter.notifyMatchFailure(
              vecFuncOp, "unexpected two-result tla compute op shape");
      } else if (computeOp->getNumResults() != 1) {
        return rewriter.notifyMatchFailure(vecFuncOp, "unexpected tla compute op shape");
      }
      if (auto anyInfo = getAnyVectorOperationInfo(computeOp)) {
        if (auto info = anyInfo->binary) {
          // Vector operands must come from a load or prior compute op. A
          // vector-scalar rhs may be captured or cloned into the helper.
          TlaBinaryOperands ops = info->operands;
          if (!ops.lhs || !ops.rhs || !producedValues.contains(ops.lhs))
            return rewriter.notifyMatchFailure(
                vecFuncOp, "expected binary op operand from load/create/update mask "
                           "or prior compute op");
          if (info->rhsKind == VectorRhsKind::Vector && !producedValues.contains(ops.rhs))
            return rewriter.notifyMatchFailure(
                vecFuncOp, "expected binary op rhs from load/create/update mask "
                           "or prior compute op");
          if (ops.mask && !producedValues.contains(ops.mask))
            return rewriter.notifyMatchFailure(
                vecFuncOp, "expected binary op mask from create/update mask, compare, "
                           "or prior mask compute op");
        } else if (auto unaryInfo = anyInfo->unary) {
          TlaUnaryOperands ops = unaryInfo->operands;
          if (!ops.operand || !producedValues.contains(ops.operand))
            return rewriter.notifyMatchFailure(
                vecFuncOp, "expected unary op operand from load/create/update mask "
                           "or prior compute op");
          if (ops.mask && !producedValues.contains(ops.mask))
            return rewriter.notifyMatchFailure(
                vecFuncOp, "expected unary op mask from create/update mask, compare, "
                           "or prior mask compute op");
        } else {
          return rewriter.notifyMatchFailure(vecFuncOp, "unexpected tla compute op");
        }
      } else if (auto cmpOp = dyn_cast<::tla::CmpOp>(computeOp)) {
        if (!producedValues.contains(cmpOp.getLhs()))
          return rewriter.notifyMatchFailure(
              vecFuncOp, "expected tla.cmp lhs from tla.load or prior compute op");
        if (isa<::tla::VectorSSAType>(cmpOp.getRhs().getType()) &&
            !producedValues.contains(cmpOp.getRhs()))
          return rewriter.notifyMatchFailure(
              vecFuncOp, "expected tla.cmp rhs from tla.load or prior compute op");
        if (cmpOp.getMask() && !producedValues.contains(cmpOp.getMask()))
          return rewriter.notifyMatchFailure(
              vecFuncOp, "expected tla.cmp mask from create/update mask or "
                         "prior mask compute op");
      } else if (auto whereOp = dyn_cast<::tla::WhereOp>(computeOp)) {
        if (!producedValues.contains(whereOp.getMask()))
          return rewriter.notifyMatchFailure(
              vecFuncOp,
              "expected tla.where mask from create/update mask, compare, "
              "SCF carrier, or prior mask compute op");
        if (!producedValues.contains(whereOp.getX()) ||
            !producedValues.contains(whereOp.getY()))
          return rewriter.notifyMatchFailure(
              vecFuncOp, "expected tla.where operand from tla.load or prior compute op");
      } else if (auto squeezeOp = dyn_cast<::tla::SqueezeOp>(computeOp)) {
        if (!producedValues.contains(squeezeOp.getSrc()))
          return rewriter.notifyMatchFailure(
              vecFuncOp, "expected tla.squeeze src from tla.load or prior compute op");
        if (!producedValues.contains(squeezeOp.getMask()))
          return rewriter.notifyMatchFailure(
              vecFuncOp, "expected tla.squeeze mask from create/update mask or "
                         "prior mask compute op");
      } else if (auto reduceOp = dyn_cast<::tla::ReduceOp>(computeOp)) {
        Value operand = reduceOp.getOperand();
        if (reduceOp.getMask() && !producedValues.contains(reduceOp.getMask()))
          return rewriter.notifyMatchFailure(
              vecFuncOp, "expected tla.reduce mask from a legal mask producer");
        if (!producedValues.contains(operand))
          return rewriter.notifyMatchFailure(
              vecFuncOp, "expected tla.reduce operand from tla.load or prior compute op");
      } else if (auto interleaveOp = dyn_cast<::tla::InterleaveOp>(computeOp)) {
        if (!producedValues.contains(interleaveOp.getSrc0()) ||
            !producedValues.contains(interleaveOp.getSrc1()))
          return rewriter.notifyMatchFailure(
            vecFuncOp, "expected tla.interleave operands from tla.load or prior compute op");
      } else if (auto deInterleaveOp = dyn_cast<::tla::DeInterleaveOp>(computeOp)) {
        if (!producedValues.contains(deInterleaveOp.getSrc0()) ||
            !producedValues.contains(deInterleaveOp.getSrc1()))
          return rewriter.notifyMatchFailure(
            vecFuncOp, "expected tla.deinterleave operands from tla.load or prior compute op");
      } else if (auto gatherOp = dyn_cast<::tla::GatherOp>(computeOp)) {
        if (gatherOp.getMask() && !producedValues.contains(gatherOp.getMask()))
          return rewriter.notifyMatchFailure(
              vecFuncOp, "expected tla.gather mask from a legal mask producer");
        if (!producedValues.contains(gatherOp.getY()))
          return rewriter.notifyMatchFailure(
              vecFuncOp, "expected tla.gather y operand from tla.load or prior compute op");
      } else if (auto castOp = dyn_cast<::tla::CastOp>(computeOp)) {
        if (castOp.getMask() && !producedValues.contains(castOp.getMask()))
          return rewriter.notifyMatchFailure(
              vecFuncOp, "expected tla.cast mask from a legal mask producer");
        if (!producedValues.contains(castOp.getSource()))
          return rewriter.notifyMatchFailure(
              vecFuncOp, "expected tla.cast source from tla.load or prior compute op");
      } else {
        return rewriter.notifyMatchFailure(vecFuncOp, "unexpected tla compute op");
      }
      for (Value result : computeOp->getResults())
        producedValues.insert(result);
    }
    for (::tla::StoreOp store : stores) {
      if (store.getMask() && !producedValues.contains(store.getMask()))
        return rewriter.notifyMatchFailure(
            vecFuncOp, "expected tla.store mask from a legal mask producer");
      if (!producedValues.contains(store.getSource()))
        return rewriter.notifyMatchFailure(
            vecFuncOp, "expected tla.store source from tla.load or compute op");
    }

    auto funcOp = vecFuncOp->getParentOfType<func::FuncOp>();
    if (!funcOp)
      return rewriter.notifyMatchFailure(vecFuncOp, "expected enclosing func.func");

    // The helper takes one full-size UB memref per referenced tensor, in body
    // order. Compute that operand list once and use it for both the helper
    // signature and the call.
    SmallVector<Value> helperOperands;
    collectVectorHelperOperands(body, helperOperands);
    if (helperOperands.empty())
      return rewriter.notifyMatchFailure(vecFuncOp, "expected vector region tensor operands");
    // Scalars captured from outside the region (e.g. a sub_block_idx computed at
    // the top of the kernel) are passed as trailing scalar arguments.
    SmallVector<Value> scalarOperands;
    collectVectorHelperScalarOperands(vecFuncOp, scalarOperands);

    auto helperOr = buildHelperFunc(module, funcOp, vecFuncOp, helperOperands, scalarOperands,
                                    nextVectorRegionId, loweredMemrefByValue);
    if (failed(helperOr))
      return rewriter.notifyMatchFailure(vecFuncOp, "failed to build vector helper function");
    auto helper = *helperOr;

    // The for/if control flow lives inside the helper, so this is a single
    // call (passing the full UB memrefs) that replaces the whole vec.func region.
    rewriter.setInsertionPoint(vecFuncOp);
    SmallVector<Value, 8> callOperands;
    callOperands.reserve(helperOperands.size());
    for (Value tensor : helperOperands) {
      // Bridged GM memref operands (from scalar_load/store) are passed as-is.
      if (isa<MemRefType>(tensor.getType())) {
        callOperands.push_back(tensor);
        continue;
      }
      auto type = getVectorHelperArgMemrefType(tensor);
      if (failed(type))
        return rewriter.notifyMatchFailure(
            vecFuncOp, "failed to type UB memref for vector helper call");
      // Materialize address-backed tla.tensor_desc operands at the call site.
      // tla-lower-tensor-desc is the sole descriptor producer, so every helper
      // operand here is a tensor_desc (raw memrefs were passed through above);
      // materialize its inttoptr base as a rank-1 helper arg when ptr-backed.
      Value ptr;
      if (auto descOp = tensor.getDefiningOp<::tla::TensorDescOp>()) {
        if (llvm::isa<::tla::PtrType>(descOp.getBase().getType()))
          ptr = descOp.getBase();
      }
      FailureOr<Value> base = failure();
      if (ptr) {
        if (!ptr.getDefiningOp<::tla::IntToPtrOp>())
          return rewriter.notifyMatchFailure(
              vecFuncOp, "expected pointer lowered to tla.inttoptr boundary");
        base = materializePtrValueAsMemref(rewriter, vecFuncOp.getLoc(), ptr, *type,
                                           vecFuncOp.getOperation());
        if (failed(base))
          return rewriter.notifyMatchFailure(
              vecFuncOp, "failed to materialize address-backed vector helper operand");
      } else {
        base = materializeBaseMemref(rewriter, vecFuncOp.getLoc(), tensor,
                                     /*loweredMemrefByValue=*/nullptr);
        if (failed(base))
          return rewriter.notifyMatchFailure(
              vecFuncOp, "failed to materialize UB memref for vector helper call");
      }
      auto arg = castMemrefToExpected(rewriter, vecFuncOp.getLoc(), *base, *type);
      if (failed(arg))
        return rewriter.notifyMatchFailure(vecFuncOp,
                                           "failed to cast helper operand to expected memref type");
      callOperands.push_back(*arg);
    }
    // Captured scalars are defined in the parent (before this region), so they
    // dominate the call — pass them directly as trailing call operands.
    for (Value scalar : scalarOperands)
      callOperands.push_back(scalar);

    auto call = rewriter.create<func::CallOp>(vecFuncOp.getLoc(), helper, callOperands);
    call->setAttr("hivm.vector_function", UnitAttr::get(rewriter.getContext()));
    call->setAttr("no_inline", UnitAttr::get(rewriter.getContext()));
    rewriter.eraseOp(vecFuncOp);
    return success();
  }

private:
  ModuleOp module;
  int &nextVectorRegionId;
  DenseMap<Value, Value> &loweredMemrefByValue;
};

class LowerCopyPattern : public OpRewritePattern<::tla::CopyOp> {
public:
  LowerCopyPattern(MLIRContext *context, DenseMap<Value, Value> &loweredMemrefByValue)
      : OpRewritePattern<::tla::CopyOp>(context, /*benefit=*/3),
        loweredMemrefByValue(loweredMemrefByValue) {}

  LogicalResult matchAndRewrite(::tla::CopyOp copyOp, PatternRewriter &rewriter) const override {
    if (copyOp->getNumOperands() != 2 || copyOp->getNumResults() != 0)
      return rewriter.notifyMatchFailure(
          copyOp, "expected tla.copy with 2 operands and 0 results");

    Value dstTile = copyOp.getDst();
    Value srcTile = copyOp.getSrc();
    auto dstDescOp = dstTile.getDefiningOp<::tla::TensorDescOp>();
    auto srcDescOp = srcTile.getDefiningOp<::tla::TensorDescOp>();
    if (!dstDescOp || !srcDescOp)
      return rewriter.notifyMatchFailure(
          copyOp, "expected tla.tensor_desc operand materialized by tla-lower-tensor-desc");
    auto dstDescOr = ::tla::descriptorFromTensorDescOp(dstDescOp);
    auto srcDescOr = ::tla::descriptorFromTensorDescOp(srcDescOp);
    const TensorDescriptor &dstDesc = *dstDescOr;
    const TensorDescriptor &srcDesc = *srcDescOr;

    std::string calleeName = ::tla::getCopyRouteCallee(
        copyOp.getContext(), srcDesc.addrspace, dstDesc.addrspace, srcDesc.layoutTag,
        dstDesc.layoutTag, srcDesc.elementType, dstDesc.elementType);
    if (calleeName.empty())
      return rewriter.notifyMatchFailure(
          copyOp, "unsupported tla.copy route (vector pass handles gm<->ub and ub->l1)");

    auto buildRuntimeMemref = [&](const TensorDescriptor &desc) -> FailureOr<Value> {
      FailureOr<Value> baseMemref = ::tla::getOrMaterializeDescriptorBaseMemref(
          rewriter, copyOp.getLoc(), desc, copyOp.getOperation(), loweredMemrefByValue);
      if (failed(baseMemref))
        return failure();
      auto baseType = dyn_cast<MemRefType>((*baseMemref).getType());
      if (!baseType)
        return failure();
      MemRefType runtimeType = ::tla::getDynamicStridedMemrefType(baseType);
      return ::tla::castMemrefToType(rewriter, copyOp.getLoc(), *baseMemref, runtimeType);
    };
    FailureOr<Value> srcRuntimeMemref = buildRuntimeMemref(srcDesc);
    FailureOr<Value> dstRuntimeMemref = buildRuntimeMemref(dstDesc);
    if (failed(srcRuntimeMemref) || failed(dstRuntimeMemref))
      return failure();
    SmallVector<Value, 20> payload =
        ::tla::buildCopyPayloadForRoute(rewriter, copyOp.getLoc(), srcDesc, dstDesc);
    SmallVector<Type, 22> operandTypes = {(*srcRuntimeMemref).getType(),
                                          (*dstRuntimeMemref).getType()};
    operandTypes.reserve(2 + payload.size());
    for (Value v : payload)
      operandTypes.push_back(v.getType());
    SmallVector<Value, 22> operands = {*srcRuntimeMemref, *dstRuntimeMemref};
    operands.append(payload.begin(), payload.end());
    auto callee = ::tla::getOrCreateRuntimeCall(copyOp->getParentOfType<ModuleOp>(), calleeName,
                                                 operandTypes);

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

    auto atomicModeAttr = copyOp->getAttrOfType<::tla::AtomicModeAttr>("atomic_mode");
    Type dstType = cast<MemRefType>((*dstRuntimeMemref).getType()).getElementType();
    bool _enable_atomic = atomicModeAttr && atomicModeAttr.getAtomicMode() != AtomicMode::none;
    if (_enable_atomic) {
      if (atomicModeAttr.getAtomicMode() != AtomicMode::add) {
        copyOp.emitError() << "currently only atomic add is supported";
        return failure();
      }

      auto modeAttr = hivm::AtomicKindAttr::get(rewriter.getContext(), getAtomicKind(atomicModeAttr.getAtomicMode()));
      rewriter.create<hivm::SetAtomicOp>(copyOp.getLoc(), modeAttr, mlir::TypeAttr::get(dstType));
    }
    rewriter.create<func::CallOp>(copyOp.getLoc(), callee, operands);                     
    if (_enable_atomic) {
      auto modeAttr = hivm::AtomicKindAttr::get(rewriter.getContext(), hivm::AtomicKind::NONE);
      rewriter.create<hivm::SetAtomicOp>(copyOp.getLoc(), modeAttr, mlir::TypeAttr::get(dstType));
    }
    rewriter.eraseOp(copyOp);
    return success();
  }

private:
  DenseMap<Value, Value> &loweredMemrefByValue;
};

class InlineVectorRegionWrapperPattern : public OpRewritePattern<::tla::VectorOp> {
public:
  explicit InlineVectorRegionWrapperPattern(MLIRContext *context)
      : OpRewritePattern<::tla::VectorOp>(context, /*benefit=*/10) {}

  LogicalResult matchAndRewrite(::tla::VectorOp vectorOp,
                                PatternRewriter &rewriter) const override {
    if (vectorOp->getNumRegions() == 0 || vectorOp.getBody().empty()) {
      rewriter.eraseOp(vectorOp);
      return success();
    }
    Block *body = &vectorOp.getBody().front();
    rewriter.inlineBlockBefore(body, vectorOp->getBlock(), vectorOp->getIterator());
    rewriter.eraseOp(vectorOp);
    return success();
  }
};

static void inlineVectorRegionWrappers(func::FuncOp funcOp) {
  SmallVector<::tla::VectorOp, 4> wrappers;
  funcOp.walk([&](::tla::VectorOp vectorOp) { wrappers.push_back(vectorOp); });

  IRRewriter rewriter(funcOp.getContext());
  for (::tla::VectorOp vectorOp : wrappers) {
    if (!vectorOp)
      continue;
    if (vectorOp->getNumRegions() == 0 || vectorOp.getBody().empty()) {
      rewriter.eraseOp(vectorOp);
      continue;
    }
    Block *body = &vectorOp.getBody().front();
    rewriter.inlineBlockBefore(body, vectorOp->getBlock(), vectorOp->getIterator());
    rewriter.eraseOp(vectorOp);
  }
}

static void populateTlaToVectorPatterns(RewritePatternSet &patterns, ModuleOp module,
                                        int &nextVectorRegionId,
                                        DenseMap<Value, Value> &loweredMemrefByValue) {
  MLIRContext *ctx = patterns.getContext();
  patterns.add<InlineVectorRegionWrapperPattern>(ctx);
  patterns.add<LowerVecFuncRegionPattern>(ctx, module, nextVectorRegionId,
                                          loweredMemrefByValue);
  patterns.add<LowerCopyPattern>(ctx, loweredMemrefByValue);
  // NOTE: no dead-tla-scaffolding DCE here. tla-vector-region lowers ops but
  // deliberately leaves the momentary tensor / ptr-bridge scaffolding and
  // unrealized casts in place; the downstream cleanup pass (tla-finalize-memref)
  // is responsible for DCE'ing them.
}

// Per-core identity queries (block_idx / block_dim / sub_block_idx) must be
// computed outside a tla.vec.func and passed in; emitting them inside the vector
// region produces an op the vector backend cannot codegen.
static bool isIllegalVecFuncArchOp(Operation *op, StringRef &dslName) {
  if (isa<::tla::BlockIdxOp>(op)) {
    dslName = "tla.arch.block_idx";
    return true;
  }
  if (isa<::tla::BlockDimOp>(op)) {
    dslName = "tla.arch.block_dim";
    return true;
  }
  if (isa<::tla::SubBlockIdxOp>(op)) {
    dslName = "tla.arch.sub_block_idx";
    return true;
  }
  return false;
}

// Fail compilation if any per-core identity query is used inside a tla.vec.func.
static LogicalResult checkNoArchOpsInVecFunc(func::FuncOp funcOp) {
  LogicalResult result = success();
  funcOp.walk([&](::tla::VecFuncOp vecFuncOp) {
    vecFuncOp.getBody().walk([&](Operation *op) {
      StringRef dslName;
      if (isIllegalVecFuncArchOp(op, dslName)) {
        op->emitOpError() << "'" << dslName
                          << "' is not allowed inside a tla.vec.func region; compute it "
                             "outside the region and pass the value in";
        result = failure();
      }
    });
  });
  return result;
}

class TlaVectorRegionPass : public PassWrapper<TlaVectorRegionPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TlaVectorRegionPass)

  StringRef getArgument() const override { return "tla-vector-region"; }
  StringRef getName() const override { return "TlaVectorRegionPass"; }
  StringRef getDescription() const override {
    return "Outline tla.vector regions and lower fragment ops to vector IR.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect, mlir::memref::MemRefDialect,
                    hivm::HIVMDialect, hivmave::AVEDialect, vector::VectorDialect,
                    ::tla::TlaDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    nextVectorRegionId = 0;

    // Snapshot the functions up front: lowering a vec.func appends a new
    // vector_region helper to the module, and that helper must not be fed back
    // through the lowering/folding driver (it already holds lowered AVE ops and
    // the carried scf control flow).
    SmallVector<func::FuncOp, 4> funcOps(module.getOps<func::FuncOp>());
    for (func::FuncOp funcOp : funcOps) {
      if (funcOp.isDeclaration())
        continue;
      // Skip the generated vector_region helpers: they already hold lowered AVE
      // ops and the carried scf control flow, and must not be re-driven.
      if (funcOp->hasAttr(kHivmVectorFunctionAttrName))
        continue;
      // Only AIV (and not-yet-split MIX) functions hold vector work. Their core
      // kind is the func_core_type set by the infer pass, falling back to the
      // module core type for pure-vector entries (whose func_core_type is
      // intentionally stripped by the HACC attr convention).
      std::optional<HivmCoreKind> coreKind = getExpectedFunctionCoreKind(funcOp.getOperation());
      if (coreKind != HivmCoreKind::AIV && coreKind != HivmCoreKind::MIX)
        continue;
      if (failed(checkNoArchOpsInVecFunc(funcOp))) {
        signalPassFailure();
        return;
      }
      inlineVectorRegionWrappers(funcOp);
      // Fresh per-function lowering state: the base-memref handoff cache shared
      // by LowerCopyPattern (gm<->ub / ub->l1 cifax runtime calls) and the
      // vec.func helper operand materialization.
      ::tla::TlaTensorMemrefLowering lowering;
      RewritePatternSet patterns(&getContext());
      populateTlaToVectorPatterns(patterns, module, nextVectorRegionId,
                                  lowering.loweredMemrefByValue);
      if (failed(mlir::applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
        signalPassFailure();
        return;
      }
    }

    // `module.getOps<func::FuncOp>()` visits only direct functions. Inline any
    // wrapper in nested modules so finalize never sees frontend scaffolding.
    SmallVector<::tla::VectorOp, 4> leftover;
    module.walk([&](::tla::VectorOp vectorOp) { leftover.push_back(vectorOp); });
    IRRewriter rewriter(module.getContext());
    for (::tla::VectorOp vectorOp : leftover) {
      if (!vectorOp)
        continue;
      if (vectorOp->getNumRegions() == 0 || vectorOp.getBody().empty()) {
        rewriter.eraseOp(vectorOp);
        continue;
      }
      Block *body = &vectorOp.getBody().front();
      rewriter.inlineBlockBefore(body, vectorOp->getBlock(), vectorOp->getIterator());
      rewriter.eraseOp(vectorOp);
    }

    // Barrier-only vec.func regions are inlined rather than outlined, and raw
    // TLAIR may place local barriers directly in a vector wrapper. Lower those
    // remaining operations here as well so this pass is the single owner of
    // tla.local_mem_bar lowering.
    SmallVector<::tla::LocalMemBarOp, 4> localMemBars;
    module.walk([&](::tla::LocalMemBarOp op) { localMemBars.push_back(op); });
    for (::tla::LocalMemBarOp op : localMemBars) {
      if (!op->getBlock())
        continue;
      rewriter.setInsertionPoint(op);
      if (failed(lowerLocalMemBar(rewriter, op))) {
        signalPassFailure();
        return;
      }
      rewriter.eraseOp(op);
    }
  }

private:
  int nextVectorRegionId = 0;
};

} // namespace

std::unique_ptr<Pass> createTlaVectorRegionPass() {
  return std::make_unique<TlaVectorRegionPass>();
}

void registerTlaVectorRegionPass() { PassRegistration<TlaVectorRegionPass>(); }

} // namespace tla
