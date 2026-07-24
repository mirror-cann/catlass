include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/DialectBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
// include "Dialect/Tla/IR/TlaOpTraits.h"
// SameOperandsAndResultShape
// let hasFolder = 1;

def TlaDialect : Dialect {
  let name = "tla";
  let summary = "Tla dialect schema for TlaIR matmul prototyping.";
  let description = [{
    Tla dialect schema for TlaIR matmul prototyping.
  }];
  let cppNamespace = "::tla";
}

class Tla_Op<string mnemonic, list<Trait> traits = []> : Op<TlaDialect, mnemonic, traits>;

def Tla_CrossFlagType : TypeDef<TlaDialect, "CrossFlag", []> {
  let mnemonic = "cross_flag";
  let summary = "Synchronization flag used for cross-core coordination.";
  let parameters = (ins "int64_t":$mode);
  let assemblyFormat = "`<` $mode `>`";
}

def Tla_FlagType : TypeDef<TlaDialect, "Flag", []> {
  let mnemonic = "flag";
  let summary = "Synchronization flag used for pipe coordination.";
  let assemblyFormat = "";
}

def Tla_MaskSSAType : TypeDef<TlaDialect, "MaskSSA", []> {
  let mnemonic = "mask";
  let summary = "Register-resident predicate mask SSA value.";
  let parameters = (ins "int64_t":$physicalLanes);
  let assemblyFormat = "`<` $physicalLanes `>`";
  let genVerifyDecl = 1;
}


def Tla_MutexType : TypeDef<TlaDialect, "Mutex", []> {
  let mnemonic = "mutex";
  let summary = "Synchronization mutex used for resource coordination.";
  let assemblyFormat = "";
}


def Tla_AddressSpaceTypeParameter
    : TypeParameter<"::AddressSpace", "Tla address-space keyword"> {
  let parser = [{ [&]() -> ::mlir::FailureOr<::AddressSpace> {
    ::llvm::StringRef kw;
    if ($_parser.parseKeyword(&kw))
      return ::mlir::failure();
    auto sym = ::symbolizeAddressSpace(kw);
    if (!sym)
      return $_parser.emitError($_parser.getCurrentLocation())
             << "unknown AddressSpace keyword: " << kw;
    return *sym;
  }() }];
  let printer = "$_printer << ::stringifyAddressSpace($_self)";
}

def Tla_LayoutTagTypeParameter
    : TypeParameter<"::LayoutTag", "Tla tensor layout keyword"> {
  let parser = [{ [&]() -> ::mlir::FailureOr<::LayoutTag> {
    ::llvm::StringRef kw;
    if ($_parser.parseKeyword(&kw))
      return ::mlir::failure();
    auto sym = ::symbolizeLayoutTag(kw);
    if (!sym)
      return $_parser.emitError($_parser.getCurrentLocation())
             << "unknown LayoutTag keyword: " << kw;
    return *sym;
  }() }];
  let printer = "$_printer << ::stringifyLayoutTag($_self)";
}

class Tla_IndexTreeType<string name, string typeMnemonic, string typeSummary>
    : TypeDef<TlaDialect, name, []> {
  let mnemonic = typeMnemonic;
  let summary = typeSummary;
  let parameters = (ins ArrayRefParameter<"int64_t", "encoded index-tree">:$tree);
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}

def Tla_ShapeType : Tla_IndexTreeType<"Shape", "shape", "Shape value for tile/broadcast.">;

def Tla_CoordType : Tla_IndexTreeType<"Coord", "coord", "Coordinate tuple for hierarchical tiling.">;

def Tla_StrideType : Tla_IndexTreeType<"Stride", "stride", "Stride tuple for layouts.">;

def Tla_VectorSSAType : TypeDef<TlaDialect, "VectorSSA", []> {
  let mnemonic = "vector";
  let summary = "Register-resident vector SSA value.";
  let parameters = (ins
    "int64_t":$validLanes,
    "::mlir::Type":$elementType
  );
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}

def Tla_ScalarValueType : AnyTypeOf<[AnyInteger, AnyFloat, Index],
    "TLA scalar SSA value type">;

def Tla_IntegerOrIndexType : AnyTypeOf<[AnyInteger, Index],
    "TLA integer-or-index SSA value type">;

def Tla_BoolAttrAttr : DialectAttr<TlaDialect, CPred<"true">, "Builtin boolean attribute placeholder.">;

def Tla_I64AttrAttr : DialectAttr<TlaDialect, CPred<"true">, "Builtin i64 attribute placeholder.">;

def Tla_StrAttrAttr : DialectAttr<TlaDialect, CPred<"true">, "Builtin string attribute placeholder.">;

def Tla_SymbolNameAttrAttr : DialectAttr<TlaDialect, CPred<"true">, "Builtin symbol name attribute placeholder.">;

def Tla_TypeAttrAttr : DialectAttr<TlaDialect, CPred<"true">, "Builtin type attribute placeholder.">;

class Tla_EnumAttr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<TlaDialect, name, traits> {
  let mnemonic = attrMnemonic;
}

class Tla_I32Enum<string name, string description, list<I32EnumAttrCase> cases>
    : I32EnumAttr<name, description, cases> {
}

def Tla_LayoutTag_RowMajor : I32EnumAttrCase<"row_major", 0>;
def Tla_LayoutTag_ColumnMajor : I32EnumAttrCase<"column_major", 1>;
def Tla_LayoutTag_zN : I32EnumAttrCase<"zN", 2>;
def Tla_LayoutTag_nZ : I32EnumAttrCase<"nZ", 3>;
def Tla_LayoutTag_zZ : I32EnumAttrCase<"zZ", 4>;
def Tla_LayoutTag_L0CLayout : I32EnumAttrCase<"L0Clayout", 5>;
def Tla_LayoutTagEnum : Tla_I32Enum<"LayoutTag", "Tensor layout tag for tla.tensor types.", [Tla_LayoutTag_RowMajor, Tla_LayoutTag_ColumnMajor, Tla_LayoutTag_zN, Tla_LayoutTag_nZ, Tla_LayoutTag_zZ, Tla_LayoutTag_L0CLayout]>;

def Tla_AddressSpace_Generic : I32EnumAttrCase<"generic", 0>;
def Tla_AddressSpace_Gm : I32EnumAttrCase<"gm", 1>;
def Tla_AddressSpace_L1 : I32EnumAttrCase<"l1", 2>;
def Tla_AddressSpace_L0a : I32EnumAttrCase<"l0a", 3>;
def Tla_AddressSpace_L0b : I32EnumAttrCase<"l0b", 4>;
def Tla_AddressSpace_L0c : I32EnumAttrCase<"l0c", 5>;
def Tla_AddressSpace_Ub : I32EnumAttrCase<"ub", 6>;
def Tla_AddressSpaceEnum : Tla_I32Enum<"AddressSpace", "Tla pointer/tensor/memref address space.", [Tla_AddressSpace_Generic, Tla_AddressSpace_Gm, Tla_AddressSpace_L1, Tla_AddressSpace_L0a, Tla_AddressSpace_L0b, Tla_AddressSpace_L0c, Tla_AddressSpace_Ub]>;

def Tla_PtrAddressSpaceAttr : Tla_EnumAttr<"PtrAddressSpace", "ptr_address_space"> {
  let parameters = (ins EnumParameter<Tla_AddressSpaceEnum>:$value);
  let assemblyFormat = "`<` $value `>`";
  let description = [{
    On-chip ``!tla.ptr`` address space; carried on pointer types (e.g. with ``tla.alloc_ptr``).
  }];
}


def Tla_PtrType : TypeDef<TlaDialect, "Ptr", []> {
  let summary = "First-class pointer SSA value (pointee, address space, alignment).";
  let description = [{
    A logical pointer that can flow through function calls, branches, and SCF
    region arguments/results like any other SSA value. ``tla-lower-ptr`` converts
    structural pointer carriers atomically to ``i64`` byte addresses.
  }];
  let mnemonic = "ptr";
  let parameters = (ins
    TypeParameter<"::mlir::Type", "Logical pointee / value element type.">:$pointee,
    Tla_AddressSpaceTypeParameter:$addrspace,
    "unsigned":$alignment
  );
  let assemblyFormat = "`<` $pointee `,` $addrspace `,` $alignment `>`";
}

def Tla_LayoutType : TypeDef<TlaDialect, "Layout", []> {
  let mnemonic = "layout";
  let summary = "Packed shape+stride+layout-tag descriptor (Tla-style).";
  let parameters = (ins
    TypeParameter<"::tla::ShapeType", "shape tree type">:$shape,
    TypeParameter<"::tla::StrideType", "stride tree type">:$stride,
    TypeParameter<"::tla::ShapeType", "origin shape">:$origin,
    Tla_LayoutTagTypeParameter:$layoutTag
  );
  let assemblyFormat = "`<` qualified($shape) `,` qualified($stride) `,` qualified($origin) `,` $layoutTag `>`";
  let genVerifyDecl = 1;
}

def Tla_TensorType : TypeDef<TlaDialect, "TlaTensor", []> {
  let summary = "First-class tensor/view with structured layout, coordinate, and pointer.";
  let description = [{
    Structured tensor descriptor:
    ``!tla.tensor<!tla.layout<...>, !tla.coord<...>, !tla.ptr<...>>``.
  }];
  let mnemonic = "tensor";
  let parameters = (ins
    TypeParameter<"::tla::LayoutType", "layout descriptor">:$layout,
    TypeParameter<"::tla::CoordType", "coord tree type">:$coord,
    TypeParameter<"::tla::PtrType", "backing pointer type">:$ptr
  );
  let assemblyFormat = "`<` qualified($layout) `,` qualified($coord) `,` qualified($ptr) `>`";
  let genVerifyDecl = 1;
}

def Tla_CrossMode_Npu : I32EnumAttrCase<"npu", 0>;
def Tla_CrossMode_VectorsCore : I32EnumAttrCase<"vectors_core", 1>;
def Tla_CrossMode_SingleCore : I32EnumAttrCase<"single_core", 2>;
def Tla_CrossModeEnum : Tla_I32Enum<"CrossMode", "Cross-core synchronization mode for tla.cross_flag.", [Tla_CrossMode_Npu, Tla_CrossMode_VectorsCore, Tla_CrossMode_SingleCore]>;
def Tla_CrossModeAttr : Tla_EnumAttr<"CrossMode", "cross_mode"> {
  let parameters = (ins EnumParameter<Tla_CrossModeEnum>:$cross_mode);
  let assemblyFormat = "`<` params `>`";
  let description = [{
    Cross-core synchronization mode for tla.cross_flag.
  }];
}

def Tla_Pipe_Scalar : I32EnumAttrCase<"scalar", 0>;
def Tla_Pipe_Vector : I32EnumAttrCase<"vector", 1>;
def Tla_Pipe_Cube : I32EnumAttrCase<"cube", 2>;
def Tla_Pipe_Mte1 : I32EnumAttrCase<"mte1", 3>;
def Tla_Pipe_Mte2 : I32EnumAttrCase<"mte2", 4>;
def Tla_Pipe_Mte3 : I32EnumAttrCase<"mte3", 5>;
def Tla_Pipe_All : I32EnumAttrCase<"all", 6>;
def Tla_Pipe_Mte4 : I32EnumAttrCase<"mte4", 7>;
def Tla_Pipe_Mte5 : I32EnumAttrCase<"mte5", 8>;
def Tla_Pipe_V2 : I32EnumAttrCase<"v2", 9>;
def Tla_Pipe_Fix : I32EnumAttrCase<"fix", 10>;
def Tla_Pipe_VirtualMte2L1a : I32EnumAttrCase<"virtual_mte2_l1a", 11>;
def Tla_Pipe_VirtualMte2L1b : I32EnumAttrCase<"virtual_mte2_l1b", 12>;
def Tla_Pipe_Num : I32EnumAttrCase<"num", 13>;
def Tla_PipeEnum : Tla_I32Enum<"Pipe", "Pipe identifier for tla.set_flag and tla.wait_flag.", [Tla_Pipe_Scalar, Tla_Pipe_Vector, Tla_Pipe_Cube, Tla_Pipe_Mte1, Tla_Pipe_Mte2, Tla_Pipe_Mte3, Tla_Pipe_All, Tla_Pipe_Mte4, Tla_Pipe_Mte5, Tla_Pipe_V2, Tla_Pipe_Fix, Tla_Pipe_VirtualMte2L1a, Tla_Pipe_VirtualMte2L1b, Tla_Pipe_Num]>;
def Tla_PipeAttr : Tla_EnumAttr<"Pipe", "pipe"> {
  let parameters = (ins EnumParameter<Tla_PipeEnum>:$pipe);
  let assemblyFormat = "`<` params `>`";
  let description = [{
    Pipe identifier for tla.set_flag and tla.wait_flag.
  }];
}

// QuantMode Enum Attr
def Tla_QuantMode_NoQuant : I32EnumAttrCase<"NO_QUANT", 0>;
def Tla_QuantMode_PerTensor : I32EnumAttrCase<"PER_TENSOR", 1>;
def Tla_QuantMode_PerChannel : I32EnumAttrCase<"PER_CHANNEL", 2>;
def Tla_QuantModeEnum : Tla_I32Enum<"QuantMode", "Quant Mode when l0c2dst.", [
  Tla_QuantMode_NoQuant,
  Tla_QuantMode_PerTensor,
  Tla_QuantMode_PerChannel
]>;
def Tla_QuantModeAttr : Tla_EnumAttr<"QuantMode", "quant_mode"> {
  let parameters = (ins EnumParameter<Tla_QuantModeEnum>:$quant_mode);
  let assemblyFormat = "`<` params `>`";
  let description = [{
    Copy L0C to Dst quant mode.
  }];
}

// L0C2UBMode Enum Attr
def Tla_L0C2UBMode_NoSplitVec0 : I32EnumAttrCase<"NO_SPLIT_VEC_0", 0>;
def Tla_L0C2UBMode_NoSplitVec1 : I32EnumAttrCase<"NO_SPLIT_VEC_1", 1>;
def Tla_L0C2UBMode_SplitM : I32EnumAttrCase<"SPLIT_M", 2>;
def Tla_L0C2UBMode_SplitN : I32EnumAttrCase<"SPLIT_N", 3>;
def Tla_L0C2UBModeEnum : Tla_I32Enum<"L0C2UBMode", "L0C 2 UB split mode.", [
  Tla_L0C2UBMode_NoSplitVec0,
  Tla_L0C2UBMode_NoSplitVec1,
  Tla_L0C2UBMode_SplitM,
  Tla_L0C2UBMode_SplitN
]>;
def Tla_L0C2UBModeAttr : Tla_EnumAttr<"L0C2UBMode", "l0c2ub_mode"> {
  let parameters = (ins EnumParameter<Tla_L0C2UBModeEnum>:$l0c2ub_mode);
  let assemblyFormat = "`<` params `>`";
  let description = [{
    Copy L0C 2 UB split mode.
  }];
}

// LoadDist Enum Attr
def Tla_LoadDist_Norm : I32EnumAttrCase<"norm", 0>;
def Tla_LoadDist_BrcB32 : I32EnumAttrCase<"brc_b32", 1>;
def Tla_LoadDist_DintlvB32 : I32EnumAttrCase<"dintlv_b32", 2>;
def Tla_LoadDistEnum : Tla_I32Enum<"LoadDist", "Load distribution for tla.load.", [
  Tla_LoadDist_Norm,
  Tla_LoadDist_BrcB32,
  Tla_LoadDist_DintlvB32
]>;
def Tla_LoadDistAttr : Tla_EnumAttr<"LoadDist", "load_dist"> {
  let parameters = (ins EnumParameter<Tla_LoadDistEnum>:$load_dist);
  let assemblyFormat = "`<` params `>`";
  let description = [{
    Broadcast/load distribution pattern for tla.load.
    ``dintlv_b32`` is dual-destination de-interleave load (even/odd b32
    elements into two vector registers); requires a second result.
  }];
}

// ---------------------------------------------------------------------------
// CastParams: the four knobs of a tla.cast, matching CastParams in the Python
// front-end (catlass/params.py). The enum keywords are the MLIR-asm tokens.
// ---------------------------------------------------------------------------

// RegSlot: which half of a packed register a narrow result lands in
// 2x casts use zero=part_even / one=part_odd; 4x casts (i32<->i8) map all four
// values onto the pack pattern zero=pp0 / one=pp1 / two=pp2 / three=pp3.
def Tla_RegSlot_Zero : I32EnumAttrCase<"zero", 0>;
def Tla_RegSlot_One : I32EnumAttrCase<"one", 1>;
def Tla_RegSlot_Two : I32EnumAttrCase<"two", 2>;
def Tla_RegSlot_Three : I32EnumAttrCase<"three", 3>;
def Tla_RegSlotEnum : Tla_I32Enum<"RegSlot", "Packed-register position for a narrow cast result.", [
  Tla_RegSlot_Zero,
  Tla_RegSlot_One,
  Tla_RegSlot_Two,
  Tla_RegSlot_Three
]>;

// ---------------------------------------------------------------------------
// AtomicMode: the atomic mode passing to ``tla.params.CopyParams``
// front-end (catlass/params.py). The enum keywords are the MLIR-asm tokens.
// ---------------------------------------------------------------------------

// AtomicMode: None, Add

def Tla_AtomicMode_None : I32EnumAttrCase<"none", 0>;
def Tla_AtomicMode_Add : I32EnumAttrCase<"add", 1>;

def Tla_AtomicMode_Enum : Tla_I32Enum<"AtomicMode", "Atomic type", [
  Tla_AtomicMode_None,
  Tla_AtomicMode_Add
]>;

def Tla_AtomicModeAttr : Tla_EnumAttr<"AtomicMode", "atomic_mode"> {
  let parameters = (ins EnumParameter<Tla_AtomicMode_Enum>:$atomic_mode);
  let assemblyFormat = "`<` params `>`";
  let description = [{
    Atomic operation kind for tla.copy

    The supported atomic mode listed below:
      - ADD
  }];
}

// SatMode: overflow behaviour of the cast (AVE `sat` BoolAttr).
def Tla_SatMode_Unknown : I32EnumAttrCase<"unknown", 0>;
def Tla_SatMode_Sat : I32EnumAttrCase<"sat", 1>;
def Tla_SatMode_Nosat : I32EnumAttrCase<"nosat", 2>;
def Tla_SatModeEnum : Tla_I32Enum<"SatMode", "Cast saturation behaviour on overflow.", [
  Tla_SatMode_Unknown,
  Tla_SatMode_Sat,
  Tla_SatMode_Nosat
]>;

// RoundMode: rounding applied on precision loss (HIVM round_mode).
def Tla_RoundMode_CastRound : I32EnumAttrCase<"cast_round", 0>;
def Tla_RoundMode_CastFloor : I32EnumAttrCase<"cast_floor", 1>;
def Tla_RoundMode_CastCeil : I32EnumAttrCase<"cast_ceil", 2>;
def Tla_RoundMode_CastTrunc : I32EnumAttrCase<"cast_trunc", 3>;
def Tla_RoundModeEnum : Tla_I32Enum<"RoundMode", "Cast rounding mode on precision loss.", [
  Tla_RoundMode_CastRound,
  Tla_RoundMode_CastFloor,
  Tla_RoundMode_CastCeil,
  Tla_RoundMode_CastTrunc
]>;

// The three cast knobs are carried on tla.cast as a builtin DenseI32ArrayAttr of
// three enum codes, in this fixed order (see catlass/params.py):
//   [0] reg_slot (RegSlot)   [1] sat_mode (SatMode)   [2] round_mode (RoundMode)
// A builtin array attr round-trips textually through every pass; the enum
// definitions above give the C++ enum types the lowering casts the codes to.

def Tla_CopyL0C2DstParamsType : TypeDef<TlaDialect, "CopyL0C2DstParams", []> {
  let mnemonic = "copy_l0c2dst_params";
  let summary = "CopyL0C2DstParams for tla.copy.";
  let assemblyFormat = "";
}

def Tla_BinaryOpsTrait : NativeOpTrait<"::mlir::tla::trait::BinaryOps">;
def Tla_BlockDimTrait : NativeOpTrait<"::mlir::tla::trait::BlockDim">;
def Tla_BlockIdxTrait : NativeOpTrait<"::mlir::tla::trait::BlockIdx">;
def Tla_ControlOpsTrait : NativeOpTrait<"::mlir::tla::trait::ControlOps">;
def Tla_CopyTrait : NativeOpTrait<"::mlir::tla::trait::Copy">;
def Tla_CrossCoreSetFlagTrait : NativeOpTrait<"::mlir::tla::trait::CrossCoreSetFlag">;
def Tla_CrossCoreWaitFlagTrait : NativeOpTrait<"::mlir::tla::trait::CrossCoreWaitFlag">;
def Tla_CrossFlagTrait : NativeOpTrait<"::mlir::tla::trait::CrossFlag">;
def Tla_CubeTrait : NativeOpTrait<"::mlir::tla::trait::Cube">;
def Tla_CubeTernaryOpsTrait : NativeOpTrait<"::mlir::tla::trait::CubeTernaryOps">;
def Tla_DependentBinaryOpsTrait : NativeOpTrait<"::mlir::tla::trait::DependentBinaryOps">;
def Tla_DependentUnaryOpsTrait : NativeOpTrait<"::mlir::tla::trait::DependentUnaryOps">;
def Tla_FlagTrait : NativeOpTrait<"::mlir::tla::trait::Flag">;
def Tla_IntermediateBinaryOpsTrait : NativeOpTrait<"::mlir::tla::trait::IntermediateBinaryOps">;
def Tla_IntermediateMemoryOpsTrait : NativeOpTrait<"::mlir::tla::trait::IntermediateMemoryOps">;
def Tla_IntermediateUnaryOpsTrait : NativeOpTrait<"::mlir::tla::trait::IntermediateUnaryOps">;
def Tla_LoadTrait : NativeOpTrait<"::mlir::tla::trait::Load">;
def Tla_MakeTileLikeTrait : NativeOpTrait<"::mlir::tla::trait::MakeTileLike">;
def Tla_MemoryOpsTrait : NativeOpTrait<"::mlir::tla::trait::MemoryOps">;
def Tla_AllocateTrait : NativeOpTrait<"::mlir::tla::trait::Allocate">;
def Tla_MmadTrait : NativeOpTrait<"::mlir::tla::trait::Mmad">;
def Tla_OpClassBinaryTrait : NativeOpTrait<"::mlir::tla::trait::OpClassBinary">;
def Tla_OpClassControlTrait : NativeOpTrait<"::mlir::tla::trait::OpClassControl">;
def Tla_OpClassMemoryTrait : NativeOpTrait<"::mlir::tla::trait::OpClassMemory">;
def Tla_OpClassUnaryTrait : NativeOpTrait<"::mlir::tla::trait::OpClassUnary">;
def Tla_PipeBarrierTrait : NativeOpTrait<"::mlir::tla::trait::PipeBarrier">;
def Tla_PipeCubeOpsTrait : NativeOpTrait<"::mlir::tla::trait::PipeCubeOps">;
def Tla_PipeDependentOpsTrait : NativeOpTrait<"::mlir::tla::trait::PipeDependentOps">;
def Tla_PipeIntermediateOpsTrait : NativeOpTrait<"::mlir::tla::trait::PipeIntermediateOps">;
def Tla_PipeScalarOpsTrait : NativeOpTrait<"::mlir::tla::trait::PipeScalarOps">;
def Tla_PipeVectorOpsTrait : NativeOpTrait<"::mlir::tla::trait::PipeVectorOps">;
def Tla_RegionControlOpsTrait : NativeOpTrait<"::mlir::tla::trait::RegionControlOps">;
def Tla_RegionOpsTrait : NativeOpTrait<"::mlir::tla::trait::RegionOps">;
def Tla_ScalarControlOpsTrait : NativeOpTrait<"::mlir::tla::trait::ScalarControlOps">;
def Tla_SetFlagTrait : NativeOpTrait<"::mlir::tla::trait::SetFlag">;
def Tla_StoreTrait : NativeOpTrait<"::mlir::tla::trait::Store">;
def Tla_SubtileTrait : NativeOpTrait<"::mlir::tla::trait::Subtile">;
def Tla_TernaryOpsTrait : NativeOpTrait<"::mlir::tla::trait::TernaryOps">;
def Tla_TileTrait : NativeOpTrait<"::mlir::tla::trait::Tile">;
def Tla_TileCountTrait : NativeOpTrait<"::mlir::tla::trait::TileCount">;
def Tla_UnaryOpsTrait : NativeOpTrait<"::mlir::tla::trait::UnaryOps">;
def Tla_VectorTrait : NativeOpTrait<"::mlir::tla::trait::Vector">;
def Tla_VectorBinaryOpsTrait : NativeOpTrait<"::mlir::tla::trait::VectorBinaryOps">;
def Tla_WaitFlagTrait : NativeOpTrait<"::mlir::tla::trait::WaitFlag">;

def Tla_BlockDimOp : Tla_Op<"arch.block_dim", [Pure]> {
  let summary = "Block dimension";
  let description = [{
    Return the grid block dimension as signless ``i32``.
  }];
  let results = (outs I32:$dim);
  let assemblyFormat = [{
    attr-dict `->` type($dim)
  }];
}

def Tla_BlockIdxOp : Tla_Op<"arch.block_idx", [Pure]> {
  let summary = "Block index";
  let description = [{
    Return the current block index as signless ``i32``.
  }];
  let results = (outs I32:$index);
  let assemblyFormat = [{
    attr-dict `->` type($index)
  }];
}

def Tla_SubBlockIdxOp : Tla_Op<"arch.sub_block_idx", [Pure]> {
  let summary = "Sub-block index";
  let description = [{
    Return the current sub-block index as signless ``i32``.
  }];
  let results = (outs I32:$index);
  let assemblyFormat = [{
    attr-dict `->` type($index)
  }];
}

def Tla_IntToPtrOp : Tla_Op<"inttoptr", [Pure]> {
  let summary = "Integer SSA to tla.ptr (``tla.inttoptr`` parity)";
  let description = [{
    Reinterprets an integer-like SSA address as the given ``!tla.ptr`` type.
    ``tla-lower-ptr`` normally converts it back to an ``i64`` byte address; an
    instance may remain temporarily at a ``tla.make_tensor{,_like}`` boundary so
    tensor lowering can construct a consumer-local memref descriptor.
  }];
  let arguments = (ins AnyType:$addr);
  let results = (outs Tla_PtrType:$ptr);
  let assemblyFormat = [{
    $addr attr-dict `:` type($addr) `->` qualified(type($ptr))
  }];
}

def Tla_RecastPtrOp : Tla_Op<"recast_ptr", [Pure]> {
  let summary = "Recast pointer element type (``tla.recast_ptr`` parity, dtype-only)";
  let description = [{
    Reinterprets a ``!tla.ptr`` SSA as a different logical pointee type while preserving
    address space and alignment.
    **No swizzle** metadata (Tla does not model swizzle on ``!tla.ptr``).
  }];
  let arguments = (ins Tla_PtrType:$src);
  let results = (outs Tla_PtrType:$ptr);
  let assemblyFormat = [{
    $src attr-dict `:` qualified(type($src)) `->` qualified(type($ptr))
  }];
}

def Tla_TensorPtrOp : Tla_Op<"tensor_ptr", [Pure]> {
  let summary = "Extract the backing pointer from a tla.tensor";
  let description = [{
    Yields the ``!tla.ptr`` SSA value backing a ``!tla.tensor``. Kernel arguments
    are represented inside the function by a root ``tla.tensor_desc``.
  }];
  let arguments = (ins Tla_TensorType:$src);
  let results = (outs Tla_PtrType:$ptr);
  let assemblyFormat = [{
    $src attr-dict `:` qualified(type($src)) `->` qualified(type($ptr))
  }];
  let hasVerifier = 1;
}

def Tla_PtrAddOp : Tla_Op<"ptr_add", [Pure]> {
  let summary = "Offset a tla.ptr by a scalar element count";
  let description = [{
    Returns a ``!tla.ptr`` advanced by ``offset`` **elements** of the pointer's pointee
    type (not bytes). The result preserves the source pointee type and address space;
    alignment is refined conservatively from a known constant offset (gcd), otherwise
    dropped to 1. The offset operand is an integer-or-index scalar SSA value.
  }];
  let arguments = (ins Tla_PtrType:$ptr, Tla_IntegerOrIndexType:$offset);
  let results = (outs Tla_PtrType:$result);
  let assemblyFormat = [{
    $ptr `,` $offset attr-dict `:` qualified(type($ptr)) `,` type($offset) `->` qualified(type($result))
  }];
  let hasVerifier = 1;
}

def Tla_FullOp : Tla_Op<"full", [Pure]> {
  let summary = "Fill vector lanes with scalar";
  let description = [{
    Broadcast a scalar value into the vector helper register type selected by the
    enclosing vector lowering context.
  }];
  let arguments = (ins AnyTypeOf<[I1, I8, I16, I32, I64, BF16, F16, F32]>:$value);
  let results = (outs Tla_VectorSSAType:$result);
  let assemblyFormat = [{
    $value attr-dict `:` type($value) `->` qualified(type($result))
  }];
  let hasVerifier = 1;
}

def Tla_ArangeOp : Tla_Op<"arange", [Pure]> {
  let summary = "Create monotonic lane indices in a vector register";
  let description = [{
    Fill the vector helper register with a monotonic sequence.
    
    With ``order = increase``(default), lane ``i`` holds ``base + i`` (i.e. ``start``, ``start + 1``, ...).
    With ``order = decrease``, lane ``i`` holds ``base + VL - 1 - i`` (i.e. ``start + VL - 1``, ``start + VL - 2``, ..., ``start``).
    
    Lowers to AVE ``vci`` with ``INCREASE`` / ``DECREASE`` mode.
  }];
  let arguments = (ins AnyTypeOf<[I8, I16, I32, I64]>:$start,
                       StrAttr:$order);
  let results = (outs Tla_VectorSSAType:$result);
  let assemblyFormat = [{
    $start attr-dict `:` type($start) `->` qualified(type($result))
  }];
  let hasVerifier = 1;
}

def Tla_CopyL0C2DstParamsOp : Tla_Op<"CopyL0C2DstParams", [Pure]> {
  let summary = "CopyL0C2DstParams op";
  let arguments = (ins 
  // attributes
    I64Attr:$unit_flag,
    BoolAttr:$relu_enable,
    Tla_QuantModeAttr:$quant_mode,
    Tla_L0C2UBModeAttr:$l0c2ub_mode,
  // operands
    Optional<AnyTypeOf<[F32, Tla_TensorType]>>:$quant_scale_or_tensor
  );
  let results = (outs Tla_CopyL0C2DstParamsType:$params);
}

def Tla_CopyOp : Tla_Op<"copy", []> {
  let summary = "Copy";
  let description = [{
    Copy between tensors/views in different address spaces. Cube data-path
    copies (GM->L1, L1->L0A, L1->L0B, L0C->GM, L0C->UB, L1->UB) must be nested
    inside a ``tla.cube`` region; vector staging copies (GM->UB, UB->GM, UB->L1)
    must be nested inside a ``tla.vector`` region.

    The atomic operation is enabled if ``atomic_mode`` provided, and currently only ``AtomicMode.ADD`` is supported, which is lowered to HIVM ``SetAtomicOp``
  }];
  let arguments = (ins
    Tla_TensorType:$dst, Tla_TensorType:$src,
    Optional<AnyTypeOf<[Tla_CopyL0C2DstParamsType]>>:$params,
    OptionalAttr<Tla_AtomicModeAttr>:$atomic_mode);
  let assemblyFormat = [{
    $dst `,` $src (`,` $params^)? attr-dict `:` qualified(type($dst)) `,` qualified(type($src)) (`,` qualified(type($params))^)?
  }];
  let hasVerifier = 1;
}

def Tla_CrossCoreSetFlagOp : Tla_Op<"cross_core_set_flag", []> {
  let summary = "Cross-core flag set";
  let description = [{
    Set a cross-core synchronization flag.
  }];
  let arguments = (ins Tla_CrossFlagType:$flag, Tla_PipeAttr:$pipe,
                       OptionalAttr<I64Attr>:$aiv_id);
  let assemblyFormat = [{
    $flag attr-dict `:` qualified(type($flag))
  }];
  let hasVerifier = 1;
}

def Tla_CrossCoreWaitFlagOp : Tla_Op<"cross_core_wait_flag", []> {
  let summary = "Cross-core flag wait";
  let description = [{
    Wait on a cross-core synchronization flag.
  }];
  let arguments = (ins Tla_CrossFlagType:$flag, Tla_PipeAttr:$pipe,
                       OptionalAttr<I64Attr>:$aiv_id);
  let assemblyFormat = [{
    $flag attr-dict `:` qualified(type($flag))
  }];
  let hasVerifier = 1;
}

def Tla_CrossFlagOp : Tla_Op<"cross_flag", [Pure]> {
  let summary = "Cross-block flag literal";
  let description = [{
    Materialize a cross-core synchronization flag.
  }];
  let arguments = (ins StrAttr:$name);
  let results = (outs Tla_CrossFlagType:$flag);
  let assemblyFormat = [{
    $name attr-dict `->` qualified(type($flag))
  }];
}

def Tla_CubeOp : Tla_Op<"cube", [NoTerminator, SingleBlock]> {
  let summary = "Cube region";
  let description = [{
    Launch a cube region for matrix multiply and related operations.
  }];
  let regions = (region AnyRegion:$body);
  let assemblyFormat = [{
    $body attr-dict
  }];
}

def Tla_FlagOp : Tla_Op<"flag", [Pure]> {
  let summary = "Flag literal";
  let description = [{
    Materialize a static synchronization flag.
  }];
  let arguments = (ins StrAttr:$name, Tla_PipeAttr:$src_pipe, Tla_PipeAttr:$dst_pipe);
  let results = (outs Tla_FlagType:$flag);
  let assemblyFormat = [{
    $name attr-dict `->` qualified(type($flag))
  }];
}

def Tla_FuncOp : Tla_Op<"func", []> {
  let summary = "Tla kernel function";
  let description = [{
    Defines a Tla kernel entry point. The body is an isolated region with Tla IR operations.
  }];
  let arguments = (ins SymbolNameAttr:$sym_name, TypeAttr:$function_type);
  let regions = (region AnyRegion:$body);
  let hasCustomAssemblyFormat = 1;
}

def Tla_LoadOp : Tla_Op<"load", []> {
  let summary = "Load";
  let description = [{
    Load a UB tensor tile into vector SSA. The source tensor must be in the UB
    address space, and the operation must be nested inside a ``tla.vec.func``
    region. When ``unaligned_ub_access`` is set, lowers to ``ave.hir.vload`` with
    unaligned UB access (``vldas``/``vldus`` inserted by the HIVMAVE pipeline).
    Dual-destination ``dintlv_b32`` produces a second result (even/odd b32
    elements), lowering to ``ave.hir.vload <DINTLV_B32>``.
  }];
  let arguments = (ins Tla_TensorType:$source,
                       OptionalAttr<Tla_LoadDistAttr>:$load_dist,
                       OptionalAttr<UnitAttr>:$unaligned_ub_access);
  let results = (outs Tla_VectorSSAType:$result,
                      Optional<Tla_VectorSSAType>:$result2);
  let assemblyFormat = [{
    $source attr-dict `:` type($source) `->` qualified(type($result))
    ( `,` qualified(type($result2))^)?
  }];
  let hasVerifier = 1;
}

// Scratch alloc uses a static ``size_bytes`` attribute (``i64``) on ``tla.alloc_ptr``.
def Tla_AllocPtrOp : Tla_Op<"alloc_ptr", [Pure]> {
  let summary = "Static on-chip scratch bytes";
  let description = [{
    Reserves one statically assigned ``size_bytes`` on-chip scratch slot. The slot
    lifetime is the whole kernel: executing an alloc nested in a region or loop
    yields the same address on every dynamic execution. Address space and element
    alignment appear only on ``!tla.ptr<pointee, addrspace, align>``.
    ``tla-lower-ptr`` replaces the result with its static ``i64`` byte address.
  }];
  let arguments = (ins I64Attr:$size_bytes);
  let results = (outs Tla_PtrType:$ptr);
  let assemblyFormat = [{
    `{` `size_bytes` `=` $size_bytes `}` attr-dict `->` qualified(type($ptr))
  }];
  let hasVerifier = 1;
}

def Tla_MakeTensorLikeOp : Tla_Op<"make_tensor_like", []> {
  let summary = "Make tensor like";
  let description = [{
    Reconstruct a typed local tensor from a raw allocation pointer and a reference tensor.
  }];
  let arguments = (ins Tla_PtrType:$ptr, Tla_TensorType:$like, StrAttr:$layoutTag);
  let results = (outs Tla_TensorType:$tile);
  let assemblyFormat = [{
    $ptr `like` $like `layoutTag` `(` $layoutTag `)` attr-dict `:` qualified(type($ptr)) `,` qualified(type($like)) `->` qualified(type($tile))
  }];
}

def Tla_MakeTensorOp : Tla_Op<"make_tensor", [Pure]> {
  let summary = "Construct a tensor from an explicit pointer, layout, and coord";
  let description = [{
    Build a `!tla.tensor<layout, coord, ptr>` from explicit operands. Unlike
    `tla.make_tensor_like` (which clones layout/coord from a reference tensor), this op
    takes the layout and coord directly as operands, mirroring the tensor type's
    `(layout, coord, ptr)` structure. Address space and element type come from the `ptr`
    operand's `!tla.ptr` type; the layout tag, shape, stride, and origin come from the
    `!tla.layout` operand.
  }];
  let arguments = (ins Tla_PtrType:$ptr, Tla_LayoutType:$layout, Tla_CoordType:$coord);
  let results = (outs Tla_TensorType:$result);
  let assemblyFormat = [{
    $ptr `,` $layout `,` $coord attr-dict `:`
    qualified(type($ptr)) `,` qualified(type($layout)) `,` qualified(type($coord)) `->` qualified(type($result))
  }];
}

def Tla_TensorDescOp : Tla_Op<"tensor_desc", [Pure]> {
  let summary = "A tile's fully-resolved tensor descriptor as a single op";
  let description = [{
    Materialized by ``tla-lower-tensor-desc`` from a tensor-view producer chain.
    ``base`` is the resolved root storage: either a builtin ``memref`` kernel
    argument or a ``!tla.ptr`` whose pointer transforms have already been applied.
    The remaining operands are the descriptor's SSA-valued coordinates, strides,
    shapes, and origin shapes. ``packed`` is empty for a linear layout or contains
    ``[packed_shape0..3, packed_stride0..3]`` for a packed layout.

    Downstream passes consume this op without interpreting the producer chain.
  }];
  let arguments = (ins
    AnyTypeOf<[Tla_PtrType, AnyMemRef]>:$base,
    Index:$row_offset, Index:$col_offset,
    Index:$stride0, Index:$stride1,
    Index:$shape0, Index:$shape1,
    Index:$origin_shape0, Index:$origin_shape1,
    Variadic<Index>:$packed);
  let results = (outs Tla_TensorType:$result);
  let assemblyFormat = [{
    $base `[` $row_offset `,` $col_offset `,` $stride0 `,` $stride1 `,` $shape0 `,`
    $shape1 `,` $origin_shape0 `,` $origin_shape1 `]` (`packed` `[` $packed^ `]`)?
    attr-dict `:` functional-type(operands, results)
  }];
  let hasVerifier = 1;
}

def Tla_MmadOp : Tla_Op<"mmad", []> {
  let summary = "Matrix multiply-accumulate";
  let description = [{
    Perform a matrix multiply-accumulate on tensor views. Must be nested inside a
    ``tla.cube`` region.
  }];
  let arguments = (ins Tla_TensorType:$acc, Tla_TensorType:$lhs, Tla_TensorType:$rhs, I1:$init_c, I64:$unit_flag);
  let assemblyFormat = [{
    $acc `,` $lhs `,` $rhs `,` $init_c `,` $unit_flag attr-dict `:`
    qualified(type($acc)) `,` qualified(type($lhs)) `,` qualified(type($rhs)) `,` type($init_c) `,` type($unit_flag)
  }];
  let hasVerifier = 1;
}

def Tla_PipeBarrierOp : Tla_Op<"pipe_barrier", []> {
  let summary = "Pipe barrier";
  let description = [{
    Apply a synchronization barrier to the specified pipe.
  }];
  let arguments = (ins Tla_PipeAttr:$pipe);
  let assemblyFormat = [{
    `[` $pipe `]` attr-dict
  }];
  let hasVerifier = 1;
}

def Tla_DebugPrintOp : Tla_Op<"debug_print", [MemoryEffects<[MemWrite]>]> {
  let summary = "Typed frontend debug print op";
  let description = [{
    Side-effecting TLA IR for exactly one builtin signless i32 or f32 value.
    Backend lowering is intentionally unsupported.
  }];
  let arguments = (ins AnyType:$value);
  let assemblyFormat = "$value attr-dict `:` type($value)";
  let hasVerifier = 1;
}

def Tla_MutexOp : Tla_Op<"mutex", [Pure]> {
  let summary = "Mutex literal";
  let description = [{
    Materialize a static mutex associated with a semantic resource.
    ``id = -1`` means the compiler may allocate the hardware MutexID later;
    ``0`` through ``31`` are user-provided hardware MutexIDs.
  }];
  let arguments = (ins StrAttr:$resource, I64Attr:$id);
  let results = (outs Tla_MutexType:$mutex);
  let assemblyFormat = [{
    $resource attr-dict `->` qualified(type($mutex))
  }];
}

def Tla_MutexLockOp : Tla_Op<"mutex_lock", []> {
  let summary = "Mutex lock";
  let description = [{
    Acquire a mutex from the specified pipe.
  }];
  let arguments = (ins Tla_MutexType:$mutex, Tla_PipeAttr:$pipe);
  let assemblyFormat = [{
    $mutex `[` $pipe `]` `:` qualified(type($mutex)) attr-dict
  }];
  let hasVerifier = 1;
}

def Tla_MutexUnlockOp : Tla_Op<"mutex_unlock", []> {
  let summary = "Mutex unlock";
  let description = [{
    Release a mutex from the specified pipe.
  }];
  let arguments = (ins Tla_MutexType:$mutex, Tla_PipeAttr:$pipe);
  let assemblyFormat = [{
    $mutex `[` $pipe `]` `:` qualified(type($mutex)) attr-dict
  }];
  let hasVerifier = 1;
}

def Tla_LocalMemBarOp : Tla_Op<"local_mem_bar", []> {
  let summary = "Local memory barrier";
  let description = [{
    Apply a synchronization barrier to the specified memory type.
    The ``barrier_kind`` is an encoded I32 immediate that selects the
    src/dst pair (see hivmave.membar for the encoding scheme).
  }];
  let arguments = (ins I64Attr:$barrier_kind);
  let assemblyFormat = "$barrier_kind attr-dict";
  let hasVerifier = 1;
}

def Tla_MakeShapeOp : Tla_Op<"make_shape", [Pure]> {
  let summary = "Shape value";
  let description = [{
    Build a ``!tla.shape`` from operands for each dynamic dimension; rank and static sizes are fixed by the result type.
  }];
  let arguments = (ins Variadic<Index>:$dyn_elems);
  let results = (outs Tla_ShapeType:$result);
  let hasCustomAssemblyFormat = 1;
}

def Tla_MakeCoordOp : Tla_Op<"make_coord", [Pure]> {
  let summary = "Coord value";
  let description = [{
    Build a ``!tla.coord`` from operands for each dynamic leaf; structure and static leaves are fixed by the result type.
  }];
  let arguments = (ins Variadic<Index>:$dyn_elems);
  let results = (outs Tla_CoordType:$result);
  let hasCustomAssemblyFormat = 1;
}

def Tla_MakeStrideOp : Tla_Op<"make_stride", [Pure]> {
  let summary = "Stride value";
  let description = [{
    Build a ``!tla.stride`` from operands for each dynamic leaf; structure and static leaves are fixed by the result type.
  }];
  let arguments = (ins Variadic<Index>:$dyn_elems);
  let results = (outs Tla_StrideType:$result);
  let hasCustomAssemblyFormat = 1;
}

def Tla_MakeLayoutOp : Tla_Op<"make_layout", [Pure]> {
  let summary = "Layout value";
  let description = [{
    Combine a packed ``!tla.shape`` and ``!tla.stride`` into a ``!tla.layout``, analogous to ``tla.make_layout(shape, stride)``.
    The result type encodes tuple-tree structure as ``!tla.layout<!tla.shape<shape_repr>, !tla.stride<stride_repr>, layout_tag>``, or
    ``!tla.layout<!tla.shape<shape_repr>, !tla.stride<stride_repr>, !tla.shape<origin_shape_repr>, layout_tag>`` when an explicit origin shape is used.

    Optional ``origin_shape`` records the logical extent (e.g. pre-padding matrix size). When omitted,
    consumers should treat the layout's logical origin shape as identical to ``shape``.
  }];
  let arguments = (ins Tla_ShapeType:$shape,
                       Tla_StrideType:$stride,
                       Optional<Tla_ShapeType>:$origin_shape,
                       OptionalAttr<StrAttr>:$layoutTag);
  let results = (outs Tla_LayoutType:$result);
  let assemblyFormat = [{
    $shape `,` $stride (`origin` $origin_shape^)? attr-dict `:`
    qualified(type($shape)) `,` qualified(type($stride))
    (`origin` qualified(type($origin_shape))^)? `->` qualified(type($result))
  }];
}

def Tla_ReturnOp : Tla_Op<"return", [Terminator]> {
  let summary = "Return";
  let description = [{
    Terminator for tla.func regions.
  }];
  let assemblyFormat = [{
    attr-dict
  }];
}

def Tla_SetFlagOp : Tla_Op<"set_flag", []> {
  let summary = "Set flag";
  let description = [{
    Signal a synchronization flag from one pipe to another.
  }];
  let arguments = (ins Tla_FlagType:$flag);
  let assemblyFormat = [{
    $flag `:` qualified(type($flag)) attr-dict
  }];
  let hasVerifier = 1;
}

def Tla_StoreOp : Tla_Op<"store", []> {
  let summary = "Store";
  let description = [{
    Store vector SSA into a UB tensor tile. The destination tensor must be in
    the UB address space. An optional mask controls which lanes are written
    (masked-out lanes are left untouched). When ``unaligned_ub_access`` is set,
    lowers to ``ave.hir.vstore`` with unaligned UB access (``vstas``/``vstus``
    inserted by the HIVMAVE pipeline).
    Must be nested inside a ``tla.vec.func`` region.
  }];
  let arguments = (ins Tla_TensorType:$dest, Tla_VectorSSAType:$source,
                       Optional<Tla_MaskSSAType>:$mask,
                       OptionalAttr<UnitAttr>:$unaligned_ub_access);
  let assemblyFormat = [{
    $dest `,` $source (`mask` $mask^)? attr-dict `:` type($dest) `,` qualified(type($source))
    (`mask` qualified(type($mask))^)?
  }];
  let hasVerifier = 1;
}

def Tla_CreateMaskOp : Tla_Op<"create_mask", [Pure]> {
  let summary = "Create a vector mask from a fixed pattern";
  let description = [{
    Create an element-wise mask from a fixed AVE ``pge`` pattern, e.g.
    ``ALL``, ``ALLF``, ``VL8`` (lowest 8 lanes), ``H`` (lowest half), ``Q``
    (lowest quarter), ``M3``/``M4`` (multiples of 3/4).

    ``dtype`` fixes the mask lane count (256 bytes / dtype size) and must match
    the enclosing vector region's element width.
  }];
  let arguments = (ins StrAttr:$pattern, TypeAttr:$dtype);
  let results = (outs Tla_MaskSSAType:$result);
  let assemblyFormat = [{
    attr-dict `:` qualified(type($result))
  }];
  let hasVerifier = 1;
}

def Tla_UpdateMaskOp : Tla_Op<"update_mask", [Pure]> {
  let summary = "Tail mask plus the remaining element count";
  let description = [{
    Tail-loop helper mapping to AVE ``plt``. Given the number of elements still
    to process (``true_shape``), produce a mask (lane ``i`` is active
    iff ``i < true_shape``, saturating to all-true once ``true_shape`` reaches
    the lane count) together with the remaining count after this iteration,
    ``new_true_shape = true_shape - lanes`` where ``lanes = 256 bytes / dtype
    size`` (= 64 for f32). Thread ``new_true_shape`` back through the loop's
    carried values to mask successive chunks, including the partial tail.

    ``dtype`` fixes the lane count and therefore the decrement; it must match the
    enclosing vector region's element width.
  }];
  let arguments = (ins Index:$true_shape, TypeAttr:$dtype);
  let results = (outs Tla_MaskSSAType:$mask, Index:$new_true_shape);
  let assemblyFormat = [{
    $true_shape `,` $dtype attr-dict `:` qualified(type($mask)) `,` type($new_true_shape)
  }];
  let hasVerifier = 1;
}

class Tla_BitwiseUnaryOp<string mnemonic, string opSummary, string opDescription>
    : Tla_Op<mnemonic, [Pure]> {
  let summary = opSummary;
  let description = opDescription # [{

    The ``mask`` operand predicates which lanes are computed; masked-out lanes
    are undefined/zeroed. When omitted, all lanes are active.
  }];
  let arguments = (ins AnyTypeOf<[Tla_VectorSSAType, Tla_MaskSSAType]>:$operand,
                       Optional<Tla_MaskSSAType>:$mask);
  let results = (outs AnyTypeOf<[Tla_VectorSSAType, Tla_MaskSSAType]>:$result);
  let assemblyFormat = [{
    $operand (`mask` $mask^)? attr-dict `:`
    qualified(type($operand)) (`mask` qualified(type($mask))^)?
    `->` qualified(type($result))
  }];
  let hasVerifier = 1;
}

class Tla_BitwiseBinaryOp<string mnemonic, string opSummary, string opDescription>
    : Tla_Op<mnemonic, [Pure]> {
  let summary = opSummary;
  let description = opDescription # [{

    The ``mask`` operand predicates which lanes are computed; masked-out lanes
    are undefined/zeroed. When omitted, all lanes are active.
  }];
  let arguments = (ins AnyTypeOf<[Tla_VectorSSAType, Tla_MaskSSAType]>:$lhs,
                       AnyTypeOf<[Tla_VectorSSAType, Tla_MaskSSAType]>:$rhs,
                       Optional<Tla_MaskSSAType>:$mask);
  let results = (outs AnyTypeOf<[Tla_VectorSSAType, Tla_MaskSSAType]>:$result);
  let assemblyFormat = [{
    $lhs `,` $rhs (`mask` $mask^)? attr-dict `:`
    qualified(type($lhs)) `,` qualified(type($rhs))
    (`mask` qualified(type($mask))^)?
    `->` qualified(type($result))
  }];
  let hasVerifier = 1;
}

def Tla_BitwiseNotOp : Tla_BitwiseUnaryOp<"bitwise_not",
    "Element-wise bitwise NOT",
    "Element-wise bitwise NOT on MaskSSA or VectorSSA values.">;

def Tla_BitwiseAndOp : Tla_BitwiseBinaryOp<"bitwise_and",
    "Element-wise bitwise AND",
    "Element-wise bitwise AND on MaskSSA or VectorSSA values.">;

def Tla_BitwiseOrOp : Tla_BitwiseBinaryOp<"bitwise_or",
    "Element-wise bitwise OR",
    "Element-wise bitwise OR on MaskSSA or VectorSSA values.">;

def Tla_BitwiseXorOp : Tla_BitwiseBinaryOp<"bitwise_xor",
    "Element-wise bitwise XOR",
    "Element-wise bitwise XOR on MaskSSA or VectorSSA values.">;

def Tla_CmpOp : Tla_Op<"cmp", [Pure]> {
  let summary = "Create a mask by comparing a vector with a vector or scalar";
  let description = [{
    Data-dependent mask: lane ``i`` is true when ``lhs[i] <mode> rhs[i]``.
    ``rhs`` may be a VectorSSA value or scalar SSA value. ``mode`` is one of
    ``lt``, ``le``, ``gt``, ``ge``, ``eq``, ``ne``. An optional mask controls
    which lanes participate in the compare.
  }];
  let arguments = (ins Tla_VectorSSAType:$lhs,
                       AnyTypeOf<[Tla_VectorSSAType, Tla_ScalarValueType]>:$rhs,
                       StrAttr:$mode,
                       Optional<Tla_MaskSSAType>:$mask);
  let results = (outs Tla_MaskSSAType:$result);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $mode $lhs `,` $rhs (`mask` $mask^)? attr-dict `:`
    qualified(type($lhs)) `,` qualified(type($rhs)) (`mask` qualified(type($mask))^)? `->` qualified(type($result))
  }];
}

def Tla_TileViewOp : Tla_Op<"tile_view", []> {
  let summary = "Tile view";
  let description = [{
    Create a tensor view over an existing ``!tla.tensor`` using packed
    ``!tla.shape`` and ``!tla.coord`` values.
  }];
  let arguments = (ins Tla_TensorType:$source, Tla_ShapeType:$shape, Tla_CoordType:$coord, OptionalAttr<StrAttr>:$layouttag);
  let results = (outs Tla_TensorType:$view);
  let assemblyFormat = [{
    $source `,` $shape `,` $coord attr-dict `:` qualified(type($source)) `,` qualified(type($shape)) `,` qualified(type($coord)) `->` qualified(type($view))
  }];
}

def Tla_ScalarLoadOp : Tla_Op<"scalar_load", []> {
  let summary = "Scalar tensor element read (GM)";
  let description = [{
    Read one GM scalar from ``!tla.tensor``. Lowered from a materialized tensor
    descriptor by ``tla-lower-scalar-access``; indices match logical rank 1 or 2.
  }];
  let arguments = (ins Tla_TensorType:$source, Variadic<Index>:$indices);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $source `[` $indices `]` attr-dict `:` type($source) `->` type($result)
  }];
}

def Tla_ScalarStoreOp : Tla_Op<"scalar_store", []> {
  let summary = "Scalar tensor element write (GM)";
  let description = [{
    Write one GM scalar to ``!tla.tensor``. Lowered from a materialized tensor
    descriptor by ``tla-lower-scalar-access``; indices match logical rank 1 or 2.
  }];
  let arguments = (ins Tla_TensorType:$dest, Variadic<Index>:$indices, AnyType:$value);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $dest `[` $indices `]` `,` $value attr-dict `:` type($dest) `,` type($value)
  }];
}

class Tla_VectorBinaryOp<string mnemonic, string opSummary, string opDescription>
    : Tla_Op<mnemonic, [Pure]> {
  let summary = opSummary;
  let description = opDescription # [{

    An optional mask controls which lanes are computed; masked-out lanes are
    undefined/zeroed.
  }];
  let arguments = (ins Tla_VectorSSAType:$lhs, Tla_VectorSSAType:$rhs,
                       Optional<Tla_MaskSSAType>:$mask);
  let results = (outs Tla_VectorSSAType:$result);
  let assemblyFormat = [{
    $lhs `,` $rhs (`mask` $mask^)? attr-dict `:`
    qualified(type($lhs)) `,` qualified(type($rhs)) (`mask` qualified(type($mask))^)? `->` qualified(type($result))
  }];
  let hasVerifier = 1;
}

def Tla_AddOp : Tla_VectorBinaryOp<"add",
    "Vector add on VectorSSA values",
    "Element-wise add on VectorSSA values.">;

def Tla_SubOp : Tla_VectorBinaryOp<"sub",
    "Vector subtract on VectorSSA values",
    "Element-wise subtract on VectorSSA values.">;

def Tla_MulOp : Tla_VectorBinaryOp<"mul",
    "Vector multiply on VectorSSA values",
    "Element-wise multiply on VectorSSA values.">;

def Tla_DivOp : Tla_VectorBinaryOp<"div",
    "Vector divide on VectorSSA values",
    "Element-wise divide on VectorSSA values.">;

def Tla_MaxOp : Tla_VectorBinaryOp<"max",
    "Vector maximum on VectorSSA values",
    "Element-wise maximum on VectorSSA values.">;

def Tla_MinOp : Tla_VectorBinaryOp<"min",
    "Vector minimum on VectorSSA values",
    "Element-wise minimum on VectorSSA values.">;

def Tla_GatherOp : Tla_Op<"gather", [Pure]> {
  let summary = "Vector gather on VectorSSA values";
  let description = [{
    Gathers elements from ``x`` to ``result``according to the indices in ``y``.
  }];
  let arguments = (ins Tla_TensorType:$x, Tla_VectorSSAType:$y,
                       Optional<Tla_MaskSSAType>:$mask);
  let results = (outs Tla_VectorSSAType:$result);
  let assemblyFormat = [{
    $x `,` $y (`mask` $mask^)? attr-dict `:`
    type($x) `,` qualified(type($y)) (`mask` qualified(type($mask))^)? `->` qualified(type($result))
  }];
  let hasVerifier = 1;
}

def Tla_SqueezeOp : Tla_Op<"squeeze", [Pure]> {
  let summary = "Mask-compress vector lanes into low indices";
  let description = [{
  Copy lanes of ``src`` selected by ``mask`` into ``result`` starting at lane 0.
  Remaining lanes in ``result`` are zeroed. This matches AscendC ``Squeeze``
  semantics (valid elements packed from low to high, not value-sorted).
  }];
  let arguments = (ins Tla_VectorSSAType:$src, Tla_MaskSSAType:$mask);
  let results = (outs Tla_VectorSSAType:$result);
  let assemblyFormat = [{
    $src `,` $mask attr-dict `:`
    qualified(type($src)) `,` qualified(type($mask)) `->` qualified(type($result))
  }];
  let hasVerifier = 1;
}

def Tla_CastOp : Tla_Op<"cast", [Pure]> {
  let summary = "Vector element-type conversion on VectorSSA values";
  let description = [{
    Convert a register-resident VectorSSA value from one element type to another
    (e.g. ``f32`` -> ``f16``, ``f32`` -> ``i32``). The ``trait`` attribute carries
    the cast knobs (rounding, saturation, register slot). An optional mask
    predicates which lanes are converted.
  }];
  let arguments = (ins Tla_VectorSSAType:$source, DenseI32ArrayAttr:$trait,
                       Optional<Tla_MaskSSAType>:$mask);
  let results = (outs Tla_VectorSSAType:$result);
  let assemblyFormat = [{
    $source `,` $trait (`mask` $mask^)? attr-dict `:`
    qualified(type($source)) (`mask` qualified(type($mask))^)? `->` qualified(type($result))
  }];
  let hasVerifier = 1;
}

def Tla_WhereOp : Tla_Op<"where", [Pure]> {
  let summary = "Vector select on VectorSSA values";
  let description = [{
    Element-wise select on VectorSSA values. For each lane the
    result holds the corresponding lane of ``x`` where the mask is
    active and ``y`` elsewhere. ``x`` and ``y`` must share the same VectorSSA type.
  }];
  let arguments = (ins Tla_MaskSSAType:$mask, Tla_VectorSSAType:$x, Tla_VectorSSAType:$y);
  let results = (outs Tla_VectorSSAType:$result);
  let assemblyFormat = [{
    $mask `,` $x `,` $y attr-dict `:`
    qualified(type($mask)) `,` qualified(type($x)) `,` qualified(type($y)) `->` qualified(type($result))
  }];
  let hasVerifier = 1;
}

class Tla_VectorScalarBinaryOp<string mnemonic, string opSummary, string opDescription>
    : Tla_Op<mnemonic, [Pure]> {
  let summary = opSummary;
  let description = opDescription # [{

    The right-hand side is a scalar SSA value. An optional mask controls which
    lanes are computed; masked-out lanes are undefined/zeroed.
  }];
  let arguments = (ins Tla_VectorSSAType:$lhs, Tla_ScalarValueType:$rhs,
                       Optional<Tla_MaskSSAType>:$mask);
  let results = (outs Tla_VectorSSAType:$result);
  let assemblyFormat = [{
    $lhs `,` $rhs (`mask` $mask^)? attr-dict `:`
    qualified(type($lhs)) `,` type($rhs) (`mask` qualified(type($mask))^)? `->` qualified(type($result))
  }];
  let hasVerifier = 1;
}

def Tla_AddsOp : Tla_VectorScalarBinaryOp<"adds",
    "Vector-scalar add on VectorSSA values",
    "Element-wise add of a scalar to VectorSSA values.">;

def Tla_SubsOp : Tla_VectorScalarBinaryOp<"subs",
    "Vector-scalar subtract on VectorSSA values",
    "Element-wise subtract of a scalar from VectorSSA values.">;

def Tla_MulsOp : Tla_VectorScalarBinaryOp<"muls",
    "Vector-scalar multiply on VectorSSA values",
    "Element-wise multiply of VectorSSA values by a scalar.">;

def Tla_MaxsOp : Tla_VectorScalarBinaryOp<"maxs",
    "Vector-scalar maximum on VectorSSA values",
    "Element-wise maximum of VectorSSA values and a scalar.">;

def Tla_MinsOp : Tla_VectorScalarBinaryOp<"mins",
    "Vector-scalar minimum on VectorSSA values",
    "Element-wise minimum of VectorSSA values and a scalar.">;

def Tla_DivsOp : Tla_VectorScalarBinaryOp<"divs",
    "Vector-scalar divide on VectorSSA values",
    "Element-wise divide of VectorSSA values by a scalar.">;

def Tla_ReduceOp : Tla_Op<"reduce", [Pure]> {
  let summary = "Vector reduction on VectorSSA values";
  let description = [{
    Reduce a register-resident VectorSSA value to a one-lane VectorSSA value.
  }];
  let arguments = (ins Tla_VectorSSAType:$operand, Optional<Tla_MaskSSAType>:$mask,
                       StrAttr:$kind);
  let results = (outs Tla_VectorSSAType:$result);
  let assemblyFormat = [{
    $operand (`mask` $mask^)? attr-dict `:`
    qualified(type($operand)) (`mask` qualified(type($mask))^)? `->` qualified(type($result))
  }];
  let hasVerifier = 1;
}

def Tla_InterleaveOp : Tla_Op<"interleave", [Pure]> {
  let summary = "Vector interleave on VectorSSA values";
  let arguments = (ins Tla_VectorSSAType:$src0, Tla_VectorSSAType:$src1);
  let results = (outs Tla_VectorSSAType:$dst0, Tla_VectorSSAType:$dst1);
  let assemblyFormat = [{
    $src0 `,` $src1 attr-dict `:`
    qualified(type($src0)) `,` qualified(type($src1)) `->` qualified(type($dst0)) `,` qualified(type($dst1))
  }];
  let hasVerifier = 1;
}

def Tla_DeInterleaveOp : Tla_Op<"deinterleave", [Pure]> {
  let summary = "Vector deinterleave on VectorSSA values";
  let arguments = (ins Tla_VectorSSAType:$src0, Tla_VectorSSAType:$src1);
  let results = (outs Tla_VectorSSAType:$dst0, Tla_VectorSSAType:$dst1);
  let assemblyFormat = [{
    $src0 `,` $src1 attr-dict `:`
    qualified(type($src0)) `,` qualified(type($src1)) `->` qualified(type($dst0)) `,` qualified(type($dst1))
  }];
  let hasVerifier = 1;
}

class Tla_VectorUnaryOp<string mnemonic, string opSummary, string opDescription>
    : Tla_Op<mnemonic, [Pure]> {
  let summary = opSummary;
  let description = opDescription # [{

    An optional mask controls which lanes are computed; masked-out lanes are
    undefined/zeroed.
  }];
  let arguments = (ins Tla_VectorSSAType:$operand,
                       Optional<Tla_MaskSSAType>:$mask);
  let results = (outs Tla_VectorSSAType:$result);
  let assemblyFormat = [{
    $operand (`mask` $mask^)? attr-dict `:`
    qualified(type($operand)) (`mask` qualified(type($mask))^)?
    `->` qualified(type($result))
  }];
  let hasVerifier = 1;
}

def Tla_ExpOp : Tla_VectorUnaryOp<"exp",
    "Vector exponential on VectorSSA values",
    "Element-wise exponential on VectorSSA values.">;

def Tla_LogOp : Tla_VectorUnaryOp<"log",
    "Vector natural logarithm on VectorSSA values",
    "Element-wise natural logarithm on VectorSSA values.">;

def Tla_SqrtOp : Tla_VectorUnaryOp<"sqrt",
    "Vector square root on VectorSSA values",
    "Element-wise square root on VectorSSA values.">;

def Tla_AbsOp : Tla_VectorUnaryOp<"abs",
    "Vector absolute value on VectorSSA values",
    "Element-wise absolute value on VectorSSA values.">;

def Tla_NegOp : Tla_VectorUnaryOp<"neg",
    "Vector negate on VectorSSA values",
    "Element-wise negate on VectorSSA values.">;

def Tla_VectorOp : Tla_Op<"vector", [NoTerminator, SingleBlock]> {
  let summary = "Vector region";
  let description = [{
    Launch a vector region for post-processing and epilogue ops.
  }];
  let regions = (region AnyRegion:$body);
}

def Tla_VecFuncOp : Tla_Op<"vec.func", [NoTerminator, SingleBlock]> {
  let summary = "Vector compute function region";
  let description = [{
    Group register-resident vector compute ops inside a vector region. Must be
    nested inside a ``tla.vector`` region.
  }];
  let regions = (region AnyRegion:$body);
  let hasVerifier = 1;
}

def Tla_WaitFlagOp : Tla_Op<"wait_flag", []> {
  let summary = "Wait flag";
  let description = [{
    Wait on a synchronization flag from one pipe to another.
  }];
  let arguments = (ins Tla_FlagType:$flag);
  let assemblyFormat = [{
    $flag `:` qualified(type($flag)) attr-dict
  }];
  let hasVerifier = 1;
}
