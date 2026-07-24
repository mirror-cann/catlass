"""Tla tensor IR values and view helpers."""

from __future__ import annotations

from typing import Any

from mlir import ir as mlir_ir  # type: ignore[assignment]

from .. import _tla_type_bridge
from .._mlir_bindings import tla_ops_gen as _tla_ops_gen
from ..base_dsl.op import dsl_user_op
from .. import runtime as _runtime
from ..base_dsl.typing import Bool, Numeric
from .typing import Tensor as TensorABC


def _dynamic_metadata_values(type_tree: Any, metadata_tree: Any) -> list[Any]:
    """Return runtime values corresponding to dynamic leaves in a type tree."""
    if isinstance(type_tree, tuple):
        values: list[Any] = []
        for type_child, metadata_child in zip(type_tree, metadata_tree, strict=True):
            values.extend(_dynamic_metadata_values(type_child, metadata_child))
        return values
    return [metadata_tree] if type_tree is None else []


class _Tensor(TensorABC):
    """Frontend proxy for an SSA ``!tla.tensor`` value."""

    def __init__(self, value: mlir_ir.Value) -> None:
        if not isinstance(value, mlir_ir.Value):
            raise TypeError(
                f"Tensor value expects mlir.ir.Value, got {type(value).__name__}"
            )
        if not _tla_type_bridge.type_is_tensor(value.type):
            raise TypeError(f"Tensor value expects !tla.tensor<...>, got {value.type}")
        self.value = value
        self.__tla_category__ = "tensor"
        _runtime._bind_frontend_value(self, value)
        _runtime._bind_frontend_category(self, "tensor")
        _runtime._bind_frontend_category(value, "tensor")

    def __tla_type__(self) -> str:
        return str(self.value.type)

    def __get_mlir_types__(self, context: mlir_ir.Context | None = None) -> list[Any]:
        del context
        return [self.value.type]

    def __extract_mlir_values__(self) -> list[Any]:
        """Flatten this tensor and the SSA leaves of its dynamic metadata."""
        from ..core_api import _tla_tensor_type_for_mlir_value

        tensor_type = _tla_tensor_type_for_mlir_value(self.value)
        values: list[Any] = [self.value]
        for field, type_tree in (
            ("shape", tensor_type.shape),
            ("stride", tensor_type.stride),
            ("coord", tensor_type.coord),
            ("origin_shape", tensor_type.origin_shape),
        ):
            values.extend(
                _dynamic_metadata_values(
                    type_tree, _tensor_metadata_field(self.value, field)
                )
            )
        return values

    def __new_from_mlir_values__(self, values: list[Any]) -> "_Tensor":
        """Rebuild an SCF-carried tensor and register its selected metadata."""
        from ..core_api import (
            _metadata_from_type_tree,
            _register_tla_tensor_metadata,
            _register_tla_tensor_type,
            _tla_tensor_type_for_mlir_value,
        )

        expected_count = len(self.__extract_mlir_values__())
        if len(values) != expected_count:
            raise ValueError(
                f"Tensor expects {expected_count} MLIR values, got {len(values)}"
            )
        result = _Tensor(values[0])
        tensor_type = _tla_tensor_type_for_mlir_value(self.value)
        _register_tla_tensor_type(result.value, tensor_type)

        from ..base_dsl.typing import as_numeric

        dynamic_values = iter(as_numeric(value) for value in values[1:])
        metadata = {
            "shape": _metadata_from_type_tree(tensor_type.shape, dynamic_values),
            "stride": _metadata_from_type_tree(tensor_type.stride, dynamic_values),
            "coord": _metadata_from_type_tree(tensor_type.coord, dynamic_values),
            "origin_shape": _metadata_from_type_tree(
                tensor_type.origin_shape, dynamic_values
            ),
            "dtype": tensor_type.element_type,
            "addrspace": tensor_type.addrspace,
            "layout_tag": tensor_type.layout_tag,
        }
        _register_tla_tensor_metadata(result.value, metadata)
        return result

    @property
    def shape(self) -> Any:
        return _tensor_metadata_field(self.value, "shape")

    @property
    def stride(self) -> Any:
        return _tensor_metadata_field(self.value, "stride")

    @property
    def coord(self) -> Any:
        return _tensor_metadata_field(self.value, "coord")

    @property
    def origin_shape(self) -> Any:
        return _tensor_metadata_field(self.value, "origin_shape")

    @property
    def dtype(self) -> str:
        return str(_tensor_metadata_field(self.value, "dtype"))

    @property
    def element_type(self) -> type[Numeric]:
        from ..core_api import _tla_tensor_descriptor_from_type_or_value

        parent = _tla_tensor_descriptor_from_type_or_value(self.value)
        return Numeric.from_mlir_type(parent.element_mlir_type())

    @property
    def addrspace(self) -> str:
        return str(_tensor_metadata_field(self.value, "addrspace"))

    @property
    def layout_tag(self) -> str:
        return str(_tensor_metadata_field(self.value, "layout_tag"))

    @property
    def ptr(self) -> Any:
        """Return the backing ``!tla.ptr`` of this tensor.

        The result is a :class:`~catlass.core_api._Pointer` and supports element-count
        offset arithmetic via ``+`` (e.g. ``a.ptr + 16`` advances by 16 elements), which
        can be fed to :func:`tla.make_tensor` to construct a tensor at an offset address.
        """
        from ..base_dsl.op import _capture_user_loc
        from ..core_api import _as_value, _emit_tensor_ptr

        loc = (
            _capture_user_loc()
            if _runtime._current_frontend_state() is not None
            else None
        )
        return _emit_tensor_ptr(_as_value(self), loc)

    @dsl_user_op
    def load(
        self,
        params: Any | None = None,
        *,
        loc: mlir_ir.Location | None = None,
    ) -> Any:
        """Load this tensor tile into vector SSA inside a tla.vec.func region.

        Single-destination modes (``DIST_NORM``, ``DIST_BRC_B32``, unalign) return
        one ``VectorSSA``. Dual-destination ``DIST_DINTLV_B32`` returns a
        ``(VectorSSA, VectorSSA)`` pair (even/odd b32 elements; f32 only).
        """
        from ..core_api import (
            VectorSSA,
            _as_value,
            _coerce_type,
            _full_vector_ssa_descriptor,
            _op_error,
            _require_frontend_state,
            _tla_tensor_type_for_mlir_value,
            _vector_ssa_type_from_tensor_descriptor,
        )
        from ..execution_lowering import TlaLoweringError
        from ..params import LoadDist, NormalLoadParams, PostMode, UnalignLoadParams

        loc = _normalize_user_loc(loc)
        _require_frontend_state("load")
        _runtime._require_enclosing_region("load", "vec.func")
        if params is None:
            params = NormalLoadParams()
        elif not isinstance(params, (NormalLoadParams, UnalignLoadParams)):
            raise TlaLoweringError(
                "load params must be NormalLoadParams or UnalignLoadParams, "
                f"got {type(params).__name__}"
            )

        if params.post_mode != PostMode.POST_MODE_NORMAL:
            raise NotImplementedError(
                f"currently unsupported post_mode {params.post_mode!r}"
            )
        if params.post_update_stride != 0:
            raise NotImplementedError(
                f"currently unsupported post_update_stride {params.post_update_stride}"
            )
        if isinstance(params, UnalignLoadParams) and params.is_pre:
            raise NotImplementedError(
                f"currently unsupported is_pre {params.is_pre}"
            )

        is_dintlv = (
            isinstance(params, NormalLoadParams)
            and params.load_dist == LoadDist.DIST_DINTLV_B32
        )

        load_kwargs: dict[str, Any] = {"loc": loc}
        if isinstance(params, UnalignLoadParams):
            load_kwargs["unaligned_ub_access"] = True
        elif isinstance(params, NormalLoadParams) and params.load_dist != LoadDist.DIST_NORM:
            ctx = loc.context if loc is not None else mlir_ir.Context.current
            load_kwargs["load_dist"] = mlir_ir.Attribute.parse(
                f"#tla.load_dist<{params.load_dist}>",
                context=ctx,
            )

        source = _as_value(self)
        source_desc = _tla_tensor_type_for_mlir_value(source)
        if source_desc.addrspace.lower() != "ub":
            _op_error(
                "load",
                "invalid argument 'source' (position 0): expected addrspace ub, "
                f"got {source_desc.addrspace}",
            )
        if is_dintlv:
            # AscendNPU-IR lowers DINTLV_B32 only to vldsx2.v64f32; reject
            # i32/u32 (and other dtypes) at the frontend until IR dispatches.
            elem = str(source_desc.element_type).strip().lower()
            if elem != "f32":
                raise TlaLoweringError(
                    "DIST_DINTLV_B32 currently requires f32 element type "
                    f"(got {source_desc.element_type})"
                )
            # Dual-destination load writes two full VL registers (even/odd).
            # Source tile is typically 2*VL elements; do not derive result
            # VectorSSA lanes from the source origin_shape.
            result_desc = _full_vector_ssa_descriptor(source_desc.element_type)
        elif (
            isinstance(params, NormalLoadParams)
            and params.load_dist == LoadDist.DIST_BRC_B32
        ):
            result_desc = _full_vector_ssa_descriptor(source_desc.element_type)
        else:
            result_desc = _vector_ssa_type_from_tensor_descriptor(source_desc)

        result_type = _coerce_type(result_desc)
        if is_dintlv:
            results = _tla_ops_gen.load(
                result_type, result_type, source, **load_kwargs
            )
            return tuple(VectorSSA(result) for result in results)

        result = _tla_ops_gen.load(result_type, None, source, **load_kwargs)
        return VectorSSA(result)

    @dsl_user_op
    def store(
        self,
        value: Any,
        params: Any | None = None,
        *,
        mask: Any | None = None,
        loc: mlir_ir.Location | None = None,
    ) -> None:
        """Store a vector SSA value into this tensor tile inside a tla.vec.func region.

        An optional ``params`` controls store mode: use ``NormalStoreParams()``
        (the default) for aligned store, or ``UnalignStoreParams()`` for unaligned
        UB access. An optional ``mask`` (a ``MaskSSA`` from ``tla.create_mask`` or
        ``tla.update_mask``) controls which lanes are written; masked-out lanes
        are left untouched. Only a ``MaskSSA`` is accepted (validated below); a
        ``mask`` here is typed ``Any`` to avoid a circular import of ``MaskSSA``.
        """
        from ..core_api import (
            _as_value,
            _op_error,
            _require_category,
            _require_frontend_state,
            _require_mask_matches_vector,
            _tla_tensor_type_for_mlir_value,
        )
        from ..execution_lowering import TlaLoweringError
        from ..params import NormalStoreParams, StoreParams, UnalignStoreParams

        loc = _normalize_user_loc(loc)
        _require_category("store", "value", value, "vector_ssa", 1)
        if mask is not None:
            _require_category("store", "mask", mask, "mask_ssa", 2)
        if params is None:
            params = NormalStoreParams()
        elif not isinstance(params, (NormalStoreParams, UnalignStoreParams)):
            raise TlaLoweringError(
                "store params must be NormalStoreParams or UnalignStoreParams, "
                f"got {type(params).__name__}"
            )
        _require_frontend_state("store")
        _runtime._require_enclosing_region("store", "vec.func")
        dest = _as_value(self)
        dest_desc = _tla_tensor_type_for_mlir_value(dest)
        if dest_desc.addrspace.lower() != "ub":
            _op_error(
                "store",
                "invalid argument 'dest' (position 0): expected addrspace ub, "
                f"got {dest_desc.addrspace}",
            )
        value_val = _as_value(value)
        mask_val = _as_value(mask) if mask is not None else None
        if mask_val is not None:
            _require_mask_matches_vector("store", mask_val, value_val)
        store_kwargs: dict[str, Any] = {"loc": loc}
        if isinstance(params, UnalignStoreParams):
            store_kwargs["unaligned_ub_access"] = True
        _tla_ops_gen.store(dest, value_val, mask=mask_val, **store_kwargs)

    def _check_can_scalar_load_store(self) -> None:
        """Phase-1 ``scalar_load``/``scalar_store`` preconditions (GM only; not ``tla.load``/``tla.store``)."""
        if self.addrspace not in ("gm",):
            raise ValueError(f"{self!r} doesn't support scalar_load/store")
        if self.layout_tag not in ("row_major", "column_major"):
            raise ValueError(
                f"{self!r} doesn't support scalar_load/store (layout={self.layout_tag!r})"
            )


    def _check_can_dereference(self) -> None:
        sub_byte_types = (Bool,)
        if self.element_type.width % 8 != 0 and self.element_type not in sub_byte_types:
            raise ValueError(
                f"Sub-byte scalar dereference not supported for type {self.element_type.__name__}"
            )

    def _cvt_to_dest(
        self,
        data: Numeric,
        dest_element_type: mlir_ir.Type,
        *,
        loc: mlir_ir.Location | None = None,
    ) -> mlir_ir.Value:
        """Require exact element type match, then ``ir_value`` / host constant."""
        from ..core_api import _op_error, _scalar_constant_for_element_type

        if not isinstance(data, Numeric):
            raise TypeError(f"expected Numeric, got {type(data).__name__}")

        dest_cls = Numeric.from_mlir_type(dest_element_type)
        src_cls = type(data)

        # No silent upcast/promote: store value dtype must match the tensor element
        # type. Callers should convert explicitly with ``.to(...)`` / ``cast``.
        if src_cls is not dest_cls:
            _op_error(
                "scalar_store",
                f"type mismatch, store {src_cls.dtype} "
                f"to Tensor with element type {dest_cls.dtype}; "
                f"cast explicitly with .to({dest_cls.__name__}) before store",
            )

        # Host literals: emit dest-typed constant (range / fraction checks).
        if isinstance(data.value, (bool, int, float)):
            return _scalar_constant_for_element_type(
                "scalar_store", data.value, dest_element_type, loc=loc
            )
        return data.ir_value(loc=loc)

    @dsl_user_op
    def __getitem__(
        self,
        crd: Any,
        *,
        loc: mlir_ir.Location | None = None,
    ) -> Any:
        """Access tensor elements at scalar coordinates."""
        from ..core_api import (
            _as_index_value,
            _as_value,
            _flatten_tla_tuple,
            _op_error,
            _require_category,
            _require_frontend_state,
            _tla_tensor_descriptor_from_type_or_value,
        )
        from ..execution_lowering import TlaLoweringError

        loc = _normalize_user_loc(loc)
        if crd is None or (type(crd) is tuple and any(part is None for part in crd)):
            raise TlaLoweringError(
                "tensor indexing does not support None/underscore coordinates; "
                "use scalar indices only"
            )

        _require_category("scalar_load", "source", self, "tensor", 0)
        _require_frontend_state("scalar_load")
        if type(crd) is tuple and not crd:
            _op_error("scalar_load", "expected at least one index")

        source_value = _as_value(self)
        parent = _tla_tensor_descriptor_from_type_or_value(source_value)
        if isinstance(self, _Tensor):
            self._check_can_scalar_load_store()
            self._check_can_dereference()
        else:
            if parent.addrspace not in ("gm",):
                raise ValueError("tensor doesn't support scalar_load")
            elem_numeric = Numeric.from_mlir_type(parent.element_mlir_type())
            sub_byte_types = (Bool,)
            if elem_numeric.width % 8 != 0 and elem_numeric not in sub_byte_types:
                raise ValueError(
                    "Sub-byte scalar dereference not supported for type "
                    f"{elem_numeric.__name__}"
                )
        if parent.layout_tag not in ("row_major", "column_major"):
            raise TlaLoweringError(
                "tla.scalar_load currently supports row_major/column_major only"
            )

        flat_shape = _flatten_tla_tuple(parent.shape)
        index_values = [
            _as_index_value(part) for part in (crd if type(crd) is tuple else (crd,))
        ]
        if len(index_values) != len(flat_shape) or len(flat_shape) not in (1, 2):
            raise TlaLoweringError(
                "tla.scalar_load index rank must match tensor logical rank "
                f"(shape rank {len(flat_shape)}, index rank {len(index_values)})"
            )

        elem_type = parent.element_mlir_type()
        result = _tla_ops_gen.scalar_load(
            elem_type,
            source_value,
            index_values,
            loc=loc,
        )
        if str(result.type) != str(elem_type):
            raise TlaLoweringError(
                f"tla.scalar_load result type mismatch: expected {elem_type}, got {result.type}"
            )
        # Bool / i1 loads: Bool Numeric (``if tensor[i]`` via coerce).
        if mlir_ir.IntegerType.isinstance(elem_type):
            int_ty = mlir_ir.IntegerType(elem_type)
            if int_ty.width == 1:
                return Bool(result)
        return Numeric.from_mlir_type(elem_type)(result)

    @dsl_user_op
    def __setitem__(
        self,
        crd: Any,
        data: Any,
        *,
        loc: mlir_ir.Location | None = None,
    ) -> None:
        """Set tensor elements at scalar coordinates.

        ``data`` may be a ``Numeric`` (e.g. from ``tensor[j]``), a bare Python
        ``int``/``float`` literal (converted to the tensor element type), or an
        ``mlir.ir.Value`` that ``as_numeric`` can wrap (``__setitem__``).
        """
        from ..base_dsl.typing import as_numeric
        from ..core_api import (
            _as_index_value,
            _as_value,
            _flatten_tla_tuple,
            _op_error,
            _require_category,
            _require_frontend_state,
            _resolve_bound_value,
            _scalar_constant_for_element_type,
            _tla_tensor_descriptor_from_type_or_value,
            _type_name,
        )
        from ..execution_lowering import TlaLoweringError

        loc = _normalize_user_loc(loc)
        if crd is None or (type(crd) is tuple and any(part is None for part in crd)):
            raise TlaLoweringError(
                "tensor indexing does not support None/underscore coordinates; "
                "use scalar indices only"
            )

        _require_category("scalar_store", "dest", self, "tensor", 0)
        _require_frontend_state("scalar_store")
        if type(crd) is tuple and not crd:
            _op_error("scalar_store", "expected at least one index")

        dest_value = _as_value(self)
        parent = _tla_tensor_descriptor_from_type_or_value(dest_value)
        if isinstance(self, _Tensor):
            self._check_can_scalar_load_store()
            self._check_can_dereference()
        else:
            if parent.addrspace not in ("gm",):
                raise ValueError("tensor doesn't support scalar_store")
        if parent.layout_tag not in ("row_major", "column_major"):
            raise TlaLoweringError(
                "tla.scalar_store currently supports row_major/column_major only"
            )

        flat_shape = _flatten_tla_tuple(parent.shape)
        index_values = [
            _as_index_value(part) for part in (crd if type(crd) is tuple else (crd,))
        ]
        if len(index_values) != len(flat_shape) or len(flat_shape) not in (1, 2):
            raise TlaLoweringError(
                "tla.scalar_store index rank must match tensor logical rank "
                f"(shape rank {len(flat_shape)}, index rank {len(index_values)})"
            )

        elem_type = parent.element_mlir_type()
        # Canonicalize to Numeric, then _cvt_to_dest → ir_value.
        # Bare Python numbers stay dest-typed constants (range / fraction checks).
        resolved = _resolve_bound_value(data)
        if isinstance(resolved, (int, float)) and not isinstance(resolved, bool):
            store_value = _scalar_constant_for_element_type(
                "scalar_store", resolved, elem_type, loc=loc
            )
        else:
            try:
                if isinstance(resolved, Numeric):
                    num = resolved
                elif isinstance(resolved, mlir_ir.Value):
                    num = as_numeric(resolved)
                elif isinstance(data, Numeric):
                    num = data
                elif isinstance(data, mlir_ir.Value):
                    num = as_numeric(data)
                else:
                    raise TypeError(type(data).__name__)
            except (TypeError, ValueError, KeyError):
                _op_error(
                    "scalar_store",
                    f"invalid argument 'value' (position 1): expected Numeric or "
                    f"scalar literal, got {_type_name(data)}",
                )
            store_value = _Tensor._cvt_to_dest(self, num, elem_type, loc=loc)
        if str(store_value.type) != str(elem_type):
            raise TlaLoweringError(
                f"tla.scalar_store value type mismatch: expected {elem_type}, got {store_value.type}"
            )
        _tla_ops_gen.scalar_store(dest_value, index_values, store_value, loc=loc)

def _normalize_user_loc(loc: mlir_ir.Location | None) -> mlir_ir.Location | None:
    if loc is None and _runtime._current_frontend_state() is not None:
        from ..core_api import _capture_user_loc

        return _capture_user_loc()
    if loc is not None and not isinstance(loc, mlir_ir.Location):
        raise TypeError(f"loc must be mlir.ir.Location or None, got {type(loc).__name__}")
    return loc


def _tensor_metadata_field(value: mlir_ir.Value, field: str) -> Any:
    from ..core_api import _tensor_metadata_field as _lookup

    return _lookup(value, field)


def _scale_coord_leaf(coord: Any, shape: Any) -> Any:
    if isinstance(coord, int):
        return coord * shape
    from ..base_dsl.typing import as_numeric

    resolved = _runtime._resolve_frontend_bound_value(coord)
    if (
        resolved is not coord
        or _runtime._resolve_frontend_bound_category(coord) == "index"
    ):
        return as_numeric(coord) * shape
    return coord * shape


def scale_tile_coord_by_shape(coord_tree: Any, shape_tree: Any) -> Any:
    """Convert tile coordinates into element offsets using the tile shape."""

    if isinstance(coord_tree, tuple) and isinstance(shape_tree, tuple):
        if len(coord_tree) != len(shape_tree):
            raise ValueError(
                "tile-view coord/shape trees must have matching tuple profiles"
            )
        return tuple(
            scale_tile_coord_by_shape(coord_part, shape_part)
            for coord_part, shape_part in zip(coord_tree, shape_tree)
        )
    if isinstance(coord_tree, tuple) or isinstance(shape_tree, tuple):
        raise ValueError(
            "tile-view coord/shape trees must have matching tuple profiles"
        )
    return _scale_coord_leaf(coord_tree, shape_tree)


def normalize_tile_view_coord(
    *,
    shape_components: tuple[Any, ...],
    coord_components: tuple[Any, ...],
) -> tuple[Any, ...]:
    """Convert ``tla.tile_view`` tile coordinates into element offsets."""

    if len(coord_components) != len(shape_components):
        raise ValueError("tile-view coord/shape ranks must match")
    return tuple(
        scale_tile_coord_by_shape(coord_part, shape_part)
        for coord_part, shape_part in zip(coord_components, shape_components)
    )


__all__ = [
    "_Tensor",
    "normalize_tile_view_coord",
    "scale_tile_coord_by_shape",
]
