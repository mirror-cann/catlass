from __future__ import annotations

from typing import Any

import pytest

import catlass as tla
import catlass.runtime as runtime_mod
from catlass.execution_lowering import UnsupportedExecutionLowering
from catlass.params import LoadDist, NormalLoadParams, PostMode, UnalignLoadParams


def _ub_tensor(
    dtype: type[tla.Numeric] = tla.Float32,
    extent: int = 64,
) -> tla.Tensor:
    with runtime_mod._eager_capture():
        shape = tla.make_shape(extent)
        return tla.Tensor(
            shape,
            dtype,
            addrspace=tla.AddressSpace.ub,
            origin_shape=shape,
            layout_tag=tla.arch.RowMajor,
        )


def _gm_tensor(
    dtype: type[tla.Numeric] = tla.Float32,
    extent: int = 64,
) -> tla.Tensor:
    with runtime_mod._eager_capture():
        shape = tla.make_shape(extent)
        return tla.Tensor(
            shape,
            dtype,
            addrspace=tla.AddressSpace.gm,
            origin_shape=shape,
            layout_tag=tla.arch.RowMajor,
        )


@tla.kernel
def load_unalign_kernel(src: tla.Tensor, dst: tla.Tensor) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            dst_tile.store(src_tile.load(UnalignLoadParams()))


@tla.kernel
def load_normal_kernel(src: tla.Tensor, dst: tla.Tensor) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            dst_tile.store(src_tile.load())


@tla.kernel
def load_unalign_loop_kernel(src: tla.Tensor, dst: tla.Tensor) -> None:
    with tla.vector():
        with tla.vec.func(mode="simd"):
            for i in tla.range(2):
                src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(i))
                dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(i))
                dst_tile.store(src_tile.load(UnalignLoadParams()))


@tla.kernel
def load_unalign_outside_vec_func(src: tla.Tensor) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        _ = src_tile.load(UnalignLoadParams())


@tla.kernel
def load_from_gm_kernel(src: tla.Tensor) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(1), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            _ = src_tile.load(NormalLoadParams(load_dist=LoadDist.DIST_BRC_B32))


def test_load_unalign_emits_tlair(compiler_tlair: Any) -> None:
    mlir = compiler_tlair(
        load_unalign_kernel,
        type_args=(_ub_tensor(), _ub_tensor()),
    )

    assert "tla.load" in mlir
    assert "unaligned_ub_access" in mlir
    assert "tla.load_unalign" not in mlir
    assert "load_unalign_pre" not in mlir
    assert "unalign_reg" not in mlir


def test_load_normal_has_no_unalign_attr(compiler_tlair: Any) -> None:
    mlir = compiler_tlair(
        load_normal_kernel,
        type_args=(_ub_tensor(), _ub_tensor()),
    )

    assert "tla.load" in mlir
    assert "unaligned_ub_access" not in mlir


def test_load_unalign_loop_emits_tlair(compiler_tlair: Any) -> None:
    mlir = compiler_tlair(
        load_unalign_loop_kernel,
        type_args=(_ub_tensor(extent=128), _ub_tensor(extent=128)),
    )

    assert "scf.for" in mlir
    assert "tla.load" in mlir
    assert "unaligned_ub_access" in mlir


@tla.kernel
def load_brc_b32_add_kernel(
    x: tla.Tensor,
    y: tla.Tensor,
    z: tla.Tensor,
) -> None:
    x_tile = tla.tile_view(x, tla.make_shape(1), tla.make_coord(0))
    y_tile = tla.tile_view(y, tla.make_shape(64), tla.make_coord(0))
    z_tile = tla.tile_view(z, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            x_v = x_tile.load(NormalLoadParams(load_dist=LoadDist.DIST_BRC_B32))
            y_v = y_tile.load()
            z_tile.store(x_v + y_v)


def test_load_brc_b32_add_emits_tlair(compiler_tlair: Any) -> None:
    mlir = compiler_tlair(
        load_brc_b32_add_kernel,
        type_args=(_ub_tensor(extent=64), _ub_tensor(), _ub_tensor()),
    )

    assert "#tla.load_dist<brc_b32>" in mlir
    assert "tla.add" in mlir
    assert "!tla.shape<64>" in mlir


def test_load_brc_b32_emits_tlair(compiler_tlair: Any) -> None:
    @tla.kernel
    def load_brc_b32_kernel(src: tla.Tensor, dst: tla.Tensor) -> None:
        src_tile = tla.tile_view(src, tla.make_shape(1), tla.make_coord(0))
        dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
        with tla.vector():
            with tla.vec.func(mode="simd"):
                dst_tile.store(
                    src_tile.load(
                        NormalLoadParams(load_dist=LoadDist.DIST_BRC_B32)
                    )
                )

    mlir = compiler_tlair(
        load_brc_b32_kernel,
        type_args=(_ub_tensor(extent=64), _ub_tensor()),
    )

    assert "#tla.load_dist<brc_b32>" in mlir
    assert "tla.load" in mlir


@tla.kernel
def load_dintlv_b32_kernel(
    src: tla.Tensor,
    dst0: tla.Tensor,
    dst1: tla.Tensor,
) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(128), tla.make_coord(0))
    dst0_tile = tla.tile_view(dst0, tla.make_shape(64), tla.make_coord(0))
    dst1_tile = tla.tile_view(dst1, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            v0, v1 = src_tile.load(
                NormalLoadParams(load_dist=LoadDist.DIST_DINTLV_B32)
            )
            dst0_tile.store(v0)
            dst1_tile.store(v1)


def test_load_dintlv_b32_emits_tlair(compiler_tlair: Any) -> None:
    mlir = compiler_tlair(
        load_dintlv_b32_kernel,
        type_args=(
            _ub_tensor(extent=128),
            _ub_tensor(),
            _ub_tensor(),
        ),
    )

    assert "#tla.load_dist<dintlv_b32>" in mlir
    load_lines = [line for line in mlir.splitlines() if "tla.load" in line]
    assert len(load_lines) == 1
    assert "->" in load_lines[0]
    # Dual-destination: two result types after ->
    assert load_lines[0].count("!") >= 3


@tla.kernel
def load_dintlv_b32_wrong_dtype_kernel(src: tla.Tensor, dst: tla.Tensor) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(256), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(128), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            v0, _v1 = src_tile.load(
                NormalLoadParams(load_dist=LoadDist.DIST_DINTLV_B32)
            )
            dst_tile.store(v0)


def test_load_dintlv_b32_rejects_non_f32(compiler_tlair: Any) -> None:
    with pytest.raises(Exception, match="f32|DIST_DINTLV_B32"):
        compiler_tlair(
            load_dintlv_b32_wrong_dtype_kernel,
            type_args=(
                _ub_tensor(dtype=tla.Float16, extent=256),
                _ub_tensor(dtype=tla.Float16, extent=128),
            ),
        )


@tla.kernel
def load_dintlv_b32_i32_kernel(
    src: tla.Tensor,
    dst0: tla.Tensor,
    dst1: tla.Tensor,
) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(128), tla.make_coord(0))
    dst0_tile = tla.tile_view(dst0, tla.make_shape(64), tla.make_coord(0))
    dst1_tile = tla.tile_view(dst1, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            v0, v1 = src_tile.load(
                NormalLoadParams(load_dist=LoadDist.DIST_DINTLV_B32)
            )
            dst0_tile.store(v0)
            dst1_tile.store(v1)


def test_load_dintlv_b32_rejects_i32(compiler_tlair: Any) -> None:
    with pytest.raises(Exception, match="f32|DIST_DINTLV_B32"):
        compiler_tlair(
            load_dintlv_b32_i32_kernel,
            type_args=(
                _ub_tensor(dtype=tla.Int32, extent=128),
                _ub_tensor(dtype=tla.Int32, extent=64),
                _ub_tensor(dtype=tla.Int32, extent=64),
            ),
        )


@tla.kernel
def load_post_mode_update_kernel(src: tla.Tensor, dst: tla.Tensor) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            dst_tile.store(
                src_tile.load(
                    NormalLoadParams(post_mode=PostMode.POST_MODE_UPDATE)
                )
            )


@tla.kernel
def load_post_update_stride_kernel(src: tla.Tensor, dst: tla.Tensor) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            dst_tile.store(
                src_tile.load(NormalLoadParams(post_update_stride=4))
            )


@tla.kernel
def load_unalign_is_pre_kernel(src: tla.Tensor, dst: tla.Tensor) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            dst_tile.store(src_tile.load(UnalignLoadParams(is_pre=True)))


@tla.kernel
def load_unalign_post_mode_update_kernel(src: tla.Tensor, dst: tla.Tensor) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            dst_tile.store(
                src_tile.load(
                    UnalignLoadParams(post_mode=PostMode.POST_MODE_UPDATE)
                )
            )


@tla.kernel
def load_unalign_post_update_stride_kernel(src: tla.Tensor, dst: tla.Tensor) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            dst_tile.store(
                src_tile.load(UnalignLoadParams(post_update_stride=8))
            )


def test_load_unalign_requires_vec_func() -> None:
    with pytest.raises(tla.TlaCoreAPIError, match="vec.func"):
        load_unalign_outside_vec_func.dump_mlir(type_args=(_ub_tensor(),))


def test_load_rejects_gm_tensor(compiler_tlair: Any) -> None:
    with pytest.raises(
        tla.TlaCoreAPIError,
        match="invalid argument 'source'.*expected addrspace ub, got gm",
    ):
        compiler_tlair(load_from_gm_kernel, type_args=(_gm_tensor(),))


@pytest.mark.parametrize(
    ("kernel", "match"),
    [
        (load_post_mode_update_kernel, "post_mode"),
        (load_post_update_stride_kernel, "post_update_stride"),
        (load_unalign_is_pre_kernel, "is_pre"),
        (load_unalign_post_mode_update_kernel, "post_mode"),
        (load_unalign_post_update_stride_kernel, "post_update_stride"),
    ],
)
def test_load_rejects_unsupported_params(
    compiler_tlair: Any, kernel: Any, match: str
) -> None:
    with pytest.raises(UnsupportedExecutionLowering, match=match):
        compiler_tlair(
            kernel,
            type_args=(_ub_tensor(), _ub_tensor()),
        )
