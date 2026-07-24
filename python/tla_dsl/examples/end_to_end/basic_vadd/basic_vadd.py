from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import catlass as tla

VECTOR_ELE = 400
VL_ELE = 64
LOOPS = (VECTOR_ELE + VL_ELE - 1) // VL_ELE
ALL_DTYPES = ("i8", "i16", "i32", "f16", "f32")

DEMO_DIR = Path(__file__).resolve().parent
DEFAULT_CACHE_DIR = DEMO_DIR / "artifacts" / "runtime-cache"
_KERNEL_DTYPE = tla.Float32
_KERNEL_ELEMENT_BYTES = 4


@tla.kernel
def basic_vadd(mem_x: tla.Tensor, mem_y: tla.Tensor, mem_z: tla.Tensor) -> None:
    ub_loaded = tla.flag("ub_loaded", tla.arch.MTE2, tla.arch.VECTOR)
    vec_done = tla.flag("vec_done", tla.arch.VECTOR, tla.arch.MTE3)

    x_gm = tla.tile_view(mem_x, tla.make_shape(VECTOR_ELE), tla.make_coord(0))
    y_gm = tla.tile_view(mem_y, tla.make_shape(VECTOR_ELE), tla.make_coord(0))
    z_gm = tla.tile_view(mem_z, tla.make_shape(VECTOR_ELE), tla.make_coord(0))

    x_ub_ptr = tla.allocate(VECTOR_ELE, _KERNEL_DTYPE, tla.AddressSpace.ub, 256)
    y_ub_ptr = tla.allocate(VECTOR_ELE, _KERNEL_DTYPE, tla.AddressSpace.ub, 256)
    z_ub_ptr = tla.allocate(VECTOR_ELE, _KERNEL_DTYPE, tla.AddressSpace.ub, 256)

    x_ub = tla.make_tensor_like(x_ub_ptr, x_gm, tla.arch.RowMajor)
    y_ub = tla.make_tensor_like(y_ub_ptr, y_gm, tla.arch.RowMajor)
    z_ub = tla.make_tensor_like(z_ub_ptr, z_gm, tla.arch.RowMajor)

    with tla.vector():
        tla.copy(x_ub, x_gm)
        tla.copy(y_ub, y_gm)

        tla.set_flag(ub_loaded)
        tla.wait_flag(ub_loaded)
        with tla.vec.func(mode="simd"):
            for i in tla.range(LOOPS):
                x_tile = tla.tile_view(
                    x_ub, tla.make_shape(VL_ELE), tla.make_coord(i)
                )
                y_tile = tla.tile_view(
                    y_ub, tla.make_shape(VL_ELE), tla.make_coord(i)
                )
                z_tile = tla.tile_view(
                    z_ub, tla.make_shape(VL_ELE), tla.make_coord(i)
                )

                x_reg = x_tile.load()
                y_reg = y_tile.load()
                z_reg = tla.add(x_reg, y_reg)
                z_tile.store(z_reg)

        tla.set_flag(vec_done)
        tla.wait_flag(vec_done)

        tla.copy(z_gm, z_ub)
        tla.pipe_barrier(tla.pipes.ALL)


@tla.kernel
def basic_vadd_mutex(mem_x: tla.Tensor, mem_y: tla.Tensor, mem_z: tla.Tensor) -> None:
    mutex_x_ub = tla.mutex(resource="x_ub", id=0)
    mutex_y_ub = tla.mutex(resource="y_ub", id=1)
    mutex_z_ub = tla.mutex(resource="z_ub", id=2)

    x_gm = tla.tile_view(mem_x, tla.make_shape(VECTOR_ELE), tla.make_coord(0))
    y_gm = tla.tile_view(mem_y, tla.make_shape(VECTOR_ELE), tla.make_coord(0))
    z_gm = tla.tile_view(mem_z, tla.make_shape(VECTOR_ELE), tla.make_coord(0))

    x_ub_ptr = tla.allocate(VECTOR_ELE, _KERNEL_DTYPE, tla.AddressSpace.ub, 256)
    y_ub_ptr = tla.allocate(VECTOR_ELE, _KERNEL_DTYPE, tla.AddressSpace.ub, 256)
    z_ub_ptr = tla.allocate(VECTOR_ELE, _KERNEL_DTYPE, tla.AddressSpace.ub, 256)

    x_ub = tla.make_tensor_like(x_ub_ptr, x_gm, tla.arch.RowMajor)
    y_ub = tla.make_tensor_like(y_ub_ptr, y_gm, tla.arch.RowMajor)
    z_ub = tla.make_tensor_like(z_ub_ptr, z_gm, tla.arch.RowMajor)

    with tla.vector():
        mutex_x_ub.lock(pipe=tla.arch.MTE2)
        tla.copy(x_ub, x_gm)
        mutex_x_ub.unlock(pipe=tla.arch.MTE2)

        mutex_y_ub.lock(pipe=tla.arch.MTE2)
        tla.copy(y_ub, y_gm)
        mutex_y_ub.unlock(pipe=tla.arch.MTE2)

        mutex_x_ub.lock(pipe=tla.arch.VECTOR)
        mutex_y_ub.lock(pipe=tla.arch.VECTOR)
        mutex_z_ub.lock(pipe=tla.arch.VECTOR)
        with tla.vec.func(mode="simd"):
            for i in tla.range(LOOPS):
                x_tile = tla.tile_view(
                    x_ub, tla.make_shape(VL_ELE), tla.make_coord(i)
                )
                y_tile = tla.tile_view(
                    y_ub, tla.make_shape(VL_ELE), tla.make_coord(i)
                )
                z_tile = tla.tile_view(
                    z_ub, tla.make_shape(VL_ELE), tla.make_coord(i)
                )

                x_reg = x_tile.load()
                y_reg = y_tile.load()
                z_reg = tla.add(x_reg, y_reg)
                z_tile.store(z_reg)
        mutex_z_ub.unlock(pipe=tla.arch.VECTOR)
        mutex_y_ub.unlock(pipe=tla.arch.VECTOR)
        mutex_x_ub.unlock(pipe=tla.arch.VECTOR)

        mutex_z_ub.lock(pipe=tla.arch.MTE3)
        tla.copy(z_gm, z_ub)
        mutex_z_ub.unlock(pipe=tla.arch.MTE3)
        tla.pipe_barrier(tla.pipes.ALL)


@tla.kernel
def basic_vadd_mutex_with(mem_x: tla.Tensor, mem_y: tla.Tensor, mem_z: tla.Tensor) -> None:
    mutex_x_ub = tla.mutex(resource="x_ub", id=0)
    mutex_y_ub = tla.mutex(resource="y_ub", id=1)
    mutex_z_ub = tla.mutex(resource="z_ub", id=2)

    x_gm = tla.tile_view(mem_x, tla.make_shape(VECTOR_ELE), tla.make_coord(0))
    y_gm = tla.tile_view(mem_y, tla.make_shape(VECTOR_ELE), tla.make_coord(0))
    z_gm = tla.tile_view(mem_z, tla.make_shape(VECTOR_ELE), tla.make_coord(0))

    x_ub_ptr = tla.allocate(VECTOR_ELE, _KERNEL_DTYPE, tla.AddressSpace.ub, 256)
    y_ub_ptr = tla.allocate(VECTOR_ELE, _KERNEL_DTYPE, tla.AddressSpace.ub, 256)
    z_ub_ptr = tla.allocate(VECTOR_ELE, _KERNEL_DTYPE, tla.AddressSpace.ub, 256)

    x_ub = tla.make_tensor_like(x_ub_ptr, x_gm, tla.arch.RowMajor)
    y_ub = tla.make_tensor_like(y_ub_ptr, y_gm, tla.arch.RowMajor)
    z_ub = tla.make_tensor_like(z_ub_ptr, z_gm, tla.arch.RowMajor)

    with tla.vector():
        with tla.mutex_guard(mutex_x_ub):
            tla.copy(x_ub, x_gm)

        with tla.mutex_guard(mutex_y_ub):
            tla.copy(y_ub, y_gm)

        with tla.mutex_guard(mutex_x_ub, mutex_y_ub, mutex_z_ub):
            with tla.vec.func(mode="simd"):
                for i in tla.range(LOOPS):
                    x_tile = tla.tile_view(
                        x_ub, tla.make_shape(VL_ELE), tla.make_coord(i)
                    )
                    y_tile = tla.tile_view(
                        y_ub, tla.make_shape(VL_ELE), tla.make_coord(i)
                    )
                    z_tile = tla.tile_view(
                        z_ub, tla.make_shape(VL_ELE), tla.make_coord(i)
                    )

                    x_reg = x_tile.load()
                    y_reg = y_tile.load()
                    z_reg = tla.add(x_reg, y_reg)
                    z_tile.store(z_reg)

        with tla.mutex_guard(mutex_z_ub):
            tla.copy(z_gm, z_ub)
        tla.pipe_barrier(tla.pipes.ALL)

@tla.kernel
def basic_vadd_atomic_add(mem_x: tla.Tensor, mem_y: tla.Tensor, mem_z: tla.Tensor) -> None:
    """This demo use one AIV to compute Z = X + Y, which is done by setting the atomic add operation."""
    ub_loaded = tla.flag("ub_loaded", tla.arch.MTE2, tla.arch.MTE3)

    x_gm = tla.tile_view(mem_x, tla.make_shape(VECTOR_ELE), tla.make_coord(0))
    y_gm = tla.tile_view(mem_y, tla.make_shape(VECTOR_ELE), tla.make_coord(0))
    z_gm = tla.make_tensor(mem_z.ptr, tla.make_layout(shape=tla.make_shape(VECTOR_ELE),stride=tla.make_stride(1)))

    x_ub_ptr = tla.allocate(VECTOR_ELE, _KERNEL_DTYPE, tla.AddressSpace.ub, 256)
    y_ub_ptr = tla.allocate(VECTOR_ELE, _KERNEL_DTYPE, tla.AddressSpace.ub, 256)

    x_ub = tla.make_tensor_like(x_ub_ptr, x_gm, tla.arch.RowMajor)
    y_ub = tla.make_tensor_like(y_ub_ptr, y_gm, tla.arch.RowMajor)

    with tla.vector():
        # To avoid possible race condition since every 
        # launched block sees the same GM tiles,
        # Restrict this work to only one block.
        if tla.arch.block_idx() == 0:
            tla.copy(x_ub, x_gm)
            tla.copy(y_ub, y_gm)

            tla.set_flag(ub_loaded)
            tla.wait_flag(ub_loaded)

            # Z = X (plain copy, overwrite z on GM)
            tla.copy(z_gm, x_ub)
            tla.pipe_barrier(tla.pipes.MTE3)

            tla.copy(z_gm, y_ub, tla.params.CopyUbToGmParams(atomic_mode=tla.params.AtomicMode.ADD))
            tla.pipe_barrier(tla.pipes.MTE3)
        tla.pipe_barrier(tla.pipes.ALL)


def _dtype_config(dtype_name: str) -> tuple[type[Any], Any, float | int, int, int]:
    try:
        import torch
    except ImportError:
        torch = None

    if dtype_name == "f32":
        return tla.Float32, torch.float32 if torch is not None else None, -7.0, 64, 4
    if dtype_name == "f16":
        return tla.Float16, torch.float16 if torch is not None else None, -7.0, 128, 2
    if dtype_name == "i16":
        return tla.Int16, torch.int16 if torch is not None else None, -7, 128, 2
    if dtype_name == "i32":
        return tla.Int32, torch.int32 if torch is not None else None, -7, 64, 4
    if dtype_name == "i8":
        return tla.Int8, torch.int8 if torch is not None else None, -101, 256, 1
    raise SystemExit(
        f"unsupported dtype={dtype_name!r}; expected one of: f32, f16, i8, i16, i32"
    )


def _set_kernel_dtype(dtype_name: str) -> tuple[type[Any], Any, float | int]:
    global VL_ELE, LOOPS, _KERNEL_DTYPE, _KERNEL_ELEMENT_BYTES
    tla_dtype, torch_dtype, default_sentinel, vl_ele, element_bytes = _dtype_config(dtype_name)
    VL_ELE = vl_ele
    LOOPS = (VECTOR_ELE + VL_ELE - 1) // VL_ELE
    _KERNEL_DTYPE = tla_dtype
    _KERNEL_ELEMENT_BYTES = element_bytes
    return tla_dtype, torch_dtype, default_sentinel


def _compile_only_type_args(dtype_name: str = "f32") -> tuple[Any, Any, Any]:
    from catlass import runtime as runtime_mod

    tla_dtype, _, _ = _set_kernel_dtype(dtype_name)
    with runtime_mod._eager_capture():
        return (
            tla.Tensor(
                tla.make_shape(VECTOR_ELE),
                tla_dtype,
                origin_shape=tla.make_shape(VECTOR_ELE),
                coord=tla.make_coord(0),
                stride=tla.make_stride(1),
                layout_tag=tla.arch.RowMajor,
            ),
            tla.Tensor(
                tla.make_shape(VECTOR_ELE),
                tla_dtype,
                origin_shape=tla.make_shape(VECTOR_ELE),
                coord=tla.make_coord(0),
                stride=tla.make_stride(1),
                layout_tag=tla.arch.RowMajor,
            ),
            tla.Tensor(
                tla.make_shape(VECTOR_ELE),
                tla_dtype,
                origin_shape=tla.make_shape(VECTOR_ELE),
                coord=tla.make_coord(0),
                stride=tla.make_stride(1),
                layout_tag=tla.arch.RowMajor,
            ),
        )


def _runtime_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "arch_scope": "aiv.c310",
        "cache": not args.no_cache,
        "cache_dir": str(Path(args.cache_dir).expanduser().resolve()),
        "force_recompile": args.force_recompile,
    }


def _select_kernel(args: argparse.Namespace) -> Any:
    if getattr(args, "use_mutex", False):
        return basic_vadd_mutex
    if getattr(args, "use_mutex_with", False):
        return basic_vadd_mutex_with
    if getattr(args, "use_atomic_add", False):
        return basic_vadd_atomic_add
    return basic_vadd


def dump_tlair(args: argparse.Namespace) -> str:
    return _select_kernel(args).dump_mlir(type_args=_compile_only_type_args(args.dtype))


def build_only(args: argparse.Namespace) -> int:
    kernel = _select_kernel(args)
    artifact = tla.compile(
        kernel,
        *_compile_only_type_args(args.dtype),
        mlir_print_ir_after_all=True,
        **_runtime_kwargs(args),
    )
    print("compile_ok=True")
    print(f"kernel.o path={artifact.kernel_binary_path}")
    return 0


def _require_torch_npu(device_id: int) -> Any:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("basic_vadd --run requires PyTorch.") from exc
    try:
        import torch_npu  # noqa: F401
    except ImportError as exc:
        raise SystemExit("basic_vadd --run requires torch_npu.") from exc
    torch.npu.set_device(device_id)
    return torch


def _create_tla_tensor(dev_buf: Any, tla_dtype: type[Any]) -> Any:
    from catlass import runtime as runtime_mod

    contiguous = dev_buf.contiguous()
    with runtime_mod._eager_capture():
        tensor = tla.Tensor(
            tla.make_shape(VECTOR_ELE),
            tla_dtype,
            origin_shape=tla.make_shape(VECTOR_ELE),
            coord=tla.make_coord(0),
            stride=tla.make_stride(1),
            data_ptr=int(contiguous.data_ptr()),
        )
    tensor._external_binding = True
    return tensor


def _run_single_case(args: argparse.Namespace, dtype_name: str, torch: Any) -> int:
    tla_dtype, torch_dtype, default_sentinel = _set_kernel_dtype(dtype_name)
    device = "npu"
    sentinel = args.sentinel if args.sentinel is not None else default_sentinel
    if dtype_name in {"i8", "i16"}:
        arange = torch.arange(VECTOR_ELE, dtype=torch.int32, device=device)
        if dtype_name == "i8":
            x = ((arange % 50) - 25).to(torch_dtype)
            y = ((arange % 30) - 15).to(torch_dtype)
        else:
            x = arange.to(torch_dtype)
            y = ((arange * 2) - 3).to(torch_dtype)
    else:
        x = torch.arange(VECTOR_ELE, dtype=torch_dtype, device=device)
        y = (torch.arange(VECTOR_ELE, dtype=torch_dtype, device=device) * 2) - 3
    z = torch.full((VECTOR_ELE,), sentinel, dtype=torch_dtype, device=device)
    expected = x + y

    tla_x = _create_tla_tensor(x, tla_dtype)
    tla_y = _create_tla_tensor(y, tla_dtype)
    tla_z = _create_tla_tensor(z, tla_dtype)

    kernel = _select_kernel(args)
    artifact = tla.compile(
        kernel,
        tla_x,
        tla_y,
        tla_z,
        **_runtime_kwargs(args),
    )
    block = max(1, args.block if args.block != -1 else tla.get_aicore_num(args.device))
    artifact(tla_x, tla_y, tla_z, block=block)

    torch.npu.synchronize()
    if dtype_name in {"f32", "f16"}:
        unchanged = torch.isclose(
            z, torch.full_like(z, sentinel), rtol=0.0, atol=args.atol
        )
        expected_match = torch.isclose(z, expected, rtol=0.0, atol=args.atol)
    else:
        unchanged = z.eq(torch.full_like(z, sentinel))
        expected_match = z.eq(expected)
    mismatch = expected_match.logical_not().nonzero(as_tuple=False)
    first_mismatch: dict[str, Any] | None = None
    if mismatch.numel():
        index = int(mismatch[0].item())
        first_mismatch = {
            "index": index,
            "actual": z[index].item(),
            "expected": expected[index].item(),
        }

    print(f"compile_ok=True host=torch_npu dtype={dtype_name} layout=row")
    print(f"kernel.o path={artifact.kernel_binary_path}")
    print("launch_ok=True")
    print(f"Z unchanged? {bool(unchanged.all())}")
    print(f"Z equals expected add? {bool(expected_match.all())}")
    print(f"Z changed count={int((~unchanged).sum().item())}")
    print(f"first mismatch={first_mismatch}")
    return 0 if first_mismatch is None else 1


def _dtype_cases(args: argparse.Namespace) -> tuple[str, ...]:
    if args.all_dtypes:
        return ALL_DTYPES
    return (args.dtype,)


def run(args: argparse.Namespace) -> int:
    tla.initialize(device=args.device)
    try:
        torch = _require_torch_npu(args.device)
        failed = 0
        for dtype_name in _dtype_cases(args):
            print("---", f"dtype={dtype_name}", "---")
            failed += _run_single_case(args, dtype_name, torch)
        return 0 if failed == 0 else 1
    finally:
        tla.finalize()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compile and run a vector add.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--build-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--device", type=int, default=2)
    parser.add_argument("--block", type=int, default=-1)
    parser.add_argument("--dtype", choices=("f32", "f16", "i8", "i16", "i32"), default="f32")
    parser.add_argument(
        "--all-dtypes",
        action="store_true",
        help="Run all supported vector add dtypes sequentially: i8, i16, i32, f16, f32.",
    )
    parser.add_argument("--sentinel", type=float, default=None)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--force-recompile", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    sync = parser.add_mutually_exclusive_group()
    sync.add_argument(
        "--use-mutex", action="store_true", help="Use explicit mutex lock/unlock sync."
    )
    sync.add_argument(
        "--use-mutex-with",
        action="store_true",
        help="Use tla.mutex_guard with-syntax sync.",
    )
    sync.add_argument(
        "--use-atomic-add",
        action="store_true",
        help="Use a block-0 plain X store followed by an atomic Y add.",
    )
    parser.add_argument("--dump-tlair", action="store_true")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if args.dump_tlair:
        if args.all_dtypes:
            raise SystemExit("--dump-tlair requires a single dtype.")
        print(dump_tlair(args))
        return 0
    if args.build_only:
        if args.all_dtypes:
            raise SystemExit("--build-only requires a single dtype.")
        return build_only(args)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
