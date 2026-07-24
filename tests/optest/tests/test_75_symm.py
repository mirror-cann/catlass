# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance
# with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS
# OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
import pytest
import torch
import torch_npu

pytestmark = pytest.mark.skipif(
    torch_npu.npu.device_count() <= 0,
    reason="torch-catlass integration tests require an available Ascend NPU",
)


def _make_symm_input(size, dtype, uplo):
    # Match the standalone example: build the full symmetric operand from the
    # triangle selected by uplo before passing it to the kernel.
    stored = torch.randn(size, size, dtype=dtype)
    tri = torch.tril(stored) if uplo == 0 else torch.triu(stored)
    full = tri + tri.T - torch.diag(torch.diag(tri))
    return full, full


@pytest.mark.parametrize(
    "side,uplo,m,n",
    [
        (0, 0, 128, 96),
        (0, 1, 128, 64),
        (1, 0, 96, 128),
        (1, 1, 64, 128),
    ],
)
def test_symm(side, uplo, m, n):
    import torch_catlass

    dtype = torch.float32

    if side == 0:
        sym_input_cpu, sym_full_cpu = _make_symm_input(m, dtype, uplo)
        dense_cpu = torch.randn(m, n, dtype=dtype)
        sym = sym_input_cpu.to("npu")
        dense = dense_cpu.to("npu")
        result = torch_catlass.symm(sym, dense, dtype, side, uplo)
        expected = torch.matmul(sym_full_cpu, dense_cpu)
    else:
        dense_cpu = torch.randn(m, n, dtype=dtype)
        sym_input_cpu, sym_full_cpu = _make_symm_input(n, dtype, uplo)
        dense = dense_cpu.to("npu")
        sym = sym_input_cpu.to("npu")
        result = torch_catlass.symm(dense, sym, dtype, side, uplo)
        expected = torch.matmul(dense_cpu, sym_full_cpu)

    assert result.shape == (m, n)
    assert result.dtype == dtype
    assert result.device.type == "npu"
    result_cpu = result.cpu()
    assert torch.allclose(result_cpu, expected, rtol=1e-2, atol=1e-2), (
        f"Results not close: max diff = {(result_cpu - expected).abs().max().item()}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
