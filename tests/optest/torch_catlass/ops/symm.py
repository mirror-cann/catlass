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

import torch
from torch import Tensor

_DTYPE_MAP = {
    "float": torch.float32,
    "float32": torch.float32,
    "torch.float32": torch.float32,
}


def _normalize_dtype(dtype: str | torch.dtype) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    try:
        return _DTYPE_MAP[dtype.lower()]
    except KeyError as exc:
        raise ValueError(f"{dtype} is not supported by torch_catlass.symm") from exc


def symm(
    mat1: Tensor,
    mat2: Tensor,
    out_dtype: str | torch.dtype = torch.float32,
    side: int = 0,
    uplo: int = 0,
    alpha: float = 1.0,
) -> Tensor:
    """Run CATLASS SYMM on NPU tensors.

    Source: example 75_symm.

    Args:
        mat1: Left input. When ``side=0`` this is the symmetric matrix.
        mat2: Right input. When ``side=1`` this is the symmetric matrix.
        out_dtype: Output dtype. The current SYMM kernel supports ``float32``.
        side: ``0`` → ``mat1 @ mat2``; ``1`` → ``mat1 @ mat2`` (``mat2`` is symmetric).
        uplo: ``0`` selects the lower-triangle branch; ``1`` selects the upper-triangle branch.
            The symmetric operand is expected to contain a full symmetric matrix.
        alpha: Output scaling factor. The current SYMM kernel supports only ``1.0``.

    Returns:
        Output tensor with shape ``(M, N)`` on the active NPU device.
    """
    return torch.ops.catlass.symm(mat1, mat2, _normalize_dtype(out_dtype), side, uplo, alpha)
