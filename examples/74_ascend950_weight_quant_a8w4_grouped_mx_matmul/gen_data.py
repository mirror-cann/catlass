#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import argparse
import math
import os
import sys
from typing import Dict, List, Tuple

import torch
import numpy as np

WORKSPACE = os.path.dirname(os.path.abspath(__file__))

_BLOCK_SIZE = 32
_EPSILON = 1e-12
_MIN_SCALE_EXP = -128
_MAX_SCALE_EXP = 127

_FP8_FORMATS = {
    "E4M3": {
        "torch_dtype": torch.float8_e4m3fn,
        "exp_bits": 4,
        "mantissa_bits": 3,
        "bias": 7,
        "emax": 8,
        "max_value": 448.0,
        "min_value": -448.0,
    },
    "E5M2": {
        "torch_dtype": torch.float8_e5m2,
        "exp_bits": 5,
        "mantissa_bits": 2,
        "bias": 15,
        "emax": 15,
        "max_value": 57344.0,
        "min_value": -57344.0,
    },
}

_FP4_FORMATS: Dict[str, Dict[str, float]] = {
    "E2M1": {
        "exp_bits": 2,
        "mantissa_bits": 1,
        "bias": 1,
        "emax": 2,
        "max_value": 6.0,
        "min_value": -6.0,
    },
    "E1M2": {
        "exp_bits": 1,
        "mantissa_bits": 2,
        "bias": 1,
        "emax": 0,
        "max_value": 1.75,
        "min_value": -1.75,
    },
}


GROUP_LIST_MODE = "group_list"
EXPECT_M_PER_GROUP_MODE = "expect_m_per_group"

def parse_group_m_list(arg: str) -> List[int]:
    """解析逗号分隔的 group M 列表"""
    values: List[int] = []
    for item in arg.split(","):
        item = item.strip()
        if not item:
            raise ValueError("group_m_list contains an empty item")
        value = int(item)
        if value < 0:
            raise ValueError("Each group M value must be greater than or equal to 0")
        values.append(value)
    if not values:
        raise ValueError("group_m_list must not be empty")
    return values


def build_random_group_m_list(group_num: int, expect_m_per_group: int, m: int) -> List[int]:
    """在 [0.7*expect, 1.3*expect] 范围内随机生成 group M 列表，保证总和不超过 m"""
    if group_num <= 0:
        raise ValueError("group_num must be greater than 0")
    if expect_m_per_group < 0:
        raise ValueError("expect_m_per_group must be greater than or equal to 0")
    if m < 0:
        raise ValueError("m must be greater than or equal to 0")

    low = int(math.floor(expect_m_per_group * 0.7))
    high = int(math.ceil(expect_m_per_group * 1.3))
    low = max(0, low)
    high = max(low, high)

    min_total_m = group_num * low
    if m < min_total_m:
        raise ValueError(
            f"m must be greater than or equal to group_num * floor(0.7 * expect_m_per_group)={min_total_m}"
        )

    if high == 0:
        return [0] * group_num

    rng = np.random.default_rng()
    for _ in range(200):
        group_m_arr = rng.integers(low, high + 1, size=group_num)
        if int(group_m_arr.sum()) <= m:
            return group_m_arr.astype(int).tolist()

    # Fallback: 从 low 开始分配剩余
    group_m_list = [low] * group_num
    remaining = m - sum(group_m_list)
    if remaining <= 0:
        return group_m_list

    capacities = [high - low for _ in range(group_num)]
    order = rng.permutation(group_num).tolist()
    while remaining > 0:
        progressed = False
        for idx in order:
            if capacities[idx] <= 0:
                continue
            group_m_list[idx] += 1
            capacities[idx] -= 1
            remaining -= 1
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            break
    return group_m_list


def parse_cli_args(argv: List[str]) -> Tuple[List[int], int, int, int, int]:
    """
    解析命令行参数，返回 (group_list, m, n, k, isNz)
    支持两种模式：
      1. group_list mode: gen_data.py group_list group_m_list m n k isNz
      2. expect_m_per_group mode: gen_data.py expect_m_per_group group_num expect_m_per_group m n k isNz
    """
    if len(argv) >= 6 and argv[1] == GROUP_LIST_MODE:
        group_m_list = parse_group_m_list(argv[2])
        m = int(argv[3])
        n = int(argv[4])
        k = int(argv[5])
        isNz = bool(int(argv[6]))
        if m < sum(group_m_list):
            raise ValueError(f"m must be greater than or equal to sum(group_m_list)={sum(group_m_list)}")

        return group_m_list, m, n, k, isNz

    if len(argv) >= 7 and argv[1] == EXPECT_M_PER_GROUP_MODE:
        group_num = int(argv[2])
        expect_m_per_group = int(argv[3])
        m = int(argv[4])
        n = int(argv[5])
        k = int(argv[6])
        isNz = bool(int(argv[7]))
        group_m_list = build_random_group_m_list(group_num, expect_m_per_group, m)

        return group_m_list, m, n, k, isNz

    raise ValueError(
        "Usage:\n"
        "  python gen_data.py group_list group_m_list m n k\n"
        "  python gen_data.py expect_m_per_group group_num expect_m_per_group m n k\n"
        "Example:\n"
        "  python gen_data.py group_list 64,128,96 400 256 128\n"
        "  python gen_data.py expect_m_per_group 4 100 400 256 128"
    )


def _build_e4m3_lut() -> torch.Tensor:
    bias = _FP8_FORMATS["E4M3"]["bias"]
    fp8_max = _FP8_FORMATS["E4M3"]["max_value"]
    values = []
    for i in range(256):
        if i < 128:
            sign, val = 1, i
        else:
            sign, val = -1, i - 128
        if val == 0:
            v = 0.0
        elif val == 127:
            v = sign * fp8_max
        else:
            exp = (val >> 3) & 0x0F
            mantissa = val & 0x07
            if exp == 0:
                v = (mantissa / 8.0) * (2.0 ** (1 - bias))
            else:
                v = (1.0 + mantissa / 8.0) * (2.0 ** (exp - bias))
            v *= sign
        v = max(min(v, fp8_max), -fp8_max)
        values.append(v)
    return torch.tensor(values, dtype=torch.float32)


def _build_e5m2_lut() -> torch.Tensor:
    bias = _FP8_FORMATS["E5M2"]["bias"]
    fp8_max = _FP8_FORMATS["E5M2"]["max_value"]
    values = []
    for i in range(256):
        if i < 128:
            sign, val = 1, i
        else:
            sign, val = -1, i - 128
        if val == 0:
            v = 0.0
        elif 124 <= val <= 127:
            v = sign * fp8_max
        else:
            exp = (val >> 2) & 0x1F
            mantissa = val & 0x03
            if exp == 0:
                v = (mantissa / 4.0) * (2.0 ** (1 - bias))
            else:
                v = (1.0 + mantissa / 4.0) * (2.0 ** (exp - bias))
            v *= sign
        v = max(min(v, fp8_max), -fp8_max)
        values.append(v)
    return torch.tensor(values, dtype=torch.float32)


_FP8_LUT_BUILDERS = {"E4M3": _build_e4m3_lut, "E5M2": _build_e5m2_lut}
_FP8_LUT_CACHE = {}
_FP8_LUT_POS_CACHE = {}


def _get_fp8_lut(format_name: str) -> torch.Tensor:
    if format_name not in _FP8_LUT_CACHE:
        _FP8_LUT_CACHE[format_name] = _FP8_LUT_BUILDERS[format_name]()
    return _FP8_LUT_CACHE[format_name]


def _get_fp8_lut_pos(format_name: str) -> torch.Tensor:
    if format_name not in _FP8_LUT_POS_CACHE:
        full = _get_fp8_lut(format_name)
        pos = full[:128].contiguous()
        diffs = pos[1:] - pos[:-1]
        if (diffs < 0).any():
            raise AssertionError(
                f"{format_name} positive LUT half is not non-decreasing")
        _FP8_LUT_POS_CACHE[format_name] = pos
    return _FP8_LUT_POS_CACHE[format_name]


def _build_fp4_lut(format_name: str) -> torch.Tensor:
    config = _FP4_FORMATS[format_name]
    exp_bits = int(config["exp_bits"])
    mantissa_bits = int(config["mantissa_bits"])
    bias = float(config["bias"])

    values = []
    for i in range(16):
        sign = (i >> 3) & 0x01
        exp = (i >> mantissa_bits) & ((1 << exp_bits) - 1)
        mantissa = i & ((1 << mantissa_bits) - 1)

        if exp == 0:
            if mantissa == 0:
                value = 0.0
            else:
                value = (mantissa / float(1 << mantissa_bits)) * (2.0 ** (1.0 - bias))
        else:
            value = (1.0 + mantissa / float(1 << mantissa_bits)) * (2.0 ** (float(exp) - bias))

        if sign == 1:
            value = -value
        values.append(value)

    return torch.tensor(values, dtype=torch.float32)


_FP4_LUT = {
    "E2M1": _build_fp4_lut("E2M1"),
    "E1M2": _build_fp4_lut("E1M2"),
}


def _e8m0_exp(max_abs: torch.Tensor, emax: int,
              epsilon: float = _EPSILON) -> torch.Tensor:
    assert max_abs.dtype == torch.float32, max_abs.dtype
    zero_mask = max_abs < epsilon
    safe = torch.where(zero_mask, torch.ones_like(max_abs), max_abs)
    bits = safe.contiguous().view(torch.int32)
    exp_bits = (bits >> 23) & 0xFF
    exp = exp_bits - 127 - emax
    exp = exp.clamp(_MIN_SCALE_EXP, _MAX_SCALE_EXP)
    return torch.where(zero_mask, torch.zeros_like(exp), exp)


def _vectorized_lut_quantize_fp8(scaled: torch.Tensor, format_name: str,
                                  fp8_dtype: torch.dtype) -> torch.Tensor:
    lut_pos = _get_fp8_lut_pos(format_name)
    last_idx = lut_pos.numel() - 1

    sign = torch.sign(scaled)
    mag = scaled.abs()

    upper_idx = torch.searchsorted(lut_pos, mag).clamp(max=last_idx)
    lower_idx = (upper_idx - 1).clamp(min=0)

    upper_val = lut_pos[upper_idx]
    lower_val = lut_pos[lower_idx]

    pick_lower = (mag - lower_val) <= (upper_val - mag)
    chosen_mag = torch.where(pick_lower, lower_val, upper_val)

    snapped_fp32 = sign * chosen_mag

    zero_mask = chosen_mag == 0
    snapped_fp32 = torch.where(
        zero_mask, torch.zeros_like(snapped_fp32), snapped_fp32)

    return snapped_fp32.to(fp8_dtype)


def _quantize_to_fp4_lut(values: torch.Tensor, format_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
    lut = _FP4_LUT[format_name].to(values.device)
    min_value = _FP4_FORMATS[format_name]["min_value"]
    max_value = _FP4_FORMATS[format_name]["max_value"]

    clamped = values.clamp(min_value, max_value)

    distances = (clamped.unsqueeze(-1) - lut).abs()
    indices = torch.argmin(distances, dim=-1)
    quantized = lut[indices]

    return quantized, indices.to(torch.uint8)


def _pack_fp4_nibbles(index_matrix: torch.Tensor) -> torch.Tensor:
    rows, cols = index_matrix.shape
    if cols % 2 != 0:
        index_matrix = torch.cat(
            [index_matrix, torch.zeros((rows, 1), dtype=torch.uint8, device=index_matrix.device)],
            dim=1,
        )

    low = index_matrix[:, 0::2]
    high = index_matrix[:, 1::2] << 4
    packed = low | high
    return packed.to(torch.uint8)


def _quantize_fp8_axis_last(matrix: torch.Tensor, format_name: str,
                             block_size: int = _BLOCK_SIZE
                             ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    M, N = matrix.shape
    fmt = _FP8_FORMATS[format_name]
    fp8_dtype = fmt["torch_dtype"]
    fp8_emax = fmt["emax"]
    fp8_max = fmt["max_value"]

    num_blocks = (N + block_size - 1) // block_size
    padded_n = num_blocks * block_size
    if padded_n != N:
        padded = torch.zeros(M, padded_n, dtype=matrix.dtype)
        padded[:, :N] = matrix
    else:
        padded = matrix

    blocks = padded.view(M, num_blocks, block_size)

    max_abs = blocks.abs().amax(dim=-1)
    exp = _e8m0_exp(max_abs, fp8_emax)
    scale = torch.exp2(exp.to(torch.float32))

    scaled = blocks / scale.unsqueeze(-1)
    scaled_clamped = scaled.clamp(-fp8_max, fp8_max)

    quant_fp8 = _vectorized_lut_quantize_fp8(scaled_clamped, format_name, fp8_dtype)

    dequant = quant_fp8.to(torch.float32) * scale.unsqueeze(-1)

    if padded_n != N:
        quant_fp8 = quant_fp8.reshape(M, padded_n)[:, :N].contiguous()
        dequant = dequant.reshape(M, padded_n)[:, :N].contiguous()
    else:
        quant_fp8 = quant_fp8.reshape(M, N)
        dequant = dequant.reshape(M, N)

    padded_blocks = ((num_blocks + 1) // 2) * 2
    if padded_blocks != num_blocks:
        scale_padded = torch.ones((M, padded_blocks), dtype=torch.float32)
        scale_padded[:, :num_blocks] = scale
        scale = scale_padded

    return quant_fp8, scale, dequant


def _quantize_fp8_axis_first(matrix: torch.Tensor, format_name: str,
                              block_size: int = _BLOCK_SIZE
                              ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    qt, st, dt = _quantize_fp8_axis_last(
        matrix.t().contiguous(), format_name, block_size)
    return (qt.t().contiguous(),
            st.t().contiguous(),
            dt.t().contiguous())


def _quantize_fp8(matrix: torch.Tensor, format_name: str, axis: int,
                   block_size: int = _BLOCK_SIZE
                   ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if axis == 0:
        return _quantize_fp8_axis_first(matrix, format_name, block_size)
    if axis == 1:
        return _quantize_fp8_axis_last(matrix, format_name, block_size)
    raise ValueError(f"axis must be 0 or 1, got {axis}")


def _quantize_fp4_axis_last(matrix: torch.Tensor, format_name: str,
                             block_size: int = _BLOCK_SIZE
                             ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m, n = matrix.shape
    padded_n = ((n + block_size - 1) // block_size) * block_size
    num_blocks = padded_n // block_size

    if padded_n != n:
        padded = torch.zeros((m, padded_n), dtype=matrix.dtype, device=matrix.device)
        padded[:, :n] = matrix
    else:
        padded = matrix

    blocks = padded.view(m, num_blocks, block_size)
    max_abs = blocks.abs().amax(dim=-1)

    exp = torch.floor(torch.log2(torch.clamp(max_abs, min=_EPSILON))) - _FP4_FORMATS[format_name]["emax"]
    exp = torch.where(max_abs < _EPSILON, torch.zeros_like(exp), exp)
    exp = exp.clamp(_MIN_SCALE_EXP, _MAX_SCALE_EXP)
    scale = torch.pow(torch.tensor(2.0, dtype=torch.float32, device=matrix.device), exp)

    scaled = blocks / scale.unsqueeze(-1)
    quantized_blocks, _ = _quantize_to_fp4_lut(scaled, format_name)
    dequant_blocks = quantized_blocks * scale.unsqueeze(-1)

    quantized = quantized_blocks.reshape(m, padded_n)
    dequantized = dequant_blocks.reshape(m, padded_n)
    if padded_n != n:
        quantized = quantized[:, :n].contiguous()
        dequantized = dequantized[:, :n].contiguous()

    padded_blocks = ((num_blocks + 1) // 2) * 2
    if padded_blocks != num_blocks:
        scale_padded = torch.ones((m, padded_blocks), dtype=torch.float32, device=matrix.device)
        scale_padded[:, :num_blocks] = scale
        scale = scale_padded

    return quantized, scale, dequantized


def _quantize_fp4_axis_first(matrix: torch.Tensor, format_name: str,
                              block_size: int = _BLOCK_SIZE
                              ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    quantized_t, scale_t, dequantized_t = _quantize_fp4_axis_last(
        matrix.t().contiguous(), format_name, block_size)
    return quantized_t.t().contiguous(), scale_t.t().contiguous(), dequantized_t.t().contiguous()


def _quantize_fp4(matrix: torch.Tensor, format_name: str, axis: int,
                   block_size: int = _BLOCK_SIZE
                   ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if axis == 0:
        return _quantize_fp4_axis_first(matrix, format_name, block_size)
    if axis == 1:
        return _quantize_fp4_axis_last(matrix, format_name, block_size)
    raise ValueError(f"axis must be 0 or 1, got {axis}")


def gen_data_fp8_e4m3(row, col, axis):

    matrix = torch.randn((row, col), dtype=torch.float32) * 10
    quant_fp8, scale_fp32, dequant_fp32 = _quantize_fp8(matrix, "E4M3", axis)
    return (quant_fp8.to(torch.float8_e4m3fn),
            scale_fp32.to(torch.float8_e8m0fnu),
            dequant_fp32)


def gen_data_fp8_e5m2(row, col, axis):
    matrix = torch.randn((row, col), dtype=torch.float32)
    quant_fp8, scale_fp32, dequant_fp32 = _quantize_fp8(matrix, "E5M2", axis)
    return (quant_fp8.to(torch.float8_e5m2),
            scale_fp32.to(torch.float8_e8m0fnu),
            dequant_fp32)


def gen_data_fp4_e2m1(row, col, axis, trans):
    matrix = torch.randn((row, col), dtype=torch.float32)
    quantized_matrix, scale_matrix, dequantized_matrix = _quantize_fp4(matrix, "E2M1", axis)

    if trans == 1:
        quantized_matrix = quantized_matrix.t().contiguous()

    _, fp4_indices = _quantize_to_fp4_lut(quantized_matrix, "E2M1")
    quantized_matrix_uint8 = _pack_fp4_nibbles(fp4_indices)

    return quantized_matrix_uint8, scale_matrix.to(torch.float8_e8m0fnu), dequantized_matrix

def trans_nd2nz(input_data):
    g, n_pad, k_pad = input_data.shape
    return input_data.reshape(g, n_pad // 16, 16, k_pad // 32, 32).permute(0, 3, 1, 2, 4)

def gen_data(group_m_list, m, n, k, isNz, trans_a = 0, trans_b = 1) -> None:
    data_dir = os.path.join(WORKSPACE, "data")
    input_dir = os.path.join(data_dir, "input")
    golden_dir = os.path.join(data_dir, "golden")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(golden_dir, exist_ok=True)

    a_fp8, a_scale, a_fp32 = gen_data_fp8_e4m3(m, k, 1)

    a_scale = a_scale.reshape(a_scale.shape[0], a_scale.shape[1] // 2, 2)

    if trans_a == 1:
        a_scale = a_scale.permute(1, 0, 2)

    g = len(group_m_list)
    b_indices_list = []      # 存储未打包的 FP4 索引矩阵 (k, n)，元素为 0~15
    b_scale_list = []
    b_fp32_list = []
    b_packed = None
    if isNz:
        for _ in range(g):
            # 生成随机 FP32 矩阵
            matrix = torch.randn((k, n), dtype=torch.float32)
            # 量化：axis=0（按行量化），返回量化后的浮点值、scale、反量化结果
            quantized_vals, scale, dequantized = _quantize_fp4(matrix, "E2M1", axis=0)

            if trans_b == 1:
                quantized_vals = quantized_vals.t().contiguous()

            # 从浮点量化值获取 FP4 索引（整数）
            _, fp4_indices = _quantize_to_fp4_lut(quantized_vals, "E2M1")
            b_indices_list.append(fp4_indices)        # 形状 (k, n)，整数类型
            b_scale_list.append(scale.to(torch.float8_e8m0fnu))
            b_fp32_list.append(dequantized)

        b_indices_stacked = torch.stack(b_indices_list, dim=0)   # (g, k, n)，整数

        pad_k = (32 - k % 32) % 32
        pad_n = (16 - n % 16) % 16
        if pad_k > 0 or pad_n > 0:
            b_indices_stacked = torch.nn.functional.pad(b_indices_stacked, (0, pad_k, 0, pad_n), "constant", 0)

        b_quant_nd = b_indices_stacked.reshape(g, n + pad_n, k + pad_k)    # (g, n_pad, k_pad)

        b_quant_nz = trans_nd2nz(b_quant_nd)             # (g, k_blk, n_blk, 16, 32)
        b_quant_flat = b_quant_nz.reshape(-1, 32)        # (total_rows, 32) 整数
        b_packed = _pack_fp4_nibbles(b_quant_flat)       # 打包为 nibble
    else:
        for _ in range(g):
            b_fp4, b_scale, b_fp32 = gen_data_fp4_e2m1(k, n, 0, trans_b)
            b_indices_list.append(b_fp4)
            b_scale_list.append(b_scale)
            b_fp32_list.append(b_fp32)

        b_packed = torch.stack(b_indices_list, dim=0)  # (g, k, n) 或 (g, n, k)

    b_scale_stacked = torch.stack(b_scale_list, dim=0)
    b_fp32_stacked = torch.stack(b_fp32_list, dim=0)

    b_scale_processed_list = []
    for i in range(g):
        scale = b_scale_stacked[i]
        scale = scale.reshape(scale.shape[0] // 2, 2, scale.shape[1])
        if trans_b == 1:
            scale = scale.permute(2, 0, 1)
        else:
            scale = scale.permute(0, 2, 1)
        b_scale_processed_list.append(scale)
    b_scale_processed = torch.stack(b_scale_processed_list, dim=0)

    a_np = torch.tensor(a_fp8.flatten().untyped_storage(), dtype=torch.int8).numpy()
    b_np = torch.tensor(b_packed.flatten().untyped_storage(), dtype=torch.int8).numpy()
    a_np.tofile(os.path.join(input_dir, "a_8.bin"))
    b_np.tofile(os.path.join(input_dir, "b_4.bin"))

    a_scale_np = torch.tensor(a_scale.flatten().untyped_storage(), dtype=torch.int8).numpy()
    b_scale_np = torch.tensor(b_scale_processed.flatten().untyped_storage(), dtype=torch.int8).numpy()
    a_scale_np.tofile(os.path.join(input_dir, "a_scale.bin"))
    b_scale_np.tofile(os.path.join(input_dir, "b_scale.bin"))

    group_list = np.array(group_m_list, dtype=np.int64)
    group_list.tofile(os.path.join(input_dir, "group_list.bin"))

    c_fp32_list = []
    m_offset = 0
    for i, group_m in enumerate(group_list):
        if group_m == 0:
            continue
        end_m = m_offset + group_m
        a_group = a_fp32[m_offset:end_m]           # (group_m, k)
        b_group = b_fp32_stacked[i]                # (k, n) 或 (n, k)
        c_group = a_group @ b_group                # (group_m, n)
        c_fp32_list.append(c_group)
        m_offset = end_m

    c_fp32 = torch.cat(c_fp32_list, dim=0)         # (m, n)

    c_np = c_fp32.numpy()

    c_np.tofile(os.path.join(golden_dir, "expected_data.bin"))


if __name__ == "__main__":
    try:
        group_list, m, n, k, isNz = parse_cli_args(sys.argv)
    except ValueError as error:
        print(error)
        sys.exit(1)

    gen_data(group_list, m, n, k, isNz)
