#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import os
import re
import itertools
import subprocess
import unittest
from typing import List, Type

import acl

CMAKE_BINARY_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "output", "bin"
)
CMAKE_EXAMPLES_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "examples"
)
os.environ["LD_LIBRARY_PATH"] = os.path.join(
    CMAKE_BINARY_PATH, "..", "shared_lib:"
) + os.environ.get("LD_LIBRARY_PATH", "")


def get_npu_arch():
    device_name = acl.get_soc_name()

    if re.match(r"Ascend910B.+", device_name, re.I) or re.search(
        r"Ascend910_93", device_name, re.I
    ):
        return 2201
    elif re.search("Ascend950(PR|DT)", device_name, re.I):
        return 3510
    else:
        raise ValueError(f"Unsupported device name: {device_name}")


def only_on_npu_arch(npu_arch: int):
    return unittest.skipIf(
        npu_arch != get_npu_arch(), f"This case only runs on {npu_arch}"
    )


only_on_2201 = only_on_npu_arch(2201)
only_on_3510 = only_on_npu_arch(3510)


class CatlassExampleTest(unittest.TestCase):
    def run_case(self, executable_name: str, args: List):
        args = [str(arg) for arg in args]

        ret = subprocess.run(
            [os.path.join(CMAKE_BINARY_PATH, executable_name)] + args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._ret_check(ret)

    def _ret_check(self, ret: subprocess.CompletedProcess):
        for error_log_line in ret.stderr.decode().splitlines():
            acl_match = re.match(r"^.*aclError:.*([1-9][0-9]{5})", error_log_line)
            rt_match = re.match(r"^.*rtError:.*([1-9][0-9]{5})", error_log_line)
            compare_match = re.match(
                r"^.*Compare failed. Error count: :.*([1-9][0-9]{5})", error_log_line
            )
            acl_code = 0 if acl_match is None else int(acl_match.group(1))
            rt_code = 0 if rt_match is None else int(rt_match.group(1))
            compare_code = 0 if compare_match is None else int(compare_match.group(1))
            self.assertEqual(acl_code, 0, f"There is an ACL error: {acl_code}")
            self.assertEqual(rt_code, 0, f"There is an RT error: {rt_code}")
            self.assertEqual(
                compare_code, 0, f"There is a compare error: {compare_code}"
            )
        self.assertEqual(
            ret.returncode, 0, f"Return code is not zero: {ret.returncode}"
        )

    @only_on_2201
    def test_19_mla(self):
        batch = 1
        q_seqlen_list = "1"
        kv_seqlen_list = "128"
        num_heads = 16
        num_blocks = 16
        block_size = 128
        dtype = "half"

        # python gen_data.py batch "qSeqlen_list" "kvSeqlen_list" numHeads numBlocks blockSize dtype
        case_py = [str(batch), q_seqlen_list, kv_seqlen_list,
                   str(num_heads), str(num_blocks), str(block_size), dtype]
        ret = subprocess.run(
            ["python", os.path.join(CMAKE_EXAMPLES_PATH, "19_mla", "gen_data.py")]
            + case_py
        )
        # ./19_mla batch "qSeqlen_list" "kvSeqlen_list" numHeads numBlocks blockSize --dtype half --datapath ...
        case_cpp = [str(batch), q_seqlen_list, kv_seqlen_list,
                    str(num_heads), str(num_blocks), str(block_size),
                    "--dtype", dtype,
                    "--datapath", os.path.join(CMAKE_EXAMPLES_PATH, "19_mla", "data")]
        self.run_case("19_mla", case_cpp)

    @only_on_2201
    def test_24_conv_bias(self):
        case_base = [
            str(i) for i in [32, 64, 1, 32, 48, 128, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
        ]
        case_py = case_base + ["float16"]
        ret = subprocess.run(
            ["python", os.path.join(CMAKE_EXAMPLES_PATH, "24_conv_bias", "gen_data.py")]
            + case_py
        )
        case_cpp = [
            str(i)
            for i in [32, 1, 4, 32, 48, 16, 128, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
        ]
        self.run_case("24_conv_bias", case_cpp)

    @only_on_2201
    def test_29_a2_fp8_e4m3_matmul(self):
        case_py = [str(i) for i in [256, 512, 1024, 0, 0]]
        ret = subprocess.run(
            [
                "python",
                os.path.join(
                    CMAKE_EXAMPLES_PATH, "29_a2_fp8_e4m3_matmul", "gen_data.py"
                ),
            ]
            + case_py
        )
        case_cpp = [str(i) for i in [256, 512, 1024, 0]]
        self.run_case("29_a2_fp8_e4m3_matmul", case_cpp)

    @only_on_2201
    def test_32_w4a8_matmul(self):
        case_py = [str(i) for i in [860, 5712, 4535, 0]]
        ret = subprocess.run(
            [
                "python",
                os.path.join(CMAKE_EXAMPLES_PATH, "32_w4a8_matmul", "gen_data.py"),
            ]
            + case_py
        )
        case_cpp = [str(i) for i in [860, 5712, 4535, 0]]
        self.run_case("32_w4a8_matmul", case_cpp)

    @only_on_2201
    def test_38_w4a4_matmul(self):
        case_py = [str(i) for i in [96, 4096, 1280]]  # (M, N, K)
        ret = subprocess.run(
            [
                "python",
                os.path.join(
                    CMAKE_EXAMPLES_PATH,
                    "38_w4a4_matmul_per_token_per_channel_dequant",
                    "gen_data.py",
                ),
            ]
            + case_py
        )
        case_cpp = case_py
        self.run_case("38_w4a4_matmul_per_token_per_channel_dequant", case_cpp)

    @only_on_2201
    def test_41_sparse_matmul_tla(self):
        case_py = [str(i) for i in [160, 320, 64]]  # (M, N, K)
        ret = subprocess.run(
            [
                "python",
                os.path.join(
                    CMAKE_EXAMPLES_PATH, "41_sparse_matmul_tla", "sparse_gen_data.py"
                ),
            ]
            + case_py
        )
        case_cpp = case_py
        self.run_case("41_sparse_matmul_tla", case_cpp)

    @only_on_3510
    def test_49_ascend950_flash_attention_infer(self):
        case_py = [str(i) for i in [1, 138, 100, 4, 2, 128, 0, 0, 0]] + ["half"]
        ret = subprocess.run(["python", os.path.join(
            CMAKE_EXAMPLES_PATH, "49_ascend950_flash_attention_infer", "gen_data.py")] + case_py)
        case_cpp = [str(i) for i in [1, 138, 100, 4, 2, 128, 0, 0, 0]] + ["--dtype", "half", "--device", "1",
            "--datapath", os.path.join(CMAKE_EXAMPLES_PATH, "49_ascend950_flash_attention_infer", "data")]
        self.run_case("49_ascend950_flash_attention_infer", case_cpp)

    @only_on_3510
    def test_53_ascend950_fp8_mx_matmul(self):
        # gen_data: trans_a=0, trans_b=1 对应 RowMajor A + ColumnMajor B（与 example 中 layout 一致）
        case_py = [str(i) for i in [256, 512, 1024, 0, 1]]
        subprocess.run(
            [
                "python",
                os.path.join(
                    CMAKE_EXAMPLES_PATH, "53_ascend950_fp8_mx_matmul", "gen_data.py"
                ),
            ]
            + case_py
        )
        case_cpp = [str(i) for i in [256, 512, 1024]] + [1]
        self.run_case("53_ascend950_fp8_mx_matmul", case_cpp)

    @only_on_3510
    def test_54_ascend950_fp4_mx_matmul(self):
        # 与 53 相同：trans_b=1 对应 B 为 ColumnMajor，与 fp4_mx_matmul.cpp 中 layout 一致
        case_py = [str(i) for i in [256, 512, 1024, 0, 1]]
        subprocess.run(
            [
                "python",
                os.path.join(
                    CMAKE_EXAMPLES_PATH, "54_ascend950_fp4_mx_matmul", "gen_data.py"
                ),
            ]
            + case_py
        )
        case_cpp = [str(i) for i in [256, 512, 1024]] + [1]
        self.run_case("54_ascend950_fp4_mx_matmul", case_cpp)

    @only_on_3510
    def test_53_ascend950_fp8_mx_matmul_aswt(self):
        case_py = [str(i) for i in [256, 512, 1024, 0, 1]]
        subprocess.run(
            [
                "python",
                os.path.join(
                    CMAKE_EXAMPLES_PATH, "53_ascend950_fp8_mx_matmul", "gen_data.py"
                ),
            ]
            + case_py
        )
        case_cpp = [str(i) for i in [256, 512, 1024]] + [1]
        self.run_case("53_ascend950_fp8_mx_matmul_aswt", case_cpp)

    @only_on_3510
    def test_54_ascend950_fp4_mx_matmul_aswt(self):
        case_py = [str(i) for i in [256, 512, 1024, 0, 1]]
        subprocess.run(
            [
                "python",
                os.path.join(
                    CMAKE_EXAMPLES_PATH, "54_ascend950_fp4_mx_matmul", "gen_data.py"
                ),
            ]
            + case_py
        )
        case_cpp = [str(i) for i in [256, 512, 1024]] + [1]
        self.run_case("54_ascend950_fp4_mx_matmul_aswt", case_cpp)

    @only_on_3510
    def test_53_ascend950_fp8_mx_matmul_small_shape(self):
        # 与默认用例不同的小形状，仍使用 trans_a=0, trans_b=1（RowMajor A + ColumnMajor B）
        case_py = [str(i) for i in [128, 256, 512, 0, 1]]
        subprocess.run(
            [
                "python",
                os.path.join(
                    CMAKE_EXAMPLES_PATH, "53_ascend950_fp8_mx_matmul", "gen_data.py"
                ),
            ]
            + case_py
        )
        case_cpp = [str(i) for i in [128, 256, 512]] + [1]
        self.run_case("53_ascend950_fp8_mx_matmul", case_cpp)

    @only_on_3510
    def test_54_ascend950_fp4_mx_matmul_small_shape(self):
        case_py = [str(i) for i in [128, 256, 512, 0, 1]]
        subprocess.run(
            [
                "python",
                os.path.join(
                    CMAKE_EXAMPLES_PATH, "54_ascend950_fp4_mx_matmul", "gen_data.py"
                ),
            ]
            + case_py
        )
        case_cpp = [str(i) for i in [128, 256, 512]] + [1]
        self.run_case("54_ascend950_fp4_mx_matmul", case_cpp)

    @only_on_3510
    def test_53_ascend950_fp8_mx_matmul_cube_1024(self):
        # 大方块形状，覆盖与 256×512×1024 不同的 tile 划分
        case_py = [str(i) for i in [1024, 1024, 1024, 0, 1]]
        subprocess.run(
            [
                "python",
                os.path.join(
                    CMAKE_EXAMPLES_PATH, "53_ascend950_fp8_mx_matmul", "gen_data.py"
                ),
            ]
            + case_py
        )
        case_cpp = [str(i) for i in [1024, 1024, 1024]] + [1]
        self.run_case("53_ascend950_fp8_mx_matmul", case_cpp)

    @only_on_3510
    def test_54_ascend950_fp4_mx_matmul_cube_1024(self):
        case_py = [str(i) for i in [1024, 1024, 1024, 0, 1]]
        subprocess.run(
            [
                "python",
                os.path.join(
                    CMAKE_EXAMPLES_PATH, "54_ascend950_fp4_mx_matmul", "gen_data.py"
                ),
            ]
            + case_py
        )
        case_cpp = [str(i) for i in [1024, 1024, 1024]] + [1]
        self.run_case("54_ascend950_fp4_mx_matmul", case_cpp)

    @only_on_3510
    def test_53_ascend950_fp8_mx_matmul_aswt_shape_512_1024_256(self):
        # tile 划分与默认用例不同，便于覆盖 ASWT 调度路径（L1 M/N=256）
        case_py = [str(i) for i in [512, 1024, 256, 0, 1]]
        subprocess.run(
            [
                "python",
                os.path.join(
                    CMAKE_EXAMPLES_PATH, "53_ascend950_fp8_mx_matmul", "gen_data.py"
                ),
            ]
            + case_py
        )
        case_cpp = [str(i) for i in [512, 1024, 256]] + [1]
        self.run_case("53_ascend950_fp8_mx_matmul_aswt", case_cpp)

    @only_on_3510
    def test_54_ascend950_fp4_mx_matmul_aswt_shape_512_1024_256(self):
        case_py = [str(i) for i in [512, 1024, 256, 0, 1]]
        subprocess.run(
            [
                "python",
                os.path.join(
                    CMAKE_EXAMPLES_PATH, "54_ascend950_fp4_mx_matmul", "gen_data.py"
                ),
            ]
            + case_py
        )
        case_cpp = [str(i) for i in [512, 1024, 256]] + [1]
        self.run_case("54_ascend950_fp4_mx_matmul_aswt", case_cpp)

    @only_on_3510
    def test_55_ascend950_mx_grouped_matmul_slice_m(self):
        for (trans, quant_type) in itertools.product(("0", "1"), ("float8_e4m3fn", "float8_e5m2", "float4_e2m1fn_x2")):
            case_py = [str(i) for i in [2, 588, 988, 1030]] + [trans, quant_type, '0']  # G, M, N, K, trans_b, quant_type, device_id
            ret = subprocess.run(
                ["python", os.path.join(
                    CMAKE_EXAMPLES_PATH, "55_ascend950_mx_grouped_matmul_slice_m", "gen_data_compare.py")] + case_py,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self._ret_check(ret)

    @only_on_3510
    def test_55_ascend950_mx_grouped_matmul_slice_m_aswt_fp4_odd_n_transb(self):
        case_py = [str(i) for i in [2, 588, 989, 1030, 1]] + [
            "float4_e2m1fn_x2",
            "0",
            "--bin",
            "55_ascend950_mx_grouped_matmul_slice_m_aswt",
        ]
        ret = subprocess.run(
            [
                "python",
                os.path.join(
                    CMAKE_EXAMPLES_PATH,
                    "55_ascend950_mx_grouped_matmul_slice_m",
                    "gen_data_compare.py",
                ),
            ] + case_py,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._ret_check(ret)
        self.assertIn("Compare success", ret.stdout.decode())

    @only_on_3510
    def test_57_ascend950_matmul_full_dequant(self):
        case_py = ["--shape"] + ["513 513 513"] + ["--x1_quant_mode"] + ["per_token"] + ["--x2_quant_mode"] + ["per_channel"]
        subprocess.run(
            [
                "python",
                os.path.join(
                    CMAKE_EXAMPLES_PATH, "57_ascend950_matmul_full_dequant", "scripts", "gen_data.py"
                ),
            ]
            + case_py
        )
        subprocess.run(
            [
                "cp", "-r", "input", CMAKE_BINARY_PATH,
            ]
        )
        subprocess.run(
            [
                "cp", "-r", "output", CMAKE_BINARY_PATH,
            ]
        )
        case_cpp = [str(i) for i in [513, 513, 513]] + ["per_token"] + ["per_channel"]
        self.run_case("57_ascend950_matmul_full_dequant", case_cpp)

    @only_on_3510
    def test_59_ascend950_a8w4_mx_matmul(self):
        case_py = [str(i) for i in [256, 128, 128, 0, 1]]
        subprocess.run(
            [
                "python",
                os.path.join(
                    CMAKE_EXAMPLES_PATH, "59_ascend950_a8w4_mx_matmul", "gen_data.py"
                ),
            ]
            + case_py
        )
        case_cpp = [str(i) for i in [256, 128, 128]] + [1]
        self.run_case("59_ascend950_a8w4_mx_matmul", case_cpp)

    @only_on_3510
    def test_62_ascend950_broadcast_matmul_perblock_quant(self):
        case_py = [str(i) for i in [5920, 128, 128, 128, 0]]  # batch_count, m, n, k, device_id
        ret = subprocess.run(
            ["python", os.path.join(
                CMAKE_EXAMPLES_PATH, "62_ascend950_broadcast_matmul_perblock_quant", "gen_data_compare.py")] + case_py,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            )
        self._ret_check(ret)

    @only_on_3510
    def test_70_ascend950_flash_attention_chunk_prefill(self):
        case_py = [str(i) for i in [1, 100, 138, 8, 1, 128, 128, 0]] + ["half"] + [str(i) for i in [2, 0, 128]] + ["nd"]
        ret = subprocess.run(["python", os.path.join(
            CMAKE_EXAMPLES_PATH, "70_ascend950_flash_attention_chunk_prefill", "gen_data.py")] + case_py)
        case_cpp = [str(i) for i in [1, 100, 138, 8, 1, 128, 128, 0, 2, 128]] + ["--dtype", "half", "--cache_layout", "nd","--device", "1",
        "--datapath", os.path.join(CMAKE_EXAMPLES_PATH, "70_ascend950_flash_attention_chunk_prefill", "data")]
        self.run_case("70_ascend950_flash_attention_chunk_prefill", case_cpp)

    @only_on_3510
    def test_74_ascend950_weight_quant_a8w4_grouped_mx_matmul(self):
        case_py = [
            "expect_m_per_group",
            "48",
            "64",
            "3072",
            "2048",
            "4096",
            "1",
        ]
        ret = subprocess.run(
            [
                "python3",
                os.path.join(
                    CMAKE_EXAMPLES_PATH,
                    "74_ascend950_weight_quant_a8w4_grouped_mx_matmul",
                    "gen_data.py",
                ),
            ]
            + case_py,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._ret_check(ret)

        case_cpp = ["48", "3072", "2048", "4096", "0"]
        self.run_case(
            "74_ascend950_weight_quant_a8w4_grouped_mx_matmul", case_cpp
        )

normal_cases_2201 = [
    "00_basic_matmul 256 512 1024 0",
    "01_batched_matmul 5 256 512 1024 0",
    "02_grouped_matmul_slice_m 128 512 1024 2048 0",
    "03_matmul_add 256 512 1024 0",
    "04_padding_matmul 256 512 1024 0",
    "05_grouped_matmul_slice_k 128 512 1024 32 0",
    "06_optimized_matmul 256 512 1024 0",
    "07_grouped_matmul_slice_m_per_token_dequant_moe 128 512 1024 2048 0",
    "08_grouped_matmul 128 512 1024 2048 0",
    "09_splitk_matmul 256 512 1024 0",
    "10_grouped_matmul_slice_m_per_token_dequant 128 512 1024 2048 0",
    "11_grouped_matmul_slice_k_per_token_dequant 128 512 1024 2048 0",
    "12_quant_matmul 256 512 1024 0",
    "13_basic_matmul_tla 256 512 1024 0",
    "14_optimized_matmul_tla 256 512 1024 0",
    "15_gemm 256 512 1024 0",
    "16_group_gemm 3 '128,256,512' '256,512,128' '512,256,128' 0",
    "17_gemv_aiv 256 512 0",
    "18_gemv_aic 256 512 0",
    "20_matmul_bias 256 512 1024 0",
    "21_basic_matmul_preload_zN 256 512 1024 0",
    "22_padding_splitk_matmul 256 512 1024 0",
    "25_matmul_full_loadA 256 512 1024 0",
    "26_matmul_relu 256 512 1024 0",
    "27_matmul_gelu 256 512 1024 0",
    "28_matmul_silu 256 512 1024 0",
    "30_w8a16_matmul 256 512 1024 0",
    "31_small_matmul 256 1024 256 0",
    "33_basic_conv2d 2 33 43 112 80 3 3 2 2 2 2 1 1 1 1 0",
    "37_streamk_matmul 256 512 1024 0",
    "34_single_core_splitk_matmul 256 512 1024 0",
    "42_quant_optimized_matmul_tla 256 512 1024 0",
    "44_quant_matmul_full_loadA_tla 256 512 1024 0",
    "45_strided_batched_matmul_tla 5 256 512 1024 0",
    "52_quant_multi_core_splitk_matmul_tla 256 512 1024 0",
    "102_dynamic_optimized_matmul 256 512 1024 0 0 0"
    "103_dynamic_optimized_quant_matmul_per_token_basic 256 512 1024 0 0 0",
]

normal_cases_3510 = [
    "43_ascend950_basic_matmul 256 512 1024 0",
    "46_ascend950_matmul_fixpipe_opti 256 512 1024 0",
    "47_ascend950_grouped_matmul_slice_m_per_token_dequant 128 512 1024 2048 0",
    "48_ascend950_grouped_matmul_slice_m_per_tensor_per_channel_dequant 128 512 1024 2048 0 0",
    "50_ascend950_basic_matmul_gemv 1 128 127 0",
    "51_ascend950_quant_matmul_per_group_per_block_tla 256 512 1024 0",
    "56_ascend950_basic_conv2d_tla 2 33 43 112 80 3 3 2 2 2 2 1 1 1 1 0",
    "60_ascend950_grouped_matmul_slice_m 128 512 1024 2048 0",
    "64_ascend950_matmul_evg_add 256 512 1024 0",
    "64_ascend950_matmul_evg_leaky_relu 256 512 1024 0",
    "64_ascend950_matmul_evg_sigmoid 256 512 1024 0",
    "64_ascend950_matmul_evg_silu 256 512 1024 0",
    "64_ascend950_matmul_evg_tanh 256 512 1024 0",
    "64_ascend950_matmul_evg_bias 256 512 1024 0",
    "64_ascend950_matmul_evg_add_ub 256 512 1024 0",
    "68_ascend950_multi_core_splitk_matmul 256 512 1024 0",
    "69_ascend950_tail_multi_core_splitk_matmul 256 512 1024 0",
    "80_grouped_matmul_slice_m 128 512 1024 2048 0",
]


def set_case(case: str, npu_arch: int):
    case_splited = case.split()
    case_executable_name = case_splited[0]
    case_args = case_splited[1:]

    def __(self: Type[CatlassExampleTest]):
        self.run_case(case_executable_name, case_args)

    setattr(
        CatlassExampleTest,
        f"test_{case_executable_name}",
        only_on_npu_arch(npu_arch)(__),
    )


for normal_case in normal_cases_2201:
    set_case(normal_case, 2201)
for normal_case in normal_cases_3510:
    set_case(normal_case, 3510)


if __name__ == "__main__":
    unittest.main()
