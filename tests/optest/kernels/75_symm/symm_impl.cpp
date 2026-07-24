/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */
#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/kernel/symm_tla.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

#include "catlass_kernel.h"
#include "common/kernel_runner.h"
#include "common/tile_shape_scaler_tla.h"

#include <type_traits>

using namespace Catlass;

#ifdef CATLASS_JIT_ELEMENT_A
using ElementA = CATLASS_JIT_ELEMENT_A;
#else
using ElementA = float;
#endif
#ifdef CATLASS_JIT_ELEMENT_B
using ElementB = CATLASS_JIT_ELEMENT_B;
#else
using ElementB = float;
#endif
#ifdef CATLASS_JIT_ELEMENT_C
using ElementC = CATLASS_JIT_ELEMENT_C;
#else
using ElementC = float;
#endif
#ifdef CATLASS_JIT_LAYOUT_A
using LayoutTagA = layout::CATLASS_JIT_LAYOUT_A;
#else
using LayoutTagA = layout::RowMajor;
#endif
#ifdef CATLASS_JIT_LAYOUT_B
using LayoutTagB = layout::CATLASS_JIT_LAYOUT_B;
#else
using LayoutTagB = layout::RowMajor;
#endif
#ifdef CATLASS_JIT_LAYOUT_C
using LayoutTagC = layout::CATLASS_JIT_LAYOUT_C;
#else
using LayoutTagC = layout::RowMajor;
#endif

#ifdef CATLASS_JIT_SYMM_SIDE
constexpr uint32_t symmSide = CATLASS_JIT_SYMM_SIDE;
#else
constexpr uint32_t symmSide = 0;
#endif
#ifdef CATLASS_JIT_SYMM_FILL
constexpr uint32_t symmFill = CATLASS_JIT_SYMM_FILL;
#else
constexpr uint32_t symmFill = 0;
#endif
#ifdef CATLASS_JIT_BLOCK_SCHEDULER
constexpr uint32_t blockSchedulerCode = CATLASS_JIT_BLOCK_SCHEDULER;
#else
constexpr uint32_t blockSchedulerCode = 30;
#endif

static_assert(symmSide <= 1, "SYMM side must be 0 (left) or 1 (right)");
static_assert(symmFill <= 1, "SYMM fill must be 0 (lower) or 1 (upper)");

constexpr bool isLeftSide = (symmSide == 0);
constexpr auto kernelSide = isLeftSide ? Gemm::Kernel::SymmMatmulSide::LEFT : Gemm::Kernel::SymmMatmulSide::RIGHT;
constexpr auto kernelFill =
    (symmFill == 0) ? Gemm::Kernel::SymmMatmulFillMode::LOWER : Gemm::Kernel::SymmMatmulFillMode::UPPER;

using ArchTag = Arch::AtlasA2;
constexpr bool enableUnitFlag = true;

using LeftL1TileShape = tla::tuple<tla::C<128>, tla::C<256>, tla::C<128>>;
using LeftL0TileShape = tla::tuple<tla::C<128>, tla::C<256>, tla::C<32>>;
using RightL1TileShape = tla::tuple<tla::C<256>, tla::C<128>, tla::C<128>>;
using RightL0TileShape = tla::tuple<tla::C<256>, tla::C<128>, tla::C<32>>;

using ScaledLeftL1TileShape = typename CatlassKernel::TileShapeScalerTLA<ElementA, float, LeftL1TileShape>::type;
using ScaledLeftL0TileShape = typename CatlassKernel::TileShapeScalerTLA<ElementA, float, LeftL0TileShape>::type;
using ScaledRightL1TileShape = typename CatlassKernel::TileShapeScalerTLA<ElementA, float, RightL1TileShape>::type;
using ScaledRightL0TileShape = typename CatlassKernel::TileShapeScalerTLA<ElementA, float, RightL0TileShape>::type;

using LeftDispatchPolicy = Gemm::MmadPingpongSymmLeft<ArchTag, enableUnitFlag>;
using RightDispatchPolicy = Gemm::MmadPingpongSymmRight<ArchTag, enableUnitFlag>;

using LeftTileCopyP =
    Gemm::Tile::PackedTileCopyTla<ArchTag, ElementA, LayoutTagA, ElementB, LayoutTagB, ElementC, LayoutTagC>;
using LeftTileCopyQ =
    Gemm::Tile::PackedTileCopyTla<ArchTag, ElementA, layout::ColumnMajor, ElementB, LayoutTagB, ElementC, LayoutTagC>;
using RightTileCopyP =
    Gemm::Tile::PackedTileCopyTla<ArchTag, ElementA, LayoutTagA, ElementB, LayoutTagB, ElementC, LayoutTagC>;
using RightTileCopyQ =
    Gemm::Tile::PackedTileCopyTla<ArchTag, ElementA, LayoutTagA, ElementB, layout::ColumnMajor, ElementC, LayoutTagC>;

using LeftBlockMmad = Gemm::Block::BlockMmadPingpongSymmLeftTla<
    LeftDispatchPolicy, ScaledLeftL1TileShape, ScaledLeftL0TileShape, ElementA, ElementB, ElementC, void, LeftTileCopyP,
    LeftTileCopyQ>;
using RightBlockMmad = Gemm::Block::BlockMmadPingpongSymmRightTla<
    RightDispatchPolicy, ScaledRightL1TileShape, ScaledRightL0TileShape, ElementA, ElementB, ElementC, void,
    RightTileCopyP, RightTileCopyQ>;
using BlockMmad = std::conditional_t<isLeftSide, LeftBlockMmad, RightBlockMmad>;
using BlockScheduler = Gemm::Block::GemmIdentityBlockSwizzle<(blockSchedulerCode / 10), (blockSchedulerCode % 10)>;

using SymmKernel = Gemm::Kernel::SymmMatmulTlaSingleKernelProducer<kernelSide, kernelFill, BlockMmad, BlockScheduler>;

extern "C" void run(uint32_t blockNum, aclrtStream stream, const CatlassKernel::SymmParams* params)
{
    if (params == nullptr) {
        return;
    }

    GemmCoord shape{params->m, params->n, params->k};
    auto layoutC = tla::MakeLayout<ElementC, LayoutTagC>(params->m, params->n);

    if constexpr (isLeftSide) {
        auto layoutSymP = tla::MakeLayout<ElementA, LayoutTagA>(params->m, params->k);
        auto layoutSymQ = tla::MakeLayout<ElementA, layout::ColumnMajor>(params->m, params->k);
        auto layoutNonSym = tla::MakeLayout<ElementB, LayoutTagB>(params->k, params->n);

        typename SymmKernel::Arguments arguments{
            shape,        params->inputAddr[0],  layoutSymP, layoutSymQ, params->inputAddr[1],
            layoutNonSym, params->outputAddr[0], layoutC};
        Catlass::RunKernel<SymmKernel>(arguments, stream, blockNum);
    } else {
        auto layoutSymP = tla::MakeLayout<ElementB, LayoutTagB>(params->k, params->n);
        auto layoutSymQ = tla::MakeLayout<ElementB, layout::ColumnMajor>(params->k, params->n);
        auto layoutNonSym = tla::MakeLayout<ElementA, LayoutTagA>(params->m, params->k);

        typename SymmKernel::Arguments arguments{
            shape,        params->inputAddr[0],  layoutSymP, layoutSymQ, params->inputAddr[1],
            layoutNonSym, params->outputAddr[0], layoutC};
        Catlass::RunKernel<SymmKernel>(arguments, stream, blockNum);
    }
}
