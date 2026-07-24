/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_BLOCK_BLOCK_MMAD_PLANAR_COMPLEX_FUSED_TLA_HPP
#define CATLASS_GEMM_BLOCK_BLOCK_MMAD_PLANAR_COMPLEX_FUSED_TLA_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/gemm/helper.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

namespace Catlass::Gemm::Block {

/// Specialization of BlockMmadTla for the MmadPlanarComplexFused policy.
///
/// Fused planar-complex block. Single K-loop with alternating A/B operands
/// and a shared L0C buffer. NEGATE_A is not a block parameter: the kernel
/// layer decides which imaginary operand to pre-negate, aliases the Signed
/// GM pointers accordingly, and passes 6 input tensors (A_real, A_imag,
/// A_imag_signed, B_real, B_imag, B_imag_signed) to the block. The block
/// always reads the Signed slots for the C_real cross term and the original
/// slots for C_imag, so it is agnostic to the negate direction.
///
/// C_real and C_imag share the same L0C buffer (time-multiplexed), so tile
/// shapes can match basic_matmul exactly (e.g. L1<128,256,256>, L0<128,256,64>).
///
/// Stage 1 - C_real (per K-tile: 2 sub-iterations, accumulate into l0C):
///   even: A_real * B_real       -> l0C
///   odd:  AImagSigned * BImagSigned -> l0C
///         The kernel pre-negates exactly one imaginary operand into a GM
///         workspace and aliases the Signed pointer to it; the other Signed
///         pointer aliases the original. The product is always -A_imag*B_imag
///         regardless of which side was negated, so the block just consumes
///         the Signed slots for the cross term - no NEGATE_A awareness here.
///   ... repeat for each K-tile, then FixPipe l0C -> GM_C_real
///
/// Stage 2 - C_imag (per K-tile: 2 sub-iterations, initC, same l0C):
///   even: A_imag * B_real -> l0C (initC on first sub)
///   odd:  A_real * B_imag -> l0C
///   ... repeat for each K-tile, then FixPipe l0C -> GM_C_imag
///
/// Pipeline overlap: FixPipe of C_real overlaps with data loading (GM->L1->L0)
/// for C_imag's first sub-iteration.
///
/// L1 layout (4 slots, same as basic_matmul):
///   [A_K0 | A_K1 | B_K0 | B_K1]
///   A and B slots are generic - their GM source alternates per sub-iteration.
///
/// L0A and L0B use dual-buffer (pingpong) to overlap L1->L0 copy and Cube MMAD.
/// L0C uses a single buffer (C_real FixPipe completes before C_imag starts).
///
/// HardEvent ID budget (Atlas A2: 0..3 per event type):
///   MTE1_MTE2 / MTE2_MTE1: {0,1} = L1A K-pingpong; {2,3} = L1B K-pingpong
///   M_MTE1    / MTE1_M:    {0,1} = L0A pingpong; {2,3} = L0B pingpong
///   MTE1_M signal:          0 = L0B data ready for MMAD
///   M_FIX     / FIX_M:     0 = FixPipe sync
template <
    class ArchTag_, bool ENABLE_SHUFFLE_K_, class L1TileShape_, class L0TileShape_, class ElementA_, class ElementB_,
    class ElementC_, class ElementBias_, class TileCopy_, class TileMmad_>
struct BlockMmadTla<
    MmadPlanarComplexFused<ArchTag_, ENABLE_SHUFFLE_K_>, L1TileShape_, L0TileShape_, ElementA_, ElementB_, ElementC_,
    ElementBias_, TileCopy_, TileMmad_> {
public:
    // Type Aliases
    using DispatchPolicy = MmadPlanarComplexFused<ArchTag_, ENABLE_SHUFFLE_K_>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using TileCopy = TileCopy_;
    using TileMmad = TileMmad_;
    using L1TileShape = L1TileShape_;
    using L0TileShape = L0TileShape_;
    using ElementA = ElementA_;
    using ElementB = ElementB_;
    using ElementCReal = ElementC_;
    using ElementCImag = ElementC_;
    using ElementAccumulator = typename TileCopy::ElementAccumulator;

    using LayoutA = typename TileCopy::LayoutA;
    using LayoutB = typename TileCopy::LayoutB; // shared by BReal/BImag
    using LayoutC = typename TileCopy::LayoutC; // shared by CReal/CImag
    using LayoutTagA = typename TileCopy::LayoutTagA;
    using LayoutTagB = typename TileCopy::LayoutTagB;
    using LayoutTagC = typename TileCopy::LayoutTagC;
    using LayoutTagL1A = typename TileCopy::LayoutTagL1A;
    using LayoutTagL1B = typename TileCopy::LayoutTagL1B;
    using LayoutTagL0A = typename TileCopy::LayoutTagL0A;
    using LayoutTagL0B = typename TileCopy::LayoutTagL0B;
    using LayoutL0C = typename TileCopy::LayoutL0C;

    using L1AAlignHelper = typename TileCopy_::L1AAlignHelper;
    using L1BAlignHelper = typename TileCopy_::L1BAlignHelper;

    static constexpr bool ENABLE_SHUFFLE_K = DispatchPolicy::ENABLE_SHUFFLE_K;
    static constexpr uint32_t STAGES = DispatchPolicy::STAGES;
    static_assert(
        STAGES == 2,
        "PlanarComplexFused block requires STAGES == 2 (L1 4-slot K-pingpong and L0 pingpong assume double buffering)");

    static constexpr uint32_t L1_TILE_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t L1_TILE_N = tla::get<1>(L1TileShape{});
    static constexpr uint32_t L1_TILE_K = tla::get<2>(L1TileShape{});
    static constexpr uint32_t L0_TILE_M = tla::get<0>(L0TileShape{});
    static constexpr uint32_t L0_TILE_N = tla::get<1>(L0TileShape{});
    static constexpr uint32_t L0_TILE_K = tla::get<2>(L0TileShape{});

    static constexpr uint32_t L1A_SIZE = L1_TILE_M * L1_TILE_K * sizeof(ElementA);
    static constexpr uint32_t L1B_SIZE = L1_TILE_K * L1_TILE_N * sizeof(ElementB);

    static constexpr uint32_t L0A_SIZE = ArchTag::L0A_SIZE;
    static constexpr uint32_t L0B_SIZE = ArchTag::L0B_SIZE;
    static constexpr uint32_t L0C_SIZE = ArchTag::L0C_SIZE;
    static constexpr uint32_t L0A_PINGPONG_BUF_SIZE = L0A_SIZE / STAGES;
    static constexpr uint32_t L0B_PINGPONG_BUF_SIZE = L0B_SIZE / STAGES;
    static constexpr uint32_t L0C_TILE_SIZE = L1_TILE_M * L1_TILE_N * sizeof(ElementAccumulator);

    static constexpr uint32_t L0A_TILE_SIZE = L0_TILE_M * L0_TILE_K * sizeof(ElementA);
    static constexpr uint32_t L0B_TILE_SIZE = L0_TILE_K * L0_TILE_N * sizeof(ElementB);

    static_assert(tla::detail::isRowMajor<LayoutC>::value, "LayoutC must be RowMajor");
    // 2 A slots + 2 B slots = 4 L1 slots (same as basic_matmul).
    static_assert((L1A_SIZE * STAGES + L1B_SIZE * STAGES) <= ArchTag::L1_SIZE, "L1 exceeds limit for A + B K-pingpong");
    // Single L0C buffer - same size as basic_matmul.
    static_assert(L0C_TILE_SIZE <= L0C_SIZE, "L0C tile doesn't fit in L0C");
    static_assert((L0A_TILE_SIZE * STAGES) <= L0A_SIZE, "L0A pingpong tiles don't fit in L0A");
    static_assert((L0B_TILE_SIZE * STAGES) <= L0B_SIZE, "L0B pingpong tiles don't fit in L0B");
    static_assert(L1_TILE_M == L0_TILE_M && L1_TILE_N == L0_TILE_N, "L1 and L0 must share M and N");
    static_assert(L0_TILE_K <= L1_TILE_K, "L0TileShape::K > L1TileShape::K");

    static constexpr auto L1A_LAYOUT =
        tla::MakeLayout<ElementA, LayoutTagL1A>(tla::Int<L1_TILE_M>{}, tla::Int<L1_TILE_K>{});
    static constexpr auto L1B_LAYOUT =
        tla::MakeLayout<ElementB, LayoutTagL1B>(tla::Int<L1_TILE_K>{}, tla::Int<L1_TILE_N>{});

    /// Construct
    CATLASS_DEVICE
    BlockMmadTla(Arch::Resource<ArchTag>& resource)
    {
        // L1 layout: [A_K0 | A_K1 | B_K0 | B_K1]
        uint32_t l1AOffset = 0;
        uint32_t l1BOffset = L1A_SIZE * STAGES;

        for (uint32_t i = 0; i < STAGES; i++) {
            l1ATensorList[i] = resource.l1Buf.template GetBufferByByte<ElementA>(l1AOffset + L1A_SIZE * i);
            l1BTensorList[i] = resource.l1Buf.template GetBufferByByte<ElementB>(l1BOffset + L1B_SIZE * i);

            l1AEventList[i] = i;     // {0, 1}
            l1BEventList[i] = 2 + i; // {2, 3}

            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[i]);
        }

        // L0A: dual-buffer pingpong
        for (uint32_t i = 0; i < STAGES; i++) {
            l0ATensorList[i] = resource.l0ABuf.template GetBufferByByte<ElementA>(L0A_PINGPONG_BUF_SIZE * i);
            l0AEventList[i] = i;
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[i]);
        }

        // L0B: dual-buffer pingpong
        for (uint32_t i = 0; i < STAGES; i++) {
            l0BTensorList[i] = resource.l0BBuf.template GetBufferByByte<ElementB>(L0B_PINGPONG_BUF_SIZE * i);
            l0BEventList[i] = STAGES + i;
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[i]);
        }

        // L0C: single buffer (shared by C_real and C_imag, time-multiplexed)
        l0CTensor = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(0);

        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
    }

    CATLASS_DEVICE
    ~BlockMmadTla()
    {
        for (uint32_t i = 0; i < STAGES; i++) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[i]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[i]);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[i]);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[i]);
        }
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
    }

    /// Single entry point - computes C = A * B for planar complex GEMM.
    ///
    /// C_real = A_R * B_R + signed imaginary cross term
    /// where the signed term is AImagSigned * BImagSigned (kernel pre-negates one side)
    /// C_imag = A_I * B_R + A_R * B_I
    ///
    /// All operands are TLA tensors (GM views with layout + coord). The caller
    /// (kernel layer) is responsible for creating them via tla::MakeTensor and
    /// tiling them to the block scope via tla::GetTile.
    template <
        class TensorAReal, class TensorAImag, class TensorAImagNeg, class TensorBReal, class TensorBImag,
        class TensorBImagNeg, class TensorCReal, class TensorCImag>
    CATLASS_DEVICE void operator()(
        TensorAReal& tensorAReal, TensorAImag& tensorAImag, TensorAImagNeg& tensorAImagNeg, TensorBReal& tensorBReal,
        TensorBImag& tensorBImag, TensorBImagNeg& tensorBImagNeg, TensorCReal& tensorCReal, TensorCImag& tensorCImag,
        GemmCoord const& actualShape)
    {
        using CopyGmToL1A = typename TileCopy::template CopyGmToL1A<TensorAReal>;
        using CopyGmToL1BReal = typename TileCopy::template CopyGmToL1B<TensorBReal>;
        using CopyGmToL1BImag = typename TileCopy::template CopyGmToL1B<TensorBImag>;
        using CopyL1ToL0A = typename TileCopy::CopyL1ToL0A;
        using CopyL1ToL0B = typename TileCopy::CopyL1ToL0B;
        using CopyL0CToGmReal = typename TileCopy::template CopyL0CToGm<TensorCReal>;
        using CopyL0CToGmImag = typename TileCopy::template CopyL0CToGm<TensorCImag>;

        CopyGmToL1A copyGmToL1A;
        CopyGmToL1BReal copyGmToL1BReal;
        CopyGmToL1BImag copyGmToL1BImag;
        CopyL1ToL0A copyL1ToL0A;
        CopyL1ToL0B copyL1ToL0B;
        CopyL0CToGmReal copyL0CToGmReal;
        CopyL0CToGmImag copyL0CToGmImag;

        uint32_t mRound = RoundUp<L1AAlignHelper::M_ALIGNED>(actualShape.m());
        uint32_t nRound = RoundUp<L1BAlignHelper::N_ALIGNED>(actualShape.n());
        auto layoutInL0C = tla::MakeLayoutL0C(mRound, nRound);
        auto tensorL0C = tla::MakeTensor(l0CTensor, layoutInL0C, Arch::PositionL0C{});

        uint32_t kTileCount = CeilDiv<L1_TILE_K>(actualShape.k());

        // K=0 (empty reduction): bail out before any MTE2/MMAD to avoid
        // modulo-by-zero UB in GetShuffleKTileIndex and uint32 underflow
        // (kTileCount-1 wraps to 0xFFFFFFFF) in GetKActual. Returning here
        // is flag-safe: constructor-set flags stay pending for the destructor.
        if (kTileCount == 0) {
            return;
        }

        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);

        // ==================================================================
        // Stage 1: C_real = A_R * B_R + A_I * (-B_I)
        // ==================================================================
        // Stage 1 runs full 2*K sub-iterations and leaves the warmup load
        // for Stage 2 to be issued by the caller (below) so that MTE2 can
        // overlap with Stage 1's tail fixpipe.
        RunStage(
            /*isRealStage=*/true, /*skipWarmup=*/false, tensorAReal, tensorAImag, tensorAImagNeg, tensorBReal,
            tensorBImag, tensorBImagNeg, tensorL0C, copyGmToL1A, copyGmToL1BReal, copyGmToL1BImag, copyL1ToL0A,
            copyL1ToL0B, kTileCount, mRound, nRound, actualShape);

        // FixPipe C_real -> GM. SetFlag<M_FIX> drains the M (MMAD) pipe before
        // fixpipe reads L0C. The fixpipe (FIX pipe) runs concurrently with
        // the Stage 2 warmup MTE2 issued below - different pipes, no L0C
        // dependency on the MTE2/MTE1 path.
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);

        copyL0CToGmReal(tensorCReal, tensorL0C);

        // ---- Stage 2 warmup: issue MTE2(GM->L1) for the first sub of Stage 2
        //      *while* fixpipe(C_real) is in flight. MTE2 has no L0C
        //      dependency, so it runs in parallel with the fixpipe. ----
        IssueWarmup(tensorAImag, tensorBReal, copyGmToL1A, copyGmToL1BReal, kTileCount, actualShape);

        // After fixpipe finishes reading L0C, hand L0C back to Cube pipe so
        // Stage 2 MMAD can write it. SetFlag<FIX_M> drains the FIX pipe.
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);

        // ==================================================================
        // Stage 2: C_imag = A_I * B_R + A_R * B_I
        // ==================================================================
        // Stage 2 skips its internal warmup - already issued above and
        // overlapped with Stage 1's fixpipe.
        RunStage(
            /*isRealStage=*/false, /*skipWarmup=*/true, tensorAReal, tensorAImag, tensorAImagNeg, tensorBReal,
            tensorBImag, tensorBImagNeg, tensorL0C, copyGmToL1A, copyGmToL1BReal, copyGmToL1BImag, copyL1ToL0A,
            copyL1ToL0B, kTileCount, mRound, nRound, actualShape);

        // FixPipe C_imag -> GM. SetFlag<M_FIX> drains M pipe before fixpipe.
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);

        copyL0CToGmImag(tensorCImag, tensorL0C);

        // SetFlag<FIX_M> drains FIX pipe so the next block's first MMAD can
        // write L0C. The matching WaitFlag is at the top of operator(),
        // allowing this block's fixpipe to overlap with the next block's
        // Stage 1 warmup MTE2.
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
    }

private:
    /// Issue MTE2(GM->L1) for the first sub-iteration of Stage 2 (C_imag)
    /// without waiting on any L0C/Cube events. Called from the host flow
    /// during Stage 1's fixpipe so MTE2 overlaps with FIX.
    ///
    /// Stage 2 even sub loads A_imag and B_real. The matching
    /// WaitFlag<MTE2_MTE1> is consumed inside RunStage's main loop on the
    /// first MMAD micro (where it normally is). RunStage is invoked with
    /// skipWarmup=true so it does not re-issue these MTE2s.
    ///
    /// curL1A / curL1B are 0 here (RunStage starts each stage from slot 0).
    template <class TensorAImag, class TensorBReal, class CopyGmToL1A, class CopyGmToL1BReal>
    CATLASS_DEVICE void IssueWarmup(
        TensorAImag& tensorAImag, TensorBReal& tensorBReal, CopyGmToL1A& copyGmToL1A, CopyGmToL1BReal& copyGmToL1BReal,
        uint32_t kTileCount, GemmCoord const& actualShape)
    {
        uint32_t firstKTileIdx = GetShuffleKTileIndex(0, kTileCount);
        uint32_t kActual0 = GetKActual(firstKTileIdx, kTileCount, actualShape.k());
        constexpr uint32_t curL1A = 0;
        constexpr uint32_t curL1B = 0;
        {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[curL1A]);
            auto tensorL1A = tla::MakeTensor(l1ATensorList[curL1A], L1A_LAYOUT, Arch::PositionL1{});
            auto tensorTileA = GetTileA(tensorAImag, 0, firstKTileIdx * L1_TILE_K, actualShape.m(), kActual0);
            copyGmToL1A(tensorL1A, tensorTileA);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[curL1A]);
        }
        {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[curL1B]);
            auto tensorL1B = tla::MakeTensor(l1BTensorList[curL1B], L1B_LAYOUT, Arch::PositionL1{});
            auto tensorTileB = GetTile(
                tensorBReal, tla::MakeCoord(firstKTileIdx * L1_TILE_K, 0), tla::MakeShape(kActual0, actualShape.n()));
            copyGmToL1BReal(tensorL1B, tensorTileB);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[curL1B]);
        }
    }

    /// Run one stage (C_real or C_imag) with 2*K_tileCount alternating sub-iterations.
    ///
    /// C_real stage:  even sub -> A_R * B_R,   odd sub -> A_I * (-B_I) or (-A_I) * B_I
    /// C_imag stage:  even sub -> A_I * B_R,   odd sub -> A_R * B_I
    template <
        class TensorAReal, class TensorAImag, class TensorAImagNeg, class TensorBReal, class TensorBImag,
        class TensorBImagNeg, class TensorL0C, class CopyGmToL1A, class CopyGmToL1BReal, class CopyGmToL1BImag,
        class CopyL1ToL0A, class CopyL1ToL0B>
    CATLASS_DEVICE void RunStage(
        bool isRealStage, bool skipWarmup, TensorAReal& tensorAReal, TensorAImag& tensorAImag,
        TensorAImagNeg& tensorAImagNeg, TensorBReal& tensorBReal, TensorBImag& tensorBImag,
        TensorBImagNeg& tensorBImagNeg, TensorL0C& tensorL0C, CopyGmToL1A& copyGmToL1A,
        CopyGmToL1BReal& copyGmToL1BReal, CopyGmToL1BImag& copyGmToL1BImag, CopyL1ToL0A& copyL1ToL0A,
        CopyL1ToL0B& copyL1ToL0B, uint32_t kTileCount, uint32_t mRound, uint32_t nRound, GemmCoord const& actualShape)
    {
        uint32_t mPartLoop = CeilDiv<L0_TILE_M>(mRound);
        uint32_t nPartLoop = CeilDiv<L0_TILE_N>(nRound);
        uint32_t totalSubs = 2 * kTileCount;

        uint32_t curL1A = 0;
        uint32_t curL1B = 0;

        // ---- Warmup: load first sub-iteration ----
        // C_real even: A_R[0], B_R[0];  C_imag even: A_I[0], B_R[0]
        // When skipWarmup=true, the caller has already issued these MTE2 loads
        // (and the matching SetFlag<MTE2_MTE1>) so we go straight to the main loop.
        if (!skipWarmup) {
            uint32_t firstKTileIdx = GetShuffleKTileIndex(0, kTileCount);
            uint32_t kActual0 = GetKActual(firstKTileIdx, kTileCount, actualShape.k());
            {
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[curL1A]);
                auto tensorL1A = tla::MakeTensor(l1ATensorList[curL1A], L1A_LAYOUT, Arch::PositionL1{});
                auto tensorTileA = isRealStage ?
                                       GetTileA(tensorAReal, 0, firstKTileIdx * L1_TILE_K, actualShape.m(), kActual0) :
                                       GetTileA(tensorAImag, 0, firstKTileIdx * L1_TILE_K, actualShape.m(), kActual0);
                copyGmToL1A(tensorL1A, tensorTileA);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[curL1A]);
            }
            {
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[curL1B]);
                auto tensorL1B = tla::MakeTensor(l1BTensorList[curL1B], L1B_LAYOUT, Arch::PositionL1{});
                auto tensorTileB = GetTile(
                    tensorBReal, tla::MakeCoord(firstKTileIdx * L1_TILE_K, 0),
                    tla::MakeShape(kActual0, actualShape.n()));
                copyGmToL1BReal(tensorL1B, tensorTileB);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[curL1B]);
            }
        }

        // ---- Main sub-iteration loop ----
        for (uint32_t subIdx = 0; subIdx < totalSubs; subIdx++) {
            uint32_t nextL1A = (curL1A + 1 < STAGES) ? (curL1A + 1) : 0;
            uint32_t nextL1B = (curL1B + 1 < STAGES) ? (curL1B + 1) : 0;

            bool isEvenSub = (subIdx % 2 == 0);
            uint32_t kTileIdx = subIdx / 2;
            uint32_t shuffleKTileIdx = GetShuffleKTileIndex(kTileIdx, kTileCount);

            // ---- Preload next sub-iteration ----
            if (subIdx < totalSubs - 1) {
                uint32_t nextSubIdx = subIdx + 1;
                bool nextIsEven = (nextSubIdx % 2 == 0);
                uint32_t nextKTileIdx = nextSubIdx / 2;
                uint32_t nextShuffleKTileIdx = GetShuffleKTileIndex(nextKTileIdx, kTileCount);

                uint32_t kActualNext = GetKActual(nextShuffleKTileIdx, kTileCount, actualShape.k());

                // ---- Preload next A ----
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[nextL1A]);
                {
                    auto tensorL1A = tla::MakeTensor(l1ATensorList[nextL1A], L1A_LAYOUT, Arch::PositionL1{});
                    auto tensorTileA = GetNextTileA(
                        isRealStage, nextIsEven, tensorAReal, tensorAImag, tensorAImagNeg,
                        nextShuffleKTileIdx * L1_TILE_K, actualShape.m(), kActualNext);
                    copyGmToL1A(tensorL1A, tensorTileA);
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[nextL1A]);

                // ---- Preload next B ----
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[nextL1B]);
                {
                    auto tensorL1B = tla::MakeTensor(l1BTensorList[nextL1B], L1B_LAYOUT, Arch::PositionL1{});
                    auto tensorTileB = GetNextTileB(
                        isRealStage, nextIsEven, tensorBReal, tensorBImag, tensorBImagNeg,
                        nextShuffleKTileIdx * L1_TILE_K, actualShape.n(), kActualNext);
                    if (nextIsEven) {
                        copyGmToL1BReal(tensorL1B, tensorTileB);
                    } else {
                        copyGmToL1BImag(tensorL1B, tensorTileB);
                    }
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[nextL1B]);
            }

            // ---- Compute inner loops ----
            uint32_t kActual = GetKActual(shuffleKTileIdx, kTileCount, actualShape.k());
            uint32_t kPartLoop = CeilDiv<L0_TILE_K>(kActual);

            uint32_t l0AListId = 0;
            uint32_t l0BListId = 0;

            auto tensorL1ACur = tla::MakeTensor(l1ATensorList[curL1A], L1A_LAYOUT, Arch::PositionL1{});
            auto tensorL1BCur = tla::MakeTensor(l1BTensorList[curL1B], L1B_LAYOUT, Arch::PositionL1{});

            for (uint32_t mPartIdx = 0; mPartIdx < mPartLoop; mPartIdx++) {
                uint32_t mPartActual = (mPartIdx < mPartLoop - 1) ? L0_TILE_M : (mRound - mPartIdx * L0_TILE_M);

                for (uint32_t kPartIdx = 0; kPartIdx < kPartLoop; kPartIdx++) {
                    uint32_t kPartActual = (kPartIdx < kPartLoop - 1) ? L0_TILE_K : (kActual - kPartIdx * L0_TILE_K);

                    // ---- Load L0A ----
                    auto layoutAInL0 = tla::MakeLayout<ElementA, LayoutTagL0A>(mPartActual, kPartActual);
                    auto tensorL0A = tla::MakeTensor(l0ATensorList[l0AListId], layoutAInL0, Arch::PositionL0A{});
                    auto tensorTileL1A =
                        GetTileA(tensorL1ACur, mPartIdx * L0_TILE_M, kPartIdx * L0_TILE_K, mPartActual, kPartActual);

                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);
                    if ((mPartIdx == 0) && (kPartIdx == 0)) {
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[curL1A]);
                    }
                    copyL1ToL0A(tensorL0A, tensorTileL1A);
                    if ((mPartIdx == mPartLoop - 1) && (kPartIdx == kPartLoop - 1)) {
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[curL1A]);
                    }

                    for (uint32_t nPartIdx = 0; nPartIdx < nPartLoop; nPartIdx++) {
                        uint32_t nPartActual = (nPartIdx < nPartLoop - 1) ? L0_TILE_N : (nRound - nPartIdx * L0_TILE_N);

                        auto layoutBInL0 = tla::MakeLayout<ElementB, LayoutTagL0B>(kPartActual, nPartActual);
                        auto tensorL0B = tla::MakeTensor(l0BTensorList[l0BListId], layoutBInL0, Arch::PositionL0B{});
                        auto tensorTileL1B = GetTile(
                            tensorL1BCur, tla::MakeCoord(kPartIdx * L0_TILE_K, nPartIdx * L0_TILE_N),
                            tla::MakeShape(kPartActual, nPartActual));

                        // initC: first sub-iteration and first kPart of that sub
                        bool isInitC = (subIdx == 0) && (kPartIdx == 0);
                        bool isFirstMicro = (mPartIdx == 0) && (kPartIdx == 0) && (nPartIdx == 0);
                        bool isLastMicro =
                            (mPartIdx == mPartLoop - 1) && (kPartIdx == kPartLoop - 1) && (nPartIdx == nPartLoop - 1);

                        // ---- Load L0B ----
                        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BListId]);
                        if (isFirstMicro) {
                            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[curL1B]);
                        }
                        copyL1ToL0B(tensorL0B, tensorTileL1B);
                        if (isLastMicro) {
                            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[curL1B]);
                        }
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);

                        // ---- MMAD: L0A * L0B -> l0C ----
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
                        {
                            auto tensorTileL0C = GetTile(
                                tensorL0C, tla::MakeCoord(mPartIdx * L0_TILE_M, nPartIdx * L0_TILE_N),
                                tla::MakeShape(mPartActual, nPartActual));
                            tileMmad(
                                tensorTileL0C, tensorL0A, tensorL0B, mPartActual, nPartActual, kPartActual, isInitC,
                                /*unitFlag=*/0);
                        }
                        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BListId]);
                        l0BListId = (l0BListId + 1 < STAGES) ? (l0BListId + 1) : 0;
                    }
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);
                    l0AListId = (l0AListId + 1 < STAGES) ? (l0AListId + 1) : 0;
                }
            }

            curL1A = nextL1A;
            curL1B = nextL1B;
        }
    }

    CATLASS_DEVICE
    uint32_t GetShuffleKTileIndex(uint32_t kTileIdx, uint32_t kTileCount)
    {
        if constexpr (ENABLE_SHUFFLE_K) {
            return (AscendC::GetBlockIdx() + kTileIdx) % kTileCount;
        } else {
            return kTileIdx;
        }
    }

    CATLASS_DEVICE
    uint32_t GetKActual(uint32_t kTileIdx, uint32_t kTileCount, uint32_t kExtent)
    {
        return (kTileIdx < kTileCount - 1) ? L1_TILE_K : (kExtent - kTileIdx * L1_TILE_K);
    }

    /// Select GM source for A based on stage and even/odd sub-iteration.
    /// C_real: even -> A_real, odd -> AImagSigned (kernel pre-negated/aliased)
    /// C_imag: even -> A_imag (original), odd -> A_real
    template <class TensorAReal, class TensorAImag, class TensorAImagNeg>
    CATLASS_DEVICE auto GetNextTileA(
        bool isRealStage, bool isEvenSub, TensorAReal& tensorAReal, TensorAImag& tensorAImag,
        TensorAImagNeg& tensorAImagNeg, uint32_t kIndex, uint32_t mSize, uint32_t kSize)
    {
        if (isRealStage) {
            return GetTileA(isEvenSub ? tensorAReal : tensorAImagNeg, 0, kIndex, mSize, kSize);
        } else {
            return GetTileA(isEvenSub ? tensorAImag : tensorAReal, 0, kIndex, mSize, kSize);
        }
    }

    /// Select GM source for B based on stage and even/odd sub-iteration.
    /// C_real: even -> B_real, odd -> BImagSigned (kernel pre-negated/aliased)
    /// C_imag: even -> B_real, odd -> B_imag (original)
    template <class TensorBReal, class TensorBImag, class TensorBImagNeg>
    CATLASS_DEVICE auto GetNextTileB(
        bool isRealStage, bool isEvenSub, TensorBReal& tensorBReal, TensorBImag& tensorBImag,
        TensorBImagNeg& tensorBImagNeg, uint32_t kIndex, uint32_t nSize, uint32_t kSize)
    {
        if (isEvenSub) {
            return GetTile(tensorBReal, tla::MakeCoord(kIndex, 0), tla::MakeShape(kSize, nSize));
        } else if (isRealStage) {
            // C_real odd sub: cross term uses the kernel-signed B operand.
            return GetTile(tensorBImagNeg, tla::MakeCoord(kIndex, 0), tla::MakeShape(kSize, nSize));
        } else {
            // C_imag odd sub: original B_imag
            return GetTile(tensorBImag, tla::MakeCoord(kIndex, 0), tla::MakeShape(kSize, nSize));
        }
    }

    template <class TensorA>
    CATLASS_DEVICE auto GetTileA(TensorA& tensorA, uint32_t mIndex, uint32_t kIndex, uint32_t mSize, uint32_t kSize)
    {
        if constexpr (tla::detail::isVector<LayoutA>::value) {
            return GetTile(tensorA, tla::MakeCoord(kIndex), tla::MakeShape(kSize));
        } else {
            return GetTile(tensorA, tla::MakeCoord(mIndex, kIndex), tla::MakeShape(mSize, kSize));
        }
    }

protected:
    // L1 tensors: generic A + B K-pingpong (4 slots)
    AscendC::LocalTensor<ElementA> l1ATensorList[STAGES];
    AscendC::LocalTensor<ElementB> l1BTensorList[STAGES];

    // L0 tensors: dual-buffer pingpong
    AscendC::LocalTensor<ElementA> l0ATensorList[STAGES];
    AscendC::LocalTensor<ElementB> l0BTensorList[STAGES];

    // L0C - single buffer, time-multiplexed between C_real and C_imag
    AscendC::LocalTensor<ElementAccumulator> l0CTensor;

    // Event IDs
    int32_t l1AEventList[STAGES]; // {0, 1}
    int32_t l1BEventList[STAGES]; // {2, 3}
    int32_t l0AEventList[STAGES]; // {0, 1}
    int32_t l0BEventList[STAGES]; // {2, 3}

    // Tile compute functor
    TileMmad tileMmad;
};

} // namespace Catlass::Gemm::Block

#endif // CATLASS_GEMM_BLOCK_BLOCK_MMAD_PLANAR_COMPLEX_FUSED_TLA_HPP
