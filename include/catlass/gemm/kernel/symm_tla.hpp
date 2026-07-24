/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_KERNEL_SYMM_TLA_HPP
#define CATLASS_GEMM_KERNEL_SYMM_TLA_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

namespace Catlass::Gemm::Kernel {

enum class SymmMatmulSide
{
    LEFT,
    RIGHT,
};

enum class SymmMatmulFillMode
{
    LOWER,
    UPPER,
};

/// TLA-style symmetric matrix multiplication kernel.
template <SymmMatmulSide Side_, SymmMatmulFillMode FillMode_, class BlockMmad_, class BlockScheduler_>
class SymmMatmulTlaSingleKernelProducer {
public:
    static constexpr SymmMatmulSide Side = Side_;
    static constexpr SymmMatmulFillMode FillMode = FillMode_;
    static constexpr bool UPPER_STORAGE = (FillMode == SymmMatmulFillMode::UPPER);
    using BlockMmad = BlockMmad_;
    using ArchTag = typename BlockMmad::ArchTag;
    using L1TileShape = typename BlockMmad::L1TileShape;
    using ElementA = typename BlockMmad::ElementA;
    using ElementB = typename BlockMmad::ElementB;
    using ElementC = typename BlockMmad::ElementC;
    using LayoutC = typename BlockMmad::LayoutC;

private:
    // Traits to extract side-specific layout types without triggering invalid-type errors
    template <SymmMatmulSide, class>
    struct SymmLayoutTraits;

    template <class B>
    struct SymmLayoutTraits<SymmMatmulSide::LEFT, B> {
        using LayoutSymP = typename B::LayoutAP;
        using LayoutSymQ = typename B::LayoutAQ;
        using LayoutNonSym = typename B::LayoutB;
    };

    template <class B>
    struct SymmLayoutTraits<SymmMatmulSide::RIGHT, B> {
        using LayoutSymP = typename B::LayoutBP;
        using LayoutSymQ = typename B::LayoutBQ;
        using LayoutNonSym = typename B::LayoutA;
    };

    using LayoutTraits = SymmLayoutTraits<Side, BlockMmad>;

public:
    // Unified layout aliases: Sym = symmetric operand, NonSym = non-symmetric operand
    using LayoutSymP = typename LayoutTraits::LayoutSymP;
    using LayoutSymQ = typename LayoutTraits::LayoutSymQ;
    using LayoutNonSym = typename LayoutTraits::LayoutNonSym;

    using BlockScheduler = BlockScheduler_;

    static constexpr uint32_t L1_TILE_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t L1_TILE_N = tla::get<1>(L1TileShape{});
    static constexpr uint32_t L1_TILE_K = tla::get<2>(L1TileShape{});

    struct Params {
        GemmCoord problemShape;
        GM_ADDR ptrA;
        GM_ADDR ptrB;
        GM_ADDR ptrC;
        LayoutSymP layoutSymP;
        LayoutSymQ layoutSymQ;
        LayoutNonSym layoutNonSym;
        LayoutC layoutC;
        uint32_t symmTiles;

        CATLASS_HOST_DEVICE
        Params()
        {}

        CATLASS_HOST_DEVICE
        Params(
            GemmCoord const& problemShape_, GM_ADDR ptrA_, LayoutSymP layoutSymP_, LayoutSymQ layoutSymQ_,
            GM_ADDR ptrB_, LayoutNonSym layoutNonSym_, GM_ADDR ptrC_, LayoutC layoutC_, uint32_t symmTiles_)
            : problemShape(problemShape_),
              ptrA(ptrA_),
              ptrB(ptrB_),
              ptrC(ptrC_),
              layoutSymP(layoutSymP_),
              layoutSymQ(layoutSymQ_),
              layoutNonSym(layoutNonSym_),
              layoutC(layoutC_),
              symmTiles(symmTiles_)
        {}
    };

    struct Arguments {
        GemmCoord problemShape;
        uint8_t* ptrA;
        LayoutSymP layoutSymP;
        LayoutSymQ layoutSymQ;
        uint8_t* ptrB;
        LayoutNonSym layoutNonSym;
        uint8_t* ptrC;
        LayoutC layoutC;
    };

    static bool CanImplement(const Arguments& args)
    {
        if constexpr (Side == SymmMatmulSide::LEFT) {
            return (args.problemShape.m() == args.problemShape.k()) && (L1_TILE_M == L1_TILE_K);
        } else {
            return (args.problemShape.k() == args.problemShape.n()) && (L1_TILE_K == L1_TILE_N);
        }
    }

    static size_t GetWorkspaceSize(const Arguments&)
    {
        return 0;
    }

    static Params ToUnderlyingArguments(const Arguments& args, uint8_t*)
    {
        uint32_t symmTiles = 0;
        if constexpr (Side == SymmMatmulSide::LEFT) {
            symmTiles = CeilDiv(args.problemShape.m(), L1_TILE_M);
        } else {
            symmTiles = CeilDiv(args.problemShape.n(), L1_TILE_N);
        }
        return Params{args.problemShape, args.ptrA, args.layoutSymP, args.layoutSymQ, args.ptrB,
                      args.layoutNonSym, args.ptrC, args.layoutC,    symmTiles};
    }

    CATLASS_DEVICE
    SymmMatmulTlaSingleKernelProducer()
    {}

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(Params const& params);

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(Params const& params)
    {
        if constexpr (Side == SymmMatmulSide::LEFT) {
            static_assert(L1_TILE_M == L1_TILE_K, "Left-symm path requires square M/K tiles.");
        } else {
            static_assert(L1_TILE_K == L1_TILE_N, "Right-symm path requires square K/N tiles.");
        }

        BlockScheduler scheduler(params.problemShape, MakeCoord(L1_TILE_M, L1_TILE_N));
        uint32_t coreLoops = scheduler.GetCoreLoops();

        Arch::Resource<ArchTag> resource;
        BlockMmad blockMmad(resource);

        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer((__gm__ ElementA*)params.ptrA);
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer((__gm__ ElementB*)params.ptrB);
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer((__gm__ ElementC*)params.ptrC);

        auto tensorC = tla::MakeTensor(gmC, params.layoutC, Arch::PositionGM{});

        if constexpr (Side == SymmMatmulSide::LEFT) {
            auto tensorSym = tla::MakeTensor(gmA, params.layoutSymP, Arch::PositionGM{});
            auto tensorSymQ = tla::MakeTensor(gmA, params.layoutSymQ, Arch::PositionGM{});
            auto tensorNonSym = tla::MakeTensor(gmB, params.layoutNonSym, Arch::PositionGM{});

            for (uint32_t loopIdx = AscendC::GetBlockIdx(); loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
                GemmCoord blockCoord = scheduler.GetBlockCoord(loopIdx);
                uint32_t iTile = blockCoord.m();
                uint32_t jTile = blockCoord.n();

                uint32_t mRemain = params.problemShape.m() - iTile * L1_TILE_M;
                uint32_t nRemain = params.problemShape.n() - jTile * L1_TILE_N;
                uint32_t mActual = (mRemain < L1_TILE_M) ? mRemain : L1_TILE_M;
                uint32_t nActual = (nRemain < L1_TILE_N) ? nRemain : L1_TILE_N;
                GemmCoord actualBlockShape{
                    static_cast<GemmCoord::Index>(mActual), static_cast<GemmCoord::Index>(nActual),
                    static_cast<GemmCoord::Index>(params.problemShape.k())};
                uint32_t tileIdx = (Side == SymmMatmulSide::LEFT) ? iTile : jTile;

                // ── Cross-block preload ──
                bool isFirstBlock = (loopIdx == AscendC::GetBlockIdx());
                bool hasNextBlock = false;
                uint32_t iTileNext = 0;
                uint32_t jTileNext = 0;
                GemmCoord actualShapeNext;
                if (loopIdx + AscendC::GetBlockNum() < coreLoops) {
                    hasNextBlock = true;
                    GemmCoord nextBlockCoord = scheduler.GetBlockCoord(loopIdx + AscendC::GetBlockNum());
                    iTileNext = nextBlockCoord.m();
                    jTileNext = nextBlockCoord.n();
                    uint32_t mRemainNext = params.problemShape.m() - iTileNext * L1_TILE_M;
                    uint32_t nRemainNext = params.problemShape.n() - jTileNext * L1_TILE_N;
                    uint32_t mActualNext = (mRemainNext < L1_TILE_M) ? mRemainNext : L1_TILE_M;
                    uint32_t nActualNext = (nRemainNext < L1_TILE_N) ? nRemainNext : L1_TILE_N;
                    actualShapeNext = GemmCoord{
                        static_cast<GemmCoord::Index>(mActualNext), static_cast<GemmCoord::Index>(nActualNext),
                        static_cast<GemmCoord::Index>(params.problemShape.k())};
                }

                auto tensorBlockC = GetTile(
                    tensorC, tla::MakeCoord(iTile * L1_TILE_M, jTile * L1_TILE_N),
                    tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));

                if constexpr (Side == SymmMatmulSide::LEFT) {
                    auto tensorBlockSym = GetTile(
                        tensorSym, tla::MakeCoord(iTile * L1_TILE_M, 0),
                        tla::MakeShape(actualBlockShape.m(), actualBlockShape.k()));
                    auto tensorBlockSymQ = GetTile(
                        tensorSymQ, tla::MakeCoord(iTile * L1_TILE_M, 0),
                        tla::MakeShape(actualBlockShape.m(), actualBlockShape.k()));
                    auto tensorBlockNonSym = GetTile(
                        tensorNonSym, tla::MakeCoord(0, jTile * L1_TILE_N),
                        tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));

                    decltype(tensorBlockSym) tensorBlockSym_Next{};
                    decltype(tensorBlockSymQ) tensorBlockSymQ_Next{};
                    decltype(tensorBlockNonSym) tensorBlockNonSym_Next{};
                    decltype(tensorBlockSym)* ptrSymNext = nullptr;
                    decltype(tensorBlockSymQ)* ptrSymQNext = nullptr;
                    decltype(tensorBlockNonSym)* ptrNonSymNext = nullptr;
                    if (hasNextBlock) {
                        tensorBlockSym_Next = GetTile(
                            tensorSym, tla::MakeCoord(iTileNext * L1_TILE_M, 0),
                            tla::MakeShape(actualShapeNext.m(), actualShapeNext.k()));
                        tensorBlockSymQ_Next = GetTile(
                            tensorSymQ, tla::MakeCoord(iTileNext * L1_TILE_M, 0),
                            tla::MakeShape(actualShapeNext.m(), actualShapeNext.k()));
                        tensorBlockNonSym_Next = GetTile(
                            tensorNonSym, tla::MakeCoord(0, jTileNext * L1_TILE_N),
                            tla::MakeShape(actualShapeNext.k(), actualShapeNext.n()));
                        ptrSymNext = &tensorBlockSym_Next;
                        ptrSymQNext = &tensorBlockSymQ_Next;
                        ptrNonSymNext = &tensorBlockNonSym_Next;
                    }

                    uint32_t tileIdxNext = (Side == SymmMatmulSide::LEFT) ? iTileNext : jTileNext;

                    blockMmad.template operator()<UPPER_STORAGE>(
                        tensorBlockSym, tensorBlockSymQ, tensorBlockNonSym, tensorBlockC, actualBlockShape, tileIdx,
                        tileIdxNext, params.symmTiles, isFirstBlock, hasNextBlock, ptrSymNext, ptrSymQNext,
                        ptrNonSymNext, actualShapeNext);
                } else {
                    // RIGHT: A is non-symmetric (tensorNonSym), B is symmetric (tensorSym)
                    auto tensorBlockNonSym = GetTile(
                        tensorNonSym, tla::MakeCoord(iTile * L1_TILE_M, 0),
                        tla::MakeShape(actualBlockShape.m(), actualBlockShape.k()));
                    auto tensorBlockSym = GetTile(
                        tensorSym, tla::MakeCoord(0, jTile * L1_TILE_N),
                        tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));
                    auto tensorBlockSymQ = GetTile(
                        tensorSymQ, tla::MakeCoord(0, jTile * L1_TILE_N),
                        tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));

                    // Next-block preload views
                    decltype(tensorBlockNonSym) tensorBlockNonSym_Next{};
                    decltype(tensorBlockSym) tensorBlockSym_Next{};
                    decltype(tensorBlockSymQ) tensorBlockSymQ_Next{};
                    decltype(tensorBlockNonSym)* ptrNonSymNext = nullptr;
                    decltype(tensorBlockSym)* ptrSymNext = nullptr;
                    decltype(tensorBlockSymQ)* ptrSymQNext = nullptr;
                    if (hasNextBlock) {
                        tensorBlockNonSym_Next = GetTile(
                            tensorNonSym, tla::MakeCoord(iTileNext * L1_TILE_M, 0),
                            tla::MakeShape(actualShapeNext.m(), actualShapeNext.k()));
                        tensorBlockSym_Next = GetTile(
                            tensorSym, tla::MakeCoord(0, jTileNext * L1_TILE_N),
                            tla::MakeShape(actualShapeNext.k(), actualShapeNext.n()));
                        tensorBlockSymQ_Next = GetTile(
                            tensorSymQ, tla::MakeCoord(0, jTileNext * L1_TILE_N),
                            tla::MakeShape(actualShapeNext.k(), actualShapeNext.n()));
                        ptrNonSymNext = &tensorBlockNonSym_Next;
                        ptrSymNext = &tensorBlockSym_Next;
                        ptrSymQNext = &tensorBlockSymQ_Next;
                    }

                    uint32_t tileIdxNext = (Side == SymmMatmulSide::LEFT) ? iTileNext : jTileNext;

                    blockMmad.template operator()<UPPER_STORAGE>(
                        tensorBlockNonSym, tensorBlockSym, tensorBlockSymQ, tensorBlockC, actualBlockShape, tileIdx,
                        tileIdxNext, params.symmTiles, isFirstBlock, hasNextBlock, ptrNonSymNext, ptrSymNext,
                        ptrSymQNext, actualShapeNext);
                }

                blockMmad.Store(tensorBlockC, actualBlockShape);
            }
        } else {
            auto tensorNonSym = tla::MakeTensor(gmA, params.layoutNonSym, Arch::PositionGM{});
            auto tensorSym = tla::MakeTensor(gmB, params.layoutSymP, Arch::PositionGM{});
            auto tensorSymQ = tla::MakeTensor(gmB, params.layoutSymQ, Arch::PositionGM{});

            for (uint32_t loopIdx = AscendC::GetBlockIdx(); loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
                GemmCoord blockCoord = scheduler.GetBlockCoord(loopIdx);
                uint32_t iTile = blockCoord.m();
                uint32_t jTile = blockCoord.n();

                uint32_t mRemain = params.problemShape.m() - iTile * L1_TILE_M;
                uint32_t nRemain = params.problemShape.n() - jTile * L1_TILE_N;
                uint32_t mActual = (mRemain < L1_TILE_M) ? mRemain : L1_TILE_M;
                uint32_t nActual = (nRemain < L1_TILE_N) ? nRemain : L1_TILE_N;
                GemmCoord actualBlockShape{
                    static_cast<GemmCoord::Index>(mActual), static_cast<GemmCoord::Index>(nActual),
                    static_cast<GemmCoord::Index>(params.problemShape.k())};
                uint32_t tileIdx = jTile;

                bool isFirstBlock = (loopIdx == AscendC::GetBlockIdx());
                bool hasNextBlock = false;
                uint32_t iTileNext = 0;
                uint32_t jTileNext = 0;
                GemmCoord actualShapeNext;
                if (loopIdx + AscendC::GetBlockNum() < coreLoops) {
                    hasNextBlock = true;
                    GemmCoord nextBlockCoord = scheduler.GetBlockCoord(loopIdx + AscendC::GetBlockNum());
                    iTileNext = nextBlockCoord.m();
                    jTileNext = nextBlockCoord.n();
                    uint32_t mRemainNext = params.problemShape.m() - iTileNext * L1_TILE_M;
                    uint32_t nRemainNext = params.problemShape.n() - jTileNext * L1_TILE_N;
                    uint32_t mActualNext = (mRemainNext < L1_TILE_M) ? mRemainNext : L1_TILE_M;
                    uint32_t nActualNext = (nRemainNext < L1_TILE_N) ? nRemainNext : L1_TILE_N;
                    actualShapeNext = GemmCoord{
                        static_cast<GemmCoord::Index>(mActualNext), static_cast<GemmCoord::Index>(nActualNext),
                        static_cast<GemmCoord::Index>(params.problemShape.k())};
                }

                auto tensorBlockC = GetTile(
                    tensorC, tla::MakeCoord(iTile * L1_TILE_M, jTile * L1_TILE_N),
                    tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));

                auto tensorBlockNonSym = GetTile(
                    tensorNonSym, tla::MakeCoord(iTile * L1_TILE_M, 0),
                    tla::MakeShape(actualBlockShape.m(), actualBlockShape.k()));
                auto tensorBlockSym = GetTile(
                    tensorSym, tla::MakeCoord(0, jTile * L1_TILE_N),
                    tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));
                auto tensorBlockSymQ = GetTile(
                    tensorSymQ, tla::MakeCoord(0, jTile * L1_TILE_N),
                    tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));

                decltype(tensorBlockNonSym) tensorBlockNonSym_Next{};
                decltype(tensorBlockSym) tensorBlockSym_Next{};
                decltype(tensorBlockSymQ) tensorBlockSymQ_Next{};
                decltype(tensorBlockNonSym)* ptrNonSymNext = nullptr;
                decltype(tensorBlockSym)* ptrSymNext = nullptr;
                decltype(tensorBlockSymQ)* ptrSymQNext = nullptr;
                if (hasNextBlock) {
                    tensorBlockNonSym_Next = GetTile(
                        tensorNonSym, tla::MakeCoord(iTileNext * L1_TILE_M, 0),
                        tla::MakeShape(actualShapeNext.m(), actualShapeNext.k()));
                    tensorBlockSym_Next = GetTile(
                        tensorSym, tla::MakeCoord(0, jTileNext * L1_TILE_N),
                        tla::MakeShape(actualShapeNext.k(), actualShapeNext.n()));
                    tensorBlockSymQ_Next = GetTile(
                        tensorSymQ, tla::MakeCoord(0, jTileNext * L1_TILE_N),
                        tla::MakeShape(actualShapeNext.k(), actualShapeNext.n()));
                    ptrNonSymNext = &tensorBlockNonSym_Next;
                    ptrSymNext = &tensorBlockSym_Next;
                    ptrSymQNext = &tensorBlockSymQ_Next;
                }

                uint32_t tileIdxNext = jTileNext;

                blockMmad.template operator()<UPPER_STORAGE>(
                    tensorBlockNonSym, tensorBlockSym, tensorBlockSymQ, tensorBlockC, actualBlockShape, tileIdx,
                    tileIdxNext, params.symmTiles, isFirstBlock, hasNextBlock, ptrNonSymNext, ptrSymNext, ptrSymQNext,
                    actualShapeNext);
                blockMmad.Store(tensorBlockC, actualBlockShape);
            }
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(Params const&)
    {}
};

} // namespace Catlass::Gemm::Kernel

#endif // CATLASS_GEMM_KERNEL_SYMM_TLA_HPP
