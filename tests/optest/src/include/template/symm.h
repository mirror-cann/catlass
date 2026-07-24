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

#ifndef OPTEST_SYMM_H
#define OPTEST_SYMM_H

#include <cmath>

#include <torch/torch.h>
#include <tiling/platform/platform_ascendc.h>

#include "catlass_kernel_jit.h"
#include "common/run_npu_func.h"
#include "torch_utils.h"

namespace CatlassKernelWrapper {

using SymmKernelFn =
    void (*)(const uint32_t, aclrtStream, const CatlassKernel::TParams&, const CatlassKernel::SymmParams&);

template <SymmKernelFn KernelFunc>
struct SymmLike {
    using OutputType = at::Tensor;

    static OutputType Run(
        const at::Tensor& mat1, const at::Tensor& mat2, const c10::ScalarType& outDType, int64_t side, int64_t uplo,
        double alpha)
    {
        TORCH_CHECK(mat1.dim() == 2 && mat2.dim() == 2, "symm expects two 2-D tensors");
        TORCH_CHECK(side == 0 || side == 1, "symm side must be 0 (left) or 1 (right)");
        TORCH_CHECK(uplo == 0 || uplo == 1, "symm uplo must be 0 (lower) or 1 (upper)");
        TORCH_CHECK(std::fabs(alpha - 1.0) < 1e-6, "symm currently supports only alpha=1.0");
        TORCH_CHECK(
            mat1.scalar_type() == at::kFloat && mat2.scalar_type() == at::kFloat,
            "symm currently supports float32 inputs");
        TORCH_CHECK(outDType == at::kFloat, "symm currently supports float32 output");

        CatlassKernel::TParams tParams;
        CatlassKernel::SymmParams params;

        tParams.element["A"] = TorchDtypeToAclDtype(mat1.scalar_type());
        tParams.element["B"] = TorchDtypeToAclDtype(mat2.scalar_type());
        tParams.element["C"] = TorchDtypeToAclDtype(outDType);
        tParams.transpose["A"] = false;
        tParams.transpose["B"] = false;
        tParams.transpose["C"] = false;
        tParams.useNz["A"] = false;
        tParams.useNz["B"] = false;
        tParams.useNz["C"] = false;

        if (side == 0) {
            TORCH_CHECK(mat1.size(0) == mat1.size(1), "left symm expects mat1 to be square");
            TORCH_CHECK(mat1.size(1) == mat2.size(0), "left symm shape mismatch");
            params.m = static_cast<uint32_t>(mat1.size(0));
            params.k = static_cast<uint32_t>(mat1.size(1));
            params.n = static_cast<uint32_t>(mat2.size(1));
        } else {
            TORCH_CHECK(mat2.size(0) == mat2.size(1), "right symm expects mat2 to be square");
            TORCH_CHECK(mat1.size(1) == mat2.size(0), "right symm shape mismatch");
            params.m = static_cast<uint32_t>(mat1.size(0));
            params.k = static_cast<uint32_t>(mat2.size(0));
            params.n = static_cast<uint32_t>(mat2.size(1));
        }

        params.side = static_cast<uint32_t>(side);
        params.uplo = static_cast<uint32_t>(uplo);
        params.alpha = static_cast<float>(alpha);
        params.inputAddr.resize(2);
        params.inputAddr[0] = static_cast<uint8_t*>(const_cast<void*>(mat1.storage().data()));
        params.inputAddr[1] = static_cast<uint8_t*>(const_cast<void*>(mat2.storage().data()));

        OutputType output = GetOutputTensor({params.m, params.n}, AclDtypeToTorchDtype(tParams.elem("C")));
        params.outputAddr.resize(1);
        params.outputAddr[0] = static_cast<uint8_t*>(const_cast<void*>(output.storage().data()));

        aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
        uint32_t coreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
        RUN_NPU_FUNC(KernelFunc, coreNum, stream, tParams, params);
        return output;
    }
};

} // namespace CatlassKernelWrapper

#endif // OPTEST_SYMM_H
