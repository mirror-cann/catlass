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

#include "catlass_kernel.h"
#include "jit_compiler.h"
#include "jit_macro_generator.h"

#include <iostream>

namespace CatlassKernel {

/**
 * @brief example 75_symm: Resolve and launch the JIT-specialized SYMM implementation.
 */
extern "C" void Symm(const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const SymmParams& params)
{
    auto macros = JitMacroGenerator<TParams>::generate("symm", tParams);
    macros["CATLASS_JIT_SYMM_SIDE"] = std::to_string(params.side);
    macros["CATLASS_JIT_SYMM_FILL"] = std::to_string(params.uplo);
    macros["CATLASS_JIT_BLOCK_SCHEDULER"] = "30";

    auto* entry = JitCompiler::instance().getKernel("symm_impl.cpp", macros, JitKernelType::MIX);
    if (entry == nullptr) {
        std::cerr << "[ERROR] Failed to resolve JIT kernel: symm_impl.cpp" << std::endl;
        return;
    }
    entry(blockNum, stream, &params);
    aclrtSynchronizeStream(stream);
}

} // namespace CatlassKernel
