/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EXAMPLES_COMMON_GOLDEN_MATRIX_INVERSE_HPP
#define EXAMPLES_COMMON_GOLDEN_MATRIX_INVERSE_HPP

#include <cmath>
#include <limits>
#include <utility>
#include <vector>
#include <cstdint>

namespace Catlass::golden {

// Row-major LU decomposition with partial pivoting (analogous to LAPACK sgetrf).
// A is N x N in row-major; ipiv receives pivot indices (0-based).
// Returns 0 on success, or the row index (+1) where a zero pivot was encountered.
template <class Element>
int SgetrfRowMajor(int N, std::vector<Element>& A, std::vector<int32_t>& ipiv)
{
    ipiv.resize(N);

    // Pivot magnitude below this threshold is treated as zero (numerically singular).
    constexpr Element kPivotThreshold = Element(64) * std::numeric_limits<Element>::epsilon();

    for (int k = 0; k < N; ++k) {
        // Find pivot
        Element maxVal = std::fabs(A[k * N + k]);
        int maxRow = k;
        for (int i = k + 1; i < N; ++i) {
            Element absVal = std::fabs(A[i * N + k]);
            if (absVal > maxVal) {
                maxVal = absVal;
                maxRow = i;
            }
        }
        ipiv[k] = maxRow;

        if (maxVal <= kPivotThreshold) {
            // Numerically singular matrix
            return k + 1; // 1-based info
        }

        // Swap rows k and maxRow
        if (maxRow != k) {
            for (int j = 0; j < N; ++j) {
                std::swap(A[k * N + j], A[maxRow * N + j]);
            }
        }

        // Compute multipliers and update trailing submatrix
        Element invPivot = Element(1) / A[k * N + k];
        for (int i = k + 1; i < N; ++i) {
            A[i * N + k] *= invPivot; // store L factor
            Element factor = A[i * N + k];
            for (int j = k + 1; j < N; ++j) {
                A[i * N + j] -= factor * A[k * N + j];
            }
        }
    }

    return 0;
}

// Solve A * X = B using LU factors and pivot info from SgetrfRowMajor.
// A contains LU factors (unit lower L, upper U) in row-major.
// B is N x nrhs in row-major; result overwrites B.
// Returns 0 on success, or -1 if `trans` is a null pointer (invalid argument).
template <class Element>
int SgetrsRowMajor(
    const char* trans, int N, int nrhs, const std::vector<Element>& A, const std::vector<int32_t>& ipiv,
    std::vector<Element>& B)
{
    // Runtime error handling: a null `trans` is an invalid argument and must be
    // rejected explicitly (not via assert, which is stripped in release builds).
    if (trans == nullptr) {
        return -1;
    }
    bool notrans = (trans[0] == 'N' || trans[0] == 'n');

    if (notrans) {
        // Solve A * X = B
        // Step 1: Apply pivots to B (P * B)
        for (int k = 0; k < N; ++k) {
            int pivRow = ipiv[k];
            if (pivRow != k) {
                for (int j = 0; j < nrhs; ++j) {
                    std::swap(B[k * nrhs + j], B[pivRow * nrhs + j]);
                }
            }
        }

        // Step 2: Forward substitution L * Y = B  (L is unit lower triangular)
        for (int k = 0; k < N; ++k) {
            for (int i = k + 1; i < N; ++i) {
                Element factor = A[i * N + k];
                for (int j = 0; j < nrhs; ++j) {
                    B[i * nrhs + j] -= factor * B[k * nrhs + j];
                }
            }
        }

        // Step 3: Back substitution U * X = Y
        for (int k = N - 1; k >= 0; --k) {
            Element invDiag = Element(1) / A[k * N + k];
            for (int j = 0; j < nrhs; ++j) {
                B[k * nrhs + j] *= invDiag;
            }
            for (int i = 0; i < k; ++i) {
                Element factor = A[i * N + k];
                for (int j = 0; j < nrhs; ++j) {
                    B[i * nrhs + j] -= factor * B[k * nrhs + j];
                }
            }
        }
    } else {
        // Solve A^T * X = B (unused for inverse, but included for completeness)
        // Step 1: Forward with U^T
        for (int k = 0; k < N; ++k) {
            for (int j = 0; j < nrhs; ++j) {
                for (int i = 0; i < k; ++i) {
                    B[k * nrhs + j] -= A[i * N + k] * B[i * nrhs + j];
                }
                B[k * nrhs + j] /= A[k * N + k];
            }
        }

        // Step 2: Backward with L^T
        for (int k = N - 1; k >= 0; --k) {
            for (int j = 0; j < nrhs; ++j) {
                for (int i = k + 1; i < N; ++i) {
                    B[k * nrhs + j] -= A[i * N + k] * B[i * nrhs + j];
                }
            }
        }

        // Step 3: Apply inverse pivots
        for (int k = N - 1; k >= 0; --k) {
            int pivRow = ipiv[k];
            if (pivRow != k) {
                for (int j = 0; j < nrhs; ++j) {
                    std::swap(B[k * nrhs + j], B[pivRow * nrhs + j]);
                }
            }
        }
    }
    return 0;
}

// Compute the inverse of an N x N matrix in row-major layout.
// The input matrix is overwritten with its inverse.
// Returns 0 on success, or a positive value if the matrix is singular.
template <class Element>
int ComputeInverseInplace(int N, std::vector<Element>& A)
{
    // Step 1: LU decomposition with partial pivoting
    std::vector<int32_t> ipiv;
    int info = SgetrfRowMajor(N, A, ipiv);
    if (info != 0) {
        return info;
    }

    // Step 2: Form the N x N identity matrix in a work buffer
    std::vector<Element> work(N * N, Element(0));
    for (int i = 0; i < N; ++i) {
        work[i * N + i] = Element(1);
    }

    // Step 3: Solve A * X = I  =>  X = A^{-1}
    int solveInfo = SgetrsRowMajor("N", N, N, A, ipiv, work);
    if (solveInfo != 0) {
        return solveInfo;
    }

    // Step 4: Copy result back to A
    A = std::move(work);

    return 0;
}

} // namespace Catlass::golden

#endif // EXAMPLES_COMMON_GOLDEN_MATRIX_INVERSE_HPP
