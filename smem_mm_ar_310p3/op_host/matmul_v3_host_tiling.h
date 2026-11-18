/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE-CANN-2.0 in the root of the software repository for the full text of the License.
 */
// Adapted from CANN ops-nn v9.1.0 MatMulV3 for Ascend 310P only.

#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
// 310P MatMulV3 CalL1TilingV200 / DoL2CacheTiling310P specialized for
// FP16 ND x NZ(transB) -> ND, K=N=2048 and eight producer cores.
// A 512-row communication-ready slab is exactly one 4x2 M/N wave.
template <typename T>
bool BuildMatmulV3Tiling(gert::TilingContext* context, T& td)
{
    auto* info = context->GetPlatformInfo();
    if (info == nullptr) return false;
    auto platform = platform_ascendc::PlatformAscendC(info);
    uint64_t l1 = 0, l0a = 0, l0b = 0, l0c = 0, ub = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, l0a);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, l0b);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0c);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
    // 6*128*64*2 + 6*256*64*2 = 288 KiB L1.
    // Reserve a dedicated 256-byte ready/trace buffer on each producer.
    if (platform.GetCoreNumAic() < 8 || l1 < 294912 ||
        l0a < 32768 || l0b < 65536 || l0c < 131072 ||
        ub <= 131072 + 256) return false;
    matmul_tiling::MultiCoreMatmulTiling mm;
    using namespace matmul_tiling;
    mm.SetAType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16, false);
    mm.SetBType(TPosition::GM, CubeFormat::NZ, DataType::DT_FLOAT16, true);
    mm.SetCType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16);
    mm.SetDim(8);
    mm.SetShape(td.rows, 2048, 2048);
    mm.SetOrgShape(td.rows, 2048, 2048);
    mm.SetBias(false);
    mm.SetBufferSpace(l1, l0c, ub - 256);
    if (mm.GetTiling(td.mmTiling) == -1) return false;
    auto& t = td.mmTiling;
    t.usedCoreNum = 8;
    t.singleCoreM = t.baseM = 128;
    t.singleCoreN = t.baseN = 256;
    t.singleCoreK = 2048;
    t.baseK = 64;
    t.depthA1 = t.depthB1 = 6;
    t.stepM = t.stepN = 1;
    t.stepKa = t.stepKb = 3;
    t.iterateOrder = 0; // ITER_COL_FIRST in MatMulV3
    t.dbL0A = t.dbL0B = 2;
    t.dbL0C = 1; // fixed for the 310P producer profile
    t.transLength = 131072;
    t.shareUbSize = 0;
    td.mmMBlock = 4;
    td.mmNBlock = 2;
    return true;
}
} // namespace
