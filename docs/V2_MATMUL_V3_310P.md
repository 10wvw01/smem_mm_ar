# V2: MatMulV3 310P producer replacement (experimental)

Baseline: smem_mm_ar main `9d8ee58632df79ee506f54116a7ce6b0468516c9` (AR flavor 10 and IAR flavor 11).
Upstream: [hicann/ops-nn v9.1.0](https://github.com/hicann/ops-nn/tree/v9.1.0/matmul/mat_mul_v3).
Sources: `mat_mul_base_kernel.h` blob c3d20980ac5ebf04669d48ff819dab1e4d181327;
`matmul_v3_base_tiling.cpp` blob 871de022cbf120d514cbacea5d62f08829e51ae6.
Derived code is subject to LICENSE-CANN-2.0; Ascend only.

## Scope
Both packages remain independent; identical producer and host helper are deliberately duplicated.
The 13 external attributes, ND x physical NZ(transB) -> ND, TP2, fixed K=N=2048,
M=1024..4096 step256, blockDim=15, pool offsets and SDMA transport remain unchanged.
This is a specialization, not a byte-identical stock MatMulV3 kernel or AOE tiling bank.
Host SDK seeds TCubeTiling; V200 ND path overrides base 128x256x64, depths 6,
steps K 3, singleCoreK 2048. Device uses MatmulImpl, GetMDLConfig(false,false,0,true),
Iterate/GetTensorC, and official MTE3_MTE2 event. L2 4x2 mapping and serpentine
N waves are specialized to 512-row readiness slabs. Actual platform UB/transLength,
L1/L0 allocation and API ABI MUST be checked on CANN 9.1 GA; this branch is not release-ready.
Reserve 256B UB for producer seat/trace separate from Matmul workspace.

## Readiness
For slab s (512 rows, final may be 256), eight seats at ib+8192+s*512+c*64.
Core c maps to row tile c/2 and column lane c%2; each traverses four N waves.
All eight publish after all their tiles are complete, including idle tail cores.
No scalar data packing. AR waits all real slabs for a 1024-row communication group;
AR consumer additionally waits local readiness before reading work (peerRdy alone
is insufficient). IAR A cursor waits one or two slabs depending on commRows,
then sends original seatUp; B/C, SDMA operations, mailbox counts and tail lanes unchanged.
AR commRows=512 is still effectively 1024 (existing AR behavior).
Per-producer trace stamp index is now slab index, maximum eight.
Do not mix V1/V2 ranks. Keep original epoch and stream discipline.

## Verification
- Host model: `python3 -m unittest discover -s tests -v`.
- Repository snapshot lacks the root package CMakeLists.txt/build.sh and README-referenced
  branchmark scripts, as well as the MemFabric SDK; restore the existing deployment
  build scaffold on the real host before CANN build. Do not use checked-in .so/.o/.whl.
- P0 gate: build both packages CANN 9.1.0 GA + memfabric_hybrid v28-orch;
  inspect raw Tiling size/fields, actual UB size and workspace requirement,
  check mixed task core mapping and event/seat visibility. Reject compile failures.
- P1: isolate MM producer; compare stock MatMulV2, stock MatMulV3, V1 MM and V2 MM.
  Same ND/NZ format, x@w.T, same shapes, and comparable eight-core configuration where
  supported. Stock may use different core count; record it instead of claiming parity.
- P2: TP2 compare all 13 M values, both commRows and packet 512 B, 64 KiB, 1 MiB, 4 MiB;
  both ranks, random/zero/identity and asymmetric rank speeds; verify tail cases, no
  hang, status==0, epoch separation and numerical tolerance vs FP16 stock MM+AR.
  IAR audit remote writes do not race local MM. Check per-rank ready/SDMA trace.
- P3: compare end-to-end V1 AR/IAR vs V2 AR/IAR and stock same invocation level.
  Use warmup, continuous same-stream async submission and only batch-end synchronization,
  no per-op sync, no downstream consumption in latency; separate correctness runs.
  Record per-rank p50/p95, slow-rank latency, msprof hardware timeline, Cube utilization,
  MTE/SDMA overlap, shape/packet/driver/CANN and exact git refs.
  Performance no-regression is a gate, not an assumption.
