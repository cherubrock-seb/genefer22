# GFN-23 wide square with 256 physical work-items

Independent default-off experiment, based on main ddc72834fff77b288dcfe19920707f006a3c492b. Define GENEFER_GPU_WIDE_SQUARE23_WG256=1 to request it; omit the definition or set it to 0 to roll back. No main change and no combination with other experiments.

## Provenance and continuity

This is a new candidate following rejected PR #4, cherubrock-seb's 90ef084b1841b582795ddf5b2fc608bceb2426e3. Its 1024 x 8192 decomposition and defensive auxiliary-program admission are retained, but the 8192 tail now maps 512 logical lanes onto 256 physical work-items. PR #4's candidate was not admitted on either tested GPU; that evidence does not isolate workgroup size as the sole cause. This candidate has no GPU evidence. PRs #6/#7 already claim copy/first-forward fusion, and #8 claims register preparation tails. Their implementations, the original shared-twiddle/fused-products branches, and active PRs #2/#5 are untouched.

## Ownership and bounds

For each prime plane and 8192-coefficient tile, physical item p in [0,255] executes logical lanes p and p+256 sequentially at each stage. At stride s in {512,128,32,8,2}, lane l accesses (l/s)*(4*s)+l%s+j*s for j in [0,3]. The 512 lanes partition all 2048 uint4 slots exactly once. Each half therefore owns 1024 distinct slots, disjoint from the other half within that stage. Processing half zero cannot overwrite half one's inputs for the same stage. All work-items execute both halves and every helper barrier uniformly; the next stage starts only after the required local fence. The square core uses adjacent groups of four vectors, also disjoint.

Global addresses are plane*N/4 + tile*2048 + lane + j*512. N=2^23; each prime owns 1024 tiles. The final index is R*N/4-1, within the allocated planes; all intermediate indices fit uint32 for R<=3. Tail global size is R*N/32, local size 256. The root indices, Montgomery arithmetic and inverse traversal match the original schedule. The CPU tests compare the actual kernel against main's 64 x 64 x 2048 schedule and an independent sparse polynomial product.

The local tile remains 32768 bytes. Sequential logical halves increase dynamic barriers; resource pressure and performance are unknown. No speedup is claimed. This is restricted to n=23, VSIZE=4, non-TUNE. Multiplication, operand preparation, normalization and source-register storage remain canonical. Device limits, both compiled outer kernels and compiled tail resources must admit the route; build/kernel/argument/resource failures retain the baseline. Enqueue errors propagate rather than silently continuing with damaged state.

## Offline evidence and reproduction

Preserved outputs are in docs/offline/wide256-*.log. Before the execution environment lost its pre-existing development dependencies, five full-length CPU synthetic configurations passed sparse and dense comparisons: 2x31, 2x32, 3x31, 3x32, and 3x32 with USE_WI (ten comparisons). Sparse cases also matched an independent polynomial oracle. Both flag-on/off host builds completed. Static checks passed 48 OpenCL syntax configurations, 24 flag-absent token comparisons, 16 non-four-lane fallback token comparisons, embedded/external identity and ownership/bounds.

The initial simulated dispatch test reached the device-limit fallback but compared it against a baseline with different limits. That changes normalization workgroup size legitimately. The test now compares the baseline under the same injected limits. Its rerun is BLOCKED: gmp.h and the previous GMP/OpenCL development libraries are absent. The preserved dispatch log records this failure. Do not report the full dispatch/fallback suite as passing. TUNE and invalid-flag host compilation checks remain pending.

With existing clang (OpenCL frontend), C++17 compiler, NumPy, GMP development headers/library and OpenCL loader available (no GPU needed):

```sh
python3 tests/check_wide256_static.py
python3 tests/check_wide256_dispatch.py
python3 tests/check_wide_square23_wg256.py
python3 tests/check_wide_square23_wg256.py --wide
python3 tests/check_wide_square23_wg256.py --rns 3
python3 tests/check_wide_square23_wg256.py --rns 3 --wide
python3 tests/check_wide_square23_wg256.py --rns 3 --wide --use-wi
```

CC_CL and CXX may select installed compilers; GMP_STATIC may select an existing archive. The native OpenCL adapter executes synthetic arrays with CPU fibers, never an OpenCL device. The mock host's internal base 1024 is only a constructor fixture, not a valid CLI validation base.

## Future hardware validation (not executed)

First finish the pending offline dispatch checks. Record exact candidate SHA from this PR, baseline SHA above, compiler/driver/device identities, build flags and binary hashes. Build baseline and candidate flag-off/on in independent directories using the repository makefile, e.g. make -B -C genefer -f makefile_linux_x64 BOINC= FLAGS_GPU='-O3 -DGPU -DGENEFER_GPU_WIDE_SQUARE23_WG256=1' ../bin/geneferg. Repeat with flag 0 and baseline main.

Use valid even CLI bases 1026, 750000, 100000000 and 1200000000 at n=23; inspect the selected arithmetic path rather than assuming a base selects it. Follow PR #9's CLI preflight. Capture actual candidate selection, kernel names, R*N/32 global size, 256 local size and compiled local-memory/kernel limits on NVIDIA and AMD. A fallback is NOT candidate execution. Capture build logs and resource refusal details if admission fails.

Only after explicit hardware authorization, compare identical bounded workloads across baseline/off/on, all source registers, square/duplicate, preparation/multiplication, Gerbicz checks, checkpoint/resume and proof/certificate verification. Require identical residues/proofs before any separately authorized timing. Also exercise resource refusal and enqueue-error behavior. Hardware execution and measurements are not part of this implementation campaign.

GPU_RUNTIME_TEST=NOT_RUN
GPU_CORRECTNESS=UNVALIDATED
GPU_PERFORMANCE=UNMEASURED
PRODUCTION_READINESS=NOT_ESTABLISHED
