# GPU shared-twiddle Montgomery multiplication

- Repository: cherubrock-seb/genefer22
- Branch: perf/gpu-shared-twiddle-20261104
- Base and inspected upstream HEAD: ddc72834fff77b288dcfe19920707f006a3c492b
- GPU_VALIDATION=DEFERRED_TO_CHERUB
- GPU_EXACTNESS=PENDING_CHERUB
- GPU_PERFORMANCE=PENDING_CHERUB
- PRODUCTION_READINESS=NOT_ESTABLISHED
- Candidate HEAD: the current branch tip; record `git rev-parse HEAD` with each experiment.

## Hypothesis checkpoint

The forward/inverse NTT vector butterflies broadcast one Montgomery twiddle
across two or four scalar residue lanes. The current mulmod2/mulmod4 API takes a
vector right operand, although every external caller supplies a scalar twiddle,
entry conversion constant, or normalization constant.

Let R=2^32, 0<=a,w<p<R, and q=p^-1 mod R. Current Montgomery multiplication is
hi(a*w) - hi(((lo(a*w)*q) mod R)*p), corrected modulo p.
Associativity modulo R gives lo(a*w)*q = a*(w*q mod R) mod R.
Compute wq once for the broadcast operand, then use hi(a*w) and
hi((a*wq mod R)*p) for each lane. The unreduced difference is identical, so
the existing submod correction and canonical output bounds remain valid for
both 31-bit and 32-bit moduli. All low-word products are unsigned modulo R.
The wide product of two uint32 values fits uint64.

For four lanes, the source-level multiplication count changes from eight
low-word and eight high-word products to five low-word and eight high-word
products: 16 to 13 (18.75% fewer). For two lanes it changes from 8 to 7.
No twiddle buffer, transform layout, barrier, kernel launch, or carry semantics
need change. This is a static operation count, not a measured speedup.
If the affected multiplies account for fraction f of total time and their cost
scales with that count, the illustrative total speedup is 1/(1-0.1875*f).
Register pressure, instruction scheduling, compiler folding and memory stalls
can negate the gain.

## Implementation and verification queue

1. Add default-off GENEFER_GPU_SHARED_TWIDDLE build option, propagated into
   generated OpenCL source.
2. Add shared-preconditioned scalar helper and explicit scalar RHS vector
   wrappers; retain the existing generic vector functions as fallback.
3. Route broadcast callers through these wrappers, regenerate the embedded
   source without running the application.
4. Run exact arithmetic boundary/random tests for all six primes and
   source-extracted C++ arithmetic checks. Compile the GPU host translation
   unit without linking or invoking an OpenCL runtime.
5. Publish a Draft PR to this fork's main with exact A/B commands and journal.

No compiler or GPU runtime installation, device probing, SSH, GPU execution or
CPU performance tuning is part of this branch.


## Implemented checkpoint

- Hypothesis commit: `ef9fc937bd7c2caf057da6396b46ddba22975d18`.
- First source commit: `630f6244b08e0e951a2fb28345fca1b00cb7042f`.
- Changed files: `ocl/kernel.cl`, `src/ocl/kernel.h`, `src/transformGPU.h`,
  `tools/embed_opencl.py`, `tests/check_gpu_shared_twiddle.py`, this journal.
- The flag defaults to zero in both the host and kernel; other values fail
  preprocessing. The host propagates it into either external or embedded source.
- All broadcast vector multiplies in forward/inverse butterflies, entry
  conversion and normalization use the shared setup when enabled. Scalar
  arithmetic and generic vector multiplication remain available unchanged.
- The candidate adds one live 32-bit preconditioner, no local/global storage and
  no workgroup assumptions. Instruction scheduling and register lifetimes still
  need inspection on each driver. Constant-folded normalization factors may
  already have been optimized by the baseline compiler.

### Executed offline evidence

`python3 tests/check_gpu_shared_twiddle.py` passes with g++ and UBSan. It extracts
arithmetic functions and radix-4 vector butterflies from the actual OpenCL
source, adapting only vector constructors to simple C++ value types.
For each of flag=0/1 and IS32=absent/present, it checks 60,300 four-lane tuples
against an independent modular-exponentiation oracle and 60,000 forward/inverse
radix-4 round trips (including a separate forward-transform oracle).
Total: 241,200 vector tuples and 240,000 butterfly round trips, with all six
prime moduli, zero, p-1, midpoint and deterministic random inputs represented.
These are arithmetic checks, not executed GPU kernels or performance evidence.

`python3 tools/embed_opencl.py --check` passes; `git diff --check` passes.
No offline OpenCL compiler was available. A host `-fsyntax-only` attempt stopped
at the pre-existing missing `gmp.h` dependency; full host build remains pending.
No GPU hardware discovery, runtime execution or benchmark was attempted.

## Cherub validation recipe (not executed here)

Use the branch tip recorded by `git rev-parse HEAD`, and preserve the source SHA,
compiler version, device identity, driver version, build flags and binary hashes.
Build on a host that already has the project's GMP/OpenCL development dependencies.
From the candidate checkout, build two separate binaries; `-B` avoids stale objects:

```sh
python3 tools/embed_opencl.py --check
python3 tests/check_gpu_shared_twiddle.py
GENEFER_EVIDENCE="$PWD/../genefer22-twiddle-evidence"
mkdir -p "$GENEFER_EVIDENCE/off" "$GENEFER_EVIDENCE/on"
make -C genefer -f makefile_linux_x64 BOINC= -B ../bin/geneferg FLAGS_GPU='-O3 -DGPU -DGENEFER_GPU_SHARED_TWIDDLE=0'
cp bin/geneferg "$GENEFER_EVIDENCE/off/geneferg"
make -C genefer -f makefile_linux_x64 BOINC= -B ../bin/geneferg FLAGS_GPU='-O3 -DGPU -DGENEFER_GPU_SHARED_TWIDDLE=1'
cp bin/geneferg "$GENEFER_EVIDENCE/on/geneferg"
sha256sum "$GENEFER_EVIDENCE/off/geneferg" "$GENEFER_EVIDENCE/on/geneferg"
```

Run from each isolated evidence directory, which has no `ocl/kernel.cl`, to
exercise the embedded source. Select the device using the existing Cherub
mapping, shown below as device 0. Use distinct `-f` paths per run to avoid resumes.
Compare these baseline-source GFN14 quick-test residues (listed in `src/main.cpp`)
for off, on, and an unchanged build of the base SHA:

| b | Expected residue |
|---|---|
| 11724168 | 063922D606F7ED14 |
| 22843784 | 45DAA25E59EAC127 |
| 1000000000 | 47A9DFAD0217519A |
| 2000000000 | 434063FE41B764E2 |

Example for one case, repeated in both directories:

```sh
./geneferg -n 14 -b 11724168 -q -d 0 -f gfn14-b11724168-run1
```

After the short correctness gate, cover every n=17..23. For each n, use b=1026
for NTT2/31-bit, the smallest even b with
`b*b >= (P1S*P2S//2)//(1<<n)` for NTT2/32-bit, b=1000000000 for NTT3/31-bit,
and b=1999999998 for NTT3/32-bit. Confirm selection from the unchanged selector.
Compare complete residues and Gerbicz checks; run a proof/server/check sequence
on selected cases to exercise multiplication and register reuse as well as
squaring. This repository processes one GFN, so there is no batch-size setting.
Example large case: `./geneferg -n 17 -b 1000000000 -q -d 0 -f gfn17-b1e9-run1`.

Use warmed, alternating A/B/B/A pairs, at least three independent pairs per
selected workload and device, with identical power settings and no concurrent
GPU job. Separate initialization/JIT from steady-state iteration time. Report
median and dispersion, complete application throughput, and profile forward,
square, inverse and normalization kernels. Capture generated ISA, low/high
multiply counts, registers and spills when the existing tools allow it.
Include NVIDIA, AMD and Intel hardware available to Cherub; no architecture
speedup is assumed. Re-run from the repository root to check external-source
loading after the embedded-source checks.

Reject on any residue/check mismatch, driver compilation failure, increased
spill traffic or reproducible end-to-end regression. Roll back by rebuilding
with `GENEFER_GPU_SHARED_TWIDDLE=0` (or using the unchanged base binary). Do not
merge or enable by default based on this source-level cost model.

Next executable engineering action: independently investigate whether fusing
Montgomery product sums in the terminal squaring stage can remove reductions
without changing carry bounds; keep it on a separate experimental branch.
