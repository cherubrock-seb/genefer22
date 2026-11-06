# Register-resident operand preparation tails

Base: `ddc72834fff77b288dcfe19920707f006a3c492b`.
Build with `GENEFER_GPU_REGISTER_PREPARE_TAIL=1` to opt in; default is 0.
This independent experiment changes only the final two operations inside
four operand-preparation kernels. It neither removes the initial copy nor
changes the first forward transform, distinguishing it from drafts #6/#7.

## Ownership and arithmetic

For VSIZE=4 and terminal K=64,256,1024,4096, let L=K/4 uint4 vectors per tile.
The last ordinary radix-4 stride is L/(K/4)=1. For work-item t within a tile,
its local pointer is `(4*t & ~3)+(t mod 1)=4*t`, the same pointer used by
`fwd4_write`. Both operations own vectors [4*t,4*t+3]. Batched tiles add
the same tile base to both pointers; intervals from different items are
disjoint. With 0 <= t < L/4, the final local index is L-1 within each tile.
The final global writes remain at `tile_base+t+j*(L/4)`, j=0,1,2,3. Quotient
and remainder by L/4 show unique complete output ownership, ending at L-1.
Prime-plane selection and all group/tile offsets are unchanged.

The candidate keeps the barrier preceding the last radix-4 load. That orders
all preceding local exchanges. It loads four uint4 values once, calls the
original `_forward4x4`, then the two original `_fwd4x2v4` operations, and
stores globally using the original stride. The intermediate local store,
barrier and reload are unnecessary: the same item consumes its own values,
and no later operation reads that local output. There is no subgroup,
warp-synchronous or cross-workgroup assumption.

At stride one, both baseline calls use exactly sj. The scalar and vector
twiddle macros are retained in separate scopes. All six primes, canonical
residue ranges, Montgomery representations and arithmetic operation order
are inherited from main. No multiplication identity, lazy range, root
convention, normalization factor or data-layout change is introduced.
Four private vectors contain the same 16 residues as in the baseline helpers.

| Kernel | Fixed exponents | Items/group | Allocated local bytes | Static barriers, base to candidate |
| --- | --- | --- | --- | --- |
| fwd64p | 12 | 64 | 4096 | 2 to 1 |
| fwd256p | 14 | 64 | 4096 | 3 to 2 |
| fwd1024p | 16,18 | 64 | 4096 | 4 to 3 |
| fwd4096p | 20,22 | 256 | 16384 | 5 to 4 |

Barrier counts were checked in Clang 18 SPIR64 -O2 output. The source-level
saving is 128 local payload bytes (64 written plus 64 read) and one barrier
per participating item. Local allocation, global payload and dispatch count
stay unchanged. Driver optimization, register pressure, spills and preparation
frequency determine any performance effect. No speedup or bank-conflict ratio
has been measured or inferred.

## Host selection and fallback

The normal OpenCL program builds first. Fixed even exponents 12 through 22
may compile a separate auxiliary program, with distinctly named
`fwd*K*p_register` entries (for example, `fwd64p_register`). Each entry is
accepted independently only after device workgroup, first-axis, kernel group
and local-memory queries and argument binding succeed. Failure preserves its
original handle. The candidate binds `_zp` and `_w`; `_z` is not passed to it.
The existing initial copy selects the source register. Square, multiply,
carry, register allocation, checkpoint and proof operations are unchanged.
TUNE, diagnostic radix-4 mode, other exponents and non-four-lane kernels retain
baseline behavior. Opt-in builds report enqueue errors without retrying after
possible partial execution. The flag defaults to zero and rejects values
other than 0/1.

## Reproduce the offline checks

Linux x86-64, Clang, a C++17 compiler, Python/NumPy, GMP development files and
an OpenCL loader suffice. No GPU or driver installation is needed. `CC_CL`,
`CXX` and `GMP_STATIC` optionally select the existing tools/libraries.
From the repository root:

```sh
python3 tests/check_register_prepare_static.py
python3 tests/run_register_prepare_matrix.py
python3 tests/check_register_prepare_dispatch.py
```

The native CPU adapter compiles actual OpenCL C and supplies cooperative
workgroup barriers. It does not invoke an OpenCL runtime. All four terminal
kernels compare every word at n=16 for the eight RNS-count, modulus-width and
inverse-table combinations. Additional 3x32 comparisons at n=12/14 cover all
groups; n=18/20/22 cover first/middle/last groups of every prime plane. Inputs
include deterministic random coefficients, 0 and p-1. There are 52 kernel
comparisons. Large-size checks are sampled, not full-length validation.

Static checks cover embedded/external equality, 48 enabled CL1.1/1.2 syntax
configurations, 24 flag-off preprocessed comparisons to the pinned base,
16 non-four-lane fallback comparisons, and the barrier counts above.
Synthetic OpenCL host checks cover all exponents 11 through 23, destination
binding, eight admission refusals for each eligible exponent and enqueue
errors. They perform no arithmetic or device execution. The ordinary host
also compiled/linked for flag 0 and 1; only version output was requested.
Invalid flag 2 was rejected. TUNE syntax passed for both flag values with
`'-DDEFINE_FORWARDP0(u)=DEFINE_FORWARDPn(u,0)'`, a compile-only workaround
for an existing baseline macro defect. This branch does not fix that defect.
Recorded outputs are in `gpu-register-prepare-tail-offline.txt` beside this file.

## Future hardware validation and rollback

Use separate worktrees at the exact base and candidate SHAs in the Draft PR.
Record the git SHA, compiler, binary hash, GPU, driver, flags, arithmetic path
and selected entry names. Build separate binaries with VALUE replaced by 0
for the base and candidate control, or 1 for the candidate:

```sh
make -B -C genefer -f makefile_linux_x64 BOINC= \
  FLAGS_GPU='-O3 -DGPU -DGENEFER_GPU_REGISTER_PREPARE_TAIL=VALUE' ../bin/geneferg
```

Test NVIDIA and AMD OpenCL devices at n=12,14,16,18,20,22 and odd controls
n=13,17,21,23. At n=20, bases 1024,1800000,100000000,1200000000 target the four
arithmetic configurations; record actual selection. Exercise source registers
0,1 and the highest allocated register, repeated preparation, and resource
refusal. Compare all source registers and the full prepared operand, followed
by products, at matched checkpoints. Exchange checkpoints in both directions.
Run from empty directories for embedded kernels, then with matching external
`ocl/kernel.cl`. Verify that admitted entries bind `_zp`, and that unavailable
entries select the original kernels without changing square/product dispatch.

Complete matched bounded jobs with `geneferg -q -n N -b B -d D -f job`,
separately `-p`, then `-s`/`-c` with the matching artifacts. Require identical
verdict, res64/old64 where present, proof keys and Gerbicz-Li checks. Check each
same certificate with both binaries against its generating server key;
server encoding is randomized. Reject any mismatch or restart failure.

Only after correctness passes, warm up and alternate at least three paired
windows. Record median/spread for operand preparation, complete iterations and
auxiliary compilation separately. Inspect local transactions, barriers,
registers, spills and occupancy using available vendor tooling. Timing inside
run-to-run noise is inconclusive. Reject enqueue errors, unsupported-path
changes or repeatable regressions. Rebuild with flag 0 to roll back; no format
migration is needed. Hardware validation has not been performed.

## Campaign provenance and handoff

On 6 November 2026, main and all seven open PRs were inspected. PR #3 at
`fceabb976967d00b0c242b7de1cbd3083e56e99c` provided the verified
cherubrock-seb-owned private-ownership precedent and MIT-licensed admission
helpers. Its square implementation is not included. CPU fixtures also follow
PR #7 at `254e62728a6ef6df981e3375c198e4e7e9badd7e`. This branch starts from
main and uses main's arithmetic helpers; no upstream or reference code was
changed. Shared-twiddle, fused-products, GFN-21/23 plans, lazy GFN-17 and copy
fusion remain separate experiments. Combined flags require separate review
and validation. No controller iteration is claimed.

Next review should inspect odd-length tail ownership before assuming the same
fusion is legal: their stride-two radix stage and contiguous output core have
different owners. Preserve their current barriers unless a new ownership
mapping and independently checked implementation justify a change.

GPU_RUNTIME_TEST=NOT_RUN
GPU_CORRECTNESS=UNVALIDATED
GPU_PERFORMANCE=UNMEASURED
PRODUCTION_READINESS=NOT_ESTABLISHED
