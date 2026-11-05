# Register-resident square core

Base `ddc72834fff77b288dcfe19920707f006a3c492b`. Set
`-DGENEFER_GPU_REGISTER_SQUARE_CORE=1` for an opt-in GPU build; omit or set
to 0 for the baseline. Separate optional entry points are compiled from the
same OpenCL source. An optional build failure keeps all baseline handles.
Each replacement is accepted independently after successful argument binding
and checks of device dimension, kernel workgroup size and local-memory
requirements. A missing entry point in an older external kernel file falls
back as well. The opt-in build checks argument and enqueue errors. Each
accepted kernel is named in verbose startup output. Auxiliary compilation
can increase startup time and should be measured separately. Odd fixed-plan exponents skip
the extra compilation.

## Ownership, arithmetic and costs

With four lanes, the last regular radix-4 stride is one for terminal lengths
64, 256, 1024 and 4096. For thread t within a block, the baseline address
formula for that stage is (4t & ~3) + (t mod 1) = 4t in uint4 units. The
square and the inverse at this stride also access slots 4t through 4t+3.
Adjacent threads own disjoint intervals. Batched 64- and 256-point tiles
include the same block offset in both pointers. The largest offset is the
last vector of that tile. Other strides or vector widths use the old code.

The first barrier orders the preceding tile exchange; the following baseline
barrier orders the next exchange. Within the interval, three operations are
performed by the same work-item, in the same order and with the same scalar
root and two-/four-wide twiddle loads, including reversed inverse lanes.
Each of the 16 residues uses the original Montgomery representation and
reduction range [0,p) for each of the six supported primes. No new product,
lazy reduction, sign change, permutation, root convention or normalization
factor is introduced. The four vectors live in private storage across the
sequence. No subgroup size, warp ordering or cross-workgroup barrier is assumed.

| Per participating work-item | Original | Candidate |
| --- | --- | --- |
| Local payload loaded then stored across the three operations | 3 × (64 B read + 64 B write) | 64 B read + 64 B write |
| Workgroup barriers inside the three-operation sequence | 3 | 1 |
| Global coefficient reads/writes and kernel launches | One terminal pass | One terminal pass |

With Clang 18 at -O2 targeting SPIR64, the count of static barrier calls in
the complete square64/256/1024/4096 functions was 4→2, 6→4, 8→6 and 10→8,
respectively. This is an intermediate-representation check.

The source-level saving is 256 local payload bytes per work-item. The tile's
allocated LDS capacity stays 4 KiB for 64/256/1024 or 16 KiB for 4096.
Fixed-plan squares at n=12,14,16,18,20,22 use these kernels. Other tails,
preparation, multiply, RNS reconstruction and carry keep their original
flow. This is a P1 investigation affecting repeated square operations. A
driver may already remove some accesses or spill the longer-lived private
vectors. Local transactions, workgroup occupancy, spills and per-square time
are the discriminating GPU measurements. No bank-conflict ratio is inferred.

The implementation uses Genefer22's MIT-licensed butterfly and quadratic-square
helpers. Genefer26's batched radix-8 routines informed the question of private
reuse; no batched layout or GPLv3 Aevum code was imported. Arithmetic experiments
on the twiddle and product branches remain separate.

## Offline reproduction

On Linux/x86-64 with Clang, a C++17 compiler and Python/NumPy run:

```sh
for r in 2 3; do
  for width in '' '--wide'; do
    python3 tests/check_register_square_core.py --rns "$r" $width
    python3 tests/check_register_square_core.py --rns "$r" $width --use-wi
  done
done
for n in 12 14 18 20 22; do
  python3 tests/check_register_square_core.py --n "$n" --rns 3 --wide
done
```

The test compares original and optional kernel entries compiled to native CPU
code. It checks all coefficients/groups for small arrays and selected first,
middle and last groups per prime plane for larger lengths, using deterministic
interior values and 0/p−1 edge values. The draft PR carries the exact revision,
GPU build and validation commands, proof/restart checks and rollback instructions.
Plain `-DTUNE` at the base requires the existing `DEFINE_FORWARDP0` alias fix.

```text
GPU_RUNTIME_TEST = NOT_RUN
GPU_CORRECTNESS = UNVALIDATED
GPU_PERFORMANCE = UNMEASURED
PRODUCTION_READINESS = NOT_ESTABLISHED
```
