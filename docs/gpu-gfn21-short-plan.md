# GFN-21 three-launch transform experiment

Base `ddc72834fff77b288dcfe19920707f006a3c492b`. Enable with
`-DGENEFER_GPU_SHORT_PLAN21=1` on the GPU host build. Omit the flag or use 0
for the original plan. The production selector requires four vector lanes,
n=21 and a fixed schedule. It checks all five participating kernel handles
before using the same layout for squaring, multiplication and operand preparation.
A failed query selects the original schedule. The tuned splitter owns its own
choice. Opt-in builds check OpenCL argument and enqueue return codes.

## Transform and cost model

Let N=2^n, R=2 or 3 active primes, and D=4RN bytes for one coefficient array.
One transform pass reads D and writes D, excluding twiddles, caching and device
transaction granularity. Each work-item handles four vectors of four residues
from one prime plane. Outer kernels transpose through workgroup-local tiles;
a launch boundary orders the next cross-group consumer. No separate transpose
is enqueued. The terminal kernel already fuses its forward butterflies,
pointwise quadratic square or product, and inverse butterflies.

| Fixed n | Outer stages, in order | Terminal length |
| --- | --- | --- |
| 11–17 | 64 | 32, 64, 128, 256, 512, 1024, 2048 respectively |
| 18–20 | 256 | 1024, 2048, 4096 respectively |
| 21 | 64, 64 | 512 |
| 22 | 1024 | 4096 |
| 23 | 64, 64 | 2048 |

The CLI accepts n=12–23. The host also contains the n=11 schedule. Square/mul
run the inverse outer stages in reverse order. The first forward stage enters
Montgomery form. The final inverse returns the raw scaled negacyclic result.
Host-generated roots are stored by prime and stage, in bit-reversed odd-power
order. Inverse lookups use the mirrored index, or the second half of the
table with USE_WI. An ordinary scalar radix-4 butterfly performs four
Montgomery products and eight modular adds/subtracts across four coefficients.
A radix-4 stage therefore costs RN such products and 2RN additions/subtractions.
A k-stage outer kernel amortizes those operations over a single 2D coefficient
pass (approximately k/8 products per coefficient byte). Entry conversion,
terminal operations and twiddle loads are additional. The quadratic square
uses four Montgomery products per coefficient pair; the corresponding
prepared product uses five.

| n=21 operation | Fixed baseline | Opt-in | Source-level difference |
| --- | --- | --- | --- |
| Square or prepared multiply | 64 → 64 → 512 → 64 → 64 | 1024 → 2048 → 1024 | Two launches and 4D coefficient bytes |
| Prepare multiplicand | copy → 64 → 64 → fwd512p | copy → 1024 → fwd2048p | One launch and 2D bytes |
| Square plus carry normalization | Seven launches | Five launches | Same two carry kernels |

Two full passes are 64 MiB for R=2 and 96 MiB for R=3 at n=21. Multiply also
reads one D-sized prepared operand in either schedule. Baseline outer64 groups
have 64 work-items and 4 KiB of local storage at VSIZE=4. Candidate outer1024
groups have 256 items and 16 KiB. Terminal512 uses 64 items and 4 KiB,
terminal2048 uses 128 and 8 KiB. The square's per-group barrier counts summed
over its successive kernel types go from 14 to 16 (not a device-wide count).
Extra local traffic, lower occupancy or less favorable global access can
outweigh the saved dispatches and bytes. Register allocation is driver-specific.

The normalizer reads all R raw planes, applies the inverse scale and signed
Garner reconstruction, carries the four base-b digits, and exchanges neighboring
carries through local memory with one barrier. It writes D bytes and one
8-byte carry per group. normalize2 processes only the first four digits of
each group, including the negative wrap at index zero. squareMul for a>2
runs the additional mulscalar and normalize2 pass. The host allocates
`_z` (D per register), `_zp` (D), `_w` (D) and `_c` (2N bytes).
The candidate adds no device buffers. In particular it leaves the cross-group
carry boundary available to order its consumers.

## Equivalence and provenance

2^21 = 64 × 64 × 512 = 1024 × 2048. These are existing radix-4 stage and
terminal kernels and a decomposition the tuned splitter can enumerate. Moving
one radix-4 level between outer and terminal kernels preserves the sequence of
power-of-two butterflies and the inverse scale. The same branch selects both
preparation and consumption of the one transformed operand. All externally
observable register reads, copies and checkpoints occur in the ordinary
normalized layout. The six primes have sufficient power-of-two roots over
the supported size range. No modular expression, width, intermediate range
or carry operation changes. For each plane a work-item index runs from 0 to
N/16−1. Both schedules divide a plane into whole groups, and all tile writes
and interstage reads stay in the original kernel bounds.

Genefer22 and the upstream inspected at this base use MIT. Genefer26 fork and
upstream `0e962f8f09f0e4dbdb2acaf76b30bd7aa30a199e` have different batched
GPU transforms. PrMers `dddee7b71d88cb05b4fe6061f7b0df72f758f081`
separates its MIT sources from GPLv3 vendored Aevum. Aevum
`fc6a3b97952a0850022c7c0c412e27460f9f34ba` illustrates the synchronization
cost of cross-group carry fusion. These references motivated keeping the
carry boundary and comparing dispatch counts. All implementation arithmetic
here comes from Genefer22. Neither floating-point FFT code nor PFA7 applies
to this power-of-two decomposition. The prior shared-twiddle and fused-product
experiments and upstream Apple-specific proposals are independent.

## Reproduction and decision

On Linux/x86-64 with Clang, a C++17 compiler and Python/NumPy run:

```sh
python3 tests/check_short_plan21.py --rns 2
python3 tests/check_short_plan21.py --rns 2 --wide
python3 tests/check_short_plan21.py --rns 3
python3 tests/check_short_plan21.py --rns 3 --wide
python3 tests/check_short_plan21.py --rns 3 --wide --use-wi
```

This executes the OpenCL C functions translated to native CPU code with a
cooperative workgroup adapter. Every coefficient is compared on a dense
square, a sparse square and a distinct-operand sparse multiply. The sparse
oracle multiplies signed polynomials modulo x^N+1 and applies the raw inverse
scale (N/2)·2^32 mod p. Inputs include sign wrap and tile edges.

P0 for n=21, moderate dispatch complexity and no new arithmetic. The draft
PR records the exact candidate SHA, build commands and future cross-device
residue, proof, restart and alternating A/B procedure. Benchmark square and
operand preparation separately. The original base has an unrelated undefined
`DEFINE_FORWARDP0` when building plain `-DTUNE`; fixed-plan builds are the
validation target. A compile-only comparison with
`'-DDEFINE_FORWARDP0(u)=DEFINE_FORWARDPn(u,0)'` is possible on both revisions.

```text
GPU_RUNTIME_TEST = NOT_RUN
GPU_CORRECTNESS = UNVALIDATED
GPU_PERFORMANCE = UNMEASURED
PRODUCTION_READINESS = NOT_ESTABLISHED
```
