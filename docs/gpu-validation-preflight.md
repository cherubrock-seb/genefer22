# GPU campaign validation input correction

Base: `ddc72834fff77b288dcfe19920707f006a3c492b`.

## Material recipe defect

The earlier future-hardware recipes for campaign PRs #2, #3, #4, #6,
#7 and #8 specified base 1024 for the low-range arithmetic configuration.
That base is invalid: `src/main.cpp` rejects powers of two after the evenness
and range checks. Use **1026**, not 1024, for this CLI case. This corrects a
recipe, not GPU arithmetic. No failed CLI invocation constitutes a GPU
correctness comparison, and no absent output can be counted as a match.

For ordinary non-CYCLO invocation the numeric conditions are: even base,
1024 <= b <= 2000000000, b not a power of two, and 12 <= n <= 23.
The internal n=11 dispatch exists but is not an accepted CLI workload.
Existing CPU synthetic checks that used b=1024 or n=11 inside a mocked host
remain internal dispatch checks; they do not demonstrate CLI acceptance.
Kernel dimensions or symbol names containing 1024 must not be renamed.

## Source-backed arithmetic matrix

The table supplies legal numeric pairs that select each of the four current
GPU arithmetic configurations. Values follow the exact integer thresholds in
`src/transform_ocl.cpp`, using primes from `src/transformGPU.h`. The 2x32
column is the first eligible even, non-power-of-two base at or above the
2x31 cutoff. Selection is source/version-specific, not a universal constant.

| n | 2x31 base | 2x32 base | 3x31 base | 3x32 base |
| --- | --- | --- | --- | --- |
| 12 | 1026 | 23448338 | 100000000 | 1200000000 |
| 13 | 1026 | 16580480 | 100000000 | 1200000000 |
| 14 | 1026 | 11724170 | 100000000 | 1200000000 |
| 15 | 1026 | 8290240 | 100000000 | 1200000000 |
| 16 | 1026 | 5862086 | 100000000 | 1200000000 |
| 17 | 1026 | 4145120 | 100000000 | 1200000000 |
| 18 | 1026 | 2931044 | 100000000 | 1200000000 |
| 19 | 1026 | 2072560 | 100000000 | 1200000000 |
| 20 | 1026 | 1465522 | 100000000 | 1200000000 |
| 21 | 1026 | 1036280 | 100000000 | 1200000000 |
| 22 | 1026 | 732762 | 100000000 | 1200000000 |
| 23 | 1026 | 518140 | 100000000 | 1200000000 |

Run from this checkout:

```sh
python3 tests/check_gpu_validation_bases.py
```

The test extracts the actual numeric b/n guards and GPU selector conditions,
compiles them in a standalone C++ program and substitutes return codes for
GPU constructors. It runs 206 CPU-only cases, including all 48 table pairs,
values around both two-prime cutoffs, base/parity/range/power-of-two negatives,
and exponent boundaries. An independent integer model checks the results.
Extraction assertions fail closed if the audited source structure changes.
No Genefer executable, GPU constructor, OpenCL API, primality run, benchmark
or driver installation is involved. A C++17 compiler and Python suffice.
The exact output is preserved in `gpu-validation-preflight-output.txt`.

This validates numeric pairs and selector branches, not all CLI token parsing,
GPU availability, experiment admission, arithmetic correctness or production
completion. A real job may exit for another reason before executing a kernel.
Future testers must record successful workload entry, selected kernels,
nonempty output artifacts and normal completion; an early exit is not GPU
coverage. A separate harness may test internal n=11, but label it accordingly.

## Use with the existing experiments

On 6 November 2026, main, all eight open PRs and branch heads were read.
The affected campaign heads were verified as authored by cherubrock-seb:

| PR | Preserved experiment commit |
| --- | --- |
| #2 | be056771628ae96219d9d79ec5407f56cd2a43ad |
| #3 | fceabb976967d00b0c242b7de1cbd3083e56e99c |
| #4 | 90ef084b1841b582795ddf5b2fc608bceb2426e3 |
| #6 | 50ffe58503516b57301b59260cf80b3223313bf1 |
| #7 | 254e62728a6ef6df981e3375c198e4e7e9badd7e |
| #8 | 4d474f866917989aa780852e013f15474c625417 |

Those commits and their immutable evidence remain preserved. Correction notices
in their draft PR descriptions supersede the invalid base in the older recipe.
This independent documentation/check branch starts at main and imports no
experimental kernel change. The original shared-twiddle/fused-products branches
and the independently active lazy-arithmetic experiment are not modified.

For future hardware work, check the exact baseline and candidate SHAs and run
this preflight against their numeric guards/selector before launching jobs.
Use each experiment's own default-off flag and recorded build procedure.
At matching exponent and base, run baseline and candidate separately on the
same NVIDIA or AMD device, driver and compiler. First compare complete register
and prepared-product data where relevant, res64/old64, Gerbicz-Li checks, proof
keys and checkpoint exchange. For a certificate, compare both binaries against
that same certificate's generating server key; server encoding is randomized.
Only after correctness passes should paired hardware timing be considered.
Retain original logs and exact identities, including invalid-input failures;
do not relabel an old run as though the corrected base had been used.

Rollback of this check is omission of the standalone script/documentation;
rollback of an experiment remains rebuilding with its own flag set to zero.
This change adds no runtime flag because it adds no runtime behavior. Main,
external/embedded kernels and all experimental defaults are unchanged. No
controller iteration, hardware artifact or speedup is claimed.

GPU_RUNTIME_TEST=NOT_RUN
GPU_CORRECTNESS=UNVALIDATED
GPU_PERFORMANCE=UNMEASURED
PRODUCTION_READINESS=NOT_ESTABLISHED
