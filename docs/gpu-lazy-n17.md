# GPU lazy forward reductions for GFN17

This is a default-off GPU performance experiment.

- Repository: `cherubrock-seb/genefer22`
- Base: `ddc72834fff77b288dcfe19920707f006a3c492b`
- Build flag: `GENEFER_GPU_LAZY_N17`
- Default: `0`
- Validated flag value: `1`
- Primary workload: `n=17`, `b=1000000000`
- Candidate keeps multiplication on the canonical reduced path.

When enabled, the experiment combines the empirically selected GPU settings
`VSIZE=2` and `USE_WI` with lazy signed forward butterflies for the production
`forward64_0 -> square2048 -> backward64_0` square path.

For the three signed moduli, `p < 2^31`, hence `2p < 2^32`. Forward values may
remain in `[0, 2p)`. Before a Montgomery multiplication, one operand is reduced
to `[0, p)` where required; for a lazy operand `x < 2p` and twiddle `w < p`,
`x*w < p*2^32`, satisfying the Montgomery input bound used by this implementation.
The square core receives fully reduced values. The backward half is unchanged.

The flag is intentionally off by default. With the flag disabled, the host uses
the original `VSIZE=4` path and does not enable `USE_WI`.

## Arithmetic validation

`tests/check_gpu_lazy_n17.py` checks all three signed moduli. It verifies the
lazy Montgomery-input equivalence and forward-butterfly modular equivalence,
including edge cases and deterministic random stress cases.

The Cherub campaign additionally ran 200,000 lazy Montgomery cases and 400,000
lazy butterfly cases per signed modulus before GPU promotion.

## Real GPU correctness

Candidate binary SHA-256 used for the full campaign:

`28ac7ab22e9f55df03e77bb7d7bea7367e70a2d2964bea65e2e5725071c654a3`

The cross-vendor regression matrix passed 10/10 cases:

- NVIDIA GeForce RTX 3080: 5/5 exact.
- AMD Radeon VII (`gfx906:sramecc+:xnack-`): 5/5 exact.

The matrix covered GFN14 bases `11724168`, `22843784`, `1000000000`,
and `2000000000`, exercising the small, signed-RNS3 and unsigned-RNS3 paths,
plus GFN17 `b=1000000000`.

Known GFN17 residue:

`041A41A3E0336675`

## Production performance

All measurements used non-TUNE production binaries and exact residues.

### NVIDIA GeForce RTX 3080

Baseline seconds:

`[108.74, 108.82, 108.82, 108.85, 108.84]`

Candidate seconds:

`[103.94, 103.94, 103.97, 103.97, 103.95]`

Median throughput speedup: **4.6849%**.

All 5/5 counterbalanced pairs were positive and all 5/5 exceeded 3%.
Paired-speedup spread was 0.0861 percentage points.

### AMD Radeon VII

Baseline seconds:

`[359.42, 361.76, 362.50]`

Candidate seconds:

`[351.32, 351.63, 351.61]`

Median throughput speedup: **2.8867%**.

All 3/3 counterbalanced pairs were positive.

## Build

Default behavior:

```sh
make -C genefer -f makefile_linux_x64 BOINC= -B ../bin/geneferg
```

Enable the experiment:

```sh
make -C genefer -f makefile_linux_x64 BOINC= -B ../bin/geneferg \
  FLAGS_GPU='-O3 -DGPU -DGENEFER_GPU_LAZY_N17=1'
```

The experiment remains draft/default-off pending broader workload coverage and
upstream review. `main` is not changed by this branch.
