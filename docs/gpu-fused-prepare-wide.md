# Wide outer transforms for fused operand preparation

Base: `ddc72834fff77b288dcfe19920707f006a3c492b`.
Enable explicitly with `GENEFER_GPU_FUSED_PREPARE_WIDE=1`; default is 0.
This independent experiment targets fixed-plan GFN-18, 19, 20 and 22 only.
It does not alter main or any other experimental branch.

## Operation and ownership

`initMultiplicand(src)` normally copies one register of `_z` into `_zp`,
then transforms `_zp`. The new `prepare256_0` and `prepare1024_0` read that
register directly and write the existing first-forward layout into `_zp`.
The entry conversion and forward helpers are unchanged. The remaining
`fwd1024p`, `fwd2048p` or `fwd4096p` is exactly the baseline terminal operation.
Square and prepared multiplication retain their original dispatch and layout.

| Exponent | First kernel | Unchanged terminal | Workgroup | Local bytes |
| --- | --- | --- | --- | --- |
| 18 | prepare256_0 | fwd1024p | 256 | 16384 |
| 19 | prepare256_0 | fwd2048p | 256 | 16384 |
| 20 | prepare256_0 | fwd4096p | 256 | 16384 |
| 22 | prepare1024_0 | fwd4096p | 256 | 16384 |

Let N=2^n, K be the outer size, B=K/4, C=CHUNK_K, and M=N/(4K).
For VSIZE=4, each modulus plane has N/4 uint4 vectors and N/16 work-items.
`blockIdx=b` is in [0,M), so `blockIdx_m=0` at this first stage.
For thread index t in [0,B), the four input vector indices are
`(j*B+t)*M+b`, j in [0,4). The four output indices are `(4*t+j)*M+b`.
Each is a bijection onto [0,N/4), proved by quotient and remainder by M
and then by B or 4. Chunking merely assigns adjacent b values to one group.
No group crosses a prime plane: N/16 is divisible by B*C=256 for every
eligible size. The local array has 4*B*C uint4 vectors, or 16384 bytes.

The input plane base is `src*R*N/4 + 4*mid`, where `4*mid=plane*N/4`.
The output plane base remains `4*mid` in `_zp`. These are distinct allocations;
all source registers are read-only, including the selected one. For
`span=R*N/4`, admission checks `src < num_regs` and
`src <= floor((UINT32_MAX-(span-1))/span)`. Thus the last vector address fits
the kernel's 32-bit `sz_t`; the host forms the offset as `src*span` only after
that check. Every scalar lane, every prime and every source register is
preserved. Roots, Montgomery conversion, modular arithmetic, barriers and
intermediate bounds are inherited without modification from main. In
particular, no lazy range or new arithmetic identity is introduced.

## Admission and fallback

The baseline OpenCL program builds first. A separate auxiliary program adds
the candidate entries. Only the selected entry is admitted after device
workgroup, first-axis, kernel-workgroup and local-memory queries succeed.
Program build, entry creation, query and initial argument-binding failures
retain the original copy-plus-forward route. A register-offset binding or
enqueue failure at execution is reported, not retried after partial execution.
Opt-in builds check enqueue errors. Other lengths, TUNE and non-four-lane
builds retain baseline scheduling. The plain baseline TUNE build has an
existing undefined `DEFINE_FORWARDP0` macro; this experiment does not repair it.

The structural saving is one dispatch and 8*R*N bytes of nominal coefficient
read/write payload per operand preparation: for two/three primes, 4/6 MiB at
n=18, 8/12 MiB at n=19, 16/24 MiB at n=20 and 64/96 MiB at n=22.
This excludes roots, cache effects and auxiliary build time. It is not a
measured speedup. Preparation frequency, compiler resource use and driver
behavior determine any whole-job benefit.

## Offline reproduction

Requirements already available for these checks were Linux x86-64, Clang 18,
GCC 13.3, Python with NumPy, GMP development files and an OpenCL loader.
No device or driver installation is required. Set `CC_CL` to Clang if it is
not the default; `CXX` and `GMP_STATIC` may override the C++ compiler and GMP
library path. Run from the repository root:

```sh
python3 tests/check_prepare_wide_static.py
python3 tests/check_fused_prepare_wide.py
python3 tests/check_prepare_wide_dispatch.py
```

The static test enumerates complete address ownership for all four lengths,
checks embedded/external equality, compiles 32 CL1.1/1.2 configurations and
compares 16 flag-off preprocessed kernels to the pinned base.
The native CPU adapter compiles the actual OpenCL C functions and supplies
cooperative workgroup barriers; it does not invoke the OpenCL runtime.
GFN-18 and GFN-22 cover two/three primes, 31/32-bit moduli and both inverse
root-table modes. GFN-19/20 additionally cover 3x32 with the inverse table.
Each case compares all first-forward and completed operand words for source
registers 0,1,2, with zero, p-1 and deterministic random coefficients. All
source words must remain identical. The test uses the production workgroup
sizes, including 64/128/256 items for the three terminal kernels.

The synthetic host test intercepts OpenCL entry points. It checks enabled and
disabled dispatch at n=12,17,18,19,20,21,22,23; destination/source bindings and
register-1 offsets; auxiliary build, missing entry, argument, source binding,
query, axis, local-memory, device and per-kernel admission failures; and
reported offset-binding/enqueue errors. It performs no GPU work or arithmetic.
The normal host was also compiled/linked with flag 0 and 1 and queried only
for its version. Invalid flag value 2 was rejected. TUNE syntax passed for
both flag values with the baseline-only macro alias shown below:

```sh
c++ -std=c++17 -DGPU -IKhronos -fsyntax-only src/transform_ocl.cpp \
  -DTUNE '-DDEFINE_FORWARDP0(u)=DEFINE_FORWARDPn(u,0)' \
  -DGENEFER_GPU_FUSED_PREPARE_WIDE=1
```

These checks do not establish GPU correctness.

## Future hardware validation and rollback

Use separate worktrees at the exact base and candidate SHAs listed in the
Draft PR. Record git SHA, compiler, binary hash, driver, device, arithmetic
configuration and selected kernel names. For each, build separately:

```sh
make -B -C genefer -f makefile_linux_x64 BOINC= \
  FLAGS_GPU='-O3 -DGPU -DGENEFER_GPU_FUSED_PREPARE_WIDE=VALUE' ../bin/geneferg
```

Use VALUE=0 for base and candidate control, and VALUE=1 for the candidate.
On NVIDIA and AMD devices test n=18,19,20,22; n=12,17,21,23 are fallback
controls. At n=22, even bases 1024, 1000000, 100000000 and 1200000000 target
the four RNS/width configurations; record actual selection rather than
assuming it. Include source registers 0,1 and the highest allocated register,
repeated preparation of distinct registers, and a resource-refusal shim.
Run from empty directories for embedded source, then repeat with matching
external `ocl/kernel.cl`. Inspect actual group size and local allocation.

Before long validation, compare full register dumps/digests after preparation
and products, including every unselected source register. Exchange checkpoints
both ways at matched iterations. Then complete bounded, identical jobs with
`geneferg -q -n N -b B -d D -f job`, separately `-p`, and matching `-s`/`-c`
artifacts. Require identical verdict, res64/old64 where reported, proof keys,
Gerbicz-Li results and restart behavior. Cross-check each same certificate
against its generating server key with both binaries; server encoding is
randomized. No proof or carry machinery changes are intended.

Only after correctness passes, warm up and alternate at least three paired
windows, recording medians/spread, preparation dispatch counts and time,
whole-iteration time and auxiliary compilation separately. Use available
vendor counters for local/global transactions, registers, spills and occupancy.
Reject on any mismatch, invalid proof, restart failure, unsupported-path change,
enqueue error or repeatable regression. Rebuild with VALUE=0 to roll back;
no data-format migration is needed. These hardware steps have not been run.

## Campaign continuity and provenance

On 6 November 2026, main and all six open PRs were inspected. PR #6 at
`50ffe58503516b57301b59260cf80b3223313bf1` is the verified cherubrock-seb-owned
GFN-12 predecessor. Its source/register approach, admission helpers and native
CPU adapter are reused under the repository's MIT license. This branch starts
from main, not from that experiment, and has its own flag and restricted sizes.
PR #2's n=21 plan, #3's register square core, #4's n=23 square and #5's n=17
lazy arithmetic do not own these preparation paths. Original shared-twiddle
and fused-product branches were left untouched. Combined experiments require
separate validation.

The next bounded review could cover remaining outer64 preparation sizes using
PR #6's ownership and evidence, or investigate another architectural operation
only after checking all then-open PRs. No controller iteration is asserted.

GPU_RUNTIME_TEST=NOT_RUN
GPU_CORRECTNESS=UNVALIDATED
GPU_PERFORMANCE=UNMEASURED
PRODUCTION_READINESS=NOT_ESTABLISHED
