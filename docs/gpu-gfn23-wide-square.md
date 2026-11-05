# GFN-23 three-launch square

Base `ddc72834fff77b288dcfe19920707f006a3c492b`. Build the GPU host with
`-DGENEFER_GPU_WIDE_SQUARE23=1` to try the experiment. The default is 0.
Only fixed-plan GFN-23 squaring with four vector lanes can use this layout.
The normal OpenCL program builds first. A separate program compiles the
candidate on eligible devices. The host checks both outer kernel limits,
the device maximum, axis-zero limit, local memory and compiled tail limit,
then checks argument binding. A failed optional build, missing entry point
or rejected limit retains the fixed plan before any coefficient is touched.
Opt-in builds check argument and enqueue errors. Verbose startup identifies
the selected route. Rebuild with 0 to roll back.

| Stage | Fixed plan | Candidate |
| --- | --- | --- |
| Start in ordinary coefficient layout | forward64_0, forward64_11 | forward1024_0 |
| Workgroup-local terminal square | square2048 | square8192 |
| End in raw inverse layout | backward64_11, backward64_0 | backward1024_0 |
| CRT and carry | normalize1, normalize2 | normalize1, normalize2 |

For N=2^23, R=2 or 3 primes, avoiding two full coefficient read/write passes
removes 16RN bytes of nominal global payload: 256 or 384 MiB per square. This
excludes twiddles and caching. Including normal carry, seven launches become
five. Register use, size-dependent bandwidth, local traffic and reduced
residency determine the performance tradeoff. No new global buffer is allocated. An eligible
instance may incur an additional OpenCL program build at startup.

## Local tile and exact transform

64 × 64 × 2048 = 1024 × 8192 = 2^23. The original combined outer forward
stages perform six regular radix-4 levels including the entry stage. The
candidate outer1024 performs five. The new terminal covers the missing level
and its inverse. One 512-item group holds 2048 uint4 values (32 KiB) belonging
to one modulus. Each item manages four vectors (16 coefficients). Local
strides on entry are 512, 128, 32, 8 and 2; square_8 then completes the
terminal core and the inverse uses the same strides in reverse order.

For stride s and lane t=qs+r, 0≤r<s, the four local addresses are
4qs+r+ks for k=0,1,2,3. For every stage they partition 0..2047. The first
and last global accesses use lane+512k within the group's disjoint 2048-vector
block. N/16 is divisible by 512, so a group never spans two RNS planes.
Producer/consumer exchanges use the original workgroup fences. The new tail
has ten barriers per group; the old 2048 tail has eight. No hardware subgroup width is
assumed. Both root layouts follow the original stage index divided by s;
the first and last outer kernels still perform the original entry conversion
and unnormalized inverse. All arithmetic calls retain the original unsigned
Montgomery operations and intermediate bounds. The three 31-bit and three
32-bit primes support the required roots. Square returns the same raw
representation before CRT, carry and checkpoints. General multiplication and
multiplicand preparation retain their matched baseline layout.

This P0 candidate is independent of the other experiments. The idea is to
move an exact radix-4 level across a kernel boundary using Genefer22's own
MIT-licensed helpers. No floating-point FFT, PFA7 algorithm or upstream code
was imported.

## Offline reproduction

On Linux/x86-64 with Clang, a C++17 compiler and Python/NumPy run:

```sh
python3 tests/check_wide_square23.py --rns 2
python3 tests/check_wide_square23.py --rns 2 --wide
python3 tests/check_wide_square23.py --rns 3
python3 tests/check_wide_square23.py --rns 3 --wide
python3 tests/check_wide_square23.py --rns 3 --wide --use-wi
```

The check translates the actual OpenCL C kernels to native CPU code. It compares
every raw coefficient of a dense square and a sparse square to the fixed plan.
A separate sparse polynomial oracle checks sign wrap at x^N = −1, including
inputs straddling 8192 and the last coefficient, and the raw inverse scale
(N/2)·2^32 mod p. The draft PR gives the exact revisions, build commands,
GPU residue, proof, restart, capability and timing gates. Plain `-DTUNE` has
the same pre-existing `DEFINE_FORWARDP0` build issue as its base.

```text
GPU_RUNTIME_TEST = NOT_RUN
GPU_CORRECTNESS = UNVALIDATED
GPU_PERFORMANCE = UNMEASURED
PRODUCTION_READINESS = NOT_ESTABLISHED
```
