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
