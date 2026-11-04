# GPU fused Montgomery product reductions

- Repository: cherubrock-seb/genefer22
- Branch: perf/gpu-fused-products-20261104
- Base: ddc72834fff77b288dcfe19920707f006a3c492b
- Independent of the shared-twiddle branch; do not stack before separate A/B tests.
- Candidate HEAD: current branch tip; capture `git rev-parse HEAD` with evidence.
- GPU_VALIDATION=DEFERRED_TO_CHERUB
- GPU_EXACTNESS=PENDING_CHERUB
- GPU_PERFORMANCE=PENDING_CHERUB
- PRODUCTION_READINESS=NOT_ESTABLISHED

## Hypothesis and range proof

The terminal SQR2/SQR2N and MUL2/MUL2N operations reduce each product separately,
then add/subtract the canonical results. Linearity permits one Montgomery
reduction of a sum/difference of two wide products, provided its input is first
brought into [0,pR), where R=2^32 and p is any of the six existing moduli.

For canonical a,b,c,d<p, let U=a*b and V=c*d. Each product fits unsigned 64 bits.
For a sum, T=U+V<2*p^2<2*p*R. If p<R/2, T<pR and fits 64 bits without adjustment.
For p>R/2, T may need 65 bits. Compute t=T mod R^2 and detect carry as t<U.
Subtract pR once when there is carry OR t>=pR. When carry occurs, unsigned
subtraction wrapping modulo R^2 produces exactly T-pR, which lies in [0,pR).
No carry may be silently dropped.

For a difference, T=U-V lies in (-p^2,p^2). If U<V, add pR in unsigned 64-bit
arithmetic to the wrapped difference; otherwise leave it unchanged. The result
again lies in [0,pR). No signed overflow or signed shift is required.

For 0<=t<pR, let m=lo(t)*q mod R, q=p^-1 mod R. Then
t-m*p is divisible by R and -(p)<(t-m*p)/R<p. The existing submod of hi(t)
and hi(m*p) yields the canonical Montgomery residue. The result is
(a*b +/- c*d)*R^-1 mod p, identical to combining individually reduced products.

SQR2 first computes B=Mont(b,b), then MontSum(a,a,B,w) and Mont(2*a mod p,b).
This changes four reductions to three and 16 limb products to 14.
MUL2 first computes B=Mont(b,d), then MontSum(a,c,B,w) and MontSum(a,d,b,c).
This changes five reductions to three and 20 limb products to 16.
The negative-twiddle variants use MontDifference for the first result.

These source-level product reductions (12.5% in terminal square arithmetic,
20% in terminal multiply arithmetic) are not end-to-end speedup predictions.
Wider additions/comparisons and register pressure can outweigh the saving.
The benefit is limited to terminal quadratic operations, not the entire NTT.

## Checkpoint queue

1. Implement default-off GENEFER_GPU_FUSED_PRODUCTS and preserve baseline macros.
2. Test source-extracted helpers and all four terminal macros against independent
   wide-integer modular oracles, especially 65-bit sums and negative differences.
3. Regenerate embedded source; record static checks and missing prerequisites.
4. Open a focused Draft PR with exact Cherub validation commands.

No GPU probing, driver/runtime installation, SSH, GPU execution or CPU
performance optimization is authorized by this work item.
