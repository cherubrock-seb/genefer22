# GPU campaign journal · 5 November 2026

Repository: cherubrock-seb/genefer22. Base for all three independent candidates:
`ddc72834fff77b288dcfe19920707f006a3c492b`. The fork and galloty/genefer22
shared this main SHA at discovery. All work below is confined to fork branches.

| Draft | Branch | Implementation milestone | Question for hardware |
| --- | --- | --- | --- |
| [#2](https://github.com/cherubrock-seb/genefer22/pull/2) | perf/gpu-gfn21-short-plan | `f3633522afbabff37388415f2636b910cdf49421`, checked enqueue `dfe8e490da861118904f9fadab0b0073684ab38a` | Do two fewer global passes offset the bigger workgroups at n=21? |
| [#3](https://github.com/cherubrock-seb/genefer22/pull/3) | perf/gpu-register-square-core | `16f2294cd4914a83072b6260254ac38d0e566772`, optional driver admission `c580787aa0b56f2e2820c9a478198bedfeb879bf` | Does retaining 16 residues privately improve the four even-length square tails without spills? |
| [#4](https://github.com/cherubrock-seb/genefer22/pull/4) | perf/gpu-gfn23-wide-square | `c2ba18be4b2c2da752e97b45df41a559b3fe343e` | Can eligible devices amortize the 32 KiB tile and 512-work-item group at n=23? |

Each PR carries its exact review head, native CPU test commands, validation
contract and individual build flag. All are default-off; combined flags are
outside the validation matrix. The existing shared-twiddle draft #1 and
fused-products branch are maintained separately.

**Dependencies.** NVIDIA and AMD OpenCL correctness, Gerbicz–Li checks,
proof/certificate interoperability, checkpoint exchange, driver counters
and end-to-end timing are open. Plain TUNE on the shared base has an undefined
DEFINE_FORWARDP0 macro. Production fixed-plan host builds are available;
the GPU hardware runs remain deferred.

**Next candidate.** Inspect the copy into the transformed multiplicand.
initMultiplicand first copies R×N coefficients from the selected register to
_zp, then its first forward pass rereads them. An out-of-place first forward
could read directly from that register and write to _zp. Preserve source
registers and independent transformed-operand layout. Measure setup separately
from repeated square time; this opportunity matters only to operations that
actually prepare an operand.
