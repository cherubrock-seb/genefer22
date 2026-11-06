# Fused operand preparation for GFN-12

Base ddc72834fff77b288dcfe19920707f006a3c492b.
Build with GENEFER_GPU_FUSED_PREPARE64=1 to enable the experiment.
The default is 0. This first candidate covers fixed-plan n=12, VSIZE=4.

The initial copy into _zp and forward64_0 are replaced by prepare64_0.
It reads the selected register directly from _z and writes the ordinary
forward output into _zp. The subsequent fwd64p uses the existing implementation.
This removes one dispatch and one full coefficient read/write pair per
operand preparation, nominally 8*R*N bytes (64 KiB or 96 KiB at n=12).
The effect on whole-job time depends on how often operands are prepared.

For each modulus the baseline input address ki becomes
src*R*N/4 + 4*mid + ki in uint4 units. The source register index is checked
against the allocation count. ki and its three radix-4 strides use the same
mapping as forward64_0, within the selected N-coefficient plane. Destination
ko uses the original map within _zp. Separate buffer allocations guarantee
non-aliasing. Every register in _z is read-only. Root indices, entry Montgomery
conversion, barriers, forward arithmetic and output layout use the existing
MIT-licensed helpers. The arithmetic bounds and all six prime definitions
are inherited exactly. Both RNS counts and inverse table modes are supported.

The optional program builds separately. Failed builds, missing kernels,
workgroup/axis/local-memory limits and argument binding select the copy path.
Opt-in builds check enqueue errors. Other lengths and TUNE use their baseline
schedule. The base's plain TUNE build has an existing DEFINE_FORWARDP0 typo.

Offline reproduction on Linux x86-64 with Clang, C++17 and Python/NumPy:

    python3 tests/check_fused_prepare64.py

The native CPU adapter executes actual OpenCL C. The test compares every
first-forward and fully prepared output word, and checks all source registers
remain identical for register indices 0,1,2. It covers two/three primes,
31/32-bit moduli and both inverse-table modes.

Future GPU validation uses fresh base and candidate worktrees, each built with:

    make -B -C genefer -f makefile_linux_x64 BOINC= FLAGS_GPU='-O3 -DGPU -DGENEFER_GPU_FUSED_PREPARE64=VALUE' ../bin/geneferg

Use VALUE=0 for the control and VALUE=1 for the candidate. Run from separate
empty directories for embedded source, then repeat with matching external
kernel files. On NVIDIA and AMD OpenCL devices, exercise n=12 with even bases
1024, 30000000, 100000000 and 1200000000 for the four arithmetic paths.
Use n=13 as a fallback control. Compare all registers after preparation,
prepared products, Gerbicz-Li results, res64/old64, proof keys and checkpoint
exchange in both directions. For each generated certificate, both binaries
must reproduce that certificate's server key. Server encoding is randomized.
Use -q, -p, -s and -c modes with the matching -n/-b/-d/-f job parameters.
Warm up and alternate at least three identical iteration windows, recording
median/spread, setup dispatch counts, full iteration and optional build time.
Reject on any mismatch, failure or repeatable regression. Rebuild with 0 to
roll back. GPU validation and benchmarks are deferred.

Campaign handoff on 6 November 2026: drafts #1 through #5 were inspected.
This candidate is independent of their twiddle, arithmetic and square plans.
The next step is extending the same out-of-place preparation to outer256
and outer1024, after checking resources and all register mappings.

GPU_RUNTIME_TEST=NOT_RUN
GPU_CORRECTNESS=UNVALIDATED
GPU_PERFORMANCE=UNMEASURED
PRODUCTION_READINESS=NOT_ESTABLISHED
