"""CPU-only exact comparison of the production operand preparation kernels."""
from pathlib import Path
import tempfile
import numpy as np
from native_opencl.kernel import compile_kernels, parameters, twiddles

# Both outer kernels cover every arithmetic mode. Intermediate terminal sizes
# also receive a full-length comparison; no GPU/OpenCL runtime is used.
cases = [(n, r, wide, wi) for n in (18, 22) for r in (2, 3)
         for wide in (False, True) for wi in (False, True)]
cases += [(n, 3, True, True) for n in (19, 20)]
for exponent, rns, wide, wi in cases:
    n = 1 << exponent
    outer = 1024 if exponent == 22 else 256
    tail = n // outer
    defs, primes, generators = parameters(n, wide, rns, wi)
    defs['GENEFER_FUSED_PREPARE_WIDE_KERNEL'] = 1
    roots = twiddles(n, primes, generators)
    rng = np.random.default_rng(2561024)
    source = np.stack([np.stack([rng.integers(0, p, n, dtype=np.uint32)
                       for p in primes]) for _ in range(3)])
    source[:, :, ::17] = 0
    for i, p in enumerate(primes):
        source[:, i, 1::17] = p - 1
    original = source.copy()
    wrappers = '\n'.join('''
void prepare_test%d(__global uint4 *z, __global const uint4 *input, __global const uint *w)
{ prepare%d_0(z, w, input, %du); }
''' % (register, outer, register*rns*n//4) for register in (0, 1, 2))
    with tempfile.TemporaryDirectory() as tmp:
        lib = compile_kernels(Path(tmp), defs, wrappers)
        for register in (0, 1, 2):
            expected = source[register].copy()
            actual = np.full_like(expected, 0xdeadbeef)
            lib.run_groups(getattr(lib, f'forward{outer}_0'), None,
                           expected.ctypes.data, None, roots.ctypes.data, 0, rns*n//16, 256)
            lib.run_groups(None, getattr(lib, f'prepare_test{register}'),
                           actual.ctypes.data, source.ctypes.data, roots.ctypes.data, 0, rns*n//16, 256)
            np.testing.assert_array_equal(actual, expected)
            # The unchanged terminal transform must consume the same layout.
            group = tail * int(defs.get(f'BLK{tail}', 1)) // 16
            for data in (expected, actual):
                lib.run_groups(getattr(lib, f'fwd{tail}p'), None, data.ctypes.data,
                               None, roots.ctypes.data, 0, rns*n//16, group)
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(source, original)
    print(f'PASS n={exponent} rns={rns} wide={wide} inverse_table={wi} registers=0,1,2', flush=True)
