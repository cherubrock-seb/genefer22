"""Compare out-of-place preparation with copied-input preparation on native CPU."""
from pathlib import Path
import tempfile
import numpy as np
from native_opencl.kernel import compile_kernels, parameters, twiddles

N = 4096
for rns in (2, 3):
    for wide in (False, True):
        for wi in (False, True):
            defs, primes, generators = parameters(N, wide, rns, wi)
            defs["GENEFER_FUSED_PREPARE64_KERNEL"] = 1
            w = twiddles(N, primes, generators)
            rng = np.random.default_rng(64012)
            source = np.stack([np.stack([rng.integers(0, p, N, dtype=np.uint32)
                               for p in primes]) for _ in range(3)])
            source[:, :, ::17] = 0
            for i, p in enumerate(primes):
                source[:, i, 1::17] = p-1
            original = source.copy()
            for register in (0, 1, 2):
                wrapper = """
void prepare_test(__global uint4 *z, __global const uint4 *input, __global const uint *w)
{ prepare64_0(z, w, input, %du); }
""" % (register*rns*N//4)
                with tempfile.TemporaryDirectory() as tmp:
                    lib = compile_kernels(Path(tmp), defs, wrapper)
                    expected = source[register].copy()
                    actual = np.full_like(expected, 0xdeadbeef)
                    lib.run_groups(lib.forward64_0, None, expected.ctypes.data, None, w.ctypes.data, 0, rns*N//16, 64)
                    lib.run_groups(None, lib.prepare_test, actual.ctypes.data, source.ctypes.data, w.ctypes.data, 0, rns*N//16, 64)
                    np.testing.assert_array_equal(actual, expected)
                    for data in (expected, actual):
                        lib.run_groups(lib.fwd64p, None, data.ctypes.data, None, w.ctypes.data, 0, rns*N//16, 64)
                    np.testing.assert_array_equal(actual, expected)
                    np.testing.assert_array_equal(source, original)
            print(f"PASS rns={rns} wide={wide} inverse_table={wi} registers=0,1,2", flush=True)
