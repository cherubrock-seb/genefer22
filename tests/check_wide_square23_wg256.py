"""Compare GFN-23 square schedules on CPU, including an independent sparse oracle."""
import argparse
from pathlib import Path
import tempfile

import numpy as np
from native_opencl.kernel import compile_kernels, parameters, twiddles
from native_opencl.polynomial import expected_product, sparse


N = 1 << 23
CONTROL = (("forward64_0", 64), ("forward64_11", 64), ("square2048", 128),
           ("backward64_11", 64), ("backward64_0", 64))
CANDIDATE = (("forward1024_0", 256), ("square8192_wg256", 256), ("backward1024_0", 256))


def calculate(lib, plan, data, w):
    out = data.copy()
    for name, local in plan:
        lib.run_groups(getattr(lib, name), None, out.ctypes.data, None, w.ctypes.data, 0, data.size//(32 if name=="square8192_wg256" else 16), local)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rns", type=int, choices=(2, 3), default=2)
    parser.add_argument("--wide", action="store_true")
    parser.add_argument("--use-wi", action="store_true")
    args = parser.parse_args()
    defs, primes, generators = parameters(N, args.wide, args.rns, args.use_wi)
    defs.update(GENEFER_WIDE_SQUARE23_WG256_KERNEL=1, MAX_WG_SZ=256)
    w = twiddles(N, primes, generators)
    terms = {0: -3, 511: 31, 2047: 5, 8191: -7, 8192: 11, N//2: -13, N-1: 17}
    rng = np.random.default_rng(2308192)
    with tempfile.TemporaryDirectory() as tmp:
        lib = compile_kernels(Path(tmp), defs)
        data = sparse(N, primes, terms)
        result = calculate(lib, CANDIDATE, data, w)
        np.testing.assert_array_equal(result, expected_product(N, primes, terms, terms))
        np.testing.assert_array_equal(result, calculate(lib, CONTROL, data, w))
        print(f"sparse square: PASS ({result.size} raw residues)", flush=True)
        for plane, p in zip(data, primes):
            plane[:] = rng.integers(0, p, N, dtype=np.uint32)
        result = calculate(lib, CANDIDATE, data, w)
        np.testing.assert_array_equal(result, calculate(lib, CONTROL, data, w))
        print(f"dense square: PASS ({result.size} raw residues)", flush=True)


if __name__ == "__main__":
    main()
