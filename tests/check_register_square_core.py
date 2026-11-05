"""Run the original and register-core square kernels on identical synthetic CPU tiles."""
import argparse
from pathlib import Path
import tempfile

import numpy as np
from native_opencl.kernel import compile_kernels, parameters, twiddles


TAIL_GROUPS = {64: 64, 256: 64, 1024: 64, 4096: 256}
FLAG = "GENEFER_GPU_REGISTER_SQUARE_CORE"


def compare(n, rns, wide, use_wi):
    defs, primes, generators = parameters(n, wide, rns, use_wi)
    w = twiddles(n, primes, generators)
    rng = np.random.default_rng(4096064)
    dense = np.stack([rng.integers(0, p, n, dtype=np.uint32) for p in primes])
    for i, p in enumerate(primes):
        dense[i, ::31] = 0
        dense[i, 1::31] = p - 1
    with tempfile.TemporaryDirectory() as base_dir, tempfile.TemporaryDirectory() as opt_dir:
        base = compile_kernels(Path(base_dir), dict(defs, **{FLAG: 0}))
        opt = compile_kernels(Path(opt_dir), dict(defs, **{FLAG: 1}))
        for tail, local in TAIL_GROUPS.items():
            control, candidate = dense.copy(), dense.copy()
            items = n // 16
            # Cover all groups for small inputs. Include both plane boundaries
            # and interior twiddle indices on larger production lengths.
            if n <= 65536:
                ranges = [(0, items*rns)]
            else:
                per_plane = items // local
                positions = sorted({0, per_plane//2, per_plane-1})
                ranges = [(plane*items + pos*local, local) for plane in range(rns) for pos in positions]
            for lib, out, suffix in ((base, control, ""), (opt, candidate, "_register")):
                kernel = getattr(lib, f"square{tail}{suffix}")
                for start, count in ranges:
                    lib.run_groups(kernel, None, out.ctypes.data, None, w.ctypes.data, start, count, local)
            np.testing.assert_array_equal(candidate, control)
            if np.array_equal(control, dense):
                raise AssertionError("test did not exercise the square")
            print(f"square{tail}: PASS ({sum(count for _, count in ranges)*16} transformed words)", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, choices=range(12, 24), default=16)
    parser.add_argument("--rns", type=int, choices=(2, 3), default=2)
    parser.add_argument("--wide", action="store_true")
    parser.add_argument("--use-wi", action="store_true")
    args = parser.parse_args()
    compare(1 << args.n, args.rns, args.wide, args.use_wi)
