"""Offline compiler, source synchronization and complete address-ownership checks."""
import ast
import os
from pathlib import Path
import subprocess
import tempfile
import numpy as np
from native_opencl.kernel import parameters
ROOT = Path(__file__).resolve().parents[1]
BASE = 'ddc72834fff77b288dcfe19920707f006a3c492b'
source = (ROOT/'ocl/kernel.cl').read_text()
embedded = ''.join(ast.literal_eval(line.rstrip(' \\')) for line in
                  (ROOT/'src/ocl/kernel.h').read_text().splitlines() if line.startswith('"') and line != '"";')
assert embedded == source
for exponent in (18,19,20,22):
    n = 1 << exponent
    outer = 1024 if exponent==22 else 256
    b, m = outer//4, n//(4*outer)
    block = np.arange(m, dtype=np.uint32)
    thread = np.arange(b, dtype=np.uint32)[:,None]
    ki, ko = block + thread*m, block + 4*thread*m
    reads = np.concatenate([(ki+j*b*m).ravel() for j in range(4)])
    writes = np.concatenate([(ko+j*m).ravel() for j in range(4)])
    expected = np.arange(n//4, dtype=np.uint32)
    np.testing.assert_array_equal(np.sort(reads), expected)
    np.testing.assert_array_equal(np.sort(writes), expected)
    for rns in (2,3):
        span = rns*n//4
        last_register = (2**32-span)//span
        assert last_register*span+span-1 <= 2**32-1
        assert (last_register+1)*span+span-1 > 2**32-1
    print(f'PASS n={exponent}: unique in-plane reads/writes and 32-bit offset bound', flush=True)
with tempfile.TemporaryDirectory() as tmp:
    baseline = Path(tmp)/'base.cl'
    baseline.write_bytes(subprocess.check_output(['git','-C',str(ROOT),'show',f'{BASE}:ocl/kernel.cl']))
    clang = os.environ.get('CC_CL','clang')
    for exponent in (18,22):
        for rns in (2,3):
            for wide in (False,True):
                for wi in (False,True):
                    defs, _, _ = parameters(1<<exponent, wide, rns, wi)
                    opts = [f'-D{k}={v}' for k,v in defs.items()]
                    for std in ('CL1.1','CL1.2'):
                        command = [clang,'-x','cl','-target','spir64',f'-cl-std={std}',*opts]
                        subprocess.run([*command,'-DGENEFER_FUSED_PREPARE_WIDE_KERNEL=1',
                                        '-fsyntax-only',str(ROOT/'ocl/kernel.cl')],check=True)
                    command = [clang,'-x','cl','-target','spir64','-cl-std=CL1.2',*opts,'-E','-P']
                    old = subprocess.check_output([*command,str(baseline)])
                    new = subprocess.check_output([*command,str(ROOT/'ocl/kernel.cl')])
                    assert old.split()==new.split()
    print('PASS embedded source, 32 enabled syntax configurations and 16 flag-off token comparisons')
