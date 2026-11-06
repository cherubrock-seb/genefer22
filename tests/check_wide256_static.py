"""CPU-only source, OpenCL compiler and barrier-structure checks."""
import ast
import os
from pathlib import Path
import re
import subprocess
import tempfile
from native_opencl.kernel import parameters
root=Path(__file__).resolve().parents[1]
flag='GENEFER_WIDE_SQUARE23_WG256_KERNEL'
source=(root/'ocl/kernel.cl').read_text()
embedded=''.join(ast.literal_eval(s.rstrip(' \\')) for s in (root/'src/ocl/kernel.h').read_text().splitlines() if s.startswith('"') and s!='"";')
assert embedded==source
with tempfile.TemporaryDirectory() as tmp:
    original=Path(tmp)/'base.cl'
    original.write_bytes(subprocess.check_output(['git','-C',str(root),'show','ddc72834fff77b288dcfe19920707f006a3c492b:ocl/kernel.cl']))
    clang=os.environ.get('CC_CL','clang')
    for lanes in (1,2,4):
        for rns in (2,3):
            for wide in (False,True):
                for wi in (False,True):
                    defs,_,_=parameters(1<<23,wide,rns,wi)
                    defs.update(VSIZE=lanes,LVSIZE={1:0,2:1,4:2}[lanes])
                    opts=[f'-D{k}={v}' for k,v in defs.items()]
                    cmd=[clang,'-x','cl','-target','spir64',*opts]
                    for std in ('CL1.1','CL1.2'):
                        subprocess.run([*cmd,f'-cl-std={std}',f'-D{flag}=1','-fsyntax-only',str(root/'ocl/kernel.cl')],check=True)
                    pp=[*cmd,'-cl-std=CL1.2','-E','-P']
                    baseline=subprocess.check_output([*pp,str(original)]).split()
                    assert baseline==subprocess.check_output([*pp,str(root/'ocl/kernel.cl')]).split()
                    if lanes!=4:
                        assert baseline==subprocess.check_output([*pp,f'-D{flag}=1',str(root/'ocl/kernel.cl')]).split()
    print('PASS embedded source; 48 enabled syntax configurations; 24 flag-off and 16 non-four-lane fallback token comparisons',flush=True)

for stride in (512,128,32,8,2):
    halves=[]
    for part in (0,1):
        addresses=[]
        for physical in range(256):
            lane=physical+part*256
            slot=(lane//stride)*(4*stride)+lane%stride
            addresses += [slot+j*stride for j in range(4)]
        assert len(addresses)==len(set(addresses))==1024
        halves.append(set(addresses))
    assert not (halves[0]&halves[1])
    assert halves[0]|halves[1]==set(range(2048))
for rns in (2,3):
    n=1<<23
    assert rns*n//32 % 256==0
    assert (n//32)//256==n//8192
    assert (rns-1)*(n//4)+(n//8192-1)*2048+2047==rns*n//4-1
print('PASS all butterfly partitions, logical lanes, plane and last-tile bounds')
