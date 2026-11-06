"""CPU-only source, OpenCL compiler and barrier-structure checks."""
import ast
import os
from pathlib import Path
import re
import subprocess
import tempfile
from native_opencl.kernel import parameters
root=Path(__file__).resolve().parents[1]
flag='GENEFER_GPU_REGISTER_PREPARE_TAIL'
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
                    defs,_,_=parameters(65536,wide,rns,wi)
                    defs.update(VSIZE=lanes,LVSIZE={1:0,2:1,4:2}[lanes])
                    opts=[f'-D{k}={v}' for k,v in defs.items()]
                    cmd=[clang,'-x','cl','-target','spir64',*opts]
                    for std in ('CL1.1','CL1.2'):
                        subprocess.run([*cmd,f'-cl-std={std}',f'-D{flag}=1','-fsyntax-only',str(root/'ocl/kernel.cl')],check=True)
                    pp=[*cmd,'-cl-std=CL1.2','-E','-P']
                    baseline=subprocess.check_output([*pp,str(original)]).split()
                    assert baseline==subprocess.check_output([*pp,f'-D{flag}=0',str(root/'ocl/kernel.cl')]).split()
                    if lanes!=4:
                        assert baseline==subprocess.check_output([*pp,f'-D{flag}=1',str(root/'ocl/kernel.cl')]).split()
    print('PASS embedded source; 48 enabled syntax configurations; 24 flag-off and 16 non-four-lane fallback token comparisons',flush=True)
    defs,_,_=parameters(65536,True,3,True)
    counts=[]
    for enabled in (0,1):
        ir=Path(tmp)/f'kernel{enabled}.ll'
        subprocess.run([clang,'-x','cl','-target','spir64','-cl-std=CL1.2','-O2','-S','-emit-llvm',
            *(f'-D{k}={v}' for k,v in defs.items()),f'-D{flag}={enabled}',str(root/'ocl/kernel.cl'),'-o',str(ir)],check=True)
        text=ir.read_text()
        counts.append([])
        for tail in (64,256,1024,4096):
            name=f'fwd{tail}p'+('_register' if enabled else '')
            body=re.search(r'define[^\n]* @'+name+r'\([^}]+^}',text,re.M)[0]
            counts[-1].append(len(re.findall(r'call[^\n]*barrier',body)))
    assert counts==[[2,3,4,5],[1,2,3,4]],counts
    print(f'PASS SPIR64 -O2 static barriers {counts[0]} -> {counts[1]}',flush=True)
