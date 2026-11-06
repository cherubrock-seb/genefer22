"""Synthetic OpenCL dispatch/resource tests; never calls a physical device."""
import os
from pathlib import Path
import subprocess
import tempfile
root=Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory() as tmp:
    binaries=[]
    for flag in (0,1):
        binary=str(Path(tmp)/f'host{flag}');binaries.append(binary)
        subprocess.run([os.environ.get('CXX','c++'),'-O2','-std=c++17','-DGPU',
            f'-DGENEFER_GPU_WIDE_SQUARE23_WG256={flag}','-I'+str(root/'Khronos'),'-I'+str(root/'src'),
            str(root/'tests/mock_opencl/drive.cpp'),str(root/'tests/mock_opencl/mock.cpp'),str(root/'src/transform_ocl.cpp'),
            os.environ.get('GMP_STATIC','-lgmp'),'-l:libOpenCL.so.1','-o',binary],check=True)
    def run(flag,n,**env):
        return subprocess.run([binaries[flag],str(n)],env=dict(os.environ,**env),capture_output=True,text=True)
    base=run(0,23);assert base.returncode==0
    for env in ({},{'LIMIT':'256'}):
        result=run(1,23,**env)
        assert result.returncode==0 and 'square8192_wg256@1[524288/256]' in result.stdout,result.stdout
        assert result.stdout.split('copyp')[1]==base.stdout.split('copyp')[1],result.stdout
    failures=[{'FAIL':s} for s in ('build','missing','arg','query','dim','device','local')]
    failures += [{'DENY':s} for s in ('forward1024_0','backward1024_0','square8192_wg256')]
    for env in failures:
        result=run(1,23,**env)
        assert result.returncode==0 and result.stdout==run(0,23,**env).stdout,(env,result.stdout)
    result=run(1,23,BAD_ENQUEUE='square8192_wg256')
    assert result.returncode!=0 and 'ERROR' in result.stdout,result.stdout
    for n in (12,17,21,22):
        a,b=run(0,n),run(1,n)
        assert a.returncode==b.returncode==0 and a.stdout==b.stdout
    print('PASS 256-item admission and half-size tail launch; 10 fallback cases; enqueue error; unchanged multiply/preparation; four size controls')
