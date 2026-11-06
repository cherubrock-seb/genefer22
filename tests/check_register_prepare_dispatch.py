"""Exercise real host routing against a synthetic OpenCL API, without a GPU."""
import os
from pathlib import Path
import subprocess
import tempfile
root=Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory() as tmp:
    binary=str(Path(tmp)/'dispatch')
    for flag in (0,1):
        subprocess.run([os.environ.get('CXX','c++'),'-std=c++17','-O2','-DGPU',
            f'-DGENEFER_GPU_REGISTER_PREPARE_TAIL={flag}', '-I'+str(root/'Khronos'),'-I'+str(root/'src'),
            str(root/'tests/mock_opencl/drive.cpp'),str(root/'tests/mock_opencl/mock.cpp'),str(root/'src/transform_ocl.cpp'),
            os.environ.get('GMP_STATIC','-lgmp'),'-l:libOpenCL.so.1','-o',binary],check=True)
        def run(n,**env):
            return subprocess.run([binary,str(n)],env=dict(os.environ,**env),capture_output=True,text=True)
        for n in (11,12,13,14,15,16,17,18,19,20,21,22,23):
            normal=run(n)
            assert normal.returncode==0,normal.stdout
            assert 'copyp@2' in normal.stdout,normal.stdout
            if flag and n in (12,14,16,18,20,22):
                tail={12:64,14:256,16:1024,18:1024,20:4096,22:4096}[n]
                name=f'fwd{tail}p_register'
                assert name+'@2' in normal.stdout,normal.stdout
                assert '_register@1' not in normal.stdout,normal.stdout
                for failure in ('build','missing','arg','query','dim','device','local'):
                    result=run(n,FAIL=failure)
                    assert result.returncode==0 and '_register@' not in result.stdout,(failure,result.stdout)
                    assert f'fwd{tail}p@2' in result.stdout,result.stdout
                result=run(n,DENY=name)
                assert result.returncode==0 and f'fwd{tail}p@2' in result.stdout,result.stdout
                result=run(n,BAD_ENQUEUE=name)
                assert result.returncode!=0 and 'ERROR' in result.stdout,result.stdout
            else:
                assert '_register@' not in normal.stdout,normal.stdout
        print(f'PASS flag={flag}: expected routing and applicable fallback/error checks at all fixed exponents',flush=True)
