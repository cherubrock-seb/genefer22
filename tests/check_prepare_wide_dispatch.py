"""Synthetic OpenCL admission and dispatch; no device enumeration or workload."""
import os
from pathlib import Path
import subprocess
import tempfile
root = Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory() as tmp:
    binary = str(Path(tmp)/'dispatch')
    for flag in (0, 1):
        subprocess.run([os.environ.get('CXX','c++'), '-O2', '-std=c++17', '-DGPU',
            f'-DGENEFER_GPU_FUSED_PREPARE_WIDE={flag}', '-I'+str(root/'Khronos'), '-I'+str(root/'src'),
            str(root/'tests/mock_opencl/drive.cpp'), str(root/'tests/mock_opencl/mock.cpp'),
            str(root/'src/transform_ocl.cpp'), os.environ.get('GMP_STATIC','-lgmp'), '-l:libOpenCL.so.1',
            '-o', binary], check=True)
        def run(n, **env):
            return subprocess.run([binary,str(n)], env=dict(os.environ, **env), capture_output=True, text=True)
        for n in (12,17,18,19,20,21,22,23):
            outer = 1024 if n==22 else 256
            name = f'prepare{outer}_0'
            result = run(n)
            assert result.returncode==0, result.stdout
            if flag and n in (18,19,20,22):
                assert f'{name}@2 source@1:{2*(1<<n)//4}' in result.stdout, result.stdout
                assert 'copyp@' not in result.stdout, result.stdout
                # Unchanged square/product still use the ordinary source buffer.
                assert f'forward{outer}_0@1' in result.stdout, result.stdout
                for failure in ('build','missing','arg','source','query','dim','local','device'):
                    result = run(n, FAIL=failure)
                    assert result.returncode==0 and 'copyp@2' in result.stdout and name not in result.stdout, (failure,result.stdout)
                result = run(n, DENY=name)
                assert result.returncode==0 and 'copyp@2' in result.stdout and name not in result.stdout
                for env in ({'FAIL':'offset'}, {'BAD_ENQUEUE':name}):
                    result = run(n, **env)
                    assert result.returncode!=0 and 'ERROR' in result.stdout, result.stdout
            else:
                assert 'copyp@2' in result.stdout and 'prepare256_0' not in result.stdout and 'prepare1024_0' not in result.stdout
        print(f'PASS flag={flag}: dispatch, register binding, fallback and checked errors', flush=True)
