"""Compile only source-extracted numeric CLI guards and GPU selector; no GPU calls."""
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
main = (ROOT/'src/main.cpp').read_text()
header = (ROOT/'src/transformGPU.h').read_text()
selector = (ROOT/'src/transform_ocl.cpp').read_text()
# Fail closed if the source structure no longer matches this audited extraction.
start = main.index('b = static_cast<uint32_t>(std::atoi(bstr.c_str()));')
base_guards = main[main.index('#if !defined(CYCLO)',start):main.index('\n\t\t\t}',start)]
nstart = main.index('n = static_cast<uint32_t>(std::atoi(nstr.c_str()));')
nguards = main[main.index('if (n <',nstart):main.index('\n\t\t\t}',nstart)]
assert 'b must not be a power of two' in base_guards and 'b must be even' in base_guards
assert 'n > 23' in nguards and 'n < 12' in nguards
macros=[]
primes={}
for name in ('P1S','P2S','P1U','P2U'):
    match=re.search(r'^#define\s+'+name+r'\s+\((\d+) \* \(uint32\(1\) << (\d+)\) \+ 1\)',header,re.M)
    assert match,name
    primes[name]=(int(match[1])<<int(match[2]))+1
    macros.append(match[0])
body=selector[selector.index('\tif (b * static_cast'):selector.index('\treturn pTransform;')]
for rns,wide,code in ((2,'false',0),(2,'true',1),(3,'false',2),(3,'true',3)):
    body,count=re.subn(r'pTransform = new transformGPUs<'+str(rns)+r', '+wide+r'>\([^;]+;',f'return {code};',body)
    assert count==1
assert 'new ' not in body and 'pTransform' not in body
cpp='''#include <cstdint>
#include <cstddef>
#include <iostream>
#include <stdexcept>
using uint32=uint32_t;
'''+ '\n'.join(macros)+'\nint check(uint32_t b,uint32_t n){\n'+base_guards+'\n'+nguards+'\n'+body+'''
throw std::runtime_error("unreachable");
}
int main(){uint32_t b,n;while(std::cin>>b>>n){try{int result=check(b,n);std::cout<<result<<'\\n';}catch(const std::runtime_error&){std::cout<<-1<<'\\n';}}}
'''

def expected(b,n):
    if not (12<=n<=23 and 1024<=b<=2000000000 and b%2==0 and b&(b-1)):
        return -1
    for mode,suffix in enumerate(('S','U')):
        if b*b < (primes['P1'+suffix]*primes['P2'+suffix]//2)//(1<<n):
            return mode
    return 2 if b<=1000000000 else 3

matrix=[]
cases=[]
for n in range(12,24):
    threshold=(primes['P1S']*primes['P2S']//2)//(1<<n)
    b=math.isqrt(threshold-1)+1
    b+=b%2
    if b&(b-1)==0: b+=2
    row=(1026,b,100000000,1200000000)
    assert [expected(x,n) for x in row]==[0,1,2,3]
    matrix.append((n,row))
    cases += [(x,n) for x in row]
    for suffix in ('S','U'):
        t=(primes['P1'+suffix]*primes['P2'+suffix]//2)//(1<<n)
        edge=math.isqrt(t)
        cases += [(x,n) for x in range(edge-2,edge+4)]
cases += [(b,12) for b in (0,1022,1024,1025,1026,2048,1000000000,1000000002,2000000000,2000000002)]
cases += [(1026,n) for n in (0,11,24,32)]
with tempfile.TemporaryDirectory() as tmp:
    src=Path(tmp)/'check.cpp';exe=Path(tmp)/'check'
    src.write_text(cpp)
    subprocess.run([os.environ.get('CXX','c++'),'-std=c++17','-Wall','-Wextra','-Werror',str(src),'-o',str(exe)],check=True)
    results=subprocess.check_output([str(exe)],input=''.join(f'{b} {n}\n' for b,n in cases),text=True).splitlines()
    assert len(results)==len(cases)
    for case,result in zip(cases,results):
        assert int(result)==expected(*case),(case,result,expected(*case))
print(f'PASS {len(cases)} numeric guard/selector cases; no Genefer binary or OpenCL API executed')
print('n | 2x31 | 2x32 | 3x31 | 3x32')
for n,row in matrix:
    print(' | '.join(map(str,(n,*row))))
print('1024 rejected; 1026 accepted in 2x31 at every supported CLI exponent')
