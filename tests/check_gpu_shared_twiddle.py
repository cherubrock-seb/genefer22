#!/usr/bin/env python3
"""Offline checks of extracted OpenCL arithmetic; never loads an OpenCL runtime."""
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / 'ocl/kernel.cl').read_text()


def function(name):
    match = re.search(r'INLINE\s+\w+\s+' + re.escape(name) + r'\([^\n]*\)\s*\{', SOURCE)
    if not match:
        raise AssertionError('Missing source function: ' + name)
    depth = 1
    end = match.end()
    while depth:
        depth += (SOURCE[end] == '{') - (SOURCE[end] == '}')
        end += 1
    # Only adapt OpenCL vector construction syntax to a C++ value type.
    return SOURCE[match.start():end].replace('(uint2_32)(', 'uint2_32(').replace('(uint4_32)(', 'uint4_32(')


def macro(name):
    start = SOURCE.index('#define ' + name + '(')
    end = SOURCE.index('\n}\n', start) + 3
    return SOURCE[start:end]


PREAMBLE = r'''
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>
using uint_32 = uint32_t;
using int_32 = int32_t;
using uint_64 = uint64_t;
#define INLINE static inline
struct uint2_32 {
    uint32_t s0, s1;
    uint2_32() = default;
    uint2_32(uint32_t a) : s0(a), s1(a) {}
    uint2_32(uint32_t a, uint32_t b) : s0(a), s1(b) {}
};
struct uint4_32 {
    uint32_t s0, s1, s2, s3;
    uint2_32 s01, s23;
    uint4_32() = default;
    uint4_32(uint32_t a) : uint4_32(a, a, a, a) {}
    uint4_32(uint32_t a, uint32_t b, uint32_t c, uint32_t d)
        : s0(a), s1(b), s2(c), s3(d), s01(a,b), s23(c,d) {}
    uint4_32(uint2_32 a, uint2_32 b) : uint4_32(a.s0,a.s1,b.s0,b.s1) {}
};
uint32_t mul_hi(uint32_t a, uint32_t b) { return uint32_t((uint64_t(a)*b) >> 32); }
uint32_t power(uint32_t a, uint32_t e, uint32_t p) {
    uint32_t r = 1;
    while(e) { if(e&1) r=uint64_t(r)*a%p; a=uint64_t(a)*a%p; e>>=1; }
    return r;
}
void require(bool ok) { if(!ok) throw std::runtime_error("arithmetic mismatch"); }
std::array<uint32_t,4> lanes(uint4_32 a) { return {a.s0,a.s1,a.s2,a.s3}; }
'''

FUNCTIONS = ['addmod', 'submod', 'mulmod', 'addmod2', 'submod2', 'mulmod2',
             'addmod4', 'submod4', 'mulmod4', 'mulmod_preconditioned',
             'mulmod2_broadcast', 'mulmod4_broadcast']
BUTTERFLIES = ['_forward4x4', '_backward4x4', '_forward4x4_0', '_backward4x4_0']

HARNESS = r'''
int main() {
#ifdef IS32
    const uint32_t primes[] = {(125u<<25)+1, (243u<<24)+1, (235u<<24)+1};
#else
    const uint32_t primes[] = {(127u<<24)+1, (63u<<25)+1, (15u<<27)+1};
#endif
    std::mt19937_64 rng(0x67656e6566657232ULL);
    size_t tuples=0, butterflies=0;
    for (uint32_t p : primes) {
        uint32_t q=1;
        for(int i=0;i<5;++i) q*=2u-p*q; // inverse modulo 2^32, independent of kernel constants
        require(p*q == 1);
        uint2_32 pq(p,q);
        const uint32_t R=uint64_t(1ULL<<32)%p, ri=power(R,p-2,p);
        auto oracle=[&](uint32_t a,uint32_t b) { return uint64_t(uint64_t(a)*b%p)*ri%p; };
        auto check=[&](uint4_32 a,uint32_t w) {
            const auto out=lanes(mulmod4_broadcast(a,w,pq));
            const auto input=lanes(a);
            auto out2=mulmod2_broadcast(a.s01,w,pq);
            require(out2.s0==out[0] && out2.s1==out[1]);
            for(int lane=0;lane<4;++lane) {
                require(out[lane]<p && out[lane]==oracle(input[lane],w));
                require(out[lane]==mulmod(input[lane],w,pq));
                require(out[lane]==mulmod_preconditioned(input[lane],w,w*q,pq));
            }
            ++tuples;
        };
        const std::vector<uint32_t> edge={0,1,2,p/2-1,p/2,p/2+1,p-3,p-2,p-1,R};
        for(auto a:edge) for(auto w:edge) check(uint4_32(a,p-1-a,(a+1)%p,(p-1)/2),w);
        for(size_t i=0;i<20000;++i) {
            check(uint4_32(rng()%p,rng()%p,rng()%p,rng()%p),rng()%p);
            uint4_32 z[4];
            uint32_t reference[4][4];
            for(int j=0;j<4;++j) {
                z[j]=uint4_32(rng()%p,rng()%p,rng()%p,rng()%p);
                auto a=lanes(z[j]);
                for(int lane=0;lane<4;++lane) reference[j][lane]=a[lane];
            }
            const uint32_t w1=1+rng()%(p-1), w2[2]={uint32_t(1+rng()%(p-1)),uint32_t(1+rng()%(p-1))};
            auto inv=[&](uint32_t w) { return uint32_t(p-uint64_t(uint64_t(R)*R%p)*power(w,p-2,p)%p); };
            const uint32_t wi1=inv(w1), wi2[2]={inv(w2[0]),inv(w2[1])};
            _forward4x4(pq,z,w1,w2);
            auto ref=std::array<std::array<uint32_t,4>,4>{};
            for(int j=0;j<4;++j) for(int lane=0;lane<4;++lane) ref[j][lane]=reference[j][lane];
            auto fwd=[&](int a,int b,uint32_t w) {
                for(int lane=0;lane<4;++lane) {
                    uint64_t t=oracle(ref[b][lane],w), old=ref[a][lane];
                    ref[a][lane]=(old+t)%p; ref[b][lane]=(old+p-t)%p;
                }
            };
            fwd(0,2,w1); fwd(1,3,w1); fwd(0,1,w2[0]); fwd(2,3,w2[1]);
            for(int j=0;j<4;++j) require(lanes(z[j])==ref[j]);
            _backward4x4(pq,z,wi1,wi2);
            for(int j=0;j<4;++j) {
                auto a=lanes(z[j]);
                for(int lane=0;lane<4;++lane) require(a[lane]==uint64_t(4)*reference[j][lane]%p);
            }
            ++butterflies;
        }
    }
    std::cout << "tuples=" << tuples << " butterfly_roundtrips=" << butterflies << " PASS\n";
}
'''


def main():
    subprocess.run(['python3', str(ROOT / 'tools/embed_opencl.py'), '--check'], check=True)
    arithmetic = '\n\n'.join(function(name) for name in FUNCTIONS)
    macros = '\n\n'.join(macro(name) for name in ['FWD2v4', 'BCK2v4'])
    butterflies = '\n\n'.join(function(name) for name in BUTTERFLIES)
    source = PREAMBLE + arithmetic + '\n' + macros + butterflies + HARNESS
    with tempfile.TemporaryDirectory(prefix='genefer-gpu-arithmetic-') as folder:
        cpp = Path(folder) / 'extracted.cpp'
        cpp.write_text(source)
        for wide in (False, True):
            for enabled in (0, 1):
                exe = Path(folder) / f'check-{wide}-{enabled}'
                cmd = shlex.split(os.environ.get('CXX', 'g++')) + [
                    '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror',
                    '-fsanitize=undefined', '-fno-sanitize-recover=all',
                    f'-DGENEFER_GPU_SHARED_TWIDDLE={enabled}']
                if wide:
                    cmd += ['-DIS32=1']
                subprocess.run(cmd + [str(cpp), '-o', str(exe)], check=True)
                print(f'IS32={int(wide)} SHARED_TWIDDLE={enabled}', flush=True)
                subprocess.run([str(exe)], check=True)
    print('STATIC_VALIDATION=PASS; GPU_EXACTNESS=PENDING_CHERUB; GPU_PERFORMANCE=PENDING_CHERUB')


if __name__ == '__main__':
    main()
