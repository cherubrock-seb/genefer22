#!/usr/bin/env python3
"""Regenerate the embedded GPU source without creating an OpenCL device."""
import argparse
from pathlib import Path


def render(source):
    preamble = source[:source.index("*/") + 2]
    header = preamble + '\n\n#pragma once\n\n#include <cstdint>\n\n'
    header += 'static const char * const src_ocl_kernel = \\\n'
    for line in source.splitlines():
        escaped = line.replace('\\', '\\\\').replace('"', '\\"')
        header += '"' + escaped + '\\n" \\\n'
    return header + '"";\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    target = root / 'src/ocl/kernel.h'
    expected = render((root / 'ocl/kernel.cl').read_text())
    if args.check:
        if target.read_text() != expected:
            raise SystemExit('Embedded OpenCL source is stale; run tools/embed_opencl.py')
        print('Embedded OpenCL source: PASS')
    else:
        target.write_text(expected)


if __name__ == '__main__':
    main()
