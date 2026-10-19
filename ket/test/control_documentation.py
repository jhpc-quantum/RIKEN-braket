#!/usr/bin/env python3
"""Compile-check the control-value examples in docs/ket.md; run the non-MPI example."""

import argparse
from pathlib import Path
import re
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cxx', default='g++')
    parser.add_argument('--mpi', action='store_true')
    parser.add_argument('--mpicxx', default='mpic++')
    parser.add_argument('--yampi', type=Path)
    args = parser.parse_args()
    ket = Path(__file__).resolve().parents[1]
    document = (ket.parent / 'docs' / 'ket.md').read_text(encoding='utf-8')
    examples = {}
    for name in ['nonmpi', 'mpi']:
        matches = re.findall(r'<!-- control-example: ' + name + r' -->\s*```c\+\+\n(.*?)\n```',
                             document, flags=re.DOTALL)
        if len(matches) != 1:
            raise AssertionError('Expected exactly one documentation example: ' + name)
        examples[name] = matches[0]
    with tempfile.TemporaryDirectory(prefix='ket-control-docs-') as directory:
        binary = str(Path(directory) / 'example')
        for standard in ['c++14', 'c++17']:
            for masks in [[], ['-DKET_USE_BIT_MASKS_EXPLICITLY']]:
                flags = ['-std=' + standard, '-O0', '-pthread', '-I' + str(ket / 'include')] + masks
                subprocess.run([args.cxx] + flags + ['-x', 'c++', '-', '-o', binary],
                               input=examples['nonmpi'], text=True, check=True)
                subprocess.run([binary], check=True)
                if args.mpi:
                    yampi = args.yampi or ket.parent.parent / 'yampi' / 'include'
                    subprocess.run([args.mpicxx] + flags + ['-I' + str(yampi), '-x', 'c++',
                                                          '-fsyntax-only', '-'],
                                   input=examples['mpi'], text=True, check=True)
    print('Control documentation examples passed (C++14/C++17, both indexing modes)')


if __name__ == '__main__':
    main()
