#!/usr/bin/env python3
"""Test exp_i in all precisions with independently compiled numerical checks.

Run: python3 ket/test/exp_i_numerical.py [--sanitize] [--cxx g++]
The subject and linker use existing debug/release optimization settings.
Only the reference/check translation unit uses -O2 (without fast-math).
No simulator build, MPI, or third-party dependency is required.

Run the ordinary matrix as well as --sanitize: UBSan changes code generation
and can mask the GCC 11.4/x86 long-double release regression (#349).
"""

import argparse
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cxx', default='g++')
    parser.add_argument('--sanitize', action='store_true', help='enable UBSan')
    options = parser.parse_args()
    ket = Path(__file__).resolve().parents[1]
    common = [options.cxx, '-std=c++14', '-Wall', '-Wextra', '-pedantic',
              '-I' + str(ket / 'include')]
    sanitizer = ['-fsanitize=undefined', '-fno-sanitize-recover=all'] if options.sanitize else []
    failures = []
    with tempfile.TemporaryDirectory(prefix='ket-exp-i-') as directory:
        build = Path(directory)
        for precision, macro in (('float', 0), ('double', 1), ('long-double', 2)):
            flags = common + [f'-DKET_TEST_PRECISION={macro}'] + sanitizer
            oracle = build / f'{precision}-oracle.o'
            subprocess.run(flags + ['-O2', '-c', str(ket / 'test/exp_i_numerical.cpp'),
                                     '-o', str(oracle)], check=True, timeout=120)
            for mode in ('debug', 'release'):
                optimization = ['-O0'] if mode == 'debug' else ['-Ofast', '-DNDEBUG']
                subject = build / f'{precision}-{mode}.o'
                executable = build / f'{precision}-{mode}'
                subprocess.run(flags + optimization
                               + ['-c', str(ket / 'test/exp_i_subject.cpp'), '-o', str(subject)],
                               check=True, timeout=120)
                subprocess.run([options.cxx, str(oracle), str(subject), '-o', str(executable)]
                               + optimization + sanitizer, check=True, timeout=120)
                print(f'{precision} {mode}:', flush=True)
                result = subprocess.run([str(executable)], timeout=120)
                if result.returncode:
                    failures.append(f'{precision} {mode}')
    if failures:
        raise SystemExit('Failed configurations: ' + ', '.join(failures))
    print('All exp_i numerical configurations passed')


if __name__ == '__main__':
    main()
