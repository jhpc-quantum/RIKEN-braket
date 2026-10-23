#!/usr/bin/env python3

"""Build exact ANGLEBITS oracle checks across precisions and optimization modes.

  python3 bra/test/classical_function_angle_numerical.py [--sanitize]
"""

import argparse
import pathlib
import resource
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cxx', default='g++')
    parser.add_argument('--sanitize', action='store_true')
    options = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    root = pathlib.Path(__file__).resolve().parents[2]
    common = [options.cxx, '-std=c++14', '-Wall', '-Wextra', '-DBRA_NO_MPI',
              '-I' + str(root / 'bra/include'), '-I' + str(root / 'ket/include')]
    sanitizer = ['-fsanitize=undefined', '-fno-sanitize-recover=all'] if options.sanitize else []

    def run(command):
        subprocess.run(command, check=True, timeout=120)

    with tempfile.TemporaryDirectory(prefix='braket-angle-kernel-') as directory:
        build = pathlib.Path(directory)
        for precision, macro in (('float', 2), ('double', 1), ('long-double', 0)):
            flags = common + [f'-DBRA_REAL_TYPE={macro}'] + sanitizer
            oracle = build / f'{precision}-oracle.o'
            run(flags + ['-O2', '-c', str(root / 'bra/test/classical_function_angle_numerical.cpp'), '-o', str(oracle)])
            for mode in ('debug', 'release', 'mixed-release'):
                evaluator = build / f'{precision}-{mode}.o'
                executable = build / f'{precision}-{mode}'
                optimization = ['-Ofast', '-DNDEBUG'] if mode == 'release' else ['-O0']
                link = ['-O0'] if mode == 'debug' else ['-Ofast', '-DNDEBUG']
                run(flags + optimization + ['-c', str(root / 'bra/src/classical_function.cpp'), '-o', str(evaluator)])
                run([options.cxx, str(oracle), str(evaluator), '-o', str(executable)] + link + sanitizer)
                run([str(executable)])
                print(f'{precision} {mode}: passed', flush=True)


if __name__ == '__main__':
    main()
