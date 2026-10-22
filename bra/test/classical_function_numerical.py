#!/usr/bin/env python3

"""Build/run standalone APPLY math tests in all precisions and build modes.

  python3 bra/test/classical_function_numerical.py
  python3 bra/test/classical_function_numerical.py --sanitize

Only a C++14 compiler, Boost, and repository headers are required; no simulator
build or MPI is needed. Evaluator and link flags mirror the existing Makefile.
An additional mixed mode checks unoptimized code with release startup's
floating-point environment, as occurs in partly instrumented simulator builds.
"""

import argparse
import pathlib
import resource
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cxx', default='g++')
    parser.add_argument('--sanitize', action='store_true', help='enable UBSan')
    options = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    root = pathlib.Path(__file__).resolve().parents[2]
    common = [options.cxx, '-std=c++14', '-Wall', '-Wextra', '-DBRA_NO_MPI',
              '-I' + str(root / 'bra/include'), '-I' + str(root / 'ket/include')]
    sanitizer = ['-fsanitize=undefined', '-fno-sanitize-recover=all'] if options.sanitize else []

    def run(command):
        subprocess.run(command, check=True, timeout=120)

    with tempfile.TemporaryDirectory(prefix='braket-apply-math-') as directory:
        build = pathlib.Path(directory)
        for precision, macro in (('float', 2), ('double', 1), ('long-double', 0)):
            flags = common + [f'-DBRA_REAL_TYPE={macro}'] + sanitizer
            test_object = build / f'{precision}-test.o'
            # Compile the oracle without fast-math, independently of the
            # evaluator flags, so optimizations cannot hide failed checks.
            run(flags + ['-O2', '-c', str(root / 'bra/test/classical_function_numerical.cpp'),
                         '-o', str(test_object)])
            for mode in ('debug', 'release', 'mixed-release'):
                optimization = ['-Ofast', '-DNDEBUG'] if mode == 'release' else ['-O0']
                link_optimization = ['-O0'] if mode == 'debug' else ['-Ofast', '-DNDEBUG']
                evaluator = build / f'{precision}-{mode}.o'
                executable = build / f'{precision}-{mode}'
                run(flags + optimization + ['-c', str(root / 'bra/src/classical_function.cpp'),
                                            '-o', str(evaluator)])
                run([options.cxx, str(test_object), str(evaluator), '-o', str(executable)]
                    + link_optimization + sanitizer)
                run([str(executable)])
                print(f'{precision} {mode}: passed', flush=True)


if __name__ == '__main__':
    main()
