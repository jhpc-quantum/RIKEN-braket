#!/usr/bin/env python3
"""Compile/run control type tests and verify invalid APIs fail to compile.

Run from any directory: python3 ket/test/control_types.py [--cxx g++]
"""

import argparse
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cxx', default='g++')
    args = parser.parse_args()
    ket = Path(__file__).resolve().parents[1]
    flags = [args.cxx, '-std=c++14', '-Wall', '-Wextra', '-pedantic',
             '-I' + str(ket / 'include')]
    preamble = '#include <ket/runtime_control.hpp>\nusing Q = ket::qubit<>;\n'
    invalid = {
        'value outside base': 'ket::control<Q, 2> c;',
        'negative template value': 'ket::control<Q, -1> c;',
        'invalid helper value': 'auto c = ket::make_control<2>(Q{0u});',
        'polarity change': 'ket::control<Q> c{ket::control<Q, 0>{Q{0u}}};',
        'runtime value loss': 'ket::control<Q> c{ket::runtime_control<Q>{Q{0u}, 0}};',
        'implicit runtime conversion': 'ket::runtime_control<Q> c = ket::control<Q, 0>{Q{0u}};',
        'implicit target conversion': 'Q q = ket::runtime_control<Q>{Q{0u}, 0};',
        'floating-point value': 'ket::runtime_control<Q> c{Q{0u}, 0.5};',
        'invalid constexpr runtime value': 'constexpr ket::runtime_control<Q> c{Q{0u}, 2};',
    }
    with tempfile.TemporaryDirectory(prefix='ket-control-types-') as directory:
        binary = str(Path(directory) / 'control_types')
        for source in ['control_types.cpp', 'runtime_control_io.cpp']:
            subprocess.run(flags + [str(ket / 'test' / source), '-o', binary], check=True)
            subprocess.run([binary], check=True)
        for name, declaration in invalid.items():
            result = subprocess.run(flags + ['-x', 'c++', '-fsyntax-only', '-'],
                                    input=preamble + declaration + '\n',
                                    text=True, capture_output=True)
            if result.returncode == 0:
                raise AssertionError('Unexpected compilation success: ' + name)
    print('Control type/I/O tests and {} compile-failure checks passed'.format(len(invalid)))


if __name__ == '__main__':
    main()
