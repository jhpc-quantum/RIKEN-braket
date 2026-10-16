#!/usr/bin/env python3

"""Execute complex array initialization through the converter and bra.

Example:
  python3 bra/test/qasm2qcx_complex_array_numerical.py --bra bra/bin/bra
"""

import argparse
import pathlib

from qasm2qcx_runtime_for_numerical import check_program


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    bra = parser.parse_args().bra

    for base in ('complex', 'complex[float[32]]', 'complex[float[64]]'):
        for real, imaginary in ((0, 0), (1, 0), (0, 1), (0, -1), (2, 3), (-2, -3)):
            check_program(bra, f'array[{base}, 1] a = {{{real} + ({imaginary}.0im)}};',
                          ('A1:0',), [f'({real},{imaginary})'])

    check_program(bra, '''int n = -2; uint u = 3; float f = 0.5; bool ready = true; bit b = 1;
        complex z = 2 + 3.0im; const complex C = -2 - 1.0im;
        array[complex, 8] a = {n, u, f, ready, complex(b), z, C, z + f};''',
                  tuple(f'A1:{i}' for i in range(8)),
                  ['(-2,0)', '(3,0)', '(0.5,0)', '(1,0)', '(1,0)', '(2,3)', '(-2,-1)', '(2.5,3)'])
    check_program(bra, '''complex z = 1 + 2.0im;
        array[complex, 4] a = {z + 1, -z, z * z, z / 2}; z = 99;''',
                  tuple(f'A1:{i}' for i in range(4)), ['(2,2)', '(-1,-2)', '(-3,4)', '(0.5,1)'])
    check_program(bra, '''const int N = 2; array[complex[float[64]], N] a = {1.0im, -1.0im};
        for int N in [0:1] {}''', ('A1:0', 'A1:1'), ['(0,1)', '(0,-1)'])
    check_program(bra, '''include "stdgates.inc"; qubit q; x q; bit b = measure q;
        array[complex, 2] a = {complex(b), complex(!b)};''',
                  ('A1:0', 'A1:1'), ['(1,0)', '(0,0)'])
    check_program(bra, '''array[int, 2] values = {2, 4}; int i = 1;
        array[complex, 2] a = {values[i] + 1.0im, complex(values[0])};''',
                  ('A1:0', 'A1:1'), ['(4,1)', '(2,0)'])
    check_program(bra, '''array[int, 1] values = {2}; int bad = 1; int zero = 0;
        array[complex, 3] a = {true || values[bad] > 0, false && bool(1 / zero), 2 > 1};''',
                  ('A1:0', 'A1:1', 'A1:2'), ['(1,0)', '(0,0)', '(1,0)'])

    print('Complex array numerical checks passed')


if __name__ == '__main__':
    main()
