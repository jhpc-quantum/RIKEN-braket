#!/usr/bin/env python3

"""Execute converted floating-point array declarations and element accesses.

Example:
  python3 bra/test/qasm2qcx_float_array_numerical.py --bra bra/bin/bra
"""

import argparse
import pathlib
import resource

from qasm2qcx_integer_array_numerical import check_error
from qasm2qcx_runtime_for_numerical import check_program


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    bra = parser.parse_args().bra
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    check_program(bra, 'array[float, 2] values;',
                  ('VALUES63:0', 'VALUES63:1'), ['0', '0'])
    check_program(bra, '''int n = 2; uint u = 3; float f = 0.25; bool b = true;
        array[float[64], 6] values = {n, u, f, f + 0.5, b, float(n)};''',
                  tuple(f'VALUES63:{i}' for i in range(6)), ['2', '3', '0.25', '0.75', '1', '2'])

    for size in (1, 2, 3):
        values = [0.25 * (i + 1) for i in range(size)]
        literal = ', '.join(map(str, values))
        for index in range(-size, size):
            for expression in (str(index), 'i'):
                check_program(bra, f'array[float, {size}] values = {{{literal}}}; '
                              f'int i = {index}; float x = values[{expression}];',
                              ('X1', 'I1'), [str(values[index]), str(index)])
        for index in (-size - 1, size):
            for statement in ('float x = values[i];', 'values[i] = 0.5;', 'values[i] += 0.5;'):
                check_error(bra, f'array[float, {size}] values; int i = {index}; ' + statement,
                            'assertion failed in ASSERT')

    for operator, expected in (('=', '2'), ('+=', '2.5'), ('-=', '-1.5'),
                                ('*=', '1'), ('/=', '0.25')):
        for index in (-2, -1, 0, 1):
            for expression in (str(index), 'i'):
                result = ['0.5', '0.5']
                result[index] = expected
                check_program(bra, f'array[float, 2] values = {{0.5, 0.5}}; '
                              f'int i = {index}; uint u = 2; values[{expression}] {operator} u;',
                              ('VALUES63:0', 'VALUES63:1', 'I1'), result + [str(index)])

    check_program(bra, '''array[float, 2] values = {0.25, 0.75}; int i = -2; int j = -1;
        values[i] += values[j] * 2; values[j] = values[i] / 2;''',
                  ('VALUES63:0', 'VALUES63:1', 'I1', 'J1'), ['1.75', '0.875', '-2', '-1'])
    check_program(bra, '''array[int, 2] indices = {1, 0}; int i = 0;
        array[float, 2] values = {0.25, 0.75};
        values[indices[i]] += values[indices[i + 1]];
        array[float, 2] copies = {values[0], values[indices[i]]};''',
                  ('VALUES63:0', 'VALUES63:1', 'COPIES63:0', 'COPIES63:1'),
                  ['0.25', '1', '0.25', '1'])

    print('Floating-point array numerical checks passed')


if __name__ == '__main__':
    main()
