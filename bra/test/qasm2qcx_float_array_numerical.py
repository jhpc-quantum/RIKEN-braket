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

    check_program(bra, '''array[float, 2] values = {0.5, 1.5}; int i = 1;
        float x = -values[i] + values[0] * 2 - values[i] / 2;
        int n = int(values[i]); uint u = uint[4](values[i]);
        complex z = complex(values[i]) + 1.0im;
        float realpart = float(z); bool b = bool(values[0]);''',
                  ('X1', 'N1', 'U1', 'REALPART255', 'B1'), ['-1.25', '1', '1', '1.5', '1'])

    # Mixed numeric comparisons must retain the fractional part of REAL values.
    for operator, expected in (('==', '0'), ('!=', '1'), ('<', '0'),
                                ('<=', '0'), ('>', '1'), ('>=', '1')):
        for rhs in ('1', 'u', 'values[0]'):
            check_program(bra, 'array[float, 2] values = {1, 1.5}; int i = 1; uint u = 1; '
                          f'bool b = values[i] {operator} {rhs}; int visits = 0; '
                          f'if (values[i] {operator} {rhs}) {{ visits = 1; }}',
                          ('B1', 'VISITS63'), [expected, expected])

    check_program(bra, '''array[float, 2] values = {0, 0.5}; int bad = 2; int zero = 0;
        bool b = false && values[bad] > 0;
        b = true || bool(values[1 / zero]);
        if (false) { values[bad] += 1; }
        while (false) { values[bad] = values[bad]; }
        for int i in [1:0] { values[bad] *= values[bad]; }
        if (bool(values[1]) && !bool(values[0])) { values[0] = 0.25; }''',
                  ('VALUES63:0', 'VALUES63:1', 'B1'), ['0.25', '0.5', '1'])

    check_program(bra, '''array[float, 3] values = {0.25, 0.5, 0.75}; int stop = 2;
        for int i in [0:2] { values[i] = -values[i]; }
        for int i in [0:stop] { if (values[i] < 0) { values[i] *= -1; } }
        int j = 0; while (values[j] < 1) { values[j] += 0.25;
            if (values[j] == 1) { break; } continue; }
        for int i in {0, stop} { values[i] += 0.25; }''',
                  ('VALUES63:0', 'VALUES63:1', 'VALUES63:2'), ['1.25', '0.5', '1'])
    check_program(bra, '''array[float, 2] values = {0, 2}; int i = 1; int total = 0;
        for int j in [int(values[0]):int(values[i])] { total += j; }
        for int j in {int(values[0]), int(values[i])} { total += j; }''',
                  ('TOTAL31',), ['5'])
    check_program(bra, '''array[float, 2] values = {0.25, 0.5}; float x = 0; int stop = 1;
        for int values in [0:stop] { x += float(values); } x += values[0];''',
                  ('X1', 'VALUES63:0'), ['1.25', '0.25'])

    for gate in ('rx(values[i])', 'ry(-values[i])', 'U(values[i], values[0], values[0])'):
        check_program(bra, 'include "stdgates.inc"; qubit q; int i = 1; '
                      'array[float, 2] values = {0, pi}; '
                      f'{gate} q; bit result = measure q; int n = int(result);', ('N1',), ['1'])

    check_error(bra, 'array[float, 1] values; int zero = 0; float x = values[1 / zero];',
                'integer division by zero in LET')
    check_error(bra, 'array[float, 1] values; int zero = 0; values[1 / zero] = 0.5;',
                'integer division by zero in LET')

    print('Floating-point array numerical checks passed')


if __name__ == '__main__':
    main()
