#!/usr/bin/env python3

"""Execute floating-point array loops and compare them with reference results.

Example:
  python3 bra/test/qasm2qcx_float_array_loop_numerical.py --bra bra/bin/bra
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

    # Binary fractions make these comparisons exact. The Python loop reads
    # each current element, mutates future elements, and keeps a copied value.
    for size in (1, 2, 4, 6):
        initial = [(i + 1) / 4 for i in range(size)]
        for mode in ('all', 'continue', 'break'):
            values = initial.copy()
            total = copied = 0.0
            position = 0
            for index in range(size):
                value = values[index]
                position += 1
                if mode == 'continue' and value == 0.5:
                    continue
                if mode == 'break' and value > 1:
                    break
                total += value * position
                if position < size:
                    values[position] += 0.25
                values[position - 1] = -value
                copied += value
            transfer = ('if (value == 0.5) { continue; }' if mode == 'continue'
                        else 'if (value > 1) { break; }' if mode == 'break' else '')
            source = f'''array[float, {size}] values = {{{', '.join(map(str, initial))}}};
                float total = 0; float copied = 0; int i = 0;
                for float value in values {{ i += 1; {transfer}
                    total += value * i;
                    if (i < {size}) {{ values[i] += 0.25; }}
                    values[i - 1] = -value; copied += value;
                }}'''
            check_program(bra, source,
                          ('TOTAL31', 'COPIED63', 'I1') + tuple(f'VALUES63:{i}' for i in range(size)),
                          [format(total, 'g'), format(copied, 'g'), str(position)]
                          + [format(value, 'g') for value in values])

    check_program(bra, '''array[float, 2] values = {0.25, 0.5}; float total = 0;
        for float value in values { for float other in values { total += value * other; } }''',
                  ('TOTAL31',), ['0.5625'])

    # Re-entering a loop resets its position, including after a previous break.
    for outer in ('for int outer in [0:2]', 'for int outer in [0:stop]',
                  'for int outer in {0, stop, 1}', 'while (visits < 3)'):
        check_program(bra, 'array[float, 3] values = {0.25, 0.5, 0.75}; '
                      'float total = 0; int stop = 2; int visits = 0; '
                      + outer + ''' { visits += 1; for float value in values {
                          if (value > 0.5) { break; } total += value; } }''',
                      ('TOTAL31', 'VISITS63'), ['2.25', '3'])

    check_program(bra, '''array[float, 2] values = {0.25, 0.5};
        bit[2] flags = "01"; int value = 7; float total = 0;
        for float value in values {
            for bit value in flags { if (value) { total += int(value); } }
            for int value in [0:1] { total += ~value; } total += value;
        } value += 1;''', ('TOTAL31', 'VALUE31'), ['-3.25', '8'])
    check_program(bra, '''array[float, 2] values = {0.25, 0.5}; float total = 0;
        for float values in values { total += values; } total += values[0];''',
                  ('TOTAL31', 'VALUES63:0'), ['1', '0.25'])

    check_program(bra, '''array[float, 2] values = {1, 2}; int total = 0;
        for float value in values {
            for int j in [0:int(value)] { total += j; }
            for int j in {int(value), 0} { total += j; }
        }''', ('TOTAL31',), ['7'])
    check_program(bra, '''array[float, 2] positions = {0, 1};
        array[float, 2] values = {0.25, 0.75}; float total = 0;
        for float position in positions { values[int(position)] += 0.25;
            total += values[int(position)]; }''',
                  ('TOTAL31', 'VALUES63:0', 'VALUES63:1'), ['1.5', '0.5', '1'])

    # A skipped body must not evaluate its runtime accesses or index arithmetic.
    check_program(bra, '''array[float, 2] values = {0, 1}; int bad = 2; int zero = 0;
        bool flag = false; float total = 0;
        if (false) { for float value in values { total += values[bad]; } }
        while (false) { for float value in values { total += values[1 / zero]; } }
        for int outer in [1:0] { for float value in values { total += values[bad]; } }
        for float value in values {
            flag = true || values[bad] > value;
            flag = false && bool(values[1 / zero]);
            continue; total += values[bad];
        } total += 0.5;''', ('TOTAL31', 'FLAG15'), ['0.5', '0'])

    check_program(bra, '''array[float, 2] values = {0, 1.5}; float total = 0;
        int n = 0; uint u = 0; bool flag = false; complex z;
        for float value in values { total += -value + value / 2;
            n += int(value); u += uint[4](value); z = complex(value) + 1.0im;
            total += float(z); flag = bool(value); }''',
                  ('TOTAL31', 'N1', 'U1', 'FLAG15'), ['0.75', '1', '1', '1'])

    for operator, expected in (('==', '0'), ('!=', '2'), ('<', '1'),
                                ('<=', '1'), ('>', '1'), ('>=', '1')):
        check_program(bra, 'array[float, 2] values = {0.5, 1.5}; uint u = 1; '
                      f'int count = 0; for float value in values {{ if (value {operator} u) {{ count += 1; }} }}',
                      ('COUNT31',), [expected])

    for base in ('float', 'float[32]', 'float[64]'):
        check_program(bra, f'array[{base}, 2] values = {{0.25, 0.5}}; float total = 0; '
                      'for float[64] value in values { total += value; }', ('TOTAL31',), ['0.75'])

    # Deterministic measurements detect accidental INT iterator truncation or
    # parameter modification. The source array must retain its original angle.
    for gate in ('rx(theta)', 'ry(-theta)', 'U(theta, 0, 0)'):
        check_program(bra, 'include "stdgates.inc"; qubit q; array[float, 2] angles = {0, pi}; '
                      f'for float theta in angles {{ {gate} q; }} '
                      'bit result = measure q; int n = int(result); int preserved = 0; '
                      'if (angles[1] == pi) { preserved = 1; }', ('N1', 'PRESERVED511'), ['1', '1'])

    check_error(bra, '''array[float, 1] values = {0.5}; float total; int bad = 1;
        for float value in values { total = value + values[bad]; }''', 'assertion failed in ASSERT')
    check_error(bra, '''array[float, 1] values = {0.5}; float total; int zero = 0;
        for float value in values { total = value + values[1 / zero]; }''',
                'integer division by zero in LET')

    print('Floating-point array loop numerical checks passed')


if __name__ == '__main__':
    main()
