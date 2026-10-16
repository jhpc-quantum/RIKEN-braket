#!/usr/bin/env python3

"""Execute complex array loops and compare results with Python reference loops.

Example:
  python3 bra/test/qasm2qcx_complex_array_loop_numerical.py --bra bra/bin/bra
"""

import argparse
import pathlib
import resource

from qasm2qcx_complex_array_numerical import check_complex_program
from qasm2qcx_integer_array_numerical import check_error
from qasm2qcx_runtime_for_numerical import check_program


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    bra = parser.parse_args().bra
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    for size in (1, 2, 4):
        for mode in ('all', 'continue', 'break'):
            initial = [complex(i + 1, i - 1) for i in range(size)]
            values = initial.copy()
            total = encoded = 0j
            visits = 0
            for index in range(size):
                z = values[index]
                visits += 1
                if mode == 'continue' and int(z.real) % 2 == 0:
                    continue
                if mode == 'break' and int(z.real) == 2:
                    break
                values[index] = -z
                if visits < size:
                    values[visits] += 2 + 1j
                total += z
                encoded = encoded * 2 + z
            transfer = ('if (uint(z) % 2 == 0) { continue; }' if mode == 'continue'
                        else 'if (uint(z) == 2) { break; }' if mode == 'break' else '')
            literal = ', '.join(f'{z.real} + ({z.imag}im)' for z in initial)
            check_complex_program(bra, f'''array[complex, {size}] a = {{{literal}}};
                int i = 0; complex total = 0; complex encoded = 0;
                for complex z in a {{ i += 1; {transfer}
                    a[i - 1] = -z; if (i < {size}) {{ a[i] += 2 + 1.0im; }}
                    total += z; encoded = encoded * 2 + z;
                }}''', ('TOTAL31', 'ENCODED127', ':COMPLEX:I1') + tuple(f'A1:{i}' for i in range(size)),
                                  [total, encoded, complex(visits)] + values)

    check_complex_program(bra, '''array[complex, 3] a = {1 + 1.0im, 2 + 2.0im, 3 + 3.0im};
        complex total = 0; for complex z in a { for complex other in a {
            if (uint(other) == 2) { continue; } total += z * other;
            if (uint(other) == 3) { break; }
        } }''', ('TOTAL31',), [48j])
    for outer in ('for int outer in [0:2]', 'for int outer in [0:stop]',
                  'for int outer in {0, stop, 1}', 'while (visits < 3)'):
        check_complex_program(bra, 'array[complex, 2] a = {1 + 1.0im, 2 + 2.0im}; '
            'complex total = 0; int visits = 0; int stop = 2; ' + outer + ''' {
                visits += 1; for complex z in a { total += z; a[0] += 1; break; }
            }''', ('TOTAL31', 'A1:0', ':COMPLEX:VISITS63'), [6 + 3j, 4 + 1j, 3 + 0j])

    check_complex_program(bra, '''const int N = 2; array[complex, N] a = {1.0im, 2.0im};
        complex total = 0; for complex N in a { total += N; }
        total += N; for complex a in a { total += a; } total += a[0];''', ('TOTAL31',), [2 + 7j])
    check_complex_program(bra, '''array[complex, 2] a = {1 + 1.0im, 2 + 2.0im};
        array[uint[4], 1] u = {1}; array[float, 1] f = {0.5}; array[bool, 1] flags = {true};
        bit[1] bits = "1"; complex z = 3.0im; complex total = 0;
        for complex z in a { for uint[4] z in u { total += z; }
            for float z in f { total += z; } for bool z in flags { total += z; }
            for bit z in bits { total += complex(z); } total += z; }
        total += z;''', ('TOTAL31', 'Z1'), [10 + 6j, 3j])

    check_complex_program(bra, '''array[complex, 2] a = {1 + 1.0im, 2 + 2.0im}; complex total = 0;
        for complex z in a { for int j in [0:uint(z)] { total += z; }
            for int j in {uint(z), 1} { total += j; } }''', ('TOTAL31',), [13 + 8j])
    check_complex_program(bra, '''array[complex, 1] a = {1.0im}; complex total = 0;
        int bad = 1; int zero = 0;
        if (false) { for complex z in a { total += a[bad]; } }
        while (false) { for complex z in a { total += a[1 / zero]; } }
        for int outer in [1:0] { for complex z in a { total += a[bad]; } }
        for complex z in a { if (true || uint(a[bad]) > 0) { total += z; }
            if (false && uint(a[1 / zero]) > 0) { total += 99; }
            continue; total += a[bad]; }''', ('TOTAL31',), [1j])

    check_program(bra, '''include "stdgates.inc"; qubit q; array[complex, 2] a = {1 + 2.0im, 0 + 3.0im};
        for complex z in a { rx(float(z) * pi) q; gphase(float(z)); }
        bit result = measure q; int n = int(result);''', ('N1',), ['1'])
    check_error(bra, '''array[complex, 1] a = {1.0im}; int bad = 1; complex total;
        for complex z in a { total = z + a[bad]; }''', 'assertion failed in ASSERT')
    check_error(bra, '''array[complex, 1] a = {1.0im}; int zero = 0; complex total;
        for complex z in a { total = z + a[1 / zero]; }''', 'integer division by zero in LET')

    print('Complex array loop numerical checks passed')


if __name__ == '__main__':
    main()
