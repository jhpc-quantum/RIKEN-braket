#!/usr/bin/env python3

"""Execute complex array initialization, access, and arithmetic through bra.

Example:
  python3 bra/test/qasm2qcx_complex_array_numerical.py --bra bra/bin/bra
"""

import argparse
import cmath
import pathlib
import resource
import subprocess

from qasm2qcx_integer_array_numerical import check_error
from qasm2qcx_runtime_for_numerical import check_program, qasm2qcx


def check_complex_program(bra: pathlib.Path, source: str,
                          outputs: tuple[str, ...], expected: list[complex]) -> None:
    lines = qasm2qcx.convert('OPENQASM 3; ' + source)
    lines.extend(f'PRINTLN {output}' for output in outputs)
    result = subprocess.run([str(bra)], input='\n'.join(lines) + '\n', check=True,
                            text=True, capture_output=True, timeout=30)
    values = [complex(*map(float, line.strip().strip('()').split(',')))
              for line in result.stdout.splitlines()]
    if len(values) != len(expected) or any(
            not cmath.isclose(value, reference, rel_tol=1e-5, abs_tol=1e-6)
            for value, reference in zip(values, expected)):
        raise RuntimeError(f'Complex array mismatch\nsource:\n{source}\n'
                           f'expected: {expected}\nstdout:\n{result.stdout}')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    bra = parser.parse_args().bra
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

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

    for size in (1, 2, 3):
        values = [complex(i + 1, 2 - i) for i in range(size)]
        literal = ', '.join(f'{value.real} + ({value.imag}im)' for value in values)
        for index in range(-size, size):
            check_complex_program(bra, f'array[complex, {size}] a = {{{literal}}}; int i = {index}; '
                                  f'complex z = a[i]; complex other = a[{index}]; a[i] = -a[i];',
                                  ('Z1', 'OTHER31', f'A1:{index % size}'),
                                  [values[index], values[index], -values[index]])

    operations = {'+': lambda a, b: a + b, '-': lambda a, b: a - b,
                  '*': lambda a, b: a * b, '/': lambda a, b: a / b}
    for real in range(-2, 3):
        for imaginary in range(-2, 3):
            value = complex(real, imaginary)
            for operator, operation in operations.items():
                check_complex_program(bra,
                    f'array[complex, 2] a = {{{real} + ({imaginary}.0im), 1 + 1.0im}}; '
                    f'int i = 0; int j = -1; complex z = a[i] {operator} a[j]; '
                    f'a[i] {operator}= a[j];', ('Z1', 'A1:0', 'A1:1'),
                    [operation(value, 1 + 1j), operation(value, 1 + 1j), 1 + 1j])

    check_complex_program(bra, '''array[complex, 2] a = {1 + 2.0im, 3 + 4.0im}; int i = 0;
        a[i] += a[1] * (a[i] + 2.0im); a[1] = a[i] - a[1];
        array[complex, 2] b = {a[i], -a[1]};''',
                          ('A1:0', 'A1:1', 'B1:0', 'B1:1'), [-12 + 18j, -15 + 14j, -12 + 18j, 15 - 14j])
    check_complex_program(bra, '''array[int, 2] indices = {1, 0}; int i = 0;
        array[complex, 2] a = {1 + 2.0im, 3 + 4.0im};
        a[indices[i]] *= a[indices[1]];''', ('A1:0', 'A1:1'), [1 + 2j, -5 + 10j])
    check_program(bra, '''array[complex, 2] a = {1.5 + 2.0im, 2.5 - 1.0im}; int i = 0;
        float f = float(a[i]); uint u = uint[4](a[1]);
        array[int, 2] values = {7, 9}; int selected = values[uint(a[i])];''',
                  ('F1', 'U1', 'SELECTED255'), ['1.5', '2', '9'])
    check_complex_program(bra, '''array[complex, 2] a = {1.0im, 2.0im}; int stop = 1;
        for int i in [0:stop] { a[i] += i; }
        int j = 0; while (j < 2) { a[j] *= 2; j += 1; }''',
                          ('A1:0', 'A1:1'), [2j, 2 + 4j])
    check_complex_program(bra, '''array[complex, 1] a = {1.0im}; int bad = 1; int zero = 0;
        if (false) { a[bad] += a[1 / zero]; } while (false) { a[bad] = 99; }
        bool off = false && uint(a[bad]) > 0;
        bool on = true || uint(a[1 / zero]) > 0;''', ('A1:0',), [1j])
    check_program(bra, '''include "stdgates.inc"; qubit q; array[complex, 1] a = {1 + 2.0im};
        int i = 0; rx(float(a[i]) * pi) q; gphase(float(a[i]));
        bit result = measure q; int n = int(result);''', ('N1',), ['1'])

    for size in (1, 2, 3):
        for index in (-size - 1, size):
            for statement in ('complex z = a[i];', 'a[i] = 1.0im;', 'a[i] += a[0];'):
                check_error(bra, f'array[complex, {size}] a; int i = {index}; ' + statement,
                            'assertion failed in ASSERT')
    check_error(bra, 'array[complex, 1] a; int zero = 0; complex z = a[1 / zero];',
                'integer division by zero in LET')
    check_error(bra, 'array[complex, 1] a; int bad = 1; a[0] = a[bad];',
                'assertion failed in ASSERT')

    print('Complex array numerical checks passed')


if __name__ == '__main__':
    main()
