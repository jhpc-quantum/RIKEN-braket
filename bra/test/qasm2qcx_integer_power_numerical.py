#!/usr/bin/env python3

"""Compare folded and runtime integer powers against bounded Python results.

Example:
  python3 bra/test/qasm2qcx_integer_power_numerical.py --bra bra/bin/bra
"""

import argparse
import math
import pathlib
import resource

from qasm2qcx_integer_array_numerical import check_error
from qasm2qcx_runtime_for_numerical import check_program, qasm2qcx


def check_assignments_and_control_flow(bra: pathlib.Path) -> None:
    converter = qasm2qcx.QASM2QCXConverter
    for base in range(-4, 5):
        for exponent in range(5):
            expected = str(pow(base, exponent))
            check_program(bra, f'''int x = {base}; int y = {base}; int n = {exponent};
                x **= n; y = y ** n;''', ('X1', 'Y1'), [expected, expected])
    for bits in (4, converter.QCX_UINT_WIDTH):
        mask = (1 << bits) - 1
        for base in (0, 1, 3, mask):
            for exponent in (0, 1, 2, 7):
                expected = str(pow(base, exponent, 1 << bits))
                check_program(bra, f'''uint[{bits}] x = {base}; uint[{bits}] y = {base};
                    uint[{bits}] n = {exponent}; x **= n; y = y ** n;''',
                              ('X1', 'Y1'), [expected, expected])
    check_program(bra, '''int x = 2; x **= x;
        uint[4] u = 3; int n = 4; u **= n;
        int y = -2; uint[4] e = 3; y **= e;
        array[int, 2] v = {2, 3}; int i = -2; v[i] **= v[i + 1];
        v[(i ** 0) - 1] **= v[1] ** 0;''',
                  ('X1', 'U1', 'Y1', 'V1:0', 'V1:1', 'I1'), ['4', '1', '-8', '8', '3', '-2'])
    check_program(bra, '''int n = 2; int x = 0; int y = 2;
        array[int, 4] v = {1, 2, 3, 4};
        for int i in [0:n ** 2] { x += i; }
        for int j in {n ** 0, n ** 1, n ** 2} { x += j; }
        while (y ** 2 < 300) { y **= 2; if (y > 10) { continue; } }
        int z = v[n ** 1]; if (x ** 0 == 1 && bool(y ** 0)) { z **= 2; }
        for int k in [0:2 ** 2] { x += 1; }
        for int k in [0:n ** 1:4] { x += k; }
        if (false) { z **= -1; } while (false) { z **= 2 ** -1; }''',
                  ('X1', 'Y1', 'Z1'), ['28', '256', '9'])
    check_program(bra, '''include "stdgates.inc"; const int N = 2 ** 1; qubit[N] q; int n = 2;
        rx(float(n ** 0) * pi) q[1]; gphase(float(n ** 2));
        bit b = measure q[1]; int x = 3;
        if (b && (n ** 2 == 4)) { x **= n; }''', ('B1', 'X1'), ['1', '9'])
    check_program(bra, '''const int N = 4; uint[N] a = 3; const uint[N] E = 2;
        uint[N] b = 0; for int N in [1:2] { a **= E; b = a ** E; }''',
                  ('A1', 'B1'), ['1', '1'])
    check_program(bra, f'''const int N = {converter.QCX_UINT_WIDTH}; int[N] a = 2;
        array[int[N], 1] v = {{2}}; for int N in [1:2] {{ a **= 2; v[0] **= 2; }}''',
                  ('A1', 'V1'), ['16', '16'])
    check_program(bra, '''array[int, 3] v = {1, 2, 3}; int x = 2;
        for int i in v { x **= i; }
        bit[2] flags = "11"; int y = 2;
        for bit b in flags { y **= int(b) + 1; }''', ('X1', 'Y1'), ['64', '16'])
    # Keep the documentation's complete exponentiation example executable.
    check_program(bra, '''const int SIZE = 2 ** 3;
        int a = -2; int n = 3; int signed_power = a ** n;
        uint[4] u = 3; uint[4] e = 4; uint[4] wrapped = u ** e;
        int promoted = u ** n; array[int, 2] values = {2, 3}; int i = 0;
        values[i] **= values[i + 1];
        if (signed_power < 0 && bool(u ** e)) { a **= 2; }
        int x = signed_power; int y = int(wrapped); int z = promoted;''',
                  ('X1', 'Y1', 'Z1', 'A1', 'VALUES63:0'), ['-8', '1', '27', '4', '8'])
    check_program(bra, '''uint[4] u = 15; int n = 1;
        int x = (u + n) ** uint[4](2); int y = (-(u + n)) ** uint[4](2);
        int z = 15; z **= (u + n) ** 0;''', ('X1', 'Y1', 'Z1'), ['256', '256', '15'])
    for source in ('int x = 2; x **= -1;', 'uint x = 2; int n = -1; x **= n;',
                   f'int x = {converter.QCX_INT_MAX}; x **= 2;'):
        check_error(bra, source, 'assertion failed in ASSERT')
    check_error(bra, f'int x = 2; uint n = {converter.QCX_UINT_WIDTH - 1}; x **= n;',
                'UINT to INT conversion out of range')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    bra = parser.parse_args().bra
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    converter = qasm2qcx.QASM2QCXConverter
    minimum, maximum, width = converter.QCX_INT_MIN, converter.QCX_INT_MAX, converter.QCX_UINT_WIDTH
    signed_cases = [(base, exponent) for base in range(-8, 9) for exponent in range(10)]
    signed_cases.extend(((minimum, 0), (minimum, 1), (minimum, 2), (maximum, 1),
                         (maximum, 2), (-2, width - 1), (2, width - 1),
                         (-2, width), (-1, maximum), (0, maximum), (1, maximum)))
    root = math.isqrt(maximum)
    signed_cases.extend((base, 2) for base in (root, root + 1, -root, -root - 1))
    for base, exponent in signed_cases:
        expected = pow(base, exponent)
        runtime = f'int a = {base}; int n = {exponent}; int x = a ** n;'
        if not minimum <= expected <= maximum:
            check_error(bra, runtime, 'assertion failed in ASSERT')
            try:
                qasm2qcx.convert(f'OPENQASM 3.0; const int C = ({base}) ** {exponent};')
            except qasm2qcx.InvalidPowerException:
                pass
            else:
                raise RuntimeError('An overflowing constant power was accepted')
            continue
        check_program(bra, runtime + f'const int C = ({base}) ** {exponent}; int y = C;',
                      ('X1', 'Y1'), [str(expected), str(expected)])
    for bits in (1, 4, 8, width):
        mask = (1 << bits) - 1
        for base in sorted({0, 1, 2 & mask, mask, 1 << (bits - 1)}):
            for exponent in sorted({0, 1, 2 & mask, 3 & mask, 4 & mask, 9 & mask, mask}):
                expected = pow(base, exponent, 1 << bits)
                check_program(bra, f'''uint[{bits}] a = {base}; uint[{bits}] n = {exponent};
                    uint[{bits}] x = a ** n;
                    const uint[{bits}] C = uint[{bits}]({base}) ** uint[{bits}]({exponent});
                    uint[{bits}] y = C;''', ('X1', 'Y1'), [str(expected), str(expected)])
    check_program(bra, '''int a = -2; uint[4] n = 3;
        int x = a ** n; uint[4] b = 3; int e = 4; int y = b ** e;
        uint u = 3; uint z = a ** u;''',
                  ('X1', 'Y1', 'Z1'), ['-8', '81', str((-8) & converter.QCX_UINT_MAX)])
    check_program(bra, f'''int n = -1; int a = 2; int z = 0;
        bool c = false && bool(a ** n);
        bool d = true || bool({maximum} ** 2);
        bool e = false && bool(a ** (1 / z));
        const bool OFF = false && bool(2 ** -1);
        bool f = OFF; if (false) {{ a = 2 ** -1; }}''',
                  ('A1', 'C1', 'D1', 'E1', 'F1'), ['2', '0', '1', '0', '0'])
    check_program(bra, '''int a = 2; int n = 3; int x = (a ** n) ** (n ** a);
        array[int, 2] values = {2, 3}; int i = 0; int y = values[i] ** values[i + 1];''',
                  ('X1', 'Y1'), [str(8 ** 9), '8'])
    for source in ('int n = -1; int x = 2 ** n;',
                   'int n = -1; uint x = uint(2) ** n;',
                   'int x = 2 ** -1;', 'uint x = uint(2) ** -1;',
                   f'int x = {maximum} ** 2;'):
        check_error(bra, source, 'assertion failed in ASSERT')
    check_error(bra, f'int a = {maximum}; int n = 2; int z = 0; int x = (a ** n) ** (1 / z);',
                'assertion failed in ASSERT')
    check_error(bra, 'int a = 0; int z = 0; int x = a ** (1 / z);',
                'integer division by zero in LET')
    check_assignments_and_control_flow(bra)
    print('Integer exponentiation numerical tests passed')


if __name__ == '__main__':
    main()
