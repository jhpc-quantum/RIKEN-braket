#!/usr/bin/env python3

"""Compare folded, runtime, and compound scalar integer shifts.

Example:
  python3 bra/test/qasm2qcx_shift_numerical.py --bra bra/bin/bra
"""

import argparse
import pathlib
import resource

from qasm2qcx_integer_array_numerical import check_error
from qasm2qcx_runtime_for_numerical import check_program, qasm2qcx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    bra = parser.parse_args().bra
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    native = qasm2qcx.QASM2QCXConverter.QCX_UINT_WIDTH
    minimum, maximum = qasm2qcx.QASM2QCXConverter.QCX_INT_MIN, qasm2qcx.QASM2QCXConverter.QCX_INT_MAX
    cases = [('int', native, (minimum, minimum + 1, -17, -3, -1, 0, 1, 3, maximum))]
    for width in (1, 4, 8, native):
        mask = (1 << width) - 1
        cases.append((f'uint[{width}]', width, sorted({0, 1, mask >> 1, mask - 1, mask})))
    for value_type, width, values in cases:
        for value in values:
            for count in range(width):
                for symbol in ('<<', '>>'):
                    expected = value << count if symbol == '<<' else value >> count
                    if value_type != 'int':
                        expected &= (1 << width) - 1
                    elif not minimum <= expected <= maximum:
                        continue
                    for count_type in ('int', 'uint'):
                        literal = f'({value})' if value_type == 'int' else f'{value_type}({value})'
                        check_program(bra, f'''const {value_type} F = {literal} {symbol} {count_type}({count});
                            {value_type} a = {value}; {count_type} b = {count};
                            {value_type} result = a {symbol} b; {value_type} folded = F;
                            {value_type} compound = a; compound {symbol}= b;''',
                                      ('RESULT63', 'FOLDED63', 'COMPOUND255', 'A1', 'B1'),
                                      [str(expected), str(expected), str(expected), str(value), str(count)])

    check_program(bra, '''uint[4] a = 15; int n = 1;
        uint[4] result = (a << n) >> n; uint[4] literal = (uint[4](15) << 1) >> 1;
        int b = -3; uint c = 1; int d = b >> c;
        array[int, 2] values = {4, 1}; int i = 0; int out = values[i] << values[i + 1];''',
                  ('RESULT63', 'LITERAL127', 'D1', 'OUT7', 'VALUES63:0', 'VALUES63:1'),
                  ['7', '7', '-2', '8', '4', '1'])
    check_program(bra, f'''int out = 7; uint[4] small = 3;
        if (false) {{ out = 1 << -1; out = {maximum} << 1; small = uint[4](1) >> 4; }}
        bool ready = false && bool(1 << -1);
        ready = true || bool({maximum} << 1);
        const bool OFF = false && bool(uint[4](1) << 4);
        bool folded = OFF;''', ('OUT7', 'SMALL31', 'READY31', 'FOLDED63'), ['7', '3', '1', '0'])
    check_program(bra, '''int n = 1; uint[4] u = 15; array[int, 3] a = {0, 1, 4}; int i = -3;
        n <<= n; u >>= u >> 3; a[2] >>= uint(n); a[a[i]] <<= a[i + 1];
        a[i + 1] <<= a[i + 2];
        if (false) { n <<= -1; u >>= 4; a[i + 999] <<= 1; }
        for int j in [1:0] { a[999] >>= -1; }''',
                  ('N1', 'U1', 'A1:0', 'A1:1', 'A1:2', 'I1'), ['2', '7', '0', '2', '1', '-3'])
    check_program(bra, '''int n = 2; int count = 1; int total = 0; array[int, 2] a = {1, 2};
        for int i in [0:n >> count] { a[i] <<= count; }
        for int j in {n >> count, n << count} { total += j; }
        while (count < 2) { total >>= count; count += 1; }
        total += a[n >> count];''', ('TOTAL31', 'A1:0', 'A1:1', 'COUNT31'), ['4', '2', '4', '2'])
    for source, diagnostic in (
            ('int a = 1; int n = -1; int out = a << n;', 'negative integer shift count'),
            (f'int a = 1; uint n = {native}; int out = a >> n;', 'integer shift count out of range'),
            (f'uint a = 1; uint n = {qasm2qcx.QASM2QCXConverter.QCX_UINT_MAX}; uint out = a << n;',
             'integer shift count out of range'),
            (f'int a = {maximum}; int n = 1; int out = a << n;', 'integer left shift overflow'),
            ('int out = 1 << -1;', 'negative integer shift count'),
            (f'int out = {maximum} << 1;', 'integer left shift overflow'),
            ('uint[4] a = 1; int n = 4; uint[4] out = a >> n;', 'assertion failed in ASSERT'),
            ('uint[4] out = uint[4](1) << 4;', 'assertion failed in ASSERT'),
            ('int a = 1; a <<= -1;', 'negative integer shift count'),
            (f'int a = {maximum}; a <<= 1;', 'integer left shift overflow'),
            (f'uint a = 1; uint n = {native}; a >>= n;', 'integer shift count out of range'),
            ('uint[4] a = 15; a <<= 4;', 'assertion failed in ASSERT'),
            ('array[int, 2] a = {1, 2}; int i = 0; a[i] <<= -1;', 'negative integer shift count'),
            (f'array[int, 2] a = {{1, {maximum}}}; int i = 1; a[i] <<= 1;', 'integer left shift overflow')):
        check_error(bra, source, diagnostic)
    print('qasm2qcx scalar shift numerical tests passed')


if __name__ == '__main__':
    main()
