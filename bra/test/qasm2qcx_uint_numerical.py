#!/usr/bin/env python3

"""Compare unsigned constant folding and generated UINT arithmetic in bra.

Example:
  python3 bra/test/qasm2qcx_uint_numerical.py --bra bra/bin/bra
"""

import argparse
import operator
import pathlib
import resource

from qasm2qcx_integer_array_numerical import check_error
from qasm2qcx_runtime_for_numerical import check_program, qasm2qcx


def check_width(bra: pathlib.Path, width: int) -> None:
    mask = (1 << width) - 1
    values = sorted({0, 1, mask >> 1, mask - 1, mask})
    operations = (('+', operator.add), ('-', operator.sub), ('*', operator.mul),
                  ('/', operator.floordiv), ('%', operator.mod),
                  ('&', operator.and_), ('|', operator.or_), ('^', operator.xor))
    for op, function in operations:
        for lhs in values:
            for rhs in values:
                if op in ('/', '%') and rhs == 0:
                    continue
                expected = function(lhs, rhs) & mask
                check_program(bra, f'''
                    const uint[{width}] F = uint[{width}]({lhs}) {op} uint[{width}]({rhs});
                    uint[{width}] a = {lhs}; uint[{width}] b = {rhs};
                    uint[{width}] result = a {op} b; uint[{width}] folded = F;
                ''', ('RESULT63', 'FOLDED63', 'A1', 'B1'),
                              [str(expected), str(expected), str(lhs), str(rhs)])
                check_program(bra, f'uint[{width}] a = {lhs}; uint[{width}] b = {rhs}; a {op}= b;',
                              ('A1', 'B1'), [str(expected), str(rhs)])
    for value in values:
        check_program(bra, f'''uint[{width}] a = {value};
            uint[{width}] b = ~a; uint[{width}] c = -a;
            uint[{width}] folded = ~uint[{width}]({value});''',
                      ('A1', 'B1', 'C1', 'FOLDED63'),
                      [str(value), str(value ^ mask), str((-value) & mask), str(value ^ mask)])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    bra = parser.parse_args().bra
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    native = qasm2qcx.QASM2QCXConverter.QCX_UINT_WIDTH
    maximum = (1 << native) - 1
    for width in (1, 4, 8, native):
        check_width(bra, width)

    check_program(bra, '''uint[8] a = 255; uint[8] one = 1; uint[8] two = 2;
        uint[8] b = (a + one) / two; uint[8] c = (a + 1) / 2;
        uint[8] d = (a * a) / two; uint[4] small = 15;
        uint[8] e = small + one; a += one;''',
                  ('A1', 'B1', 'C1', 'D1', 'E1'), ['0', '0', '128', '0', '16'])
    check_program(bra, f'''uint a = {maximum}; uint b = a + uint(1);
        uint[8] c = uint[8](a); int n = -1; uint d = n + uint(1);
        uint e = a % uint(7); bool equal = a == -1;
        const bool SAME = uint(-1) == -1; bool folded = SAME;
        uint[8] small = 255; bool greater = small > -1;''',
                  ('A1', 'B1', 'C1', 'D1', 'E1', 'EQUAL31', 'FOLDED63', 'GREATER127'),
                  [str(maximum), '0', '255', '0', str(maximum % 7), '1', '1', '1'])

    # Small UINT promotes to native INT when mixed with INT. Compound
    # assignment computes that result first, then converts/masks the target.
    check_program(bra, '''uint[8] a = 5; int divisor = -1;
        a /= divisor; uint[8] b = 5; b /= -1;
        uint[8] c = 5; c %= divisor; int n = -3; int r = n + uint[8](7);''',
                  ('A1', 'B1', 'C1', 'R1'), ['251', '251', '0', '4'])
    for op, lhs, rhs, expected in (('+=', -1, 1, 0), ('-=', -1, maximum, 0),
                                   ('*=', -1, 0, 0), ('/=', 1, maximum, 0),
                                   ('%=', -1, maximum, 0), ('&=', -1, 0, 0)):
        check_program(bra, f'int a = {lhs}; uint b = {rhs}; a {op} b;',
                      ('A1', 'B1'), [str(expected), str(rhs)])

    check_program(bra, '''float f = 257.9; complex z = 3.9 + 2im;
        uint[8] a = uint[8](f); uint[8] b = uint[8](257.9);
        uint c = uint(z); uint d = uint(true); uint e = uint(-0.9);
        float wide = a; int n = int(a); bool ready = bool(c);''',
                  ('A1', 'B1', 'C1', 'D1', 'E1', 'N1', 'READY31'),
                  ['1', '1', '3', '1', '0', '1', '1'])
    check_program(bra, '''uint a = 7; int b = 3;
        if (false) { a = uint(-1.0); b = int(uint(-1)); a /= 0; }
        bool ready = false && bool(uint(-1.0));
        ready = true || bool(int(uint(-1)));''',
                  ('A1', 'B1', 'READY31'), ['7', '3', '1'])
    for source, diagnostic in (
            ('uint a = uint(-1.0);', 'REAL to UINT conversion out of range'),
            ('float f = -1.0; uint a = uint(f);', 'REAL to UINT conversion out of range'),
            (f'float f = {maximum + 1}.0; uint a = uint(f);', 'REAL to UINT conversion out of range'),
            ('float f = 0.0; f /= 0; uint a = uint(f);', 'REAL to UINT conversion out of range'),
            ('uint a = uint(-1); int b = int(a);', 'UINT to INT conversion out of range'),
            ('int b = int(uint(-1));', 'UINT to INT conversion out of range'),
            ('uint a = 7; a /= uint(0);', 'integer division by zero in LET'),
            ('uint[8] a = 7; a %= uint[8](0);', 'integer division by zero in LET')):
        check_error(bra, source, diagnostic)
    print('qasm2qcx native and narrow UINT numerical tests passed')


if __name__ == '__main__':
    main()
