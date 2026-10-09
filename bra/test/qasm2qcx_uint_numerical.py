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


def check_integration(bra: pathlib.Path) -> None:
    for start in range(4):
        for stop in range(4):
            for step in (1, 2, 3):
                values = list(range(start, stop + 1, step))
                check_program(bra, f'''uint[8] first = {start}; uint last = {stop};
                    uint[4] step = {step}; int total = 0; int visits = 0;
                    for int i in [first:step:last] {{ total += i; visits += 1; }}''',
                              ('TOTAL31', 'VISITS63'), [str(sum(values)), str(len(values))])
    check_program(bra, '''uint first = 1; uint[8] last = 5; uint step = 2; int total = 0;
        for int i in [first:step:last] {
            first = 9; last = 0; step = 1;
            if (i == 3) { continue; } total += i;
        }
        for int i in {first, last, first} { first = 0; total += i; }''',
                  ('TOTAL31',), ['24'])
    check_program(bra, '''const uint[8] FIRST = 1; const uint[8] LAST = 5;
        uint[8] n = 0; int total = 0;
        for int i in [FIRST:uint[8](2):LAST] { total += i; }
        for int i in {FIRST, LAST, FIRST} { total += i; }
        while (n < uint[8](4)) {
            n += uint[8](1); if (n == uint[8](2)) { continue; }
            if (n == uint[8](4)) { break; } total += int(n);
        }''', ('TOTAL31', 'N1'), ['20', '4'])
    check_program(bra, '''uint[8] i = 1; const uint[8] I = 1;
        array[int, 3] a = {uint[8](4), i, uint(7)}; bit[3] flags = "101";
        int out = a[i]; a[i] += uint(3); flags[i] = !flags[I];
        flags[i] ^= flags[uint[8](2)];
        i = uint[8](0); a[i] = a[uint[8](2)];
        complex z = 2.5 + 9im; out += a[uint(z)];''',
                  ('A1:0', 'A1:1', 'A1:2', 'FLAGS31:1', 'OUT7'), ['7', '4', '7', '0', '8'])
    check_program(bra, '''uint[8] n = 1; int total = 0; array[int, 2] a = {3, 4};
        bit[2] flags = "01"; uint bad = uint(-1); uint zero = 0;
        if (false) {
            total = a[bad]; flags[bad] = true;
            for int i in [bad:bad] { total += i; }
            for int i in {bad, n} { total += i; }
        }
        bool ready = true || bool(a[bad]);
        ready = false && bool(flags[bad]);
        while (false && bool(n / zero)) { n += uint[8](1); }
        for int i in [n:n] { if (n == uint[8](1)) { total += i; } }''',
                  ('TOTAL31', 'N1', 'READY31'), ['1', '1', '0'])
    check_program(bra, '''uint i = 0; uint all = uint(-1); uint[8] mask = 7;
        array[int, 1] a = {-1}; a[i] %= all;
        a[i] = -1; a[i] ^= all; int out = a[i];
        a[i] = -1; a[i] &= mask; out += a[i];''',
                  ('A1:0', 'OUT7'), ['7', '7'])
    check_program(bra, '''include "stdgates.inc"; qubit q; uint theta = 0;
        uint i = 1; bit[2] flags = "00"; rx(theta) q; gphase(theta);
        x q; flags[i] = measure q;
        bit[1] single = "0"; i = 0; single[i] = flags[uint(1)];''',
                  ('FLAGS31:0', 'FLAGS31:1', 'SINGLE63'), ['0', '1', '1'])
    maximum = qasm2qcx.QASM2QCXConverter.QCX_INT_MAX
    check_program(bra, f'''uint start = {maximum}; uint stop = start; uint step = 1;
        int out = 0; for int i in [start:step:stop] {{ out = i; }}''',
                  ('OUT7',), [str(maximum)])
    for source, diagnostic in (
            ('uint bad = uint(-1); for int i in [bad:bad] {}', 'UINT to INT conversion out of range'),
            ('uint bad = uint(-1); for int i in {0, bad} {}', 'UINT to INT conversion out of range'),
            ('uint step = uint(-1); for int i in [0:step:2] {}', 'UINT to INT conversion out of range'),
            ('uint step = 0; for int i in [0:step:2] {}', 'assertion failed in ASSERT'),
            ('uint i = uint(-1); array[int, 2] a; int out = a[i];', 'UINT to INT conversion out of range'),
            ('uint i = uint(-1); bit[2] flags; bit out = flags[i];', 'UINT to INT conversion out of range'),
            ('uint i = 2; bit[2] flags; bit out = flags[i];', 'assertion failed in ASSERT'),
            ('uint bad = uint(-1); array[int, 1] a = {bad};', 'UINT to INT conversion out of range')):
        check_error(bra, source, diagnostic)


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
    check_integration(bra)
    print('qasm2qcx native and narrow UINT numerical tests passed')


if __name__ == '__main__':
    main()
