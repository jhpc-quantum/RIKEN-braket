#!/usr/bin/env python3

"""Execute converted scalar bitwise expressions with bra.

Example:
  python3 bra/test/qasm2qcx_bitwise_numerical.py --bra bra/bin/bra
"""

import argparse
import operator
import pathlib
import resource

from qasm2qcx_integer_array_numerical import check_error
from qasm2qcx_runtime_for_numerical import check_program, qasm2qcx


def check_assignments(bra: pathlib.Path, values: tuple[int, ...]) -> None:
    for op, function in (('&=', operator.and_), ('|=', operator.or_), ('^=', operator.xor)):
        for lhs in values:
            for rhs in values:
                check_program(bra, f'int a = {lhs}; int b = {rhs}; a {op} b;',
                              ('A1', 'B1'), [str(function(lhs, rhs)), str(rhs)])
            check_program(bra, f'int a = {lhs}; a {op} a;', ('A1',), [str(function(lhs, lhs))])
        for lhs in (0, 1):
            for rhs in (0, 1):
                check_program(bra, f'bit a = {lhs}; bit b = {rhs}; a {op} b;',
                              ('A1', 'B1'), [str(function(lhs, rhs)), str(rhs)])

        # Overlapping runtime element targets and RHS reads retain old values.
        for pattern in range(4):
            bits = [(pattern >> index) & 1 for index in range(2)]
            for target in (-2, -1, 0, 1):
                for source in (-2, -1, 0, 1):
                    expected = bits.copy()
                    expected[target] = function(bits[target], bits[source] ^ bits[target])
                    check_program(bra, f'''bit[2] flags = "{pattern:02b}";
                        int i = {target}; int j = {source};
                        flags[i] {op} flags[j] ^ flags[i];''',
                                  ('FLAGS31:0', 'FLAGS31:1', 'I1', 'J1'),
                                  [*map(str, expected), str(target), str(source)])
        for target in (-2, -1, 0, 1):
            expected = [7, -3]
            expected[target] = function(expected[target], 2)
            check_program(bra, f'''array[int, 2] a = {{7, -3}}; int i = {target};
                a[i] {op} a[0] & ~a[1];''', ('A1:0', 'A1:1', 'I1'),
                          [*map(str, expected), str(target)])
        check_program(bra, f'''bit[1] flags = "1"; int i = -1; flags[i] {op} ~flags[i];
            array[int, 1] a = {{7}}; a[i] {op} a[i];''', ('FLAGS31', 'A1:0', 'I1'),
                      [str(function(1, 0)), str(function(7, 7)), '-1'])
        for index in (-3, 2, qasm2qcx.QASM2QCXConverter.QCX_INT_MIN,
                      qasm2qcx.QASM2QCXConverter.QCX_INT_MAX):
            check_error(bra, f'bit[2] flags = "01"; int i = {index}; flags[i] {op} bit(true);',
                        'assertion failed in ASSERT')
            check_error(bra, f'array[int, 2] a = {{1, 3}}; int i = {index}; a[i] {op} 1;',
                        'assertion failed in ASSERT')
        # Destination failure must take precedence over an invalid RHS access.
        check_error(bra, f'''array[int, 2] a = {{1, 3}}; int bad = 2; int zero = 0;
            a[bad] {op} 1 / zero;''', 'assertion failed in ASSERT')
        check_error(bra, f'''bit[2] flags = "01"; int bad = 2; int zero = 0;
            flags[bad] {op} bit(bool(1 / zero));''', 'assertion failed in ASSERT')
        check_error(bra, f'bit[2] flags = "01"; int bad = 2; flags[0] {op} flags[bad];',
                    'assertion failed in ASSERT')

    check_program(bra, '''array[int, 2] a = {1, 0}; int i = 0;
        a[a[i]] ^= a[i];''', ('A1:0', 'A1:1'), ['1', '1'])
    check_program(bra, '''bit[2] flags = "01"; int i = 0;
        flags[int(flags[i])] ^= flags[i];''', ('FLAGS31:0', 'FLAGS31:1'), ['1', '1'])
    check_program(bra, '''bit[2] flags = "01"; int zero = 0; int bad = 2;
        flags[0] &= bit(false && bool(1 / zero));
        flags[1] |= bit(true || flags[bad]);
        if (false) { flags[bad] ^= flags[0]; }
        while (false) { bad ^= 1 / zero; }''', ('FLAGS31:0', 'FLAGS31:1', 'BAD7'), ['0', '1', '2'])
    check_program(bra, '''array[int, 2] a = {0, 1}; bit[2] flags = "01";
        int n = 7; int first = 0; int last = 1;
        for int i in [first:last] { n ^= i; flags[i] ^= flags[1 - i]; }
        for int i in {first, last} { a[i] |= n & 1; }
        for int i in a { n &= ~i; }
        for bit value in flags { flags[int(value)] &= value; }''',
                  ('N1', 'FLAGS31:0', 'FLAGS31:1', 'A1:0', 'A1:1'), ['6', '1', '1', '0', '1'])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    bra = parser.parse_args().bra
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    values = (qasm2qcx.QASM2QCXConverter.QCX_INT_MIN, -8, -3, -1, 0, 1, 7,
              qasm2qcx.QASM2QCXConverter.QCX_INT_MAX)
    for op, function in (('&', operator.and_), ('|', operator.or_), ('^', operator.xor)):
        for lhs in values:
            for rhs in values:
                result = function(lhs, rhs)
                check_program(bra, f'''const int FOLDED = ({lhs}) {op} ({rhs});
                    int a = {lhs}; int b = {rhs}; int c = a {op} b;
                    int d = FOLDED;''', ('C1', 'D1', 'A1', 'B1'),
                              [str(result), str(result), str(lhs), str(rhs)])
        for lhs in (0, 1):
            for rhs in (0, 1):
                result = function(lhs, rhs)
                check_program(bra, f'''bit a = {lhs}; bit b = {rhs};
                    bit out = a {op} b; bool ready = a {op} b; int n = 0;
                    if (a {op} b) {{ n = 1; }}
                    int c = int(bit(bool({lhs})) {op} bit(bool({rhs})));''',
                              ('OUT7', 'READY31', 'N1', 'C1', 'A1', 'B1'),
                              [str(result)] * 4 + [str(lhs), str(rhs)])

    for value in values:
        check_program(bra, f'int a = {value}; int c = ~a; int d = ~({value});',
                      ('C1', 'D1', 'A1'), [str(~value), str(~value), str(value)])
    for value in (0, 1):
        check_program(bra, f'''bit a = {value}; bit out = ~a; int n = 0;
            if (~a) {{ n = 1; }} bool ready = ~a && true;''',
                      ('OUT7', 'N1', 'READY31', 'A1'), [str(1 - value)] * 3 + [str(value)])

    check_program(bra, '''int a = -3; int b = 7;
        int c = ~(a & b) | (a ^ b); int d = a | b ^ 1 & 3;''',
                  ('C1', 'D1', 'A1', 'B1'), ['-6', '-1', '-3', '7'])
    check_program(bra, '''array[int, 2] a = {0, 1}; bit[2] flags = "01"; int i = 0;
        int c = a[i] ^ a[i + 1]; bit out = ~flags[i] | flags[i + 1];
        flags[i & 1] = flags[i] ^ flags[i + 1];
        a[i ^ 1] = (a[i] | 4) & ~2;''', ('C1', 'OUT7', 'FLAGS31:0', 'A1:1'), ['1', '0', '1', '4'])
    check_program(bra, '''bit[1] flags = "1"; int i = -1; bit out = ~flags[i];
        flags[i] = flags[i] ^ bit(true); bool ready = flags[i] | out;''',
                  ('FLAGS31', 'OUT7', 'READY31'), ['0', '0', '0'])
    check_program(bra, '''bit[2] flags = "01"; bit a = 1; bit b = 0;
        int n = 1; int total = 0;
        if (~a | b) { total = 100; }
        for int i in [0:n & 1] { total += i ^ n; }
        for int i in {~n, n | 2} { total += i; }
        for bit value in flags { a = value ^ b; if (~value) { b = a; } }
        while (a & b) { break; }''', ('TOTAL31', 'A1', 'B1'), ['2', '0', '0'])
    check_program(bra, '''bit[2] flags = "01"; int zero = 0; int bad = 2;
        bool ready = false && bool(~(1 / zero));
        ready = true || (flags[bad] & flags[bad]);
        if (false) { flags[0] = flags[bad] ^ flags[0]; }
        while (false) { bad = ~bad & (1 / zero); }''', ('READY31', 'BAD7'), ['1', '2'])
    check_program(bra, '''include "stdgates.inc"; qubit q;
        int a = 1; bit out = 0; rx(float(a & 1) * pi) q;
        out = measure q;''', ('OUT7',), ['1'])
    check_program(bra, '''const bool SKIP = false && bool(~(1 / 0));
        const bool TAKE = true || bool((1 / 0) ^ 3); bool ready = SKIP || TAKE;''',
                  ('READY31',), ['1'])

    # Bitwise binary operators are eager, but surrounding logical operators
    # short-circuit them. The new binary lowering evaluates its LHS first.
    check_error(bra, '''array[int, 2] a = {0, 1}; int bad = 2; int zero = 0;
        int c = a[bad] & (1 / zero);''', 'assertion failed in ASSERT')
    check_error(bra, '''array[int, 2] a = {0, 1}; int bad = 2; int zero = 0;
        int c = (1 / zero) | a[bad];''', 'integer division by zero in LET')
    check_error(bra, 'bit a = 0; bit[2] flags = "01"; int bad = 2; bit out = a & flags[bad];',
                'assertion failed in ASSERT')
    check_assignments(bra, values)
    print('qasm2qcx scalar bitwise numerical tests passed')


if __name__ == '__main__':
    main()
