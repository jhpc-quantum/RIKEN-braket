#!/usr/bin/env python3

"""Check native QCX INT/UINT remainder assignment and reached/skipped errors.

  python3 bra/test/integer_remainder_numerical.py --bra /path/to/bra

Expected remainders use arbitrary-precision integer division with explicit
truncation toward zero, not Python's signed % convention.
"""

import argparse
import ctypes
import pathlib
import resource
import subprocess


def remainder(lhs, rhs):
    quotient = abs(lhs) // abs(rhs)
    if (lhs < 0) != (rhs < 0):
        quotient = -quotient
    return lhs - quotient * rhs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    bra = parser.parse_args().bra.resolve()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    width = ctypes.sizeof(ctypes.c_int) * 8
    minimum, maximum = -(1 << (width - 1)), (1 << (width - 1)) - 1
    unsigned_maximum = (1 << (ctypes.sizeof(ctypes.c_uint) * 8)) - 1
    programs = 0

    def check(source, expected, diagnostic=''):
        nonlocal programs
        result = subprocess.run([str(bra)], input=source, text=True, capture_output=True, timeout=30)
        if expected is None:
            valid = result.returncode != 0 and not result.stdout.strip() and diagnostic in result.stderr
        else:
            valid = result.returncode == 0 and result.stdout.splitlines() == expected
        if not valid:
            raise RuntimeError(f'Remainder test failed:\n{source}\n{result}')
        programs += 1

    for kind, values in (('INT', (minimum, minimum + 1, -17, -7, -3, -1, 0, 1, 3, 7, 17, maximum)),
                         ('UINT', (0, 1, 3, 7, 17, maximum, maximum + 1, unsigned_maximum))):
        lines = ['QUBITS 0', f'VAR A {kind}', f'VAR B {kind}']
        expected = []
        for lhs in values:
            for rhs in values:
                if rhs == 0:
                    continue
                for operand in (str(rhs), 'B'):
                    lines += [f'LET A := {lhs}', f'LET B := {rhs}', f'LET A %= {operand}', 'PRINTLN A B']
                    expected.append(f'{remainder(lhs, rhs)} {rhs}')
        check('\n'.join(lines) + '\n', expected)
        for lhs in values:
            check(f'QUBITS 0\nVAR A {kind}\nLET A := {lhs}\nLET A %= A\nPRINTLN A\n',
                  None if lhs == 0 else ['0'], 'integer remainder by zero in LET A %= A' if lhs == 0 else '')
        cast = ':INT:' if kind == 'INT' else ':UINT:'
        for divisor in ('0', 'B', cast + '0.5', cast + '-0.5', cast + '-0.0', cast + ':COMPLEX:0.5'):
            prefix = f'QUBITS 0\nVAR A {kind} 2\nVAR B {kind}\nVAR I INT\nLET I := 1\nLET A:1 := 7\n'
            instruction = f'LET A:I %= {divisor}'
            check(prefix + instruction + '\nPRINTLN A:1\n', None, 'integer remainder by zero in ' + instruction)
            for jump in ('JUMP DONE', 'JUMPIF DONE I == 1'):
                check(prefix + jump + '\n' + instruction + '\n@DONE\nPRINTLN A:1\n', ['7'])
        for divisor in ('R', 'Z', 'P', ':REAL:3', ':COMPLEX:3', '3.5', 'MISSING', 'B:99', 'B:-1',
                        'B:POINTERS:99', ':INT:R:99' if kind == 'INT' else ':UINT:R:99',
                        ':INT:1e309' if kind == 'INT' else ':UINT:1e309',
                        ':UINT:3' if kind == 'INT' else ':INT:3', 'U' if kind == 'INT' else 'S'):
            prefix = (f'QUBITS 0\nVAR A {kind}\nVAR B {kind}\nVAR R REAL\nVAR Z COMPLEX\n'
                      'VAR P PAULISS\nVAR U UINT\nVAR S INT\nVAR POINTERS INT\nLET A := 7\n')
            instruction = f'LET A %= {divisor}\n'
            check(prefix + instruction + 'PRINTLN A\n', None)
            check(prefix + 'JUMP DONE\n' + instruction + '@DONE\nPRINTLN A\n', ['7'])
        for source in (cast + 'R', cast + 'Z', cast + '3.9', cast + (':UINT:3' if kind == 'INT' else ':INT:3')):
            check(f'QUBITS 0\nVAR A {kind}\nVAR R REAL\nVAR Z COMPLEX\nLET A := 7\n'
                  f'LET R := 3.9\nLET Z := :COMPLEX:3.9\nLET A %= {source}\nPRINTLN A R :REAL:Z\n',
                  ['1 3.9 3.9'])
        for numerator in ('0.0', '1.0', '-1.0'):
            check(f'QUBITS 0\nVAR A {kind}\nVAR R REAL\nLET A := 7\nLET R := {numerator}\n'
                  f'LET R /= 0.0\nLET A %= {cast}R\nPRINTLN A\n', None, 'conversion out of range')
        check(f'''QUBITS 0
VAR A {kind} 2
VAR B {kind} 2
VAR POINTERS INT 2
VAR I INT
LET I := 1
LET POINTERS:1 := 0
LET A:0 := 3
LET A:1 := 7
LET B:0 := 3
let a:i %= b:pointers:i
PRINTLN A:0 A:1 B:0 I
''', ['3 1 3 1'])
        for target in ('A:99', 'A:-1', 'A:POINTERS:99'):
            prefix = f'QUBITS 0\nVAR A {kind}\nVAR POINTERS INT\nLET A := 7\n'
            check(prefix + f'LET {target} %= 3\nPRINTLN A\n', None, 'index')
            check(prefix + f'JUMP DONE\nLET {target} %= 3\n@DONE\nPRINTLN A\n', ['7'])

    check(f'QUBITS 0\nVAR A INT\nVAR U UINT\nLET A := 7\nLET U := {unsigned_maximum}\n'
          'LET A %= :INT:U\nPRINTLN A\n', None, 'UINT to INT conversion out of range')
    check('QUBITS 0\nVAR U UINT\nLET U := 7\nLET U %= :UINT:-1\nPRINTLN U\n', ['7'])
    for divisor in ('-1', '-0', str(unsigned_maximum) + '0'):
        check('QUBITS 0\nVAR U UINT\nLET U := 7\n' + f'LET U %= {divisor}\nPRINTLN U\n', None)
    for kind in ('REAL', 'COMPLEX', 'PAULISS'):
        check(f'QUBITS 0\nVAR A {kind}\nLET A %= 3\nPRINTLN A\n', None, 'A %= 3')
    for malformed in ('LET A %=', 'LET %= 3', 'LET A %= 3 EXTRA', 'LET A %=3'):
        check('QUBITS 0\nVAR A INT\nJUMP DONE\n' + malformed + '\n@DONE\nPRINTLN A\n', None)
    check('''QUBITS 0
VAR A INT 2
VAR I INT
LET I := 1
LET A:0 := 3
LET A:1 := 1
LET A:I %= A:A:I
PRINTLN A:0 A:1 I
LET A:1 := 1
LET I %= A:I
PRINTLN A:0 A:1 I
''', ['3 0 1', '3 1 0'])
    for fused in (False, True):
        begin, end = ('BEGIN FUSION\n', 'END FUSION\n') if fused else ('', '')
        check('QUBITS 1\nVAR A INT\nLET A := 7\n' + begin
              + 'LET A %= 3\nH 0\n' + end + 'PRINTLN A\n', ['1'])
    print(f'Native integer remainder programs checked: {programs}')


if __name__ == '__main__':
    main()
