#!/usr/bin/env python3

"""Check native QCX INT/UINT shifts and their runtime failures.

Example:
  python3 bra/test/integer_shift_numerical.py --bra bra/bin/bra
"""

import argparse
import ctypes
import pathlib
import resource

from integer_bitwise_numerical import check_program


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    bra = parser.parse_args().bra
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    width = ctypes.sizeof(ctypes.c_uint) * 8
    minimum, maximum, mask = -(1 << (width - 1)), (1 << (width - 1)) - 1, (1 << width) - 1
    for kind, values in (('INT', (minimum, minimum + 1, -17, -3, -1, 0, 1, 3, 17, maximum)),
                         ('UINT', (0, 1, 3, 17, maximum, maximum + 1, mask))):
        for symbol in ('<<=', '>>='):
            lines = ['QUBITS 0', f'VAR A {kind}', 'VAR S INT', 'VAR U UINT']
            expected = []
            for value in values:
                for count in range(width):
                    result = value << count if symbol == '<<=' else value >> count
                    if kind == 'UINT':
                        result &= mask
                    elif not minimum <= result <= maximum:
                        check_program(bra, f'QUBITS 0\nVAR A INT\nLET A := {value}\n'
                                      f'LET A {symbol} {count}\nPRINTLN A\n', None,
                                      'integer left shift overflow')
                        continue
                    for operand in (str(count), 'S', 'U', ':INT:S', ':UINT:U'):
                        lines.extend((f'LET A := {value}', f'LET S := {count}', f'LET U := {count}',
                                      f'LET A {symbol} {operand}', 'PRINTLN A S U'))
                        expected.append(f'{result} {count} {count}')
            check_program(bra, '\n'.join(lines) + '\n', expected)
            for operand, diagnostic in (('-1', 'negative integer shift count'),
                                        (str(width), 'integer shift count out of range'),
                                        (str(mask), 'integer shift count out of range'),
                                        ('S', 'negative integer shift count'),
                                        ('U', 'integer shift count out of range')):
                check_program(bra, f'''QUBITS 0
VAR A {kind}
VAR S INT
VAR U UINT
LET A := 7
LET S := -1
LET U := {mask}
LET A {symbol} {operand}
PRINTLN A
''', None, diagnostic)
            check_program(bra, f'''QUBITS 0
VAR A {kind} 2
VAR COUNTS UINT 2
VAR I INT
LET I := 1
LET A:0 := 3
LET A:1 := 8
LET COUNTS:1 := 2
let a:i {symbol} COUNTS:I
PRINTLN A:0 A:1 COUNTS:I I
JUMP SKIPPED
LET A:I {symbol} -1
LET MISSING {symbol} 1
@SKIPPED
PRINTLN A:1
''', ['3 32 2 1', '32'] if symbol == '<<=' else ['3 2 2 1', '2'])
            check_program(bra, f'QUBITS 0\nVAR A {kind}\nLET A := 2\n'
                          f'LET A {symbol} A\nPRINTLN A\n', ['8' if symbol == '<<=' else '0'])
            for operand in ('1.5', 'R', 'Z', 'P', ':REAL', ':PI', 'MISSING'):
                check_program(bra, f'''QUBITS 0
VAR A {kind}
VAR R REAL
VAR Z COMPLEX
VAR P PAULISS
LET A := 7
LET A {symbol} {operand}
PRINTLN A
''', None)
            for instruction in (f'LET A {symbol}', f'LET A {symbol} 1 EXTRA', f'LET 0 {symbol} 1'):
                check_program(bra, f'QUBITS 0\nVAR A {kind}\n{instruction}\nPRINTLN A\n', None)
    for symbol in ('<<=', '>>='):
        for kind in ('REAL', 'COMPLEX', 'PAULISS'):
            instruction = f'A {symbol} 1'
            check_program(bra, f'QUBITS 0\nVAR A {kind}\nLET {instruction}\nPRINTLN :INT:99\n',
                          None, f'"{instruction}" is a wrong argument')
    print('Native INT/UINT shift numerical tests passed')


if __name__ == '__main__':
    main()
