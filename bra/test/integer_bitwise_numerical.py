#!/usr/bin/env python3

"""Check integer-only QCX LET bitwise operators against Python results.

Example:
  python3 bra/test/integer_bitwise_numerical.py --bra bra/bin/bra
"""

import argparse
import ctypes
import operator
import pathlib
import resource
import subprocess


def check_program(bra: pathlib.Path, source: str, expected: list[str] | None,
                  diagnostic: str = '') -> None:
    result = subprocess.run([str(bra)], input=source, text=True,
                            capture_output=True, timeout=30)
    if expected is None:
        valid = result.returncode != 0 and not result.stdout.strip() and diagnostic in result.stderr
    else:
        valid = result.returncode == 0 and result.stdout.splitlines() == expected
    if not valid:
        raise RuntimeError(f'Integer bitwise test failed\nsource:\n{source}\n{result}')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    bra = parser.parse_args().bra
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    width = ctypes.sizeof(ctypes.c_int) * 8
    minimum, maximum = -(1 << (width - 1)), (1 << (width - 1)) - 1
    operands = (minimum, minimum + 1, -17, -8, -3, -1, 0, 1, 3, 8, 17, maximum)
    for op, function in (('&=', operator.and_), ('|=', operator.or_), ('^=', operator.xor)):
        lines = ['QUBITS 0', 'VAR A INT', 'VAR B INT']
        expected = []
        for lhs in operands:
            for rhs in operands:
                for operand in (str(rhs), 'B'):
                    lines.extend([f'LET A := {lhs}', f'LET B := {rhs}',
                                  f'LET A {op} {operand}', 'PRINTLN A B'])
                    expected.append(f'{function(lhs, rhs)} {rhs}')
        check_program(bra, '\n'.join(lines) + '\n', expected)
        for lhs in operands:
            check_program(bra, f'QUBITS 0\nVAR A INT\nLET A := {lhs}\n'
                          f'LET A {op} A\nPRINTLN A\n', [str(function(lhs, lhs))])

        # Both indices are evaluated through existing recursive integer operands.
        check_program(bra, f'''QUBITS 0
VAR VALUES INT 2
VAR POINTERS INT 2
VAR INDEX INT
LET INDEX := 1
LET POINTERS:INDEX := 0
LET VALUES:0 := -3
LET VALUES:1 := 7
LET VALUES:INDEX {op} VALUES:POINTERS:INDEX
PRINTLN VALUES:0 VALUES:1 INDEX
''', [f'-3 {function(7, -3)} 1'])
        check_program(bra, f'''QUBITS 0
VAR A INT
LET A := 7
let a {op} :INT:3.5
PRINTLN A
JUMP SKIPPED
LET MISSING {op} 1
@SKIPPED
JUMPIF DONE A == {function(7, 3)}
LET A {op} :REAL
@DONE
PRINTLN A
''', [str(function(7, 3))] * 2)

        for kind in ('REAL', 'COMPLEX', 'PAULISS'):
            instruction = f'R:1 {op} 1'
            check_program(bra, f'QUBITS 0\nVAR R {kind} 2\nLET {instruction}\nPRINTLN :INT:99\n',
                          None, f'"{instruction}" is a wrong argument')
        for operand in ('1.5', 'R', 'R:0', 'Z', 'P', ':REAL', ':PI', ':COMPLEX', 'MISSING'):
            check_program(bra, f'''QUBITS 0
VAR A INT
VAR R REAL
VAR Z COMPLEX
VAR P PAULISS
LET A := 7
LET A {op} {operand}
PRINTLN A
''', None)
        for instruction in (f'LET A {op}', f'LET A {op} 1 EXTRA', f'LET 0 {op} 1'):
            check_program(bra, 'QUBITS 0\nVAR A INT\n' + instruction + '\nPRINTLN A\n', None)

    # Integer complement and one-bit complement are distinct masks.
    for value in operands:
        check_program(bra, f'QUBITS 0\nVAR A INT\nLET A := {value}\nLET A ^= -1\nPRINTLN A\n',
                      [str(~value)])
    for value in (0, 1):
        check_program(bra, f'QUBITS 0\nVAR A INT\nLET A := {value}\nLET A ^= 1\nPRINTLN A\n',
                      [str(1 - value)])
    for op in ('&', '|', '^', '~='):
        check_program(bra, f'QUBITS 0\nVAR A INT\nLET A {op} 1\nPRINTLN A\n', None)
    print('Integer bitwise numerical tests passed')


if __name__ == '__main__':
    main()
