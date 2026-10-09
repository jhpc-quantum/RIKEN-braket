#!/usr/bin/env python3

"""Check native UINT arithmetic, comparisons, and explicit conversions.

Example:
  python3 bra/test/uint_operations_numerical.py --bra bra/bin/bra
"""

import argparse
import ctypes
import operator
import pathlib
import resource

from uint_storage_numerical import check_program


def check_operations(bra: pathlib.Path, values: tuple[int, ...], maximum: int) -> None:
    operations = (('+=', operator.add), ('-=', operator.sub), ('*=', operator.mul),
                  ('/=', operator.floordiv), ('&=', operator.and_),
                  ('|=', operator.or_), ('^=', operator.xor))
    for op, function in operations:
        lines = ['QUBITS 0', 'VAR A UINT', 'VAR B UINT']
        expected = []
        for lhs in values:
            for rhs in values:
                if op == '/=' and rhs == 0:
                    continue
                for operand in (str(rhs), 'B'):
                    lines.extend([f'LET A := {lhs}', f'LET B := {rhs}',
                                  f'LET A {op} {operand}', 'PRINTLN A B'])
                    expected.append(f'{function(lhs, rhs) & maximum} {rhs}')
            if op != '/=' or lhs != 0:
                lines.extend([f'LET A := {lhs}', f'LET A {op} A', 'PRINTLN A'])
                expected.append(str(function(lhs, lhs) & maximum))
        check_program(bra, '\n'.join(lines) + '\n', expected)
        check_program(bra, f'''QUBITS 0
VAR VALUES UINT 2
VAR POINTERS INT 2
VAR INDEX INT
LET INDEX := 1
LET POINTERS:INDEX := 0
LET VALUES:0 := {maximum}
LET VALUES:1 := 7
LET VALUES:INDEX {op} VALUES:POINTERS:INDEX
PRINTLN VALUES:0 VALUES:1 INDEX
''', [f'{maximum} {function(7, maximum) & maximum} 1'])

        for operand in ('S', 'R', 'Z', 'P', '-1', '1.5', ':INT:1', ':REAL:1'):
            check_program(bra, f'''QUBITS 0
VAR A UINT
VAR S INT
VAR R REAL
VAR Z COMPLEX
VAR P PAULISS
LET A := 7
LET A {op} {operand}
''', None)
        check_program(bra, f'''QUBITS 0
VAR A UINT
LET A := 7
LET A {op} :UINT:-1
PRINTLN A
JUMP DONE
LET A {op} -1
@DONE
PRINTLN A
''', [str(function(7, maximum) & maximum)] * 2)
    for operand in ('0', 'A', ':UINT:0.5'):
        check_program(bra, f'QUBITS 0\nVAR A UINT\nLET A /= {operand}\n',
                      None, f'integer division by zero in LET A /= {operand}')
    check_program(bra, '''QUBITS 0
VAR A UINT
LET A := 7
JUMPIF DONE A == 7
LET A /= 0
LET A := :UINT::REAL:-1.0
ASSERT A == -1
@DONE
PRINTLN A
''', ['7'])


def check_comparisons(bra: pathlib.Path, values: tuple[int, ...]) -> None:
    for op, function in (('==', operator.eq), ('\\=', operator.ne), ('>', operator.gt),
                         ('<', operator.lt), ('>=', operator.ge), ('<=', operator.le)):
        lines = ['QUBITS 0', 'VAR A UINT 2', 'VAR B UINT', 'VAR INDEX INT', 'LET INDEX := 1']
        expected = []
        counter = 0
        for lhs in values:
            for rhs in values:
                for operand in (str(rhs), 'B'):
                    counter += 1
                    lines.extend([f'LET A:INDEX := {lhs}', f'LET B := {rhs}',
                                  f'JUMPIF TRUE{counter} A:INDEX {op} {operand}',
                                  'PRINTLN 0', f'JUMP END{counter}', f'@TRUE{counter}',
                                  f'ASSERT A:INDEX {op} {operand}', 'PRINTLN 1', f'@END{counter}'])
                    expected.append(str(int(function(lhs, rhs))))
        check_program(bra, '\n'.join(lines) + '\n', expected)
    check_program(bra, '''QUBITS 0
VAR A UINT
VAR B INT
JUMPIF DONE A == B
@DONE
''', None)
    check_program(bra, '''QUBITS 0
VAR A UINT
VAR B INT
ASSERT B == A
''', None)


def check_casts(bra: pathlib.Path, maximum: int, width: int) -> None:
    int_maximum = maximum >> 1
    int_minimum = -(1 << (width - 1))
    check_program(bra, f'''QUBITS 0
VAR U UINT 2
VAR S INT
LET S := -1
LET U:0 := :UINT:S
LET U:1 := :UINT:{int_minimum}
PRINTLN U:0 U:1 :UINT :UINT:+{maximum} :UINT::UINT:{maximum}
LET U:1 := :UINT:{int_maximum}
LET S := :INT:U:1
PRINTLN S :INT::UINT:7 :UINT::INT:-1
VAR R REAL
VAR Z COMPLEX
LET R := :REAL:U:1
ASSERT R == {int_maximum}
LET Z := :COMPLEX::UINT:7
LET Z += :I
LET U:1 := :UINT:Z
PRINTLN U:1 :IMAG:U:0 :IMAG::UINT:7
LET R := :IMAG::UINT:{maximum}
ASSERT R == 0
PRINTLN :UINT::REAL:3.9 :UINT:.9 :UINT:-.9 :UINT::COMPLEX:2.5
JUMP DONE
LET S := :INT:U:0
LET U:0 /= 0
@DONE
PRINTLN U:0
''', [f'{maximum} {1 << (width - 1)} 0 {maximum} {maximum}',
      f'{int_maximum} 7 {maximum}', '7 0 0', '3 0 0 2', str(maximum)])
    for operand in ('U', f':UINT:{maximum}', f':UINT:{int_maximum + 1}'):
        check_program(bra, f'QUBITS 0\nVAR U UINT\nVAR S INT\nLET U := {maximum}\n'
                      f'LET S := :INT:{operand}\n', None, 'UINT to INT conversion out of range')
    for value in ('-1.0', '-1.9', str(1 << width) + '.0'):
        check_program(bra, f'QUBITS 0\nVAR U UINT\nLET U := :UINT:{value}\n',
                      None, 'REAL to UINT conversion out of range')
    for value in ('0', '1', '-1'):
        check_program(bra, f'QUBITS 0\nVAR U UINT\nVAR R REAL\nLET R := {value}\n'
                      'LET R /= 0\nLET U := :UINT:R\n',
                      None, 'REAL to UINT conversion out of range')
    for operand in (':UINT:', ':UINT:UNKNOWN', ':UINT::PAULIS:X'):
        check_program(bra, f'QUBITS 0\nVAR U UINT\nLET U := {operand}\n', None)
    check_program(bra, f'''QUBITS 0
VAR U UINT 2
LET U:0 := {maximum}
LET U:1 := 1
PRINTLN U::INT:U:1 :UINT:U::INT:U:1 :UINT::IMAG:U:0
''', ['1 1 0'])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    bra = parser.parse_args().bra
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    width = ctypes.sizeof(ctypes.c_uint) * 8
    maximum = (1 << width) - 1
    values = (0, 1, 2, 7, maximum >> 1, (maximum >> 1) + 1, maximum - 1, maximum)
    check_operations(bra, values, maximum)
    check_comparisons(bra, values)
    check_casts(bra, maximum, width)
    check_program(bra, f'QUBITS 0\nVAR A UINT\nLET A := {maximum}\nASSERT A < 1\n',
                  None, f'evaluated: {maximum} < 1')
    print('Native UINT operations numerical tests passed')


if __name__ == '__main__':
    main()
