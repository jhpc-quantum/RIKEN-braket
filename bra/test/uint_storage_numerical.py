#!/usr/bin/env python3

"""Check native UINT declarations, assignment, indexed reads, and printing.

Example:
  python3 bra/test/uint_storage_numerical.py --bra bra/bin/bra
"""

import argparse
import ctypes
import pathlib
import resource
import subprocess


def check_program(bra: pathlib.Path, source: str, expected: list[str] | None,
                  diagnostic: str = '') -> None:
    result = subprocess.run([str(bra)], input=source, text=True,
                            capture_output=True, timeout=30)
    if expected is None:
        valid = result.returncode != 0 and diagnostic in result.stderr
    else:
        valid = result.returncode == 0 and result.stdout.splitlines() == expected
    if not valid:
        raise RuntimeError(f'UINT storage test failed\nsource:\n{source}\n{result}')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    bra = parser.parse_args().bra
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    width = ctypes.sizeof(ctypes.c_uint) * 8
    maximum = (1 << width) - 1
    int_maximum = (1 << (width - 1)) - 1
    for value in (0, 1, int_maximum, int_maximum + 1, maximum - 1, maximum):
        for literal in (str(value), f'+{value}'):
            check_program(bra, f'''QUBITS 0
VAR A UINT
VAR VALUES UINT 3
VAR POINTERS INT 2
VAR INDEX INT
PRINTLN A VALUES:0 VALUES:1 VALUES:2
LET INDEX := 1
LET POINTERS:INDEX := 2
LET A := {literal}
LET VALUES:POINTERS:INDEX := A
LET VALUES:0 := VALUES:2
LET A := A
LET VALUES:2 := VALUES:2
PRINTLN A VALUES:0 VALUES:1 VALUES:2 INDEX
''', ['0 0 0 0', f'{value} {value} 0 {value} 1'])

    check_program(bra, f'''QUBITS 0
var a uint
var b uint
let a := {maximum}
let b := a
PRINT A
PRINTLN B
VAR SIGNED INT
LET SIGNED := -1
PRINTLN SIGNED A
JUMP DONE
LET A := -1
@DONE
PRINTLN A
''', [f'{maximum}{maximum}', f'-1 {maximum}', str(maximum)])

    for literal in ('-1', '-0', '1.5', '1e2', str(maximum + 1), str(1 << (width * 2))):
        check_program(bra, f'QUBITS 0\nVAR A UINT\nLET A := {literal}\n', None)
    for kind in ('INT', 'REAL', 'COMPLEX', 'PAULISS'):
        check_program(bra, f'QUBITS 0\nVAR A UINT\nVAR B {kind}\nLET A := B\n', None)
        for first, second in (('UINT', kind), (kind, 'UINT')):
            check_program(bra, f'QUBITS 0\nVAR A {first}\nVAR A {second}\n',
                          None, 'variable already declared: A')
    check_program(bra, 'QUBITS 0\nVAR A UINT\nVAR B INT\nLET B := A\n', None)
    check_program(bra, 'QUBITS 0\nVAR A UINT\nVAR A UINT\n',
                  None, 'variable already declared: A')
    for size in ('0', '-1'):
        check_program(bra, f'QUBITS 0\nVAR A UINT {size}\n',
                      None, 'UINT variable size must be positive: A')
    for index in ('-1', '2'):
        for instruction in (f'LET A:{index} := 1', f'PRINTLN A:{index}',
                            f'LET B := A:{index}'):
            check_program(bra, f'QUBITS 0\nVAR A UINT 2\nVAR B UINT\n{instruction}\n', None)
    print('Native UINT storage numerical tests passed')


if __name__ == '__main__':
    main()
