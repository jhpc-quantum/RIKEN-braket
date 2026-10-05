#!/usr/bin/env python3

"""Check QCX integer division, including executed and skipped zero divisors.

Example:
  python3 bra/test/integer_division_numerical.py --bra bra/bin/bra
"""

import argparse
import importlib.util
import pathlib
import subprocess


def check_program(bra: pathlib.Path, source: str, expected: list[str] | None,
                  instruction: str = '') -> None:
    result = subprocess.run(
        [str(bra)], input=source, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, timeout=30,
    )
    if expected is None:
        # The CLI currently leaves runtime exceptions uncaught. Require the
        # specific diagnostic, rather than accepting any nonzero status/signal.
        valid = (result.returncode != 0
                 and f'integer division by zero in {instruction}' in result.stderr)
    else:
        valid = (result.returncode == 0
                 and [line.strip() for line in result.stdout.splitlines()] == expected)
    if not valid:
        raise RuntimeError(
            f'Integer division test failed\nsource:\n{source}\n'
            f'exit status: {result.returncode}\nstdout:\n{result.stdout}\n'
            f'stderr:\n{result.stderr}')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    arguments = parser.parse_args()

    prefix = 'QUBITS 0\nVAR A INT\nLET A := 7\nVAR B INT\n'
    for divisor in ('0', 'B', 'A'):
        setup = 'LET A := 0\n' if divisor == 'A' else ''
        check_program(arguments.bra, prefix + setup + f'LET A /= {divisor}\n',
                      None, f'LET A /= {divisor}')

    for target, divisor in (('VALUES:0', '0'), ('VALUES:1', 'VALUES:0'),
                            ('VALUES:INDEX', 'VALUES:INDEX')):
        source = ('QUBITS 0\nVAR VALUES INT 2\nVAR INDEX INT\n'
                  f'LET {target} /= {divisor}\n')
        check_program(arguments.bra, source, None, f'LET {target} /= {divisor}')

    # to_int can also yield zero after an explicit real-to-integer conversion.
    check_program(arguments.bra, prefix + 'LET A /= :INT:0.5\n',
                  None, 'LET A /= :INT:0.5')

    for lhs in (-7, -6, -1, 0, 1, 6, 7):
        for rhs in (-3, -1, 1, 3):
            expected = abs(lhs) // abs(rhs) * (-1 if (lhs < 0) != (rhs < 0) else 1)
            for divisor in (str(rhs), 'B'):
                source = (prefix + f'LET A := {lhs}\nLET B := {rhs}\n'
                          f'LET A /= {divisor}\nPRINTLN A B\n')
                check_program(arguments.bra, source, [f'{expected} {rhs}'])

    source = prefix + '''JUMP AFTER_LITERAL
LET A /= 0
@AFTER_LITERAL
JUMPIF AFTER_VARIABLE A == 7
LET A /= B
@AFTER_VARIABLE
LET A /= 3
PRINTLN A
'''
    check_program(arguments.bra, source, ['2'])

    source = '''QUBITS 0
VAR VALUES INT 2
VAR INDEX INT
LET INDEX := 1
LET VALUES:INDEX := -7
LET VALUES:0 := 3
LET VALUES:INDEX /= VALUES:0
PRINTLN VALUES:1 VALUES:0
'''
    check_program(arguments.bra, source, ['-2 3'])

    converter_path = pathlib.Path(__file__).parents[1] / 'qcx' / 'qasm2qcx.py'
    spec = importlib.util.spec_from_file_location('qasm2qcx', converter_path)
    converter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(converter)
    for statement in ('a = a / b;', 'a /= b;', 'a = a % b;', 'a %= b;'):
        lines = converter.convert('OPENQASM 3.0; int a = 7; int b = 0; ' + statement)
        instruction = next(line for line in lines if ' /= ' in line)
        check_program(arguments.bra, '\n'.join(lines) + '\n', None, instruction)


if __name__ == '__main__':
    main()
