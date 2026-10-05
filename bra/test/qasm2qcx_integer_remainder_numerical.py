#!/usr/bin/env python3

"""Execute converted integer remainder expressions with bra.

Example:
  python3 bra/test/qasm2qcx_integer_remainder_numerical.py --bra bra/bin/bra
"""

import argparse
import importlib.util
import pathlib
import subprocess


CONVERTER_PATH = pathlib.Path(__file__).parents[1] / "qcx" / "qasm2qcx.py"
SPEC = importlib.util.spec_from_file_location("qasm2qcx", CONVERTER_PATH)
qasm2qcx = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(qasm2qcx)


def check_program(
        bra: pathlib.Path, source: str, outputs: tuple[str, ...],
        expected: list[str]) -> None:
    lines = qasm2qcx.convert(source)
    lines.extend(f'PRINTLN {output}' for output in outputs)
    result = subprocess.run(
        [str(bra)], input="\n".join(lines) + "\n", check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30,
    )
    if [line.strip() for line in result.stdout.splitlines()] != expected:
        raise RuntimeError(
            f"Integer remainder numerical test failed\nsource:\n{source}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bra", required=True, type=pathlib.Path)
    arguments = parser.parse_args()

    for lhs in (-7, -6, -1, 0, 1, 6, 7):
        for rhs in (-3, -1, 1, 3):
            expected = abs(lhs) % abs(rhs) * (-1 if lhs < 0 else 1)
            for expression in ('a % b', f'a % ({rhs})', f'({lhs}) % b'):
                source = f'''OPENQASM 3.0; int a = {lhs}; int b = {rhs};
                    int result = {expression};'''
                check_program(arguments.bra, source, ('RESULT63', 'A1', 'B1'),
                              [str(expected), str(lhs), str(rhs)])
            for operand in ('b', f'({rhs})'):
                source = f'''OPENQASM 3.0; int a = {lhs}; int b = {rhs};
                    a %= {operand};'''
                check_program(arguments.bra, source, ('A1', 'B1'),
                              [str(expected), str(rhs)])

    for expression, expected in (
            ('(a + 1) % (b + 1)', 0), ('a % (b % a)', 1),
            ('(a % b) + (b % a)', 4), ('(a % b) % (b - 1)', 1),
            ('-a % b', -1), ('a + b % 2 * 3', 10), ('int(f) % b', 1)):
        source = f'''OPENQASM 3.0; int a = 7; int b = 3; float f = 7.5;
            int result = {expression};'''
        check_program(arguments.bra, source, ('RESULT63', 'A1', 'B1'),
                      [str(expected), '7', '3'])

    source = '''OPENQASM 3.0; int a = 7; int b = 3;
        int result = a % b; result = b % a; a = a % a;
        result = result + b;'''
    check_program(arguments.bra, source, ('RESULT63', 'A1', 'B1'), ['6', '0', '3'])

    source = '''OPENQASM 3.0; uint a = 7; int b = 3; uint result = a % b;'''
    check_program(arguments.bra, source, ('RESULT63',), ['1'])

    for expression, expected in (
            ('a && 7 % 0 == 0', '0'), ('!a || bool(7 % 0)', '1'),
            ('a && 7 % divisor == 0', '0'), ('a && divisor % 0 == 0', '0')):
        source = f'''OPENQASM 3.0; bool a = false; int divisor = 0;
            bool result = {expression}; divisor = divisor + 1;'''
        check_program(arguments.bra, source, ('RESULT63', 'DIVISOR127'),
                      [expected, '1'])

    source = '''OPENQASM 3.0; bool a = false; int n = 7; int result = 0;
        if (a) { result = 7 % 0; }
        result = n % 3; n = n + 1;'''
    check_program(arguments.bra, source, ('RESULT63', 'N1'), ['1', '8'])

    source = '''OPENQASM 3.0; int n = -7; bool ready = n % 3 == -1;
        int result = 0;
        if (ready && n % 2 != 0) { result = 1; }'''
    check_program(arguments.bra, source, ('RESULT63', 'READY31'), ['1', '1'])

    for operand, expected in (
            ('a', 0), ('a - b', 3), ('(a % b) + 1', 1), ('int(f)', 1)):
        source = f'''OPENQASM 3.0; int a = 7; uint b = 3; float f = 3.5;
            a %= {operand};'''
        check_program(arguments.bra, source, ('A1', 'B1'), [str(expected), '3'])

    source = '''OPENQASM 3.0; uint a = 7; int b = 3;
        a %= b; b %= a; a += 2;'''
    check_program(arguments.bra, source, ('A1', 'B1'), ['3', '0'])

    source = '''OPENQASM 3.0; int a = 7;
        if (false) { a %= 0; }
        a %= 3;
        if (true) { a %= 1; } else { a %= 0; }'''
    check_program(arguments.bra, source, ('A1',), ['0'])


if __name__ == "__main__":
    main()
