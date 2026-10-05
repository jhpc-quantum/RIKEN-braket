#!/usr/bin/env python3

"""Execute converted constant-range for loops with bra.

Example:
  python3 bra/test/qasm2qcx_for_loop_numerical.py --bra bra/bin/bra
"""

import argparse
import importlib.util
import pathlib
import subprocess


CONVERTER_PATH = pathlib.Path(__file__).parents[1] / 'qcx' / 'qasm2qcx.py'
SPEC = importlib.util.spec_from_file_location('qasm2qcx', CONVERTER_PATH)
qasm2qcx = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(qasm2qcx)


def check_program(bra: pathlib.Path, source: str, outputs: tuple[str, ...],
                  expected: list[str]) -> None:
    lines = qasm2qcx.convert(source)
    lines.extend(f'PRINTLN {output}' for output in outputs)
    result = subprocess.run(
        [str(bra)], input='\n'.join(lines) + '\n', check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30,
    )
    if [line.strip() for line in result.stdout.splitlines()] != expected:
        raise RuntimeError(f'For-loop numerical test failed\nsource:\n{source}\n'
                           f'stdout:\n{result.stdout}\nstderr:\n{result.stderr}')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    arguments = parser.parse_args()

    source = '''OPENQASM 3.0; int sum = 0; int i = 99;
        for int i in [0:3] { sum += i; }
        for int i in [3:-2:0] { sum += i; }
        for int i in [3:0] { sum += i; }
        sum += i;'''
    check_program(arguments.bra, source, ('SUM7', 'I1'), ['109', '99'])

    source = '''OPENQASM 3.0; include "stdgates.inc";
        qubit[4] q; bit[4] flags; int sum = 0;
        for int i in [0:3] {
            if (i % 2 == 0) { x q[i]; }
            flags[i] = measure q[i];
            if (flags[i]) { sum += i + 1; }
            reset q[i];
        }
        for int i in [0:3] { flags[i] = measure q[i]; }
        if (flags[0] || flags[1] || flags[2] || flags[3]) { sum += 100; }'''
    check_program(arguments.bra, source, ('SUM7',), ['4'])

    source = '''OPENQASM 3.0; int sum = 0; int divisor = 0;
        for int i in [0:3] {
            if (i == 0 || false && 1 / divisor > 0) { sum += 1; }
            else { sum += i; }
        }
        sum += 10;'''
    check_program(arguments.bra, source, ('SUM7',), ['17'])

    source = '''OPENQASM 3.0; const int i = 8; int sum = 0; int value = 7;
        for int i in [0:2] { sum += value % (i + 1); value -= 1; }
        sum += i;'''
    check_program(arguments.bra, source, ('SUM7', 'VALUE31'), ['10', '4'])


if __name__ == '__main__':
    main()
