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

    source = '''OPENQASM 3.0; int sum = 0;
        for int i in [0:2] {
            for int j in [0:i] { sum += i * 10 + j; }
        }'''
    check_program(arguments.bra, source, ('SUM7',), ['84'])

    source = '''OPENQASM 3.0; const int i = 9; int sum = 0;
        for int i in [1:2] {
            sum += i;
            for int i in [0:i] { sum += i; }
            sum += i;
        }
        sum += i;'''
    check_program(arguments.bra, source, ('SUM7',), ['19'])

    source = '''OPENQASM 3.0; include "stdgates.inc";
        qubit[4] q; bit[4] flags; int sum = 0;
        for int i in [0:1] {
            for int j in [0:1] {
                x q[2 * i + j];
                flags[2 * i + j] = measure q[2 * i + j];
                if (flags[2 * i + j]) { sum += 2 * i + j + 1; }
            }
        }'''
    check_program(arguments.bra, source,
                  ('FLAGS31:0', 'FLAGS31:1', 'FLAGS31:2', 'FLAGS31:3', 'SUM7'),
                  ['1', '1', '1', '1', '10'])

    source = '''OPENQASM 3.0; bool ready = false; int value = 7; int sum = 0;
        if (ready) {
            for int i in [0:1] { for int j in [0:1] { sum += value / j; } }
        }
        for int i in [0:1] { for int j in [1:2] { sum += value % j; } }
        sum += value;'''
    check_program(arguments.bra, source, ('SUM7',), ['9'])

    source = '''OPENQASM 3.0; int sum = 0;
        for int i in [2:-1:0] {
            for int j in [i:-1:1] { sum += i * 10 + j; }
        }'''
    check_program(arguments.bra, source, ('SUM7',), ['54'])

    # Complete classical example from bra/qcx/qasm2qcx.md.
    source = '''OPENQASM 3.0; int total = 0;
        for int i in [1:3] {
            for int j in [0:i] { total += i + j; }
        }'''
    check_program(arguments.bra, source, ('TOTAL31',), ['30'])


if __name__ == '__main__':
    main()
