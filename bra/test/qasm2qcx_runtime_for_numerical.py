#!/usr/bin/env python3

"""Execute converted runtime-bound for loops with bra.

Example:
  python3 bra/test/qasm2qcx_runtime_for_numerical.py --bra bra/bin/bra
"""

import argparse
import ctypes
import importlib.util
import pathlib
import subprocess


CONVERTER_PATH = pathlib.Path(__file__).parents[1] / 'qcx' / 'qasm2qcx.py'
SPEC = importlib.util.spec_from_file_location('qasm2qcx', CONVERTER_PATH)
qasm2qcx = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(qasm2qcx)


def check_program(bra: pathlib.Path, source: str, outputs: tuple[str, ...],
                  expected: list[str]) -> None:
    lines = qasm2qcx.convert('OPENQASM 3.0; ' + source)
    lines.extend(f'PRINTLN {output}' for output in outputs)
    result = subprocess.run([str(bra)], input='\n'.join(lines) + '\n', check=True,
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
    if [line.strip() for line in result.stdout.splitlines()] != expected:
        raise RuntimeError(f'Runtime-for numerical test failed\nsource:\n{source}\n'
                           f'stdout:\n{result.stdout}\nstderr:\n{result.stderr}')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    arguments = parser.parse_args()

    for start, stop, step in ((0, 3, 1), (3, 0, -1), (2, 2, 1), (2, 2, -1),
                              (3, 0, 1), (0, 3, -1)):
        values = list(range(start, stop + step, step))
        check_program(arguments.bra, f'''int first = {start}; int last = {stop};
            int visits = 0; int total = 0;
            for int i in [first:{step}:last] {{ visits += 1; total += i; }}''',
                      ('VISITS63', 'TOTAL31'), [str(len(values)), str(sum(values))])

    check_program(arguments.bra, '''int first = 1; int last = 3; int total = 0;
        for int i in [first:last] { first = 100; last = -100; total += i; }''',
                  ('TOTAL31', 'FIRST31', 'LAST15'), ['6', '100', '-100'])

    check_program(arguments.bra, '''int i = 1; int last = 3; int total = 0;
        for int i in [i:last] { total += i; } total += i;''',
                  ('TOTAL31', 'I1'), ['7', '1'])

    check_program(arguments.bra, '''int limit = 5; int total = 0;
        for int i in [0:limit] {
            if (i == 2) { continue; } if (i == 4) { break; }
            total += i;
        }''', ('TOTAL31',), ['4'])

    check_program(arguments.bra, '''int limit = 2; int total = 0;
        for int i in [1:limit] {
            for int j in [0:i] { total += j; }
            for int k in {1, 1} { total += k; }
        }''', ('TOTAL31',), ['8'])

    check_program(arguments.bra, '''int limit = 2; int total = 0;
        for int i in [1:limit] {
            for int i in [0:i] { total += i; }
            total += i;
        }''', ('TOTAL31',), ['7'])

    check_program(arguments.bra, '''int value = 7; int limit = -1; int total = 0;
        for int i in [0:limit] { total += value % 0; }
        total += value + 1;
        for int i in [0:2] {
            for int j in [i:limit] { total += value % 0; }
        }''', ('TOTAL31',), ['8'])

    check_program(arguments.bra, '''bool active = false; int limit = 3; int total = 0;
        if (active) {
            for int i in [0:limit + 1 / 0] { total += i; }
        }
        total += limit + 1;''', ('TOTAL31',), ['4'])

    check_program(arguments.bra, '''include "stdgates.inc";
        qubit q; bit outcome = 0; int limit = 3; int visits = 0;
        for int i in [0:limit] {
            visits += 1; if (i == 1) { x q; }
            outcome = measure q; if (outcome) { break; }
        }''', ('VISITS63', 'OUTCOME127'), ['2', '1'])

    check_program(arguments.bra, '''include "stdgates.inc";
        qubit q; bit outcome; int limit = 1;
        for int i in [0:limit] { rx(pi * float(i)) q; }
        outcome = measure q;''', ('OUTCOME127',), ['1'])

    # bra::int_type is C++ int. Match the host C integer endpoints rather than
    # assuming that the converter's accepted OpenQASM widths are enforced.
    bits = ctypes.sizeof(ctypes.c_int) * 8
    maximum, minimum = 2 ** (bits - 1) - 1, -(2 ** (bits - 1))
    for start, stop, step in ((maximum - 1, maximum, 1), (minimum + 1, minimum, -1)):
        check_program(arguments.bra, f'''int first = {start}; int last = {stop}; int visits = 0;
            for int i in [first:{step}:last] {{ visits += 1; continue; }}''',
                      ('VISITS63',), ['2'])


if __name__ == '__main__':
    main()
