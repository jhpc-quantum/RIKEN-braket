#!/usr/bin/env python3

"""Execute converted runtime while loops with bra.

Example:
  python3 bra/test/qasm2qcx_while_loop_numerical.py --bra bra/bin/bra
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
    lines = qasm2qcx.convert('OPENQASM 3.0; ' + source)
    lines.extend(f'PRINTLN {output}' for output in outputs)
    result = subprocess.run(
        [str(bra)], input='\n'.join(lines) + '\n', check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30,
    )
    if [line.strip() for line in result.stdout.splitlines()] != expected:
        raise RuntimeError(f'While-loop numerical test failed\nsource:\n{source}\n'
                           f'stdout:\n{result.stdout}\nstderr:\n{result.stderr}')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    arguments = parser.parse_args()

    check_program(arguments.bra, '''int n = 0; int total = 0;
        while (n < 5) { n += 1; total += n; }''', ('N1', 'TOTAL31'), ['5', '15'])

    # Neither the body nor its zero-divisor computation executes; the following
    # expression must still be able to reuse the temporary storage.
    check_program(arguments.bra, '''int value = 7; int total = 0;
        while (false) { total += value % 0; }
        total += value + 1;''', ('TOTAL31',), ['8'])

    check_program(arguments.bra, '''int n = 0; int total = 0;
        while (n < 6) {
            n += 1;
            if (n == 2) { continue; }
            if (n == 4) { break; }
            total += n;
        }
        total += 100;''', ('N1', 'TOTAL31'), ['4', '104'])

    # Arithmetic in the condition must be recomputed on every back edge.
    check_program(arguments.bra, '''int n = 5;
        while (n % 3 != 0) { n -= 1; }''', ('N1',), ['3'])

    # Short-circuiting keeps runtime zero divisors out of executed paths.
    check_program(arguments.bra, '''int n = 2; int total = 0;
        while (n > 0 && 4 / n > 0) { total += n; n -= 1; }''',
                  ('N1', 'TOTAL31'), ['0', '3'])

    check_program(arguments.bra, '''include "stdgates.inc";
        qubit q; bit outcome = 0; int visits = 0;
        while (!outcome) {
            visits += 1;
            if (visits == 2) { x q; }
            outcome = measure q;
        }''', ('VISITS63', 'OUTCOME127'), ['2', '1'])

    check_program(arguments.bra, '''int n = 0; int total = 0;
        while (n < 2) {
            n += 1;
            for int i in [0:2] {
                if (i == 0) { continue; }
                total += n + i;
                break;
            }
        }''', ('N1', 'TOTAL31'), ['2', '5'])

    check_program(arguments.bra, '''int n = 0; int total = 0;
        for int i in [1:2] {
            n = 0;
            while (n < 4) {
                n += 1;
                if (n == 1) { continue; }
                total += i;
                break;
            }
            total += 10;
        }''', ('N1', 'TOTAL31'), ['2', '23'])

    check_program(arguments.bra, '''int outer = 0; int inner = 0; int total = 0;
        while (outer < 2) {
            outer += 1;
            inner = 0;
            while (inner < 3) {
                inner += 1;
                if (inner == 1) { continue; }
                total += outer;
                break;
            }
            total += 10;
        }''', ('TOTAL31',), ['23'])

    # First use of body temporaries is skipped by continue. Later iterations
    # and code after break reuse the same storage safely.
    check_program(arguments.bra, '''int value = 7; int n = 0; int total = 0;
        while (n < 4) {
            n += 1;
            if (n == 1) { continue; }
            total += value % n;
            if (n == 3) { break; }
        }
        total += value + 1;''', ('N1', 'TOTAL31'), ['3', '10'])

    check_program(arguments.bra, '''bool ready = true; int n = 2;
        while (ready && bool(n)) { n -= 1; ready = n > 0; }''',
                  ('N1', 'READY31'), ['0', '0'])


if __name__ == '__main__':
    main()
