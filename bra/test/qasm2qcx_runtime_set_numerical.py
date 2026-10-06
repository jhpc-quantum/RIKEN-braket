#!/usr/bin/env python3

"""Execute converted runtime-valued integer-set loops with bra.

Example (disable core dumps for deliberate runtime-error checks):
  ulimit -c 0
  python3 bra/test/qasm2qcx_runtime_set_numerical.py --bra bra/bin/bra
"""

import argparse
import ctypes
import importlib.util
import itertools
import pathlib
import subprocess


CONVERTER_PATH = pathlib.Path(__file__).parents[1] / 'qcx' / 'qasm2qcx.py'
SPEC = importlib.util.spec_from_file_location('qasm2qcx', CONVERTER_PATH)
qasm2qcx = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(qasm2qcx)


def check_program(bra: pathlib.Path, source: str, outputs: tuple[str, ...],
                  expected: list[str] | None) -> None:
    lines = qasm2qcx.convert('OPENQASM 3.0; ' + source)
    lines.extend(f'PRINTLN {output}' for output in outputs)
    result = subprocess.run([str(bra)], input='\n'.join(lines) + '\n', text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
    if expected is None:
        instruction = next(line for line in lines if ' /= ' in line)
        valid = (result.returncode != 0
                 and f'integer division by zero in {instruction}' in result.stderr)
    else:
        valid = (result.returncode == 0
                 and [line.strip() for line in result.stdout.splitlines()] == expected)
    if not valid:
        raise RuntimeError(f'Runtime-set numerical test failed\nsource:\n{source}\n'
                           f'exit status: {result.returncode}\nstdout:\n{result.stdout}\n'
                           f'stderr:\n{result.stderr}')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    arguments = parser.parse_args()

    for values in ((2, -1, 2), (0, 0, 0), (-3, 1, -3), (1, 2, 3)):
        a, b, c = values
        check_program(arguments.bra, f'''int a = {a}; int b = {b}; int c = {c};
            int total = 0; int visits = 0;
            for int i in {{a, b, c}} {{ total += i; visits += 1; }}''',
                      ('TOTAL31', 'VISITS63'), [str(sum(values)), '3'])

    # Transfers by position distinguish repeated values and test source order.
    for values in itertools.product((-1, 0, 2), repeat=3):
        for skip, stop in ((0, 2), (1, -1), (-1, 1)):
            total, visits = 0, 0
            for index, value in enumerate(values):
                visits += 1
                if index == skip:
                    continue
                if index == stop:
                    break
                total = total * 7 + value
            a, b, c = values
            check_program(arguments.bra, f'''int a = {a}; int b = {b}; int c = {c};
                int total = 0; int visits = 0; int n = -1;
                for int i in {{a, b, c}} {{
                    n += 1; visits += 1;
                    if (n == {skip}) {{ continue; }} if (n == {stop}) {{ break; }}
                    total = total * 7 + i;
                }}''', ('TOTAL31', 'VISITS63'), [str(total), str(visits)])

    check_program(arguments.bra, '''int a = 1; int b = 3; int total = 0; int order = 0;
        for int i in {a, b, a + 1, a} {
            total += i; order = order * 10 + i; a = 9; b = 9;
        }''', ('TOTAL31', 'ORDER31', 'A1', 'B1'), ['7', '1321', '9', '9'])

    check_program(arguments.bra, '''int i = 2; int total = 0;
        for int i in {i, i + 1} { total += i; } total += i;''',
                  ('TOTAL31', 'I1'), ['7', '2'])

    check_program(arguments.bra, '''int a = 1; int b = 3; int n = 0; int total = 0;
        for int i in {a, a, b} {
            n += 1; if (n == 1) { continue; } if (n == 3) { break; } total += i;
        }''', ('TOTAL31', 'N1'), ['1', '3'])

    check_program(arguments.bra, '''int a = 1; int total = 0;
        for int i in {a, a + 1} {
            for int i in {i, i + 1} { total += i; }
            total += i;
        }''', ('TOTAL31',), ['11'])

    check_program(arguments.bra, '''int last = 2; int total = 0;
        for int i in [0:last] { for int j in {i, i + 1} { total += j; } }''',
                  ('TOTAL31',), ['9'])

    check_program(arguments.bra, '''int a = 1; int total = 0;
        for int i in {a, a + 1} {
            for int j in [0:i] { total += j; }
            for int j in {1, 1} { total += j; }
        }''', ('TOTAL31',), ['8'])

    check_program(arguments.bra, '''int a = 1; int tick = 0; int total = 0;
        while (tick < 3) {
            tick += 1; for int i in {a, a + 1} { total += i; a += 1; }
        }''', ('TOTAL31', 'A1'), ['21', '7'])

    check_program(arguments.bra, '''int a = 1; int total = 0;
        for int i in {0, 1} {
            for int j in {a, i, a + 1} { total += j; a += 1; }
        }''', ('TOTAL31', 'A1'), ['13', '7'])

    check_program(arguments.bra, '''int a = 1; int total = 0; int tick = 0;
        for int i in {a, a + 1} {
            tick = 0;
            while (tick < 4) {
                tick += 1; if (tick == 1) { continue; }
                total += i; if (tick == 3) { break; }
            }
            for int k in {0, 1} { if (k == 0) { continue; } total += k; break; }
            total += 10;
        }''', ('TOTAL31', 'TICK15'), ['28', '3'])

    check_program(arguments.bra, '''uint a = 1; float f = 2.5; bool ready = true;
        bit[2] flags = "01"; int total = 0;
        for int i in {a, int(f), int(ready), int(flags[0]), int(1.0im)} { total += i; }''',
                  ('TOTAL31',), ['5'])

    check_program(arguments.bra, '''int a = 7; int b = 3; int total = 0;
        for int i in {a % b, (a + b) / 2, -b} { total += i; }
        total += a + b;''', ('TOTAL31',), ['13'])

    check_program(arguments.bra, '''int a = 1; int total = 0;
        if (false) { for int i in {a / 0, a % 0} { total += i; } }
        while (false) { for int i in {a / 0} { total += i; } }
        for int i in {a, a} { continue; total += a % 0; }
        total += a + 1;''', ('TOTAL31',), ['2'])

    check_program(arguments.bra, '''int a = 1; int total = 0;
        for int i in [a:-1] { for int j in {a / 0, a} { total += j; } }
        if (true) { total += 1; } else { for int j in {a % 0} { total += j; } }
        total += a;''', ('TOTAL31',), ['2'])

    check_program(arguments.bra, '''bool ready = false; int a = 1; int total = 0;
        for int i in {int(ready && bool(a / 0)), int(!ready || bool(a / 0)), a} {
            total += i;
        }''', ('TOTAL31',), ['2'])

    check_program(arguments.bra, '''bit i = 1; int total = 0;
        for int i in {int(i), int(i) + 1} { total += i; } total += int(i);''',
                  ('TOTAL31', 'I1'), ['4', '1'])

    check_program(arguments.bra, '''int QASM2QCX_INT_ = 9; int a = 1; int total = 0;
        for int i in {a, a + 1} { total += i; }
        total += QASM2QCX_INT_;''', ('TOTAL31', 'QASM2QCX_INT_0'), ['12', '9'])

    check_program(arguments.bra, '''int a = 7; int total = 0;
        for int i in {a, a + 1} { total += i; break; }
        for int i in {a % 3, a / 3} { total += i; continue; }
        total += (a + 2) % 4;''', ('TOTAL31',), ['11'])

    # Sets do not advance an iterator, even at the backend integer endpoints.
    bits = ctypes.sizeof(ctypes.c_int) * 8
    maximum, minimum = 2 ** (bits - 1) - 1, -(2 ** (bits - 1))
    check_program(arguments.bra, f'''int a = {maximum}; int b = {minimum}; int visits = 0;
        int positive = 0; int negative = 0;
        for int i in {{a, b, a, b}} {{
            visits += 1; if (i == a) {{ positive += 1; continue; }} negative += 1;
        }}''', ('VISITS63', 'POSITIVE255', 'NEGATIVE255'), ['4', '2', '2'])

    check_program(arguments.bra, '''int a = 1; int total = 0;
        for int i in {a, a} {
            break; for int j in {a / 0, a} { total += j; }
        }
        total += a + 1;''', ('TOTAL31',), ['2'])

    # Even a break in the first body cannot skip capture of a later element.
    check_program(arguments.bra, '''int a = 1; int zero = 0;
        for int i in {a, a / zero} { break; }''', (), None)

    check_program(arguments.bra, '''include "stdgates.inc";
        qubit q; bit outcome = 0; int a = 0; int visits = 0;
        for int i in {a, a + 1, a + 2} {
            visits += 1; if (i == 1) { x q; }
            outcome = measure q; if (outcome) { break; }
        }''', ('VISITS63', 'OUTCOME127'), ['2', '1'])

    check_program(arguments.bra, '''include "stdgates.inc";
        qubit q; bit outcome; int a = 1;
        for int i in {a, 0} { rx(pi * float(i)) q; }
        outcome = measure q;''', ('OUTCOME127',), ['1'])


if __name__ == '__main__':
    main()
