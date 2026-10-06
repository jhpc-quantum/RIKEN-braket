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

    # Cross zero in both directions, with equal and reversed endpoints too.
    for start, stop, step in ((start, stop, step)
                              for start in range(-2, 3)
                              for stop in range(-2, 3)
                              for step in (1, -1, 2, -2, 3, -3)):
        values = list(range(start, stop + (1 if step > 0 else -1), step))
        check_program(arguments.bra, f'''int first = {start}; int last = {stop};
            int visits = 0; int total = 0;
            for int i in [first:{step}:last] {{ visits += 1; total += i; }}''',
                      ('VISITS63', 'TOTAL31'), [str(len(values)), str(sum(values))])

    check_program(arguments.bra, '''int first = 1; int last = 3; int total = 0;
        for int i in [first:last] { first = 100; last = -100; total += i; }''',
                  ('TOTAL31', 'FIRST31', 'LAST15'), ['6', '100', '-100'])

    check_program(arguments.bra, '''int first = 1; int last = 6; int total = 0;
        for int i in [first:2:last] { first = 100; last = -100; total += i; }''',
                  ('TOTAL31', 'FIRST31', 'LAST15'), ['9', '100', '-100'])

    check_program(arguments.bra, '''int first = 6; int last = -1; int total = 0;
        for int i in [first:-3:last] { first = -100; last = 100; total += i; }''',
                  ('TOTAL31', 'FIRST31', 'LAST15'), ['9', '-100', '100'])

    check_program(arguments.bra, '''int first = 7; int last = 2; int total = 0;
        for int i in [first / 3:2:(last + 2) * 2] { total += i; }
        total += first % last;''', ('TOTAL31',), ['21'])

    check_program(arguments.bra, '''int i = 1; int last = 3; int total = 0;
        for int i in [i:last] { total += i; } total += i;''',
                  ('TOTAL31', 'I1'), ['7', '1'])

    check_program(arguments.bra, '''int limit = 5; int total = 0;
        for int i in [0:limit] {
            if (i == 2) { continue; } if (i == 4) { break; }
            total += i;
        }''', ('TOTAL31',), ['4'])

    for first, last, step, skip, stop in ((1, 8, 2, 3, 7), (8, -1, -3, 5, 2)):
        total, visits = 0, 0
        for value in range(first, last + (1 if step > 0 else -1), step):
            visits += 1
            if value == skip:
                continue
            if value == stop:
                break
            total += value
        check_program(arguments.bra, f'''int first = {first}; int last = {last};
            int total = 0; int visits = 0;
            for int i in [first:{step}:last] {{
                visits += 1; if (i == {skip}) {{ continue; }} if (i == {stop}) {{ break; }}
                total += i;
            }}''', ('TOTAL31', 'VISITS63'), [str(total), str(visits)])

    check_program(arguments.bra, '''int last = 4; int total = 0;
        for int i in [0:2:last] {
            for int j in [i:-3:-2] { total += j; }
            for int k in {i, i + 1} { total += k; }
        }''', ('TOTAL31',), [str(sum(sum(range(i, -3, -3)) + i + (i + 1)
                                   for i in range(0, 5, 2)))])

    check_program(arguments.bra, '''int last = 5; int total = 0;
        for int stride in {2, -3} {
            for int i in [0:stride:last] { total += i; }
        }''', ('TOTAL31',), ['6'])

    check_program(arguments.bra, '''int first = 1; int total = 0;
        for int i in {first, first + 1} { for int j in [i:2:5] { total += j; } }''',
                  ('TOTAL31',), ['15'])

    check_program(arguments.bra, '''int first = 1; int last = -1; int total = 0;
        for int i in [first:2:last] { total += first / 0; }
        if (false) { for int i in [0:3:last + first / 0] { total += i; } }
        total += first + 1;''', ('TOTAL31',), ['2'])

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

    # Arithmetic bound temporaries must not overwrite either captured value.
    check_program(arguments.bra, '''int first = 7; int last = 3; int total = 0;
        for int i in [first % last:(last + 1) * 2 / 2] {
            total += (i + first) % last; first = 100; last = 100;
        }
        total += 2;''', ('TOTAL31',), ['13'])

    for declarations, bounds, expected in (
            ('float first = -1.5; uint last = 2;', '[int(first):last]', '2'),
            ('bool ready = true; int last = 2;', '[int(!ready):last]', '3'),
            ('bit[2] flags = "01"; int last = 2;', '[int(flags[0]):last]', '3'),
            ('int first = 0; int last = 2;', '[int(first < last):last]', '3'),
            ('const complex first = 2.0 + 1.0im; int last = 3;',
             '[int(first):last]', '5'),
            ('const int stride = 1; int last = -1;', '[2:-stride:last]', '2')):
        check_program(arguments.bra, declarations + f'''int total = 0;
            for int i in {bounds} {{ total += i; }}''', ('TOTAL31',), [expected])

    # Inner transfers target their nearest loop, not the enclosing runtime for.
    check_program(arguments.bra, '''int limit = 2; int total = 0; int tick = 0;
        for int i in [0:limit] {
            tick = 0;
            while (tick < 4) {
                tick += 1; if (tick == 1) { continue; }
                total += i; if (tick == 3) { break; }
            }
            for int j in {1, 1, 2} {
                if (j == 1) { continue; } total += j; break;
            }
            total += 10;
        }''', ('TOTAL31', 'TICK15'), ['42', '3'])

    # A runtime loop inside an expanded loop captures fresh bounds each entry.
    check_program(arguments.bra, '''int last = 1; int total = 0;
        for int k in {1, 2, 3} {
            for int i in [0:last] { total += i; last += 1; }
        }''', ('TOTAL31', 'LAST15'), ['35', '15'])

    check_program(arguments.bra, '''int limit = 1; int total = 0; int tick = 0;
        while (tick < 3) {
            tick += 1;
            for int i in [0:limit] { if (i == 0) { continue; } total += tick; break; }
            limit += 1;
        }''', ('TOTAL31', 'LIMIT31'), ['6', '4'])

    check_program(arguments.bra, '''int first = 4; int last = 0; int total = 0;
        for int i in [first:-1:last] {
            if (i == 3) { continue; } if (i == 1) { break; } total += i;
        }''', ('TOTAL31',), ['6'])

    # Ordinary source variables with the iterator name are restored afterwards.
    check_program(arguments.bra, '''bit i = 1; int last = 2; int total = 0;
        for int i in [int(i):last] { total += i; } total += int(i);''',
                  ('TOTAL31', 'I1'), ['4', '1'])

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
    for start, stop, step in ((maximum - 1, maximum, 1), (minimum + 1, minimum, -1),
                              (maximum, maximum, 1), (minimum, minimum, -1),
                              (maximum, minimum, 1), (minimum, maximum, -1),
                              (maximum - 3, maximum, 2), (minimum + 3, minimum, -2),
                              (maximum - 2, maximum, 2), (minimum + 2, minimum, -2),
                              (maximum, maximum, 2), (minimum, minimum, -2),
                              (maximum, minimum, 2), (minimum, maximum, -2),
                              (minimum, maximum, maximum), (maximum, minimum, minimum),
                              (0, minimum, minimum), (0, maximum, maximum),
                              (minimum, -1, maximum), (maximum, 0, minimum),
                              (minimum, 0, maximum), (maximum, -1, minimum)):
        check_program(arguments.bra, f'''int first = {start}; int last = {stop}; int visits = 0;
            for int i in [first:{step}:last] {{ visits += 1; continue; }}''',
                      ('VISITS63',), [str(len(range(start, stop + (1 if step > 0 else -1), step)))])


if __name__ == '__main__':
    main()
