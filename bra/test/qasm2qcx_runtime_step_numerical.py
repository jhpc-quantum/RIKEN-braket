#!/usr/bin/env python3

"""Execute converted runtime-valued for-loop steps with bra.

Example:
  python3 bra/test/qasm2qcx_runtime_step_numerical.py --bra bra/bin/bra
"""

import argparse
import pathlib
import resource
import subprocess

from qasm2qcx_runtime_for_numerical import check_program, qasm2qcx


def check_error(bra: pathlib.Path, source: str, diagnostic: str) -> None:
    lines = qasm2qcx.convert('OPENQASM 3.0; ' + source)
    lines.append('PRINTLN :INT:99')
    result = subprocess.run([str(bra)], input='\n'.join(lines) + '\n',
                            text=True, capture_output=True, timeout=10)
    if result.returncode == 0 or diagnostic not in result.stderr or result.stdout.strip():
        raise RuntimeError(f'Expected runtime error {diagnostic!r}: {result}')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    bra = parser.parse_args().bra
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    # Bounds are literal: the step alone must select runtime lowering.
    for first in range(-2, 3):
        for last in range(-2, 3):
            for step in (1, -1, 2, -2, 3, -3):
                values = list(range(first, last + (1 if step > 0 else -1), step))
                check_program(bra, f'''int stride = {step}; int visits = 0; int total = 0;
                    for int i in [{first}:stride:{last}] {{ visits += 1; total += i; }}''',
                              ('VISITS63', 'TOTAL31'), [str(len(values)), str(sum(values))])

    for first, last, step in ((1, 6, 2), (6, -1, -3)):
        values = list(range(first, last + (1 if step > 0 else -1), step))
        check_program(bra, f'''int first = {first}; int last = {last}; int stride = {step};
            int total = 0;
            for int i in [first:stride:last] {{
                first = 100; last = -100; stride = 0; total += i;
            }}''', ('TOTAL31',), [str(sum(values))])

    check_program(bra, '''int stride = 2; int total = 0;
        for int i in [0:stride + 1:8] {
            if (i == 3) { continue; } if (i == 6) { break; } total += 1;
        }''', ('TOTAL31',), ['1'])
    check_program(bra, '''int limit = 2; int total = 0;
        for int i in [1:limit] { for int j in [0:i:4] { total += j; } }''',
                  ('TOTAL31',), ['16'])
    check_program(bra, '''int stride = 2; int total = 0;
        for int s in {stride, -stride} {
            for int i in [0:s:s * 2] { total += i; continue; }
        }''', ('TOTAL31',), ['0'])
    check_program(bra, '''int stride = 0; int total = 1;
        if (false) { for int i in [0:stride:3] { total += i; } }
        while (false) { for int i in [0:stride:3] {} }
        for int outer in [1:0] { for int i in [0:stride:3] {} }''',
                  ('TOTAL31',), ['1'])

    # Transfers are selected by iteration position, not by whether the stop is reached.
    for first, last, step, skip, stop in (
            (first, last, step, skip, stop)
            for first in (-3, 0, 3) for last in (-3, 0, 4)
            for step in (-3, -2, 2, 3) for skip in (-1, 0, 1) for stop in (-1, 1)):
        total, visits = 0, 0
        for position, value in enumerate(range(first, last + (1 if step > 0 else -1), step)):
            visits += 1
            if position == skip:
                continue
            if position == stop:
                break
            total = total * 7 + value
        check_program(bra, f'''int stride = {step}; int visits = 0; int total = 0; int position = -1;
            for int i in [{first}:stride:{last}] {{
                position += 1; visits += 1;
                if (position == {skip}) {{ continue; }} if (position == {stop}) {{ break; }}
                total = total * 7 + i;
            }}''', ('VISITS63', 'TOTAL31'), [str(visits), str(total)])

    # The same generated loop must recapture values on each entry, including its sign.
    check_program(bra, '''int first = 1; int last = 6; int stride = 2;
        int tick = 0; int total = 0; int seen = 0;
        while (tick < 3) {
            for int i in [first:stride:last] { total += i; seen = i; stride = 0; }
            tick += 1;
            if (tick == 1) { first = 6; last = -1; stride = -3; }
            else { first = -4; last = 2; stride = 2; }
        }''', ('TOTAL31', 'SEEN15', 'TICK15'), ['14', '2', '3'])

    check_program(bra, '''int stride = -2; int total = 0;
        for int i in [4:stride:0] {
            for int s in {2, -2} {
                for int j in [i:s:i + s * 2] {
                    if (j == i + s) { continue; }
                    if (j == i + s * 2) { break; }
                    total += j;
                }
            }
        }''', ('TOTAL31',), ['12'])

    check_program(bra, '''int stride = 2; int total = 0; int tick = 0;
        for int i in [0:stride:4] {
            for int k in {1, 2} { if (k == 1) { continue; } break; }
            tick = 0;
            while (tick < 2) { tick += 1; if (tick == 1) { continue; } break; }
            total += i;
        }''', ('TOTAL31', 'TICK15'), ['6', '2'])

    check_program(bra, '''int i = 2; int total = 0;
        for int i in [i:i:6] { total += i; } total += i;
        total += i % 3;''', ('TOTAL31', 'I1'), ['16', '2'])

    for expression in ('stride / 2', 'stride % 3', 'int(f)', 'uint(f)', 'int(flags[0]) + 1'):
        step = 1 if expression in ('stride / 2',) else 2
        values = list(range(1, 7, step))
        check_program(bra, f'''int stride = 2; float f = 2.5; bit[2] flags = "01"; int total = 0;
            for int i in [1:{expression}:6] {{ total += i; }}''', ('TOTAL31',), [str(sum(values))])

    # Captures and guard scratch must not disturb body temporaries or later uses.
    check_program(bra, '''int stride = 2; int total = 0;
        for int i in [1:stride:5] { total += i % stride; continue; total += 1 / stride; }
        total += stride % 3;''', ('TOTAL31',), ['5'])

    check_program(bra, '''include "stdgates.inc";
        qubit q; bit outcome; int stride = 2; int visits = 0;
        for int i in [1:stride:5] {
            visits += 1; if (i == 3) { x q; }
            outcome = measure q; if (outcome) { break; }
        }''', ('VISITS63',), ['2'])

    check_program(bra, '''int stride = 0; int value = 3; int total = 1;
        if (false) { for int i in [0:value / stride:3] {} }
        while (false) { for int i in [0:value % stride:3] {} }
        for int i in [1:2:stride] { for int j in [0:value / stride:3] {} }
        for int i in {1} { continue; for int j in [0:stride:3] {} }
        for int i in [0:1] { break; for int j in [0:stride:3] {} }''', ('TOTAL31',), ['1'])

    for step in ('stride - 2', 'stride / 3', 'stride % 2', 'int(f)'):
        check_error(bra, f'''int stride = 2; float f = 0.5;
            for int i in [3:{step}:0] {{}}''', r'(evaluated: 0 \= 0)')
    check_error(bra, '''int stride = 2; int tick = 0;
        while (tick < 2) { for int i in [0:stride:3] {} stride = 0; tick += 1; }''',
                r'(evaluated: 0 \= 0)')
    check_error(bra, '''int stride = 2;
        for int s in {stride, 0} { for int i in [0:s:3] {} }''', r'(evaluated: 0 \= 0)')
    for expression in ('value / stride', 'value % stride'):
        check_error(bra, f'''int value = 3; int stride = 0;
            for int i in [3:{expression}:0] {{}}''', 'integer division by zero in LET')

    minimum, maximum = qasm2qcx.QASM2QCXConverter.QCX_INT_MIN, qasm2qcx.QASM2QCXConverter.QCX_INT_MAX
    for first, last, step in (
            (minimum, maximum, maximum), (maximum, minimum, minimum),
            (minimum, -1, maximum), (maximum, -1, minimum),
            (minimum, minimum + 3, 2), (maximum, maximum - 3, -2),
            (maximum - 1, maximum, 2), (minimum + 1, minimum, -2)):
        values = list(range(first, last + (1 if step > 0 else -1), step))
        check_program(bra, f'''int stride = {step}; int visits = 0; int seen = 0;
            for int i in [{first}:stride:{last}] {{ visits += 1; seen = i; continue; }}''',
                      ('VISITS63', 'SEEN15'), [str(len(values)), str(values[-1] if values else 0)])

    # Large strides bound the number of iterations while exercising endpoint signs.
    endpoints = (minimum, minimum + 1, -1, 0, 1, maximum - 1, maximum)
    for first in endpoints:
        for last in endpoints:
            for step in (minimum, -maximum, -(maximum - 1), maximum - 1, maximum):
                values = list(range(first, last + (1 if step > 0 else -1), step))
                check_program(bra, f'''int first = {first}; int last = {last}; int stride = {step};
                    int visits = 0; int seen = 0;
                    for int i in [first:stride:last] {{ visits += 1; seen = i; continue; }}''',
                              ('VISITS63', 'SEEN15'), [str(len(values)), str(values[-1] if values else 0)])

    # Zero is an error even if neither direction would enter the body.
    for first, last in ((0, 3), (3, 0), (0, 0)):
        lines = qasm2qcx.convert(f'''OPENQASM 3.0; int stride = 0;
            for int i in [{first}:stride:{last}] {{}}''')
        lines.append('PRINTLN STRIDE63')
        result = subprocess.run([str(bra)], input='\n'.join(lines) + '\n',
                                text=True, capture_output=True, timeout=10)
        if (result.returncode == 0 or 'assertion failed in ASSERT' not in result.stderr
                or r'(evaluated: 0 \= 0)' not in result.stderr or result.stdout.strip()):
            raise RuntimeError(f'Zero runtime step did not fail explicitly: {result}')


if __name__ == '__main__':
    main()
