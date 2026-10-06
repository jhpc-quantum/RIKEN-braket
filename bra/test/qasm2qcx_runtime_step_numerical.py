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
