#!/usr/bin/env python3

"""Execute converted one-dimensional integer-array declarations and loops with bra.

Example:
  python3 bra/test/qasm2qcx_integer_array_numerical.py --bra bra/bin/bra
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
        raise RuntimeError(f'Expected array runtime error {diagnostic!r}: {result}')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    bra = parser.parse_args().bra
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    for source, expected in (
            ('array[int, 3] values = {1, 3, 5};', ['1', '3', '5']),
            ('int seed = 3; float f = 2.5; '
             'array[int[8], 3] values = {seed, seed + 1, int(f)}; seed = 9;', ['3', '4', '2']),
            ('const uint N = 3; int seed = -4; '
             'array[int, N] values = {seed % 3, -seed, seed / 3};', ['-1', '4', '-1']),
            ('bit[2] flags = "01"; '
             'array[int, 3] values = {int(flags[0]), int(flags[1]), int(true)};', ['1', '0', '1']),
            ('int seed = 2; bool flag = true; '
             'array[int, 3] values = {int(flag), seed * seed + 1, seed % 3}; '
             'int after = seed % 3;', ['1', '5', '2'])):
        check_program(bra, source, ('VALUES63:0', 'VALUES63:1', 'VALUES63:2'), expected)

    check_program(bra, 'array[int, 1] values = {-7};', ('VALUES63:0',), ['-7'])

    for outer in ('for int N in values', 'for int N in {0, 3}'):
        check_program(bra, 'const int N = 2; array[int, N] values = {1, 2}; int total = 0; '
                      + outer + ' { for int j in values { total += j; } } '
                      'for int j in values { total += j; }', ('TOTAL31',), ['9'])

    check_program(bra, '''array[int, 3] values = {1, 3, 5}; int total = 0;
        for int value in values { total += value; values[1] = 10; }''',
                  ('TOTAL31', 'VALUES63:1'), ['16', '10'])
    check_program(bra, '''array[int, 3] values = {1, 3, 5}; int total = 0;
        for int value in values { values[0] = 100; total += value; }''',
                  ('TOTAL31', 'VALUES63:0'), ['9', '100'])
    check_program(bra, '''array[int, 3] values = {2, 2, -1}; int total = 0;
        for int value in values { total = total * 10 + value; }''', ('TOTAL31',), ['219'])
    check_program(bra, '''array[int, 3] values = {1, 3, 5}; int total = 0;
        for int value in values {
            if (value == 1) { values[1] = 10; continue; }
            if (value == 10) { break; } total += value;
        }''', ('TOTAL31',), ['0'])
    check_program(bra, '''array[int, 2] values = {1, 2}; int total = 0; int tick = 0;
        while (tick < 3) {
            for int value in values { total += value; }
            values[0] += 1; values[1] += 1; tick += 1;
        }''', ('TOTAL31', 'VALUES63:0', 'VALUES63:1'), ['15', '4', '5'])
    check_program(bra, '''array[int, 2] values = {1, 2}; int total = 0;
        for int i in values {
            for int j in values { total += j; values[1] = 3; }
            total += i;
        }''', ('TOTAL31',), ['12'])
    check_program(bra, '''array[int, 1] values = {-7}; int total = 0;
        for int i in values { total += i; continue; }
        total += values[-1];''', ('TOTAL31',), ['-14'])
    check_program(bra, '''array[int, 3] values = {-7, 3, 5}; const int LAST = -1;
        values[-3] %= 3; values[1] *= 2; values[LAST] += values[0];
        values[1] -= 1; values[1] /= 2; int total = values[0] + values[1] + values[2];''',
                  ('VALUES63:0', 'VALUES63:1', 'VALUES63:2', 'TOTAL31'), ['-1', '2', '4', '5'])
    check_program(bra, '''array[int, 3] values = {0, 2, 4}; int total = 0;
        for int i in [values[0]:values[1]:values[2]] { total += i; }
        for int j in {values[1], values[2]} { total += j; }''', ('TOTAL31',), ['12'])
    check_program(bra, '''array[int, 3] values = {1, 3, 5}; int total = 0;
        for int values in values { total += values; } total += values[0];''', ('TOTAL31',), ['10'])
    check_program(bra, '''array[int, 3] values = {1, 3, 5}; int total = 0;
        for int k in [0:2] { total += values[k]; }
        if (values[-1] == 5) { total += 1; }''', ('TOTAL31',), ['10'])

    # Model live element reads explicitly; Python's value iterator is not used
    # so mutation of a current/future element is independent of the implementation.
    for initial, target, replacement, skip, stop in (
            (initial, target, replacement, skip, stop)
            for initial in ((1, 3, 5), (2, 2, -1), (0, -2, 1), (-1, 0, -1))
            for target in range(3) for replacement in (-3, 0, 7)
            for skip in (-1, 0, 1) for stop in (-1, 1)):
        working = list(initial)
        total, visits = 0, 0
        for position in range(len(working)):
            value = working[position]
            visits += 1
            working[target] = replacement
            if position == skip:
                continue
            if position == stop:
                break
            total = total * 7 + value
        literal = ', '.join(map(str, initial))
        check_program(bra, f'''array[int, 3] values = {{{literal}}};
            int total = 0; int visits = 0; int position = -1;
            for int value in values {{
                position += 1; visits += 1; values[{target}] = {replacement};
                if (position == {skip}) {{ continue; }} if (position == {stop}) {{ break; }}
                total = total * 7 + value;
            }}''', ('TOTAL31', 'VISITS63', 'VALUES63:0', 'VALUES63:1', 'VALUES63:2'),
                      [str(total), str(visits), *map(str, working)])

    check_program(bra, '''int seed = 2; array[int, 3] values = {seed, seed + 1, seed - 1};
        int total = 0; int tick = 0;
        while (tick < 3) {
            for int value in values {
                for int s in {value, -value} {
                    for int j in [0:1] { if (j == 0) { continue; } break; }
                    total += s; if (s < 0) { break; }
                }
                total += value;
            }
            values[1] += 1; tick += 1;
        }''', ('TOTAL31', 'TICK15'), ['21', '3'])

    # Array elements in discrete sets are snapshots; direct array iteration is live.
    check_program(bra, '''array[int, 2] values = {1, 2}; int total = 0;
        for int value in {values[0], values[1]} { total += value; values[1] = 10; }
        for int value in values { total += value; }''', ('TOTAL31',), ['14'])

    check_program(bra, '''array[int, 3] values = {1, 0, 3}; int total = 0;
        for int value in values {
            if (value == 0) { continue; }
            total += values[0] / value;
        }''', ('TOTAL31',), ['1'])
    check_program(bra, '''array[int, 2] values = {1, 0}; int total = 1;
        if (false) { for int value in values { values[0] /= values[1]; } }
        while (false) { for int value in values { values[0] %= values[1]; } }
        for int value in values { break; values[0] /= values[1]; }
        for int value in values { continue; values[0] %= values[1]; }
        for int i in [1:0] { for int value in values { values[0] /= values[1]; } }
        total += values[0];''', ('TOTAL31',), ['2'])

    check_program(bra, '''include "stdgates.inc"; array[int, 3] values = {1, 3, 5};
        qubit q; bit outcome; int visits = 0;
        for int value in values {
            visits += 1; if (value == 3) { x q; }
            outcome = measure q; if (outcome) { break; }
        }''', ('VISITS63',), ['2'])
    check_program(bra, '''include "stdgates.inc"; array[int, 3] values = {0, 1, 2};
        qubit q; bit outcome;
        for int value in values { rx(value * 3.141592653589793) q; }
        outcome = measure q; int total = int(outcome);''', ('TOTAL31',), ['1'])

    minimum, maximum = qasm2qcx.QASM2QCXConverter.QCX_INT_MIN, qasm2qcx.QASM2QCXConverter.QCX_INT_MAX
    check_program(bra, f'''array[int, 3] values = {{{minimum}, {maximum}, {minimum}}};
        int seen = 0; int visits = 0;
        for int value in values {{ seen = value; visits += 1; continue; }}''',
                  ('SEEN15', 'VISITS63'), [str(minimum), '3'])
    for operation in ('/=', '%='):
        check_error(bra, f'''array[int, 2] values = {{1, 0}};
            for int value in values {{ values[0] {operation} values[1]; }}''',
                    'integer division by zero in LET')

    # Declarations allocate storage but initializer calculations must execute.
    lines = qasm2qcx.convert('''OPENQASM 3.0; int seed = 3; int zero = 0;
        array[int, 2] values = {seed, seed / zero};''')
    lines.append('PRINTLN VALUES63:0')
    result = subprocess.run([str(bra)], input='\n'.join(lines) + '\n',
                            text=True, capture_output=True, timeout=10)
    if (result.returncode == 0 or 'integer division by zero in LET' not in result.stderr
            or result.stdout.strip()):
        raise RuntimeError(f'Array initializer did not execute its division: {result}')


if __name__ == '__main__':
    main()
