#!/usr/bin/env python3

"""Execute converted one-dimensional integer-array declarations with bra.

Example:
  python3 bra/test/qasm2qcx_integer_array_numerical.py --bra bra/bin/bra
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
