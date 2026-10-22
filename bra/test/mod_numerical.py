#!/usr/bin/env python3

"""Check native real MOD with a default-double bra executable.

  python3 bra/test/mod_numerical.py --bra /path/to/bra

All-precision kernel tests are in classical_function_numerical.py.
"""

import argparse
import math
import pathlib
import resource
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    bra = parser.parse_args().bra.resolve()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    programs = 0

    def run(source):
        nonlocal programs
        programs += 1
        return subprocess.run([str(bra)], input=source, text=True, capture_output=True, timeout=30)

    setup = 'QUBITS 1\nVAR R REAL 2\nVAR I INT\nLET I := 1\nLET R:0 := 7.5\nLET R:1 := 3.0\n'
    for dividend in (-7.5, -6.0, -0.0, 0.0, 6.0, 7.5, 1e308, -1e308, 1e-300):
        for divisor in (-3.0, -1.0, 1.0, 3.0, 1e308, 1e-300):
            expected = math.fmod(dividend, divisor)
            source = (setup + f'LET R:0 := {dividend!r}\nLET R:1 := {divisor!r}\n'
                      'apply r:0 <- mod r:0 r:i\nPRINTLN R:0 R:1\n')
            result = run(source)
            actual = [float(value) for value in result.stdout.split()]
            if (result.returncode or len(actual) != 2
                    or not math.isclose(actual[0], expected, rel_tol=1e-5, abs_tol=0.0)
                    or math.copysign(1, actual[0]) != math.copysign(1, expected)
                    or actual[1] != divisor):
                raise RuntimeError(f'MOD value/sign mismatch: {source}\n{result}')

    for instruction, expected in (
            ('APPLY R:I <- MOD R:0 R:I', '7.5 1.5'),
            ('APPLY R:0 <- MOD R:0 R:0', '0 3'),
            ('APPLY R:0 <- MOD 7.5 -3.0', '1.5 3')):
        result = run(setup + instruction + '\nPRINTLN R:0 R:1\n')
        if result.returncode or result.stdout.strip() != expected:
            raise RuntimeError(f'MOD alias/literal mismatch: {instruction}\n{result}')

    failures = (
        ('APPLY R:0 <- MOD R:0 0.0', 'MOD domain error: zero divisor'),
        ('APPLY R:0 <- MOD R:0 -0.0', 'MOD domain error: zero divisor'),
        ('APPLY R:0 <- MOD R:0 I', 'unsupported operand signature'),
        ('APPLY R:0 <- MOD :I R:1', 'unsupported operand signature'),
        ('APPLY I <- MOD R:0 R:1', 'wrong destination type'),
        ('APPLY R:0 <- MOD R:0', 'wrong operand count'),
        ('APPLY R:0 <- MOD R:0 R:1 R:0', 'wrong operand count'),
        ('APPLY R:0 <- MOD R:0 R:I:999', 'index out of range'),
        ('APPLY R:99 <- MOD R:0 R:1', 'index out of range'),
    )
    for instruction, diagnostic in failures:
        result = run(setup + instruction + '\nPRINTLN R:0\n')
        if (not result.returncode or result.stdout or instruction not in result.stderr
                or diagnostic not in result.stderr):
            raise RuntimeError(f'MOD expected {diagnostic}: {result}')
        for jump in ('JUMP DONE', 'JUMPIF DONE I == 1'):
            result = run(setup + jump + '\n' + instruction + '\n@DONE\nPRINTLN R:0\n')
            if result.returncode or result.stdout.strip() != '7.5':
                raise RuntimeError(f'Skipped MOD evaluated its operands: {result}')

    for numerator in ('0.0', '1.0', '-1.0'):
        for operands in ('BAD R:1', 'R:0 BAD'):
            instruction = 'APPLY R:0 <- MOD ' + operands
            result = run(setup + f'VAR BAD REAL\nLET BAD := {numerator}\nLET BAD /= 0.0\n'
                         + instruction + '\nPRINTLN R:0\n')
            if (not result.returncode or result.stdout or instruction not in result.stderr
                    or 'MOD domain error: nonfinite input' not in result.stderr):
                raise RuntimeError(f'MOD accepted nonfinite input: {result}')

    for fused in (False, True):
        begin, end = ('BEGIN FUSION\n', 'END FUSION\n') if fused else ('', '')
        result = run(setup + begin + 'APPLY R:I <- MOD :REAL:7 :REAL:3\nH 0\n'
                     + end + 'PRINTLN R:0 R:1 I\n')
        if result.returncode or result.stdout.strip() != '7.5 1 1':
            raise RuntimeError(f'MOD cast/fusion failed: {result}')
    print(f'Native real MOD programs checked: {programs}')


if __name__ == '__main__':
    main()
