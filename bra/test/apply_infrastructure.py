#!/usr/bin/env python3

"""Check APPLY syntax, basic evaluation, and skipped execution.

  python3 bra/test/apply_infrastructure.py --bra /path/to/bra
"""

import argparse
import math
import pathlib
import resource
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    bra = parser.parse_args().bra.resolve()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    def run(program):
        return subprocess.run([str(bra)], input=program, text=True, capture_output=True, timeout=10)

    setup = 'QUBITS 1\nVAR R REAL\nVAR N INT\nLET R := 2.0\n'
    # Known function calls on skipped classical paths must not resolve invalid
    # operands, destination indices, counts, or types.
    for jump in ('JUMP DONE', 'JUMPIF DONE N == 0'):
        result = run(setup + jump + '''
apply missing:n:999 <- exp nonexistent
APPLY R:N:999 <- ARG R
APPLY R R <- EXP 1
APPLY R:N:999 <- ABS nonexistent
APPLY R:N:999 <- SQRT nonexistent
APPLY missing <- COMPLEX nonexistent R:N:999
APPLY R:N:999 <- LOG nonexistent
APPLY R:N:999 <- SIN nonexistent
APPLY R:N:999 <- COS nonexistent
APPLY R:N:999 <- TAN nonexistent
APPLY R:N:999 <- ARCSIN nonexistent
APPLY R:N:999 <- ARCCOS nonexistent
APPLY R:N:999 <- ARCTAN nonexistent
@DONE
PRINTLN R
''')
        if result.returncode != 0 or result.stdout.strip() != '2':
            raise RuntimeError(f'skipped APPLY evaluated operands: {result}')

    for malformed in (
        'APPLY', 'APPLY R', 'APPLY <- EXP R', 'APPLY R <-',
        'APPLY R <- EXP', 'APPLY R := EXP R', 'APPLY R<-EXP R',
        'APPLY R <- EXP <- R',
    ):
        result = run(setup + f'JUMP DONE\n{malformed}\n@DONE\nPRINTLN R\n')
        if result.returncode == 0 or 'APPLY' not in result.stderr or result.stdout.strip():
            raise RuntimeError(f'malformed APPLY was not rejected while parsing: {result}')

    result = run(setup + 'JUMP DONE\nAPPLY R <- BOGUS R\n@DONE\n')
    if result.returncode == 0 or 'unknown function BOGUS' not in result.stderr:
        raise RuntimeError(f'unknown function was not rejected while parsing: {result}')

    result = run(setup + 'apply r <- exp r\nPRINTLN R\n')
    if result.returncode != 0 or not math.isclose(float(result.stdout.strip()), math.exp(2), rel_tol=1e-5):
        raise RuntimeError(f'in-place EXP failed: {result}')
    result = run(setup + 'apply r <- log r\nPRINTLN R\n')
    if result.returncode != 0 or not math.isclose(float(result.stdout.strip()), math.log(2), rel_tol=1e-5):
        raise RuntimeError(f'in-place LOG failed: {result}')
    print('APPLY infrastructure CLI checks passed')


if __name__ == '__main__':
    main()
