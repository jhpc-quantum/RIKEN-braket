#!/usr/bin/env python3

"""Check APPLY syntax, basic evaluation, and skipped execution.

  python3 bra/test/apply_infrastructure.py --bra /path/to/bra
"""

import argparse
import math
import pathlib
import re
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
APPLY R:N:999 <- CEILING nonexistent
APPLY R:N:999 <- FLOOR nonexistent
APPLY R:N:999 <- MOD nonexistent R:N:999
APPLY R:N:999 <- ANGLEBITS2PHASE nonexistent R:N:999 INV POW 0.5
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

    # Literal descriptors are instruction metadata: reject malformed chains
    # while loading even if the instruction will be skipped.
    for suffix in ('INV extra', 'POW', 'POW N', 'POW 0 POW NaN', 'POW 1e9999',
                   'POW ' + '0' * 4097, 'INV ' * 1025):
        instruction = 'APPLY R <- ANGLEBITS2PHASE missing missing ' + suffix
        result = run(setup + f'JUMP DONE\n{instruction}\n@DONE\n')
        if result.returncode == 0 or 'APPLY R <- ANGLEBITS2PHASE' not in result.stderr:
            raise RuntimeError(f'invalid skipped descriptor accepted: {result}')
    for instruction in ('APPLY R <- ANGLEBITS2PHASE missing',
                        'APPLY R R <- ANGLEBITS2PHASE missing missing'):
        result = run(setup + f'JUMP DONE\n{instruction}\n@DONE\n')
        if result.returncode == 0 or instruction.upper() not in result.stderr:
            raise RuntimeError(f'invalid ANGLEBITS2PHASE shape accepted: {result}')

    result = run(setup + '''VAR U UINT
LET U := 4
apply r <- anglebits2phase u 3 inv pow 0.5
PRINTLN R
APPLY R <- ANGLEBITS2PHASE :UINT:4 :INT:3 POW 0.5 INV
PRINTLN R
APPLY R <- ANGLEBITS2PHASE U 3 POW 2 POW 0.5
PRINTLN R
''')
    if result.returncode or len(result.stdout.splitlines()) != 3:
        raise RuntimeError(f'ANGLEBITS2PHASE CLI failed: {result}')
    for actual, expected in zip(map(float, result.stdout.splitlines()), (math.pi / 2, -math.pi / 2, 0)):
        if not math.isclose(actual, expected, rel_tol=1e-5, abs_tol=1e-7):
            raise RuntimeError(f'ANGLEBITS2PHASE CLI phase: {actual} != {expected}')

    document = (pathlib.Path(__file__).parents[2] / 'docs' / 'bra.md').read_text()
    examples = re.findall(r'<!-- anglebits2phase-example -->\s*```text\n(.*?)\n```', document, flags=re.DOTALL)
    if len(examples) != 1:
        raise RuntimeError('expected one documented ANGLEBITS2PHASE example')
    result = run(examples[0] + '\n')
    if result.returncode or len(result.stdout.splitlines()) != 3:
        raise RuntimeError(f'documented ANGLEBITS2PHASE example failed: {result}')
    for actual, expected in zip(map(float, result.stdout.splitlines()), (math.pi / 2, -math.pi / 2, 0)):
        if not math.isclose(actual, expected, rel_tol=1e-5, abs_tol=1e-7):
            raise RuntimeError(f'documented ANGLEBITS2PHASE phase: {actual} != {expected}')

    result = run(setup + 'apply r <- exp r\nPRINTLN R\n')
    if result.returncode != 0 or not math.isclose(float(result.stdout.strip()), math.exp(2), rel_tol=1e-5):
        raise RuntimeError(f'in-place EXP failed: {result}')
    result = run(setup + 'apply r <- log r\nPRINTLN R\n')
    if result.returncode != 0 or not math.isclose(float(result.stdout.strip()), math.log(2), rel_tol=1e-5):
        raise RuntimeError(f'in-place LOG failed: {result}')
    print('APPLY infrastructure CLI checks passed')


if __name__ == '__main__':
    main()
