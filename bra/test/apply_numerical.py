#!/usr/bin/env python3

"""End-to-end APPLY calls and ARG(EXP(i*theta)) versus native phase gates.

  python3 bra/test/apply_numerical.py --bra /path/to/default-double/bra

Uses raw QCX and independent Python complex arithmetic; no converter or
OpenQASM dependencies. Precision-independent kernel tests are provided by
classical_function_numerical.py.
"""

import argparse
import cmath
import math
import pathlib
import re
import resource
import subprocess


SETUP = '''QUBITS 2
VAR R REAL 3
VAR Z COMPLEX 3
VAR N INT 3
VAR U UINT
VAR OUT REAL
VAR COUT COMPLEX
LET R:0 := 0.25
LET R:1 := -0.5
LET R:2 := 0.75
LET N:0 := 1
LET N:1 := 2
LET N:2 := 0
LET U := :UINT:2
LET Z:0 := :COMPLEX:R:0
LET Z:0 += :I
LET Z:1 := :COMPLEX:R:1
LET Z:1 -= :I
'''


def run(bra, program):
    return subprocess.run([str(bra)], input=program, text=True, capture_output=True, timeout=30)


def check_values(bra, program, expected):
    result = run(bra, program)
    if result.returncode:
        raise RuntimeError(f'QCX failed:\n{program}\n{result.stderr}')
    actual = [float(token) for token in result.stdout.split()]
    if len(actual) != len(expected) or any(
            not math.isclose(a, b, rel_tol=1e-5, abs_tol=2e-6)
            for a, b in zip(actual, expected)):
        raise RuntimeError(f'Value mismatch: {actual} != {expected}\n{program}')


def check_failure(bra, instruction, reason, preparation=''):
    result = run(bra, SETUP + preparation + instruction + '\nPRINTLN :INT:99\n')
    if (not result.returncode or instruction.upper() not in result.stderr
            or reason not in result.stderr or result.stdout.strip()):
        raise RuntimeError(f'Expected stopped APPLY with {reason!r}: {result}')


def amplitudes(bra, program):
    result = run(bra, program + 'DO AMPLITUDES\n')
    if result.returncode:
        raise RuntimeError(f'Phase program failed:\n{program}\n{result.stderr}')
    values = {int(index, 2): complex(float(real), float(imaginary))
              for index, real, imaginary in re.findall(
                  r'^([01]+) => (\S+) \+ (\S+) i$', result.stdout, re.MULTILINE)}
    if set(values) != set(range(4)):
        raise RuntimeError(f'Missing phase amplitudes: {result.stdout}')
    return [values[index] for index in range(4)]


def check_phase(bra, angle, gate, active, fused):
    setup = f'''QUBITS 2
VAR THETA REAL 2
VAR BETA REAL 2
VAR COEFFICIENT COMPLEX 2
VAR INDEX INT
LET INDEX := 1
LET THETA:INDEX := {angle!r}
H 0
H 1
'''
    # Evaluate APPLY inside the fusion block, not just before it. The source
    # parameter and index must survive both in-place and distinct-result calls.
    begin, end = ('BEGIN FUSION\n', 'END FUSION\n') if fused else ('', '')
    conversion = '''LET COEFFICIENT:INDEX := :I
LET COEFFICIENT:INDEX *= :COMPLEX:THETA:INDEX
APPLY COEFFICIENT:INDEX <- EXP COEFFICIENT:INDEX
APPLY BETA:INDEX <- ARG COEFFICIENT:INDEX
'''
    assertions = f'ASSERT INDEX == 1\nASSERT THETA:INDEX == {angle!r}\n'
    native = amplitudes(bra, setup + begin + gate.format(phase='THETA:INDEX') + '\n' + end)
    wrapped = amplitudes(bra, setup + begin + conversion + assertions
                         + gate.format(phase='BETA:INDEX') + '\n' + end)
    coefficient = cmath.exp(complex(0, angle))
    expected = [coefficient / 2 if active(basis) else 0.5 + 0j for basis in range(4)]
    for label, values in (('native', native), ('wrapped', wrapped)):
        if any(not cmath.isclose(value, reference, rel_tol=1e-5, abs_tol=2e-6)
               for value, reference in zip(values, expected)):
            raise RuntimeError(f'{label} phase mismatch: {angle=}, {gate=}, {fused=}\n'
                               f'{values} != {expected}')
    if any(not cmath.isclose(a, b, rel_tol=1e-5, abs_tol=2e-6) for a, b in zip(native, wrapped)):
        raise RuntimeError(f'Native/wrapped phase mismatch: {angle=}, {gate=}, {fused=}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    bra = parser.parse_args().bra.resolve()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    calls = 0
    for source, value in (('R:0', 0.25), ('R:N:N', 0.75), ('1.0', 1), ('1.', 1),
                          ('1E+0', 1), ('.5', 0.5), ('-0.0', -0.0), (':REAL:N:1', 2),
                          (':REAL:U', 2), (':REAL:Z:0', 0.25), (':IMAG:Z:0', 1),
                          (':REAL::INT:1.9', 1), (':REAL::UINT:2', 2)):
        check_values(bra, SETUP + f'APPLY OUT <- EXP {source}\nPRINTLN OUT R:0 R:1 R:2 N:0 N:1 N:2 U\n',
                     [math.exp(value), 0.25, -0.5, 0.75, 1, 2, 0, 2])
        calls += 1
    for source, value in (('Z:0', complex(0.25, 1)), ('Z:1', complex(-0.5, -1)),
                          ('Z:N:N:N:0', complex(0.25, 1)), (':COMPLEX:R:1', complex(-0.5)),
                          (':I', 1j), (':MINUS_I', -1j)):
        expected = cmath.exp(value)
        check_values(bra, SETUP + f'APPLY COUT <- EXP {source}\nPRINTLN :REAL:COUT :IMAG:COUT\n',
                     [expected.real, expected.imag])
        check_values(bra, SETUP + f'LET COUT := {source}\nAPPLY COUT <- EXP COUT\nPRINTLN :REAL:COUT :IMAG:COUT\n',
                     [expected.real, expected.imag])
        check_values(bra, SETUP + f'APPLY OUT <- ARG {source}\nPRINTLN OUT\n', [cmath.phase(value)])
        calls += 3
    check_values(bra, SETUP + 'APPLY R:N:N <- EXP R:N:N\nPRINTLN R:0 R:1 R:2 N:0 N:1 N:2\n',
                 [0.25, -0.5, math.exp(0.75), 1, 2, 0])
    check_values(bra, SETUP + 'APPLY OUT <- ARG :COMPLEX:-1\nPRINTLN OUT\n', [math.pi])
    check_values(bra, SETUP + 'APPLY OUT <- EXP -1000000.0\nAPPLY COUT <- EXP :COMPLEX:-1000000.0\n'
                 'PRINTLN OUT :REAL:COUT :IMAG:COUT\n', [0, 0, 0])
    calls += 3

    failures = [
        ('APPLY OUT <- EXP 1', 'unsupported operand signature', ''),
        ('APPLY OUT <- EXP U', 'unsupported operand signature', ''),
        ('APPLY COUT <- EXP R:0', 'wrong destination type', ''),
        ('APPLY OUT <- ARG R:0', 'unsupported operand signature', ''),
        ('APPLY OUT OUT <- EXP 1.0', 'wrong destination count', ''),
        ('APPLY OUT <- EXP 1.0 2.0', 'wrong operand count', ''),
        ('APPLY R:-1 <- EXP 1.0', 'negative index', ''),
        ('APPLY OUT <- EXP R:N:999', 'index out of range', ''),
        ('APPLY OUT <- EXP :REAL:R:N:999', 'index out of range', ''),
        ('APPLY OUT <- EXP MISSING', 'unknown variable', ''),
        ('APPLY :REAL:1 <- EXP 1.0', 'destination must be a mutable variable', ''),
        ('APPLY R:U <- EXP 1.0', 'index requires INT', ''),
        ('APPLY OUT <- EXP :REAL:R::INT:1e100', 'REAL to INT conversion out of range', ''),
        ('APPLY OUT <- ARG :COMPLEX:0', 'zero complex input', ''),
        ('APPLY OUT <- EXP 1000000.0', 'overflow/nonfinite result', ''),
        ('APPLY COUT <- EXP :COMPLEX:1000000.0', 'overflow/nonfinite result', ''),
    ]
    for numerator in ('1.0', '-1.0', '0.0'):
        preparation = f'LET OUT := {numerator}\nLET OUT /= 0.0\nLET COUT := :COMPLEX:OUT\n'
        failures.extend((
            ('APPLY R:0 <- EXP OUT', 'nonfinite input', preparation),
            ('APPLY Z:0 <- EXP COUT', 'nonfinite input', preparation),
            ('APPLY R:0 <- ARG COUT', 'nonfinite input', preparation),
        ))
    for instruction, reason, preparation in failures:
        check_failure(bra, instruction, reason, preparation)
    check_values(bra, SETUP + 'JUMP DONE\n' + '\n'.join(item[0] for item in failures)
                 + '\n@DONE\nPRINTLN R:0 R:1 R:2\n', [0.25, -0.5, 0.75])

    angles = [0.0, -0.0, 0.7, -0.7, math.pi, -math.pi, 3 * math.pi / 2,
              math.nextafter(math.pi, 0), math.nextafter(math.pi, math.inf),
              -math.nextafter(math.pi, 0), -math.nextafter(math.pi, math.inf),
              1e20, -1e20, 1e100, -1e100, 1e308, -1e308]
    gates = [('U1 0 {phase}', lambda basis: basis & 1),
             ('CU1 1 0 {phase}', lambda basis: (basis & 3) == 3),
             ('CU1 1#0 0 {phase}', lambda basis: (basis & 3) == 1),
             ('PHASE {phase}', lambda basis: True)]
    phases = 0
    for angle in angles:
        for gate, active in gates:
            for fused in (False, True):
                check_phase(bra, angle, gate, active, fused)
                phases += 1
    print(f'APPLY end-to-end checks passed: {calls} value cases, {len(failures)} failures, '
          f'{phases} native/wrapped phase cases (including fusion)')


if __name__ == '__main__':
    main()
