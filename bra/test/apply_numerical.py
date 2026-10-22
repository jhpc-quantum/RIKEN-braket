#!/usr/bin/env python3

"""End-to-end APPLY calls, complex magnitudes, and phase-gate equivalence.

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
    for name, function in (('SIN', math.sin), ('COS', math.cos), ('TAN', math.tan)):
        for source, value in (('R:0', 0.25), ('R:N:N', 0.75), (':REAL:N:1', 2),
                              (':REAL:U', 2), (':REAL:Z:0', 0.25), (':IMAG:Z:0', 1),
                              ('1.0', 1), ('-0.5', -0.5), ('1e20', 1e20), ('-1e20', -1e20),
                              ('1e308', 1e308), ('-1e308', -1e308),
                              (repr(math.nextafter(math.pi / 2, 0)), math.nextafter(math.pi / 2, 0)),
                              (repr(math.nextafter(math.pi / 2, math.inf)), math.nextafter(math.pi / 2, math.inf))):
            check_values(bra, SETUP + f'APPLY OUT <- {name} {source}\nPRINTLN OUT R:0 R:1 R:2 N:1 U\n',
                         [function(value), 0.25, -0.5, 0.75, 2, 2])
            calls += 1
        check_values(bra, SETUP + f'apply r:n:n <- {name.lower()} r:n:n\nPRINTLN R:0 R:1 R:2\n',
                     [0.25, -0.5, function(0.75)])
        calls += 1
        for value in (0.0, -0.0):
            result = run(bra, SETUP + f'APPLY OUT <- {name} {value!r}\nPRINTLN OUT\n')
            if result.returncode:
                raise RuntimeError(f'{name} failed for zero: {result}')
            actual = float(result.stdout)
            expected = function(value)
            if actual != expected or math.copysign(1, actual) != math.copysign(1, expected):
                raise RuntimeError(f'{name} lost signed-zero behavior: {actual!r} != {expected!r}')
            calls += 1
    for source, value in (('R:0', 0.25), ('R:N:N', 0.75), ('1.0', 1), (':REAL:N:1', 2),
                          (':REAL:U', 2), (':REAL:Z:0', 0.25), (':IMAG:Z:0', 1),
                          ('1e308', 1e308), ('1e-300', 1e-300), ('5e-324', 5e-324),
                          ('1e-310', 1e-310), (repr(math.nextafter(1.0, 0.0)), math.nextafter(1.0, 0.0)),
                          (repr(math.nextafter(1.0, 2.0)), math.nextafter(1.0, 2.0))):
        check_values(bra, SETUP + f'APPLY OUT <- LOG {source}\nPRINTLN OUT R:0 R:1 R:2 N:1 U\n',
                     [math.log(value), 0.25, -0.5, 0.75, 2, 2])
        calls += 1
    check_values(bra, SETUP + 'apply r:n:n <- log r:n:n\nPRINTLN R:0 R:1 R:2\n',
                 [0.25, -0.5, math.log(0.75)])
    calls += 1
    for re_part in (0.0, -0.0, 3.0, -3.0):
        for im_part in (0.0, -0.0, 4.0, -4.0):
            result = run(bra, SETUP + f'APPLY COUT <- COMPLEX {re_part!r} {im_part!r}\n'
                         'PRINTLN :REAL:COUT :IMAG:COUT\n')
            if result.returncode:
                raise RuntimeError(f'COMPLEX construction failed: {result}')
            actual = list(map(float, result.stdout.split()))
            if len(actual) != 2 or any(a != e or math.copysign(1, a) != math.copysign(1, e)
                                      for a, e in zip(actual, (re_part, im_part))):
                raise RuntimeError(f'COMPLEX changed components: {actual} != {(re_part, im_part)}')
            calls += 1
    check_values(bra, SETUP + 'LET COUT := :I\nLET COUT += :COMPLEX:2.0\n'
                 'apply cout <- complex :imag:cout :real:cout\nPRINTLN :REAL:COUT :IMAG:COUT\n', [1, 2])
    calls += 1
    for source, value in (('R:0', 0.25), ('R:N:N', 0.75), ('4.0', 4), ('-0.0', -0.0),
                          (':REAL:N:1', 2), (':REAL:U', 2), ('1e308', 1e308), ('1e-300', 1e-300)):
        check_values(bra, SETUP + f'APPLY R:N:N <- SQRT {source}\nPRINTLN R:2 R:0 R:1 N:1 U\n',
                     [math.sqrt(value), 0.25, -0.5, 2, 2])
        calls += 1
    for source, value in (('Z:0', complex(0.25, 1)), ('Z:1', complex(-0.5, -1)),
                          ('Z:N:N:N:0', complex(0.25, 1)), (':COMPLEX:-4.0', -4+0j),
                          (':I', 1j), (':MINUS_I', -1j), (':COMPLEX:1e308', 1e308+0j)):
        expected = cmath.sqrt(value)
        check_values(bra, SETUP + f'APPLY COUT <- SQRT {source}\nPRINTLN :REAL:COUT :IMAG:COUT\n',
                     [expected.real, expected.imag])
        check_values(bra, SETUP + f'LET COUT := {source}\nAPPLY COUT <- SQRT COUT\nPRINTLN :REAL:COUT :IMAG:COUT\n',
                     [expected.real, expected.imag])
        calls += 2
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
        check_values(bra, SETUP + f'apply r:n:n <- abs {source}\nPRINTLN R:2\n', [abs(value)])
        calls += 4
    for re_part, im_part, scale, expected in ((3, 4, 1, 5), (-3, -4, 1, 5),
            (1e200, -1e200, 1e200, math.sqrt(2)),
            (1e-200, -1e-200, 1e-200, math.sqrt(2)),
            (1e308, 1e308, 1e308, math.sqrt(2)), (0, 0, 1, 0)):
        check_values(bra, SETUP + f'LET COUT := :I\nLET COUT *= :COMPLEX:{im_part!r}\n'
                     f'LET COUT += :COMPLEX:{re_part!r}\nAPPLY OUT <- ABS COUT\n'
                     f'LET OUT /= {scale!r}\nPRINTLN OUT :REAL:COUT :IMAG:COUT\n',
                     [expected, re_part, im_part])
        calls += 1
    check_values(bra, SETUP + 'APPLY OUT <- ABS :COMPLEX:-3\nPRINTLN OUT\n', [3])
    check_values(bra, SETUP + 'APPLY R:1 <- ABS :COMPLEX:R:1\nPRINTLN R:1\n', [0.5])
    calls += 2
    for fused in (False, True):
        begin, end = ('BEGIN FUSION\n', 'END FUSION\n') if fused else ('', '')
        program = (SETUP + 'LET COUT := :I\nLET COUT *= :COMPLEX:0.4\n'
                   'LET COUT += :COMPLEX:0.3\nH 0\nH 1\n' + begin
                   + 'APPLY OUT <- ABS COUT\nU1 0 OUT\n'
                   'APPLY OUT <- ABS :COMPLEX:-0.7\nU1 1 OUT\n' + end)
        expected = [cmath.exp(1j * (0.5 * bool(index & 1) + 0.7 * bool(index & 2))) / 2
                    for index in range(4)]
        actual = amplitudes(bra, program)
        if any(not cmath.isclose(a, e, rel_tol=1e-5, abs_tol=2e-6) for a, e in zip(actual, expected)):
            raise RuntimeError(f'ABS gate parameter mismatch: {fused=}, {actual=}, {expected=}')
        calls += 1
        program = (SETUP + 'H 0\nH 1\n' + begin
                   + 'APPLY OUT <- SQRT 0.25\nU1 0 OUT\n'
                   'APPLY COUT <- SQRT :COMPLEX:-4.0\nLET OUT := :IMAG:COUT\nU1 1 OUT\n' + end)
        expected = [cmath.exp(1j * (0.5 * bool(index & 1) + 2 * bool(index & 2))) / 2
                    for index in range(4)]
        actual = amplitudes(bra, program)
        if any(not cmath.isclose(a, e, rel_tol=1e-5, abs_tol=2e-6) for a, e in zip(actual, expected)):
            raise RuntimeError(f'SQRT gate parameter mismatch: {fused=}, {actual=}, {expected=}')
        calls += 1
        program = (SETUP + 'H 0\nH 1\n' + begin
                   + 'APPLY OUT <- LOG 2.0\nU1 0 OUT\nAPPLY OUT <- LOG 0.5\nU1 1 OUT\n' + end)
        expected = [cmath.exp(1j * math.log(2) * (bool(index & 1) - bool(index & 2))) / 2
                    for index in range(4)]
        actual = amplitudes(bra, program)
        if any(not cmath.isclose(a, e, rel_tol=1e-5, abs_tol=2e-6) for a, e in zip(actual, expected)):
            raise RuntimeError(f'LOG gate parameter mismatch: {fused=}, {actual=}, {expected=}')
        calls += 1
        for name, function in (('SIN', math.sin), ('COS', math.cos), ('TAN', math.tan)):
            program = (SETUP + 'H 0\nH 1\n' + begin
                       + f'APPLY OUT <- {name} 0.5\nU1 0 OUT\n' + end)
            expected = [cmath.exp(1j * function(0.5) * bool(index & 1)) / 2 for index in range(4)]
            actual = amplitudes(bra, program)
            if any(not cmath.isclose(a, e, rel_tol=1e-5, abs_tol=2e-6) for a, e in zip(actual, expected)):
                raise RuntimeError(f'{name} gate parameter mismatch: {fused=}, {actual=}, {expected=}')
            calls += 1
    check_values(bra, SETUP + 'APPLY R:N:N <- EXP R:N:N\nPRINTLN R:0 R:1 R:2 N:0 N:1 N:2\n',
                 [0.25, -0.5, math.exp(0.75), 1, 2, 0])
    check_values(bra, SETUP + 'APPLY OUT <- ARG :COMPLEX:-1\nPRINTLN OUT\n', [math.pi])
    check_values(bra, SETUP + 'APPLY OUT <- EXP -1000000.0\nAPPLY COUT <- EXP :COMPLEX:-1000000.0\n'
                 'PRINTLN OUT :REAL:COUT :IMAG:COUT\n', [0, 0, 0])
    calls += 3

    failures = [
        ('APPLY OUT <- LOG -1.0', 'nonpositive real input', ''),
        ('APPLY OUT <- LOG 0.0', 'nonpositive real input', ''),
        ('APPLY OUT <- LOG -0.0', 'nonpositive real input', ''),
        ('APPLY OUT <- LOG -5e-324', 'nonpositive real input', ''),
        ('APPLY OUT <- LOG 1', 'unsupported operand signature', ''),
        ('APPLY OUT <- LOG U', 'unsupported operand signature', ''),
        ('APPLY OUT <- LOG Z:0', 'unsupported operand signature', ''),
        ('APPLY COUT <- LOG R:0', 'wrong destination type', ''),
        ('APPLY OUT OUT <- LOG R:0', 'wrong destination count', ''),
        ('APPLY OUT <- LOG R:0 R:1', 'wrong operand count', ''),
        ('APPLY R:999 <- LOG R:0', 'index out of range', ''),
        ('APPLY OUT <- LOG R:999', 'index out of range', ''),
        ('APPLY OUT <- COMPLEX R:0 R:1', 'wrong destination type', ''),
        ('APPLY COUT <- COMPLEX N:0 R:0', 'unsupported operand signature', ''),
        ('APPLY COUT <- COMPLEX R:0 U', 'unsupported operand signature', ''),
        ('APPLY COUT <- COMPLEX R:0', 'wrong operand count', ''),
        ('APPLY COUT <- COMPLEX R:0 R:1 R:2', 'wrong operand count', ''),
        ('APPLY COUT <- COMPLEX R:0 R:999', 'index out of range', ''),
        ('APPLY OUT <- SQRT -1.0', 'negative real input', ''),
        ('APPLY OUT <- SQRT 1', 'unsupported operand signature', ''),
        ('APPLY OUT <- SQRT U', 'unsupported operand signature', ''),
        ('APPLY COUT <- SQRT R:0', 'wrong destination type', ''),
        ('APPLY OUT <- SQRT Z:0', 'wrong destination type', ''),
        ('APPLY OUT OUT <- SQRT R:0', 'wrong destination count', ''),
        ('APPLY OUT <- SQRT R:0 R:1', 'wrong operand count', ''),
        ('APPLY R:999 <- SQRT R:0', 'index out of range', ''),
        ('APPLY OUT <- SQRT R:999', 'index out of range', ''),
        ('APPLY OUT <- EXP 1', 'unsupported operand signature', ''),
        ('APPLY OUT <- EXP U', 'unsupported operand signature', ''),
        ('APPLY COUT <- EXP R:0', 'wrong destination type', ''),
        ('APPLY OUT <- ARG R:0', 'unsupported operand signature', ''),
        ('APPLY OUT <- ABS R:0', 'unsupported operand signature', ''),
        ('APPLY OUT <- ABS 1', 'unsupported operand signature', ''),
        ('APPLY OUT <- ABS U', 'unsupported operand signature', ''),
        ('APPLY COUT <- ABS Z:0', 'wrong destination type', ''),
        ('APPLY OUT OUT <- ABS Z:0', 'wrong destination count', ''),
        ('APPLY OUT <- ABS Z:0 Z:1', 'wrong operand count', ''),
        ('APPLY R:999 <- ABS Z:0', 'index out of range', ''),
        ('APPLY OUT <- ABS Z:999', 'index out of range', ''),
        ('APPLY OUT <- ABS COUT', 'overflow/nonfinite result',
         'LET COUT := :I\nLET COUT *= :COMPLEX:1.3e308\nLET COUT += :COMPLEX:1.3e308\n'),
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
    for name in ('SIN', 'COS', 'TAN'):
        failures.extend((
            (f'APPLY OUT <- {name} 1', 'unsupported operand signature', ''),
            (f'APPLY OUT <- {name} U', 'unsupported operand signature', ''),
            (f'APPLY OUT <- {name} Z:0', 'unsupported operand signature', ''),
            (f'APPLY COUT <- {name} R:0', 'wrong destination type', ''),
            (f'APPLY OUT OUT <- {name} R:0', 'wrong destination count', ''),
            (f'APPLY OUT <- {name} R:0 R:1', 'wrong operand count', ''),
            (f'APPLY R:999 <- {name} R:0', 'index out of range', ''),
            (f'APPLY OUT <- {name} R:999', 'index out of range', ''),
        ))
    for numerator in ('1.0', '-1.0', '0.0'):
        preparation = f'LET OUT := {numerator}\nLET OUT /= 0.0\nLET COUT := :COMPLEX:OUT\n'
        failures.extend((f'APPLY R:0 <- {name} OUT', 'nonfinite input', preparation)
                        for name in ('SIN', 'COS', 'TAN'))
        failures.extend((
            ('APPLY R:0 <- LOG OUT', 'nonfinite input', preparation),
            ('APPLY Z:0 <- COMPLEX OUT R:0', 'nonfinite input', preparation),
            ('APPLY Z:0 <- COMPLEX R:0 OUT', 'nonfinite input', preparation),
            ('APPLY R:0 <- SQRT OUT', 'nonfinite input', preparation),
            ('APPLY Z:0 <- SQRT COUT', 'nonfinite input', preparation),
            ('APPLY R:0 <- EXP OUT', 'nonfinite input', preparation),
            ('APPLY Z:0 <- EXP COUT', 'nonfinite input', preparation),
            ('APPLY R:0 <- ARG COUT', 'nonfinite input', preparation),
            ('APPLY R:0 <- ABS COUT', 'nonfinite input', preparation),
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
