#!/usr/bin/env python3

"""Check QCX control values against dense complex matrices (no converter needed).

  python3 bra/test/negative_controls_numerical.py --bra /path/to/bra

Extra bra options may be supplied with --bra-arg=--threads=2, for example.
"""

import argparse
import cmath
import itertools
import math
import pathlib
import re
import resource
import subprocess


def operator_matrix(gate, targets, parameters):
    if gate.endswith('+'):
        base = operator_matrix(gate[:-1], targets, parameters)
        return [[base[col][row].conjugate() for col in range(len(base))]
                for row in range(len(base))]
    size = 1 << len(targets)
    identity = [[complex(row == col) for col in range(size)] for row in range(size)]
    if gate in ('I', 'IC'):
        return identity
    if gate == 'SZ' and targets:
        return [[1j ** bin(row).count('1') if row == col else 0j
                 for col in range(size)] for row in range(size)]
    if gate == 'H':
        return [[1 / math.sqrt(2), 1 / math.sqrt(2)],
                [1 / math.sqrt(2), -1 / math.sqrt(2)]]
    if gate in ('U2', 'U3'):
        theta, phi, lam = (math.pi / 2, *parameters) if gate == 'U2' else parameters
        c, s = math.cos(theta / 2), math.sin(theta / 2)
        return [[c, -cmath.exp(1j * lam) * s],
                [cmath.exp(1j * phi) * s, cmath.exp(1j * (phi + lam)) * c]]
    if gate in ('Z', 'SZ', 'S', 'T', 'U1', 'R') and not targets:
        angle = {'Z': math.pi, 'SZ': math.pi / 2, 'S': math.pi / 2,
                 'T': math.pi / 4}.get(gate)
        if angle is None:
            angle = parameters[0] if gate == 'U1' else 2 * math.pi / 2 ** parameters[0]
        return [[cmath.exp(1j * angle)]]
    axis = gate[-1]
    pauli = [[0j for _ in range(size)] for _ in range(size)]
    for col in range(size):
        if gate in ('SWAP', 'ESWAP'):
            row = ((col & 1) << 1) | ((col >> 1) & 1)
            coefficient = 1
        elif axis == 'Z':
            row = col
            coefficient = (-1) ** bin(col).count('1')
        else:
            row = col ^ (size - 1)
            coefficient = (1j ** len(targets) * (-1) ** bin(col).count('1')
                           if axis == 'Y' else 1)
        pauli[row][col] = coefficient
    if gate.startswith('S') and gate != 'SWAP':
        a, b = (1 + 1j) / 2, (1 - 1j) / 2
    elif gate.startswith('E'):
        a, b = math.cos(parameters[0]), 1j * math.sin(parameters[0])
    elif gate in ('+X', '-X', '+Y', '-Y'):
        a = 1 / math.sqrt(2)
        b = (1j if gate[0] == '+' else -1j) / math.sqrt(2)
    else:
        a, b = 0, 1
    return [[a * identity[row][col] + b * pauli[row][col]
             for col in range(size)] for row in range(size)]


def apply(state, matrix, controls, targets, values):
    result = state.copy()
    mask = sum(1 << target for target in targets)
    for base in range(len(state)):
        if base & mask or any((base >> wire) & 1 != value
                              for wire, value in zip(controls, values)):
            continue
        indices = [base | sum(((local >> bit) & 1) << wire
                              for bit, wire in enumerate(targets))
                   for local in range(len(matrix))]
        for row, index in enumerate(indices):
            result[index] = sum(matrix[row][col] * state[source]
                                for col, source in enumerate(indices))
    return result


def run(command, program):
    return subprocess.run(command, input=program, text=True, capture_output=True, timeout=60)


def check(command, program, expected):
    result = run(command, program + 'DO AMPLITUDES\n')
    actual = {int(bits, 2): complex(float(real), float(imag)) for bits, real, imag in
              re.findall(r'^([01]+) => (\S+) \+ (\S+) i$', result.stdout, re.MULTILINE)}
    if result.returncode or set(actual) != set(range(len(expected))) or any(
            not cmath.isclose(actual[index], value, rel_tol=1e-5, abs_tol=2e-6)
            for index, value in enumerate(expected)):
        raise RuntimeError(f'QCX control-value mismatch:\n{program}\n'
                           f'expected: {expected}\n{result.stdout}\n{result.stderr}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    parser.add_argument('--bra-arg', action='append', default=[])
    args = parser.parse_args()
    command = [str(args.bra.resolve()), *args.bra_arg]
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    setup = 'QUBITS 5\n'
    state = [1 + 0j] + [0j] * 31
    # Unequal complex amplitudes distinguish control/target permutations and
    # relative phases; a uniform real state can hide several wrong gates.
    for wire in range(5):
        parameters = (0.4 + wire / 7, 0.2 + wire / 11, -0.3)
        setup += f'U3 {wire} ' + ' '.join(map(str, parameters)) + '\n'
        state = apply(state, operator_matrix('U3', (wire,), parameters), (), (wire,), ())

    cases = []
    for count in (1, 2, 3):
        prefix = 'C' if count == 1 else f'C{count}'
        # Deliberately unsorted, nonconsecutive controls and targets.
        controls = (4, 0, 3)[:count]
        for gate, parameters in (
                ('H', ()), ('NOT', ()), ('X', ()), ('Y', ()),
                ('SX', ()), ('SY', ()), ('U2', (0.2, -0.3)),
                ('U3', (0.4, 0.2, -0.3)), ('+X', ()), ('-X', ()),
                ('+Y', ()), ('-Y', ()), ('EX', (0.3,)),
                ('EY', (0.3,)), ('EZ', (0.3,)), ('I', ())):
            cases.append((prefix + gate, 'X' if gate == 'NOT' else gate,
                          controls, (2,), parameters))
        for gate, parameters in (('SWAP', ()), ('ESWAP', (0.3,)),
                                 ('X', ()), ('Y', ()), ('Z', ()),
                                 ('SZ', ()), ('EX', (0.3,)), ('EY', (0.3,)), ('EZ', (0.3,))):
            mnemonic = gate if 'SWAP' in gate else gate + '2'
            cases.append((prefix + mnemonic, gate, controls, (2, 1), parameters))
    # Intrinsic phase controls are genuine control operands too, including
    # the sole operand of Z, S, T, R, U1 and their controlled variants.
    for count in (1, 2, 3):
        prefix = '' if count == 1 else 'C' if count == 2 else 'C2'
        for gate, parameters in (('Z', ()), ('SZ', ()), ('S', ()), ('T', ()),
                                 ('U1', (0.3,)), ('R', (3,)), ('IC', ())):
            cases.append((prefix + gate, gate, (4, 0, 3)[:count], (), parameters))
    cases.extend([('TOFFOLI', 'X', (4, 0), (2,), ()),
                  ('U', 'R', (4, 0), (), (3,)),
                  ('CCX', 'X', (4, 0), (2,), ()),
                  ('CCXX', 'X', (4, 0), (2, 1), ())])
    cases += [(mnemonic + '+', gate + '+', controls, targets, parameters)
              for mnemonic, gate, controls, targets, parameters in cases
              if gate in ('SX', 'SY', 'SZ', 'S', 'T', 'U1', 'U2', 'U3',
                          'R', 'EX', 'EY', 'EZ', 'ESWAP')]

    checked = 0
    for mnemonic, gate, controls, targets, parameters in cases:
        matrix = operator_matrix(gate, targets, parameters)
        for values in itertools.product((0, 1), repeat=len(controls)):
            operands = [f'{wire}#{value}' for wire, value in zip(controls, values)]
            operands += list(map(str, targets)) + list(map(str, parameters))
            instruction = mnemonic + ' ' + ' '.join(operands) + '\n'
            expected = apply(state, matrix, controls, targets, values)
            check(command, setup + instruction, expected)
            check(command, setup + 'BEGIN FUSION\n' + instruction + 'END FUSION\n', expected)
            checked += 2
        # Explicit positive annotations preserve legacy bare-wire behavior.
        bare = mnemonic + ' ' + ' '.join(map(str, (*controls, *targets, *parameters))) + '\n'
        check(command, setup + bare, apply(state, matrix, controls, targets, (1,) * len(controls)))
        checked += 1

    # A wire can be a zero-control in one gate and a one-control in another.
    # Include a target change and repeated phase gates to exercise fusion's
    # outer-control selection and phase batching boundaries.
    sequence = [('CU1', 'U1', (4, 0), (), (0, 1), (0.3,)),
                ('CU1', 'U1', (4, 0), (), (1, 1), (0.2,)),
                ('CH', 'H', (0,), (4,), (0,), ()),
                ('C2X', 'X', (4, 0), (2,), (1, 0), ())]
    body, expected = '', state
    for mnemonic, gate, controls, targets, values, parameters in sequence:
        body += mnemonic + ' ' + ' '.join(
            [f'{wire}#{value}' for wire, value in zip(controls, values)]
            + list(map(str, targets)) + list(map(str, parameters))) + '\n'
        expected = apply(expected, operator_matrix(gate, targets, parameters), controls, targets, values)
    check(command, setup + 'BEGIN FUSION\n' + body + 'END FUSION\n', expected)

    # ! is the comment marker; # is part of an operand even beside a comment.
    check(command, setup + '! CH 4#1 2\nCH 4#0 2 ! negative control #0\n',
          apply(state, operator_matrix('H', (2,), ()), (4,), (2,), (0,)))

    invalid = ['H 0#0', 'X 0#1', 'ZZ 0#0 1', 'CH 0 1#0',
               'C2X 0#0 0#1 1', 'CX 0#0 0', 'CU1 0#0 0 0.3',
               'CH 5#0 1', 'CH 0#0 5', 'CH 4294967296#0 1',
               'CH 0:0 1', 'CH 0:1 1', '# comment']
    invalid += [f'CH {operand} 1' for operand in
                ('0#', '#0', '0#2', '0#-1', '0#+1', '0#01', '0#1#0',
                 '-1#0', '+0#0', 'a#0', '0 #0', '0# 0')]
    for instruction in invalid:
        result = run(command, 'QUBITS 5\n' + instruction + '\nPRINTLN "unexpected"\n')
        if result.returncode == 0 or 'unexpected' in result.stdout:
            raise RuntimeError(f'invalid control operand accepted: {instruction}\n{result}')
    print(f'{checked + 2} numerical cases and {len(invalid)} invalid operands passed')


if __name__ == '__main__':
    main()
