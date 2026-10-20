#!/usr/bin/env python3

"""Compare inverse-modified circuits with independent complex matrix calculations.

Example:
  python3 bra/test/qasm2qcx_inverse_numerical.py --bra bra/bin/bra
"""

import argparse
import cmath
import math
import pathlib
import re
import resource
import subprocess

from qasm2qcx_integer_array_numerical import check_error
from qasm2qcx_runtime_for_numerical import check_program, qasm2qcx
from qasm2qcx_u_phase_numerical import (
    builtin_u_matrix, conventional_u3_matrix, documented_cu_active_matrix)


def matrix(gate: str, parameters: tuple[float, ...]) -> list[list[complex]]:
    if gate in ('U', 'u3', 'cu', 'u2'):
        theta, phi, lam = ((math.pi / 2, *parameters) if gate == 'u2' else parameters[:3])
        # Use the independently transcribed published matrix, rather than
        # mirroring the converter's QCX phase-correction calculation.
        if gate == 'U':
            return builtin_u_matrix(theta, phi, lam)
        if gate == 'cu':
            return documented_cu_active_matrix(*parameters)
        return conventional_u3_matrix(theta, phi, lam)
    if gate in ('p', 'phase', 'u1', 's', 'sdg', 't', 'tdg', 'z'):
        phase = {'s': math.pi / 2, 'sdg': -math.pi / 2,
                 't': math.pi / 4, 'tdg': -math.pi / 4, 'z': math.pi}.get(gate)
        if phase is None:
            phase = parameters[0]
        return [[1, 0], [0, cmath.exp(1j * phase)]]
    if gate == 'h':
        scale = 1 / math.sqrt(2)
        return [[scale, scale], [scale, -scale]]
    if gate == 'id':
        return [[1, 0], [0, 1]]
    if gate == 'x':
        return [[0, 1], [1, 0]]
    if gate == 'y':
        return [[0, -1j], [1j, 0]]
    if gate == 'sx':
        return [[(1 + 1j) / 2, (1 - 1j) / 2], [(1 - 1j) / 2, (1 + 1j) / 2]]
    if gate in ('rx', 'ry', 'rz'):
        theta = parameters[0]
        c, s = math.cos(theta / 2), math.sin(theta / 2)
        if gate == 'rx':
            return [[c, -1j * s], [-1j * s, c]]
        if gate == 'ry':
            return [[c, -s], [s, c]]
        return [[cmath.exp(-1j * theta / 2), 0], [0, cmath.exp(1j * theta / 2)]]
    raise ValueError(f'No reference matrix for {gate}')


def apply_gate(state: list[complex], gate: str, parameters: tuple[float, ...],
               qubits: tuple[int, ...], inverse: bool = False) -> list[complex]:
    controls: tuple[int, ...] = ()
    controlled = {'cx': 'x', 'CX': 'x', 'cy': 'y', 'cz': 'z', 'ch': 'h',
                  'cp': 'p', 'cphase': 'p', 'crx': 'rx', 'cry': 'ry', 'crz': 'rz'}
    if gate in controlled or gate in ('cu', 'ccx', 'cswap'):
        controls = qubits[:-1] if gate == 'ccx' else qubits[:1]
        gate = 'x' if gate == 'ccx' else 'swap' if gate == 'cswap' else controlled.get(gate, gate)
    result = state.copy()
    if gate == 'swap':
        left, right = qubits[-2:]
        for index, amplitude in enumerate(state):
            if all(index & (1 << control) for control in controls):
                destination = index ^ ((1 << left) | (1 << right)) if ((index >> left) ^ (index >> right)) & 1 else index
                result[destination] = amplitude
        return result
    operator = matrix(gate, parameters)
    if inverse:
        operator = [[operator[column][row].conjugate() for column in range(2)] for row in range(2)]
    target = qubits[-1]
    for index in range(len(state)):
        if index & (1 << target) or not all(index & (1 << control) for control in controls):
            continue
        partner = index | (1 << target)
        result[index] = operator[0][0] * state[index] + operator[0][1] * state[partner]
        result[partner] = operator[1][0] * state[index] + operator[1][1] * state[partner]
    return result


def check_amplitudes(bra: pathlib.Path, source: str, expected: list[complex]) -> None:
    lines = qasm2qcx.convert('OPENQASM 3; include "stdgates.inc"; ' + source)
    lines.append('DO AMPLITUDES')
    result = subprocess.run([str(bra)], input='\n'.join(lines) + '\n', check=True,
                            text=True, capture_output=True, timeout=30)
    amplitudes = {int(index, 2): complex(float(real), float(imaginary)) for index, real, imaginary in
                  re.findall(r'^([01]+) => (\S+) \+ (\S+) i$', result.stdout, re.MULTILINE)}
    # bra prints six significant digits. Compare actual complex amplitudes,
    # without aligning away global phase or reducing them to probabilities.
    if set(amplitudes) != set(range(len(expected))) or any(
            not cmath.isclose(amplitudes[index], value, rel_tol=1e-5, abs_tol=2e-6)
            for index, value in enumerate(expected)):
        raise RuntimeError(f'Inverse amplitude mismatch\nsource:\n{source}\n'
                           f'expected: {expected}\nstdout:\n{result.stdout}')


def prepared_state(basis: int | None) -> tuple[str, list[complex]]:
    state = [0j] * 8
    state[0 if basis is None else basis] = 1
    if basis is not None:
        return ''.join(f'x q[{index}]; ' for index in range(3) if basis & (1 << index)), state
    for index in range(3):
        state = apply_gate(state, 'h', (), (index,))
    state = apply_gate(state, 's', (), (1,))
    state = apply_gate(state, 't', (), (2,))
    return 'h q; s q[1]; t q[2]; ', state


def inverse_step(state: list[complex], theta: float) -> list[complex]:
    state = apply_gate(state, 'U', (theta, theta / 2, -theta), (0,), True)
    state = apply_gate(state, 'cu', (theta, theta / 2, -theta, theta + 0.1), (0, 1), True)
    return [value * cmath.exp(-1j * theta / 3) for value in state]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    bra = parser.parse_args().bra
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    # Enumerate the public gate signatures only; reference matrices and the
    # adjoint operation above are independent of QCX mappings and lowering.
    for gate, (parameter_count, qubit_count) in qasm2qcx.QASM2QCXConverter.gate_signatures.items():
        parameters = tuple((index + 1) / 5 for index in range(parameter_count))
        arguments = '(' + ', '.join(map(str, parameters)) + ')' if parameters else ''
        operands = ', '.join(f'q[{index}]' for index in range(qubit_count))
        instruction = f'{gate}{arguments} {operands};'
        for basis in (*range(8), None):
            preparation, state = prepared_state(basis)
            for count in (1, 2, 3):
                expected = apply_gate(state, gate, parameters, tuple(range(qubit_count)), bool(count % 2))
                check_amplitudes(bra, 'qubit[3] q; ' + preparation + 'inv @ ' * count + instruction, expected)
        # Runtime expressions must not change either the angle conversion or
        # the phase convention. Include a nontrivial complex input state.
        preparation, state = prepared_state(None)
        runtime_args = '(' + ', '.join(f'theta + {index / 5}' for index in range(parameter_count)) + ')' if parameters else ''
        expected = apply_gate(state, gate, parameters, tuple(range(qubit_count)), True)
        check_amplitudes(bra, 'qubit[3] q; float theta = 0.2; ' + preparation
                         + f'inv @ {gate}{runtime_args} {operands};', expected)

    preparation, state = prepared_state(None)
    for phase_source, phase in (('0.7', 0.7), ('theta', 0.7), ('2', 2), ('uint(2)', 2)):
        for count in (1, 2, 3):
            expected = [value * cmath.exp(1j * phase * (-1 if count % 2 else 1)) for value in state]
            check_amplitudes(bra, 'qubit[3] q; float theta = 0.7; ' + preparation
                             + 'inv @ ' * count + f'gphase({phase_source});', expected)

    for gate in ('h', 'rx', 'U'):
        parameters = () if gate == 'h' else (0.3,) if gate == 'rx' else (0.3, 0.5, 0.7)
        arguments = '(' + ', '.join(map(str, parameters)) + ')' if parameters else ''
        expected = state
        for target in range(3):
            expected = apply_gate(expected, gate, parameters, (target,), True)
        check_amplitudes(bra, 'qubit[3] q; ' + preparation + f'inv @ {gate}{arguments} q;', expected)
    expected = apply_gate(state, 'cu', (0.3, 0.5, 0.7, 0.9), (0, 1), True)
    expected = apply_gate(expected, 'cu', (0.3, 0.5, 0.7, 0.9), (0, 2), True)
    check_amplitudes(bra, 'qubit[3] q; ' + preparation
                     + 'inv @ cu(0.3, 0.5, 0.7, 0.9) q[0], q[1:2];', expected)

    body = '''inv @ U(theta, theta / 2, -theta) q[0];
        inv @ cu(theta, theta / 2, -theta, theta + 0.1) q[0], q[1];
        inv @ gphase(theta / 3);'''
    cases = [
        ('', 'for int i in [0:2]', '0.2 + float(i) / 5', [0.2, 0.4, 0.6], ''),
        ('int stop = 2;', 'for int i in [0:stop]', '0.2 + float(i) / 5', [0.2, 0.4, 0.6], ''),
        ('int step = 1;', 'for int i in [0:step:2]', '0.2 + float(i) / 5', [0.2, 0.4, 0.6], ''),
        ('int value = 1;', 'for int i in {0, value, 2}', '0.2 + float(i) / 5', [0.2, 0.4, 0.6], ''),
        ('array[int, 3] values = {0, 1, 2};', 'for int i in values', '0.2 + float(i) / 5', [0.2, 0.4, 0.6], ''),
        ('array[uint[4], 3] values = {0, 1, 2};', 'for uint[4] i in values', '0.2 + float(i) / 5', [0.2, 0.4, 0.6], ''),
        ('array[float, 3] values = {0.2, 0.4, 0.6};', 'for float i in values', 'i', [0.2, 0.4, 0.6], ''),
        ('array[bool, 3] values = {true, false, true};', 'for bool i in values', '0.2 + float(i) / 5', [0.4, 0.2, 0.4], ''),
        ('bit[3] values = "101";', 'for bit i in values', '0.2 + float(i) / 5', [0.4, 0.2, 0.4], ''),
        ('array[complex, 3] values = {1.0im, 1 + 1.0im, 2 + 1.0im};', 'for complex i in values',
         '0.2 + float(i) / 5', [0.2, 0.4, 0.6], ''),
        ('int i = 0;', 'while (i < 3)', '0.2 + float(i) / 5', [0.2, 0.4, 0.6], 'i += 1;'),
    ]
    for declaration, header, expression, angles, update in cases:
        expected = state
        for theta in angles:
            expected = inverse_step(expected, theta)
        check_amplitudes(bra, 'qubit[3] q; ' + preparation + declaration + 'float theta; '
                         + header + '{ theta = ' + expression + ';' + body + update + '}', expected)

    expected = state
    for i in range(2):
        for j in (0, 2):
            expected = inverse_step(expected, 0.2 + (i + j) / 5)
    check_amplitudes(bra, 'qubit[3] q; ' + preparation + '''int stop = 3; float theta;
        for int i in [0:1] { for int j in [0:stop] {
            if (j == 1) { continue; } if (j == 3) { break; }
            theta = 0.2 + float(i + j) / 5;''' + body + '}}', expected)

    check_amplitudes(bra, 'qubit[3] q; ' + preparation + '''array[float, 1] values = {0.3};
        int bad = 1; int zero = 0;
        if (false) { inv @ U(values[bad], 0, 0) q[0]; }
        while (false) { inv @ gphase(values[1 / zero]); }
        for int i in [1:0] { inv @ rx(values[bad]) q[0]; }
        if (false && bool(1 / zero)) { inv @ gphase(values[bad]); }
        for int i in [0:1] { continue; inv @ rx(values[bad]) q[0]; }''', state)

    # A measurement-derived condition is deterministic here; it must not
    # cause parameters to be evaluated on the skipped path.
    check_program(bra, '''include "stdgates.inc"; qubit q; x q; bit b = measure q; int zero = 0;
        float theta = pi; if (b) { inv @ rx(theta) q; } else { inv @ rx(1 / zero) q; }
        bit result = measure q; int n = int(result);''', ('N1', 'THETA31'), ['0', '3.14159'])
    check_error(bra, '''include "stdgates.inc"; qubit q; array[float, 1] values = {0.3}; int bad = 1;
        inv @ rx(values[bad]) q;''', 'assertion failed in ASSERT')
    check_error(bra, 'qubit q; int zero = 0; inv @ gphase(1 / zero);', 'integer division by zero in LET')
    check_program(bra, '''include "stdgates.inc"; qubit q; float theta = 0.3;
        inv @ U(theta, theta + 1, theta * 2) q; inv @ rx(theta) q; inv @ gphase(theta);
        uint u = 2; inv @ gphase(u);''', ('THETA31', 'U1'), ['0.3', '2'])

    print('Inverse modifier numerical checks passed')


if __name__ == '__main__':
    main()
