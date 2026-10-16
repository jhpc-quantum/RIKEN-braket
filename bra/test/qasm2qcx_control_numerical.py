#!/usr/bin/env python3

"""Check control modifiers against independently conditioned complex matrices.

Example:
  python3 bra/test/qasm2qcx_control_numerical.py --bra bra/bin/bra
"""

import argparse
import cmath
import pathlib
import resource

from qasm2qcx_integer_array_numerical import check_error
from qasm2qcx_inverse_numerical import apply_gate, check_amplitudes
from qasm2qcx_runtime_for_numerical import check_program, qasm2qcx


def controlled_apply(state: list[complex], gate: str, parameters: tuple[float, ...],
                     controls: tuple[int, ...], operands: tuple[int, ...]) -> list[complex]:
    # Apply the independent base matrix only within the all-one control block.
    # No QCX mnemonic or converter modifier helper is used in this reference.
    transformed = apply_gate(state, gate, parameters, operands)
    mask = sum(1 << control for control in controls)
    return [transformed[index] if index & mask == mask else value for index, value in enumerate(state)]


def conditional_phase(state: list[complex], controls: tuple[int, ...], phase: float) -> list[complex]:
    mask = sum(1 << control for control in controls)
    factor = cmath.exp(1j * phase)
    return [value * factor if index & mask == mask else value for index, value in enumerate(state)]


def prepared_state(qubits: int, basis: int | None = None) -> tuple[str, list[complex]]:
    state = [0j] * (1 << qubits)
    state[0 if basis is None else basis] = 1
    if basis is not None:
        return ''.join(f'x q[{index}]; ' for index in range(qubits) if basis & (1 << index)), state
    for index in range(qubits):
        state = apply_gate(state, 'h', (), (index,))
    state = apply_gate(state, 's', (), (0,))
    state = apply_gate(state, 't', (), (qubits - 1,))
    return f'h q; s q[0]; t q[{qubits - 1}]; ', state


def controlled_step(state: list[complex], theta: float) -> list[complex]:
    state = controlled_apply(state, 'U', (theta, theta / 2, -theta), (0,), (2,))
    state = controlled_apply(state, 'cu', (theta, theta / 2, -theta, theta + 0.1), (0,), (1, 2))
    return conditional_phase(state, (0, 1), theta / 3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    bra = parser.parse_args().bra
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    for gate, (parameter_count, base_qubits) in qasm2qcx.QASM2QCXConverter.gate_signatures.items():
        parameters = tuple((index + 1) / 5 for index in range(parameter_count))
        arguments = '(' + ', '.join(map(str, parameters)) + ')' if parameters else ''
        for count in (1, 2):
            qubits = base_qubits + count
            controls = tuple(range(count))
            operands = tuple(range(count, qubits))
            quantum_arguments = ', '.join(f'q[{index}]' for index in range(qubits))
            modifier = 'ctrl' if count == 1 else 'ctrl(2)'
            for basis in (*range(1 << qubits), None):
                preparation, state = prepared_state(qubits, basis)
                expected = controlled_apply(state, gate, parameters, controls, operands)
                check_amplitudes(bra, f'qubit[{qubits}] q; ' + preparation
                                 + f'{modifier} @ {gate}{arguments} {quantum_arguments};', expected)
            preparation, state = prepared_state(qubits)
            expected = controlled_apply(state, gate, parameters, controls, operands)
            runtime_args = '(' + ', '.join(f'theta + {index / 5}' for index in range(parameter_count)) + ')' if parameters else ''
            check_amplitudes(bra, f'qubit[{qubits}] q; float theta = 0.2; const uint N = {count}; '
                             + preparation + f'ctrl(N) @ {gate}{runtime_args} {quantum_arguments};', expected)
            if count == 2:
                check_amplitudes(bra, f'qubit[{qubits}] q; ' + preparation
                                 + f'ctrl @ ctrl @ {gate}{arguments} {quantum_arguments};', expected)

    for count in (1, 2, 3):
        preparation, state = prepared_state(count)
        operands = ', '.join(f'q[{index}]' for index in reversed(range(count)))
        for expression, phase in (('0.7', 0.7), ('theta', 0.7), ('n', 2), ('u', 2)):
            expected = conditional_phase(state, tuple(range(count)), phase)
            check_amplitudes(bra, f'qubit[{count}] q; float theta = 0.7; int n = 2; uint u = 2; '
                             + preparation + f'ctrl({count}) @ gphase({expression}) {operands};', expected)

    preparation, state = prepared_state(4)
    # Nonconsecutive/permuted controls expose accidental reliance on physical
    # bit order; control and target register broadcasting are checked separately.
    for gate, parameters, controls, operands in (
            ('U', (0.2, 0.4, 0.6), (3, 1), (0,)),
            ('cu', (0.2, 0.4, 0.6, 0.8), (3,), (2, 0)),
            ('swap', (), (3, 0), (2, 1))):
        expected = controlled_apply(state, gate, parameters, controls, operands)
        arguments = '(' + ', '.join(map(str, parameters)) + ')' if parameters else ''
        quantum_arguments = ', '.join(f'q[{index}]' for index in controls + operands)
        check_amplitudes(bra, 'qubit[4] q; ' + preparation
                         + f'ctrl({len(controls)}) @ {gate}{arguments} {quantum_arguments};', expected)

    expected = state
    for target in (1, 2, 3):
        expected = controlled_apply(expected, 'U', (0.2, 0.4, 0.6), (0,), (target,))
    check_amplitudes(bra, 'qubit[4] q; ' + preparation + 'ctrl @ U(0.2, 0.4, 0.6) q[0], q[1:3];', expected)
    expected = state
    for control in (0, 1, 2):
        expected = controlled_apply(expected, 'rx', (0.7,), (control,), (3,))
    check_amplitudes(bra, 'qubit[4] q; ' + preparation + 'ctrl @ rx(0.7) q[0:2], q[3];', expected)
    expected = state
    for control, target in ((0, 2), (1, 3)):
        expected = controlled_apply(expected, 'u3', (0.2, 0.4, 0.6), (control,), (target,))
    check_amplitudes(bra, 'qubit[4] q; ' + preparation + 'ctrl @ u3(0.2, 0.4, 0.6) q[0:1], q[2:3];', expected)
    expected = conditional_phase(conditional_phase(state, (0, 2), 0.7), (1, 3), 0.7)
    check_amplitudes(bra, 'qubit[4] q; ' + preparation + 'ctrl(2) @ gphase(0.7) q[0:1], q[2:3];', expected)

    preparation, state = prepared_state(3)
    body = '''ctrl @ U(theta, theta / 2, -theta) q[0], q[2];
        ctrl @ cu(theta, theta / 2, -theta, theta + 0.1) q[0], q[1], q[2];
        ctrl(2) @ gphase(theta / 3) q[0], q[1];'''
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
            expected = controlled_step(expected, theta)
        check_amplitudes(bra, 'qubit[3] q; ' + preparation + declaration + 'float theta; '
                         + header + '{ theta = ' + expression + ';' + body + update + '}', expected)

    expected = state
    for i in range(2):
        for j in (0, 2):
            expected = controlled_step(expected, 0.2 + (i + j) / 5)
    check_amplitudes(bra, 'qubit[3] q; ' + preparation + '''int stop = 3; float theta;
        for int i in [0:1] { for int j in [0:stop] {
            if (j == 1) { continue; } if (j == 3) { break; }
            theta = 0.2 + float(i + j) / 5;''' + body + '}}', expected)
    expected = controlled_step(state, 0.2)
    check_amplitudes(bra, 'qubit[3] q; ' + preparation + '''array[float, 4] angles = {0.2, 0.1, -0.2, 0.3}; int i = 0;
        ctrl @ U(angles[i], angles[i + 1], angles[i + 2]) q[0], q[2];
        ctrl @ cu(angles[i], angles[i + 1], angles[i + 2], angles[i + 3]) q[0], q[1], q[2];
        ctrl(2) @ gphase(angles[i] / 3) q[0], q[1];''', expected)

    check_amplitudes(bra, 'qubit[3] q; ' + preparation + '''array[float, 1] angles = {0.2}; int bad = 1; int zero = 0;
        if (false) { ctrl @ U(angles[bad], 0, 0) q[0], q[1]; }
        while (false) { ctrl @ gphase(angles[1 / zero]) q[0]; }
        for int i in [1:0] { ctrl @ rx(angles[bad]) q[0], q[1]; }
        if (false && bool(1 / zero)) { ctrl @ gphase(angles[bad]) q[0]; }
        for int i in [0:1] { continue; ctrl @ rx(angles[bad]) q[0], q[1]; }''', state)
    check_program(bra, '''include "stdgates.inc"; qubit[2] q; x q[0]; bit b = measure q[0];
        int zero = 0; if (b) { ctrl @ x q[0], q[1]; } else { ctrl @ rx(1 / zero) q[0], q[1]; }
        bit result = measure q[1]; int n = int(result);''', ('N1',), ['1'])
    # A quantum control in |0> does not short-circuit classical parameter
    # evaluation. Unlike an unexecuted classical branch, this must fail.
    check_error(bra, '''include "stdgates.inc"; qubit[2] q; array[float, 1] angles = {0.2}; int bad = 1;
        ctrl @ rx(angles[bad]) q[0], q[1];''', 'assertion failed in ASSERT')
    check_error(bra, 'qubit q; int zero = 0; ctrl @ gphase(1 / zero) q;', 'integer division by zero in LET')
    check_program(bra, '''qubit[2] q; float theta = 0.3; uint u = 2;
        ctrl @ U(theta, theta + 1, theta * 2) q[0], q[1]; ctrl @ gphase(theta) q[0];
        ctrl @ gphase(u) q[0];''', ('THETA31', 'U1'), ['0.3', '2'])

    print('Control modifier numerical checks passed')


if __name__ == '__main__':
    main()
