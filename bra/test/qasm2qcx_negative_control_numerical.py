#!/usr/bin/env python3

"""Check negctrl and mixed modifiers against independently conditioned matrices.

  python3 bra/test/qasm2qcx_negative_control_numerical.py --bra /path/to/bra
"""

import argparse
import cmath
import pathlib
import resource
import subprocess

from qasm2qcx_control_numerical import prepared_state
from qasm2qcx_inverse_numerical import apply_gate, check_amplitudes
from qasm2qcx_runtime_for_numerical import qasm2qcx


def conditioned(state, gate, parameters, controls, operands, values, inverse=False):
    transformed = apply_gate(state, gate, parameters, operands, inverse)
    return [transformed[index] if all((index >> wire) & 1 == value
                                     for wire, value in zip(controls, values)) else amplitude
            for index, amplitude in enumerate(state)]


def phase_reference(state, controls, values, angle):
    return [amplitude * cmath.exp(1j * angle) if all((index >> wire) & 1 == value
                                                   for wire, value in zip(controls, values)) else amplitude
            for index, amplitude in enumerate(state)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    bra = parser.parse_args().bra.resolve()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    profiles = [('negctrl', (0,), False), ('negctrl(2)', (0, 0), False),
                ('ctrl @ negctrl', (1, 0), False), ('negctrl @ ctrl', (0, 1), False),
                ('negctrl @ ctrl @ negctrl', (0, 1, 0), False),
                ('inv @ negctrl', (0,), True), ('negctrl @ inv @ ctrl', (0, 1), True),
                ('ctrl @ inv @ negctrl(2) @ inv', (1, 0, 0), False),
                ('ctrl @ inv', (1,), True)]
    checked = 0
    for gate, (parameter_count, base_qubits) in qasm2qcx.QASM2QCXConverter.gate_signatures.items():
        parameters = tuple((index + 1) / 5 for index in range(parameter_count))
        arguments = '(' + ', '.join(map(str, parameters)) + ')' if parameters else ''
        for profile_index, (modifiers, values, inverse) in enumerate(profiles):
            qubits = len(values) + base_qubits
            # Reverse physical wire order; the controls are not sorted by index.
            order = tuple(reversed(range(qubits)))
            controls, operands = order[:len(values)], order[len(values):]
            quantum_arguments = ', '.join(f'q[{wire}]' for wire in order)
            bases = (*range(1 << qubits), None) if profile_index < 2 else (0, (1 << qubits) - 1, None)
            for basis in bases:
                preparation, state = prepared_state(qubits, basis)
                expected = conditioned(state, gate, parameters, controls, operands, values, inverse)
                check_amplitudes(bra, f'qubit[{qubits}] q; ' + preparation
                                 + f'{modifiers} @ {gate}{arguments} {quantum_arguments};', expected)
                checked += 1
        # Every parameterized gate also takes runtime expressions under an
        # inverted negative control, including U's and cu's phase corrections.
        if parameters:
            qubits = base_qubits + 1
            preparation, state = prepared_state(qubits)
            controls, operands = (qubits - 1,), tuple(range(base_qubits))
            expected = conditioned(state, gate, parameters, controls, operands, (0,), True)
            runtime = '(' + ', '.join(f'theta + {index / 5}' for index in range(parameter_count)) + ')'
            quantum_arguments = ', '.join(f'q[{wire}]' for wire in controls + operands)
            check_amplitudes(bra, f'qubit[{qubits}] q; float theta = 0.2; ' + preparation
                             + f'negctrl @ inv @ {gate}{runtime} {quantum_arguments};', expected)
            checked += 1

    preparation, state = prepared_state(4)
    for modifiers, values, inverse in profiles:
        controls = (3, 0, 2)[:len(values)]
        operands = ', '.join(f'q[{wire}]' for wire in controls)
        for expression, angle in (('0.7', 0.7), ('theta', 0.7), ('n', 2), ('u', 2)):
            expected = phase_reference(state, controls, values, -angle if inverse else angle)
            check_amplitudes(bra, 'qubit[4] q; float theta = 0.7; int n = 2; uint u = 2; '
                             + preparation + f'{modifiers} @ gphase({expression}) {operands};', expected)
            checked += 1

    # Broadcasting across control and target registers, ranges and selections.
    expected = state
    for negative, positive in ((0, 2), (1, 3)):
        expected = phase_reference(expected, (negative, positive), (0, 1), -0.7)
    check_amplitudes(bra, 'qubit[4] q; ' + preparation
                     + 'negctrl @ inv @ ctrl @ gphase(0.7) q[0:1], q[2:3];', expected)
    checked += 1
    expected = state
    for target in (0, 2):
        expected = conditioned(expected, 'U', (0.2, 0.4, 0.6), (3,), (target,), (0,), True)
    check_amplitudes(bra, 'qubit[4] q; ' + preparation
                     + 'negctrl @ inv @ U(0.2, 0.4, 0.6) q[3], q[{0, 2}];', expected)
    checked += 1

    def step(current, theta):
        current = conditioned(current, 'U', (theta, theta / 2, -theta), (3,), (2,), (0,), True)
        current = conditioned(current, 'cu', (theta, theta / 2, -theta, theta + 0.1),
                              (3, 0), (1, 2), (1, 0), True)
        return phase_reference(current, (3, 0), (0, 1), -theta / 3)

    body = '''negctrl @ inv @ U(theta, theta / 2, -theta) q[3], q[2];
        ctrl @ negctrl @ inv @ cu(theta, theta / 2, -theta, theta + 0.1) q[3], q[0], q[1], q[2];
        negctrl @ inv @ ctrl @ gphase(theta / 3) q[3], q[0];'''
    for declaration, header, expression, angles, update in (
            ('', 'for int i in [0:2]', '0.2 + float(i) / 5', [0.2, 0.4, 0.6], ''),
            ('int stop = 2;', 'for int i in [0:stop]', '0.2 + float(i) / 5', [0.2, 0.4, 0.6], ''),
            ('int stride = 1;', 'for int i in [0:stride:2]', '0.2 + float(i) / 5', [0.2, 0.4, 0.6], ''),
            ('int element = 1;', 'for int i in {0, element, 2}', '0.2 + float(i) / 5', [0.2, 0.4, 0.6], ''),
            ('array[float, 3] values = {0.2, 0.4, 0.6};', 'for float i in values', 'i', [0.2, 0.4, 0.6], ''),
            ('int i = 0;', 'while (i < 3)', '0.2 + float(i) / 5', [0.2, 0.4, 0.6], 'i += 1;')):
        expected = state
        for theta in angles:
            expected = step(expected, theta)
        check_amplitudes(bra, 'qubit[4] q; ' + preparation + declaration + 'float theta; '
                         + header + '{ theta = ' + expression + ';' + body + update + '}', expected)
        checked += 1

    # A quantum control selects a unitary block; it does not short-circuit
    # classical parameter evaluation. Conversely, skipped classical paths do.
    for circuit in ('', 'x q[0];'):
        lines = qasm2qcx.convert('OPENQASM 3; include "stdgates.inc"; qubit[2] q; '
            'array[float, 1] angles; int bad = 1; ' + circuit
            + 'negctrl @ rx(angles[bad]) q[0], q[1];')
        result = subprocess.run([str(bra)], input='\n'.join(lines) + '\n',
                                text=True, capture_output=True, timeout=30)
        if result.returncode == 0 or 'assertion failed' not in result.stderr:
            raise RuntimeError(f'negative control suppressed a parameter check: {result}')
    check_amplitudes(bra, 'qubit[4] q; ' + preparation + '''array[float, 1] angles; int bad = 1;
        if (false) { negctrl @ U(angles[bad], 0, 0) q[0], q[1]; }
        while (false) { ctrl @ negctrl @ inv @ gphase(angles[bad]) q[0], q[1]; }
        for int i in [1:0] { negctrl @ rx(angles[bad]) q[0], q[1]; }''', state)
    checked += 1
    print(f'{checked} negative/mixed modifier numerical cases and 2 parameter failures passed')


if __name__ == '__main__':
    main()
