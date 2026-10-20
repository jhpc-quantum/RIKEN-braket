#!/usr/bin/env python3

"""Compare integer-powered gates with independent complex matrix references.

  python3 bra/test/qasm2qcx_power_numerical.py --bra /path/to/bra
"""

import argparse
import pathlib
import resource

from qasm2qcx_control_numerical import prepared_state
from qasm2qcx_inverse_numerical import check_amplitudes
from qasm2qcx_negative_control_numerical import conditioned, phase_reference
from qasm2qcx_runtime_for_numerical import qasm2qcx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    bra = parser.parse_args().bra.resolve()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    profiles = [('', (), False), ('ctrl @ ', (1,), False),
                ('negctrl @ ', (0,), False), ('negctrl @ inv @ ctrl @ ', (0, 1), True)]
    checked = 0
    for gate, (parameter_count, base_qubits) in qasm2qcx.QASM2QCXConverter.gate_signatures.items():
        parameters = tuple((index + 1) / 5 for index in range(parameter_count))
        arguments = '(' + ', '.join(map(str, parameters)) + ')' if parameters else ''
        for prefix, values, inverse in profiles:
            qubits = base_qubits + len(values)
            order = tuple(reversed(range(qubits)))
            controls, operands = order[:len(values)], order[len(values):]
            quantum_arguments = ', '.join(f'q[{wire}]' for wire in order)
            exponents = (-3, -1, 0, 1, 2, 3)
            if gate in ('s', 'sdg', 't', 'tdg', 'sx'):
                exponents += (-9, -8, -7, -5, -4, -2, 4, 5, 6, 7, 8, 9)
            for exponent in exponents:
                for basis in (None, 0, (1 << qubits) - 1):
                    preparation, state = prepared_state(qubits, basis)
                    expected = state
                    for _ in range(abs(exponent)):
                        expected = conditioned(expected, gate, parameters, controls, operands,
                                               values, inverse != (exponent < 0))
                    check_amplitudes(bra, f'qubit[{qubits}] q; ' + preparation
                        + f'{prefix}pow({exponent}) @ {gate}{arguments} {quantum_arguments};', expected)
                    checked += 1
            # Runtime parameter expressions, including U/cu phase corrections,
            # must be captured once and reused for all inverse repetitions.
            if parameters:
                preparation, state = prepared_state(qubits)
                expected = state
                for _ in range(3):
                    expected = conditioned(expected, gate, parameters, controls, operands, values, not inverse)
                arguments_runtime = '(' + ', '.join(f'theta + {index / 5}'
                                                    for index in range(parameter_count)) + ')'
                check_amplitudes(bra, f'qubit[{qubits}] q; float theta = 0.2; ' + preparation
                    + f'{prefix}pow(-3) @ {gate}{arguments_runtime} {quantum_arguments};', expected)
                checked += 1

    preparation, state = prepared_state(3)
    for prefix, values, inverse in profiles:
        controls = (2, 0)[:len(values)]
        operands = ', '.join(f'q[{wire}]' for wire in controls)
        for exponent in (-3, 0, 2):
            expected = phase_reference(state, controls, values, (-1 if inverse else 1) * exponent * 0.7)
            check_amplitudes(bra, 'qubit[3] q; float theta = 0.7; ' + preparation
                + f'{prefix}pow({exponent}) @ gphase(theta) {operands};', expected)
            checked += 1

    # Nested powers and mixed-control broadcasting exercise modifier order
    # independently of the single-gate reference cases above.
    preparation, state = prepared_state(4)
    expected = state
    for target, control in ((0, 3), (1, 2)):
        for _ in range(6):
            expected = conditioned(expected, 'U', (0.2, 0.4, 0.6), (control,), (target,), (0,), True)
    check_amplitudes(bra, 'qubit[4] q; ' + preparation
        + 'negctrl @ pow(2) @ inv @ pow(3) @ U(0.2, 0.4, 0.6) q[{3,2}], q[0:1];', expected)
    checked += 1

    # Reached loop entries must recompute runtime parameters, while repeated
    # gates within one entry reuse that entry's snapshots.
    preparation, state = prepared_state(3)
    expected = state
    for theta in (0.2, 0.3):
        for _ in range(2):
            expected = conditioned(expected, 'U', (theta, theta + 1, theta + 2),
                                   (2,), (0,), (0,), True)
    for header in ('for int i in [0:1]', 'for int i in [0:stop]',
                   'for int i in {0, stop}', 'while (visits < 2)'):
        check_amplitudes(bra, 'qubit[3] q; float theta = 0.2; int stop = 1; int visits = 0; '
            + preparation + header + ' { negctrl @ pow(-2) @ U(theta, theta + 1, theta + 2) q[2], q[0]; '
            'theta += 0.1; visits += 1; }', expected)
        checked += 1
    check_amplitudes(bra, 'qubit[3] q; array[float, 1] angles = {0.2}; int bad = 9; '
        + preparation + 'if (false) { pow(3) @ rx(angles[bad]) q[0]; }', state)
    checked += 1

    # Rounded/overflowing scaled angles must retain the original applications,
    # for literal and runtime-valued parameters alike.
    preparation, state = prepared_state(1)
    for gate in ('p', 'rx', 'gphase'):
        for angle, exponent in ((1e308, 2), (1e100, 3)):
            expected = state
            for _ in range(exponent):
                expected = (phase_reference(expected, (), (), angle) if gate == 'gphase'
                            else conditioned(expected, gate, (angle,), (), (0,), ()))
            for expression in (str(angle), 'theta'):
                operands = '' if gate == 'gphase' else 'q[0]'
                check_amplitudes(bra, f'qubit q; float theta = {angle}; ' + preparation
                    + f'pow({exponent}) @ {gate}({expression}) {operands};', expected)
                checked += 1
    print(f'{checked} integer-power modifier amplitude comparisons passed')


if __name__ == '__main__':
    main()
