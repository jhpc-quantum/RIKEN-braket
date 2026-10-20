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
            for exponent in (-3, -1, 0, 1, 2, 3):
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
    print(f'{checked} integer-power modifier amplitude comparisons passed')


if __name__ == '__main__':
    main()
