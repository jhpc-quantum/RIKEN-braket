#!/usr/bin/env python3

"""Check runtime-integer gate powers against independent matrix references.

  python3 bra/test/qasm2qcx_runtime_power_numerical.py --bra /path/to/bra
"""

import argparse
import pathlib
import resource
import subprocess

from qasm2qcx_control_numerical import prepared_state
from qasm2qcx_integer_array_numerical import check_error
from qasm2qcx_inverse_numerical import check_amplitudes
from qasm2qcx_negative_control_numerical import conditioned, phase_reference
from qasm2qcx_runtime_for_numerical import check_program, qasm2qcx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    bra = parser.parse_args().bra.resolve()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    profiles = [('', (), False), ('ctrl @ ', (1,), False), ('negctrl @ ', (0,), False),
                ('negctrl @ inv @ ctrl @ ', (0, 1), True)]
    checked = 0
    for gate, (parameter_count, base_qubits) in qasm2qcx.QASM2QCXConverter.gate_signatures.items():
        parameters = tuple((index + 1) / 5 for index in range(parameter_count))
        arguments = '(' + ', '.join(map(str, parameters)) + ')' if parameters else ''
        for prefix, values, inverse in profiles:
            qubits = base_qubits + len(values)
            order = tuple(reversed(range(qubits)))
            controls, operands = order[:len(values)], order[len(values):]
            quantum_arguments = ', '.join(f'q[{wire}]' for wire in order)
            for value_type, exponent in [('int', n) for n in (-3, -1, 0, 1, 2, 3)] + [('uint[3]', n) for n in (0, 1, 3)]:
                for basis in (None, 0, (1 << qubits) - 1):
                    preparation, state = prepared_state(qubits, basis)
                    expected = state
                    for _ in range(abs(exponent)):
                        expected = conditioned(expected, gate, parameters, controls, operands,
                                               values, inverse != (exponent < 0))
                    check_amplitudes(bra, f'qubit[{qubits}] q; {value_type} n = {exponent}; ' + preparation
                        + f'{prefix}pow(n) @ {gate}{arguments} {quantum_arguments};', expected)
                    checked += 1

    preparation, state = prepared_state(3)
    for prefix, values, inverse in profiles:
        operands = ', '.join(f'q[{wire}]' for wire in (2, 0)[:len(values)])
        for exponent in (-3, 0, 2):
            expected = phase_reference(state, (2, 0)[:len(values)], values,
                                       (-1 if inverse else 1) * exponent * 0.7)
            check_amplitudes(bra, f'qubit[3] q; int n = {exponent}; float theta = 0.7; '
                + preparation + f'{prefix}pow(n) @ gphase(theta) {operands};', expected)
            checked += 1

    preparation, state = prepared_state(4)
    expected = state
    for control, target in ((3, 0), (2, 1)):
        for _ in range(2):
            expected = conditioned(expected, 'U', (0.2, 0.4, 0.6), (control,), (target,), (0,), True)
    check_amplitudes(bra, 'qubit[4] q; int n = -2; int index = 0; array[int, 1] counts = {n}; '
        'array[float, 3] angles = {0.2, 0.4, 0.6}; ' + preparation
        + 'negctrl @ pow(counts[index]) @ U(angles[index], angles[index + 1], angles[index + 2]) q[{3,2}], q[0:1];', expected)
    checked += 1

    preparation, state = prepared_state(3)
    for header in ('for int n in [-2:2]', 'for int n in [-2:stop]', 'for int n in {-2, -1, 0, 1, 2}'):
        expected = state
        for n in (-2, -1, 0, 1, 2):
            for _ in range(abs(n)):
                expected = conditioned(expected, 'U', (0.2, 0.4, 0.6), (2,), (0,), (0,), n < 0)
        check_amplitudes(bra, 'qubit[3] q; int stop = 2; float theta = 0.2; ' + preparation
            + header + ' { negctrl @ pow(n) @ U(theta, 0.4, 0.6) q[2], q[0]; }', expected)
        checked += 1
    check_amplitudes(bra, 'qubit[3] q; int n = 1; int bad = 9; array[float, 1] angles = {0.2}; '
        + preparation + 'if (false) { pow(n / 0) @ U(angles[bad], 0, 0) q[0]; }', state)
    checked += 1
    expected = state
    for _ in range(4):
        expected = conditioned(expected, 'U', (0.2, 0.4, 0.6), (), (0,), ())
    check_amplitudes(bra, 'qubit[3] q; array[uint[3], 3] counts = {0, 1, 3}; ' + preparation
        + 'for uint[3] n in counts { pow(n) @ U(0.2, 0.4, 0.6) q[0]; }', expected)
    checked += 1
    check_program(bra, 'qubit q; int n = -2; float theta = 0.2; pow(n) @ U(theta, 0.4, 0.6) q;',
                  ('N1', 'THETA31'), ['-2', '0.2'])
    checked += 1

    # Verify native endpoint magnitudes directly, without executing billions
    # of gate repetitions. The same normalization helper feeds public loops.
    limits = qasm2qcx.QASM2QCXConverter
    for value_type, value in (('int', limits.QCX_INT_MIN), ('int', -1), ('int', 0),
                              ('int', limits.QCX_INT_MAX), ('uint', limits.QCX_UINT_MAX)):
        program = qasm2qcx.openqasm3.parser.parse(f'OPENQASM 3; {value_type} n = {value};')
        converter = qasm2qcx.QASM2QCXConverter(program)
        converter.visit(program)
        modifiers = qasm2qcx.openqasm3.parser.parse('OPENQASM 3; pow(n) @ gphase(0.5);').statements[0].modifiers
        info = converter._QASM2QCXConverter__analyze_gate_modifiers(modifiers, allow_runtime=True)
        with converter._QASM2QCXConverter__runtime_gate_power(info) as runtime:
            converter._QASM2QCXConverter__qcx_lines.append(f'PRINTLN {runtime.magnitudes[0]}')
        result = subprocess.run([str(bra)], input='\n'.join(converter) + '\n', text=True,
                                capture_output=True, check=True, timeout=30)
        if result.stdout.strip() != str(abs(value)):
            raise AssertionError(f'Wrong magnitude for {value_type} {value}: {result.stdout}')
        checked += 1
    check_error(bra, 'qubit q; int n = 0; array[float, 1] angles = {0.2}; int bad = 9; '
                'pow(n) @ U(angles[bad], 0, 0) q;', 'assertion failed in ASSERT')

    for gate in ('U', 'cu', 't', 'sx', 'rz', 'gphase'):
        parameter_count, base_qubits = ((1, 0) if gate == 'gphase'
                                        else limits.gate_signatures[gate])
        parameters = tuple((index + 1) / 5 for index in range(parameter_count))
        arguments = '(' + ', '.join(map(str, parameters)) + ')' if parameters else ''
        for prefix, values, inverse in profiles:
            qubits = max(1, base_qubits + len(values))
            order = tuple(reversed(range(base_qubits + len(values))))
            controls, operands = order[:len(values)], order[len(values):]
            quantum_arguments = ', '.join(f'q[{wire}]' for wire in order)
            preparation, state = prepared_state(qubits)
            for n, m in ((-2, -3), (-2, 3), (2, -3), (0, 3), (3, 0), (2, 3)):
                exponent = n * m
                expected = state
                if gate == 'gphase':
                    expected = phase_reference(state, controls, values, (-1 if inverse else 1) * exponent * parameters[0])
                else:
                    for _ in range(abs(exponent)):
                        expected = conditioned(expected, gate, parameters, controls, operands,
                                               values, inverse != (exponent < 0))
                m_type = 'uint[3]' if m >= 0 else 'int'
                check_amplitudes(bra, f'qubit[{qubits}] q; int n = {n}; {m_type} m = {m}; '
                    + preparation + f'{prefix}pow(n) @ pow(m) @ {gate}{arguments} {quantum_arguments};', expected)
                checked += 1

    # Periodic reductions must work at native boundaries without executing
    # billions of gates, or multiplying full-width counts in QCX.
    for gate, period in (('x', 2), ('s', 4), ('t', 8), ('sx', 4), ('cx', 2), ('ccx', 2)):
        base_qubits = limits.gate_signatures[gate][1]
        qubits = base_qubits + 1
        order = tuple(reversed(range(qubits)))
        controls, operands = order[:1], order[1:]
        preparation, state = prepared_state(qubits)
        for n in (limits.QCX_INT_MIN, -1, limits.QCX_INT_MAX):
            expected = state
            for _ in range((abs(n) * limits.QCX_UINT_MAX) % period):
                expected = conditioned(expected, gate, (), controls, operands, (0,), n < 0)
            quantum_arguments = ', '.join(f'q[{wire}]' for wire in order)
            check_amplitudes(bra, f'qubit[{qubits}] q; int n = {n}; uint m = {limits.QCX_UINT_MAX}; '
                + preparation + f'negctrl @ pow(n) @ pow(m) @ {gate} {quantum_arguments};', expected)
            checked += 1

    preparation, state = prepared_state(1)
    for factors in (f'pow(large) @ pow(zero)', f'pow(zero) @ pow(large)',
                    f'pow({1 << (limits.QCX_UINT_WIDTH // 2)}) @ pow(zero) @ pow({1 << (limits.QCX_UINT_WIDTH // 2)})'):
        check_amplitudes(bra, f'qubit q; uint large = {limits.QCX_UINT_MAX}; int zero = 0; '
            + preparation + factors + ' @ U(0.2, 0.4, 0.6) q;', state)
        checked += 1
    check_error(bra, 'qubit q; int zero = 0; array[int, 1] counts = {2}; int bad = 9; '
                'pow(zero) @ pow(counts[bad]) @ U(1, 2, 3) q;', 'assertion failed in ASSERT')
    for gate in ('U', 'cu', 'gphase'):
        parameter_count, base_qubits = ((1, 0) if gate == 'gphase' else limits.gate_signatures[gate])
        parameters = tuple((index + 1) / 5 for index in range(parameter_count))
        arguments = '(' + ', '.join(map(str, parameters)) + ')'
        qubits = base_qubits + 1
        preparation, state = prepared_state(qubits)
        controls, operands = (qubits - 1,), tuple(range(base_qubits))
        quantum_arguments = ', '.join(f'q[{wire}]' for wire in controls + operands)
        for modifiers, exponent in (('pow(-2) @ pow(n) @ pow(m)', 12),
                                    ('pow(n) @ pow(m) @ pow(k)', 6)):
            expected = state
            if gate == 'gphase':
                expected = phase_reference(state, controls, (0,), exponent * parameters[0])
            else:
                for _ in range(exponent):
                    expected = conditioned(expected, gate, parameters, controls, operands, (0,))
            check_amplitudes(bra, f'qubit[{qubits}] q; int n = -2; int m = 3; int k = -1; '
                + preparation + f'negctrl @ {modifiers} @ {gate}{arguments} {quantum_arguments};', expected)
            checked += 1
    print(f'{checked} runtime-integer gate power checks passed')


if __name__ == '__main__':
    main()
