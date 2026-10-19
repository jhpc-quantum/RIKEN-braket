#!/usr/bin/env python3

"""Check native/converted control values across distributed bra execution paths.

  python3 bra/test/negative_controls_mpi_numerical.py --bra /path/to/mpi-bra
  python3 bra/test/negative_controls_mpi_numerical.py --bra /path/to/mpi-bra \
      --ranks 4 --mode unit --page-qubits 2 --cache-qubits 3

Requires an MPI build. Each invocation checks normal and fused circuits with
identity and nontrivial initial permutations. Ten wires leave room for five
operated wires even with four ranks and two page qubits. Small fusion caches
exercise cache-aware dispatch when enabled in the executable.
"""

import argparse
import itertools
import pathlib
import resource
import subprocess
import tempfile

from negative_controls_numerical import apply, check_result, operator_matrix
from qasm2qcx_negative_control_numerical import conditioned, phase_reference
from qasm2qcx_runtime_for_numerical import qasm2qcx


WIRES = 10
ORDER = (9, 0, 7, 3, 5, 1, 8, 2, 6, 4)


def prefix(count):
    return '' if count == 0 else 'C' if count == 1 else f'C{count}'


def preparation():
    state = [1 + 0j] + [0j] * ((1 << WIRES) - 1)
    text = ''
    for wire in range(WIRES):
        parameters = (0.3 + wire / 13, 0.1 + wire / 17, -0.2)
        text += f'U3 {wire} ' + ' '.join(map(str, parameters)) + '\n'
        state = apply(state, operator_matrix('U3', (wire,), parameters), (), (wire,), ())
    return text, state


def native_blocks(state):
    cases = []
    for count in (1, 2, 3):
        for gate, parameters in (
                ('H', ()), ('X', ()), ('Y', ()), ('SX', ()), ('SX+', ()),
                ('SY', ()), ('SY+', ()), ('+X', ()), ('-X', ()),
                ('+Y', ()), ('-Y', ()), ('U2', (0.2, -0.3)),
                ('U2+', (0.2, -0.3)), ('U3', (0.4, 0.2, -0.3)),
                ('U3+', (0.4, 0.2, -0.3)), ('EX', (0.3,)), ('EY+', (0.3,)),
                ('EZ', (0.3,)), ('I', ())):
            cases.append((prefix(count) + ('I1' if gate == 'I' else gate), gate, count, 1, parameters))
        for gate, parameters in (('SWAP', ()), ('ESWAP+', (0.3,)),
                                 ('X', ()), ('Y', ()), ('Z', ()), ('SZ+', ()),
                                 ('EX', (0.3,)), ('EY+', (0.3,)), ('EZ', (0.3,))):
            root = gate if 'SWAP' in gate else gate.rstrip('+') + '2' + ('+' if gate.endswith('+') else '')
            cases.append((prefix(count) + root, gate, count, 2, parameters))
    for count in (1, 2, 3):
        for gate, parameters in (('Z', ()), ('SZ+', ()), ('S', ()), ('S+', ()),
                                 ('T', ()), ('T+', ()), ('U1', (0.3,)), ('U1+', (0.3,)),
                                 ('R', (3,)), ('R+', (3,)), ('IC', ())):
            cases.append((prefix(count - 1) + gate, gate, count, 0, parameters))
    cases.extend([('TOFFOLI', 'X', 2, 1, ()), ('CCX', 'X', 2, 1, ()),
                  ('CCXX', 'X', 2, 2, ()), ('U+', 'R+', 2, 0, (3,))])
    blocks = []
    for block in range(3):
        pool = tuple(ORDER[(block * 3 + index) % WIRES] for index in range(5))
        text = ''
        for index, (mnemonic, gate, count, target_count, parameters) in enumerate(cases):
            controls = pool[:count]
            targets = pool[5 - target_count:] if target_count else ()
            # Vary all control sectors, including adjacent gates that require
            # conflicting polarities on the same wire inside one fusion block.
            values = tuple(((index + block) >> bit) & 1 for bit in range(count))
            operands = [f'{wire}#{value}' for wire, value in zip(controls, values)]
            text += mnemonic + ' ' + ' '.join(operands + list(map(str, (*targets, *parameters)))) + '\n'
            state = apply(state, operator_matrix(gate, targets, parameters), controls, targets, values)
        blocks.append(text)
    # Keep one wire positive-only within each block, so MPI can factor it
    # into rank/unit filtering while zero-controls remain inside the block.
    # Rotate that wire through every logical position to exercise both kinds
    # of distributed control after the preceding interchange operations.
    for positive in range(WIRES):
        negative, first, second = ((positive + offset) % WIRES for offset in (3, 7, 1))
        text = ''
        for mnemonic, gate, targets, values, parameters in (
                ('C2H', 'H', (first,), (0, 1), ()),
                ('CU1', 'U1', (), (0, 1), (0.37,)),
                ('C2X', 'X', (second,), (0, 1), ()),
                ('CU1', 'U1', (), (1, 1), (0.22,))):
            controls = (negative, positive)
            operands = [f'{wire}#{value}' for wire, value in zip(controls, values)]
            text += mnemonic + ' ' + ' '.join(operands + list(map(str, (*targets, *parameters)))) + '\n'
            state = apply(state, operator_matrix(gate, targets, parameters), controls, targets, values)
        blocks.append(text)
    return blocks, state


def converted_blocks(state):
    blocks = []
    for block in range(3):
        pool = tuple(ORDER[(block * 3 + index) % WIRES] for index in range(5))
        source = ''
        for index, (gate, (parameter_count, qubits)) in enumerate(
                qasm2qcx.QASM2QCXConverter.gate_signatures.items()):
            parameters = tuple((index + bit + 1) / 19 for bit in range(parameter_count))
            controls, targets = pool[:2], pool[2:2 + qubits]
            values = (0, 1) if index % 2 else (1, 0)
            modifiers = 'negctrl @ inv @ ctrl' if index % 2 else 'ctrl @ negctrl @ inv'
            arguments = '(' + ', '.join(map(str, parameters)) + ')' if parameters else ''
            operands = ', '.join(f'q[{wire}]' for wire in controls + targets)
            source += f'{modifiers} @ {gate}{arguments} {operands};\n'
            state = conditioned(state, gate, parameters, controls, targets, values, True)
        controls, values = pool[:3], (0, 1, 0)
        source += ('inv @ negctrl @ ctrl @ negctrl @ gphase(0.7) '
                   + ', '.join(f'q[{wire}]' for wire in controls) + ';')
        state = phase_reference(state, controls, values, -0.7)
        lines = qasm2qcx.convert(f'OPENQASM 3; include "stdgates.inc"; qubit[{WIRES}] q; ' + source)
        blocks.append('\n'.join(lines[1:]) + '\n')
    return blocks, state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    parser.add_argument('--mpiexec', default='mpiexec')
    parser.add_argument('--ranks', type=int, choices=(2, 4), default=2)
    parser.add_argument('--mode', choices=('simple', 'unit'), default='simple')
    parser.add_argument('--page-qubits', type=int, choices=(0, 2), default=2)
    parser.add_argument('--cache-qubits', type=int, default=3)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    command = [args.mpiexec, '-n', str(args.ranks), str(args.bra.resolve()),
               '--mode', args.mode, '--page-qubits', str(args.page_qubits),
               '--num-cache-qubits', str(args.cache_qubits), '--threads', str(args.threads)]
    if args.mode == 'unit':
        command += ['--unit-qubits', '1', '--unit-processes', '2']
    text, initial = preparation()
    circuits = [('native', *native_blocks(initial)), ('converted', *converted_blocks(initial))]
    permutations = [tuple(range(WIRES)), (9, 2, 7, 0, 5, 8, 1, 6, 4, 3)]
    checked = 0
    # mpiexec normally forwards stdin only to rank zero; bra parses its input
    # independently on every rank, so all ranks must open the same QCX file.
    with tempfile.TemporaryDirectory(prefix='bra-negative-controls-mpi-') as directory:
        path = pathlib.Path(directory) / 'circuit.qcx'
        for (name, blocks, expected), permutation, fused in itertools.product(circuits, permutations, (False, True)):
            program = f'QUBITS {WIRES}\nBIT ASSIGNMENT ' + ' '.join(map(str, permutation)) + '\n' + text
            program += ''.join('BEGIN FUSION\n' + body + 'END FUSION\n' if fused else body for body in blocks)
            path.write_text(program + 'DO AMPLITUDES\n', encoding='utf-8')
            result = subprocess.run(command + ['--file', str(path)],
                                    text=True, capture_output=True, timeout=60)
            check_result(result, program, expected)
            checked += 1
            print(f'{name}: permutation={permutation != permutations[0]}, fused={fused} passed', flush=True)
    print(f'{checked} MPI matrix-reference circuits passed: ranks={args.ranks}, mode={args.mode}, '
          f'pages={args.page_qubits}, cache={args.cache_qubits}, threads={args.threads}')


if __name__ == '__main__':
    main()
