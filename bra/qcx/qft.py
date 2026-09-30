import argparse
import math
import sys

def write_swapped_fourier_transform(qubits: list[int], file) -> None:
    num_qubits: int = len(qubits)

    for index in range(num_qubits):
        target_qubit_index: int = num_qubits - index - 1
        print('H', qubits[target_qubit_index], file=file)

        for phase_exponent in range(2, num_qubits - index + 1):
            control_qubit_index: int = target_qubit_index - (phase_exponent - 1)
            print('U', qubits[control_qubit_index], qubits[target_qubit_index], phase_exponent, file=file)

def write_inversed_swapped_fourier_transform(qubits: list[int], file) -> None:
    num_qubits: int = len(qubits)

    for index in range(num_qubits):
        target_qubit_index: int = index

        for phase_exponent in range(index + 1, 1, -1):
            control_qubit_index: int = target_qubit_index - (phase_exponent - 1)
            print('U', qubits[control_qubit_index], qubits[target_qubit_index], -phase_exponent, file=file)

        print('H', qubits[target_qubit_index], file=file)

def write_communication_oriented_fourier_transform(num_qubits: int, num_global_qubits: int, is_fused: bool, file) -> None:
    if num_global_qubits != 2:
        sys.exit('communication-oriented QFT currently requires two global qubits')
    if num_qubits < 4:
        sys.exit('communication-oriented QFT requires at least four qubits')

    if is_fused:
        print('BEGIN FUSION', file=file)

    for target_qubit in range(num_qubits - 1, 1, -1):
        print('H', target_qubit, file=file)
        for control_qubit in range(target_qubit - 1, -1, -1):
            phase = math.pi / (1 << (target_qubit - control_qubit))
            print('CU1', target_qubit, control_qubit, format(phase, '.17g'), file=file)

    for qubit in range(num_qubits // 2 - 1, 1, -1):
        print('SWAP', qubit, num_qubits - qubit - 1, file=file)

    if is_fused:
        print('END FUSION', file=file)
        print('BEGIN FUSION', file=file)

    print('H', 1, file=file)
    print('CU1', 1, 0, format(math.pi / 2, '.17g'), file=file)
    print('SWAP', 1, num_qubits - 2, file=file)
    print('H', 0, file=file)
    print('SWAP', 0, num_qubits - 1, file=file)

    if is_fused:
        print('END FUSION', file=file)

def main(num_qubits: int, is_swapped: bool, is_inversed: bool, adds_measurement: bool, adds_amplitudes: bool,
         bit_assignment: str, uses_descending_bit_assignment: bool, num_global_qubits: int,
         is_fused: bool, initial_state: int, file) -> None:
    if uses_descending_bit_assignment:
        bit_assignment = ' '.join(str(qubit) for qubit in range(num_qubits - 1, -1, -1))
    if bit_assignment and len(bit_assignment.split()) != num_qubits:
        sys.exit('wrong bit assignment')
    if initial_state is not None and (initial_state < 0 or initial_state >= (1 << num_qubits)):
        sys.exit('wrong initial state')
    if num_global_qubits is not None and num_global_qubits != 2:
        sys.exit('communication-oriented QFT currently requires two global qubits')
    if is_fused and num_global_qubits is None:
        sys.exit('--fused requires --num-global-qubits')
    if num_global_qubits is not None and (is_swapped or is_inversed):
        sys.exit('--num-global-qubits cannot be used with --swapped or --inversed')

    print('QUBITS', num_qubits, file=file)
    if initial_state is not None:
        print('INITIAL STATE', initial_state, file=file)
    if bit_assignment:
        print('BIT ASSIGNMENT', bit_assignment, file=file)

    if num_global_qubits is not None:
        write_communication_oriented_fourier_transform(num_qubits, num_global_qubits, is_fused, file)
    elif is_inversed:
        write_inversed_swapped_fourier_transform(range(num_qubits), file)
    else:
        write_swapped_fourier_transform(range(num_qubits), file)

    if num_global_qubits is None and not is_swapped:
        for qubit in range(num_qubits // 2):
            print('SWAP', qubit, num_qubits - qubit -1, file=file)

    if adds_measurement:
        print('DO MEASUREMENT', file=file)

    if adds_amplitudes:
        print('DO AMPLITUDES', file=file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate quantum circuit to perform quantum Fourier transform')
    parser.add_argument('num_qubits', type=int, help='the number of qubits (>=1)')
    parser.add_argument('-s', '--swapped', action='store_true', help='swapped QFT')
    parser.add_argument('-i', '--inversed', action='store_true', help='inversed QFT')
    parser.add_argument('-m', '--measure', action='store_true', help='add measurement operation after the other operations')
    parser.add_argument('-a', '--amplitudes', action='store_true', help='add amplitudes operation after the other operations')
    bit_assignment_group = parser.add_mutually_exclusive_group()
    bit_assignment_group.add_argument('-b', '--bitassign', type=str, help='add bit assignment operation')
    bit_assignment_group.add_argument('--descending-bitassign', action='store_true',
                                      help='assign logical qubits to physical qubits in descending order')
    parser.add_argument('-g', '--num-global-qubits', type=int,
                        help='order QFT for communication-oriented gate fusion (currently must be 2)')
    parser.add_argument('-f', '--fused', action='store_true', help='add communication-oriented gate-fusion blocks')
    parser.add_argument('--initial-state', type=int, help='initial computational-basis state')
    parser.add_argument('-o', '--output', type=str, help='output filename (default: stdout)')
    args = parser.parse_args()

    if args.output:
        with open(args.output, mode='w') as file:
            main(args.num_qubits, args.swapped, args.inversed, args.measure, args.amplitudes,
                 args.bitassign, args.descending_bitassign, args.num_global_qubits, args.fused,
                 args.initial_state, file)
    else:
        main(args.num_qubits, args.swapped, args.inversed, args.measure, args.amplitudes,
             args.bitassign, args.descending_bitassign, args.num_global_qubits, args.fused,
             args.initial_state, sys.stdout)
