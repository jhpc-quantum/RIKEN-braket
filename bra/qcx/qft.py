import sys
import argparse

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

def main(num_qubits: int, is_swapped: bool, is_inversed: bool, adds_measurement: bool, adds_amplitudes: bool, bit_assignment: str, file) -> None:
    if bit_assignment and len(bit_assignment.split()) != num_qubits:
        sys.exit('wrong bit assignment')

    print('QUBITS', num_qubits, file=file)
    if bit_assignment:
        print('BIT ASSIGNMENT', bit_assignment, file=file)

    if is_inversed:
        write_inversed_swapped_fourier_transform(range(num_qubits), file)
    else:
        write_swapped_fourier_transform(range(num_qubits), file)

    if not is_swapped:
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
    parser.add_argument('-b', '--bitassign', type=str, help='add bit assignment operation')
    parser.add_argument('-o', '--output', type=str, help='output filename (default: stdout)')
    args = parser.parse_args()

    if args.output:
        with open(args.output, mode='w') as file:
            main(args.num_qubits, args.swapped, args.inversed, args.measure, args.amplitudes, args.bitassign, file)
    else:
        main(args.num_qubits, args.swapped, args.inversed, args.measure, args.amplitudes, args.bitassign, sys.stdout)

