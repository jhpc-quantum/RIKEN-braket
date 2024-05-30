import sys
import argparse

def main(num_qubits: int, is_in_descending_order: bool, adds_measurement: bool, bit_assignment: str, file) -> None:
    if bit_assignment and len(bit_assignment.split()) != num_qubits:
        sys.exit('wrong bit assignment')

    print('QUBITS', num_qubits, file=file)
    if bit_assignment:
        print('BIT ASSIGNMENT', bit_assignment, file=file)

    if not is_in_descending_order:
        for qubit in range(num_qubits):
            print('H', qubit, file=file)
    else:
        for qubit in range(num_qubits-1, -1, -1):
            print('H', qubit, file=file)

    if adds_measurement:
        print('BEGIN MEASUREMENT', file=file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate quantum circuit to operate Hadamard gate to every qubit in ascending order')
    parser.add_argument('num_qubits', type=int, help='the number of qubits (>=1)')
    parser.add_argument('-d', '--descend', action='store_true', help='in descending order')
    parser.add_argument('-m', '--measure', action='store_true', help='add measurement operation after the other operations')
    parser.add_argument('-b', '--bitassign', type=str, help='add bit assignment operation')
    parser.add_argument('-o', '--output', type=str, help='output filename (default: stdout)')
    args = parser.parse_args()

    if args.output:
        with open(args.output, mode='w') as file:
            main(args.num_qubits, args.descend, args.measure, args.bitassign, file)
    else:
        main(args.num_qubits, args.descend, args.measure, args.bitassign, sys.stdout)

