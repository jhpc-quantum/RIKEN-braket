import sys
import argparse
import qft

def write_addition(result_register_index: int, register_index: int, register_size: int, file) -> None:
    if register_size <= 0:
        sys.exit('wrong register size')
    if result_register_index < 0 or register_index < 0:
        sys.exit('wrong register index')
    if result_register_index == register_index:
        sys.exit('different registers should be specified')

    result_qubits: list[int] = [qubit_in_register + register_size * result_register_index for qubit_in_register in range(register_size)]
    qubits: list[int] = [qubit_in_register + register_size * register_index for qubit_in_register in range(register_size)]

    qft.write_swapped_fourier_transform(result_qubits, file)

    for phase_exponent in range(1, register_size + 1):
        for index in range(register_size - (phase_exponent - 1)):
            target_qubit_index: int = register_size - index - 1
            control_qubit_index: int = register_size - phase_exponent - index # (register_size - (phase_exponent - 1)) - index - 1
            print('U', qubits[control_qubit_index], result_qubits[target_qubit_index], phase_exponent, file=file)

    qft.write_inversed_swapped_fourier_transform(result_qubits, file)

def write_subtraction(result_register_index: int, register_index: int, register_size: int, file) -> None:
    if register_size <= 0:
        sys.exit('wrong register size')
    if result_register_index < 0 or register_index < 0:
        sys.exit('wrong register index')
    if result_register_index == register_index:
        sys.exit('different registers should be specified')

    result_qubits: list[int] = [qubit_in_register + register_size * result_register_index for qubit_in_register in range(register_size)]
    qubits: list[int] = [qubit_in_register + register_size * register_index for qubit_in_register in range(register_size)]

    qft.write_swapped_fourier_transform(result_qubits, file)

    for phase_exponent in range(register_size, 0, -1):
        for index in range(register_size - (phase_exponent - 1)):
            target_qubit_index: int = register_size - index - 1
            control_qubit_index: int = register_size - phase_exponent - index # (register_size - (phase_exponent - 1)) - index - 1
            print('U', qubits[control_qubit_index], result_qubits[target_qubit_index], -phase_exponent, file=file)

    qft.write_inversed_swapped_fourier_transform(result_qubits, file)

def write_assignment(assignment_register_index: int, assignment_value: int, register_size: int, file) -> None:
    if register_size <= 0:
        sys.exit('wrong register size')
    if assignment_register_index < 0:
        sys.exit('wrong register index')
    if assignment_value < 0:
        sys.exit('wrong value')

    assignment_qubits: list[int] = [qubit_in_register + register_size * assignment_register_index for qubit_in_register in range(register_size)]
    gates: list[str] = ["+X", "-X", "+Y", "-Y"]
    gate_index = 0

    for index in range(register_size):
        if (assignment_value >> index) & 0b1 == 0b1:
            print(gates[gate_index], assignment_qubits[index], file=file)
            print(gates[gate_index], assignment_qubits[index], file=file)
            gate_index += 1
            if gate_index >= len(gates):
                gate_index = 0

def main(register_size: int, num_registers: int, expression: str, adds_measurement: bool, adds_amplitudes: bool, bit_assignment: str, file) -> None:
    num_qubits: int = num_registers * register_size
    if bit_assignment and len(bit_assignment.split()) != num_qubits:
        sys.exit('wrong bit assignment')

    print('QUBITS', num_qubits, file=file)
    if bit_assignment:
        print('BIT ASSIGNMENT', bit_assignment, file=file)

    for operation in expression.split(';'):
        if operation.strip() == '':
            continue

        register_strs: list[str] = []
        is_assignment: bool = False
        is_addition: bool = True
        if '+=' in operation:
            register_strs = [register_str.strip() for register_str in operation.split('+=')]
        elif '-=' in operation:
            register_strs = [register_str.strip() for register_str in operation.split('-=')]
            is_addition = False
        elif '=' in operation:
            register_strs = [register_str.strip() for register_str in operation.split('=')]
            is_assignment = True

        if len(register_strs) != 2:
            sys.exit('wrong expression')

        if is_assignment:
            if not register_strs[0].startswith('@'):
                sys.exit('wrong expression')
            assignment_register_index: int = int(register_strs[0][1:])
            assignment_value: int = int(register_strs[1])

            if assignment_register_index >= num_registers:
                sys.exit('wrong expression')

            write_assignment(assignment_register_index, assignment_value, register_size, file)
        else:
            if not register_strs[0].startswith('@'):
                sys.exit('wrong expression')
            result_register_index: int = int(register_strs[0][1:])
            if not register_strs[1].startswith('@'):
                sys.exit('wrong expression')
            register_index: int = int(register_strs[1][1:])

            if result_register_index >= num_registers or register_index >= num_registers:
                sys.exit('wrong expression')

            if is_addition:
                write_addition(result_register_index, register_index, register_size, file)
            else:
                write_subtraction(result_register_index, register_index, register_size, file)

    if adds_measurement:
        print('DO MEASUREMENT', file=file)

    if adds_amplitudes:
        print('DO AMPLITUDES', file=file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate quantum circuit to perform additions and subtractions of integers')
    parser.add_argument('register_size', type=int, help='the size of registers (>=1)')
    parser.add_argument('num_registers', type=int, help='the number of registers (>=2)')
    parser.add_argument('expression', type=str, help='expression (e.g. "@0=7; @1=4; @2=6; @0+=@1; @0-=@2")')
    parser.add_argument('-m', '--measure', action='store_true', help='add measurement operation after the other operations')
    parser.add_argument('-a', '--amplitudes', action='store_true', help='add amplitudes operation after the other operations')
    parser.add_argument('-b', '--bitassign', type=str, help='add bit assignment operation')
    parser.add_argument('-o', '--output', type=str, help='output filename (default: stdout)')
    args = parser.parse_args()

    if args.output:
        with open(args.output, mode='w') as file:
            main(args.register_size, args.num_registers, args.expression, args.measure, args.amplitudes, args.bitassign, file)
    else:
        main(args.register_size, args.num_registers, args.expression, args.measure, args.amplitudes, args.bitassign, sys.stdout)

