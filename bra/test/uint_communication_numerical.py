#!/usr/bin/env python3

"""Check UINT communication through the CLI scheduler and MPI backend.

Examples:
  python3 bra/test/uint_communication_numerical.py --bra bra/bin/bra
  python3 bra/test/uint_communication_numerical.py --bra /path/to/mpi/bra --mpiexec mpiexec --mpi-processes 3
"""

import argparse
import ctypes
import pathlib
import resource
import subprocess
import tempfile


def program(count: int, root: int, maximum: int) -> str:
    # MPI needs the circuit count before computing the qubit distribution.
    lines = ['CIRCUITS 3', 'QUBITS 2']
    values = [[maximum - circuit, (maximum >> 1) + 1 + circuit][:count]
              for circuit in range(3)]
    count_argument = '' if count == 1 else f' {count}'
    for circuit in range(3):
        lines.extend([f'BEGIN CIRCUIT {circuit}', 'VAR A UINT 3', 'VAR B UINT 4',
                      'VAR G UINT 8', 'VAR DEST UINT 8', 'VAR INDEX INT', 'LET INDEX := 1'])
        for offset, value in enumerate(values[circuit]):
            lines.extend([f'LET A:{1 + offset} := {value}', f'LET G:{1 + offset} := {value}'])
        # MPI in-place gather keeps the root's contribution in its own block.
        if circuit == root:
            for offset, value in enumerate(values[root]):
                lines.append(f'LET G:{1 + count * root + offset} := {value}')
        if circuit == 0:
            lines.append(f'SEND 2 A:INDEX UINT{count_argument}')
        elif circuit == 2:
            lines.append(f'RECEIVE 0 B:2 UINT{count_argument}')
        # Self-transfers must not wait or alter values.
        lines.extend([f'SEND {circuit} A UINT', f'RECEIVE {circuit} A UINT',
                      f'BROADCAST 2 B:2 UINT{count_argument}'])
        for offset, value in enumerate(values[0]):
            lines.append(f'ASSERT B:{2 + offset} == {value}')
        lines.append(f'GATHER {root} A:INDEX UINT{count_argument} TO DEST:1')
        if circuit == root:
            for peer in range(3):
                for offset, value in enumerate(values[peer]):
                    lines.append(f'ASSERT DEST:{1 + count * peer + offset} == {value}')
        else:
            lines.append('ASSERT DEST:1 == 0')
        lines.append(f'SCATTER {root} B:INDEX UINT{count_argument} FROM DEST:1')
        for offset, value in enumerate(values[circuit]):
            lines.append(f'ASSERT B:{1 + offset} == {value}')
        lines.append(f'GATHER {root} G:INDEX UINT{count_argument}')
        if circuit == root:
            for peer in range(3):
                for offset, value in enumerate(values[peer]):
                    lines.append(f'ASSERT G:{1 + count * peer + offset} == {value}')
        lines.append(f'SCATTER {root} G:INDEX UINT{count_argument}')
        expected = values[0] if circuit == root else values[circuit]
        for offset, value in enumerate(expected):
            lines.append(f'ASSERT G:{1 + offset} == {value}')
        # Signed communication continues to use a separate scheduler protocol.
        if circuit == 0:
            lines.extend(['LET INDEX := -1', 'SEND 1 INDEX INT'])
        elif circuit == 1:
            lines.extend(['RECEIVE 0 INDEX INT', 'ASSERT INDEX == -1'])
        lines.extend(['VAR SIGNED INT 3', f'LET SIGNED := {-circuit - 1}',
                      'BROADCAST 1 SIGNED INT', 'ASSERT SIGNED == -2',
                      'GATHER 0 SIGNED INT', 'SCATTER 0 SIGNED INT', 'ASSERT SIGNED == -2'])
        if circuit == 0:
            lines.append('PRINTLN 12345')
        lines.append('END CIRCUIT')
    return '\n'.join(lines) + '\n'


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    parser.add_argument('--mpiexec', help='MPI launcher; omit for the non-MPI backend')
    parser.add_argument('--mpi-processes', type=int, default=3)
    args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    if args.mpiexec and (args.mpi_processes < 3 or args.mpi_processes % 3 != 0):
        parser.error('--mpi-processes must be a positive multiple of three')
    command = [str(args.bra)]
    if args.mpiexec:
        command = [args.mpiexec, '-n', str(args.mpi_processes), *command, '--page-qubits', '0']
    maximum = (1 << (ctypes.sizeof(ctypes.c_uint) * 8)) - 1
    for count in (1, 2):
        for root in range(3):
            source = program(count, root, maximum)
            if args.mpiexec:
                # MPI launchers normally forward stdin only to rank zero;
                # bra's interpreter reads the program independently on each rank.
                with tempfile.TemporaryDirectory(prefix='bra-uint-communication-') as directory:
                    source_path = pathlib.Path(directory) / 'communication.qcx'
                    source_path.write_text(source)
                    result = subprocess.run([*command, '--file', str(source_path)],
                                            text=True, capture_output=True, timeout=60)
            else:
                result = subprocess.run(command, input=source, text=True, capture_output=True, timeout=60)
            if result.returncode != 0 or result.stdout.splitlines() != ['12345']:
                raise RuntimeError(f'UINT communication failed\nsource:\n{source}\n{result}')
    if not args.mpiexec:
        # Non-MPI errors avoid leaving other MPI ranks blocked in a collective.
        for instruction in ('BROADCAST 0 A:1 UINT 2', 'GATHER 0 A UINT 3',
                            'SCATTER 0 A UINT 3', 'BROADCAST 0 A UINT 0',
                            'BROADCAST 0 A UINT -1'):
            source = f'QUBITS 0\nVAR A UINT 2\n{instruction}\n'
            result = subprocess.run(command, input=source, text=True, capture_output=True, timeout=30)
            diagnostic = 'UINT communication count must be positive' if instruction.endswith((' 0', ' -1')) else 'UINT communication range out of bounds'
            if result.returncode == 0 or diagnostic not in result.stderr:
                raise RuntimeError(f'Invalid UINT communication accepted\n{source}\n{result}')
    print('Native UINT communication numerical tests passed')


if __name__ == '__main__':
    main()
