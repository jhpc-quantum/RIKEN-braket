#!/usr/bin/env python3
"""Build and run MPI control-value tests (requires MPI and YAMPI).

Examples:
  python3 ket/test/mpi/control_gate_numerical.py
  python3 ket/test/mpi/control_gate_numerical.py --variant variadic --basis 128
  python3 ket/test/mpi/control_gate_numerical.py --variant iterators --unit --datatype --ranks 4
  python3 ket/test/mpi/control_gate_numerical.py --bit-masks --ubsan --std c++17
"""

import argparse
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cxx', default='mpic++')
    parser.add_argument('--mpiexec', default='mpiexec')
    parser.add_argument('--yampi', type=Path)
    parser.add_argument('--std', default='c++14', choices=['c++14', 'c++17'])
    parser.add_argument('--ranks', type=int, default=2, choices=[2, 4])
    parser.add_argument('--basis', type=int, default=0,
                        help='number of computational-basis states, in addition to a complex superposition')
    parser.add_argument('--group', type=int, choices=[1, 2, 3], action='append')
    parser.add_argument('--variant', default='ranges',
                        choices=['ranges', 'variadic', 'runtime-variadic', 'iterators', 'homogeneous'])
    parser.add_argument('--bit-masks', action='store_true')
    parser.add_argument('--ubsan', action='store_true')
    parser.add_argument('--diagonal', action='store_true')
    parser.add_argument('--unit', action='store_true')
    parser.add_argument('--datatype', action='store_true')
    parser.add_argument('--aliases', action='store_true', help='also exercise controlled gate aliases in variadic mode')
    parser.add_argument('--page-qubits', type=int, default=2, choices=range(1, 6))
    parser.add_argument('--regressions', action='store_true')
    args = parser.parse_args()
    if args.basis < 0:
        parser.error('--basis must be nonnegative')
    ket = Path(__file__).resolve().parents[2]
    yampi = args.yampi or ket.parent.parent / 'yampi' / 'include'
    flags = [args.cxx, '-std=' + args.std, '-DNDEBUG', '-O0', '-fmax-errors=2', '-pthread',
             '-I' + str(ket / 'include'), '-I' + str(yampi)]
    defines = {
        'variadic': ['-DKET_MPI_TEST_VARIADIC=1'],
        'runtime-variadic': ['-DKET_MPI_TEST_VARIADIC=2'],
        'iterators': ['-DKET_MPI_TEST_ITERATORS'],
        'homogeneous': ['-DKET_MPI_TEST_HOMOGENEOUS'],
        'ranges': [],
    }
    flags += defines[args.variant]
    flags.append('-DKET_MPI_TEST_PAGE_QUBITS=' + str(args.page_qubits))
    for enabled, define in [(args.bit_masks, 'KET_USE_BIT_MASKS_EXPLICITLY'),
                            (args.diagonal, 'KET_USE_DIAGONAL_LOOP'),
                            (args.unit, 'KET_MPI_TEST_UNIT'),
                            (args.datatype, 'KET_MPI_TEST_DATATYPE'),
                            (args.aliases, 'KET_MPI_TEST_ALIASES')]:
        if enabled:
            flags.append('-D' + define)
    if args.ubsan:
        flags += ['-fsanitize=undefined', '-fno-sanitize-recover=all']
    tests = ket / 'test' / 'mpi'
    with tempfile.TemporaryDirectory(prefix='ket-mpi-control-gates-') as directory:
        binary = str(Path(directory) / 'test')
        subprocess.run(flags + [str(tests / 'control_gate_api.cpp'), '-o', binary], check=True)
        subprocess.run([binary], check=True)
        preamble = '''#include <complex>
#include <vector>
#include <ket/mpi/gate/hadamard.hpp>
#include <ket/mpi/gate/pauli_x.hpp>
using Q = ket::qubit<>;
using N = ket::control<Q, 0u>;
using P = ket::control<Q, 1u>;
void invalid(yampi::communicator const& comm, yampi::environment const& env) {
  std::vector<std::complex<double>> state(16), buffer;
  ket::mpi::qubit_permutation<> permutation(4u);
  auto const mpi = ket::mpi::utility::policy::make_simple_mpi();
  auto const seq = ket::utility::policy::make_sequential();
'''
        prefix = 'mpi, seq, state, permutation, buffer, comm, env, '
        invalid = {
            'interleaved targets': 'ket::mpi::gate::pauli_x(' + prefix + 'Q{0u}, N{1u}, Q{2u}, P{3u});',
            'target after controls': 'ket::mpi::gate::hadamard(' + prefix + 'Q{0u}, N{1u}, Q{2u});',
            'target-only control range': 'std::vector<Q> controls; ket::mpi::gate::runtime::ranges::hadamard(' + prefix + 'Q{0u}, controls);',
            'mismatched control range': 'std::vector<ket::control<ket::qubit<std::uint32_t, unsigned>, 0u>> controls; ket::mpi::gate::runtime::ranges::hadamard(' + prefix + 'Q{0u}, controls);',
        }
        for name, statement in invalid.items():
            result = subprocess.run(flags + ['-x', 'c++', '-fsyntax-only', '-'],
                                    input=preamble + statement + '\n}', text=True, capture_output=True)
            if result.returncode == 0:
                raise AssertionError('Unexpected compilation success: ' + name)
        print('MPI API and {} compile-failure tests passed'.format(len(invalid)), flush=True)
        for group in args.group or [1, 2, 3]:
            print('MPI control tests: group', group, args.variant, flush=True)
            subprocess.run(flags + ['-DKET_MPI_TEST_GROUP=' + str(group),
                                    str(tests / 'control_gate_numerical.cpp'), '-o', binary], check=True)
            subprocess.run([args.mpiexec, '-n', str(args.ranks), binary, str(args.basis)], check=True)
        if args.regressions:
            for source in ['runtime_gate_numerical.cpp', 'fused_runtime_gate_numerical.cpp']:
                subprocess.run(flags + [str(tests / source), '-o', binary], check=True)
                subprocess.run([args.mpiexec, '-n', '2', binary], check=True)
            subprocess.run(flags + ['-DKET_TEST_NEGATIVE_CONTROLS',
                                    str(tests / 'fused_runtime_gate_numerical.cpp'), '-o', binary], check=True)
            subprocess.run([args.mpiexec, '-n', '2', binary], check=True)


if __name__ == '__main__':
    main()
