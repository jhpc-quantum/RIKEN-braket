#!/usr/bin/env python3
"""Build/run ordinary control-value gate tests (C++14 by default).

Examples:
  python3 ket/test/control_gate_numerical.py
  python3 ket/test/control_gate_numerical.py --bit-masks --ubsan
"""

import argparse
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cxx', default='g++')
    parser.add_argument('--std', default='c++14', choices=['c++14', 'c++17'])
    parser.add_argument('--bit-masks', action='store_true')
    parser.add_argument('--ubsan', action='store_true')
    parser.add_argument('--group', type=int, choices=range(1, 5), action='append')
    args = parser.parse_args()
    ket = Path(__file__).resolve().parents[1]
    flags = [args.cxx, '-std=' + args.std, '-O0', '-fmax-errors=3',
             '-I' + str(ket / 'include')]
    if args.bit_masks:
        flags.append('-DKET_USE_BIT_MASKS_EXPLICITLY')
    if args.ubsan:
        flags += ['-fsanitize=undefined', '-fno-sanitize-recover=all']
    with tempfile.TemporaryDirectory(prefix='ket-control-gates-') as directory:
        binary = str(Path(directory) / 'test')
        for source in ['control_gate_api.cpp', 'control_gate_ranges.cpp']:
            subprocess.run(flags + [str(ket / 'test' / source), '-o', binary], check=True)
            subprocess.run([binary], check=True)
        for group in args.group or range(1, 5):
            subprocess.run(flags + ['-DKET_TEST_GROUP=' + str(group),
                                    str(ket / 'test' / 'control_gate_numerical.cpp'), '-o', binary], check=True)
            subprocess.run([binary], check=True)
        preamble = ('#include <complex>\n#include <vector>\n'
                    '#include <ket/gate/hadamard.hpp>\n#include <ket/gate/pauli_x.hpp>\n'
                    'using Q = ket::qubit<>;\nusing P = ket::control<Q, 1>;\n'
                    'using N = ket::control<Q, 0>;\n'
                    'int main() { std::vector<std::complex<double>> state(32);\n')
        invalid = {
            'interleaved targets': 'ket::gate::pauli_x(state.begin(), state.end(), Q{0u}, N{1u}, Q{2u}, P{3u});',
            'target after mixed controls': 'ket::gate::hadamard(state.begin(), state.end(), Q{0u}, N{1u}, P{2u}, Q{3u});',
            'target-only control range': 'std::vector<Q> controls{Q{1u}}; ket::gate::runtime::qubit_ranges::hadamard(state.begin(), state.end(), Q{0u}, controls);',
            'mismatched control range': 'std::vector<ket::runtime_control<ket::qubit<std::uint32_t, unsigned int>>> controls; ket::gate::runtime::qubit_ranges::hadamard(state.begin(), state.end(), Q{0u}, controls);',
        }
        for name, statement in invalid.items():
            result = subprocess.run(flags + ['-x', 'c++', '-fsyntax-only', '-'],
                                    input=preamble + statement + '\n}\n', text=True, capture_output=True)
            if result.returncode == 0:
                raise AssertionError('Unexpected compilation success: ' + name)
    print('Control gate API tests and {} compile-failure checks passed'.format(len(invalid)))


if __name__ == '__main__':
    main()
