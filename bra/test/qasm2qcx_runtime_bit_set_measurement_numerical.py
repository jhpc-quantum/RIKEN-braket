#!/usr/bin/env python3
"""Audit measurement into runtime bit-register index-set destinations (#446)."""

import argparse
import pathlib
import resource
import unittest

import qasm2qcx_angle_array_numerical as arrays


class BackendTests(unittest.TestCase):
    bra = None
    checked = 0
    setUp = arrays.BackendTests.setUp
    program = arrays.BackendTests.program
    execute = arrays.BackendTests.execute
    check_initializers = arrays.BackendTests.check_initializers

    def test_basis_states_ordering_both_syntaxes_and_unchanged_bits(self):
        for width in (1, 2, 4, 33, 65, 129):
            positions = sorted({0, width // 2, width - 1}, reverse=True)
            count = len(positions)
            declarations = ''.join(
                f'int i{j} = {position - width};' if j % 2 == 0 else f'uint i{j} = {position};'
                for j, position in enumerate(positions))
            target = 'flags[{' + ','.join(f'i{j}' for j in range(count)) + '}]'
            for source_positions in (tuple(range(count)), tuple(reversed(range(count)))):
                source = 'q[{' + ','.join(map(str, source_positions)) + '}]'
                for pattern in range(1 << count):
                    gates = ''.join(f'x q[{j}];' for j in range(count) if pattern & (1 << j))
                    for before in sorted({'0' * width, ('101100' * width)[:width]}):
                        bits = list(map(int, reversed(before)))
                        for destination, qubit in zip(positions, source_positions):
                            bits[destination] = (pattern >> qubit) & 1
                        expected = ''.join(map(str, reversed(bits)))
                        for statement in (f'{target} = measure {source};', f'measure {source} -> {target};'):
                            self.check_initializers('include "stdgates.inc"; qubit pad; '
                                f'qubit[{count}] q; bit[{width}] flags = "{before}"; '
                                + declarations + gates + statement + f'bool same = flags == "{expected}";',
                                (), (), ('same',), (1,))

    def test_whole_ranges_constants_repeated_qubits_and_singleton_registers(self):
        for source, values in (('q', (1, 0, 1)), ('q[2:-1:0]', (1, 0, 1)),
                               ('q[0:2:2]', (1, 1)), ('q[{HIGH - 1, -3}]', (0, 1)),
                               ('q[{1, 1}]', (0, 0)), ('q[{2}]', (1,)), ('q[1:1]', (0,))):
            before = [1, 0, 1, 0]
            for position, value in enumerate(values):
                before[position] = value
            expected = ''.join(map(str, reversed(before)))
            target = '{index' + ''.join(f', {j}' for j in range(1, len(values))) + '}'
            self.check_initializers('include "stdgates.inc"; const int HIGH = 2; qubit[3] q; '
                'x q[0]; x q[2]; bit[4] flags = "0101"; int index = 0; '
                f'flags[{target}] = measure {source}; bool same = flags == "{expected}";',
                (), (), ('same',), (1,))
        for value in (0, 1):
            self.check_initializers('include "stdgates.inc"; qubit[1] q; bit[1] flags = "1"; '
                f'int index = -1; {"x q;" if value else ""} flags[{{index}}] = measure q;',
                (), (), ('flags', 'index'), (value, -1))

    def test_nested_destinations_read_original_bits_and_capture_once(self):
        # The first result changes flags[0]. The second address must already
        # have been selected using its original value, so bit 2 receives zero.
        self.check_initializers('include "stdgates.inc"; qubit[2] q; x q[0]; '
            'bit[4] flags = "1110"; int index = 0; array[int, 2] positions = {2, 3}; '
            'flags[{index, positions[int(flags[index])]}] = measure q; '
            'bool same = flags == "1011";', (), (), ('same', 'index'), (1, 0))

    def test_bad_destinations_fail_before_first_measurement(self):
        limits = self.module.QASM2QCXConverter
        cases = []
        for width in (1, 4, 65, 129):
            for kind, value in (('int', -width - 1), ('int', width),
                                ('int', limits.QCX_INT_MIN), ('uint', limits.QCX_UINT_MAX)):
                cases.append((width, f'{kind} index = {value};', 'index, 0', ''))
            cases.extend((width, f'int index = {width - 1};', indices, '')
                         for indices in ('index, index', 'index, -1'))
        cases.extend(((4, 'int index = 0; int zero = 0;', 'index, 1 / zero', 'integer division by zero'),
                      (4, 'int index = 0; array[int, 1] positions = {0};',
                       'index, positions[index + 1]', ''),
                      (4, 'int index = 0;', 'index, 999999999999999999999999999999999999', '')))
        for width, declarations, indices, message in cases:
            with self.subTest(width=width, indices=indices, declarations=declarations):
                lines = self.program(f'qubit[2] q; bit[{width}] flags; ' + declarations +
                                     f'measure q -> flags[{{{indices}}}];')
                first_m = next(i for i, line in enumerate(lines) if line.startswith('M '))
                # A reached sentinel would prove that validation allowed
                # execution to get as far as the first measurement.
                lines.insert(first_m, 'PRINTLN 12345')
                result = self.execute(lines)
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertNotIn('12345', result.stdout)
                self.assertNotIn('runtime error:', result.stderr)
                if message:
                    self.assertIn(message, result.stderr)
                else:
                    self.assertTrue('assertion failed in ASSERT' in result.stderr
                                    or 'conversion' in result.stderr, result.stderr)
                type(self).checked += 1


def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    args = parser.parse_args()
    BackendTests.bra = args.bra.resolve()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(BackendTests))
    print(f'runtime bit-set measurement programs checked: {BackendTests.checked}')
    raise SystemExit(not result.wasSuccessful())


if __name__ == '__main__':
    main()
