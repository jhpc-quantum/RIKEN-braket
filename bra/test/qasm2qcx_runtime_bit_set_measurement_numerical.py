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

    def test_skipped_measurements_do_not_evaluate_destinations_or_collapse_state(self):
        cases = ('bad, 0', 'index, index', 'index, 1 / zero', 'index, 999999999999999999999999',
                 'index, int(flags[bad])')
        wrappers = ('if (false) {{ {} }}', 'while (false) {{ {} }}',
                    'for int outer in [1:0] {{ {} }}',
                    'for int outer in [0:1] {{ if (outer < 0) {{ {} }} }}')
        for indices in cases:
            for wrapper in wrappers:
                self.check_initializers('include "stdgates.inc"; qubit[2] q; h q[0]; '
                    'bit[4] flags = "1010"; bit[2] observed; int index = 0; int bad = 4; int zero = 0; '
                    + wrapper.format(f'measure q -> flags[{{{indices}}}];') +
                    'h q[0]; observed = measure q; '
                    'bool same = flags == "1010"; bool untouched = observed == "00";',
                    (), (), ('same', 'untouched'), (1, 1))

    def test_range_set_array_and_register_loops_capture_live_indices(self):
        loops = ('for int i in [first:last]', 'for int i in {-4, -3, -2, -1}',
                 'for int i in {first, first + 1, last - 1, last}', 'for int i in positions')
        for loop in loops:
            self.check_initializers('include "stdgates.inc"; qubit[1] q; bit[4] flags = "0000"; '
                'int first = -4; int last = -1; array[int, 4] positions = {-4, -3, -2, -1}; '
                + loop + ' { reset q; if (i % 2 == 0) { x q; } flags[{i}] = measure q; } '
                'bool same = flags == "0101";', (), (), ('same',), (1,))
        self.check_initializers('include "stdgates.inc"; qubit[1] q; bit[2] indices = "10"; '
            'bit[4] flags = "1010"; for bit i in indices { reset q; if (i) { x q; } '
            'flags[{int(i)}] = measure q; } bool same = flags == "1010";',
            (), (), ('same',), (1,))

    def test_while_reentry_break_continue_and_nested_loop_transfers(self):
        self.check_initializers('include "stdgates.inc"; qubit[2] q; x q[0]; '
            'bit[4] flags = "0000"; int i = 0; while (i < 4) { '
            'flags[{i, (i + 1) % 4}] = measure q; i += 1; } '
            'bool same = flags == "1110";', (), (), ('same', 'i'), (1, 4))
        self.check_initializers('include "stdgates.inc"; qubit[1] q; x q; bit[4] flags = "0000"; '
            'int first = 0; int last = 3; int visits = 0; '
            'for int i in [first:last] { if (i == 1) { continue; } '
            'for int j in [0:2] { if (j == 1) { continue; } '
            'flags[{i}] = measure q; visits += 1; break; } if (i == 2) { break; } } '
            'bool same = flags == "0101";', (), (), ('same', 'visits'), (1, 2))

    def test_iterator_shadowing_restores_constants_classical_variables_and_qubits(self):
        self.check_initializers('include "stdgates.inc"; const int I = 3; int index = -1; '
            'qubit[1] q; x q; bit[4] flags = "0000"; int first = 0; int last = 1; '
            'for int I in [first:last] { flags[{I}] = measure q; } '
            'for int index in {2, 3} { flags[{index}] = measure q; } '
            'for int q in [0:1] {} flags[{I}] = measure q; '
            'bool same = flags == "1111"; int restored = I;',
            (), (), ('same', 'restored', 'index'), (1, 3, -1))

    def test_entangled_correlations_repeated_measurement_and_feedback(self):
        # Do not assume a particular random outcome or require both branches
        # to occur. Correlations and repeatability hold for either outcome.
        for anti in (0, 1):
            for known in (0, 1):
                for _ in range(8):
                    self.check_initializers('include "stdgates.inc"; qubit[3] q; h q[0]; cx q[0], q[1]; '
                        + ('x q[1];' if anti else '') + ('x q[2];' if known else '') +
                        'bit[4] flags = "0100"; bit[3] repeated; bit[1] corrected; int i = -1; uint j = 1; '
                        'flags[{i, 0, j}] = measure q[{1, 2, 0}]; repeated = measure q; '
                        f'bool correlation = (flags[3] != flags[1]) == bool({anti}); '
                        f'bool ordered = flags[0] == bool({known}); bool preserved = flags[2]; '
                        'bool stable = flags[1] == repeated[0] && flags[3] == repeated[1] && flags[0] == repeated[2]; '
                        'if (flags[3]) { x q[1]; } corrected[{j - 1}] = measure q[{1}]; '
                        'bool feedback = corrected == "0";',
                        (), (), ('correlation', 'ordered', 'preserved', 'stable', 'feedback'), (1, 1, 1, 1, 1))

    def test_later_iteration_failures_stop_before_new_measurements(self):
        for second, update, error in (('other', '', 'assertion failed in ASSERT'),
                                      ('other', 'other += 3;', 'assertion failed in ASSERT'),
                                      ('1 / (1 - index)', '', 'integer division by zero')):
            lines = self.program('qubit[2] q; bit[4] flags; int index = 0; int other = 1; '
                f'while (index < 2) {{ flags[{{index, {second}}}] = measure q; index += 1; {update} }}')
            instrumented = []
            for line in lines:
                if line.startswith('M '):
                    instrumented.append('PRINTLN 12345')
                instrumented.append(line)
            result = self.execute(instrumented)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(error, result.stderr)
            self.assertNotIn('runtime error:', result.stderr)
            self.assertEqual(result.stdout.split(), ['12345', '12345'])
            type(self).checked += 1

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
