#!/usr/bin/env python3
"""Audit measurement into runtime bit-register range destinations (#454)."""

import argparse
import pathlib
import resource
import unittest
from unittest.mock import patch

import qasm2qcx_angle_array_numerical as arrays


class BackendTests(unittest.TestCase):
    bra = None
    checked = 0
    setUp = arrays.BackendTests.setUp
    program = arrays.BackendTests.program
    execute = arrays.BackendTests.execute
    check_initializers = arrays.BackendTests.check_initializers

    def test_basis_states_both_syntaxes_ordering_and_unselected_bits(self):
        for width in (1, 2, 4, 33, 65, 129):
            stride = max(1, (width - 1) // 2)
            for start, step, stop in {(0, stride, width - 1), (-1, -stride, -width), (-1, 1, -1), (-1, 1, 0)}:
                raw = list(range(start, stop + (1 if step > 0 else -1), step))
                positions = [value % width for value in raw]
                if len(positions) != len(set(positions)):
                    continue
                count = len(positions)
                for order in (tuple(range(count)), tuple(reversed(range(count)))):
                    source = 'q[{' + ','.join(map(str, order)) + '}]'
                    for pattern in range(1 << count):
                        gates = ''.join(f'x q[{i}];' for i in range(count) if pattern & (1 << i))
                        for before in sorted({'0' * width, ('101100' * width)[:width]}):
                            bits = list(map(int, reversed(before)))
                            for destination, qubit in zip(positions, order):
                                bits[destination] = (pattern >> qubit) & 1
                            expected = ''.join(map(str, reversed(bits)))
                            target = 'flags[start:step:stop]'
                            for statement in (f'{target} = measure {source};', f'measure {source} -> {target};'):
                                self.check_initializers('include "stdgates.inc"; qubit pad; '
                                    f'qubit[{count}] q; reset q; bit[{width}] flags = "{before}"; '
                                    f'int start = {start}; {"uint" if step > 0 else "int"} step = {step}; '
                                    f'{"uint" if stop >= 0 else "int"} stop = {stop}; ' + gates + statement +
                                    f'bool same = flags == "{expected}";', (), (), ('same',), (1,))

    def test_whole_static_ranges_constants_and_repeated_qubits(self):
        for source, order in (('q', (0, 1, 2)), ('q[HIGH:-1:LOW]', (2, 1, 0)),
                               ('q[0:2:2]', (0, 2)), ('q[{-1, LOW}]', (2, 0)),
                               ('q[{1, 1}]', (1, 1)), ('q[1:1]', (1,))):
            for pattern in range(8):
                bits = [0, 1, 0, 1]
                for i, qubit in enumerate(order):
                    bits[i] = (pattern >> qubit) & 1
                expected = ''.join(map(str, reversed(bits)))
                gates = ''.join(f'x q[{i}];' for i in range(3) if pattern & (1 << i))
                self.check_initializers('include "stdgates.inc"; const int LOW = 0; const uint HIGH = 2; '
                    'qubit[3] q; reset q; bit[4] flags = "1010"; int start = 0; '
                    f'int stop = {len(order) - 1}; ' + gates +
                    f'flags[start:stop] = measure {source}; bool same = flags == "{expected}";',
                    (), (), ('same',), (1,))

    def test_singleton_storage_and_original_value_capture(self):
        for value in (0, 1):
            self.check_initializers('include "stdgates.inc"; qubit[1] q; reset q; bit[1] one = "1"; '
                f'int index = -1; {"x q;" if value else ""} one[index:index] = measure q;',
                (), (), ('one',), (value,))
        # The first write changes the bit used to compute the stride. Later
        # writes must still use the complete original captured range.
        self.check_initializers('include "stdgates.inc"; qubit[2] q; reset q; x q[1]; '
            'bit[4] flags = "1001"; int index = 0; array[int, 2] bounds = {0, 3}; '
            'flags[bounds[index]:int(flags[0]) + 1:bounds[index + 1]] = measure q; '
            'bool same = flags == "1100";', (), (), ('same',), (1,))

    def test_native_range_steps_and_unselected_stop_values(self):
        limits = self.module.QASM2QCXConverter
        for kind, step, start, stop, expected in (
                ('uint', limits.QCX_UINT_MAX, 0, 3, '0001'),
                ('int', limits.QCX_INT_MIN, -1, -4, '1000'),
                ('uint', 5, 0, 4, '0001')):
            self.check_initializers('include "stdgates.inc"; qubit[1] q; reset q; x q; '
                f'bit[4] flags = "0000"; int start = {start}; {kind} step = {step}; int stop = {stop}; '
                f'measure q -> flags[start:step:stop]; bool same = flags == "{expected}";',
                (), (), ('same',), (1,))

    def test_storage_limit_allows_short_measurements_into_wide_destinations(self):
        with patch.object(self.module.QASM2QCXConverter, 'MAX_RUNTIME_BIT_RANGE_POSITIONS', 1):
            self.check_initializers('include "stdgates.inc"; qubit[1] q; reset q; x q; '
                'bit[129] flags = "' + '0' * 129 + '"; int start = -1; int stop = -1; '
                'measure q -> flags[start:stop]; bool same = flags == "1' + '0' * 128 + '";',
                (), (), ('same',), (1,))

    def test_reentry_changes_order_and_uses_separate_source_widths_without_stale_outcomes(self):
        for width in (4, 33, 65, 129):
            before = ('101100' * width)[:width]
            bits = list(map(int, reversed(before)))
            stride = 1 if width == 4 else 2
            checks = []
            for visit in range(5):
                if visit % 2 == 0:
                    start, step = (0, stride) if visit % 4 == 0 else (width - 1, -stride)
                    positions = [start + i * step for i in range(3)]
                    values = [(visit >> i) & 1 for i in range(3)]
                else:
                    positions, values = [visit % width], [visit & 1]
                for position, value in zip(positions, values):
                    bits[position] = value
                text = ''.join(map(str, reversed(bits)))
                checks.append(f'if (visit == {visit}) {{ same = same && flags == "{text}"; }}')
            self.check_initializers('include "stdgates.inc"; qubit[3] q; '
                f'bit[{width}] flags = "{before}"; int first = 0; int last = 4; '
                f'int start = 0; int step = {stride}; int stop = 0; bool same = true; '
                'for int visit in [first:last] { reset q; if ((visit & 1) != 0) { x q[0]; } '
                'if ((visit & 2) != 0) { x q[1]; } if ((visit & 4) != 0) { x q[2]; } '
                f'if (visit % 2 == 0) {{ start = 0; step = {stride}; '
                f'if (visit % 4 == 2) {{ start = {width - 1}; step = -step; }} stop = start + 2 * step; '
                'flags[start:step:stop] = measure q; } '
                f'else {{ start = visit % {width}; flags[start:start] = measure q[0:0]; }} '
                + ''.join(checks) + '}', (), (), ('same',), (1,))

    def test_range_set_array_and_live_bit_register_loops(self):
        for loop, indices in (('for int i in [first:last]', (0, 1, 2)),
                               ('for int i in {2, 0, 1}', (2, 0, 1)),
                               ('for int i in {last, first, last - 1}', (2, 0, 1)),
                               ('for int i in indices', (2, 0, 1))):
            bits = [0] * 4
            for i in indices:
                bits[i], bits[i + 1] = 1, 0
            expected = ''.join(map(str, reversed(bits)))
            self.check_initializers('include "stdgates.inc"; qubit[2] q; reset q; x q[0]; '
                'bit[4] flags = "0000"; int first = 0; int last = 2; array[int, 3] indices = {2, 0, 1}; '
                + loop + ' { measure q -> flags[i:i + 1]; } '
                f'bool same = flags == "{expected}";', (), (), ('same',), (1,))
        self.check_initializers('qubit[1] q; reset q; bit[4] flags = "1111"; int next = 1; int total = 0; '
            'for bit b in flags { if (b && next < 4) { flags[next:next] = measure q; next += 1; } '
            'total += int(b); } bool same = flags == "0001";', (), (), ('same', 'total'), (1, 3))

    def test_nested_transfers_and_iterator_shadowing_restore_outer_bindings(self):
        self.check_initializers('qubit[1] q; reset q; const int I = 3; int start = -1; '
            'bit[4] flags = "1111"; int first = 0; int last = 3; int visits = 0; '
            'for int I in [first:last] { if (I == 1) { continue; } for int start in [0:2] { '
            'if (start == 1) { continue; } measure q -> flags[I:I]; visits += 1; break; } '
            'if (I == 2) { break; } } flags[I:I] = measure q; '
            'bool same = flags == "0010"; int restored = I;',
            (), (), ('same', 'visits', 'start', 'restored'), (1, 2, -1, 3))

    def test_skipped_invalid_ranges_do_not_evaluate_indices_or_collapse_qubits(self):
        selectors = ('start:0:stop', 'bad:stop', '-2:2:stop + 1', 'start:start',
                     'start:1 / zero:stop', 'bounds[bad]:stop')
        for selector in selectors:
            for wrapper in ('if (false) {{ {} }}', 'while (false) {{ {} }}',
                            'for int outer in [1:0] {{ {} }}',
                            'for int outer in [0:1] {{ continue; {} }}'):
                self.check_initializers('include "stdgates.inc"; qubit[3] q; reset q; h q[0]; '
                    'bit[4] flags = "1010"; int start = 0; int stop = 1; int bad = 4; int zero = 0; '
                    'array[int, 2] bounds = {0, 1}; '
                    + wrapper.format(f'measure q -> flags[{selector}];') +
                    'h q[0]; bit[3] observed = measure q; bool same = flags == "1010"; '
                    'bool untouched = observed == "000";', (), (), ('same', 'untouched'), (1, 1))

    def test_entangled_correlations_repeatability_and_measurement_feedback(self):
        for anti in (0, 1):
            for known in (0, 1):
                for _ in range(8):
                    self.check_initializers('include "stdgates.inc"; qubit[3] q; reset q; h q[0]; cx q[0], q[1]; '
                        + ('x q[1];' if anti else '') + ('x q[2];' if known else '') +
                        'bit[4] flags = "0001"; bit[3] repeated; bit[1] corrected; '
                        'int start = -1; int step = -1; int stop = -3; uint i = 0; '
                        'flags[start:step:stop] = measure q[{1, 2, 0}]; repeated = measure q; '
                        f'bool correlation = (flags[3] != flags[1]) == bool({anti}); '
                        f'bool ordered = flags[2] == bool({known}); bool preserved = flags[0]; '
                        'bool stable = flags[1] == repeated[0] && flags[3] == repeated[1] && flags[2] == repeated[2]; '
                        'if (flags[3]) { x q[1]; } corrected[i:i] = measure q[{1}]; '
                        'bool feedback = corrected == "0";',
                        (), (), ('correlation', 'ordered', 'preserved', 'stable', 'feedback'), (1, 1, 1, 1, 1))
        for anti in (0, 1):
            for _ in range(8):
                self.check_initializers('include "stdgates.inc"; qubit[2] q; reset q; h q[0]; cx q[0], q[1]; '
                    + ('x q[1];' if anti else '') +
                    'bit[2] flags; int start = 0; int stop = 1; measure q[{0, 0}] -> flags[start:stop]; '
                    f'bit other = measure q[1]; bool same = flags[0] == flags[1] && (flags[0] != other) == bool({anti});',
                    (), (), ('same',), (1,))

    def test_later_iteration_failures_precede_all_new_measurements_and_writes(self):
        for selector, update, error in (
                ('start:step:stop', 'stop = 0;', 'assertion failed'),
                ('start:step:stop', 'stop = 2;', 'assertion failed'),
                ('start:step:stop', 'step = 0;', 'assertion failed'),
                ('start:step:stop', 'start = -1; step = 4; stop = 3;', 'assertion failed'),
                ('start:step / divisor:stop', 'divisor = 0;', 'division by zero')):
            lines = self.program('include "stdgates.inc"; qubit[2] q; reset q; x q[0]; '
                'bit[4] flags = "1010"; int start = 0; int stop = 1; int step = 1; int divisor = 1; int i = 0; '
                f'while (i < 2) {{ flags[{selector}] = measure q; i += 1; {update} }}')
            last_write = max(i for i, line in enumerate(lines) if line.startswith('LET FLAGS31:QASM2QCX_'))
            write_end = next(i for i in range(last_write + 1, len(lines)) if lines[i].startswith('@'))
            instrumented = []
            for i, line in enumerate(lines):
                if line.startswith('M '):
                    instrumented.append('PRINTLN 12345')
                if line.startswith('LET FLAGS31:QASM2QCX_'):
                    instrumented.append('PRINTLN 67890')
                instrumented.append(line)
                if i == write_end:
                    instrumented.append('PRINTLN FLAGS31:3 FLAGS31:2 FLAGS31:1 FLAGS31:0')
            result = self.execute(instrumented)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(error, result.stderr)
            self.assertNotIn('runtime error:', result.stderr)
            self.assertEqual([''.join(line.split()) for line in result.stdout.splitlines()],
                             ['12345', '12345', '67890', '67890', '1001'])
            type(self).checked += 1

    def test_bad_destinations_fail_before_first_measurement_or_write(self):
        limits = self.module.QASM2QCXConverter
        cases = (
            ('int start = 0; int stop = 1;', 'start:0:stop', 2, 'assertion failed'),
            ('int start = 3; int stop = 0;', 'start:stop', 2, 'assertion failed'),
            ('int start = 0; int stop = 0;', 'start:stop', 2, 'assertion failed'),
            ('int start = 0; int stop = 2;', 'start:stop', 2, 'assertion failed'),
            ('int start = -2; int stop = 2;', 'start:2:stop', 3, 'assertion failed'),
            ('int start = -1; int stop = 3;', 'start:stop', 5, 'assertion failed'),
            ('int start = -5; int stop = -4;', 'start:stop', 2, 'assertion failed'),
            ('int start = 3; int stop = 4;', 'start:stop', 2, 'assertion failed'),
            (f'uint start = {limits.QCX_UINT_MAX}; uint stop = start;', 'start:stop', 1, 'assertion failed'),
            (f'int start = {limits.QCX_INT_MIN}; int stop = start;', 'start:stop', 1, 'assertion failed'),
            ('int start = 0; int stop = 1; int zero = 0;', 'start:1 / zero:stop', 2, 'division by zero'),
            ('int start = 0; int stop = 1; array[int, 1] bounds = {0};',
             'start:bounds[stop]:stop', 2, 'assertion failed'),
        )
        for declarations, selector, count, error in cases:
            for statement in (f'flags[{selector}] = measure q;', f'measure q -> flags[{selector}];'):
                with self.subTest(selector=selector, declarations=declarations, statement=statement):
                    lines = self.program(f'qubit[{count}] q; bit[4] flags = "1010"; ' + declarations + statement)
                    instrumented = []
                    for line in lines:
                        if line.startswith(('M ', 'LET FLAGS31:QASM2QCX_')):
                            instrumented.append('PRINTLN 12345')
                        instrumented.append(line)
                    self.assertIn('PRINTLN 12345', instrumented)
                    result = self.execute(instrumented)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(error, result.stderr)
                    self.assertNotIn('runtime error:', result.stderr)
                    self.assertNotIn('12345', result.stdout)
                    type(self).checked += 1


def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    args = parser.parse_args()
    BackendTests.bra = args.bra.resolve()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(BackendTests))
    print(f'runtime bit-range measurement programs checked: {BackendTests.checked}')
    raise SystemExit(not result.wasSuccessful())


if __name__ == '__main__':
    main()
