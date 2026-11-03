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
