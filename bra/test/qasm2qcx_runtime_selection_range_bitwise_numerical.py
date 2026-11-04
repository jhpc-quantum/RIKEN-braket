#!/usr/bin/env python3
"""Audit runtime-selected bitwise assignments with runtime range RHS values (#466)."""

import argparse
import operator
import pathlib
import re
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
    operations = (('&=', operator.and_), ('|=', operator.or_), ('^=', operator.xor))

    @staticmethod
    def declarations(values):
        return ''.join(f'{"uint" if value >= 0 else "int"} {name} = {value}; ' for name, value in values)

    def test_independent_set_and_range_destinations_preserve_order_and_unselected_bits(self):
        for width in (1, 4, 33, 65, 129):
            for start, step, stop in ((0, 1, width - 1), (width - 1, -1, 0),
                                       (-1, -1, -width), (0, 2, width - 1), (-1, width, width - 1)):
                raw_source = list(range(start, stop + (1 if step > 0 else -1), step))
                count = len(raw_source)
                target_width = 2 * count + 3
                before = ('01101001' * target_width)[:target_width]
                source = ('100110' * width)[:width]
                source_bits = list(map(int, reversed(source)))
                selected = [source_bits[index % width] for index in raw_source]
                descending = tuple(range(target_width - 2, target_width - 2 - 2 * count, -2))
                permuted = descending[1:] + descending[:1]
                targets = (('result[dstart:dstep:dstop]', descending,
                            self.declarations((('dstart', descending[0]), ('dstep', -2), ('dstop', descending[-1])))),
                           ('result[dstart:dstep:dstop]', tuple(range(1, 2 * count, 2)),
                            self.declarations((('dstart', -target_width + 1), ('dstep', 2), ('dstop', -4)))),
                           ('result[{index' + ''.join(f', {i}' for i in permuted[1:]) + '}]', permuted,
                            f'int index = {permuted[0] - target_width}; '))
                for target, positions, declarations in targets:
                    for constant in (False, True):
                        for symbol, function in self.operations:
                            expected = list(map(int, reversed(before)))
                            for destination, bit in zip(positions, selected):
                                expected[destination] = function(expected[destination], bit)
                            after = ''.join(map(str, reversed(expected)))
                            self.check_initializers(('const ' if constant else '') +
                                f'bit[{width}] source = "{source}"; bit[{target_width}] result = "{before}"; '
                                + declarations + self.declarations((('start', start), ('step', step), ('stop', stop))) +
                                f'{target} {symbol} source[start:step:stop]; '
                                f'bool same = result == "{after}"; bool unchanged = source == "{source}";',
                                (), (), ('same', 'unchanged'), (1, 1))

    def test_self_referential_destination_indices_and_rhs_bounds_use_original_values(self):
        for pattern in range(16):
            bits = [(pattern >> i) & 1 for i in range(4)]
            for target, positions in (('flags[{int(flags[0]), int(flags[1]) + 2}]', (bits[0], bits[1] + 2)),
                                       ('flags[int(flags[0]):int(flags[0]) + 1]', (bits[0], bits[0] + 1))):
                for symbol, function in self.operations:
                    expected = bits.copy()
                    for destination, source in zip(positions, (bits[2], bits[2] + 1)):
                        expected[destination] = function(bits[destination], bits[source])
                    after = ''.join(map(str, reversed(expected)))
                    self.check_initializers(f'bit[4] flags = "{pattern:04b}"; '
                        f'{target} {symbol} flags[int(flags[2]):int(flags[2]) + 1]; '
                        f'bool same = flags == "{after}";', (), (), ('same',), (1,))

    def test_singleton_storage_native_extreme_strides_and_small_position_limits(self):
        limits = self.module.QASM2QCXConverter
        cases = ((0, limits.QCX_UINT_MAX, limits.QCX_UINT_MAX - 1, 'uint'),
                 (0, limits.QCX_INT_MIN, -4, 'int'), (-1, limits.QCX_INT_MAX, 3, 'int'))
        for target in ('one[{index}]', 'one[index:index]'):
            for start, step, stop, kind in cases:
                for symbol, function in self.operations:
                    expected = function(1, int(start == -1))
                    self.check_initializers('const bit[4] C = "1000"; bit[1] one = "1"; int index = -1; '
                        f'int start = {start}; {kind} step = {step}; {kind} stop = {stop}; '
                        f'{target} {symbol} C[start:step:stop]; bool same = one == "{expected}";',
                        (), (), ('same',), (1,))
        for target in ('result[{first, last}]', 'result[first:128:last]'):
            for symbol, function in self.operations:
                with patch.object(limits, 'MAX_RUNTIME_BIT_RANGE_POSITIONS', 2):
                    expected = function(0, 1) * ((1 << 128) | 1)
                    self.check_initializers('const bit[129] C = "' + '1' + '0' * 127 + '1"; '
                        'bit[129] result = "' + '0' * 129 + '"; int first = 0; uint last = 128; '
                        f'{target} {symbol} C[first:128:last]; bool same = result == "{expected:0129b}";',
                        (), (), ('same',), (1,))

    def test_loop_families_refresh_runtime_addresses_and_original_values(self):
        loops = (('for int i in [0:3]', 'i', (0, 1, 2, 3)),
                 ('for int i in [first:last]', 'i', (0, 1, 2, 3)),
                 ('for int i in [ufirst:ulast]', 'i', (0, 1, 2, 3)),
                 ('for int i in [first:stride:last]', 'i', (0, 1, 2, 3)),
                 ('for int i in {0, 1, 2, 3}', 'i', (0, 1, 2, 3)),
                 ('for int i in {first, first + 1, last - 1, last}', 'i', (0, 1, 2, 3)),
                 ('for int i in indices', 'i', (0, 1, 2, 3)),
                 ('for uint i in unsigned_indices', 'i', (0, 1, 2, 3)),
                 ('for bit i in flags', 'int(i)', (0, 1, 0, 1)),
                 ('for bit i in C', 'int(i)', (0, 1, 0, 1)))
        source = [0, 1, 0, 0, 1, 1, 0, 1]
        for loop, index, visits in loops:
            for constant in (False, True):
                for is_set in (False, True):
                    target = f'result[{{{index} + 4, {index}}}]' if is_set else f'result[{index}:{index} + 1]'
                    for symbol, function in self.operations:
                        expected = list(map(int, reversed('11001100')))
                        total = 0
                        for value in visits:
                            positions = (value + 4, value) if is_set else (value, value + 1)
                            for position, bit in zip(positions, source[value:value + 2]):
                                expected[position] = function(expected[position], bit)
                            total += expected[positions[0]]
                        after = ''.join(map(str, reversed(expected)))
                        self.check_initializers(('const ' if constant else '') + 'bit[8] source = "10110010"; '
                            'bit[8] result = "11001100"; bit[4] flags = "1010"; const bit[4] C = "1010"; '
                            'array[int,4] indices = {0,1,2,3}; array[uint,4] unsigned_indices = {0,1,2,3}; '
                            'int first = 0; int last = 3; uint ufirst = 0; uint ulast = 3; int stride = 1; '
                            'int i = -9; int total = 0; ' + loop +
                            f' {{ {target} {symbol} source[{index}:{index} + 1]; '
                            f'total += int(result[{index}' + (' + 4' if is_set else '') + ']); } '
                            f'bool same = result == "{after}"; bool unchanged = source == "10110010";',
                            (), (), ('same', 'unchanged', 'total', 'i'), (1, 1, total, -9))

    def test_reentry_refreshes_directions_counts_and_source_values(self):
        ranges = (((7, -2, 1), (0, 1, 3)), ((0, 1, 0), (-1, 1, -1)),
                  ((6, -2, 2), (3, -1, 1)), ((1, 2, 3), (-1, 4, 3)))
        sets = (((7, 1), (0, 1, 1)), ((0, 7), (3, -1, 2)),
                ((6, 2), (-1, 4, 3)), ((1, 3), (1, 2, 3)))
        for is_set in (False, True):
            plans = sets if is_set else ranges
            dstarts = [d[0] for d, _ in plans]
            dstops = [d[-1] for d, _ in plans]
            dsteps = [1 if is_set else d[1] for d, _ in plans]
            sstarts, ssteps, sstops = zip(*(s for _, s in plans))
            arrays_text = ''.join(f'array[int,4] {name} = {{' + ','.join(map(str, values)) + '}; '
                                  for name, values in (('dstarts', dstarts), ('dsteps', dsteps), ('dstops', dstops),
                                                       ('sstarts', sstarts), ('ssteps', ssteps), ('sstops', sstops)))
            target = 'result[{dstarts[i], dstops[i]}]' if is_set else 'result[dstarts[i]:dsteps[i]:dstops[i]]'
            for constant in (False, True):
                for symbol, function in self.operations:
                    expected = list(map(int, reversed('11001100')))
                    source = [0, 1, 0, 1]
                    total = 0
                    for destination, (start, step, stop) in plans:
                        if is_set:
                            positions = destination
                        else:
                            first, stride, last = destination
                            positions = range(first, last + (1 if stride > 0 else -1), stride)
                        raw = range(start, stop + (1 if step > 0 else -1), step)
                        for position, index in zip(positions, raw):
                            expected[position] = function(expected[position], source[index % 4])
                        total += expected[destination[0]]
                        if not constant:
                            source[1] ^= 1
                            source[2] ^= 1
                    after = ''.join(map(str, reversed(expected)))
                    self.check_initializers(('const ' if constant else '') + 'bit[4] source = "1010"; '
                        'bit[8] result = "11001100"; int i = 0; int total = 0; ' + arrays_text +
                        f'while (i < 4) {{ {target} {symbol} source[sstarts[i]:ssteps[i]:sstops[i]]; '
                        'total += int(result[dstarts[i]]); ' + ('' if constant else 'source ^= "0110"; ') +
                        f'i += 1; }} bool same = result == "{after}";', (), (), ('same', 'total', 'i'), (1, total, 4))

    def test_branches_refresh_overlapping_self_references_with_different_counts(self):
        for pattern in range(16):
            for symbol, function in self.operations:
                expected = [(pattern >> i) & 1 for i in range(4)]
                for iteration in range(4):
                    original = expected.copy()
                    positions, source = ((tuple(range(4)), tuple(reversed(original))) if iteration % 2 == 0
                                         else ((3, 0), original[:2]))
                    for position, bit in zip(positions, source):
                        expected[position] = function(original[position], bit)
                after = ''.join(map(str, reversed(expected)))
                self.check_initializers(f'bit[4] flags = "{pattern:04b}"; int i = 0; '
                    'int dstart = 0; int dstop = 3; int first = 3; int step = -1; int last = 0; '
                    'int left = 3; int right = 0; int start = 0; int stop = 1; '
                    f'while (i < 4) {{ if (i % 2 == 0) {{ flags[dstart:dstop] {symbol} flags[first:step:last]; }} '
                    f'else {{ flags[{{left,right}}] {symbol} flags[start:stop]; }} i += 1; }} '
                    f'bool same = flags == "{after}";', (), (), ('same', 'i'), (1, 4))

    def test_live_bit_iteration_observes_completed_updates(self):
        for is_set in (False, True):
            target = 'flags[{first,1,2,last}]' if is_set else 'flags[first:last]'
            for symbol, function in self.operations:
                initial, source = {'&=': (15, 0), '|=': (1, 15), '^=': (0, 15)}[symbol]
                expected = [(initial >> i) & 1 for i in range(4)]
                total = 0
                for i in range(4):
                    total += expected[i]
                    expected = [function(bit, (source >> j) & 1) for j, bit in enumerate(expected)]
                after = ''.join(map(str, reversed(expected)))
                self.check_initializers(f'const bit[4] C = "{source:04b}"; bit[4] flags = "{initial:04b}"; '
                    'int first = 0; int last = 3; int total = 0; for bit b in flags { '
                    f'total += int(b); {target} {symbol} C[first:last]; }} bool same = flags == "{after}";',
                    (), (), ('same', 'total'), (1, total))

    def test_nested_transfers_and_constant_shadows_restore_destination_names(self):
        for symbol, function in self.operations:
            expected = list(map(int, reversed('11001100')))
            source = [0, 1, 0, 1]
            visits = 0
            for outer in (0, 1, 2):
                for index in (1, 2):
                    positions = (outer, 7) if index == 1 else (outer, outer + 1)
                    for position, bit in zip(positions, source[index:index + 2]):
                        expected[position] = function(expected[position], bit)
                    visits += 1
            after = ''.join(map(str, reversed(expected)))
            self.check_initializers('const int LAST = 7; const bit[4] C = "1010"; bit[8] result = "11001100"; '
                'int first = 0; int last = 2; int i = -1; int visits = 0; '
                'for int LAST in [first:last] { for int i in [0:3] { if (i == 0) { continue; } '
                f'if (i == 1) {{ result[{{LAST,7}}] {symbol} C[i:i + 1]; }} '
                f'else {{ result[LAST:LAST + 1] {symbol} C[i:i + 1]; }} '
                'visits += 1; if (i == 2) { break; } } } '
                f'bool same = result == "{after}"; int restored = LAST;',
                (), (), ('same', 'visits', 'restored', 'i'), (1, visits, 7, -1))

    def test_skipped_invalid_destinations_and_sources_do_not_evaluate_values(self):
        assignments = (('result[{bad,0}]', 'start:1 / zero:stop'),
                       ('result[{left,left}]', 'start:stop'), ('result[left:0:right]', 'start:stop'),
                       ('result[left:right]', 'start:0:stop'), ('result[{left,right}]', 'bounds[bad]:stop'),
                       ('result[left:right]', 'start:stop + 1'))
        wrappers = ('if (false) {{ {} }}', 'while (false) {{ {} }}', 'for int outer in [1:0] {{ {} }}',
                    'for int outer in [first:last] {{ if (outer < 0) {{ {} }} }}',
                    'while (visits < 2) {{ visits += 1; continue; {} }}', 'while (true) {{ break; {} }}')
        for constant in (False, True):
            for symbol, _ in self.operations:
                for target, bounds in assignments:
                    for wrapper in wrappers:
                        self.check_initializers(('const ' if constant else '') + 'bit[4] source = "1010"; '
                            'bit[8] result = "11001100"; int left = 1; int right = 2; int start = 0; int stop = 1; '
                            'int bad = 9; int zero = 0; int first = 0; int last = 1; int visits = 0; '
                            'array[int,2] bounds = {0,1}; ' + wrapper.format(f'{target} {symbol} source[{bounds}];') +
                            'bool same = result == "11001100";', (), (), ('same',), (1,))

    def test_measured_values_choose_destinations_ranges_and_quantum_feedback(self):
        for is_set in (False, True):
            target = 'result[{left,right}]' if is_set else 'result[left:dstep:right]'
            for anti in (0, 1):
                for symbol, _ in self.operations:
                    for _ in range(8):
                        self.check_initializers('include "stdgates.inc"; qubit[2] q; reset q; h q[0]; cx q[0], q[1]; '
                            + ('x q[1]; ' if anti else '') +
                            'bit[2] measured = measure q; bit[8] original = "11001100"; bit[8] result = original; '
                            'int left = 7; int right = 1; int dstep = -6; int start = 0; int stop = 1; int step = 1; '
                            'if (measured[0]) { left = 1; right = 7; dstep = 6; } '
                            'if (measured[1]) { start = 1; stop = 0; step = -1; } '
                            f'{target} {symbol} measured[start:step:stop]; '
                            f'bool same = int(result[left]) == (int(original[left]) {symbol[0]} int(measured[start])) '
                            f'&& int(result[right]) == (int(original[right]) {symbol[0]} int(measured[stop])); '
                            'bool unchanged = (result & "01111101") == "01001100"; '
                            'if (result[left]) { x q[0]; } bit[2] observed = measure q; '
                            'bool feedback = bool(observed[0]) == (bool(measured[0]) != bool(result[left])) '
                            '&& observed[1] == measured[1];',
                            (), (), ('same', 'unchanged', 'feedback'), (1, 1, 1))

    def test_later_failures_preserve_the_previous_successful_result(self):
        cases = (('destination duplicate', 'left = -1; right = 7; dstep = 8;'),
                 ('destination bounds', 'left = 9; right = 10;'),
                 ('destination capacity', 'right = 3;'),
                 ('source count', 'stop = 0;'), ('source capacity', 'stop = 2;'),
                 ('source step', 'step = 0;'), ('source division', 'divisor = 0;'))
        for is_set in (False, True):
            target = 'result[{left,right}]' if is_set else 'result[left:dstep:right]'
            for constant in (False, True):
                for symbol, function in self.operations:
                    for failure, update in cases:
                        if is_set and failure == 'destination duplicate':
                            update = 'right = left;'
                        if is_set and failure == 'destination capacity':
                            continue  # The number of set entries is fixed.
                        expected = list(map(int, reversed('11001100')))
                        for position, bit in zip((1, 2), (0, 1)):
                            expected[position] = function(expected[position], bit)
                        after = ''.join(map(str, reversed(expected)))
                        with patch.object(self.module.QASM2QCXConverter, 'MAX_RUNTIME_BIT_RANGE_POSITIONS', 2):
                            lines = self.program(('const ' if constant else '') + 'bit[4] source = "1010"; '
                                'bit[8] result = "11001100"; int left = 1; int right = 2; int dstep = 1; '
                                'int start = 0; int stop = 1; int step = 1; int divisor = 1; int i = 0; '
                                f'while (i < 2) {{ {target} {symbol} source[start:step / divisor:stop]; i += 1; {update} }}')
                        writes = [i for i, line in enumerate(lines) if line.startswith('LET RESULT63:')
                                  and ' := QASM2QCX_RANGE_BITWISE_ORIGINAL_' in line]
                        self.assertEqual(len(writes), 2 if is_set else 1)
                        end = writes[-1] if is_set else next(i for i, line in enumerate(lines)
                            if line.startswith('JUMPIF QASM2QCX_RANGE_TO_RANGE_COPY_') and '_WRITE ' in line)
                        instrumented = []
                        for i, line in enumerate(lines):
                            if line.startswith('LET QASM2QCX_RANGE_COPY_SOURCE_'):
                                instrumented.append('PRINTLN 12345')
                            if i in writes:
                                instrumented.append('PRINTLN 67890')
                            instrumented.append(line)
                            if i == end:
                                instrumented.append('PRINTLN ' + ' '.join(f'RESULT63:{bit}' for bit in reversed(range(8))))
                        result = self.execute(instrumented)
                        self.assertNotEqual(result.returncode, 0)
                        self.assertIn('division by zero' if failure == 'source division' else 'assertion failed', result.stderr)
                        self.assertNotIn('runtime error:', result.stderr)
                        self.assertEqual([''.join(line.split()) for line in result.stdout.splitlines()],
                                         ['12345', '12345', '67890', '67890', after])
                        type(self).checked += 1

    def test_updated_angle_phase_parameters_work_with_and_without_fusion(self):
        import cmath
        import math
        from qasm2qcx_u_power_numerical import BackendAuditTests
        for target in ('result[{first,1,2,last}]', 'result[first:last]'):
            for symbol, function in self.operations:
                value = function(12, 10)
                source = ('include "stdgates.inc"; qubit q; h q; const bit[4] C = "1010"; '
                          'bit[4] result = "1100"; int first = 0; int last = 3; int visits = 0; '
                          f'while (visits < 1) {{ {target} {symbol} C[first:last]; visits += 1; }} '
                          'p(angle[4](result)) q;')
                expected = (1 / math.sqrt(2), cmath.exp(1j * value * math.pi / 8) / math.sqrt(2))
                for fused in (False, True):
                    BackendAuditTests.check(self, self.program(source, fused=fused), expected)

    def test_documented_runtime_selection_range_bitwise_example(self):
        document = (pathlib.Path(__file__).parents[1] / 'qcx/qasm2qcx.md').read_text(encoding='utf-8')
        examples = re.findall(r'<!-- runtime-selection-range-bitwise-example -->\s*```qasm\n(.*?)\n```',
                              document, flags=re.DOTALL)
        self.assertEqual(len(examples), 1)
        source = examples[0].removeprefix('OPENQASM 3.0;')
        self.check_initializers(source + '\nbool same = flags == "11111111" && one == "1";',
                                (), (), ('same', 'set_result', 'range_result', 'one'), (1, 1, 1, 1))

    def assert_failure(self, source, error, original_reads):
        lines = self.program(source)
        instrumented = []
        for line in lines:
            if line.startswith('LET QASM2QCX_RANGE_BITWISE_ORIGINAL_') and ' := RESULT63' in line:
                instrumented.append('PRINTLN 12345')
            if (line.startswith('LET QASM2QCX_RANGE_COPY_SOURCE_') or
                    (line.startswith('LET RESULT63:') and ' := QASM2QCX_RANGE_BITWISE_ORIGINAL_' in line)):
                instrumented.append('PRINTLN 67890')
            instrumented.append(line)
        self.assertIn('PRINTLN 12345', instrumented)
        self.assertIn('PRINTLN 67890', instrumented)
        result = self.execute(instrumented)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(error, result.stderr)
        self.assertNotIn('runtime error:', result.stderr)
        self.assertEqual(result.stdout.split(), ['12345'] * original_reads)
        type(self).checked += 1

    def test_destination_failures_precede_original_reads_and_rhs_division(self):
        targets = ('result[{index, 0}]', 'result[index:step:stop]')
        cases = ((-5, 1, -4), (4, 1, 5), (0, 0, 1), (1, -1, 2), (-1, 4, 3))
        for symbol, _ in self.operations:
            for target in targets:
                for index, step, stop in cases:
                    # Sets need a bad/duplicate entry; step/stop are irrelevant.
                    if '{' in target and index not in (-5, 4):
                        index = 0
                    self.assert_failure('const bit[4] C = "1010"; bit[4] result = "1100"; '
                        f'int index = {index}; int step = {step}; int stop = {stop}; int start = 0; int zero = 0; '
                        f'{target} {symbol} C[start:1 / zero:stop];', 'assertion failed', 0)

    def test_source_range_failures_allow_original_snapshot_but_no_rhs_reads_or_writes(self):
        cases = (('0', '0', '1'), ('1', '1', '0'), ('-5', '1', '-4'),
                 ('3', '1', '4'), ('0', '1', '0'), ('0', '1', '2'))
        for symbol, _ in self.operations:
            for target in ('result[{left, right}]', 'result[left:right]'):
                for start, step, stop in cases:
                    self.assert_failure('bit[4] source = "1010"; bit[4] result = "1100"; '
                        'int left = 1; int right = 2; '
                        f'int start = {start}; int step = {step}; int stop = {stop}; '
                        f'{target} {symbol} source[start:step:stop];', 'assertion failed', 2)
                self.assert_failure('bit[4] source = "1010"; bit[4] result = "1100"; '
                    'int left = 1; int right = 2; int start = 0; int stop = 1; int zero = 0; '
                    f'{target} {symbol} source[start:1 / zero:stop];', 'division by zero', 2)


def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    args = parser.parse_args()
    BackendTests.bra = args.bra.resolve()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(BackendTests))
    print(f'runtime selection range bitwise programs checked: {BackendTests.checked}')
    raise SystemExit(not result.wasSuccessful())


if __name__ == '__main__':
    main()
