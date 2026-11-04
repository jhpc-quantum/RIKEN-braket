#!/usr/bin/env python3
"""Audit runtime bit-register range-to-range copies (#462)."""

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

    @staticmethod
    def declarations(values):
        return ''.join(f'{"uint" if value >= 0 else "int"} {name} = {value};' for name, value in values)

    def test_independent_ranges_preserve_order_sources_and_unselected_destination_bits(self):
        for width in (1, 4, 33, 65, 129):
            sources = {(0, 1, width - 1), (width - 1, -1, 0), (-1, -1, -width),
                       (0, 2, width - 1), (-1, 1, width - 1), (-width, 1, width - 1)}
            for sstart, sstep, sstop in sorted(sources):
                raw_source = list(range(sstart, sstop + (1 if sstep > 0 else -1), sstep))
                count = len(raw_source)
                target_width = 2 * count + 3
                before = ('01101001' * target_width)[:target_width]
                destinations = ((1, 1, count), (target_width - 2, -2, 2),
                                (-target_width + 1, 2, -target_width + 1 + 2 * (count - 1)))
                for dstart, dstep, dstop in destinations:
                    raw_destination = list(range(dstart, dstop + (1 if dstep > 0 else -1), dstep))
                    self.assertEqual(len(raw_destination), count)
                    positions = [index % target_width for index in raw_destination]
                    declarations = self.declarations((('dstart', dstart), ('dstep', dstep), ('dstop', dstop),
                                                      ('sstart', sstart), ('sstep', sstep), ('sstop', sstop)))
                    for pattern in sorted({'0' * width, '1' * width, ('100110' * width)[:width]}):
                        bits = list(map(int, reversed(pattern)))
                        values = [bits[index % width] for index in raw_source]
                        expected = list(map(int, reversed(before)))
                        for position, value in zip(positions, values):
                            expected[position] = value
                        text = ''.join(map(str, reversed(expected)))
                        for constant in (False, True):
                            self.check_initializers(f'{"const " if constant else ""}bit[{width}] source = "{pattern}"; '
                                f'bit[{target_width}] result = "{before}"; ' + declarations +
                                'result[dstart:dstep:dstop] = source[sstart:sstep:sstop]; '
                                f'bool same = result == "{text}"; bool unchanged = source == "{pattern}";',
                                (), (), ('same', 'unchanged'), (1, 1))

    def test_overlapping_self_copies_use_original_bits_and_captured_ranges(self):
        for pattern in range(16):
            bits = [(pattern >> i) & 1 for i in range(4)]
            cases = ((1, 1, 3, 0, 1, 2), (3, -1, 0, 0, 1, 3),
                     (0, 1, 3, -1, -1, -4), (-1, -1, -3, 0, 1, 2), (0, 1, 1, -1, 4, 3))
            for dstart, dstep, dstop, sstart, sstep, sstop in cases:
                positions = list(range(dstart, dstop + (1 if dstep > 0 else -1), dstep))
                raw = list(range(sstart, sstop + (1 if sstep > 0 else -1), sstep))
                values = [bits[index % 4] for index in raw]
                expected = bits.copy()
                for position, value in zip(positions, values):
                    expected[position % 4] = value
                text = ''.join(map(str, reversed(expected)))
                declarations = self.declarations((('dstart', dstart), ('dstep', dstep), ('dstop', dstop),
                                                  ('sstart', sstart), ('sstep', sstep), ('sstop', sstop)))
                self.check_initializers(f'bit[4] flags = "{pattern:04b}"; ' + declarations +
                    'flags[dstart:dstep:dstop] = flags[sstart:sstep:sstop]; '
                    f'bool same = flags == "{text}";', (), (), ('same',), (1,))
        self.check_initializers('bit[4] flags = "0101"; int first = 0; int last = 1; '
            'flags[int(flags[first]):int(flags[last]) + 3] = flags[first:last + 1]; '
            'bool same = flags == "1011";', (), (), ('same',), (1,))

    def test_default_steps_singleton_storage_and_repeated_scalar_storage_source(self):
        for constant in (False, True):
            for value in ('0', '1'):
                self.check_initializers(f'{"const " if constant else ""}bit[1] source = "{value}"; '
                    'bit[1] one = "0"; bit[8] result = "00000000"; int low = -1; uint high = 7; '
                    'one[low:low] = source[low:low]; result[high:high] = source[low:low]; '
                    f'bool same = one == "{value}" && result == "{value}' + '0' * 7 + '";',
                    (), (), ('same',), (1,))
                repeated = value * 2
                self.check_initializers(f'{"const " if constant else ""}bit[1] source = "{value}"; '
                    'bit[2] result = "00"; int low = 0; int high = 1; int first = -1; int last = 0; '
                    f'result[low:high] = source[first:last]; bool same = result == "{repeated}";',
                    (), (), ('same',), (1,))

    def test_native_endpoint_steps_and_unselected_stops_on_both_sides(self):
        limits = self.module.QASM2QCXConverter
        steps = (('uint', limits.QCX_UINT_MAX, 0, limits.QCX_UINT_MAX - 1),
                 ('int', limits.QCX_INT_MIN, 0, -4), ('int', limits.QCX_INT_MAX, -1, 3))
        for dkind, dstep, dstart, dstop in steps:
            for skind, sstep, sstart, sstop in steps:
                expected = ['0'] * 4
                expected[dstart % 4] = '1' if sstart == -1 else '0'
                text = ''.join(reversed(expected))
                self.check_initializers('bit[4] source = "1010"; bit[4] result = "0000"; '
                    f'int dstart = {dstart}; {dkind} dstep = {dstep}; {dkind} dstop = {dstop}; '
                    f'int sstart = {sstart}; {skind} sstep = {sstep}; {skind} sstop = {sstop}; '
                    'result[dstart:dstep:dstop] = source[sstart:sstep:sstop]; '
                    f'bool same = result == "{text}";', (), (), ('same',), (1,))

    def test_small_capacity_allows_short_copies_on_wide_registers_and_bounds_all_buffers(self):
        for constant in (False, True):
            with patch.object(self.module.QASM2QCXConverter, 'MAX_RUNTIME_BIT_RANGE_POSITIONS', 3):
                self.check_initializers(f'{"const " if constant else ""}bit[129] source = "1' + '0' * 127 + '1"; '
                    'bit[129] result = "' + '0' * 129 + '"; int dstart = -2; int dstop = -1; '
                    'int sstart = 127; uint sstop = 128; result[dstart:dstop] = source[sstart:sstop]; '
                    'bool same = result == "1' + '0' * 128 + '";', (), (), ('same',), (1,))
        for dstart, dstop, sstart, sstop in ((0, 3, 0, 3), (0, 1, 0, 3), (0, 1, 0, 2)):
            with patch.object(self.module.QASM2QCXConverter, 'MAX_RUNTIME_BIT_RANGE_POSITIONS', 3):
                lines = self.program('bit[8] source = "10101010"; bit[8] result = "11001100"; '
                    f'int dstart = {dstart}; int dstop = {dstop}; int sstart = {sstart}; int sstop = {sstop}; '
                    'result[dstart:dstop] = source[sstart:sstop];')
            self.assert_failure_before_reads_or_writes(lines, 'assertion failed')

    def assert_failure_before_reads_or_writes(self, lines, error):
        instrumented = []
        for line in lines:
            if (line.startswith('LET QASM2QCX_RANGE_COPY_SOURCE_')
                    or (line.startswith('LET RESULT63') and ' := QASM2QCX_RANGE_COPY_SOURCE_' in line)):
                instrumented.append('PRINTLN 67890')
            instrumented.append(line)
        self.assertIn('PRINTLN 67890', instrumented)
        result = self.execute(instrumented)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(error, result.stderr)
        self.assertNotIn('runtime error:', result.stderr)
        self.assertNotIn('67890', result.stdout)
        type(self).checked += 1

    def test_loop_families_recapture_both_ranges_and_preserve_live_iteration(self):
        loops = (('for int i in [0:3]', 'i', '00001010'),
                 ('for int i in [first:last]', 'i', '00001010'),
                 ('for int i in [ufirst:ulast]', 'i', '00001010'),
                 ('for int i in [first:step:last]', 'i', '00001010'),
                 ('for int i in {0, 1, 2, 3}', 'i', '00001010'),
                 ('for int i in {first, first + 1, last - 1, last}', 'i', '00001010'),
                 ('for int i in indices', 'i', '00001010'),
                 ('for uint i in unsigned_indices', 'i', '00001010'),
                 ('for bit i in flags', 'int(i)', '00000010'),
                 ('for bit i in C', 'int(i)', '00000010'))
        for loop, index, expected in loops:
            for source in ('flags', 'C'):
                self.check_initializers('const bit[4] C = "1010"; bit[4] flags = "1010"; '
                    'bit[8] result = "00000000"; int first = 0; int last = 3; uint ufirst = 0; uint ulast = 3; '
                    'int step = 1; int i = -1; int total = 0; array[int, 4] indices = {0, 1, 2, 3}; '
                    'array[uint, 4] unsigned_indices = {0, 1, 2, 3}; ' + loop +
                    f' {{ result[{index}:{index}] = {source}[{index}:{index}]; total += int(result[{index}]); }} '
                    f'bool same = result == "{expected}"; bool unchanged = flags == "1010";',
                    (), (), ('same', 'unchanged', 'total', 'i'), (1, 1, 2, -1))
        self.check_initializers('const bit[4] C = "1111"; bit[4] flags = "0001"; '
            'int first = 0; int last = 3; int total = 0; '
            'for bit b in flags { total += int(b); flags[first:last] = C[first:last]; } '
            'bool same = flags == "1111";', (), (), ('same', 'total'), (1, 4))

    def test_reentry_changes_counts_directions_source_values_and_both_position_tables(self):
        pattern, before = '10110110', '01011001'
        for constant in (False, True):
            for rounds in range(1, 7):
                source_bits = list(map(int, reversed(pattern)))
                expected = list(map(int, reversed(before)))
                count, dstart, dstep, sstart, sstep = 1, 0, 2, 0, 1
                for _ in range(rounds):
                    values = [source_bits[sstart + j * sstep] for j in range(count)]
                    for j, value in enumerate(values):
                        expected[dstart + j * dstep] = value
                    if not constant:
                        source_bits[0] ^= 1
                        source_bits[1] ^= 1
                    count = count % 3 + 1
                    dstart, dstep, sstart, sstep = 7 - dstart, -dstep, 7 - sstart, -sstep
                text = ''.join(map(str, reversed(expected)))
                final_source = ''.join(map(str, reversed(source_bits)))
                self.check_initializers(('const ' if constant else '') + f'bit[8] source = "{pattern}"; '
                    f'bit[8] result = "{before}"; int count = 1; int dstart = 0; int dstep = 2; '
                    'int dstop = 0; int sstart = 0; int sstep = 1; int sstop = 0; int i = 0; '
                    f'while (i < {rounds}) {{ dstop = dstart + (count - 1) * dstep; '
                    'sstop = sstart + (count - 1) * sstep; result[dstart:dstep:dstop] = source[sstart:sstep:sstop]; '
                    + ('' if constant else 'source ^= "00000011"; ') +
                    'count = count % 3 + 1; dstart = 7 - dstart; dstep = -dstep; '
                    'sstart = 7 - sstart; sstep = -sstep; i += 1; } '
                    f'bool same = result == "{text}"; bool source_ok = source == "{final_source}";',
                    (), (), ('same', 'source_ok', 'i', 'count'), (1, 1, rounds, count))
        self.check_initializers('bit[4] source = "1010"; bit[8] result = "00000000"; '
            'array[int, 2] destinations = {0, 7}; array[int, 2] sources = {0, 1}; '
            'int dstep = 7; int sstep = 1; int i = 0; while (i < 2) { '
            'result[destinations[0]:dstep:destinations[1]] = source[sources[0]:sstep:sources[1]]; '
            'destinations[0] = 7; destinations[1] = 0; dstep = -7; '
            'sources[0] = -1; sources[1] = 3; sstep = 4; i += 1; } '
            'bool same = result == "10000001";', (), (), ('same', 'i'), (1, 2))
        self.check_initializers('bit[4] source = "1010"; bit[1] one = "0"; int low = -1; int i = 0; int total = 0; '
            'while (i < 4) { one[low:low] = source[i:i]; total += int(one[0]); i += 1; } '
            'bool same = one == "1";', (), (), ('same', 'total', 'i'), (1, 2, 4))

    def test_nested_transfers_shadowing_and_differently_sized_branch_copies(self):
        self.check_initializers('const int LAST = 7; const bit[4] C = "1010"; bit[8] result = "00000000"; '
            'int first = 0; int last = 2; int start = -1; int index = 0; int visits = 0; '
            'for int LAST in [first:last] { for int start in [0:3] { if (start == 0) { continue; } '
            'result[LAST:LAST] = C[start:start]; visits += 1; break; } } '
            'for int C in {0, 1} {} result[LAST - index:LAST - index] = C[first + 1:first + 1]; '
            'bool same = result == "10000111"; int restored = LAST;',
            (), (), ('same', 'visits', 'restored', 'start'), (1, 3, 7, -1))
        self.check_initializers('const bit[4] C = "1010"; bit[8] result = "00000000"; '
            'int i = 0; int index = 0; int dstart = 0; int dstop = 1; int sstart = 0; int sstop = 1; '
            'while (i < 4) { if (i % 2 == 0) { dstart = index; dstop = index + 1; '
            'result[dstart:dstop] = C[sstart:sstop]; } else { dstart = index + 1; dstop = dstart; '
            'result[dstart:dstop] = C[sstop:sstop]; } index += 1; i += 1; } '
            'bool same = result == "00011010";', (), (), ('same', 'i'), (1, 4))
        self.check_initializers('bit[4] source = "1010"; bit[8] result = "11001100"; int i = 0; '
            'while (i < 4) { i += 1; if (i == 2) { continue; } '
            'result[i - 1:i - 1] = source[i - 1:i - 1]; if (i == 3) { break; } } '
            'bool same = result == "11001000";', (), (), ('same', 'i'), (1, 3))

    def test_skipped_destination_and_source_failures_preserve_registers(self):
        ranges = (('dstart:0:dstop', 'sstart:1 / zero:sstop'),
                  ('bad:dstep:dstop', 'sstart:sstep:sstop'),
                  ('dstop:1:dstart', 'sstart:sstep:sstop'),
                  ('-1:8:dstop', 'sstart:sstep:sstop'),
                  ('dstart:1 / zero:dstop', 'sstart:sstep:sstop'),
                  ('bounds[bad]:dstep:dstop', 'sstart:sstep:sstop'),
                  ('dstart:dstep:dstop', 'sstart:0:sstop'),
                  ('dstart:dstep:dstop', 'sstart:sstart'),
                  ('dstart:dstep:dstop', 'sstart:1 / zero:sstop'),
                  ('dstart:dstep:dstop', 'sstart:sstep:bad'))
        wrappers = ('if (false) {{ {} }}', 'while (false) {{ {} }}', 'for int outer in [1:0] {{ {} }}',
                    'for int outer in [first:last] {{ if (outer < 0) {{ {} }} }}',
                    'while (visits < 2) {{ visits += 1; continue; {} }}', 'while (true) {{ break; {} }}')
        for constant in (False, True):
            for destination, source in ranges:
                for wrapper in wrappers:
                    self.check_initializers(('const ' if constant else '') + 'bit[4] source = "1010"; '
                        'bit[8] result = "11001100"; int dstart = 0; int dstep = 1; int dstop = 7; '
                        'int sstart = 0; int sstep = 1; int sstop = 1; int zero = 0; int bad = 8; '
                        'int first = 0; int last = 1; int visits = 0; array[int, 2] bounds = {0, 7}; '
                        + wrapper.format(f'result[{destination}] = source[{source}];') +
                        'bool same = result == "11001100"; bool unchanged = source == "1010";',
                        (), (), ('same', 'unchanged'), (1, 1))

    def test_measurements_choose_both_ranges_and_copied_bits_drive_feedback(self):
        for anti in (0, 1):
            for _ in range(16):
                self.check_initializers('include "stdgates.inc"; qubit[2] q; reset q; h q[0]; cx q[0], q[1]; '
                    + ('x q[1]; ' if anti else '') +
                    'bit[2] measured = measure q; bit[8] result = "01010101"; bit[1] one = "0"; '
                    'int dstart = 6; int dstep = -5; int dstop = 1; int sstart = 0; int sstep = 1; int sstop = 1; '
                    'int high = -1; int low = 0; '
                    'if (measured[0]) { dstart = 1; dstep = 5; dstop = 6; sstart = 1; sstep = -1; sstop = 0; } '
                    'result[dstart:dstep:dstop] = measured[sstart:sstep:sstop]; '
                    'result[high:high] = measured[sstart:sstart]; one[low:low] = measured[sstop:sstop]; '
                    'bool ordered = result[dstart] == measured[sstart] && result[dstop] == measured[sstop]; '
                    f'bool correlation = (result[dstart] != result[dstop]) == bool({anti}); '
                    'bool unchanged = (result & "00111101") == "00010101"; '
                    'bool singleton = one[0] == result[dstop]; '
                    'if (result[high]) { if (sstart == 0) { x q[0]; } else { x q[1]; } } '
                    'bit[2] observed = measure q; '
                    'bool feedback = !observed[sstart] && observed[sstop] == measured[sstop];',
                    (), (), ('ordered', 'correlation', 'unchanged', 'singleton', 'feedback'), (1, 1, 1, 1, 1))

    def test_later_failures_preserve_previous_copy_and_destination_error_priority(self):
        expected = list(reversed('11001100'))
        for position, value in zip((6, 1), ('0', '1')):
            expected[position] = value
        completed = ''.join(reversed(expected))
        cases = (('dstep = 0;', 'assertion failed', 8),
                 ('dstart = -9;', 'assertion failed', 8),
                 ('dstart = -1; dstep = 8; dstop = 7;', 'assertion failed', 8),
                 ('dstart = 1; dstop = 6;', 'assertion failed', 8),
                 ('dstop = -4;', 'assertion failed', 8),
                 ('sstop = 0;', 'assertion failed', 8), ('sstop = 2;', 'assertion failed', 8),
                 ('sstep = 0;', 'assertion failed', 8), ('sstart = 3; sstop = 4;', 'assertion failed', 8),
                 ('source_divisor = 0;', 'division by zero', 8),
                 ('destination_divisor = 0;', 'division by zero', 8),
                 ('dstart = -1; dstep = 8; dstop = 7; source_divisor = 0;', 'assertion failed', 8),
                 ('dstart = -9; source_divisor = 0;', 'assertion failed', 8),
                 ('dstart = 0; dstep = 1; dstop = 3; sstop = 3;', 'assertion failed', 3),
                 ('sstop = 3;', 'assertion failed', 3))
        for update, error, capacity in cases:
            with patch.object(self.module.QASM2QCXConverter, 'MAX_RUNTIME_BIT_RANGE_POSITIONS', capacity):
                lines = self.program('bit[4] source = "1010"; bit[8] result = "11001100"; '
                    'int dstart = 6; int dstep = -5; int dstop = 1; int destination_divisor = 1; '
                    'int sstart = 0; int sstep = 1; int sstop = 1; int source_divisor = 1; int i = 0; '
                    'while (i < 2) { result[dstart:dstep / destination_divisor:dstop] = '
                    f'source[sstart:sstep / source_divisor:sstop]; i += 1; {update} }}')
            writes = [i for i, line in enumerate(lines) if line.startswith('LET RESULT63:')
                      and ' := QASM2QCX_RANGE_COPY_SOURCE_' in line]
            self.assertEqual(len(writes), 1)
            write_end = next(i for i, line in enumerate(lines)
                             if line.startswith('JUMPIF QASM2QCX_RANGE_TO_RANGE_COPY_') and '_WRITE ' in line)
            instrumented = []
            for i, line in enumerate(lines):
                if line.startswith('LET QASM2QCX_RANGE_COPY_SOURCE_'):
                    instrumented.append('PRINTLN 12345')
                if i in writes:
                    instrumented.append('PRINTLN 67890')
                instrumented.append(line)
                if i == write_end:
                    instrumented.append('PRINTLN ' + ' '.join(f'RESULT63:{bit}' for bit in reversed(range(8))))
            result = self.execute(instrumented)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(error, result.stderr)
            self.assertNotIn('runtime error:', result.stderr)
            self.assertEqual([''.join(line.split()) for line in result.stdout.splitlines()],
                             ['12345', '12345', '67890', '67890', completed])
            type(self).checked += 1

    def test_range_to_range_copies_feed_phase_parameters_and_fusion(self):
        import cmath
        import math
        from qasm2qcx_u_power_numerical import BackendAuditTests
        source = ('include "stdgates.inc"; qubit q; h q; bit[4] source = "1010"; bit[4] result = "0000"; '
                  'int dstart = 3; int dstep = -1; int dstop = 0; int sstart = 0; int sstop = 3; int visits = 0; '
                  'while (visits < 1) { result[dstart:dstep:dstop] = source[sstart:sstop]; visits += 1; } '
                  'p(angle[4](result)) q; dstart = 0; dstep = 1; dstop = 3; '
                  'result[dstart:dstep:dstop] = source[sstart:sstop];')
        expected = (1 / math.sqrt(2), cmath.exp(1j * 5 * math.pi / 8) / math.sqrt(2))
        for fused in (False, True):
            BackendAuditTests.check(self, self.program(source, fused=fused), expected)

    def test_destination_errors_precede_source_bound_evaluation(self):
        cases = ((0, 0, 1), (2, 1, 0), (-5, 1, -4), (3, 1, 4), (-1, 4, 3), (-1, 1, 3))
        for dstart, dstep, dstop in cases:
            lines = self.program('bit[4] source = "1010"; bit[4] result = "1100"; '
                f'int dstart = {dstart}; int dstep = {dstep}; int dstop = {dstop}; '
                'int sstart = 0; int sstop = 1; int zero = 0; '
                'result[dstart:dstep:dstop] = source[sstart:1 / zero:sstop];')
            source_division = next(i for i, line in enumerate(lines) if '/= ZERO15' in line)
            lines.insert(source_division, 'PRINTLN 12345')
            instrumented = []
            for line in lines:
                if line.startswith('LET RESULT63:') and ' := QASM2QCX_RANGE_COPY_SOURCE_' in line:
                    instrumented.append('PRINTLN 67890')
                instrumented.append(line)
            result = self.execute(instrumented)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('assertion failed', result.stderr)
            self.assertNotIn('runtime error:', result.stderr)
            self.assertNotIn('12345', result.stdout)
            self.assertNotIn('67890', result.stdout)
            type(self).checked += 1

    def test_source_errors_and_count_mismatches_precede_all_reads_and_writes(self):
        for sstart, sstep, sstop in ((0, 0, 1), (2, 1, 0), (-5, 1, -4), (3, 1, 4),
                                    (0, 1, 0), (0, 1, 2)):
            lines = self.program('bit[4] source = "1010"; bit[4] result = "1100"; '
                'int dstart = 1; int dstop = 2; '
                f'int sstart = {sstart}; int sstep = {sstep}; int sstop = {sstop}; '
                'result[dstart:dstop] = source[sstart:sstep:sstop];')
            self.assert_failure_before_reads_or_writes(lines, 'assertion failed')
        lines = self.program('bit[4] source = "1010"; bit[4] result = "1100"; '
            'int dstart = 1; int dstop = 2; int sstart = 0; int sstop = 1; int zero = 0; '
            'result[dstart:dstop] = source[sstart:1 / zero:sstop];')
        self.assert_failure_before_reads_or_writes(lines, 'division by zero')


def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    args = parser.parse_args()
    BackendTests.bra = args.bra.resolve()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(BackendTests))
    print(f'runtime range-to-range copy programs checked: {BackendTests.checked}')
    raise SystemExit(not result.wasSuccessful())


if __name__ == '__main__':
    main()
