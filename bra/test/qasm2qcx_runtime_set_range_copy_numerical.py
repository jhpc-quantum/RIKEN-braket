#!/usr/bin/env python3
"""Audit runtime bit-range copies into runtime index-set destinations (#460)."""

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

    def test_mixed_runtime_indices_preserve_order_sources_and_unselected_bits(self):
        for width in (1, 4, 33, 65, 129):
            ranges = {(0, 1, width - 1), (width - 1, -1, max(0, width - 3)),
                      (-1, -1, max(-width, -3)), (0, 2, min(width - 1, 5)),
                      (-1, 1, min(width - 1, 1))}
            for start, step, stop in sorted(ranges):
                raw = list(range(start, stop + (1 if step > 0 else -1), step))
                count = len(raw)
                target_width = 2 * count + 1
                before = ('01101001' * target_width)[:target_width]
                odd = tuple(2 * i + 1 for i in range(count))
                layouts = (tuple(range(count)), odd[1:] + odd[:1])
                declarations = ''.join(f'{"uint" if value >= 0 else "int"} {name} = {value};'
                    for name, value in (('start', start), ('step', step), ('stop', stop)))
                for pattern in sorted({'0' * width, '1' * width, ('100110' * width)[:width]}):
                    bits = list(map(int, reversed(pattern)))
                    values = [bits[index % width] for index in raw]
                    for constant in (False, True):
                        for positions in layouts:
                            expected = list(map(int, reversed(before)))
                            for index, value in zip(positions, values):
                                expected[index] = value
                            text = ''.join(map(str, reversed(expected)))
                            signed = ','.join(str(position - target_width) for position in positions)
                            unsigned = ','.join(map(str, positions))
                            entries = ','.join(f'{"signed_indices" if i % 2 == 0 else "unsigned_indices"}[{i}]'
                                               for i in range(count))
                            self.check_initializers(f'{"const " if constant else ""}bit[{width}] source = "{pattern}"; '
                                f'bit[{target_width}] result = "{before}"; '
                                f'array[int, {count}] signed_indices = {{{signed}}}; '
                                f'array[uint, {count}] unsigned_indices = {{{unsigned}}}; ' + declarations +
                                f'result[{{{entries}}}] = source[start:step:stop]; '
                                f'bool same = result == "{text}"; bool unchanged = source == "{pattern}";',
                                (), (), ('same', 'unchanged', 'start', 'step', 'stop'), (1, 1, start, step, stop))

    def test_overlapping_self_copies_snapshot_sources_before_writes(self):
        for pattern in range(16):
            bits = [(pattern >> i) & 1 for i in range(4)]
            for start, step, stop, positions in ((0, 1, 2, (1, 2, 3)),
                                                (3, -1, 0, (2, 0, 3, 1)),
                                                (-1, -1, -4, (0, 1, 2, 3)),
                                                (-1, 4, 3, (3, 0))):
                raw = list(range(start, stop + (1 if step > 0 else -1), step))
                values = [bits[index % 4] for index in raw]
                expected = bits.copy()
                for index, value in zip(positions, values):
                    expected[index] = value
                text = ''.join(map(str, reversed(expected)))
                declarations = ''.join(f'int d{i} = {position - 4};' for i, position in enumerate(positions))
                entries = ','.join(f'd{i}' for i in range(len(positions)))
                self.check_initializers(f'bit[4] flags = "{pattern:04b}"; ' + declarations +
                    f'int start = {start}; int step = {step}; int stop = {stop}; '
                    f'flags[{{{entries}}}] = flags[start:step:stop]; bool same = flags == "{text}";',
                    (), (), ('same',), (1,))

    def test_self_referential_destination_indices_use_original_values(self):
        for pattern in range(16):
            bits = [(pattern >> i) & 1 for i in range(4)]
            positions = (bits[0], bits[1] + 2)
            values = bits[:2]
            expected = bits.copy()
            for index, value in zip(positions, values):
                expected[index] = value
            text = ''.join(map(str, reversed(expected)))
            self.check_initializers(f'bit[4] flags = "{pattern:04b}"; int start = 0; int stop = 1; '
                'flags[{int(flags[0]), int(flags[1]) + 2}] = flags[start:stop]; '
                f'bool same = flags == "{text}";', (), (), ('same',), (1,))

    def test_singletons_and_mixed_constant_entries(self):
        for constant in (False, True):
            for value in ('0', '1'):
                self.check_initializers(f'{"const " if constant else ""}bit[1] source = "{value}"; '
                    'bit[1] one = "0"; bit[8] result = "00000000"; int low = -1; uint high = 7; '
                    'one[{low}] = source[low:low]; result[{high}] = source[low:low]; '
                    f'bool same = one == "{value}" && result == "{value}' + '0' * 7 + '";',
                    (), (), ('same',), (1,))
        self.check_initializers('const bit[4] C = "1010"; bit[8] result = "00000000"; '
            'int index = -1; int start = -1; int stop = 1; '
            'result[{0, index, 3}] = C[start:stop]; bool same = result == "00001001";',
            (), (), ('same', 'index'), (1, -1))

    def test_wide_constant_word_boundaries_and_partial_final_words(self):
        native = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        for width in (native, native + 1, 2 * native + 1, 129):
            for position in sorted({0, native - 1, min(native, width - 1), width - 1}):
                pattern = f'{1 << position:0{width}b}'
                self.check_initializers(f'const bit[{width}] C = "{pattern}"; '
                    'bit[129] result = "' + '0' * 129 + f'"; int index = -1; int source_index = {position - width}; '
                    'result[{index}] = C[source_index:source_index]; '
                    'bool same = result == "1' + '0' * 128 + '";', (), (), ('same',), (1,))

    def test_native_source_steps_and_repeated_source_positions(self):
        limits = self.module.QASM2QCXConverter
        for kind, step, start, stop in (('uint', limits.QCX_UINT_MAX, 0, limits.QCX_UINT_MAX - 1),
                                        ('int', limits.QCX_INT_MIN, 0, -4),
                                        ('int', limits.QCX_INT_MAX, -1, 3)):
            expected = '10000000' if start == -1 else '00000000'
            self.check_initializers('bit[4] source = "1010"; bit[8] result = "00000000"; uint index = 7; '
                f'int start = {start}; {kind} step = {step}; {kind} stop = {stop}; '
                f'result[{{index}}] = source[start:step:stop]; bool same = result == "{expected}";',
                (), (), ('same',), (1,))
        self.check_initializers('bit[4] source = "1010"; bit[8] result = "00000000"; '
            'int index = 0; int start = -1; int stop = 3; '
            'result[{index, 1, 2, 3, 4}] = source[start:stop]; bool same = result == "00010101";',
            (), (), ('same',), (1,))

    def test_loop_families_capture_fresh_destination_indices_and_sources(self):
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
                    f' {{ result[{{{index}}}] = {source}[{index}:{index}]; total += int(result[{index}]); }} '
                    f'bool same = result == "{expected}"; bool unchanged = flags == "1010";',
                    (), (), ('same', 'unchanged', 'total', 'i'), (1, 1, 2, -1))
        self.check_initializers('const bit[4] C = "1111"; bit[4] flags = "0001"; '
            'int first = 0; int start = 0; int stop = 3; int total = 0; '
            'for bit b in flags { total += int(b); flags[{first, 1, 2, 3}] = C[start:stop]; } '
            'bool same = flags == "1111";', (), (), ('same', 'total'), (1, 4))

    def test_while_reentry_refreshes_addresses_directions_and_source_values(self):
        for constant in (False, True):
            self.check_initializers(('const ' if constant else '') + 'bit[4] source = "1010"; '
                'bit[8] result = "00000000"; int left = 0; uint right = 7; '
                'int start = 0; int stop = 1; int step = 1; int i = 0; '
                'while (i < 3) { result[{left, right}] = source[start:step:stop]; '
                'left += 1; right -= 1; start = 1 - start; stop = 1 - stop; step = -step; i += 1; } '
                'bool same = result == "10100010";', (), (), ('same', 'left', 'right', 'i'), (1, 3, 4, 3))
            self.check_initializers(('const ' if constant else '') + 'bit[4] source = "1010"; '
                'bit[8] result = "00000000"; int left = 0; int right = 7; '
                'int start = 0; int stop = 1; int step = 1; int i = 0; '
                'while (i < 2) { result[{left, right}] = source[start:step:stop]; '
                'left = 7 - left; right = 7 - right; start = -1; stop = 3; step = 4; i += 1; } '
                'bool same = result == "10000001";', (), (), ('same', 'i'), (1, 2))
        self.check_initializers('bit[4] source = "1001"; bit[8] result = "00000000"; '
            'int left = 5; int right = 3; int start = 0; int stop = 1; int i = 0; '
            'while (i < 4) { result[{left, right}] = source[start:stop]; source ^= "0011"; '
            'left = 8 - left; right = 8 - right; i += 1; } '
            'bool same = result == "00100000" && source == "1001";', (), (), ('same', 'i'), (1, 4))
        self.check_initializers('bit[4] source = "1010"; bit[8] result = "00000000"; '
            'array[int, 2] addresses = {0, 7}; array[int, 2] bounds = {0, 1}; int step = 1; int i = 0; '
            'while (i < 3) { result[{addresses[0], addresses[1]}] = source[bounds[0]:step:bounds[1]]; '
            'addresses[0] += 1; addresses[1] -= 1; bounds[0] = 1 - bounds[0]; '
            'bounds[1] = 1 - bounds[1]; step = -step; i += 1; } '
            'bool same = result == "10100010";', (), (), ('same', 'i'), (1, 3))
        self.check_initializers('bit[4] source = "1010"; bit[1] one = "0"; int i = 0; int total = 0; '
            'while (i < 4) { one[{i - i}] = source[i:i]; total += int(one[0]); i += 1; } '
            'bool same = one == "1";', (), (), ('same', 'total', 'i'), (1, 2, 4))

    def test_nested_transfers_shadowing_and_differently_sized_branch_copies(self):
        self.check_initializers('const int LAST = 7; const bit[4] C = "1010"; bit[8] result = "00000000"; '
            'int first = 0; int last = 2; int start = -1; int index = 5; int visits = 0; '
            'for int LAST in [first:last] { for int start in [0:3] { if (start == 0) { continue; } '
            'result[{LAST, 4}] = C[start:start + 1]; visits += 1; break; } } '
            'for int C in {0, 1} {} result[{LAST, index}] = C[first:first + 1]; '
            'bool same = result == "00100111"; int restored = LAST;',
            (), (), ('same', 'visits', 'restored', 'start'), (1, 3, 7, -1))
        self.check_initializers('const bit[4] C = "1010"; bit[8] result = "00000000"; '
            'int i = 0; int index = 0; int start = 0; int stop = 1; '
            'while (i < 4) { if (i % 2 == 0) { result[{index, 7}] = C[start:stop]; } '
            'else { result[{index + 1}] = C[stop:stop]; } index += 1; i += 1; } '
            'bool same = result == "10010000";', (), (), ('same', 'i', 'index'), (1, 4, 4))
        self.check_initializers('bit[4] source = "1010"; bit[8] result = "11001100"; int i = 0; '
            'while (i < 4) { i += 1; if (i == 2) { continue; } '
            'result[{i - 1}] = source[i - 1:i - 1]; if (i == 3) { break; } } '
            'bool same = result == "11001000";', (), (), ('same', 'i'), (1, 3))

    def test_skipped_destination_and_source_failures_preserve_registers(self):
        statements = ('result[{bad, 0}] = source[start:1 / zero:stop];',
                      'result[{index, index}] = source[start:stop];',
                      'result[{-1, index}] = source[start:stop];',
                      'result[{index / zero, 0}] = source[start:stop];',
                      'result[{addresses[bad], 0}] = source[start:stop];',
                      'result[{index, 0}] = source[start:0:stop];',
                      'result[{index, 0}] = source[start:start];',
                      'result[{index, 0}] = source[start:1 / zero:stop];',
                      'result[{index, 0}] = source[bad:stop];')
        wrappers = ('if (false) {{ {} }}', 'while (false) {{ {} }}', 'for int outer in [1:0] {{ {} }}',
                    'for int outer in [first:last] {{ if (outer < 0) {{ {} }} }}',
                    'while (visits < 2) {{ visits += 1; continue; {} }}', 'while (true) {{ break; {} }}')
        for constant in (False, True):
            for statement in statements:
                for wrapper in wrappers:
                    self.check_initializers(('const ' if constant else '') + 'bit[4] source = "1010"; '
                        'bit[8] result = "11001100"; int index = 7; int start = 0; int stop = 1; '
                        'int zero = 0; int bad = 8; int first = 0; int last = 1; int visits = 0; '
                        'array[int, 2] addresses = {0, 7}; ' + wrapper.format(statement) +
                        'bool same = result == "11001100"; bool unchanged = source == "1010";',
                        (), (), ('same', 'unchanged'), (1, 1))

    def test_measurement_driven_addresses_and_ranges_control_quantum_feedback(self):
        for anti in (0, 1):
            for _ in range(16):
                self.check_initializers('include "stdgates.inc"; qubit[2] q; reset q; h q[0]; cx q[0], q[1]; '
                    + ('x q[1]; ' if anti else '') +
                    'bit[2] measured = measure q; bit[8] result = "01010101"; bit[1] one = "0"; '
                    'int left = 6; uint right = 1; int high = -1; int low = 0; '
                    'int start = 0; int stop = 1; int step = 1; '
                    'if (measured[0]) { left = 1; right = 6; start = 1; stop = 0; step = -1; } '
                    'result[{left, right}] = measured[start:step:stop]; '
                    'result[{high}] = measured[start:start]; one[{low}] = measured[stop:stop]; '
                    'bool ordered = result[left] == measured[start] && result[right] == measured[stop]; '
                    f'bool correlation = (result[left] != result[right]) == bool({anti}); '
                    'bool unchanged = (result & "00111101") == "00010101"; '
                    'bool singleton = one[0] == result[right]; '
                    'if (result[high]) { if (start == 0) { x q[0]; } else { x q[1]; } } '
                    'bit[2] observed = measure q; '
                    'bool feedback = !observed[start] && observed[stop] == measured[stop];',
                    (), (), ('ordered', 'correlation', 'unchanged', 'singleton', 'feedback'), (1, 1, 1, 1, 1))

    def test_later_failures_preserve_successful_copies_and_destination_error_priority(self):
        expected = list(reversed('11001100'))
        for index, value in zip((6, 1), ('0', '1')):
            expected[index] = value
        completed_copy = ''.join(reversed(expected))
        cases = (('left = 8;', 'assertion failed'), ('left = -9;', 'assertion failed'),
                 ('left = -1; right = 7;', 'assertion failed'),
                 ('address_divisor = 0;', 'division by zero'),
                 ('stop = 0;', 'assertion failed'), ('stop = 2;', 'assertion failed'),
                 ('step = 0;', 'assertion failed'), ('start = 3; stop = 4;', 'assertion failed'),
                 ('source_divisor = 0;', 'division by zero'),
                 ('left = 8; source_divisor = 0;', 'assertion failed'),
                 ('left = -1; right = 7; source_divisor = 0;', 'assertion failed'))
        for update, error in cases:
            lines = self.program('bit[4] source = "1010"; bit[8] result = "11001100"; '
                'int left = 6; int right = 1; int address_divisor = 1; int source_divisor = 1; '
                'int start = 0; int stop = 1; int step = 1; int i = 0; '
                'while (i < 2) { result[{left / address_divisor, right}] = source[start:step / source_divisor:stop]; '
                f'i += 1; {update} }}')
            writes = [i for i, line in enumerate(lines) if line.startswith('LET RESULT63:')
                      and ' := QASM2QCX_RANGE_COPY_SOURCE_' in line]
            self.assertEqual(len(writes), 2)
            instrumented = []
            for i, line in enumerate(lines):
                if line.startswith('LET QASM2QCX_RANGE_COPY_SOURCE_'):
                    instrumented.append('PRINTLN 12345')
                if i in writes:
                    instrumented.append('PRINTLN 67890')
                instrumented.append(line)
                if i == writes[-1]:
                    instrumented.append('PRINTLN ' + ' '.join(f'RESULT63:{bit}' for bit in reversed(range(8))))
            result = self.execute(instrumented)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(error, result.stderr)
            self.assertNotIn('runtime error:', result.stderr)
            self.assertEqual([''.join(line.split()) for line in result.stdout.splitlines()],
                             ['12345', '12345', '67890', '67890', completed_copy])
            type(self).checked += 1

    def test_runtime_destination_copies_feed_phase_parameters_and_fusion(self):
        import cmath
        import math
        from qasm2qcx_u_power_numerical import BackendAuditTests
        source = ('include "stdgates.inc"; qubit q; h q; bit[4] source = "1010"; bit[4] result = "0000"; '
                  'int left = 3; uint right = 0; int start = 0; int stop = 3; int visits = 0; '
                  'while (visits < 1) { result[{left, 2, 1, right}] = source[start:stop]; visits += 1; } '
                  'p(angle[4](result)) q; left = 0; right = 3; '
                  'result[{left, 1, 2, right}] = source[start:stop];')
        expected = (1 / math.sqrt(2), cmath.exp(1j * 5 * math.pi / 8) / math.sqrt(2))
        for fused in (False, True):
            BackendAuditTests.check(self, self.program(source, fused=fused), expected)

    def test_destination_failures_precede_source_evaluation_and_all_writes(self):
        cases = (('int i = 8;', 'i, 0', 'assertion failed'),
                 ('int i = -9;', '0, i', 'assertion failed'),
                 ('int i = 1;', 'i, i', 'assertion failed'),
                 ('int i = -1;', 'i, 7', 'assertion failed'),
                 (f'uint i = {self.module.QASM2QCXConverter.QCX_UINT_MAX};', 'i, 0',
                  'UINT to INT conversion out of range'),
                 ('int i = 0;', 'i / zero, 1', 'division by zero'),
                 ('array[int, 2] indices = {0, 1}; int i = 2;', 'indices[i], 1', 'assertion failed'))
        for declarations, entries, error in cases:
            lines = self.program('bit[4] source = "1010"; bit[8] result = "11001100"; '
                'int zero = 0; int start = 0; int stop = 1; ' + declarations +
                f'result[{{{entries}}}] = source[start:1 / zero:stop];')
            # A marker before source-step arithmetic proves destination errors
            # take priority, even over an RHS division by zero.
            source_division = max(i for i, line in enumerate(lines) if '/= ZERO15' in line)
            instrumented = []
            for i, line in enumerate(lines):
                if i == source_division:
                    instrumented.append('PRINTLN 12345')
                if line.startswith('LET RESULT63:') and ' := QASM2QCX_RANGE_COPY_SOURCE_' in line:
                    instrumented.append('PRINTLN 67890')
                instrumented.append(line)
            self.assertIn('PRINTLN 12345', instrumented)
            result = self.execute(instrumented)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(error, result.stderr)
            self.assertNotIn('runtime error:', result.stderr)
            self.assertNotIn('12345', result.stdout)
            self.assertNotIn('67890', result.stdout)
            type(self).checked += 1

    def test_source_failures_after_destination_capture_precede_reads_and_writes(self):
        for declarations, bounds, error in (
                ('int start = 0; int stop = 1; int step = 0;', 'start:step:stop', 'assertion failed'),
                ('int start = 1; int stop = 0;', 'start:stop', 'assertion failed'),
                ('int start = -5; int stop = -4;', 'start:stop', 'assertion failed'),
                ('int start = 3; int stop = 4;', 'start:stop', 'assertion failed'),
                ('int start = 0; int stop = 0;', 'start:stop', 'assertion failed'),
                ('int start = 0; int stop = 2;', 'start:stop', 'assertion failed'),
                ('int start = 0; int stop = 1; int zero = 0;', 'start:1 / zero:stop', 'division by zero')):
            lines = self.program('bit[4] source = "1010"; bit[8] result = "11001100"; int index = -1; '
                + declarations + f'result[{{index, 0}}] = source[{bounds}];')
            instrumented = []
            for line in lines:
                if (line.startswith('LET QASM2QCX_RANGE_COPY_SOURCE_')
                        or (line.startswith('LET RESULT63:') and ' := QASM2QCX_RANGE_COPY_SOURCE_' in line)):
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
    print(f'runtime set range-copy programs checked: {BackendTests.checked}')
    raise SystemExit(not result.wasSuccessful())


if __name__ == '__main__':
    main()
