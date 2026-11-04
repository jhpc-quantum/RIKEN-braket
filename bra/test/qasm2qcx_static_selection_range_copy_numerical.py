#!/usr/bin/env python3
"""Audit runtime bit-range copies into static register selections (#458)."""

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

    def test_static_ranges_and_sets_preserve_order_and_unselected_bits(self):
        for width in (1, 2, 4, 33, 65, 129):
            ranges = {(0, 1, width - 1), (width - 1, -1, 0), (-width, 1, -1),
                      (-1, -1, -width), (0, 2, width - 1), (width - 1, -2, 0),
                      (-1, 1, width - 1), (-width, 1, width - 1)}
            for start, step, stop in sorted(ranges):
                raw = list(range(start, stop + (1 if step > 0 else -1), step))
                count = len(raw)
                target_width = 2 * count + 1
                before = ('01101001' * target_width)[:target_width]
                permuted = tuple(2 * i + 1 for i in range(count))
                permuted = permuted[1:] + permuted[:1]
                targets = ((f'0:{count - 1}', tuple(range(count))),
                           (f'-1:-2:{-(2 * count - 1)}', tuple(range(2 * count, 0, -2))),
                           ('{' + ','.join(map(str, permuted)) + '}', permuted))
                declarations = ''.join(
                    f'{"uint" if value >= 0 else "int"} {name} = {value};'
                    for name, value in (('start', start), ('step', step), ('stop', stop)))
                for pattern in sorted({'0' * width, '1' * width, ('100110' * width)[:width]}):
                    source_bits = list(map(int, reversed(pattern)))
                    selected = [source_bits[index % width] for index in raw]
                    for constant in (False, True):
                        for selector, positions in targets:
                            expected = list(map(int, reversed(before)))
                            for index, value in zip(positions, selected):
                                expected[index] = value
                            text = ''.join(map(str, reversed(expected)))
                            self.check_initializers(f'{"const " if constant else ""}bit[{width}] source = "{pattern}"; '
                                f'bit[{target_width}] result = "{before}"; ' + declarations +
                                f'result[{selector}] = source[start:step:stop]; '
                                f'bool same = result == "{text}"; bool unchanged = source == "{pattern}";',
                                (), (), ('same', 'unchanged', 'start', 'step', 'stop'), (1, 1, start, step, stop))

    def test_overlapping_self_copies_use_original_source_bits(self):
        for pattern in range(16):
            bits = [(pattern >> i) & 1 for i in range(4)]
            for start, step, stop, destinations in (
                    (0, 1, 2, ((1, 2, 3), (3, 0, 1))),
                    (3, -1, 0, ((3, 2, 1, 0), (2, 0, 3, 1))),
                    (-1, -1, -4, ((0, 1, 2, 3), (1, 3, 0, 2))),
                    (-1, 4, 3, ((0, 1), (3, 0)))):
                raw = list(range(start, stop + (1 if step > 0 else -1), step))
                selected = [bits[index % 4] for index in raw]
                for positions in destinations:
                    expected = bits.copy()
                    for index, value in zip(positions, selected):
                        expected[index] = value
                    text = ''.join(map(str, reversed(expected)))
                    selector = '{' + ','.join(map(str, positions)) + '}'
                    self.check_initializers(f'bit[4] flags = "{pattern:04b}"; '
                        f'int start = {start}; int step = {step}; int stop = {stop}; '
                        f'flags[{selector}] = flags[start:step:stop]; bool same = flags == "{text}";',
                        (), (), ('same',), (1,))

    def test_singleton_storage_default_steps_and_negative_destinations(self):
        for constant in (False, True):
            for value in ('0', '1'):
                for destination in ('one[0:0]', 'one[{-1}]', 'flags[4:4]', 'flags[{-1}]'):
                    expected = ('00010000' if destination == 'flags[4:4]' else '10000000') if value == '1' else '00000000'
                    self.check_initializers(f'{"const " if constant else ""}bit[1] source = "{value}"; '
                        'bit[1] one = "0"; bit[8] flags = "00000000"; int index = -1; '
                        f'{destination} = source[index:index]; '
                        f'bool same = flags == "{expected if destination.startswith("flags") else "00000000"}"; '
                        f'bool singleton = one == "{value if destination.startswith("one") else "0"}";',
                        (), (), ('same', 'singleton'), (1, 1))
        self.check_initializers('bit[8] source = "10101010"; bit[8] result = "00000000"; '
            'array[int, 2] bounds = {1, 3}; int index = 0; '
            'result[5:-2:1] = source[bounds[index]:bounds[index + 1]]; '
            'bool same = result == "00100010";', (), (), ('same',), (1,))

    def test_wide_constant_word_boundaries_and_partial_final_words(self):
        native = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        for width in (native - 1, native, native + 1, 2 * native + 1, 129):
            for position in sorted({0, width - 1, min(native - 1, width - 1), min(native, width - 1)}):
                pattern = f'{1 << position:0{width}b}'
                for index in (position, position - width):
                    self.check_initializers(f'const bit[{width}] C = "{pattern}"; bit[129] result = "' + '0' * 129 +
                        f'"; int index = {index}; '
                        'result[{128}] = C[index:index]; '
                        'bool same = result == "1' + '0' * 128 + '";', (), (), ('same',), (1,))

    def test_native_endpoint_strides_and_unselected_stops(self):
        limits = self.module.QASM2QCXConverter
        for kind, step, start, stop in (('uint', limits.QCX_UINT_MAX, 0, limits.QCX_UINT_MAX - 1),
                                        ('int', limits.QCX_INT_MIN, 0, -4),
                                        ('int', limits.QCX_INT_MAX, -1, 3)):
            expected = '10000000' if start == -1 else '00000000'
            self.check_initializers('bit[4] source = "1010"; bit[8] result = "00000000"; '
                f'int start = {start}; {kind} step = {step}; {kind} stop = {stop}; '
                f'result[7:7] = source[start:step:stop]; bool same = result == "{expected}";',
                (), (), ('same', 'step', 'stop'), (1, step, stop))

    def test_loop_families_capture_fresh_sources_and_preserve_unselected_bits(self):
        loops = (('for int i in [0:3]', 'i'), ('for int i in [first:last]', 'i'),
                 ('for int i in [ufirst:ulast]', 'i'), ('for int i in [first:stride:last]', 'i'),
                 ('for int i in {0, 1, 2, 3}', 'i'), ('for int i in {first, first + 1, last - 1, last}', 'i'),
                 ('for int i in indices', 'i'), ('for uint i in unsigned_indices', 'i'),
                 ('for bit i in flags', 'int(i)'), ('for bit i in C', 'int(i)'))
        for loop, index in loops:
            for source in ('flags', 'C'):
                for target, scalar, expected, one in (
                        ('result[4:4]', 'result[4]', '00010000', '0'),
                        ('result[{-1}]', 'result[7]', '10000000', '0'),
                        ('one[0:0]', 'one[0]', '00000000', '1')):
                    self.check_initializers('const bit[4] C = "1010"; bit[4] flags = "1010"; '
                        'bit[8] result = "00000000"; bit[1] one = "0"; '
                        'int first = 0; int last = 3; uint ufirst = 0; uint ulast = 3; int stride = 1; '
                        'int i = -1; int total = 0; array[int, 4] indices = {0, 1, 2, 3}; '
                        'array[uint, 4] unsigned_indices = {0, 1, 2, 3}; ' + loop +
                        f' {{ {target} = {source}[{index}:{index}]; total += int({scalar}); }} '
                        f'bool same = result == "{expected}" && one == "{one}"; '
                        'bool unchanged = flags == "1010";', (), (), ('same', 'unchanged', 'total', 'i'), (1, 1, 2, -1))
        self.check_initializers('const bit[4] C = "1111"; bit[4] flags = "0001"; '
            'int start = 0; int stop = 3; int total = 0; '
            'for bit b in flags { total += int(b); flags[3:-1:0] = C[start:stop]; } '
            'bool same = flags == "1111";', (), (), ('same', 'total'), (1, 4))

    def test_while_reentry_refreshes_direction_values_bounds_and_repeated_positions(self):
        for constant in (False, True):
            self.check_initializers(('const ' if constant else '') + 'bit[4] source = "1010"; '
                'bit[8] result = "01010101"; int start = 0; int stop = 3; int step = 1; int i = 0; int total = 0; '
                'while (i < 4) { result[{7, 1, 5, 3}] = source[start:step:stop]; '
                'total += int(result[7]); start = 3 - start; stop = 3 - stop; step = -step; i += 1; } '
                'bool same = result == "11110101";', (), (), ('same', 'i', 'total', 'step'), (1, 4, 2, 1))
            self.check_initializers(('const ' if constant else '') + 'bit[4] source = "1010"; '
                'bit[8] result = "00000000"; int start = 0; int stop = 1; int step = 1; int i = 0; '
                'while (i < 2) { result[{7, 0}] = source[start:step:stop]; '
                'start = -1; step = 4; stop = 3; i += 1; } bool same = result == "10000001";',
                (), (), ('same', 'i'), (1, 2))
        self.check_initializers('bit[4] source = "1001"; bit[8] result = "00000000"; '
            'int start = 0; int stop = 1; int i = 0; while (i < 4) { '
            'result[5:-2:3] = source[start:stop]; source ^= "0011"; i += 1; } '
            'bool same = result == "00001000" && source == "1001";', (), (), ('same', 'i'), (1, 4))
        self.check_initializers('bit[4] source = "1010"; bit[8] result = "00000000"; '
            'array[int, 2] bounds = {0, 0}; int i = 0; while (i < 4) { '
            'result[4:4] = source[bounds[0]:bounds[1]]; bounds[0] += 1; bounds[1] += 1; i += 1; } '
            'bool same = result == "00010000";', (), (), ('same', 'i'), (1, 4))

    def test_nested_transfers_and_shadowed_names_restore_static_destination_constants(self):
        self.check_initializers('const int LAST = 7; const bit[4] C = "1010"; bit[8] result = "00000000"; '
            'int first = 0; int last = 2; int start = -1; int visits = 0; int total = 0; '
            'for int LAST in [first:last] { for int start in [0:3] { '
            'if (start == 0) { continue; } result[0:0] = C[start:start]; '
            'visits += 1; total += int(result[0]); break; } } '
            'for int C in {0, 1} {} result[{LAST, 1}] = C[first:first + 1]; '
            'bool same = result == "00000011"; int restored = LAST;',
            (), (), ('same', 'visits', 'total', 'restored', 'start'), (1, 3, 3, 7, -1))
        self.check_initializers('const bit[4] C = "1010"; bit[8] result = "00000000"; '
            'int i = 0; int start = 0; int stop = 1; int total = 0; '
            'while (i < 4) { if (i % 2 == 0) { result[0:1] = C[start:stop]; } '
            'else { result[{7, 3}] = C[start:stop]; } '
            'total += int(result[1]); i += 1; } bool same = result == "00001010";',
            (), (), ('same', 'i', 'total'), (1, 4, 4))
        self.check_initializers('bit[4] source = "1010"; bit[8] result = "11001100"; int i = 0; '
            'while (i < 4) { i += 1; if (i == 2) { continue; } '
            'result[7:7] = source[i - 1:i - 1]; if (i == 3) { break; } } '
            'bool same = result == "01001100";', (), (), ('same', 'i'), (1, 3))

    def test_skipped_invalid_source_ranges_do_not_modify_static_destinations(self):
        bounds = ('start:0:stop', 'bad:stop', 'start:1 / zero:stop', 'start:start',
                  'start:stop + 1', 'bounds[bad]:stop', 'start:-1:stop', 'start:step:stop')
        wrappers = ('if (false) {{ {} }}', 'while (false) {{ {} }}', 'for int outer in [1:0] {{ {} }}',
                    'for int outer in [first:last] {{ if (outer < 0) {{ {} }} }}',
                    'while (visits < 2) {{ visits += 1; continue; {} }}', 'while (true) {{ break; {} }}')
        for constant in (False, True):
            for selector in ('0:1', '7:-3:4', '{6, 1}'):
                for bounds_expression in bounds:
                    for wrapper in wrappers:
                        self.check_initializers(('const ' if constant else '') + 'bit[4] source = "1010"; '
                            'bit[8] result = "11001100"; int start = 0; int stop = 1; int step = 0; '
                            'int zero = 0; int bad = 4; int first = 0; int last = 1; int visits = 0; '
                            'array[int, 2] bounds = {0, 1}; ' +
                            wrapper.format(f'result[{selector}] = source[{bounds_expression}];') +
                            'bool same = result == "11001100"; bool unchanged = source == "1010";',
                            (), (), ('same', 'unchanged'), (1, 1))

    def test_measured_source_ranges_and_selected_destinations_drive_quantum_feedback(self):
        for anti in (0, 1):
            for _ in range(16):
                self.check_initializers('include "stdgates.inc"; qubit[2] q; reset q; h q[0]; cx q[0], q[1]; '
                    + ('x q[1]; ' if anti else '') +
                    'bit[2] measured = measure q; bit[8] result = "01010101"; bit[1] one = "0"; '
                    'int start = 0; int stop = 1; int step = 1; '
                    'if (measured[0]) { start = 1; stop = 0; step = -1; } '
                    'result[{6, 1}] = measured[start:step:stop]; result[7:7] = measured[start:start]; '
                    'one[{0}] = measured[stop:stop]; '
                    'bool ordered = result[6] == measured[start] && result[1] == measured[stop]; '
                    f'bool correlation = (result[6] != result[1]) == bool({anti}); '
                    'bool unchanged = (result & "00111101") == "00010101"; '
                    'bool singleton = one[0] == result[1]; '
                    'if (result[7]) { if (start == 0) { x q[0]; } else { x q[1]; } } '
                    'bit[2] observed = measure q; '
                    'bool feedback = !observed[start] && observed[stop] == measured[stop];',
                    (), (), ('ordered', 'correlation', 'unchanged', 'singleton', 'feedback'), (1, 1, 1, 1, 1))

    def test_later_iteration_failures_precede_all_new_reads_and_selected_writes(self):
        for selector, positions in (('0:1', (0, 1)), ('7:-3:4', (7, 4)), ('{6, 1}', (6, 1))):
            expected = list(map(int, reversed('11001100')))
            for index, value in zip(positions, (0, 1)):
                expected[index] = value
            text = ''.join(map(str, reversed(expected)))
            for bounds, update, error in (
                    ('start:step:stop', 'stop = 0;', 'assertion failed'),
                    ('start:step:stop', 'stop = 2;', 'assertion failed'),
                    ('start:step:stop', 'step = 0;', 'assertion failed'),
                    ('start:step:stop', 'start = -5; stop = -4;', 'assertion failed'),
                    ('start:step:stop', 'start = 3; stop = 4;', 'assertion failed'),
                    ('start:step / divisor:stop', 'divisor = 0;', 'division by zero')):
                lines = self.program('bit[4] source = "1010"; bit[8] result = "11001100"; '
                    'int start = 0; int stop = 1; int step = 1; int divisor = 1; int i = 0; '
                    f'while (i < 2) {{ result[{selector}] = source[{bounds}]; i += 1; {update} }}')
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
                                 ['12345', '12345', '67890', '67890', text])
                type(self).checked += 1

    def test_selected_copies_feed_phase_parameters_and_fusion(self):
        import cmath
        import math
        from qasm2qcx_u_power_numerical import BackendAuditTests
        source = ('include "stdgates.inc"; qubit q; h q; bit[4] source = "1010"; bit[4] result = "0000"; '
                  'int start = 0; int stop = 3; int visits = 0; '
                  'while (visits < 1) { result[3:-1:0] = source[start:stop]; visits += 1; } '
                  'p(angle[4](result)) q; result[0:3] = source[start:stop];')
        expected = (1 / math.sqrt(2), cmath.exp(1j * 5 * math.pi / 8) / math.sqrt(2))
        for fused in (False, True):
            BackendAuditTests.check(self, self.program(source, fused=fused), expected)

    def test_invalid_source_ranges_and_counts_fail_before_reads_or_any_destination_write(self):
        cases = (
            ('int start = 0; int stop = 1; int step = 0;', 'start:step:stop', 'assertion failed'),
            ('int start = 1; int stop = 0;', 'start:stop', 'assertion failed'),
            ('int start = -5; int stop = -4;', 'start:stop', 'assertion failed'),
            ('int start = 3; int stop = 4;', 'start:stop', 'assertion failed'),
            ('int start = 0; int stop = 0;', 'start:stop', 'assertion failed'),
            ('int start = 0; int stop = 2;', 'start:stop', 'assertion failed'),
            ('int start = 0; int stop = 1; int zero = 0;', 'start:1 / zero:stop', 'division by zero'),
            ('array[int, 2] bounds = {0, 1}; int index = 2; int stop = 1;', 'bounds[index]:stop', 'assertion failed'))
        for selector in ('0:1', '7:-3:4', '{6, 1}'):
            for declarations, bounds, error in cases:
                lines = self.program('bit[4] source = "1010"; bit[8] result = "11001100"; '
                                     + declarations + f'result[{selector}] = source[{bounds}];')
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
    print(f'static selection range-copy programs checked: {BackendTests.checked}')
    raise SystemExit(not result.wasSuccessful())


if __name__ == '__main__':
    main()
