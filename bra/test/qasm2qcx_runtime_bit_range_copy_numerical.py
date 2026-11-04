#!/usr/bin/env python3
"""Audit copies from runtime bit-register ranges into whole registers (#456)."""

import argparse
import pathlib
import re
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

    def test_mutable_and_constant_sources_preserve_order_and_source_bits(self):
        for width in (1, 2, 4, 33, 65, 129):
            ranges = {(0, 1, width - 1), (width - 1, -1, 0), (-width, 1, -1),
                      (-1, -1, -width), (0, 2, width - 1), (width - 1, -2, 0),
                      (0, width + 1, width), (-1, 1, width - 1)}
            for start, step, stop in sorted(ranges):
                raw = list(range(start, stop + (1 if step > 0 else -1), step))
                positions = [index % width for index in raw]
                for pattern in sorted({'0' * width, '1' * width, ('100110' * width)[:width]}):
                    bits = list(map(int, reversed(pattern)))
                    expected = ''.join(str(bits[index]) for index in reversed(positions))
                    for constant in (False, True):
                        for mixed in (False, True):
                            declarations = ''.join(
                                f'{"uint" if mixed and value >= 0 else "int"} {name} = {value};'
                                for name, value in (('start', start), ('stride', step), ('stop', stop)))
                            self.check_initializers(
                                f'{"const " if constant else ""}bit[{width}] flags = "{pattern}"; '
                                f'bit[{len(raw)}] result; ' + declarations +
                                'result = flags[start:stride:stop]; '
                                f'bool same = result == "{expected}"; bool unchanged = flags == "{pattern}";',
                                (), (), ('same', 'unchanged', 'start', 'stride', 'stop'),
                                (1, 1, start, step, stop))

    def test_in_place_reverse_snapshots_all_bits_before_writes(self):
        for width in (1, 2, 4, 33, 65, 129):
            patterns = ({f'{value:04b}' for value in range(16)} if width == 4
                        else {'0' * width, '1' * width, ('100110' * width)[:width]})
            for pattern in sorted(patterns):
                for start, step, stop in ((width - 1, -1, 0), (-1, -1, -width)):
                    self.check_initializers(f'bit[{width}] flags = "{pattern}"; '
                        f'int start = {start}; int stride = {step}; int stop = {stop}; '
                        'flags = flags[start:stride:stop]; '
                        f'bool same = flags == "{pattern[::-1]}";', (), (), ('same',), (1,))

    def test_default_step_repeated_normalized_positions_and_singletons(self):
        for constant in (False, True):
            prefix = ('const ' if constant else '') + 'bit[4] flags = "1010"; '
            self.check_initializers(prefix + 'bit[5] result; int start = -1; int stop = 3; '
                'result = flags[start:stop]; bool same = result == "10101";', (), (), ('same',), (1,))
            self.check_initializers(prefix + 'bit[8] result; int start = -4; int stop = 3; '
                'result = flags[start:stop]; bool same = result == "10101010";', (), (), ('same',), (1,))
            self.check_initializers(prefix + 'bit[1] result; int start = -1; '
                'result = flags[start:start]; bool same = result == "1";', (), (), ('same',), (1,))
            self.check_initializers(prefix + 'bit[4] result; array[int, 2] bounds = {-4, -1}; int i = 0; '
                'result = flags[bounds[i]:bounds[i + 1]]; bool same = result == "1010";',
                (), (), ('same',), (1,))

    def test_native_signed_unsigned_limits_and_unselected_endpoints(self):
        limits = self.module.QASM2QCXConverter
        for kind, step, start, stop in (('uint', limits.QCX_UINT_MAX, 0, 3),
                                        ('int', limits.QCX_INT_MAX, -1, 3),
                                        ('int', limits.QCX_INT_MIN, 0, -4),
                                        ('uint', limits.QCX_UINT_MAX, 0, limits.QCX_UINT_MAX - 1)):
            expected = '1' if start == -1 else '0'
            self.check_initializers('bit[4] flags = "1010"; bit[1] result; '
                f'{kind} stride = {step}; int start = {start}; {kind} stop = {stop}; '
                f'result = flags[start:stride:stop]; bool same = result == "{expected}";',
                (), (), ('same', 'stride', 'stop'), (1, step, stop))

    def test_loop_families_recapture_positions_and_restore_iterator_names(self):
        for loop, index in (
                ('for int i in [0:3]', 'i'), ('for int i in [first:last]', 'i'),
                ('for int i in [ufirst:ulast]', 'i'), ('for int i in {0, 1, 2, 3}', 'i'),
                ('for int i in {first, first + 1, last - 1, last}', 'i'),
                ('for int i in indices', 'i'), ('for uint i in unsigned_indices', 'i'),
                ('for bit i in flags', 'int(i)'), ('for bit i in C', 'int(i)')):
            for source in ('flags', 'C'):
                self.check_initializers('const bit[4] C = "1010"; bit[4] flags = "1010"; bit[1] result; '
                    'int first = 0; int last = 3; uint ufirst = 0; uint ulast = 3; int i = -1; int total = 0; '
                    'array[int, 4] indices = {0, 1, 2, 3}; array[uint, 4] unsigned_indices = {0, 1, 2, 3}; '
                    + loop + f' {{ result = {source}[{index}:{index}]; total += int(result[0]); }} '
                    'bool same = result == "1"; bool unchanged = flags == "1010";',
                    (), (), ('same', 'unchanged', 'total', 'i'), (1, 1, 2, -1))
        # Changing the source of a live bit iterator must affect later visits.
        self.check_initializers('const bit[4] C = "1111"; bit[4] flags = "0001"; '
            'int start = 0; int stop = 3; int total = 0; '
            'for bit b in flags { total += int(b); flags = C[start:stop]; } '
            'bool same = flags == "1111";', (), (), ('same', 'total'), (1, 4))

    def test_while_reentry_refreshes_direction_source_values_and_repeated_indices(self):
        for constant in (False, True):
            self.check_initializers(('const ' if constant else '') + 'bit[4] flags = "1010"; '
                'bit[4] result; int start = 0; int stop = 3; int stride = 1; int i = 0; int total = 0; '
                'while (i < 4) { result = flags[start:stride:stop]; total += int(result[0]); '
                'start = 3 - start; stop = 3 - stop; stride = -stride; i += 1; } '
                'bool same = result == "0101";', (), (), ('same', 'i', 'total', 'stride'), (1, 4, 2, 1))
            self.check_initializers(('const ' if constant else '') + 'bit[4] flags = "1010"; '
                'bit[8] result; int start = -4; int stop = 3; int stride = 1; int i = 0; '
                'while (i < 3) { result = flags[start:stride:stop]; '
                'start = -1 - start; stop = -1 - stop; stride = -stride; i += 1; } '
                'bool same = result == "10101010";', (), (), ('same', 'i'), (1, 3))
        self.check_initializers('bit[4] flags = "1001"; bit[2] result; int start = 0; int stop = 1; int i = 0; '
            'while (i < 4) { result = flags[start:stop]; flags ^= "0011"; i += 1; } '
            'bool same = result == "10"; bool restored = flags == "1001";',
            (), (), ('same', 'restored', 'i'), (1, 1, 4))
        self.check_initializers('bit[4] flags = "1010"; bit[1] result; array[int, 2] bounds = {0, 0}; int i = 0; '
            'while (i < 4) { result = flags[bounds[0]:bounds[1]]; bounds[0] += 1; bounds[1] += 1; i += 1; } '
            'bool same = result == "1";', (), (), ('same', 'i'), (1, 4))
        self.check_initializers('bit[4] flags = "1010"; bit[2] result; '
            'int start = 0; int stop = 1; int stride = 1; int i = 0; '
            'while (i < 2) { result = flags[start:stride:stop]; '
            'start = -1; stride = 4; stop = 3; i += 1; } '
            'bool same = result == "11";', (), (), ('same', 'i'), (1, 2))

    def test_nested_break_continue_and_shadowed_constants_restore_outer_bindings(self):
        self.check_initializers('const int I = 3; const bit[4] C = "1010"; bit[1] result; '
            'int first = 0; int last = 2; int start = -1; int visits = 0; int total = 0; '
            'for int I in [first:last] { for int start in [0:3] { '
            'if (start == 0) { continue; } result = C[start:start]; visits += 1; '
            'total += int(result[0]); break; } } '
            'for int C in {0, 1} {} result = C[I:I]; int restored = I; bool same = result == "1";',
            (), (), ('same', 'visits', 'total', 'restored', 'start'), (1, 3, 3, 3, -1))
        self.check_initializers('bit[4] flags = "1010"; bit[1] result; int i = 0; int total = 0; '
            'while (i < 4) { i += 1; if (i == 2) { continue; } '
            'result = flags[i - 1:i - 1]; total += int(result[0]); if (i == 3) { break; } } '
            'bool same = result == "0";', (), (), ('same', 'total', 'i'), (1, 0, 3))
        # Both arms reuse temporary names, but each reached copy must use
        # its own source, positions and count on every new visit.
        self.check_initializers('const bit[4] C = "1001"; bit[4] flags = "0110"; bit[2] result; '
            'int i = 0; int start = 0; int stop = 1; int total = 0; '
            'while (i < 4) { if (i % 2 == 0) { result = C[start:stop]; } '
            'else { result = flags[start:stop]; } total += int(result[0]); i += 1; } '
            'bool same = result == "10";', (), (), ('same', 'total'), (1, 2))

    def test_skipped_invalid_ranges_do_not_read_sources_or_modify_destinations(self):
        bounds = ('start:0:stop', 'bad:stop', 'start:1 / zero:stop', 'start:start',
                  'start:stop + 1', 'bounds[bad]:stop', 'start:-1:stop', 'start:stride:stop')
        wrappers = ('if (false) {{ {} }}', 'while (false) {{ {} }}',
                    'for int outer in [1:0] {{ {} }}',
                    'for int outer in [0:1] {{ if (outer < 0) {{ {} }} }}',
                    'for int outer in [first:last] {{ if (outer < 0) {{ {} }} }}',
                    'while (visits < 2) {{ visits += 1; continue; {} }}',
                    'while (true) {{ break; {} }}')
        for constant in (False, True):
            for selector in bounds:
                for wrapper in wrappers:
                    self.check_initializers(('const ' if constant else '') + 'bit[4] flags = "1010"; '
                        'bit[2] result = "11"; int start = 0; int stop = 1; int stride = 0; '
                        'int zero = 0; int bad = 4; int first = 0; int last = 1; int visits = 0; '
                        'array[int, 2] bounds = {0, 1}; ' + wrapper.format(f'result = flags[{selector}];') +
                        'bool same = result == "11"; bool unchanged = flags == "1010";',
                        (), (), ('same', 'unchanged'), (1, 1))

    def test_measurements_select_ranges_and_copied_bits_control_quantum_feedback(self):
        for anti in (0, 1):
            for _ in range(16):
                self.check_initializers('include "stdgates.inc"; qubit[2] q; reset q; h q[0]; cx q[0], q[1]; '
                    + ('x q[1]; ' if anti else '') +
                    'bit[2] measured = measure q; bit[2] copied; bit[1] selected; '
                    'int start = 0; int stop = 1; int stride = 1; '
                    'if (measured[0]) { start = 1; stop = 0; stride = -1; } '
                    'copied = measured[start:stride:stop]; selected = measured[start:start]; '
                    'bool ordered = copied[0] == measured[start] && copied[1] == measured[stop]; '
                    f'bool correlation = (copied[0] != copied[1]) == bool({anti}); '
                    'bool chosen = selected[0] == copied[0]; '
                    'if (selected[0]) { if (start == 0) { x q[0]; } else { x q[1]; } } '
                    'bit[2] observed = measure q; '
                    'bool feedback = !observed[start] && observed[stop] == measured[stop];',
                    (), (), ('ordered', 'correlation', 'chosen', 'feedback'), (1, 1, 1, 1))

    def test_later_iteration_failures_precede_new_source_reads_and_writes(self):
        for selector, update, error in (
                ('start:stride:stop', 'stop = 0;', 'assertion failed'),
                ('start:stride:stop', 'stop = 2;', 'assertion failed'),
                ('start:stride:stop', 'stride = 0;', 'assertion failed'),
                ('start:stride:stop', 'start = -5; stop = -4;', 'assertion failed'),
                ('start:stride:stop', 'start = 3; stop = 4;', 'assertion failed'),
                ('start:stride / divisor:stop', 'divisor = 0;', 'division by zero')):
            lines = self.program('bit[4] flags = "1010"; bit[2] result = "11"; '
                'int start = 0; int stop = 1; int stride = 1; int divisor = 1; int i = 0; '
                f'while (i < 2) {{ result = flags[{selector}]; i += 1; {update} }}')
            write_end = next(i for i, line in enumerate(lines)
                             if line.startswith('JUMPIF QASM2QCX_RANGE_COPY_') and '_WRITE ' in line)
            instrumented = []
            for i, line in enumerate(lines):
                if line.startswith('LET QASM2QCX_RANGE_COPY_SOURCE_'):
                    instrumented.append('PRINTLN 12345')
                if line.startswith('LET RESULT63:QASM2QCX_'):
                    instrumented.append('PRINTLN 67890')
                instrumented.append(line)
                if i == write_end:
                    instrumented.append('PRINTLN RESULT63:1 RESULT63:0')
            result = self.execute(instrumented)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(error, result.stderr)
            self.assertNotIn('runtime error:', result.stderr)
            self.assertEqual([''.join(line.split()) for line in result.stdout.splitlines()],
                             ['12345', '12345', '67890', '67890', '10'])
            type(self).checked += 1

    def test_copied_registers_feed_phase_gates_and_fusion(self):
        import cmath
        import math
        from qasm2qcx_u_power_numerical import BackendAuditTests
        source = ('include "stdgates.inc"; qubit q; h q; bit[4] flags = "1010"; '
                  'int start = 3; int stop = 0; int stride = -1; int visits = 0; '
                  'while (visits < 1) { flags = flags[start:stride:stop]; visits += 1; } '
                  'p(angle[4](flags)) q;')
        expected = (1 / math.sqrt(2), cmath.exp(1j * 5 * math.pi / 8) / math.sqrt(2))
        for fused in (False, True):
            BackendAuditTests.check(self, self.program(source, fused=fused), expected)

    def test_documented_runtime_range_copy_example(self):
        document = (pathlib.Path(__file__).parents[1] / 'qcx/qasm2qcx.md').read_text(encoding='utf-8')
        examples = re.findall(r'<!-- runtime-bit-range-copy-example -->\s*```qasm\n(.*?)\n```',
                              document, flags=re.DOTALL)
        self.assertEqual(len(examples), 1)
        source = examples[0].removeprefix('OPENQASM 3.0;')
        self.check_initializers(source + '\nbool same = result == "111" && flags == "01010101" '
                                '&& repeated == "10101" && one == "0";',
                                (), (), ('same', 'one', 'start', 'stride', 'stop', 'reverse'), (1, 0, -1, 2, 3, -1))

    def test_bad_ranges_and_lengths_fail_before_any_source_read_or_destination_write(self):
        cases = (
            ('int start = 0; int stop = 3; int step = 0;', 'start:step:stop', 4, 'assertion failed'),
            ('int start = 2; int stop = 0;', 'start:stop', 1, 'assertion failed'),
            ('int start = -5; int stop = -4;', 'start:stop', 2, 'assertion failed'),
            ('int start = 0; int stop = 4;', 'start:stop', 5, 'assertion failed'),
            ('int start = 0; int stop = 1;', 'start:stop', 3, 'assertion failed'),
            ('int start = 0; int stop = 3;', 'start:stop', 3, 'assertion failed'),
            ('int start = -1; int stop = 3;', 'start:stop', 4, 'assertion failed'),
            (f'uint start = {self.module.QASM2QCXConverter.QCX_UINT_MAX}; uint stop = start;',
             'start:stop', 1, 'assertion failed'),
            ('int start = 0; int stop = 1; int zero = 0;', 'start:1 / zero:stop', 2, 'division by zero'),
            ('array[int, 2] bounds = {0, 1}; int i = 2; int stop = 1;',
             'bounds[i]:stop', 2, 'assertion failed'),
        )
        for declarations, bounds, width, error in cases:
            with self.subTest(bounds=bounds, declarations=declarations, width=width):
                lines = self.program('bit[4] flags = "1010"; '
                    f'bit[{width}] result = "' + '1' * width + '"; ' + declarations +
                    f'result = flags[{bounds}];')
                instrumented = []
                for line in lines:
                    if (line.startswith('LET QASM2QCX_RANGE_COPY_SOURCE_')
                            or line.startswith('LET RESULT63:QASM2QCX_')
                            or line.startswith('LET RESULT63 := QASM2QCX_')):
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
    print(f'runtime bit-range copy programs checked: {BackendTests.checked}')
    raise SystemExit(not result.wasSuccessful())


if __name__ == '__main__':
    main()
