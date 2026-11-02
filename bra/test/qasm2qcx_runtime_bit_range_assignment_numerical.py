#!/usr/bin/env python3
"""Audit plain assignment into runtime bit-register ranges (#448)."""

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

    def test_signed_unsigned_ranges_preserve_order_and_unselected_bits(self):
        for width in (1, 2, 4, 33, 65, 129):
            ranges = {(0, 1, width - 1), (width - 1, -1, 0), (-width, 1, -1), (-1, -1, -width),
                      (0, 2, width - 1), (width - 1, -2, 0), (0, width, width - 1),
                      (-1, 1, min(1, width - 2))}
            for start, step, stop in sorted(ranges):
                raw = list(range(start, stop + (1 if step > 0 else -1), step))
                positions = [value % width for value in raw]
                if not raw or len(set(positions)) != len(positions):
                    continue
                count = len(positions)
                for payload in sorted({'0' * count, '1' * count, ('101100' * count)[:count]}):
                    before = ('100110' * width)[:width]
                    expected = list(map(int, reversed(before)))
                    for position, bit in zip(positions, reversed(payload)):
                        expected[position] = int(bit)
                    text = ''.join(map(str, reversed(expected)))
                    for mixed in (False, True):
                        # UINT endpoints/strides remain full-width operands;
                        # descending and negative values retain signed types.
                        declarations = ''.join(
                            f'{"uint" if mixed and value >= 0 else "int"} {name} = {value};'
                            for name, value in (('start', start), ('stride', step), ('stop', stop)))
                        self.check_initializers(f'bit[{width}] flags = "{before}"; ' + declarations +
                            f'flags[start:stride:stop] = "{payload}"; bool same = flags == "{text}";',
                            (), (), ('same', 'start', 'stride', 'stop'), (1, start, step, stop))

    def test_overlapping_rhs_is_copied_before_first_write(self):
        for pattern in range(16):
            bits = [(pattern >> i) & 1 for i in range(4)]
            for positions, rhs in (((1, 2, 3), 'flags[0:2]'), ((3, 2, 1, 0), 'flags'),
                                   ((0, 2), 'flags[{3, index}]')):
                values = (bits[:3] if len(positions) == 3 else bits.copy() if len(positions) == 4
                          else [bits[3], bits[0]])
                expected = bits.copy()
                for position, value in zip(positions, values):
                    expected[position] = value
                step = positions[1] - positions[0]
                text = ''.join(map(str, reversed(expected)))
                self.check_initializers(f'bit[4] flags = "{pattern:04b}"; int index = 0; '
                    f'int start = {positions[0]}; int stop = {positions[-1]}; int stride = {step}; '
                    f'flags[start:stride:stop] = {rhs}; bool same = flags == "{text}";',
                    (), (), ('same',), (1,))

    def test_nested_bounds_and_existing_rhs_expression_consumers(self):
        self.check_initializers('bit[4] flags = "1001"; int index = 0; array[int, 2] bounds = {0, 3}; '
            'flags[bounds[index]:int(flags[index]) + 1:bounds[index + 1]] = "10"; '
            'bool same = flags == "1100";', (), (), ('same',), (1,))
        for rhs in ('rotl(pair, count)', '~pair', 'C[{index, 3}]',
                    'a[0:1]', 'flags[{3, index}] << count'):
            # Expected bits are specified independently of destination
            # lowering; every case yields the two-bit pattern 10.
            self.check_initializers('bit[4] flags = "1000"; bit[2] pair = "01"; uint count = 1; '
                'const bit[4] C = "1000"; angle[2] a = angle[2]("10"); int index = 0; '
                f'int start = 1; flags[start:2] = {rhs}; bool same = flags == "1100";',
                (), (), ('same',), (1,))

    def test_native_endpoint_strides_and_unselected_stop_values(self):
        limits = self.module.QASM2QCXConverter
        for kind, step, start, stop in (('uint', limits.QCX_UINT_MAX, 0, 3),
                                        ('int', limits.QCX_INT_MAX, -1, 3),
                                        ('int', limits.QCX_INT_MIN, 0, -4)):
            expected = '1000' if start == -1 else '0001'
            self.check_initializers(f'bit[4] flags = "0000"; {kind} stride = {step}; '
                f'int start = {start}; int stop = {stop}; flags[start:stride:stop] = "1"; '
                f'bool same = flags == "{expected}";', (), (), ('same',), (1,))
        # The stop bound is not selected and is not itself an array access.
        self.check_initializers('bit[4] flags = "0000"; uint stride = 5; '
            'flags[0:stride:4] = "1"; bool same = flags == "0001";', (), (), ('same',), (1,))

    def test_loops_capture_fresh_positions_and_preserve_live_register_iteration(self):
        for loop in ('for int i in [first:last]', 'for int i in {0, 1, 2, 3}',
                     'for int i in {first, first + 1, last - 1, last}', 'for int i in indices'):
            self.check_initializers('bit[4] flags = "0000"; int first = 0; int last = 3; '
                'array[int, 4] indices = {0, 1, 2, 3}; ' + loop +
                ' { flags[i:i] = "1"; } bool same = flags == "1111";', (), (), ('same',), (1,))
        self.check_initializers('bit[4] flags = "1010"; bit[2] indices = "10"; '
            'for bit i in indices { flags[int(i):int(i)] = "1"; } '
            'bool same = flags == "1011";', (), (), ('same',), (1,))
        # The iterator sees later bits after writes to the live source.
        self.check_initializers('bit[4] flags = "0001"; int next = 1; int total = 0; '
            'for bit b in flags { total += int(b); '
            'if (next < 4) { flags[next:next] = "1"; next += 1; } } '
            'bool same = flags == "1111";', (), (), ('same', 'total'), (1, 4))

    def test_while_reentry_and_changing_direction_capture_new_bounds(self):
        self.check_initializers('bit[4] flags = "0000"; int i = 0; while (i < 3) { '
            'flags[i:i + 1] = "10"; i += 1; } bool same = flags == "1000";',
            (), (), ('same', 'i'), (1, 3))
        self.check_initializers('bit[4] flags = "0000"; int i = 0; int start = 0; int stop = 3; int step = 1; '
            'while (i < 3) { flags[start:step:stop] = "1010"; '
            'start = 3 - start; stop = 3 - stop; step = -step; i += 1; } '
            'bool same = flags == "1010";', (), (), ('same', 'start', 'stop', 'step'), (1, 3, 0, -1))

    def test_nested_transfers_shadowing_and_outer_binding_restoration(self):
        self.check_initializers('const int I = 3; int index = -1; bit[4] flags = "0000"; '
            'int first = 0; int last = 1; int visits = 0; '
            'for int I in [first:last] { for int index in [0:2] { '
            'if (index == 1) { continue; } flags[I:I] = "1"; visits += 1; break; } } '
            'for int index in {2, 3} { flags[index:index] = "1"; } '
            'flags[I:I] = "0"; bool same = flags == "0111"; int restored = I;',
            (), (), ('same', 'visits', 'restored', 'index'), (1, 2, 3, -1))
        self.check_initializers('bit[4] flags = "0000"; int i = 0; while (i < 4) { '
            'i += 1; if (i == 2) { continue; } flags[i - 1:i - 1] = "1"; '
            'if (i == 3) { break; } } bool same = flags == "0101";', (), (), ('same', 'i'), (1, 3))

    def test_skipped_bounds_width_duplicates_and_rhs_failures(self):
        statements = ('flags[start:0:stop] = "11";', 'flags[bad:stop] = "11";',
                      'flags[start:1 / zero:stop] = "11";', 'flags[start:stop] = "1";',
                      'flags[-1:stop + 2] = "11111";', 'flags[start:stop] = flags[{start, bad}];',
                      'flags[start:stop + 2] = flags << (1 / zero);')
        wrappers = ('if (false) {{ {} }}', 'while (false) {{ {} }}',
                    'for int outer in [1:0] {{ {} }}',
                    'for int outer in [0:1] {{ if (outer < 0) {{ {} }} }}')
        for statement in statements:
            for wrapper in wrappers:
                self.check_initializers('bit[4] flags = "1010"; int start = 0; int stop = 1; '
                    'int zero = 0; int bad = 4; ' + wrapper.format(statement) +
                    'bool same = flags == "1010";', (), (), ('same',), (1,))

    def test_measurement_controlled_ranges_and_feedback(self):
        for _ in range(16):
            self.check_initializers('include "stdgates.inc"; qubit q; h q; bit outcome = measure q; '
                'bit[4] flags = "0000"; int start = 0; int stop = 1; '
                'if (outcome) { start = 2; stop = 3; } flags[start:stop] = "10"; '
                'bool same = (outcome && flags == "1000") || (!outcome && flags == "0010"); '
                'reset q; if (flags[start + 1]) { x q; } bit observed = measure q;',
                (), (), ('same', 'observed'), (1, 1))

    def test_later_iteration_failures_preserve_previous_successful_writes(self):
        for selector, rhs, update, expected, error in (
                ('start:stop', '"11"', 'stop = 0;', ['1011'], 'assertion failed'),
                ('start:stop', '"11"', 'stop = 2;', ['1011'], 'assertion failed'),
                ('start:stop', 'flags[{start, other}]', 'other = 4;', ['1010'], 'assertion failed'),
                ('start:stop', 'pair << (1 / divisor)', 'divisor = 0;', ['1010'], 'division by zero')):
            lines = self.program('bit[4] flags = "1010"; bit[2] pair = "01"; '
                'int start = 0; int stop = 1; int other = 1; int divisor = 1; int i = 0; '
                f'while (i < 2) {{ flags[{selector}] = {rhs}; i += 1; {update} }}')
            # Mark every destination write and report each completed store.
            # Only the first iteration may reach these markers.
            last_write = max(i for i, line in enumerate(lines) if line.startswith('LET FLAGS31:QASM2QCX_'))
            instrumented = []
            for i, line in enumerate(lines):
                if line.startswith('LET FLAGS31:QASM2QCX_'):
                    instrumented.append('PRINTLN 12345')
                instrumented.append(line)
                if i == last_write:
                    instrumented.append('PRINTLN FLAGS31:3 FLAGS31:2 FLAGS31:1 FLAGS31:0')
            result = self.execute(instrumented)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(error, result.stderr)
            self.assertNotIn('runtime error:', result.stderr)
            self.assertEqual([''.join(line.split()) for line in result.stdout.splitlines()], ['12345', '12345'] + expected)
            type(self).checked += 1

    def test_updated_ranges_feed_phase_gates_and_fusion(self):
        import cmath
        import math
        from qasm2qcx_u_power_numerical import BackendAuditTests
        source = ('include "stdgates.inc"; qubit q; h q; bit[4] flags = "1010"; '
                  'int start = 3; int stop = 0; int stride = -1; int visits = 0; '
                  'while (visits < 1) { flags[start:stride:stop] = flags; visits += 1; } '
                  'p(angle[4](flags)) q; start = 0; stop = 3; stride = 1; '
                  'flags[start:stride:stop] = "0000";')
        expected = (1 / math.sqrt(2), cmath.exp(1j * 5 * math.pi / 8) / math.sqrt(2))
        for fused in (False, True):
            BackendAuditTests.check(self, self.program(source, fused=fused), expected)

    def test_documented_runtime_range_assignment_example(self):
        document = (pathlib.Path(__file__).parents[1] / 'qcx/qasm2qcx.md').read_text(encoding='utf-8')
        examples = re.findall(r'<!-- runtime-bit-range-assignment-example -->\s*```qasm\n(.*?)\n```',
                              document, flags=re.DOTALL)
        self.assertEqual(len(examples), 1)
        source = examples[0].removeprefix('OPENQASM 3.0;')
        self.check_initializers(source + '\nbool same = flags == "00001000";',
                                (), (), ('same', 'one', 'start', 'stride', 'stop', 'reverse', 'single'),
                                (1, 1, 5, 2, 1, -2, -1))

    def test_destination_and_rhs_failures_precede_all_destination_writes(self):
        cases = (
            ('int start = 0; int stop = 1;', 'start:0:stop', '"11"', 'assertion failed'),
            ('int start = 3; int stop = 0;', 'start:stop', '"11"', 'assertion failed'),
            ('int start = 0; int stop = 0;', 'start:stop', '"11"', 'assertion failed'),
            ('int start = 0; int stop = 2;', 'start:stop', '"11"', 'assertion failed'),
            ('int start = -1; int stop = 3;', 'start:stop', '"11111"', 'assertion failed'),
            ('int start = -5; int stop = -4;', 'start:stop', '"11"', 'assertion failed'),
            ('int start = 3; int stop = 4;', 'start:stop', '"11"', 'assertion failed'),
            (f'uint start = {self.module.QASM2QCXConverter.QCX_UINT_MAX}; int stop = 0;',
             'start:-1:stop', '"11"', 'assertion failed'),
            ('int start = 0; int stop = 1; int zero = 0;', 'start:1 / zero:stop', '"11"', 'division by zero'),
            ('int start = 0; int stop = 1; int zero = 0;', 'start:stop', 'flags << (1 / zero)', 'assertion failed'),
            ('int start = 0; int stop = 3; int zero = 0;', 'start:stop', 'flags << (1 / zero)', 'division by zero'),
            ('int start = 0; int stop = 1; int bad = 4;', 'start:stop', 'flags[{0, bad}]', 'assertion failed'),
        )
        for declarations, selector, rhs, error in cases:
            with self.subTest(selector=selector, declarations=declarations, rhs=rhs):
                lines = self.program('bit[4] flags = "1010"; ' + declarations + f'flags[{selector}] = {rhs};')
                instrumented = []
                for line in lines:
                    if line.startswith('LET FLAGS31:QASM2QCX_'):
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
    print(f'runtime bit-range assignment programs checked: {BackendTests.checked}')
    raise SystemExit(not result.wasSuccessful())


if __name__ == '__main__':
    main()
