#!/usr/bin/env python3
"""Audit bitwise compound assignments into runtime bit-register ranges (#450)."""

import argparse
import pathlib
import resource
import unittest

import qasm2qcx_angle_array_numerical as arrays


def combine(left, right, operator):
    return left & right if operator == '&' else left | right if operator == '|' else left ^ right


class BackendTests(unittest.TestCase):
    bra = None
    checked = 0
    setUp = arrays.BackendTests.setUp
    program = arrays.BackendTests.program
    execute = arrays.BackendTests.execute
    check_initializers = arrays.BackendTests.check_initializers

    def test_all_operators_ranges_and_wide_registers_against_bit_list_references(self):
        for width in (1, 4, 33, 65, 129):
            stride = max(1, (width - 1) // 2)
            for start, step, stop in {(0, stride, width - 1), (width - 1, -stride, 0), (-1, 1, -1)}:
                positions = [i % width for i in range(start, stop + (1 if step > 0 else -1), step)]
                count = len(positions)
                for before in sorted({'0' * width, ('101100' * width)[:width]}):
                    for payload in sorted({'0' * count, '1' * count, ('10' * count)[:count]}):
                        for operator in ('&', '|', '^'):
                            bits = list(map(int, reversed(before)))
                            for position, bit in zip(positions, reversed(payload)):
                                bits[position] = combine(bits[position], int(bit), operator)
                            expected = ''.join(map(str, reversed(bits)))
                            for rhs in ('source', f'"{payload}"'):
                                self.check_initializers(f'bit[{width}] flags = "{before}"; '
                                    f'bit[{count}] source = "{payload}"; int start = {start}; '
                                    f'{"uint" if step > 0 else "int"} step = {step}; int stop = {stop}; '
                                    f'flags[start:step:stop] {operator}= {rhs}; bool same = flags == "{expected}";',
                                    (), (), ('same', 'start', 'step', 'stop'), (1, start, step, stop))

    def test_overlapping_explicit_rhs_selections_use_original_bits(self):
        for pattern in range(16):
            for start, step, stop in ((0, 1, 3), (3, -1, 0), (0, 2, 3)):
                positions = list(range(start, stop + (1 if step > 0 else -1), step))
                rhs_positions = list(reversed(positions))
                rhs = '{' + ','.join('index' if i == rhs_positions[0] else str(i) for i in rhs_positions) + '}'
                for operator in ('&', '|', '^'):
                    bits = [(pattern >> i) & 1 for i in range(4)]
                    expected = bits.copy()
                    for destination, source in zip(positions, rhs_positions):
                        expected[destination] = combine(bits[destination], bits[source], operator)
                    text = ''.join(map(str, reversed(expected)))
                    self.check_initializers(f'bit[4] flags = "{pattern:04b}"; int start = {start}; '
                        f'int step = {step}; int stop = {stop}; uint index = {rhs_positions[0]}; '
                        f'flags[start:step:stop] {operator}= flags[{rhs}]; bool same = flags == "{text}";',
                        (), (), ('same',), (1,))

    def test_nested_bounds_and_register_rhs_consumers(self):
        for operator in ('&', '|', '^'):
            for rhs in ('rotl(pair, count)', '~pair', 'a[0:1]', 'C[{0, 3}]',
                        'flags[{index, 3}]', '"01" << count'):
                old = [1, 0, 0, 1]
                for position, value in zip((3, 1), (0, 1)):
                    old[position] = combine(old[position], value, operator)
                expected = ''.join(map(str, reversed(old)))
                self.check_initializers('bit[4] flags = "1001"; bit[2] pair = "01"; '
                    'const bit[4] C = "1000"; angle[2] a = angle[2]("10"); uint count = 1; int index = 2; '
                    'array[int, 2] bounds = {3, 1}; int start = 0; '
                    f'flags[bounds[start]:-2:bounds[start + 1]] {operator}= {rhs}; '
                    f'bool same = flags == "{expected}";', (), (), ('same',), (1,))
        self.check_initializers('bit[4] flags = "1001"; int index = 0; '
            'flags[int(flags[index]):int(flags[3]) + 1:3] ^= "11"; bool same = flags == "0011";',
            (), (), ('same',), (1,))

    def test_full_width_self_operations_and_native_endpoint_steps(self):
        for width in (1, 33, 65, 129):
            before = ('101100' * width)[:width]
            for operator in ('&', '|', '^'):
                expected = '0' * width if operator == '^' else before
                self.check_initializers(f'bit[{width}] flags = "{before}"; int start = 0; '
                    f'int stop = {width - 1}; flags[start:stop] {operator}= flags; '
                    f'bool same = flags == "{expected}";', (), (), ('same',), (1,))
        limits = self.module.QASM2QCXConverter
        for kind, step, start, stop in (('uint', limits.QCX_UINT_MAX, 0, 3),
                                        ('int', limits.QCX_INT_MIN, -1, -4)):
            self.check_initializers(f'bit[4] flags = "1001"; {kind} step = {step}; '
                f'int start = {start}; int stop = {stop}; flags[start:step:stop] ^= "1"; '
                f'bool same = flags == "{"1000" if start == 0 else "0001"}";',
                (), (), ('same',), (1,))

    def test_loop_forms_capture_new_ranges_and_original_bits_on_each_visit(self):
        for operator in ('&', '|', '^'):
            bits = [0, 1, 0, 1]
            for i in range(4):
                bits[i] = combine(bits[i], 1, operator)
            expected = ''.join(map(str, reversed(bits)))
            for loop in ('for int i in [first:last]', 'for int i in {0, 1, 2, 3}',
                         'for int i in {first, first + 1, last - 1, last}', 'for int i in indices'):
                self.check_initializers('bit[4] flags = "1010"; int first = 0; int last = 3; '
                    'array[int, 4] indices = {0, 1, 2, 3}; ' + loop +
                    f' {{ flags[i:i] {operator}= "1"; }} bool same = flags == "{expected}";',
                    (), (), ('same',), (1,))

    def test_live_register_iteration_observes_compound_writes_but_keeps_iterator_copy(self):
        for operator, payload in (('&', 0), ('|', 1), ('^', 1)):
            bits = [1, 1, 0, 1]
            next_index, total = 1, 0
            for i in range(4):
                value = bits[i]
                if value and next_index < 4:
                    bits[next_index] = combine(bits[next_index], payload, operator)
                    next_index += 1
                total += value
            expected = ''.join(map(str, reversed(bits)))
            self.check_initializers('bit[4] flags = "1011"; int next = 1; int total = 0; '
                f'for bit b in flags {{ if (b && next < 4) {{ flags[next:next] {operator}= "{payload}"; '
                f'next += 1; }} total += int(b); }} bool same = flags == "{expected}";',
                (), (), ('same', 'total', 'next'), (1, total, next_index))

    def test_while_reentry_and_changing_direction(self):
        for operator in ('&', '|', '^'):
            bits = [0, 1, 0, 1]
            mask = [0, 1, 1, 0]
            for i in range(3):
                positions = range(4) if i % 2 == 0 else range(3, -1, -1)
                for position, value in zip(positions, mask):
                    bits[position] = combine(bits[position], value, operator)
            expected = ''.join(map(str, reversed(bits)))
            self.check_initializers('bit[4] flags = "1010"; int i = 0; int start = 0; int stop = 3; int step = 1; '
                f'while (i < 3) {{ flags[start:step:stop] {operator}= "0110"; '
                'start = 3 - start; stop = 3 - stop; step = -step; i += 1; } '
                f'bool same = flags == "{expected}";', (), (), ('same', 'i'), (1, 3))

    def test_nested_transfers_and_shadowing_restore_outer_bindings(self):
        for operator in ('&', '|', '^'):
            bits = [0, 1, 0, 1]
            for i in (0, 1, 2, 3, 3):
                bits[i] = combine(bits[i], 1, operator)
            expected = ''.join(map(str, reversed(bits)))
            self.check_initializers('const int I = 3; int start = -1; bit[4] flags = "1010"; '
                'int first = 0; int last = 1; int visits = 0; for int I in [first:last] { '
                f'for int start in [0:2] {{ if (start == 1) {{ continue; }} flags[I:I] {operator}= "1"; '
                'visits += 1; break; } } '
                f'for int start in {{2, 3}} {{ flags[start:start] {operator}= "1"; }} '
                f'flags[I:I] {operator}= "1"; bool same = flags == "{expected}"; int restored = I;',
                (), (), ('same', 'visits', 'restored', 'start'), (1, 2, 3, -1))

    def test_skipped_range_and_rhs_failures_leave_targets_unchanged(self):
        for operator in ('&=', '|=', '^='):
            for selector, rhs in (('start:0:stop', '"11"'), ('bad:stop', '"11"'),
                                   ('start:1 / zero:stop', '"11"'), ('start:stop', '"1"'),
                                   ('-1:stop + 2', '"11111"'), ('start:stop', 'flags[{start, bad}]'),
                                   ('start:stop + 2', 'flags << (1 / zero)')):
                statement = f'flags[{selector}] {operator} {rhs};'
                for wrapper in ('if (false) {{ {} }}', 'while (false) {{ {} }}',
                                'for int outer in [1:0] {{ {} }}'):
                    self.check_initializers('bit[4] flags = "1010"; int start = 0; int stop = 1; '
                        'int bad = 4; int zero = 0; ' + wrapper.format(statement) +
                        'bool same = flags == "1010";', (), (), ('same',), (1,))

    def test_later_iteration_failures_do_not_write_any_destination(self):
        for operator in ('&', '|', '^'):
            for rhs, update, error in (('"11"', 'stop = 0;', 'assertion failed'),
                                       ('"11"', 'stop = 2;', 'assertion failed'),
                                       ('flags[{start, bad}]', 'bad = 4;', 'assertion failed'),
                                       ('pair << (1 / divisor)', 'divisor = 0;', 'division by zero')):
                bits = [0, 1, 0, 1]
                values = [1, 1] if rhs == '"11"' else [0, 1]
                for i in range(2):
                    bits[i] = combine(bits[i], values[i], operator)
                expected = ''.join(map(str, reversed(bits)))
                lines = self.program('bit[4] flags = "1010"; bit[2] pair = "01"; '
                    'int start = 0; int stop = 1; int bad = 1; int divisor = 1; int i = 0; '
                    f'while (i < 2) {{ flags[start:stop] {operator}= {rhs}; i += 1; {update} }}')
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
                self.assertEqual([''.join(line.split()) for line in result.stdout.splitlines()],
                                 ['12345', '12345', expected])
                type(self).checked += 1

    def test_measurement_controlled_compounds_and_feedback(self):
        for operator in ('&', '|', '^'):
            for _ in range(8):
                bits = [0, 1, 0, 1]
                outcomes = []
                for start in (0, 2):
                    expected = bits.copy()
                    for i, value in zip((start, start + 1), (1, 0)):
                        expected[i] = combine(bits[i], value, operator)
                    outcomes.append(''.join(map(str, reversed(expected))))
                self.check_initializers('include "stdgates.inc"; qubit q; h q; bit outcome = measure q; '
                    'bit[4] flags = "1010"; int start = 0; int stop = 1; '
                    f'if (outcome) {{ start = 2; stop = 3; }} flags[start:stop] {operator}= "01"; '
                    f'bool same = (!outcome && flags == "{outcomes[0]}") || (outcome && flags == "{outcomes[1]}"); '
                    'reset q; if (same) { x q; } bit observed = measure q;',
                    (), (), ('same', 'observed'), (1, 1))

    def test_updated_ranges_feed_phase_gates_and_fusion(self):
        import cmath
        import math
        from qasm2qcx_u_power_numerical import BackendAuditTests
        for operator in ('&', '|', '^'):
            bits = [0, 1, 0, 1]
            for position, value in zip((3, 1), (0, 1)):
                bits[position] = combine(bits[position], value, operator)
            pattern = sum(bit << i for i, bit in enumerate(bits))
            source = ('include "stdgates.inc"; qubit q; h q; bit[4] flags = "1010"; '
                      'int start = 3; int stop = 0; int step = -2; int i = 0; '
                      f'while (i < 1) {{ flags[start:step:stop] {operator}= "10"; i += 1; }} '
                      'p(angle[4](flags)) q; flags[start:step:stop] = "00";')
            expected = (1 / math.sqrt(2), cmath.exp(1j * pattern * math.pi / 8) / math.sqrt(2))
            for fused in (False, True):
                BackendAuditTests.check(self, self.program(source, fused=fused), expected)

    def test_destination_and_rhs_failures_happen_before_any_write(self):
        cases = (
            ('start:0:stop', '"11"', 'assertion failed'),
            ('start:stop + 1', '"11"', 'assertion failed'),
            ('start:start', '"11"', 'assertion failed'),
            ('stop:start', '"11"', 'assertion failed'),
            ('-1:stop + 2', '"11111"', 'assertion failed'),
            ('bad:stop', '"11"', 'assertion failed'),
            ('start:1 / zero:stop', '"11"', 'division by zero'),
            ('start:stop', 'flags << (1 / zero)', 'assertion failed'),
            ('start:stop + 2', 'flags << (1 / zero)', 'division by zero'),
            ('start:stop', 'flags[{start, bad}]', 'assertion failed'),
        )
        for operator in ('&=', '|=', '^='):
            for selector, rhs, error in cases:
                with self.subTest(operator=operator, selector=selector, rhs=rhs):
                    lines = self.program('bit[4] flags = "1010"; int start = 0; int stop = 1; '
                        f'int bad = 4; int zero = 0; flags[{selector}] {operator} {rhs};')
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
    print(f'runtime bit-range bitwise programs checked: {BackendTests.checked}')
    raise SystemExit(not result.wasSuccessful())


if __name__ == '__main__':
    main()
