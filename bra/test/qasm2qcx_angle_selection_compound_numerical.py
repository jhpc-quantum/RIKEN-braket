#!/usr/bin/env python3
"""Audit compound assignments to static angle bit selections (#424).

Pass --bra /path/to/bra for the default-double non-MPI backend audit.
"""

import argparse
import cmath
import math
import pathlib
import resource
import unittest

import qasm2qcx_static_angle_bit_selection_numerical as selections


class BackendTests(unittest.TestCase):
    bra = None
    checked = 0
    setUp = selections.BackendTests.setUp
    execute = selections.BackendTests.execute
    program = selections.BackendTests.program
    check_initializers = selections.BackendTests.check_initializers
    bits = staticmethod(selections.BackendTests.bits)
    replace_bits = staticmethod(selections.BackendTests.replace_bits)

    @staticmethod
    def combine(lhs, rhs, operator):
        if operator == '&=':
            return [left & right for left, right in zip(lhs, rhs)]
        if operator == '|=':
            return [left | right for left, right in zip(lhs, rhs)]
        return [left ^ right for left, right in zip(lhs, rhs)]

    @staticmethod
    def shift(bits, distance, operator):
        if distance >= len(bits):
            return [0] * len(bits)
        if operator == '<<=':
            return [0] * distance + bits[:len(bits) - distance]
        return bits[distance:] + [0] * distance

    def test_bitwise_compounds_at_every_width_preserve_unselected_bits(self):
        native = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        for width in range(1, native + 1):
            mask = (1 << width) - 1
            alternating = sum(1 << bit for bit in range(0, width, 2))
            for raw in sorted({0, alternating, mask}):
                sparse = (width - 1, 0) if width > 1 else (0,)
                selectors = (('0', (0,)), ('-1', (width - 1,)),
                             (f'0:{width - 1}', tuple(range(width))),
                             (f'{width - 1}:-1:0', tuple(reversed(range(width)))),
                             ('{' + ', '.join(map(str, sparse)) + '}', sparse),
                             (f'0:2:{width - 1}', tuple(range(0, width, 2))))
                source = (f'const angle[{width}] A = angle[{width}]("{raw:0{width}b}"); '
                          f'angle[{width}] value = A; angle[{width}] other = ~A; '
                          f'array[angle[{width}], {9 * len(selectors)}] results; ')
                expected = []
                for number, (selector, indices) in enumerate(selectors):
                    pattern = [index % 2 for index in range(len(indices))]
                    literal = '"' + ''.join(map(str, reversed(pattern))) + '"'
                    reverse = tuple(reversed(indices))
                    reverse_selector = '{' + ', '.join(map(str, reverse)) + '}'
                    source += f'bit[{len(indices)}] input{number} = {literal}; '
                    mixed = [left ^ right for left, right in zip(pattern, self.bits(raw ^ mask, indices))]
                    operands = ((literal, pattern), (f'input{number} ^ other[{selector}]', mixed),
                                (f'value[{reverse_selector}]', self.bits(raw, reverse)))
                    for operator in ('&=', '|=', '^='):
                        for rhs, right in operands:
                            bits = self.combine(self.bits(raw, indices), right, operator)
                            source += f'value = A; value[{selector}] {operator} {rhs}; results[{len(expected)}] = value; '
                            expected.append(self.replace_bits(raw, indices, bits))
                with self.subTest(width=width, raw=raw):
                    self.check_initializers(source, (('results', len(expected)),), expected,
                                            ('value', 'other'), (expected[-1], raw ^ mask))

    def test_array_destinations_negative_indices_and_overlapping_rhs(self):
        for operator in ('&=', '|=', '^='):
            for index in (-2, -1, 0, 1):
                target = index % 2
                values = [5, 10]
                bits = self.combine(self.bits(values[target], (0, 1, 2)),
                                    self.bits(values[1 - target], (3, 1, 0)), operator)
                values[target] = self.replace_bits(values[target], (0, 1, 2), bits)
                self.check_initializers('array[angle[4], 2] phases = {angle[4]("0101"), angle[4]("1010")}; '
                    f'int index = {index}; int other = {1 - target}; '
                    f'phases[index][0:2] {operator} phases[other][{{3, 1, 0}}];',
                    (('phases', 2),), values)
            values = [3, 10]
            bits = self.combine(self.bits(10, (3, 0)), self.bits(10, (0, 3)), operator)
            values[1] = self.replace_bits(10, (3, 0), bits)
            self.check_initializers('array[angle[4], 2] phases = {angle[4]("0011"), angle[4]("1010")}; '
                f'int index = -1; phases[index][{{3, 0}}] {operator} phases[index][{{0, 3}}];',
                (('phases', 2),), values)

    def test_singleton_arrays_constant_sources_and_repeated_rhs_bits(self):
        self.check_initializers('array[angle[1], 1] phases = {pi}; int index = -1; '
            'bit[1] one = "1"; phases[index][0] ^= one; phases[index][0] |= "1"; phases[index][0] &= "0";',
            (('phases', 1),), (0,))
        self.check_initializers('const angle[4] A = angle[4]("1001"); '
            'angle[4] value = angle[4]("1010"); value[0:3] ^= A[{0, 0, 3, 3}]; '
            'value[{3, 0}] |= A[{0, 0}];', (), (), ('value',), (13,))

    def test_reentry_transfers_and_shadowed_iterators(self):
        values = [3, 10]
        for _ in range(3):
            for index in range(2):
                bits = self.combine(self.bits(values[index], (0, 1, 2)), self.bits(values[index], (1, 2, 3)), '^=')
                values[index] = self.replace_bits(values[index], (0, 1, 2), bits)
        for outer in ('for int outer in [0:2]', 'for int outer in [0:stop]', 'while (visits < 3)'):
            self.check_initializers('array[angle[4], 2] phases = {angle[4]("0011"), angle[4]("1010")}; '
                'int stop = 2; int visits = 0; ' + outer + ' { visits += 1; '
                'for int index in [0:1] { phases[index][0:2] ^= phases[index][1:3]; }}',
                (('phases', 2),), values, ('visits',), (3,))
        self.check_initializers('const angle[4] value = pi; array[angle[4], 3] phases = {0.0, pi / 2, pi}; '
            'int index = 0; for angle[4] value in phases { '
            'if (value == angle[4](0.0)) { index += 1; continue; } '
            'phases[index][0:3] ^= value[3:-1:0]; index += 1; if (index == 2) { break; }} '
            'phases[0][0] |= value[-1];', (('phases', 3),), (1, 6, 8), ('index',), (2,))

    def test_skipped_checks_and_reached_errors(self):
        prefix = 'array[angle[4], 1] phases = {pi}; int bad = 1; int zero = 0; bit[2] pair = "11"; '
        for body in ('phases[bad][0:1] &= pair;', 'phases[0][0:1] |= phases[bad][0:1];',
                     'phases[1 / zero][0:1] ^= pair;', 'phases[0][0:1] ^= pair << (1 / zero);'):
            for wrapper in ('if (false) {{ {} }}', 'while (false) {{ {} }}', 'for int i in [1:0] {{ {} }}'):
                self.check_initializers(prefix + wrapper.format(body), (('phases', 1),), (8,))
            result = self.execute(self.program(prefix + body))
            self.assertNotEqual(result.returncode, 0, body)
            self.assertNotIn('runtime error:', result.stderr)
            type(self).checked += 1

    def test_shifts_at_every_width_follow_selection_order_and_bound_counts(self):
        native = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        maximum = self.module.QASM2QCXConverter.QCX_UINT_MAX
        for width in range(1, native + 1):
            mask = (1 << width) - 1
            alternating = sum(1 << bit for bit in range(0, width, 2))
            for raw in sorted({alternating, mask}):
                sparse = (width - 1, 0) if width > 1 else (0,)
                selectors = (('0', (0,)), ('-1', (width - 1,)),
                             (f'0:{width - 1}', tuple(range(width))),
                             (f'{width - 1}:-1:0', tuple(reversed(range(width)))),
                             ('{' + ', '.join(map(str, sparse)) + '}', sparse),
                             (f'0:2:{width - 1}', tuple(range(0, width, 2))))
                source = (f'const angle[{width}] A = angle[{width}]("{raw:0{width}b}"); '
                          f'angle[{width}] value = A; int count = 1; uint huge = uint(-1); '
                          f'array[angle[{width}], {14 * len(selectors)}] results; ')
                expected = []
                for selector, indices in selectors:
                    counts = (('', '0', 0), ('', '1', 1), ('', str(len(indices)), len(indices)),
                              ('', str(1 << 200), 1 << 200), ('count = 1;', 'count', 1),
                              (f'count = {len(indices) - 1};', 'count', len(indices) - 1), ('', 'huge', maximum))
                    for operator in ('<<=', '>>='):
                        for setup, count, distance in counts:
                            bits = self.shift(self.bits(raw, indices), distance, operator)
                            source += f'value = A; {setup} value[{selector}] {operator} {count}; results[{len(expected)}] = value; '
                            expected.append(self.replace_bits(raw, indices, bits))
                with self.subTest(width=width, raw=raw):
                    self.check_initializers(source, (('results', len(expected)),), expected,
                                            ('value',), (expected[-1],))

    def test_array_shift_indices_and_angle_derived_counts(self):
        for operator in ('<<=', '>>='):
            for index in (-2, -1, 0, 1):
                target = index % 2
                values = [9, 6]
                selected = self.bits(values[target], (3, 0, 2))
                values[target] = self.replace_bits(values[target], (3, 0, 2), self.shift(selected, 1, operator))
                self.check_initializers('array[angle[4], 2] phases = {angle[4]("1001"), angle[4]("0110")}; '
                    f'int index = {index}; uint[8] count = 1; phases[index][{{3, 0, 2}}] {operator} count;',
                    (('phases', 2),), values)
        self.check_initializers('array[angle[4], 3] phases = {angle[4]("1001"), angle[4]("0110"), angle[4]("0011")}; '
            'phases[0][{3, 0}] <<= popcount(phases[0][0:3]); '
            'phases[1][0:2] >>= phases[1] / angle[4]("0010"); '
            'phases[2][0:1] <<= uint[8](1);', (('phases', 3),), (0, 0, 2))

    def test_nested_runtime_reentry_transfers_and_count_shadowing(self):
        for loop in ('for int count in [-1:2]', 'for int count in [first:last]'):
            self.check_initializers('const angle[4] count = pi; '
                'array[angle[4], 2] phases = {angle[4]("0011"), 0.0}; '
                'int visits = 0; int first = -1; int last = 2; while (visits < 2) { visits += 1; ' + loop +
                ' { if (count < 0) { continue; } phases[0][0:2] <<= count; if (count == 1) { break; } } '
                'phases[1][0] |= count[-1]; }', (('phases', 2),), (4, 1), ('visits',), (2,))

    def test_negative_counts_and_operand_failures_only_on_reached_paths(self):
        prefix = ('array[angle[4], 1] phases = {angle[4]("1001")}; '
                  'int negative = -1; int minimum = -2147483648; int zero = 0; int bad = 1; '
                  'array[int, 1] counts = {1}; ')
        for operator in ('<<=', '>>='):
            for count in ('-1', 'negative', 'minimum', '1 / zero', 'counts[bad]'):
                body = f'phases[0][0:2] {operator} {count};'
                for wrapper in ('if (false) {{ {} }}', 'while (false) {{ {} }}', 'for int i in [1:0] {{ {} }}'):
                    self.check_initializers(prefix + wrapper.format(body), (('phases', 1),), (9,))
                result = self.execute(self.program(prefix + body))
                self.assertNotEqual(result.returncode, 0, (operator, count))
                self.assertNotIn('runtime error:', result.stderr)
                type(self).checked += 1

    def test_all_compound_operators_preserve_captured_gate_parameters_with_fusion(self):
        from qasm2qcx_u_power_numerical import BackendAuditTests
        changes = (('&=', '"110"', 7, 6), ('|=', '"100"', 2, 6),
                   ('^=', '"101"', 3, 6), ('<<=', 'count', 3, 6), ('>>=', 'count', 6, 3))
        gates = (
            ('p({angle}) q[0];', 1, lambda index: bool(index & 1)),
            ('inv @ p({angle}) q[0];', -1, lambda index: bool(index & 1)),
            ('pow(exponent) @ p({angle}) q[0];', 2, lambda index: bool(index & 1)),
            ('pow(0.5) @ p({angle}) q[0];', 0.5, lambda index: bool(index & 1)),
            ('cp({angle}) q[0], q[1];', 1, lambda index: index == 3),
            ('negctrl @ p({angle}) q[0], q[1];', 1, lambda index: index == 2),
            ('gphase({angle});', 1, lambda index: True),
        )
        for operator, rhs, before, after in changes:
            phase = (before + after) * math.pi / 8
            for array in (False, True):
                operand = 'phases[index]' if array else 'value'
                declaration = (f'array[angle[4], 1] phases = {{angle[4]("{before:04b}")}}; int index = 0; '
                               if array else f'angle[4] value = angle[4]("{before:04b}"); ')
                for gate, factor, active in gates:
                    operation = gate.format(angle=operand)
                    source = ('include "stdgates.inc"; qubit[2] q; h q; int count = 1; int exponent = 2; '
                              + declaration + operation + f'{operand}[0:2] {operator} {rhs}; '
                              + operation + f'{operand}[0:3] ^= "1111";')
                    expected = tuple(cmath.exp(1j * factor * phase) / 2 if active(index) else 0.5 for index in range(4))
                    for fused in (False, True):
                        with self.subTest(operator=operator, gate=gate, array=array, fused=fused):
                            BackendAuditTests.check(self, self.program(source, fused=fused), expected)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra', type=pathlib.Path)
    arguments = parser.parse_args()
    BackendTests.bra = arguments.bra.resolve() if arguments.bra else None
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(BackendTests))
    print(f'angle-selection compound programs checked: {BackendTests.checked}')
    raise SystemExit(not result.wasSuccessful())
