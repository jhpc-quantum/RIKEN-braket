#!/usr/bin/env python3
"""Audit compound assignments to static angle bit selections (#424).

Pass --bra /path/to/bra for the default-double non-MPI backend audit.
"""

import argparse
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra', type=pathlib.Path)
    arguments = parser.parse_args()
    BackendTests.bra = arguments.bra.resolve() if arguments.bra else None
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(BackendTests))
    print(f'angle-selection compound programs checked: {BackendTests.checked}')
    raise SystemExit(not result.wasSuccessful())
