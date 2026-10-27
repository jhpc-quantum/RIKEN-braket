#!/usr/bin/env python3
"""Audit scalar angle bitwise operations with exact stored-bit fixtures (#404).

Run with --bra /path/to/bra (including a UBSan build).
"""

import argparse
import pathlib
import resource
import unittest

import qasm2qcx_angle_arithmetic_numerical as arithmetic


class BackendTests(unittest.TestCase):
    bra = None
    checked = 0
    setUp = arithmetic.BackendTests.setUp
    program = arithmetic.BackendTests.program
    execute = arithmetic.BackendTests.execute
    exact = arithmetic.BackendTests.exact

    def test_exact_bitwise_results_at_every_supported_width(self):
        for width in range(1, self.module.QASM2QCXConverter.QCX_UINT_WIDTH + 1):
            mask = (1 << width) - 1
            patterns = sorted({0, 1, 1 << (width - 1), mask})
            for lhs in patterns:
                for rhs in patterns:
                    self.exact(f'''angle[{width}] a; angle[{width}] b;
                        angle[{width}] complement = ~a; angle[{width}] both = a & b;
                        angle[{width}] either = a | b; angle[{width}] exclusive = a ^ b;
                        angle[{width}] nested = ~(a & b) | (a ^ b);
                        angle[{width}] self = a; self = ~(self ^ b);
                        bool nonzero = bool(a & b); bool equal = (a ^ b) == angle[{width}](0.0);''',
                        ('a', 'b', 'complement', 'both', 'either', 'exclusive', 'nested', 'self', 'nonzero', 'equal'),
                        (lhs, rhs, lhs ^ mask, lhs & rhs, lhs | rhs, lhs ^ rhs,
                         ((lhs & rhs) ^ mask) | (lhs ^ rhs), (lhs ^ rhs) ^ mask, int(bool(lhs & rhs)), int(lhs == rhs)),
                        raw=(('a', lhs), ('b', rhs)))

    def test_mixed_width_alignment_and_destination_narrowing(self):
        native = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        for left_width in sorted({1, 4, 8, native}):
            for right_width in sorted({1, 4, 8, native}):
                width, small = max(left_width, right_width), min(left_width, right_width)
                mask = (1 << width) - 1
                for lhs, rhs in ((1, 1), ((1 << left_width) - 1, 1), (1, (1 << right_width) - 1)):
                    left, right = lhs << (width - left_width), rhs << (width - right_width)
                    self.exact(f'''angle[{left_width}] a; angle[{right_width}] b;
                        angle[{width}] both = a & b; angle[{width}] either = a | b;
                        angle[{width}] exclusive = a ^ b; angle[{small}] narrow = a | b;
                        angle[{width}] before = ~a ^ b; angle[{width}] after = ~(a | b);''',
                        ('a', 'b', 'both', 'either', 'exclusive', 'narrow', 'before', 'after'),
                        (lhs, rhs, left & right, left | right, left ^ right, (left | right) >> (width - small),
                         (((1 << left_width) - 1) ^ lhs) << (width - left_width) ^ right, (left | right) ^ mask),
                        raw=(('a', lhs), ('b', rhs)))

    def test_native_sizedness_composition_and_reentry(self):
        width = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        mask = (1 << width) - 1
        self.exact(f'''angle a; angle[{width}] b;
            angle result = ~a ^ b; angle[{width}] sized = a | b;
            int count = 0; while (count < 2) {{ result = ~result; count += 1; }}''',
            ('a', 'b', 'result', 'sized'), (1, mask, 1, mask), raw=(('a', 1), ('b', mask)))
        self.exact('''angle[4] a; angle[8] b; angle[8] result = (a | b) * uint[8](2);
            uint[8] ratio = (a | b) / b; uint count = popcount((a | b) / b);
            bool greater = (a | b) > a;''', ('a', 'b', 'result', 'ratio', 'count', 'greater'),
            (7, 1, 226, 113, 4, 1), raw=(('a', 7), ('b', 1)))

    def test_zero_patterns_do_not_suppress_operand_checks(self):
        prefix = 'angle[4] a; uint[4] zero = 0; array[float, 2] values = {0.0, 0.1}; int index = 2; '
        for expression in ('a & (a / zero)', '(a / zero) | a', '~(a / zero)',
                           'a & angle[4](1e309)', 'a & angle[4](values[index])'):
            result = self.execute(self.program(prefix + f'angle[4] result = {expression};'))
            self.assertNotEqual(result.returncode, 0, expression)
            type(self).checked += 1
            for body in (f'bool result = false && bool({expression});',
                         f'bool result = true || bool({expression});',
                         f'for int i in [1:0] {{ a = {expression}; }}'):
                self.exact(prefix + body, ('a',), (0,))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    BackendTests.bra = parser.parse_args().bra.resolve()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(BackendTests))
    print(f'angle bitwise programs checked: {BackendTests.checked}')
    raise SystemExit(not result.wasSuccessful())


if __name__ == '__main__':
    main()
