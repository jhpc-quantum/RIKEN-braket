#!/usr/bin/env python3
"""Audit scalar angle comparisons using exact stored-bit fixtures (#402).

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

    def compare(self, left_width, left, right_width, right):
        width = max(left_width, right_width)
        lhs = left << (width - left_width)
        rhs = right << (width - right_width)
        self.exact(f'''angle[{left_width}] a; angle[{right_width}] b;
            bool equal = a == b; bool unequal = a != b;
            bool less = a < b; bool less_equal = a <= b;
            bool greater = a > b; bool greater_equal = a >= b;
            int branch = 0; if (a < b) {{ branch = 1; }} else {{ branch = 2; }}
            bool combined = !(a == b) && (a < b || b < a);''',
            ('a', 'b', 'equal', 'unequal', 'less', 'less_equal', 'greater', 'greater_equal', 'branch', 'combined'),
            (left, right, int(lhs == rhs), int(lhs != rhs), int(lhs < rhs), int(lhs <= rhs),
             int(lhs > rhs), int(lhs >= rhs), 1 if lhs < rhs else 2, int(lhs != rhs)),
            raw=(('a', left), ('b', right)))

    def test_all_operators_at_every_supported_width(self):
        for width in range(1, self.module.QASM2QCXConverter.QCX_UINT_WIDTH + 1):
            for lhs in sorted({0, 1, 1 << (width - 1), (1 << width) - 1}):
                for rhs in sorted({0, 1, 1 << (width - 1), (1 << width) - 1}):
                    self.compare(width, lhs, width, rhs)

    def test_mixed_width_promotion_preserves_fractional_turns(self):
        native = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        for left_width in sorted({1, 4, 8, native}):
            for right_width in sorted({1, 4, 8, native}):
                for lhs, rhs in ((1, 1), (1 << (left_width - 1), 1 << (right_width - 1)),
                                 ((1 << left_width) - 1, (1 << right_width) - 1)):
                    self.compare(left_width, lhs, right_width, rhs)

    def test_constants_literal_casts_computed_operands_and_reentry(self):
        self.exact('''const angle[4] A = pi; angle[4] a = A; angle[8] b = A;
            bool equal = A == b; bool cast_equal = angle[4](pi) == b;
            bool wrapped = (a * uint[4](2)) < a;
            bool quotient = (a / a) == uint[4](1);
            int count = 0; while (a < angle[4](3 * pi / 2)) { a += angle[4](pi / 8); count += 1; }
            bool final = a > b;''', ('a', 'b', 'equal', 'cast_equal', 'wrapped', 'quotient', 'count', 'final'),
            (12, 128, 1, 1, 1, 1, 4, 1))
        self.exact('const angle[4] A = pi; bool result = A == A;', ('result',), (1,))

    def test_operand_checks_follow_reached_and_short_circuit_paths(self):
        for expression in ('a / zero == a', 'angle[4](1e309) < a', 'a < angle[4](values[index])'):
            prefix = 'angle[4] a; uint[4] zero = 0; array[float, 2] values = {0.0, 0.1}; int index = 2; '
            result = self.execute(self.program(prefix + f'bool result = {expression};'))
            self.assertNotEqual(result.returncode, 0, expression)
            type(self).checked += 1
            for statement in (f'bool result = false && ({expression});',
                              f'bool result = true || ({expression});',
                              f'for int i in [1:0] {{ if ({expression}) {{ }} }}'):
                self.exact(prefix + statement, ('a',), (7,), raw=(('a', 7),))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    BackendTests.bra = parser.parse_args().bra.resolve()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(BackendTests))
    print(f'angle comparison programs checked: {BackendTests.checked}')
    raise SystemExit(not result.wasSuccessful())


if __name__ == '__main__':
    main()
