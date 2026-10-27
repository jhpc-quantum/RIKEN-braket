#!/usr/bin/env python3
"""Audit scalar angle comparisons using exact stored-bit fixtures (#402).

Run with --bra /path/to/bra (including a UBSan build).
"""

import argparse
import cmath
import math
import pathlib
import re
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
    check_phase = arithmetic.BackendTests.check_phase
    check_zero_qubit = arithmetic.BackendTests.check_zero_qubit

    def test_documented_comparison_examples(self):
        document = (pathlib.Path(__file__).parents[1] / 'qcx' / 'qasm2qcx.md').read_text()
        examples = {}
        for marker in ('angle-comparison-example', 'angle-comparison-phase-example'):
            matches = re.findall(r'<!-- ' + marker + r' -->\s*```qasm\n(.*?)\n```', document, flags=re.DOTALL)
            self.assertEqual(len(matches), 1)
            examples[marker] = matches[0]
        self.exact(examples['angle-comparison-example'],
                   ('a', 'b', 'greater', 'wrapped', 'fine_order', 'count', 'equal', 'skipped'),
                   (8, 128, 1, 1, 0, 4, 1, 0))
        for fused in (False, True):
            lines = self.program(examples['angle-comparison-phase-example'])
            if fused:
                lines.insert(1, 'BEGIN FUSION')
                lines.insert(len(lines) - 1, 'END FUSION')
            result = self.execute(lines)
            self.assertEqual(result.returncode, 0, result.stderr)
            actual = {int(index, 2): complex(float(real), float(imaginary))
                      for index, real, imaginary in re.findall(r'^([01]+) => (\S+) \+ (\S+) i$', result.stdout, re.MULTILINE)}
            self.assertEqual(set(actual), set(range(8)), result.stdout)
            for index in range(8):
                phase = (5 * math.pi / 8 if index & 1 else 0)
                phase += 5 * math.pi / 16 if not index & 4 else 0
                self.assertTrue(cmath.isclose(actual[index], cmath.exp(1j * phase) / math.sqrt(8),
                                             rel_tol=1e-5, abs_tol=2e-6), (index, actual[index]))
            type(self).checked += 1

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

    def test_exhaustive_four_bit_runtime_truth_tables(self):
        for lhs in range(16):
            for rhs in range(16):
                self.compare(4, lhs, 4, rhs)

    def test_sized_unsized_native_operands_and_boolean_destinations(self):
        width = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        for lhs, rhs in ((0, 0), (1, 1), ((1 << width) - 1, 0), (0, (1 << width) - 1)):
            self.exact(f'''angle a; angle[{width}] b;
                bool equal = a == b; bool less = b < a;
                int numeric = a >= b; bit flag = a != b;
                array[bool, 2] results = {{false, false}}; int index = 1;
                results[index] = a <= b; bool readback = results[index];''',
                ('a', 'b', 'equal', 'less', 'numeric', 'flag', 'readback'),
                (lhs, rhs, int(lhs == rhs), int(rhs < lhs), int(lhs >= rhs), int(lhs != rhs), int(lhs <= rhs)),
                raw=(('a', lhs), ('b', rhs)))

    def test_iterator_shadowing_and_outer_angle_binding_restoration(self):
        self.exact('''const angle[4] A = pi; angle[4] a; bool result = false; int hits = 0;
            array[float, 3] values = {pi / 2, pi, 3 * pi / 2};
            for float A in values { result = a < angle[8](A); if (result) { hits += 1; } }
            bool restored = a < A;
            for int A in {0, 1, 2} { if (A < A + 1) { hits += 1; } }
            bool restored_again = a < A;''', ('a', 'result', 'hits', 'restored', 'restored_again'),
            (7, 1, 5, 1, 1), raw=(('a', 7),))

    def test_bit_register_comparisons_with_angle_quotient_counts(self):
        for operand in ('B << uint(A / A)', 'B >> uint(A / A)',
                        'rotl(B, uint(A / A))', 'rotr(B, uint(A / A))'):
            for declarations in ('const angle[4] A = pi; const bit[4] B = "0001";',
                                 'angle[4] A = pi; bit[4] B = "0001";'):
                self.exact(declarations + f'''bool equal = ({operand}) == B;
                    bool unequal = ({operand}) != B; int branch = 0;
                    if (({operand}) != B) {{ branch = 1; }}
                    for int i in [1:0] {{ if (({operand}) == B) {{ branch = 2; }} }}''',
                    ('equal', 'unequal', 'branch'), (0, 1, 1))

    def test_repeated_conditions_break_continue_and_measurement(self):
        self.exact('''angle[4] a; angle[8] limit = 3 * pi / 2; int count = 0; int hits = 0;
            while (a < limit) { a += angle[4](pi / 8); count += 1;
                if (a == angle[8](pi / 4)) { continue; }
                hits += 1; if (a >= angle[4](pi / 2)) { break; } }
            for int i in [0:2] { if (a <= angle[8](pi)) { a += angle[4](pi / 8); } }''',
            ('a', 'count', 'hits', 'limit'), (7, 4, 3, 192), raw=(('a', 0),))
        for measured in (0, 1):
            self.exact('''include "stdgates.inc"; qubit q; bit outcome;
                angle[4] a; angle[8] b = pi;
                ''' + ('x q; ' if measured else '') + '''outcome = measure q;
                if (outcome && a < b) { a += angle[4](pi / 8); }
                else { a -= angle[4](pi / 8); }
                bool result = a == b;''', ('a', 'b', 'outcome', 'result'),
                (8 if measured else 6, 128, measured, measured), raw=(('a', 7),))

    def test_conditional_phase_matrices_preserve_full_phase_and_sources(self):
        cases = (('==', False), ('!=', True), ('<', True), ('<=', True), ('>', False), ('>=', False))
        prefix = 'angle[4] a = 5 * pi / 8; angle[8] b = pi; '
        for fused in (False, True):
            for op, expected in cases:
                for basis in (*range(8), None):
                    self.check_phase(prefix + f'''if (a {op} b) {{
                        negctrl @ p(a) q[2], q[0];
                    }} else {{ negctrl @ p(-a) q[2], q[0]; }}''',
                        4, 5 if expected else 11, (((2, 0), 0),), fused=fused, basis=basis,
                        preserved=(('a', 5), ('b', 128)))
            self.check_phase(prefix + 'if (a < b && a != b) { gphase(a); }',
                             4, 5, ((),), fused=fused, preserved=(('a', 5), ('b', 128)))
            self.check_zero_qubit(prefix + 'if (a < b) { pow(0.5) @ gphase(a); }',
                                  5, 5, fused=fused, preserved=(('a', 5), ('b', 128)))

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

    def test_comparison_does_not_skip_evaluation_of_either_operand(self):
        prefix = 'angle[4] a; const angle[4] A = pi; const uint[4] ZERO = 0; '
        for expression in ('A / ZERO == a', 'a < A / ZERO',
                           'uint[4](0) * (A / ZERO) == a', 'a == angle[4](1e309)'):
            result = self.execute(self.program(prefix + f'bool result = {expression};'))
            self.assertNotEqual(result.returncode, 0, expression)
            type(self).checked += 1
            for body in (f'if (false) {{ if ({expression}) {{ }} }}',
                         f'while (false) {{ if ({expression}) {{ }} }}',
                         f'bool result = true || ({expression});'):
                self.exact(prefix + body, ('a',), (7,), raw=(('a', 7),))


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
