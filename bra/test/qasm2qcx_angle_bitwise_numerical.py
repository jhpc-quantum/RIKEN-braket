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
    check_phase = arithmetic.BackendTests.check_phase
    check_zero_qubit = arithmetic.BackendTests.check_zero_qubit

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

    def test_compound_assignments_and_self_reference_at_every_width(self):
        for width in range(1, self.module.QASM2QCXConverter.QCX_UINT_WIDTH + 1):
            mask = (1 << width) - 1
            for lhs, rhs in ((0, mask), (mask, mask), (1, mask // 3), (mask // 3, mask * 2 // 3)):
                self.exact(f'''angle[{width}] a; angle[{width}] b;
                    angle[{width}] both = a; both &= b; angle[{width}] either = a; either |= b;
                    angle[{width}] exclusive = a; exclusive ^= b;
                    angle[{width}] ones = a; ones |= ~ones;
                    angle[{width}] zero = a; zero ^= zero;
                    angle[{width}] self = a; self ^= ~self | b;''',
                    ('a', 'b', 'both', 'either', 'exclusive', 'ones', 'zero', 'self'),
                    (lhs, rhs, lhs & rhs, lhs | rhs, lhs ^ rhs, mask, 0, lhs ^ ((lhs ^ mask) | rhs)),
                    raw=(('a', lhs), ('b', rhs)))

    def test_mixed_width_compound_results(self):
        native = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        for left_width in sorted({1, 4, 8, native}):
            for right_width in sorted({1, 4, 8, native}):
                width = max(left_width, right_width)
                for lhs, rhs in ((1, 1), ((1 << left_width) - 1, 1), (1, (1 << right_width) - 1)):
                    left, right = lhs << (width - left_width), rhs << (width - right_width)
                    inverse_left = (lhs ^ ((1 << left_width) - 1)) << (width - left_width)
                    self.exact(f'''angle[{left_width}] a; angle[{right_width}] b;
                        angle[{left_width}] both = a; both &= b;
                        angle[{left_width}] either = a; either |= b;
                        angle[{left_width}] exclusive = a; exclusive ^= b;
                        angle[{left_width}] self = a; self ^= ~self | b;''',
                        ('a', 'b', 'both', 'either', 'exclusive', 'self'),
                        (lhs, rhs, (left & right) >> (width - left_width), (left | right) >> (width - left_width),
                         (left ^ right) >> (width - left_width), (left ^ (inverse_left | right)) >> (width - left_width)),
                        raw=(('a', lhs), ('b', rhs)))

    def test_compound_reentry_shadowing_and_measurement_control(self):
        self.exact('''const angle[4] A = pi; angle[4] a; array[float, 2] values = {pi / 2, pi};
            for float A in values { a |= angle[8](A); } a &= A;
            int count = 0; while (count < 3) { a ^= angle[4](pi / 4); count += 1; }
            for int i in [0:3] { if (i == 1) { continue; } a ^= angle[4](pi / 8); if (i == 2) { break; } }''',
            ('a', 'count'), (10, 3), raw=(('a', 1),))
        for measured in (0, 1):
            self.exact('''include "stdgates.inc"; qubit q; bit outcome; angle[4] a;
                ''' + ('x q; ' if measured else '') + '''outcome = measure q;
                if (outcome) { a ^= angle[8](pi); } else { a &= ~a; }
                bool nonzero = bool(a);''', ('a', 'outcome', 'nonzero'),
                (15 if measured else 0, measured, measured), raw=(('a', 7),))

    def test_compound_phase_matrices_modifiers_and_fusion(self):
        prefix = '''angle[4] a = 7 * pi / 8; angle[8] fine = pi / 128;
            a |= fine; a ^= angle[4](pi / 4); a &= angle[8](15 * pi / 8); '''
        for fused in (False, True):
            for gate in ('p', 'phase', 'u1', 'cp', 'cphase'):
                controlled = gate in ('cp', 'cphase')
                wires, operands = ((2, 0), 'q[2], q[0]') if controlled else ((0,), 'q[0]')
                for basis in (*range(8), None):
                    self.check_phase(prefix + f'{gate}(a) {operands};', 4, 5, (wires,),
                                     fused=fused, basis=basis, preserved=(('a', 5), ('fine', 1)))
            self.check_phase(prefix + 'negctrl @ inv @ pow(0.5) @ p(a) q[2], q[0];',
                             5, 27, (((2, 0), 0),), fused=fused)
            self.check_phase(prefix + 'if ((~a & fine) != a) { gphase(a); }',
                             4, 5, ((),), fused=fused)
            self.check_zero_qubit(prefix + 'pow(0.5) @ gphase(a);', 5, 5, fused=fused)

    def test_compound_operand_checks_follow_reached_and_skipped_paths(self):
        prefix = 'angle[4] a; const angle[4] A = pi; const uint[4] ZERO = 0; '
        for operator in ('&=', '|=', '^='):
            for rhs in ('A / ZERO', 'angle[4](1e309)'):
                statement = f'a {operator} {rhs};'
                result = self.execute(self.program(prefix + statement))
                self.assertNotEqual(result.returncode, 0)
                type(self).checked += 1
                for wrapper in ('if (false) { %s }', 'while (false) { %s }', 'for int i in [1:0] { %s }'):
                    self.exact(prefix + wrapper % statement, ('a',), (7,), raw=(('a', 7),))

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
