#!/usr/bin/env python3
"""Audit scalar angle shifts with exact stored-bit fixtures (#406).

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

    def test_every_width_and_boundary_counts_with_both_count_types(self):
        native = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        for width in range(1, native + 1):
            mask = (1 << width) - 1
            for bits in sorted({0, 1, 1 << (width - 1), mask}):
                for count in sorted({0, 1, width - 1, width, width + 1, self.module.QASM2QCXConverter.QCX_UINT_MAX}):
                    for count_type in ('int', 'uint'):
                        if count_type == 'int' and count > self.module.QASM2QCXConverter.QCX_INT_MAX:
                            continue
                        left = (bits << count) & mask if count < width else 0
                        right = bits >> count if count < width else 0
                        self.exact(f'''angle[{width}] a; {count_type} count = {count};
                            angle[{width}] left = a << count; angle[{width}] right = a >> count;''',
                            ('a', 'count', 'left', 'right'), (bits, count, left, right), raw=(('a', bits),))

    def test_literal_counts_sizedness_and_rhs_widths(self):
        native = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        for count in (0, 1, 4, self.module.QASM2QCXConverter.QCX_UINT_MAX, 10**100):
            self.exact(f'''const angle[4] A = 9 * pi / 8; angle[4] a = A;
                angle[4] result = a << {count}; angle[4] folded = A >> {count};''',
                ('a', 'result', 'folded'), (9, (9 << count) & 15 if count < 4 else 0, 9 >> count if count < 4 else 0))
        self.exact('''angle[4] a; uint[8] wide = 16; uint[1] small = 1;
            angle[4] zero = a >> wide; angle[4] shifted = a >> small;''',
            ('a', 'wide', 'small', 'zero', 'shifted'), (9, 16, 1, 0, 4), raw=(('a', 9),))
        self.exact(f'''angle a; uint[1] small = 1; angle result = a >> small;
            angle[{native}] sized = a << small;''', ('a', 'result', 'sized'),
            ((1 << native) - 1, (1 << (native - 1)) - 1, (1 << native) - 2), raw=(('a', (1 << native) - 1),))

    def test_negative_counts_fail_only_when_reached(self):
        minimum = self.module.QASM2QCXConverter.QCX_INT_MIN
        for count in (-1, minimum):
            for operator in ('<<', '>>'):
                for expression in (str(count), 'count'):
                    prefix = f'const angle[4] A = pi; angle[4] a = A; int count = {count}; '
                    result = self.execute(self.program(prefix + f'angle[4] result = A {operator} {expression};'))
                    self.assertNotEqual(result.returncode, 0)
                    type(self).checked += 1
                    for body in (f'if (false) {{ a = a {operator} {expression}; }}',
                                 f'bool skipped = false && bool(a {operator} {expression});',
                                 f'while (false) {{ a = a {operator} {expression}; }}',
                                 f'for int i in [1:0] {{ a = a {operator} {expression}; }}'):
                        self.exact(prefix + body, ('a',), (8,))

    def test_operands_are_evaluated_even_for_zero_results(self):
        prefix = 'angle[4] a; uint zero = 0; array[float, 2] values = {0.0, 0.1}; int index = 2; '
        for expression in ('(a / uint[4](0)) << 1000', 'angle[4](1e309) >> 1000',
                           'angle[4](values[index]) << 1000', 'a >> (1 / zero)'):
            result = self.execute(self.program(prefix + f'angle[4] result = {expression};'))
            self.assertNotEqual(result.returncode, 0, expression)
            type(self).checked += 1
            self.exact(prefix + f'bool skipped = true || bool({expression});', ('a',), (0,))

    def test_composition_and_repeated_execution(self):
        self.exact('''angle[4] a; array[uint[8], 3] counts = {0, 1, 4}; int index = 0;
            angle[4] result; while (index < 3) { result = a << counts[index]; index += 1; }
            angle[8] combined = (a >> 1) | angle[8](pi / 128);
            uint[4] ratio = (a >> 1) / angle[4](pi / 8);
            bool smaller = (a >> 1) < a;''', ('a', 'result', 'combined', 'ratio', 'smaller'),
            (9, 0, 65, 4, 1), raw=(('a', 9),))

    def test_compound_shifts_at_every_width_and_count_boundary(self):
        native = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        for width in range(1, native + 1):
            mask = (1 << width) - 1
            for bits in sorted({0, 1, 1 << (width - 1), mask}):
                for count in sorted({0, 1, width - 1, width, width + 1, self.module.QASM2QCXConverter.QCX_UINT_MAX}):
                    left = (bits << count) & mask if count < width else 0
                    right = bits >> count if count < width else 0
                    self.exact(f'''angle[{width}] a; uint count = {count};
                        angle[{width}] left = a; left <<= count;
                        angle[{width}] right = a; right >>= count;''',
                        ('a', 'count', 'left', 'right'), (bits, count, left, right), raw=(('a', bits),))

    def test_self_referencing_counts_and_repeated_assignments(self):
        self.exact('''const angle[4] UNIT = pi / 8; angle[4] a;
            a >>= a / a; a <<= a / UNIT;
            angle native = pi / 2; uint[1] small = 1; native <<= small; native >>= small;''',
            ('a', 'native'), (0, 1 << (self.module.QASM2QCXConverter.QCX_UINT_WIDTH - 2)), raw=(('a', 9),))
        self.exact('''const int COUNT = 1; angle[4] a; array[uint[8], 3] counts = {0, 1, 4};
            for uint[8] COUNT in counts { a >>= COUNT; } a <<= COUNT;''', ('a',), (0,), raw=(('a', 15),))
        self.exact('''angle[4] a; int count = 0;
            while (count < 3) { a >>= count; if (a < angle[4](pi)) { a <<= 1; } count += 1; }
            for int i in [0:3] { if (i == 1) { continue; } a >>= 1; if (i == 2) { break; } }''',
            ('a', 'count'), (1, 3), raw=(('a', 15),))

    def test_compound_failures_and_skipped_execution(self):
        prefix = 'angle[4] a; int negative = -1; uint zero = 0; '
        for operator in ('<<=', '>>='):
            for count in ('negative', '-1', '1 / zero', 'a / angle[4](0.0)'):
                statement = f'a {operator} {count};'
                result = self.execute(self.program(prefix + statement))
                self.assertNotEqual(result.returncode, 0)
                type(self).checked += 1
                for wrapper in ('if (false) { %s }', 'while (false) { %s }', 'for int i in [1:0] { %s }'):
                    self.exact(prefix + wrapper % statement, ('a',), (9,), raw=(('a', 9),))

    def test_measurement_controlled_updates(self):
        for measured in (0, 1):
            self.exact('''include "stdgates.inc"; qubit q; bit outcome; angle[4] a;
                ''' + ('x q; ' if measured else '') + '''outcome = measure q;
                if (outcome) { a <<= uint[8](1); } else { a >>= 1; }
                bool smaller = a < angle[4](pi);''', ('a', 'outcome', 'smaller'),
                (2 if measured else 4, measured, 1), raw=(('a', 9),))

    def test_shifted_phase_matrices_modifiers_and_fusion(self):
        prefix = '''angle[4] a = 9 * pi / 8; uint[8] count = 1;
            a >>= count; a <<= 1; a ^= angle[4](pi / 8); '''
        for fused in (False, True):
            for gate in ('p', 'phase', 'u1', 'cp', 'cphase'):
                controlled = gate in ('cp', 'cphase')
                wires, operands = ((2, 0), 'q[2], q[0]') if controlled else ((0,), 'q[0]')
                for basis in (*range(8), None):
                    self.check_phase(prefix + f'{gate}(a) {operands};', 4, 9, (wires,),
                                     fused=fused, basis=basis, preserved=(('a', 9), ('count', 1)))
            self.check_phase(prefix + 'negctrl @ inv @ pow(0.5) @ p(a) q[2], q[0];',
                             5, 7, (((2, 0), 0),), fused=fused)
            self.check_phase(prefix + 'if ((a >> 1) < a) { gphase(a); }', 4, 9, ((),), fused=fused)
            self.check_zero_qubit(prefix + 'pow(0.5) @ gphase(a);', 5, 25, fused=fused)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    BackendTests.bra = parser.parse_args().bra.resolve()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(BackendTests))
    print(f'angle shift programs checked: {BackendTests.checked}')
    raise SystemExit(not result.wasSuccessful())


if __name__ == '__main__':
    main()
