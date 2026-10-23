#!/usr/bin/env python3

"""Check scalar angle storage with a default-double non-MPI bra.

Stage 1 injects valid raw UINT bit patterns into generated QCX to test storage
and copying independently of the later floating-point conversion kernel.

  python3 bra/test/qasm2qcx_angle_numerical.py --bra /path/to/bra
"""

import argparse
import math
import pathlib
import resource
import unittest
from fractions import Fraction

import qasm2qcx_math_function_numerical as functions


class BackendTests(unittest.TestCase):
    bra = None
    checked = 0
    setUp = functions.BackendTests.setUp
    program = functions.BackendTests.program
    execute = functions.BackendTests.execute

    def check_raw_copy(self, source, raw):
        lines = self.program(source, (('origin', False), ('out', False)))
        program = self.module.openqasm3.parser.parse('OPENQASM 3;')
        converter = self.module.QASM2QCXConverter(program)
        origin = converter._QASM2QCXConverter__capitalize_variable_name('origin')
        declaration = lines.index(f'VAR {origin} UINT')
        lines.insert(declaration + 1, f'LET {origin} := {raw}')
        result = self.execute(lines)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.split(), [str(raw), str(raw)])
        type(self).checked += 1

    def test_explicit_width_storage_and_copies_preserve_exact_bits(self):
        native = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        for width in sorted({1, 4, 8, native}):
            mask = (1 << width) - 1
            for raw in sorted({0, 1, mask, mask // 3}):
                with self.subTest(width=width, raw=raw):
                    self.check_raw_copy(f'angle[{width}] origin; angle[{width}] out = origin; out = origin;', raw)

    def test_default_width_matches_native_uint_storage(self):
        native = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        for raw in (0, 1, (1 << native) - 1):
            self.check_raw_copy(f'angle origin; angle[{native}] out = origin;', raw)

    def test_width_constant_shadowing_does_not_change_copy_representation(self):
        self.check_raw_copy('''const int WIDTH = 4; angle[WIDTH] origin; angle[WIDTH] out;
            for int WIDTH in [0:1] { out = origin; }''', 13)

    @staticmethod
    def reference(value, width):
        scaled = Fraction(value) / Fraction(math.tau) * (1 << width)
        quotient, remainder = divmod(scaled.numerator, scaled.denominator)
        if remainder * 2 > scaled.denominator or (remainder * 2 == scaled.denominator and quotient % 2):
            quotient += 1
        return quotient % (1 << width)

    def check_exact(self, source, outputs, expected):
        result = self.execute(self.program(source, tuple((name, False) for name in outputs)))
        self.assertEqual(result.returncode, 0, result.stderr + '\n' + source)
        self.assertEqual(result.stdout.split(), [str(value) for value in expected], source)
        type(self).checked += 1

    def test_constant_literal_and_runtime_float_conversions(self):
        native = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        for width in sorted({1, 2, 4, 8, native}):
            tie = math.ldexp(math.tau, -width - 1)
            values = (0.0, -0.0, math.pi, -math.pi, math.tau, -math.tau, 0.1, -0.1,
                      1e308, -1e308, tie, -tie,
                      math.nextafter(tie, 0.0), math.nextafter(tie, math.inf),
                      math.nextafter(-tie, 0.0), math.nextafter(-tie, -math.inf),
                      math.nextafter(math.tau, 0.0), math.nextafter(math.tau, math.inf))
            for value in values:
                expected = self.reference(value, width)
                self.check_exact(f'''const angle[{width}] CONSTANT = {value!r};
                    angle[{width}] folded = CONSTANT; float value = {value!r};
                    angle[{width}] live = value;
                    angle[{width}] direct = angle[{width}]({value!r});
                    bool nonzero = bool(live);''',
                                 ('folded', 'live', 'direct', 'nonzero'),
                                 (expected, expected, expected, int(expected != 0)))

    def test_constant_and_runtime_angle_width_conversion(self):
        native = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        value = math.tau * 45 / 256
        for source in sorted({1, 4, 8, native}):
            for target in sorted({1, 4, 8, native}):
                bits = self.reference(value, source)
                expected = bits << (target - source) if target >= source else bits >> (source - target)
                self.check_exact(f'''const angle[{source}] SOURCE = {value!r};
                    const angle[{target}] TARGET = SOURCE;
                    angle[{source}] original = SOURCE;
                    angle[{target}] folded = TARGET; angle[{target}] live = original;
                    live = angle[{target}](original);
                    bool nonzero = bool(live);''',
                                 ('original', 'folded', 'live', 'nonzero'),
                                 (bits, expected, expected, int(expected != 0)))

    def test_nonfinite_runtime_failure_identifies_native_conversion(self):
        for source in ('float huge = 1e308; float bad = huge * huge; angle[4] out = bad;',
                       'float huge = 1e308; float bad = -huge * huge; angle[4] out = angle[4](bad);',
                       'angle[4] out = angle[4](1e309);', 'angle[4] out = -1e309;'):
            result = self.execute(self.program(source))
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('ANGLEBITS', result.stderr)
            self.assertIn('nonfinite', result.stderr)
            self.assertEqual(result.stdout, '')
            type(self).checked += 1

    def test_short_circuiting_skips_runtime_conversion_failures(self):
        self.check_exact('''float huge = 1e308; float bad = huge * huge; int zero = 0;
            bool off = false && bool(angle[4](bad));
            bool on = true || bool(angle[4](1.0 / zero));
            bool literal = false && bool(angle[4](1e309));
            const bool SKIP = false && bool(angle[4](1e309));
            bool skipped = SKIP;''', ('off', 'on', 'literal', 'skipped'), (0, 1, 0, 0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    BackendTests.bra = parser.parse_args().bra.resolve()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__)))
    print(f'angle programs checked: {BackendTests.checked}')
    raise SystemExit(not result.wasSuccessful())


if __name__ == '__main__':
    main()
