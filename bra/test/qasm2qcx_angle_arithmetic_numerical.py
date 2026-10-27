#!/usr/bin/env python3
"""Check exact scalar angle arithmetic and resulting phase gates (#398).

  python3 bra/test/qasm2qcx_angle_arithmetic_numerical.py --bra /path/to/bra

Raw UINT fixtures isolate arithmetic from floating-point angle quantization.
Expected results use independent integer alignment and modular arithmetic.
"""

import argparse
import cmath
import math
import pathlib
import re
import resource
import unittest

import qasm2qcx_angle_global_phase_numerical as angles


class BackendTests(unittest.TestCase):
    bra = None
    checked = 0
    setUp = angles.BackendTests.setUp
    program = angles.BackendTests.program
    execute = angles.BackendTests.execute
    check_phase = angles.BackendTests.check_phase
    check_zero_qubit = angles.BackendTests.check_zero_qubit

    def exact(self, source, outputs, expected, *, raw=()):
        lines = self.program(source, tuple((name, False) for name in outputs))
        # The name mapping is used only to seed declared storage, never to
        # calculate expected values or reproduce converter arithmetic.
        empty = self.module.QASM2QCXConverter(self.module.openqasm3.parser.parse('OPENQASM 3;'))
        for name, bits in raw:
            operand = empty._QASM2QCXConverter__capitalize_variable_name(name)
            declaration = lines.index(f'VAR {operand} UINT')
            lines.insert(declaration + 1, f'LET {operand} := {bits}')
        result = self.execute(lines)
        self.assertEqual(result.returncode, 0, result.stderr + '\n' + source)
        self.assertEqual(result.stdout.split(), list(map(str, expected)), source)
        type(self).checked += 1

    def test_exact_runtime_results_at_every_width(self):
        for width in range(1, self.module.QASM2QCXConverter.QCX_UINT_WIDTH + 1):
            modulus = 1 << width
            values = sorted({0, 1, modulus // 2 - 1, modulus // 2, modulus - 1})
            for lhs in values:
                for rhs in values:
                    source = f'''angle[{width}] a; angle[{width}] b;
                        angle[{width}] sum = a + b; angle[{width}] difference = a - b;
                        angle[{width}] negative = -a; angle[{width}] nested = -(a - (b + a));
                        angle[{width}] self = a; self = self + b; self = self - b;
                        angle[{width}] plus = a; plus += b; angle[{width}] minus = a; minus -= b;
                        angle[{width}] double = a; double += double;
                        angle[{width}] zero_angle = b; zero_angle -= zero_angle;
                        angle[{width}] alias = a; alias += b - alias;
                        bool zero = bool(a - a);'''
                    self.exact(source, ('a', 'b', 'sum', 'difference', 'negative', 'nested', 'self',
                                        'plus', 'minus', 'double', 'zero_angle', 'alias', 'zero'),
                               (lhs, rhs, (lhs + rhs) % modulus, (lhs - rhs) % modulus,
                                (-lhs) % modulus, rhs, lhs, (lhs + rhs) % modulus,
                                (lhs - rhs) % modulus, (2 * lhs) % modulus, 0, rhs, 0), raw=(('a', lhs), ('b', rhs)))

    def test_mixed_width_promotion_and_destination_narrowing(self):
        native = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        for left_width in sorted({1, 4, 8, native}):
            for right_width in sorted({1, 4, 8, native}):
                width, small = max(left_width, right_width), min(left_width, right_width)
                for lhs, rhs in ((0, 1), (1, 1), ((1 << left_width) - 1, 1),
                                  (1, (1 << right_width) - 1)):
                    left = lhs << (width - left_width)
                    right = rhs << (width - right_width)
                    total = (left + right) % (1 << width)
                    difference = (left - right) % (1 << width)
                    source = f'''angle[{left_width}] a; angle[{right_width}] b;
                        angle[{width}] sum = a + b; angle[{width}] difference = a - b;
                        angle[{small}] narrow_sum = a + b; angle[{small}] narrow_difference = a - b;
                        angle[{width}] negative_sum = -(a + b);
                        angle[{left_width}] plus = a; plus += b;
                        angle[{left_width}] minus = a; minus -= b;
                        angle[{left_width}] alias_plus = a; alias_plus += b - alias_plus;
                        angle[{left_width}] alias_minus = a; alias_minus -= b - alias_minus;'''
                    self.exact(source, ('a', 'b', 'sum', 'difference', 'narrow_sum', 'narrow_difference',
                                        'negative_sum', 'plus', 'minus', 'alias_plus', 'alias_minus'),
                               (lhs, rhs, total, difference, total >> (width - small), difference >> (width - small),
                                (-total) % (1 << width), total >> (width - left_width), difference >> (width - left_width),
                                right >> (width - left_width), ((2 * left - right) % (1 << width)) >> (width - left_width)),
                               raw=(('a', lhs), ('b', rhs)))

    def test_repeated_compound_assignments_and_classical_control_flow(self):
        declaration = 'const angle[4] START = pi / 8; const angle[8] EPSILON = pi / 128; angle[4] a = START; angle[8] fine = EPSILON; '
        for body in ('int count = 0; while (count < 3) { a -= fine; count += 1; }',
                     'int stop = 2; for int i in [0:stop] { a -= fine; }'):
            self.exact(declaration + body, ('a', 'fine'), (14, 1))
        self.exact(declaration + 'for int i in {0, 1} { a -= fine; }', ('a', 'fine'), (15, 1))
        self.exact(declaration + '''array[float, 2] steps = {pi / 128, pi / 128};
            for float step in steps { a -= angle[8](step); }''', ('a', 'fine'), (15, 1))
        self.exact(declaration + '''int count = 0; while (count < 4) { count += 1;
            if (count == 2) { continue; } a -= fine; if (count == 3) { break; } }''',
                   ('a', 'fine', 'count'), (15, 1, 3))
        self.exact(declaration + '''int count = 0; while (count < 2) {
            if (count == 0) { a -= fine; } else { a += fine; } count += 1; }''', ('a', 'fine'), (0, 1))
        self.exact(declaration + '''array[float, 2] steps = {pi / 128, pi / 128};
            for float START in steps { a -= angle[8](START); } a += START;''', ('a', 'fine'), (0, 1))
        for preparation, measured, expected in (('', 0, 1), ('x q;', 1, 0)):
            self.exact('include "stdgates.inc"; qubit q; ' + preparation + 'bit flag = measure q; ' + declaration
                       + 'if (flag) { a -= fine; } else { a += fine; }',
                       ('a', 'fine', 'flag'), (expected, 1, measured))

    def test_compound_updates_feed_phase_gates_without_losing_global_phase(self):
        declaration = 'const angle[4] A = pi; const angle[8] B = pi / 128; angle[4] a = A; angle[8] fine = B; a -= fine; '
        for gate in ('p', 'phase', 'u1', 'cp', 'cphase', 'gphase'):
            wires = (2, 0) if gate in ('cp', 'cphase') else () if gate == 'gphase' else (1,)
            operands = ', '.join(f'q[{wire}]' for wire in wires)
            for modifier, width, bits in (('', 4, 7), ('inv @ ', 4, 9), ('pow(0.5) @ ', 5, 7)):
                for fused in (False, True):
                    self.check_phase(declaration + f'{modifier}{gate}(a) {operands};', width, bits, (wires,),
                                     fused=fused, preserved=(('a', 7), ('fine', 1)))
        for fused in (False, True):
            self.check_phase(declaration + 'negctrl @ ctrl @ pow(0.5) @ cp(a) q[3], q[2], q[1], q[0];',
                             5, 7, (((3, 0), 2, 1, 0),), fused=fused, qubits=4)
            self.check_zero_qubit(declaration + 'pow(0.5) @ gphase(a);', 5, 7,
                                  fused=fused, preserved=(('a', 7), ('fine', 1)))

    def test_constants_unsized_angles_and_executable_array_casts(self):
        native = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        self.exact('''const angle[4] A = 15 * pi / 8; angle[4] a = A;
            const angle[8] B = pi / 128; angle[4] result = a - B;
            angle[8] widened = a + B; angle[4] negative = -a;''',
                   ('a', 'result', 'widened', 'negative'), (15, 14, 241, 1))
        self.exact('''angle a; const angle[4] B = pi / 8;
            angle sum = a + B; angle difference = a - B; angle negative = -a;''',
                   ('a', 'sum', 'difference', 'negative'),
                   (1, 1 + (1 << (native - 4)), (1 - (1 << (native - 4))) % (1 << native), (1 << native) - 1),
                   raw=(('a', 1),))
        self.exact('''array[float, 2] data = {pi, pi / 2}; int index = 1;
            angle[4] a; angle[8] sum = a + angle[8](data[index]);
            angle[4] difference = angle[8](data[index]) - a;
            angle[8] negative = -(angle[4](data[index]) + angle[8](data[index - 1]));''',
                   ('a', 'sum', 'difference', 'negative'), (15, 48, 5, 64), raw=(('a', 15),))

    def test_computed_phase_parameters_preserve_full_matrices_and_sources(self):
        declaration = 'const angle[4] A = pi; const angle[8] B = pi / 128; angle[4] a = A; angle[8] b = B; '
        for gate in ('p', 'phase', 'u1', 'cp', 'cphase', 'gphase'):
            wires = (2, 0) if gate in ('cp', 'cphase') else () if gate == 'gphase' else (1,)
            operands = ', '.join(f'q[{wire}]' for wire in wires)
            for modifier, width, bits in (('', 8, 127), ('inv @ ', 8, 129), ('pow(0.5) @ ', 9, 127)):
                for fused in (False, True):
                    self.check_phase(declaration + f'{modifier}{gate}(-(a + b)) {operands};',
                                     width, bits, (wires,), fused=fused, preserved=(('a', 8), ('b', 1)))
        for fused in (False, True):
            self.check_phase(declaration + 'negctrl @ pow(0.5) @ p(-(a + b)) q[2], q[0:1];',
                             9, 127, (((2, 0), 0), ((2, 0), 1)), fused=fused)
            self.check_zero_qubit(declaration + 'pow(0.5) @ gphase(-(a + b));', 9, 127,
                                  fused=fused, preserved=(('a', 8), ('b', 1)))

    def test_skipped_calls_and_reached_conversion_and_access_errors(self):
        for wrapper in ('if (flag) {{ {} }}', 'for int i in [1:0] {{ {} }}', 'while (flag) {{ {} }}'):
            self.exact('angle[4] a; angle[8] out; bool flag = false; array[float, 1] data = {0}; int index = 2; '
                       + wrapper.format('out = -(a + angle[8](1e309)); out = angle[8](data[index]) - a; '
                                        'a += angle[8](1e309); a -= angle[8](data[index]);'),
                       ('a', 'out'), (5, 0), raw=(('a', 5),))
        for expression in ('a + angle[8](1e309)', 'angle[8](data[index]) - a', '-angle[8](1e309)'):
            source = 'angle[4] a; angle[8] out; array[float, 1] data = {0}; int index = 2; '
            result = self.execute(self.program(source + f'out = {expression};') + ['DO AMPLITUDES'])
            self.assertNotEqual(result.returncode, 0, expression)
            self.assertNotIn(' => ', result.stdout)
            type(self).checked += 1
        for operator in ('+=', '-='):
            for expression in ('angle[8](1e309)', 'angle[8](data[index])'):
                source = 'angle[4] a; array[float, 1] data = {0}; int index = 2; '
                result = self.execute(self.program(source + f'a {operator} {expression};') + ['DO AMPLITUDES'])
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn(' => ', result.stdout)
                type(self).checked += 1
        self.exact('''angle[4] a; bool flag = false;
            bool and_result = flag && bool(a + angle[8](1e309));
            bool or_result = !flag || bool(-angle[8](1e309));''',
                   ('and_result', 'or_result'), (0, 1))

    def test_documented_arithmetic_examples(self):
        document = (pathlib.Path(__file__).parents[1] / 'qcx' / 'qasm2qcx.md').read_text()
        examples = re.findall(r'<!-- angle-arithmetic-example -->\s*```qasm\n(.*?)\n```', document, flags=re.DOTALL)
        self.assertEqual(len(examples), 1)
        self.exact(examples[0], ('wrapped', 'a', 'b', 'difference', 'negative', 'nonzero'), (0, 0, 17, 0, 15, 1))
        examples = re.findall(r'<!-- angle-arithmetic-phase-example -->\s*```qasm\n(.*?)\n```', document, flags=re.DOTALL)
        self.assertEqual(len(examples), 1)
        for fused in (False, True):
            lines = self.program(examples[0])
            if fused:
                lines.insert(1, 'BEGIN FUSION')
                lines.insert(len(lines) - 1, 'END FUSION')  # Flush before pragma-requested amplitudes.
            result = self.execute(lines)
            self.assertEqual(result.returncode, 0, result.stderr)
            actual = {int(index, 2): complex(float(real), float(imaginary))
                      for index, real, imaginary in re.findall(r'^([01]+) => (\S+) \+ (\S+) i$', result.stdout, re.MULTILINE)}
            self.assertEqual(set(actual), set(range(8)), result.stdout)
            for index in range(8):
                phase = (7 * math.pi / 8 if index & 1 else 0) - (7 * math.pi / 8 if index & 2 else 0)
                phase -= math.pi / 16 if not index & 4 else 0
                self.assertTrue(cmath.isclose(actual[index], cmath.exp(1j * phase) / math.sqrt(8), rel_tol=1e-5, abs_tol=2e-6))
            type(self).checked += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    BackendTests.bra = parser.parse_args().bra.resolve()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(BackendTests))
    print(f'angle arithmetic programs checked: {BackendTests.checked}')
    raise SystemExit(not result.wasSuccessful())


if __name__ == '__main__':
    main()
