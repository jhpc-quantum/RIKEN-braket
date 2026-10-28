#!/usr/bin/env python3
"""Audit one-dimensional angle-array storage (#418).

Pass --bra /path/to/bra for the default-double non-MPI backend audit.
Inspect exact UINT storage independently of source-level element reads.
"""

import argparse
import cmath
from fractions import Fraction
import math
import pathlib
import re
import resource
import unittest

import qasm2qcx_math_function_numerical as functions


class BackendTests(unittest.TestCase):
    bra = None
    checked = 0
    setUp = functions.BackendTests.setUp
    execute = functions.BackendTests.execute
    program = functions.BackendTests.program
    check = functions.BackendTests.check

    def check_initializers(self, source, arrays, expected, scalars=(), scalar_expected=()):
        program = self.module.openqasm3.parser.parse('OPENQASM 3; ' + source)
        converter = self.module.QASM2QCXConverter(program)
        converter.visit(program)
        self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())
        lines = list(converter)
        for name, length in arrays:
            operand = converter._QASM2QCXConverter__capitalize_variable_name(name)
            for index in range(length):
                lines.append(f'PRINTLN {operand}:{index}')
        for name in scalars:
            operand = converter._QASM2QCXConverter__capitalize_variable_name(name)
            lines.append(f'PRINTLN {operand}')
        result = self.execute(lines)
        self.assertEqual(result.returncode, 0, result.stderr + '\n' + '\n'.join(lines))
        self.assertNotIn('runtime error:', result.stderr)
        self.assertEqual(result.stdout.split(), [str(value) for value in (*expected, *scalar_expected)])
        type(self).checked += 1

    def test_exact_initializers_at_every_width_preserve_sources(self):
        for width in range(1, self.module.QASM2QCXConverter.QCX_UINT_WIDTH + 1):
            modulus = 1 << width
            for bits in sorted({0, 1, modulus // 2, modulus - 1, 0xAAAAAAAA & (modulus - 1)}):
                self.check_initializers(f'const angle[{width}] SEED = angle[{width}]("{bits:0{width}b}"); '
                                        f'angle[{width}] live = SEED; '
                                        f'array[angle[{width}], 4] a = {{SEED, angle[{width}]("{bits:0{width}b}"), live, -live}};',
                                        (('a', 4),), (bits, bits, bits, -bits % modulus), ('live',), (bits,))

    def test_angle_width_conversion_and_unsized_elements(self):
        for width in (1, 4, 8, self.module.QASM2QCXConverter.QCX_UINT_WIDTH):
            expected = [bits << (width - 4) if width >= 4 else bits >> (4 - width) for bits in (3, 13)]
            self.check_initializers('const angle[4] SEED = angle[4]("0011"); angle[4] live = -SEED; '
                                    f'array[angle[{width}], 2] a = {{SEED, live}};', (('a', 2),), expected,
                                    ('live',), (13,))
        width = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        self.check_initializers('const angle[4] SEED = angle[4]("0011"); array[angle, 1] a = {SEED};',
                                (('a', 1),), (3 << (width - 4),))

    def test_float_inputs_quantize_with_scalar_angle_rules(self):
        for width in (1, 4, 8, self.module.QASM2QCXConverter.QCX_UINT_WIDTH):
            tie = math.ldexp(math.tau, -width - 1)
            for value in (0.0, -0.0, math.pi, -math.pi, math.tau, 0.1, -0.1, 1e308,
                          tie, -tie, math.nextafter(tie, 0), math.nextafter(tie, math.inf)):
                expected = round(Fraction(value) / Fraction(math.tau) * (1 << width)) % (1 << width)
                self.check_initializers(f'float x = {value!r}; array[angle[{width}], 1] a = {{x}};',
                                        (('a', 1),), (expected,))

    def test_initializer_order_and_dimension_constant_shadowing(self):
        self.check_initializers('const int W = 4; const int N = 2; '
                                'angle[4] seed = angle[4]("0011"); '
                                'array[angle[W], N] a = {seed, seed + angle[4]("0001")}; '
                                'for int W in [0:1] {} for int N in [0:1] {}',
                                (('a', 2),), (3, 4), ('seed',), (3,))
        self.check_initializers('array[float, 2] radians = {0.0, pi}; int i = 1; '
                                'array[angle[4], 2] a = {radians[i], radians[0]};',
                                (('a', 2),), (8, 0), ('i',), (1,))

    def test_reads_and_writes_at_every_width(self):
        for width in range(1, self.module.QASM2QCXConverter.QCX_UINT_WIDTH + 1):
            modulus = 1 << width
            for bits in sorted({0, 1, modulus // 2, modulus - 1}):
                negative = -bits % modulus
                self.check_initializers(f'const angle[{width}] A = angle[{width}]("{bits:0{width}b}"); '
                                        f'array[angle[{width}], 2] a = {{A, -A}}; int i = -1; '
                                        f'angle[{width}] copied = a[i]; a[0] = copied; a[i] = A;',
                                        (('a', 2),), (negative, bits), ('copied', 'i'), (negative, -1))

    def test_compound_assignments_preserve_promoted_width_until_final_write(self):
        native = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        for width in range(1, native + 1):
            mask = (1 << width) - 1
            bits = ((1 << (width - 1)) | 1) & mask
            wider = min(width + 1, native)
            difference = wider - width
            common_mask = (1 << wider) - 1
            lhs, rhs = bits << difference, 1
            for operator in ('+=', '-=', '&=', '|=', '^=', '*=', '/=', '<<=', '>>='):
                if operator in ('*=', '/='):
                    factor = (3 if operator == '*=' else 2) if width >= 2 else 1
                    operand = f'uint[{width}]({factor})'
                    expected = (bits * factor if operator == '*=' else bits // factor) & mask
                elif operator in ('<<=', '>>='):
                    count = width + 1 if operator == '<<=' else 1
                    operand = str(count)
                    expected = (bits << count if operator == '<<=' else bits >> count) & mask
                else:
                    operand = f'angle[{wider}]("{rhs:0{wider}b}")'
                    result = {'+=': lhs + rhs, '-=': lhs - rhs, '&=': lhs & rhs,
                              '|=': lhs | rhs, '^=': lhs ^ rhs}[operator]
                    expected = (result & common_mask) >> difference
                self.check_initializers(f'array[angle[{width}], 2] a = {{angle[{width}]("{bits:0{width}b}"), '
                                        f'angle[{width}]("{bits:0{width}b}")}}; int i = -1; '
                                        f'a[i] {operator} {operand};', (('a', 2),), (bits, expected))

    def test_singleton_arrays_self_references_and_element_conversions(self):
        self.check_initializers('array[angle[4], 1] a = {angle[4]("0011")}; int i = -1; '
                                'a[i] += a[0]; angle[8] wide = a[i]; '
                                'array[angle[8], 1] b = {wide}; b[0] -= angle[8]("00000001"); '
                                'a[0] = b[0];', (('a', 1), ('b', 1)), (5, 95), ('wide',), (96,))
        self.check_initializers('array[angle[4], 2] a = {angle[4]("0111"), angle[4]("1001")}; '
                                'int i = 0; int j = 1; a[i] = a[j]; '
                                'a[j] = a[i] + angle[4]("0001");', (('a', 2),), (9, 10))

    def test_runtime_bounds_checks_and_skipped_access(self):
        for index in (-3, 2, 2147483647):
            self.check_initializers(f'array[angle[4], 2] a = {{pi, pi / 2}}; int i = {index}; '
                                    'if (false) { a[i] += angle[4](pi / 8); }',
                                    (('a', 2),), (8, 4))
            for statement in ('angle[4] out = a[i];', 'a[i] = pi;', 'a[i] += angle[4](pi / 8);'):
                result = self.execute(self.program(f'array[angle[4], 2] a = {{pi, pi}}; int i = {index}; ' + statement))
                self.assertNotEqual(result.returncode, 0, (index, statement))
                self.assertNotIn('runtime error:', result.stderr)
                type(self).checked += 1

    def test_bit_register_casts_rotations_and_population_counts(self):
        self.check_initializers('array[angle[4], 2] a = {angle[4]("0011"), angle[4]("1001")}; int i = -1; '
                                'bit[4] bits = bit[4](a[i]); '
                                'array[angle[8], 2] wide = {angle[8](angle[4](bits)), -a[0]}; '
                                'a[0] = angle[4](bits); uint count = popcount(a[i]); '
                                'a[i] = rotl(a[i], -1); a[0] = rotr(a[0], 1);',
                                (('a', 2), ('wide', 2), ('bits', 4)), (12, 12, 144, 208, 1, 0, 0, 1), ('count',), (2,))
        self.check_initializers('array[angle[4], 2] a = {angle[4]("0011"), angle[4]("1100")}; '
                                'a[popcount(a[0]) - uint(1)] ^= a[0]; '
                                'uint ratio = a[1] / a[0]; bool ordered = a[1] > a[0];',
                                (('a', 2),), (3, 15), ('ratio', 'ordered'), (5, 1))

    def test_trigonometric_calls_nested_functions_and_float_destinations(self):
        from qasm2qcx_angle_trigonometric_numerical import reference
        sine, cosine, tangent = (reference(name, 3, 4) for name in ('sin', 'cos', 'tan'))
        self.check('array[angle[4], 1] a = {angle[4]("0011")}; int i = -1; '
                   'array[float, 3] out = {sin(a[i]), cos(a[0]), tan(a[i])}; '
                   'out[0] += cos(sin(a[i])); '
                   'float x = out[0]; float y = out[1]; float z = out[2]; '
                   'a[0] = arccos(0.0); float exact = sin(a[i]);',
                   (('x', False), ('y', False), ('z', False), ('exact', False)),
                   (sine + math.cos(sine), cosine, tangent, 1))

    def test_conditions_loops_and_dimension_constant_shadowing(self):
        self.check_initializers('const int W = 4; const int N = 2; '
                                'array[angle[W], N] a = {angle[4]("0000"), pi}; int count = 0; '
                                'while (a[0] < a[1]) { a[0] += angle[4]("0001"); '
                                'if (a[0] == angle[4]("0011")) { continue; } count += 1; } '
                                'for int W in [0:1] { a[W] ^= angle[4]("0001"); } '
                                'for int N in [0:1] { a[N] <<= 1; }',
                                (('a', 2),), (2, 2), ('count',), (7,))
        self.check_initializers('array[angle[4], 2] a = {angle[4]("0011"), angle[4]("0101")}; '
                                'int start = 0; int stop = 1; uint total = 0; '
                                'for int i in [start:stop] { total += popcount(a[i]); '
                                'for int j in {i, 1 - i} { a[j] += angle[4]("0001"); } }',
                                (('a', 2),), (5, 7), ('total',), (4,))

    def test_array_names_shadowed_by_scalar_iterators(self):
        self.check_initializers('array[angle[4], 1] a = {pi}; array[float, 2] values = {0.0, pi / 2}; '
                                'angle[4] out; uint total = 0; '
                                'for float a in values { out = angle[4](a); total += popcount(out); } '
                                'out = a[0];', (('a', 1),), (8,), ('out', 'total'), (8, 1))

    def test_skipped_body_domains_and_reached_operand_errors(self):
        for statement in ('a[i] /= zero;', 'a[i] <<= negative;',
                          'a[i] = angle[4](1e309);', 'a[i] = tan(a[i]);'):
            for wrapper in ('if (flag) {{ {} }}', 'while (flag) {{ {} }}',
                            'for int j in [first:last] {{ {} }}', 'for int j in [1:0] {{ {} }}'):
                self.check_initializers('array[angle[4], 1] a = {pi / 2}; int i = 0; uint[4] zero = 0; '
                                        'int negative = -1; bool flag = false; int first = 2; int last = 1; '
                                        + wrapper.format(statement), (('a', 1),), (4,))
            result = self.execute(self.program('array[angle[4], 1] a = {pi / 2}; int i = 0; '
                                               'uint[4] zero = 0; int negative = -1; ' + statement))
            self.assertNotEqual(result.returncode, 0, statement)
            self.assertNotIn('runtime error:', result.stderr)
            type(self).checked += 1
        self.check_initializers('array[angle[4], 1] a = {pi / 2}; int bad = 2; uint[4] zero = 0; '
                                'bool x = false && bool(a[bad]); bool y = true || bool(a[0] / zero);',
                                (('a', 1),), (4,), ('x', 'y'), (0, 1))

    def test_phase_gates_modifiers_and_fusion(self):
        from qasm2qcx_u_power_numerical import BackendAuditTests
        phase = 3 * math.pi / 4
        source = ('include "stdgates.inc"; qubit[2] q; array[angle[4], 2] phases = '
                  '{angle[4]("0110"), angle[4]("1000")}; int i = 0; h q; ')
        for operation, expected in (
                ('p(phases[i]) q[0];', tuple(cmath.exp(1j * phase) / 2 if index & 1 else 0.5 for index in range(4))),
                ('pow(0.5) @ p(phases[i]) q[0];', tuple(cmath.exp(0.5j * phase) / 2 if index & 1 else 0.5 for index in range(4))),
                ('pow(popcount(phases[i])) @ p(phases[i]) q[0];', tuple(cmath.exp(2j * phase) / 2 if index & 1 else 0.5 for index in range(4))),
                ('ctrl @ p(phases[i]) q[0], q[1];', (0.5, 0.5, 0.5, cmath.exp(1j * phase) / 2)),
                ('negctrl @ gphase(phases[i]) q[0];', tuple(0.5 if index & 1 else cmath.exp(1j * phase) / 2 for index in range(4)))):
            for fused in (False, True):
                # Later mutation must not alter parameters captured by gates
                # queued in a fusion block.
                lines = self.program(source + operation + 'phases[i] = phases[1];', fused=fused)
                BackendAuditTests.check(self, lines, expected)
        for fused in (False, True):
            lines = self.program('include "stdgates.inc"; qubit q; array[angle[4], 1] phases = {pi / 2}; '
                                 'int i = 0; rx(sin(phases[i])) q;', fused=fused)
            BackendAuditTests.check(self, lines, (math.cos(0.5), -1j * math.sin(0.5)))

    def test_documented_angle_array_example(self):
        from qasm2qcx_u_power_numerical import BackendAuditTests
        document = (pathlib.Path(__file__).parents[1] / 'qcx/qasm2qcx.md').read_text(encoding='utf-8')
        examples = re.findall(r'<!-- angle-array-example -->\s*```qasm\n(.*?)\n```', document, flags=re.DOTALL)
        self.assertEqual(len(examples), 1)
        source = examples[0].removeprefix('OPENQASM 3.0;')
        self.check_initializers(source, (('phases', 3), ('bits', 4)), (0, 6, 6, 0, 1, 1, 0))
        expected = (1 / math.sqrt(2), cmath.exp(3j * math.pi / 4) / math.sqrt(2))
        for fused in (False, True):
            BackendAuditTests.check(self, self.program(examples[0], fused=fused), expected)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra', type=pathlib.Path)
    arguments = parser.parse_args()
    BackendTests.bra = arguments.bra.resolve() if arguments.bra else None
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(BackendTests))
    print(f'angle-array programs checked: {BackendTests.checked}')
    raise SystemExit(not result.wasSuccessful())
