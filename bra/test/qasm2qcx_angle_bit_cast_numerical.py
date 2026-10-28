#!/usr/bin/env python3
"""Audit exact angle/bit-register casts (#412).

Run with --bra /path/to/bra for a non-MPI backend audit. References use
integer patterns and individual little-endian bits, without floating point.
"""

import argparse
import cmath
import math
import pathlib
import random
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

    def check_exact(self, source, outputs, expected):
        program = self.module.openqasm3.parser.parse('OPENQASM 3; ' + source)
        converter = self.module.QASM2QCXConverter(program)
        converter.visit(program)
        self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())
        lines = list(converter)
        for name, width in outputs:
            operand = converter._QASM2QCXConverter__capitalize_variable_name(name)
            if width is None:
                lines.append(f'PRINTLN {operand}')
            else:
                for index in range(width):
                    bit = converter._QASM2QCXConverter__qcx_bit_name(operand, width, index)
                    lines.append(f'PRINTLN {bit}')
        result = self.execute(lines)
        self.assertEqual(result.returncode, 0, result.stderr + '\n' + '\n'.join(lines))
        self.assertNotIn('runtime error:', result.stderr)
        actual = list(map(int, result.stdout.split()))
        position, values = 0, []
        for _, width in outputs:
            if width is None:
                values.append(actual[position])
                position += 1
            else:
                bits = actual[position:position + width]
                self.assertEqual(len(bits), width)
                self.assertTrue(all(bit in (0, 1) for bit in bits))
                values.append(sum(bit << index for index, bit in enumerate(bits)))
                position += width
        self.assertEqual(position, len(actual))
        self.assertEqual(values, list(expected), source)
        type(self).checked += 1

    def test_all_widths_exact_round_trips_and_source_preservation(self):
        random_source = random.Random(412)
        for width in range(1, self.module.QASM2QCXConverter.QCX_UINT_WIDTH + 1):
            modulus = 1 << width
            patterns = (range(modulus) if width <= 6 else
                        sorted({0, 1, modulus - 1, modulus // 2, modulus // 2 - 1}
                               | {random_source.randrange(modulus) for _ in range(4)}))
            for bits in patterns:
                self.check_exact(f'bit[{width}] bits = "{bits:0{width}b}"; '
                                 f'angle[{width}] a = angle[{width}](bits); '
                                 f'bit[{width}] back = bit[{width}](a); '
                                 f'angle[{width}] out = angle[{width}](back); '
                                 f'bit[{width}] nested = bit[{width}](angle[{width}](bits));',
                                 (('bits', width), ('a', None), ('back', width), ('out', None), ('nested', width)),
                                 (bits,) * 5)

    def test_executable_literals_and_named_constants(self):
        for width in (1, 4, self.module.QASM2QCXConverter.QCX_UINT_WIDTH):
            bits = (1 << width) - 1
            self.check_exact(f'const angle[{width}] A = angle[{width}]("{bits:0{width}b}"); '
                             f'const bit[{width}] B = bit[{width}](A); '
                             f'angle[{width}] a = angle[{width}](B); '
                             f'angle[{width}] literal = angle[{width}]("{bits:0{width}b}"); '
                             f'bit[{width}] b = bit[{width}](A);',
                             (('a', None), ('literal', None), ('b', width)), (bits,) * 3)
        width = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        self.check_exact(f'bit[{width}] bits = "1{"0" * (width - 1)}"; '
                         f'angle a = angle(bits); bit[{width}] back = bit[{width}](a);',
                         (('a', None), ('back', width)), (1 << (width - 1),) * 2)

    def test_overlapping_destinations_and_nested_temporary_ownership(self):
        self.check_exact('bit[4] bits = "1101"; '
                         'bits = bit[4](angle[4](bits[{3, 2, 1, 0}])); '
                         'angle[4] a = angle[4](bits); '
                         'a = angle[4](bit[4](a) ^ "0110"); '
                         'bit[4] back = bit[4](a); '
                         'back[{3, 2, 1, 0}] = bit[4](angle[4](back));',
                         (('bits', 4), ('a', None), ('back', 4)), (11, 13, 11))

    def test_expressions_runtime_rotations_shifts_and_static_selections(self):
        bits = 0b10110101
        for distance in (0, 1, 3, 8, 9, -1):
            count = distance % 8
            rotated = ((bits << count) | (bits >> (8 - count))) & 255
            self.check_exact(f'bit[8] b = "{bits:08b}"; int count = {distance}; '
                             'angle[8] a = angle[8](rotl(b, count)); '
                             'bit[8] back = bit[8](a + angle[8]("00000001")); '
                             'angle[8] out = angle[8](back ^ "01010101");',
                             (('b', 8), ('a', None), ('back', 8), ('out', None)),
                             (bits, rotated, (rotated + 1) & 255, ((rotated + 1) & 255) ^ 0x55))
        for distance in (0, 1, 3, 8, 9):
            shifted = (bits << distance) & 255
            self.check_exact(f'bit[8] b = "{bits:08b}"; uint count = {distance}; '
                             'angle[8] a = angle[8](b << count); '
                             'bit[8] back = bit[8](a);',
                             (('b', 8), ('a', None), ('back', 8)), (bits, shifted, shifted))
        self.check_exact('bit[8] b = "11010110"; '
                         'angle[4] a = angle[4](b[{7, 0, 5, 1}]); '
                         'angle[4] repeated = angle[4](b[{0, 0, 1, 1}]); '
                         'b[0:3] = bit[4](a); '
                         'angle[4] slice = angle[4](b[0:3]);',
                         (('a', None), ('repeated', None), ('b', 8), ('slice', None)), (9, 12, 0xD9, 9))

    def test_conditions_runtime_loops_and_loop_bounds(self):
        self.check_exact('angle[4] a = angle[4]("0000"); int count = 0; '
                         'while (bit[4](a) != "1000") { '
                         'a += angle[4]("0001"); '
                         'if (bit[4](a) == "0011") { continue; } count += 1; '
                         'if (bit[4](a) == "1111") { break; } } '
                         'bit[4] out = bit[4](a);',
                         (('a', None), ('count', None), ('out', 4)), (8, 7, 8))
        self.check_exact('angle[4] a = angle[4]("0111"); int total = 0; '
                         'int last = int(popcount(bit[4](a))); '
                         'for int i in [int(popcount(bit[4](a))):last + 2] { total += i; } '
                         'for int i in {int(popcount(bit[4](a))), int(popcount(bit[4](a + a)))} '
                         '{ total += i; }', (('total', None), ('a', None)), (18, 7))
        self.check_exact('array[float, 3] values = {0.0, pi / 2, pi}; '
                         'angle[4] a; bit[4] b; uint total = 0; '
                         'for float x in values { a = angle[4](x); b = bit[4](a); '
                         'total += popcount(bit[4](angle[4](b))); }',
                         (('a', None), ('b', 4), ('total', None)), (8, 8, 2))

    def test_skipped_paths_suppress_cast_operand_domains(self):
        for call in ('a = angle[4](b << count);',
                     'b = bit[4](a / zero);',
                     'b = bit[4](angle[4](1e309));'):
            for wrapper in ('if (flag) {{ {} }}', 'while (flag) {{ {} }}',
                            'for int i in [first:last] {{ {} }}', 'for int i in [1:0] {{ {} }}'):
                self.check_exact('angle[4] a = angle[4]("0011"); bit[4] b = "1010"; '
                                 'int count = -1; uint[4] zero = 0; bool flag = false; int first = 2; int last = 1; '
                                 + wrapper.format(call), (('a', None), ('b', 4)), (3, 10))
        for condition, expected in (('false && (bit[4](a / zero) == b)', 0),
                                    ('true || bool(angle[4](b << count))', 1)):
            self.check_exact('angle[4] a = pi; bit[4] b = "1000"; uint[4] zero = 0; int count = -1; '
                             'bool out = ' + condition + ';', (('out', None),), (expected,))
        for call in ('a = angle[4](b << count);', 'b = bit[4](a / zero);',
                     'b = bit[4](angle[4](1e309));'):
            result = self.execute(self.program('angle[4] a = pi; bit[4] b = "1000"; '
                                               'int count = -1; uint[4] zero = 0; ' + call))
            self.assertNotEqual(result.returncode, 0, call)
            self.assertNotIn('runtime error:', result.stderr)
            type(self).checked += 1

    def test_gates_preserve_quantum_phases_with_and_without_fusion(self):
        from qasm2qcx_u_power_numerical import BackendAuditTests
        phase = 3 * math.pi / 4
        expected = (1 / math.sqrt(2), cmath.exp(1j * phase) / math.sqrt(2))
        for fused in (False, True):
            lines = self.program('include "stdgates.inc"; qubit q; bit[4] bits = "0110"; '
                                 'angle[4] a = angle[4](bits); h q; p(a) q; '
                                 'bits = bit[4](a);', fused=fused)
            BackendAuditTests.check(self, lines, expected)
            theta = 1.0  # sin(pi/2), with no float-to-angle initialization.
            lines = self.program('include "stdgates.inc"; qubit q; bit[4] bits = "0100"; '
                                 'rx(sin(angle[4](bits))) q;', fused=fused)
            BackendAuditTests.check(self, lines, (math.cos(theta / 2), -1j * math.sin(theta / 2)))

    def test_iterator_shadowing_and_measurement_sources(self):
        self.check_exact('bit[4] b = "1001"; angle[4] a; uint total = 0; '
                         'array[float, 2] values = {0.0, pi / 2}; '
                         'for float b in values { a = angle[4](b); total += popcount(bit[4](a)); } '
                         'angle[4] out = angle[4](b);',
                         (('a', None), ('total', None), ('out', None), ('b', 4)), (4, 1, 9, 9))
        self.check_exact('include "stdgates.inc"; qubit q; x q; bit[4] flags = "0000"; '
                         'flags[2] = measure q; angle[4] a = angle[4](flags); bit[4] out = bit[4](a);',
                         (('flags', 4), ('a', None), ('out', 4)), (4, 4, 4))

    def test_documented_angle_bit_cast_example(self):
        from qasm2qcx_u_power_numerical import BackendAuditTests
        document = (pathlib.Path(__file__).parents[1] / 'qcx/qasm2qcx.md').read_text(encoding='utf-8')
        examples = re.findall(r'<!-- angle-bit-cast-example -->\s*```qasm\n(.*?)\n```',
                              document, flags=re.DOTALL)
        self.assertEqual(len(examples), 1)
        source = examples[0].removeprefix('OPENQASM 3.0;')
        self.check_exact(source, (('bits', 4), ('a', None), ('copied', 4)), (6, 6, 6))
        expected = (1 / math.sqrt(2), cmath.exp(3j * math.pi / 4) / math.sqrt(2))
        for fused in (False, True):
            BackendAuditTests.check(self, self.program(examples[0], fused=fused), expected)

    def test_direct_cast_phase_gate_arguments_and_modifiers(self):
        from qasm2qcx_u_power_numerical import BackendAuditTests
        phase = 3 * math.pi / 4
        for expression in ('angle[4](bits)', 'angle[4](BITS)', 'angle[4]("0110")',
                           'angle[4](bit[4](a))', 'angle[8](angle[4](bits))'):
            source = ('include "stdgates.inc"; qubit[2] q; const bit[4] BITS = "0110"; '
                      'bit[4] bits = BITS; angle[4] a = angle[4](bits); h q; ')
            for gate, expected in (
                    (f'p({expression}) q[0];',
                     tuple(cmath.exp(1j * phase) / 2 if index & 1 else 0.5 for index in range(4))),
                    (f'pow(0.5) @ p({expression}) q[0];',
                     tuple(cmath.exp(0.5j * phase) / 2 if index & 1 else 0.5 for index in range(4))),
                    (f'ctrl @ p({expression}) q[0], q[1];', (0.5, 0.5, 0.5, cmath.exp(1j * phase) / 2)),
                    (f'negctrl @ gphase({expression}) q[0];',
                     tuple(0.5 if index & 1 else cmath.exp(1j * phase) / 2 for index in range(4)))):
                for fused in (False, True):
                    BackendAuditTests.check(self, self.program(source + gate, fused=fused), expected)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra', type=pathlib.Path)
    arguments = parser.parse_args()
    BackendTests.bra = arguments.bra.resolve() if arguments.bra else None
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(BackendTests))
    print(f'angle/bit-register cast programs checked: {BackendTests.checked}')
    raise SystemExit(not result.wasSuccessful())
