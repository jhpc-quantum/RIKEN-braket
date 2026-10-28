#!/usr/bin/env python3
"""Audit exact angle/bit-register casts (#412).

Run with --bra /path/to/bra for a non-MPI backend audit. References use
integer patterns and individual little-endian bits, without floating point.
"""

import argparse
import pathlib
import random
import resource
import unittest

import qasm2qcx_math_function_numerical as functions


class BackendTests(unittest.TestCase):
    bra = None
    checked = 0
    setUp = functions.BackendTests.setUp
    execute = functions.BackendTests.execute

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra', type=pathlib.Path)
    arguments = parser.parse_args()
    BackendTests.bra = arguments.bra.resolve() if arguments.bra else None
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(BackendTests))
    print(f'angle/bit-register cast programs checked: {BackendTests.checked}')
    raise SystemExit(not result.wasSuccessful())
