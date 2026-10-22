#!/usr/bin/env python3

"""Execute converted popcount calls with a default-double non-MPI bra.

  python3 bra/test/qasm2qcx_popcount_numerical.py --bra /path/to/bra
"""

import argparse
import pathlib
import re
import resource
import unittest

import qasm2qcx_math_function_numerical as functions


class BackendTests(unittest.TestCase):
    bra = None
    checked = 0
    setUp = functions.BackendTests.setUp
    program = functions.BackendTests.program
    execute = functions.BackendTests.execute
    check = functions.BackendTests.check

    def test_literals_constants_registers_and_widths_beyond_uint(self):
        for width in (1, 2, 3, 8, 32, 65, 129):
            for bits in ('0' * width, '1' * width, ('10' * width)[:width], '1' + '0' * (width - 1)):
                expected = sum(character == '1' for character in bits)
                for mode in ('literal', 'constant', 'variable'):
                    source = f'bit[{width}] flags = "{bits}"; '
                    if mode == 'constant':
                        source += f'const uint C = popcount("{bits}"); '
                    expression = 'C' if mode == 'constant' else f'popcount("{bits}")' if mode == 'literal' else 'popcount(flags)'
                    self.check(source + f'uint out = {expression}; uint saved = popcount(flags);',
                               (('out', False), ('saved', False)), (expected, expected))

    def test_static_selections_and_register_operators(self):
        for expression, expected in (('flags[2:5]', 3), ('flags[7:-2:1]', 2), ('flags[-3:-1]', 2),
                                      ('flags[{2, 2, 7}]', 3), ('~flags', 4),
                                      ('flags & "01010101"', 2), ('flags | "01010101"', 6),
                                      ('(flags ^ "01010101") >> 1', 3), ('flags[2:5] ^ "1111"', 1)):
            self.check(f'bit[8] flags = "10110100"; uint out = popcount({expression}); '
                       'uint saved = popcount(flags);', (('out', False), ('saved', False)), (expected, 4))

    def test_runtime_shifts_unsigned_arithmetic_and_narrow_assignment(self):
        for count in (0, 1, 4, 8, 99):
            expected = sum(character == '1' for character in format((0xb4 << count) & 255, '08b'))
            self.check(f'bit[8] flags = "10110100"; int shift = {count}; '
                       'uint out = popcount(flags << shift); uint power = popcount(flags) ** 2; '
                       'uint[3] narrow = popcount("11111111"); uint total = popcount(flags) + 1;',
                       (('out', False), ('power', False), ('narrow', False), ('total', False)),
                       (expected, 16, 0, 5))
        self.check('bit[8] flags = "10110100"; uint shift = uint(-1); '
                   'uint out = popcount(flags >> shift);', (('out', False),), (0,))

    def test_nested_functions_array_assignments_and_self_references(self):
        self.check('bit[4] flags = "1011"; uint n = 1; '
                   'n = popcount(flags << n) + popcount(~flags); '
                   'uint remainder = uint(mod(popcount(flags), 2)); '
                   'float out = sqrt(popcount(flags) + 1);',
                   (('n', False), ('remainder', False), ('out', False)), (3, 1, 2))
        self.check('bit[4] flags = "1011"; array[uint, 2] counts = {0, 0}; int i = 0; '
                   'counts[i] = popcount(flags << counts[i]); uint total = 0; '
                   'for uint count in counts { total += count; }', (('total', False),), (3,))

    def test_conditions_runtime_loops_and_scoped_iterators(self):
        self.check('bit[4] flags = "1011"; uint total = 0; int visits = 0; '
                   'while (popcount(flags) > 0) { flags >>= 1; total += popcount(flags); visits += 1; }',
                   (('total', False), ('visits', False)), (4, 4))
        self.check('bit[4] flags = "1011"; uint total = 0; '
                   'for int i in [0:popcount(flags)] { total += 1; } '
                   'for int j in {popcount(flags), popcount("1")} { total += j; } '
                   'for int k in [0:1] { total += popcount(flags[0:1]); }', (('total', False),), (12,))
        self.check('const uint SIZE = 4; bit[SIZE] flags = "1011"; uint total = 0; '
                   'for int SIZE in [0:1] { total += popcount(flags); } '
                   'for bit flag in flags { total += popcount(flags); }', (('total', False),), (18,))

    def test_skipped_shift_errors_and_reached_diagnostics(self):
        for wrapper in ('if (ready) {{ {} }}', 'while (ready) {{ {} }}',
                        'for int i in [first:last] {{ {} }}', 'for int i in [1:0] {{ {} }}'):
            self.check('bit[4] flags = "1011"; bool ready = false; int first = 1; int last = 0; '
                       'int count = -1; uint out = 7; ' + wrapper.format('out = popcount(flags << count);'),
                       (('out', False),), (7,))
        self.check('bit[4] flags = "1011"; int zero = 0; '
                   'bool ready = false && (popcount(flags << (7 / zero)) > 0);', (('ready', False),), (0,))
        for expression, diagnostic in (('flags << count', 'assertion failed in ASSERT'),
                                      ('flags << (7 / zero)', 'integer division by zero in LET')):
            result = self.execute(self.program('bit[4] flags = "1011"; int count = -1; int zero = 0; '
                                              f'uint out = popcount({expression});', (('out', False),)))
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(diagnostic, result.stderr)
            self.assertEqual(result.stdout, '')
            type(self).checked += 1

    def test_measurement_results(self):
        self.check('include "stdgates.inc"; qubit[2] q; x q[0]; bit[2] flags = measure q; '
                   'uint out = popcount(flags); uint inverted = popcount(~flags);',
                   (('out', False), ('inverted', False)), (1, 1))

    def test_gate_modifiers_with_and_without_fusion(self):
        from qasm2qcx_u_power_numerical import BackendAuditTests
        source = 'include "stdgates.inc"; qubit q; bit[4] flags = "1011"; pow(popcount(flags)) @ x q;'
        for fused in (False, True):
            BackendAuditTests.check(self, self.program(source, fused=fused), (0j, complex(1)))

    def test_documented_example(self):
        document = (pathlib.Path(__file__).parents[1] / 'qcx/qasm2qcx.md').read_text(encoding='utf-8')
        examples = re.findall(r'<!-- popcount-example -->\s*```qasm\n(.*?)\n```', document, flags=re.DOTALL)
        self.assertEqual(len(examples), 1)
        for fused in (False, True):
            self.check(examples[0], (('count', False), ('selected', False)), (4, 3), fused=fused)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    BackendTests.bra = parser.parse_args().bra.resolve()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__)))
    print(f'popcount programs checked: {BackendTests.checked}')
    raise SystemExit(not result.wasSuccessful())


if __name__ == '__main__':
    main()
