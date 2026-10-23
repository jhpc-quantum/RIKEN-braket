#!/usr/bin/env python3

"""Check converted circular rotations with a default-double non-MPI bra.

  python3 bra/test/qasm2qcx_rotation_numerical.py --bra /path/to/bra
"""

import argparse
import pathlib
import resource
import unittest

import qasm2qcx_math_function_numerical as functions


def reference(value, width, distance, name):
    # Independent oracle: rotate a textual, most-significant-bit-first word.
    bits = format(value, f'0{width}b')
    count = (distance if name == 'rotl' else -distance) % width
    return int(bits[count:] + bits[:count], 2)


class BackendTests(unittest.TestCase):
    bra = None
    checked = 0
    setUp = functions.BackendTests.setUp
    program = functions.BackendTests.program
    execute = functions.BackendTests.execute

    def check(self, source, names, expected):
        lines = self.program(source, tuple((name, False) for name in names))
        result = self.execute(lines)
        self.assertEqual(result.returncode, 0, result.stderr + '\n' + '\n'.join(lines))
        # Integer results must be exact, including full-width UINT values.
        self.assertEqual([int(value) for value in result.stdout.split()], list(expected), source)
        type(self).checked += 1

    def test_unsigned_boundary_values_and_distances(self):
        native = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        minimum = self.module.QASM2QCXConverter.QCX_INT_MIN
        maximum = self.module.QASM2QCXConverter.QCX_INT_MAX
        for width in (1, 3, 6, 8, native):
            mask = (1 << width) - 1
            for value in sorted({0, 1, 37 & mask, mask >> 1, mask}):
                for distance in sorted({0, 1, -1, width, -width, width * 3 + 2, minimum, maximum}):
                    for name in ('rotl', 'rotr'):
                        expected = reference(value, width, distance, name)
                        self.check(f'''const uint[{width}] C = {name}(uint[{width}]({value}), {distance});
                            uint[{width}] u = {value}; int d = {distance};
                            uint[{width}] out = {name}(u, d);
                            uint[{width}] literal = {name}(uint[{width}]({value}), {distance});
                            uint[{width}] folded = C; u = {name}(u, d);''',
                                   ('out', 'literal', 'folded', 'u', 'd'),
                                   (expected, expected, expected, expected, distance))

    def test_nesting_arithmetic_indexed_operands_and_assignment(self):
        self.check('''uint[6] u = 37; int d = -2;
            uint[6] restored = rotr(rotl(u, d), d);
            uint complement = ~rotl(u, 0);
            uint[3] narrow = rotl(u, 3);
            uint power = rotl(u, 3) ** 2;
            uint narrowpower = rotl(u, 3) ** uint[6](2);
            uint[6] nested = rotl(rotl(u, 1), int(rotr(u, 2)));
            array[uint[6], 3] a = {37, 1, 2}; array[int, 2] counts = {-2, 0};
            int i = 0; a[i] = rotl(a[i], counts[i]);
            a[a[1]] = rotr(a[a[1]], int(a[2]));
            uint first = a[0]; uint second = a[1];''',
                   ('restored', 'complement', 'narrow', 'power', 'narrowpower', 'nested', 'first', 'second', 'u', 'd'),
                   (37, 26, 4, 44 ** 2, (44 ** 2) & 63,
                    reference(reference(37, 6, 1, 'rotl'), 6, reference(37, 6, 2, 'rotr'), 'rotl'),
                    reference(37, 6, -2, 'rotl'), reference(1, 6, 2, 'rotr'), 37, -2))

    def test_unsigned_and_boolean_distance_promotions(self):
        for count_type, count in (('uint', 0), ('uint', 13), ('uint', 2147483647),
                                  ('bool', 'true'), ('bool', 'false')):
            distance = 1 if count == 'true' else 0 if count == 'false' else count
            self.check(f'uint[6] u = 37; {count_type} d = {count}; uint out = rotl(u, d);',
                       ('out',), (reference(37, 6, distance, 'rotl'),))

    def test_conditions_loops_and_short_circuiting(self):
        self.check('''uint[6] u = 37; int count = 0; int d = -1;
            while (rotl(u, count) != u || count == 0) { count += 1; }
            for int i in [0:2] { u = rotr(u, d); }
            int visits = 0;
            for int i in [0:rotl(uint[3](1), 1)] { visits += 1; }
            bool equal = rotl(u, 6) == u;''',
                   ('u', 'count', 'visits', 'equal'), (44, 6, 3, 1))
        self.check('''uint[6] u = 37; uint bad = uint(-1); int zero = 0;
            if (false) { u = rotl(u, bad); u = rotr(u, 1 / zero); }
            bool off = false && bool(rotl(u, bad));
            bool on = true || bool(rotr(uint[6](37), uint(-1)));
            for int i in [1:0] { u = rotl(u, bad); }''',
                   ('u', 'off', 'on'), (37, 0, 1))

    def test_reached_conversion_and_division_errors(self):
        for expression, diagnostic in (('rotl(u, bad)', 'INT'),
                                        ('rotl(uint[6](37), uint(-1))', 'INT'),
                                        ('rotr(u, 1 / zero)', 'integer division by zero')):
            lines = self.program('uint[6] u = 37; uint bad = uint(-1); int zero = 0; '
                                 f'uint out = {expression};', (('out', False),))
            result = self.execute(lines)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(diagnostic, result.stderr)
            self.assertEqual(result.stdout, '')
            type(self).checked += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    BackendTests.bra = parser.parse_args().bra.resolve()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__)))
    print(f'rotation programs checked: {BackendTests.checked}')
    raise SystemExit(not result.wasSuccessful())


if __name__ == '__main__':
    main()
