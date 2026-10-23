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

    def test_register_widths_distances_and_self_assignment(self):
        minimum = self.module.QASM2QCXConverter.QCX_INT_MIN
        maximum = self.module.QASM2QCXConverter.QCX_INT_MAX
        for width in (1, 4, 8, 65, 129):
            bits = ('1001011' * width)[:width]
            value = int(bits, 2)
            for distance in sorted({0, 1, -1, width, -width, width * 3 + 2, minimum, maximum}):
                for name in ('rotl', 'rotr'):
                    expected = format(reference(value, width, distance, name), f'0{width}b')
                    self.check(f'''bit[{width}] b = "{bits}"; int d = {distance};
                        bit[{width}] out = {name}(b, d);
                        bit[{width}] literal = {name}(b, {distance});
                        bit[{width}] runtime_literal = {name}("{bits}", d);
                        bit[{width}] folded = {name}("{bits}", {distance});
                        bool sourceok = b == "{bits}";
                        b = {name}(b, d);
                        bool outok = out == "{expected}";
                        bool literalok = literal == "{expected}";
                        bool runtimeok = runtime_literal == "{expected}";
                        bool foldedok = folded == "{expected}";
                        bool selfok = b == "{expected}";''',
                               ('sourceok', 'outok', 'literalok', 'runtimeok', 'foldedok', 'selfok'), (1,) * 6)

    def test_register_selections_overlap_and_operator_composition(self):
        for distance in (0, 1, -2, 9):
            selected = 0b1101  # flags[7:-2:1], low-index-first selection
            expected = format(reference(selected, 4, distance, 'rotl'), '04b')
            self.check(f'''bit[8] flags = "11001010"; int d = {distance};
                bit[4] out = rotl(flags[7:-2:1], d);
                bool correct = out == "{expected}";
                bit[4] duplicates = rotr(flags[{{1, 1, 6, 7}}], d);
                bool duplicateok = duplicates == "{format(reference(15, 4, distance, 'rotr'), '04b')}";
                bit[4] original = flags[7:-2:1];
                flags[7:-2:1] = rotl(flags[7:-2:1], d);
                bool overlapok = flags[7:-2:1] == "{expected}";
                bit[4] restored = rotr(out, d);
                bool restoredok = restored == original;''',
                       ('correct', 'duplicateok', 'overlapok', 'restoredok'), (1, 1, 1, 1))
            reversed_bits = format(reference(0b1011, 4, distance, 'rotl'), '04b')[::-1]
            self.check(f'''bit[4] b = "1011"; bit[4] c = "1011"; int d = {distance};
                b[3:-1:0] = rotl(b, d); c[3:-1:0] = rotl(c, {distance});
                bool runtimeok = b == "{reversed_bits}";
                bool literalok = c == "{reversed_bits}";''', ('runtimeok', 'literalok'), (1, 1))
        self.check('''bit[4] b = "1011"; int zero = 0;
            b[3:-1:0] = rotl(b, zero); bool reverseok = b == "1101";
            bit[4] a = "1011"; a[3:-1:0] = rotr(a, 0); bool aliasok = a == "1101";
            bit[4] c = "1011"; int d = 1;
            bit[4] out = ~rotr(rotl(c, d), d) | "0001";
            bool operatorok = out == "0101";
            bit[4] shifted = rotl(c, d) << 1; bool shiftok = shifted == "1110";
            c ^= rotl(c, d); bool compoundok = c == "1100";
            uint count = popcount(rotr(out, d));''',
                   ('reverseok', 'aliasok', 'operatorok', 'shiftok', 'compoundok', 'count'), (1, 1, 1, 1, 1, 2))

    def test_register_runtime_contexts_and_unsigned_distance(self):
        self.check('''bit[4] b = "1011"; uint d = 13;
            bit[4] out = rotl(b, d); bool ok = out == "0111";
            bit[4] nested = rotr(rotl(b, int(popcount(b))), int(popcount(b)));
            bool nestedok = nested == b;
            uint[3] u = 1; bit[4] mixed = rotl(b, int(rotl(u, 1)));
            bool mixedok = mixed == "1110";
            int visits = 0; while (rotl(b, visits) != b || visits == 0) { visits += 1; }
            uint total = 0; for int i in [0:1] { total += popcount(rotl(b, i)); }
            bool yes = true; bit[4] promoted = rotr(b, yes);
            bool promotedok = promoted == "1101";''',
                   ('ok', 'nestedok', 'mixedok', 'visits', 'total', 'promotedok'), (1, 1, 1, 4, 6, 1))

    def test_register_skipped_and_reached_errors(self):
        self.check('''bit[4] b = "1011"; uint bad = uint(-1); int zero = 0;
            if (false) { b = rotl(b, bad); b = rotr("1011", 1 / zero); }
            bool off = false && (popcount(rotl(b, bad)) > 0);
            bool on = true || (rotl(b, 1 / zero) == b);
            for int i in [1:0] { b = rotl(b, bad); }
            bool unchanged = b == "1011";''', ('off', 'on', 'unchanged'), (0, 1, 1))
        for expression, diagnostic in (('rotl(b, bad)', 'INT'),
                                       ('rotr("1011", uint(-1))', 'INT'),
                                       ('rotl(b, 1 / zero)', 'integer division by zero'),
                                       ('rotr("1011", 1 / 0)', 'integer division by zero')):
            lines = self.program('bit[4] b = "1011"; uint bad = uint(-1); int zero = 0; '
                                 f'bit[4] out = {expression};')
            result = self.execute(lines)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(diagnostic, result.stderr)
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
