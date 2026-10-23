#!/usr/bin/env python3

"""Check constant bit-register folding with a default-double non-MPI bra.

  python3 bra/test/qasm2qcx_constant_bit_register_numerical.py --bra /path/to/bra
"""

import argparse
import pathlib
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

    def test_named_constant_counts_and_rotation_round_trips(self):
        for width in (1, 3, 8, 32, 65, 129):
            for bits in ('0' * width, '1' * width, ('100101' * width)[:width]):
                source = f'''const bit[{width}] FLAGS = "{bits}";
                    const bit[{width}] COPY = FLAGS;
                    const bit[{width}] TURNED = rotl(~COPY, -2147483648);
                    const uint COUNT = popcount(FLAGS);
                    const uint INVERTED = popcount(TURNED);
                    const bool SAME = rotr(TURNED, -2147483648) == ~FLAGS;
                    uint out = COUNT; uint inverse = INVERTED; bool same = SAME;'''
                lines = self.program(source, (('out', False), ('inverse', False), ('same', False)))
                for name in ('FLAGS', 'COPY', 'TURNED'):
                    self.assertFalse(any(line.startswith(f'VAR {name}0 ') for line in lines))
                result = self.execute(lines)
                self.assertEqual(result.returncode, 0, result.stderr)
                count = bits.count('1')
                self.assertEqual(result.stdout.split(), [str(count), str(width - count), '1'])
                type(self).checked += 1

    def test_static_selection_bit_order_and_scalar_elements(self):
        bits = '10001111'
        for selector, indices in (('2:5', (2, 3, 4, 5)), ('7:-2:1', (7, 5, 3, 1)),
                                  ('-3:-1', (5, 6, 7)), ('{2, 2, 7}', (2, 2, 7)),
                                  ('{7, 0, 4, 1}', (7, 0, 4, 1)), ('0:0', (0,))):
            selected = ''.join(bits[::-1][index] for index in indices)[::-1]
            self.check(f'''const bit[8] FLAGS = "{bits}";
                const bit[{len(indices)}] SELECTED = FLAGS[{selector}];
                const bool SAME = SELECTED == "{selected}";
                const uint COUNT = popcount(SELECTED);
                const bool LOW = FLAGS[0]; const bool HIGH = FLAGS[-1];
                uint count = COUNT; bool same = SAME; bool low = LOW; bool high = HIGH;''',
                       (('count', False), ('same', False), ('low', False), ('high', False)),
                       (selected.count('1'), 1, 1, 1))

    def test_derived_widths_and_skipped_constant_errors(self):
        self.check('''const bit[8] FLAGS = "10001111";
            const uint WIDTH = popcount(FLAGS); const bit[WIDTH] ONES = "11111";
            const bool OFF = false && (popcount(FLAGS << (1 / 0)) > 0);
            const bool ON = true || (rotr(FLAGS, 1 / 0) == FLAGS);
            const uint COUNT = popcount(ONES);
            uint count = COUNT; bool off = OFF; bool on = ON;''',
                   (('count', False), ('off', False), ('on', False)), (5, 0, 1))

    def test_runtime_register_copies_and_scalar_elements(self):
        for width in (1, 8, 65, 129):
            bits = ('100101' * width)[:width]
            self.check(f'''const bit[{width}] FLAGS = "{bits}";
                bit[{width}] live = FLAGS; bool copied = live == FLAGS;
                live = ~FLAGS; bool inverse = live == ~FLAGS;
                bit low = FLAGS[0]; bool high = FLAGS[-1];
                uint count = popcount(FLAGS); uint changed = popcount(live);''',
                       (('copied', False), ('inverse', False), ('low', False), ('high', False),
                        ('count', False), ('changed', False)),
                       (1, 1, int(bits[-1]), int(bits[0]), bits.count('1'), bits.count('0')))
        self.check('''const bit[8] FLAGS = "10001111";
            bit[4] live = FLAGS[7:-2:1]; bool copied = live == "1101";
            live[3:-1:0] = FLAGS[{7, 0, 4, 1}]; bool selected = live == "1101";
            bit low; low = FLAGS[4]; bit[4] target; target[2] = FLAGS[0];
            bool high = FLAGS[7] && !FLAGS[4]; bool bitwise = bool(FLAGS[0] & FLAGS[7]);
            int index = 1; target[index] = FLAGS[0];
            bool scalar = target[1] && target[2];''',
                   (('copied', False), ('selected', False), ('low', False), ('high', False),
                    ('scalar', False), ('bitwise', False)), (1, 1, 0, 1, 1, 1))

    def test_mixed_register_expressions_and_runtime_distances(self):
        self.check('''const bit[8] FLAGS = "10001111"; bit[8] live = "10110100";
            int d = -2; bit[8] out = rotr(live ^ FLAGS, d);
            bool correct = out == "11101100";
            bit[8] turned = rotl(FLAGS, d); bool rotated = turned == "11100011";
            bit[8] shifted = FLAGS << int(popcount(live)); bool shiftok = shifted == "11110000";
            live ^= FLAGS; bool compound = live == "00111011";
            live &= ~FLAGS; bool masked = live == "00110000";
            uint count = popcount(FLAGS ^ live);''',
                   (('correct', False), ('rotated', False), ('shiftok', False),
                    ('compound', False), ('masked', False), ('count', False)), (1, 1, 1, 1, 1, 7))

    def test_conditions_loops_shadowing_and_short_circuiting(self):
        self.check('''const int WIDTH = 8; const bit[WIDTH] FLAGS = "10001111";
            bit[8] live = FLAGS; int visits = 0; uint total = 0;
            while (popcount(live) > 0) { live >>= 1; visits += 1; }
            for int WIDTH in [0:1] { total += popcount(FLAGS[2:5]); }
            for int FLAGS in [1:2] { total += FLAGS; }
            bit[2] flags = "10"; bit copied;
            for bit FLAGS in flags { copied = FLAGS; }
            if (FLAGS[0] && !FLAGS[4]) { total += popcount(FLAGS); }
            int bad = -1;
            bool off = false && (popcount(FLAGS << bad) > 0);
            bool on = true || (rotr(FLAGS, 1 / 0) == live);
            for int i in [1:0] { live = FLAGS << bad; }''',
                   (('visits', False), ('total', False), ('copied', False), ('off', False), ('on', False)),
                   (8, 12, 1, 0, 1))

    def test_runtime_errors_and_temporary_reuse(self):
        self.check('''const bit[8] FLAGS = "10001111"; bit[8] live = FLAGS; int bad = -1;
            uint count = 0;
            for int i in [0:3] { live = rotr(FLAGS ^ live, i); count += popcount(live); }
            bool ok = FLAGS == "10001111";
            if (false) { live = FLAGS << bad; }
            bool off = false && (popcount(FLAGS << bad) > 0);
            bool on = true || (popcount(rotr(FLAGS, uint(-1))) > 0);''',
                   (('ok', False), ('off', False), ('on', False)), (1, 0, 1))
        for expression, diagnostic in (('FLAGS << bad', 'assertion failed in ASSERT'),
                                        ('FLAGS >> (1 / zero)', 'integer division by zero'),
                                        ('rotl(FLAGS, uint(-1))', 'INT')):
            result = self.execute(self.program('''const bit[8] FLAGS = "10001111";
                bit[8] live = FLAGS; int bad = -1; int zero = 0;
                live = ''' + expression + ';'))
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
    print(f'constant bit-register programs checked: {BackendTests.checked}')
    raise SystemExit(not result.wasSuccessful())


if __name__ == '__main__':
    main()
