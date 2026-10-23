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
