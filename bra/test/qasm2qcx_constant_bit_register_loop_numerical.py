#!/usr/bin/env python3

"""Check constant bit-register iteration with a default-double non-MPI bra.

  python3 bra/test/qasm2qcx_constant_bit_register_loop_numerical.py --bra /path/to/bra
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

    def test_counts_visits_and_order_at_singleton_and_wide_widths(self):
        for width in (1, 3, 8, 32, 65, 129):
            for bits in ('0' * width, '1' * width, ('101100' * width)[:width]):
                checksum = 0
                for bit in reversed(bits):
                    checksum = (checksum * 3 + int(bit)) % 997
                self.check(f'''const bit[{width}] FLAGS = "{bits}";
                    uint count = 0; int visits = 0; uint checksum = 0;
                    for bit value in FLAGS {{
                        count += uint(value); visits += 1;
                        checksum = (checksum * 3 + uint(value)) % 997;
                    }}
                    bool unchanged = FLAGS == "{bits}";''',
                           (('count', False), ('visits', False), ('checksum', False), ('unchanged', False)),
                           (bits.count('1'), width, checksum, 1))

    def test_conditional_break_and_continue(self):
        self.check('''const bit[8] FLAGS = "10001111";
            uint count = 0; int visits = 0;
            for bit value in FLAGS {
                visits += 1;
                if (!value) { continue; }
                count += uint(value);
                if (count == 3) { break; }
            }
            bool same = FLAGS == "10001111";''',
                   (('count', False), ('visits', False), ('same', False)), (3, 3, 1))
        self.check('''const bit[4] FLAGS = "1010";
            int visits = 0; int ones = 0;
            for bit value in FLAGS {
                visits += 1; if (!value) { continue; } ones += int(value);
            }''', (('visits', False), ('ones', False)), (4, 2))

    def test_unconditional_transfers(self):
        for transfer, expected in (('break', 1), ('continue', 4)):
            self.check('''const bit[4] FLAGS = "1011"; int visits = 0; int skipped = 0;
                for bit value in FLAGS { visits += 1; ''' + transfer + '''; skipped += int(value); }''',
                       (('visits', False), ('skipped', False)), (expected, 0))

    def test_nested_constant_register_loops(self):
        self.check('''const bit[3] A = "101"; const bit[4] B = "1011";
            int count = 0; int visits = 0;
            for bit outer in A {
                for bit inner in B { count += int(outer && inner); visits += 1; }
            }''', (('count', False), ('visits', False)), (6, 12))

    def test_same_name_iterator_and_outer_binding_restoration(self):
        self.check('''const bit[4] FLAGS = "1011"; uint count = 0; bit copy;
            for bit FLAGS in FLAGS { copy = FLAGS; count += uint(FLAGS); }
            bool restored = FLAGS == "1011";''',
                   (('count', False), ('copy', False), ('restored', False)), (3, 1, 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    BackendTests.bra = parser.parse_args().bra.resolve()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__)))
    print(f'constant bit-register loop programs checked: {BackendTests.checked}')
    raise SystemExit(not result.wasSuccessful())


if __name__ == '__main__':
    main()
