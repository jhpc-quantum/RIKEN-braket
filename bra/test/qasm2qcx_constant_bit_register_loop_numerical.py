#!/usr/bin/env python3

"""Check constant bit-register iteration with a default-double non-MPI bra.

  python3 bra/test/qasm2qcx_constant_bit_register_loop_numerical.py --bra /path/to/bra
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

    def test_runtime_reentry_and_mixed_loop_nesting(self):
        self.check('''const bit[3] FLAGS = "101"; bit[2] live = "10";
            array[int, 2] numbers = {1, 2}; int stop = 1; int total = 0; int visits = 0;
            for int outer in [0:stop] {
                for bit value in FLAGS {
                    for bit inner in live { total += int(value && inner); }
                    for int number in numbers { total += number * int(value); }
                    visits += 1;
                }
            }
            int rounds = 0;
            while (rounds < 2) { for bit value in FLAGS { total += int(value); } rounds += 1; }
            bool unchanged = FLAGS == "101";''',
                   (('total', False), ('visits', False), ('rounds', False), ('unchanged', False)), (20, 6, 2, 1))

    def test_live_mutable_inner_register_and_nearest_transfers(self):
        self.check('''const bit[4] FLAGS = "1011"; bit[3] live = "001";
            int outervisits = 0; int innervisits = 0; int ones = 0;
            for bit value in FLAGS {
                outervisits += 1;
                for bit inner in live {
                    innervisits += 1;
                    if (inner) { live[1] = true; ones += int(inner); continue; }
                    break;
                }
                if (!value) { continue; }
                ones += int(value);
            }
            bool changed = live == "011"; bool unchanged = FLAGS == "1011";''',
                   (('outervisits', False), ('innervisits', False), ('ones', False),
                    ('changed', False), ('unchanged', False)), (4, 12, 11, 1, 1))

    def test_shadowing_and_declaration_scope_width(self):
        self.check('''const uint WIDTH = 4; const bit[WIDTH] FLAGS = "1011";
            const bit value = false; int total = 0;
            for int WIDTH in [0:1] {
                for bit value in FLAGS { total += int(value); }
            }
            for bit FLAGS in FLAGS {
                for int FLAGS in [0:1] { total += FLAGS; }
                total += int(FLAGS);
            }
            bool scalar = !value; bool restored = FLAGS == "1011";
            uint width = WIDTH;''',
                   (('total', False), ('scalar', False), ('restored', False), ('width', False)), (13, 1, 1, 4))

    def test_measurement_controlled_execution(self):
        for initial in (0, 1):
            self.check(f'''include "stdgates.inc"; qubit q;
                const bit[4] FLAGS = "1011"; {'x q;' if initial else ''}
                bit measured = measure q; int count = 0; int visits = 0;
                for bit value in FLAGS {{
                    if (measured && value) {{ count += int(value); }}
                    if (measured) {{ continue; }} visits += 1;
                }}
                bool unchanged = FLAGS == "1011";''',
                       (('count', False), ('visits', False), ('unchanged', False)),
                       (3 * initial, 4 * (1 - initial), 1))

    def test_skipped_and_reached_errors_and_temporary_reuse(self):
        self.check('''const bit[4] FLAGS = "1011"; int zero = 0; int total = 0;
            for bit value in FLAGS {
                if (false) { total += int(value) / zero; }
                if (!value) { continue; }
                if (false && bool(int(value) / zero)) { total = 99; }
                total += int(value & ~bit(false));
            }
            for bit value in FLAGS { if (value) { break; } total = 99; }
            bool unchanged = FLAGS == "1011";''',
                   (('total', False), ('unchanged', False)), (3, 1))
        result = self.execute(self.program('''const bit[3] FLAGS = "101"; int zero = 0; int out = 0;
            for bit value in FLAGS { if (!value) { out = int(value) / zero; } }'''))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('integer division by zero', result.stderr)
        self.assertEqual(result.stdout, '')
        type(self).checked += 1

    def test_gate_parameters_modifiers_and_fusion(self):
        from qasm2qcx_u_power_numerical import BackendAuditTests
        source = '''include "stdgates.inc"; qubit q; const bit[3] FLAGS = "101";
            for bit value in FLAGS { pow(int(value)) @ x q; }
            for bit value in FLAGS { rx(pi * float(value)) q; }'''
        for fused in (False, True):
            BackendAuditTests.check(self, self.program(source, fused=fused), (-1 + 0j, 0j))

    def test_documented_example(self):
        document = (pathlib.Path(__file__).parents[1] / 'qcx' / 'qasm2qcx.md').read_text()
        examples = re.findall(r'<!-- constant-bit-register-loop-example -->\s*```qasm\n(.*?)\n```',
                              document, flags=re.DOTALL)
        self.assertEqual(len(examples), 1)
        for fused in (False, True):
            self.check(examples[0], (('count', False), ('visits', False), ('unchanged', False)),
                       (5, 8, 1), fused=fused)


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
