#!/usr/bin/env python3

"""Check scalar constant bits with a default-double non-MPI bra.

  python3 bra/test/qasm2qcx_constant_scalar_bit_numerical.py --bra /path/to/bra
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

    def test_scalar_copies_and_bitwise_truth_tables(self):
        for a in (0, 1):
            for b in (0, 1):
                self.check(f'''const bit A = {a}; const bit B = {b};
                    bit copied = A; bit both = A & B; bit either = A | B;
                    bit different = A ^ B; bit inverse = ~A;
                    bool equal = A == B; bool logical = A && !B;
                    bit live = B; live ^= A; bool mixed = live == (A ^ B);''',
                           (('copied', False), ('both', False), ('either', False),
                            ('different', False), ('inverse', False), ('equal', False),
                            ('logical', False), ('mixed', False)),
                           (a, a & b, a | b, a ^ b, 1 - a, int(a == b), a & (1 - b), 1))

    def test_register_elements_and_indexed_mutable_destinations(self):
        self.check('''const bit[8] FLAGS = "10001111";
            const bit LOW = FLAGS[0]; const bit OFF = FLAGS[4];
            const bit TOP = FLAGS[-1]; bit[3] live;
            live[0] = LOW; live[1] = OFF; int index = 2; live[index] = TOP;
            bool correct = live == "101"; bit copy; copy = LOW;
            bool ready = OFF; ready = TOP;''',
                   (('correct', False), ('copy', False), ('ready', False)), (1, 1, 1))

    def test_casts_arrays_and_nested_functions(self):
        self.check('''const bit ON = true; const bit OFF = false;
            const bit COPY = ON; const bool READY = ON && !OFF;
            const uint VALUE = uint(COPY); const float ROOT = sqrt(float(ON));
            array[bool, 2] flags = {OFF, ON}; flags[0] = ON;
            array[int, 2] numbers = {int(ON), int(OFF)}; numbers[1] = int(ON);
            bool ready = READY && flags[0] && flags[1]; uint value = VALUE;
            float root = ROOT; float nested = log(exp(float(ON)));
            int sum = numbers[0] + numbers[1];''',
                   (('ready', False), ('value', False), ('root', False),
                    ('nested', False), ('sum', False)), (1, 1, 1, 1, 2))

    def test_conditions_and_short_circuiting(self):
        self.check('''const bit ON = true; const bit OFF = false;
            const bool A = OFF && bool(1 / 0); const bool B = ON || bool(1 / 0);
            int zero = 0; bool skipped = OFF && bool(1 / zero);
            bool bypass = ON || bool(1 / zero); int result = 0;
            if (ON) { result += 1; } else { result = 99; }
            if (OFF) { result = 99; } else { result += 2; }
            bool folded = A || B;''',
                   (('skipped', False), ('bypass', False), ('result', False), ('folded', False)),
                   (0, 1, 3, 1))

    def test_reached_runtime_error_is_not_hidden_by_constant_storage(self):
        result = self.execute(self.program('''const bit ON = true; int zero = 0;
            bool reached = ON && bool(1 / zero);'''))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('integer division by zero', result.stderr)
        self.assertEqual(result.stdout, '')
        type(self).checked += 1

    def test_nested_loops_transfers_and_temporary_reuse(self):
        self.check('''const bit ON = true; const bit OFF = false;
            int stop = 3; int total = 0; int visits = 0; bit live = OFF;
            for int i in [0:stop] {
                if (i == 1 && ON) { continue; }
                for int j in [0:int(ON)] { total += int(ON ^ live); }
                live ^= ON; visits += 1;
                if (i == 2 && ON) { break; }
            }
            int count = 0;
            while (ON && count < 3) { count += 1; if (OFF) { break; } }
            bool unchanged = ON && !OFF;''',
                   (('total', False), ('visits', False), ('live', False),
                    ('count', False), ('unchanged', False)), (2, 2, 0, 3, 1))

    def test_iterator_shadowing_and_binding_restoration(self):
        self.check('''const bit FLAG = true; int total = 0; int stop = 2;
            for int FLAG in [0:stop] { total += FLAG; }
            for int FLAG in {0, stop} { total += FLAG; }
            bit[2] bits = "10"; bit copy;
            for bit FLAG in bits { copy = FLAG; }
            array[bool, 2] bools = {true, false}; bit copied;
            for bool FLAG in bools { copied = FLAG; }
            array[uint, 2] integers = {1, 2};
            for uint FLAG in integers { total += int(FLAG); }
            for int FLAG in [0:1] {
                total += FLAG;
                for int FLAG in [2:3] { total += FLAG; }
                total += FLAG;
            }
            bool restored = FLAG;''',
                   (('total', False), ('copy', False), ('copied', False), ('restored', False)),
                   (20, 1, 0, 1))

    def test_constant_casts_in_nested_loop_bounds(self):
        self.check('''const bit ON = true; int total = 0;
            for int i in [0:int(ON)] {
                for int j in [0:int(ON)] { total += int(ON); }
                for int j in {int(ON), int(!ON)} { total += j; }
            }
            for int i in [1:0] { total = 99; }''', (('total', False),), (6,))

    def test_measurement_controlled_execution_and_source_preservation(self):
        for initial in (0, 1):
            self.check(f'''include "stdgates.inc"; qubit q;
                const bit ON = true; const bit OFF = false;
                {'x q;' if initial else ''} bit measured = measure q;
                bit out = OFF; uint visits = 0;
                if (ON && measured) {{ out = ON; }}
                while (out && ON) {{ visits += uint(ON); out = OFF; }}
                bool same = measured == bit(bool({initial}));
                bool unchanged = ON && !OFF;''',
                       (('visits', False), ('out', False), ('same', False), ('unchanged', False)),
                       (initial, 0, 1, 1))

    def test_gate_parameters_powers_and_fusion(self):
        from qasm2qcx_u_power_numerical import BackendAuditTests
        source = '''include "stdgates.inc"; qubit q;
            const bit ON = true; const bit OFF = false;
            pow(int(ON)) @ x q;
            if (ON && !OFF) { rx(pi * float(ON)) q; }
            rz(pi * float(OFF)) q;'''
        for fused in (False, True):
            BackendAuditTests.check(self, self.program(source, fused=fused), (-1j, 0j))

    def test_documented_example(self):
        document = (pathlib.Path(__file__).parents[1] / 'qcx' / 'qasm2qcx.md').read_text()
        examples = re.findall(r'<!-- constant-scalar-bit-example -->\s*```qasm\n(.*?)\n```',
                              document, flags=re.DOTALL)
        self.assertEqual(len(examples), 1)
        for fused in (False, True):
            self.check(examples[0], (('live', False), ('ready', False), ('value', False)),
                       (1, 1, 1), fused=fused)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    BackendTests.bra = parser.parse_args().bra.resolve()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__)))
    print(f'constant scalar-bit programs checked: {BackendTests.checked}')
    raise SystemExit(not result.wasSuccessful())


if __name__ == '__main__':
    main()
