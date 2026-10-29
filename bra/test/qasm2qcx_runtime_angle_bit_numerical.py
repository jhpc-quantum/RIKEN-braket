#!/usr/bin/env python3
"""Audit runtime single-bit selections on sized angles (#426).

Pass --bra /path/to/bra for the default-double non-MPI backend audit.
"""

import argparse
import pathlib
import re
import resource
import unittest

import qasm2qcx_angle_array_numerical as arrays


class BackendTests(unittest.TestCase):
    bra = None
    checked = 0
    setUp = arrays.BackendTests.setUp
    execute = arrays.BackendTests.execute
    program = arrays.BackendTests.program
    check_initializers = arrays.BackendTests.check_initializers

    def test_reads_at_every_width_with_signed_unsigned_and_negative_indices(self):
        for width in range(1, self.module.QASM2QCXConverter.QCX_UINT_WIDTH + 1):
            mask = (1 << width) - 1
            for raw in sorted({0, 1 << (width - 1), mask, 0xAAAAAAAA & mask}):
                source = (f'const angle[{width}] C = angle[{width}]("{raw:0{width}b}"); '
                          f'angle[{width}] value = C; array[angle[{width}], 1] phases = {{C}}; '
                          'int element = -1; ')
                outputs, expected = [], []
                for bit in sorted({0, width // 2, width - 1}):
                    source += f'int signed{bit} = {bit}; int negative{bit} = {bit - width}; uint unsigned{bit} = {bit}; '
                    for kind in ('signed', 'negative', 'unsigned'):
                        for origin, operand in enumerate(('C', 'value', 'phases[element]')):
                            name = f'out{kind}{bit}_{origin}'
                            source += f'bit[1] {name} = {operand}[{kind}{bit}]; '
                            outputs.append((name, 1))
                            expected.append((raw >> bit) & 1)
                self.check_initializers(source, tuple(outputs) + (('phases', 1),),
                                        expected + [raw], ('value',), (raw,))

    def test_register_consumers_and_nested_index_expressions(self):
        self.check_initializers('angle[4] a = angle[4]("1010"); int i = -1; uint u = 0; '
            'array[angle[4], 2] phases = {a, angle[4]("0011")}; '
            'array[int, 2] indices = {-1, 0}; int element = 1; '
            'bit[1] direct = a[i]; bit[1] inverse = ~a[u]; '
            'bit[1] combined = a[i] ^ phases[element][u]; '
            'bit[1] rotated = rotl(a[i], i); bit[1] shifted = a[i] >> u; '
            'uint count = popcount(a[i]); bool same = a[i] == "1"; '
            'bool different = a[u] != a[i]; angle[1] roundtrip = angle[1](a[i]); '
            'bit[1] nested = phases[popcount(a[i])][indices[element]]; '
            'bit[1] recursive = phases[element][popcount(a[i])]; '
            'bit[1] arithmetic = a[indices[element] + 3];',
            (('direct', 1), ('inverse', 1), ('combined', 1), ('rotated', 1), ('shifted', 1),
             ('nested', 1), ('recursive', 1), ('arithmetic', 1)),
            (1, 1, 0, 1, 1, 1, 1, 1), ('count', 'same', 'different', 'roundtrip'), (1, 1, 1, 1))

    def test_reentry_shadowing_and_angle_iterators(self):
        self.check_initializers('const int index = 100; const angle[4] value = pi; '
            'array[angle[4], 2] phases = {angle[4]("0011"), angle[4]("1010")}; '
            'uint total = 0; bit[1] out; int stop = 3; int visits = 0; '
            'for int index in [0:3] { total += popcount(value[index]); } '
            'for int index in [0:stop] { for angle[4] value in phases { '
            'out = value[index]; total += popcount(out); }} '
            'while (visits < 2) { visits += 1; for int index in {-1, 0} { '
            'total += popcount(phases[-1][index]); }}',
            (('out', 1),), (1,), ('total', 'visits'), (7, 2))

    def test_bounds_and_domain_failures_only_execute_on_reached_paths(self):
        native = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        maximum = (1 << native) - 1
        prefix = ('angle[4] value = pi; array[angle[4], 1] phases = {pi}; '
                  'int high = 4; int low = -5; int zero = 0; int element = 1; '
                  f'uint huge = {maximum}; int minimum = {self.module.QASM2QCXConverter.QCX_INT_MIN}; '
                  'bit[1] out = "1"; ')
        for operand in ('value[high]', 'value[low]', 'value[minimum]', 'value[huge]',
                        'value[1 / zero]', 'phases[element][zero]',
                        'phases[zero][high]', 'phases[1 / zero][high]'):
            for wrapper in ('if (false) {{ out = {}; }}', 'while (false) {{ out = {}; }}',
                            'for int outer in [1:0] {{ out = {}; }}'):
                self.check_initializers(prefix + wrapper.format(operand), (('out', 1),), (1,))
            self.check_initializers(prefix + f'bool yes = true || {operand} == "0"; '
                                    f'bool no = false && {operand} == "0";',
                                    (('out', 1),), (1,), ('yes', 'no'), (1, 0))
            result = self.execute(self.program(prefix + f'out = {operand};'))
            self.assertNotEqual(result.returncode, 0, operand)
            self.assertNotIn('runtime error:', result.stderr)
            type(self).checked += 1


    def test_writes_and_compounds_preserve_other_bits_at_every_width(self):
        for width in range(1, self.module.QASM2QCXConverter.QCX_UINT_WIDTH + 1):
            mask = (1 << width) - 1
            for raw in sorted({0, mask, 0xAAAAAAAA & mask}):
                source = (f'angle[{width}] value = angle[{width}]("{raw:0{width}b}"); '
                          f'array[angle[{width}], 1] phases = {{value}}; int element = -1; '
                          f'int last = -1; uint first = 0; uint huge = {2**self.module.QASM2QCXConverter.QCX_UINT_WIDTH - 1}; ')
                snapshots, expected = [], []
                for origin, operand in enumerate(('value', 'phases[element]')):
                    current = raw
                    step = 0
                    for index_name, bit_index in (('last', width - 1), ('first', 0)):
                        for operator, rhs, count in (('=', '"1"', 1), ('&=', '"0"', 0),
                                ('|=', '"1"', 1), ('^=', '"1"', 1),
                                ('=', f'{operand}[{index_name}] ^ "1"', None),
                                ('<<=', '0', 0), ('>>=', 'huge', 1),
                                ('|=', '"1"', 1), ('<<=', str(2**200), 1),
                                ('|=', '"1"', 1), ('>>=', '1', 1)):
                            old = (current >> bit_index) & 1
                            if operator == '=':
                                new = 1 - old if count is None else count
                            elif operator == '&=':
                                new = old & count
                            elif operator == '|=':
                                new = old | count
                            elif operator == '^=':
                                new = old ^ count
                            else:
                                new = old if count == 0 else 0
                            current = (current & ~(1 << bit_index)) | (new << bit_index)
                            name = f'snapshot{origin}_{step}'
                            source += f'{operand}[{index_name}] {operator} {rhs}; angle[{width}] {name} = {operand}; '
                            snapshots.append(name)
                            expected.append(current)
                            step += 1
                self.check_initializers(source, (), (), tuple(snapshots), tuple(expected))

    def test_writes_with_overlapping_sources_and_nested_index_reads(self):
        self.check_initializers('angle[4] a = angle[4]("1010"); int index = -1; uint zero = 0; '
            'array[angle[4], 2] phases = {a, angle[4]("0011")}; int element = -1; '
            'a[index] ^= a[zero]; phases[element][index] = a[index]; '
            'phases[element][zero] &= phases[element][index]; '
            'a[popcount(phases[element][zero])] = ~a[index]; '
            'phases[popcount(a[index])][zero] >>= popcount(a[index]); '
            'a[0] |= a[index]; a[index] &= a[{0}];',
            (('phases', 2),), (10, 10), ('a', 'index', 'element'), (9, -1, -1))

    def test_write_reentry_and_iterator_indices(self):
        self.check_initializers('angle[4] a; array[angle[4], 2] phases; int stop = 3; int visits = 0; '
            'for int i in [0:3] { a[i] = "1"; } '
            'for int i in [0:stop] { phases[-1][i] ^= a[i]; } '
            'while (visits < 2) { visits += 1; for int i in {0, -1} { phases[0][i] ^= "1"; }} '
            'for int i in [0:3] { if (i == 1) { continue; } a[i] ^= "1"; if (i == 2) { break; }}',
            (('phases', 2),), (0, 15), ('a', 'visits'), (10, 2))

    def test_write_failures_are_guarded_on_skipped_paths(self):
        prefix = ('angle[4] a = pi; array[angle[4], 1] phases = {pi}; int high = 4; '
                  'int low = -5; int zero = 0; int element = 1; int negative = -1; '
                  f'int minimum = {self.module.QASM2QCXConverter.QCX_INT_MIN}; '
                  f'uint huge = {2**self.module.QASM2QCXConverter.QCX_UINT_WIDTH - 1}; ')
        for statement in ('a[high] = "1";', 'a[low] ^= "1";', 'a[huge] |= "1";',
                          'a[1 / zero] = "1";', 'phases[element][zero] &= "1";',
                          'phases[zero][high] >>= 0;', 'a[zero] ^= a[high];',
                          'a[zero] = phases[element][zero];', 'a[zero] <<= negative;',
                          'a[zero] >>= 1 / zero;', 'a[minimum] = "1";', 'a[zero] >>= minimum;'):
            for wrapper in ('if (false) {{ {} }}', 'while (false) {{ {} }}',
                            'for int outer in [1:0] {{ {} }}'):
                self.check_initializers(prefix + wrapper.format(statement), (('phases', 1),), (8,), ('a',), (8,))
            result = self.execute(self.program(prefix + statement))
            self.assertNotEqual(result.returncode, 0, statement)
            self.assertNotIn('runtime error:', result.stderr)
            type(self).checked += 1

    def test_gate_parameters_remain_captured_across_runtime_selection_writes(self):
        import cmath
        import math
        from qasm2qcx_u_power_numerical import BackendAuditTests
        gates = (
            ('p(value) q[0];', 1, lambda i: bool(i & 1)),
            ('inv @ p(value) q[0];', -1, lambda i: bool(i & 1)),
            ('pow(exponent) @ p(value) q[0];', 2, lambda i: bool(i & 1)),
            ('pow(0.5) @ p(value) q[0];', 0.5, lambda i: bool(i & 1)),
            ('cp(value) q[0], q[1];', 1, lambda i: i == 3),
            ('negctrl @ p(value) q[0], q[1];', 1, lambda i: i == 2),
            ('gphase(value);', 1, lambda i: True),
        )
        for array in (False, True):
            operand = 'phases[element]' if array else 'value'
            declaration = ('array[angle[4], 1] phases = {angle[4]("0011")}; int element = -1; '
                           if array else 'angle[4] value = angle[4]("0011"); ')
            for gate, factor, active in gates:
                gate = gate.replace('value', operand)
                source = ('include "stdgates.inc"; qubit[2] q; h q; int index = -1; int exponent = 2; '
                          + declaration + gate + f'{operand}[index] ^= "1"; ' + gate)
                # Fractional powers use each gate's principal eigenphase;
                # the second angle (11*pi/8) wraps to -5*pi/8.
                phase = (-2 if factor == 0.5 else 14) * math.pi / 8
                expected = tuple(cmath.exp(1j * factor * phase) / 2 if active(i) else 0.5
                                 for i in range(4))
                for fused in (False, True):
                    BackendAuditTests.check(self, self.program(source, fused=fused), expected)


    def test_documented_runtime_angle_bit_selection_example(self):
        import cmath
        import math
        from qasm2qcx_u_power_numerical import BackendAuditTests
        document = (pathlib.Path(__file__).parents[1] / 'qcx/qasm2qcx.md').read_text(encoding='utf-8')
        examples = re.findall(r'<!-- runtime-angle-bit-selection-example -->\s*```qasm\n(.*?)\n```',
                              document, flags=re.DOTALL)
        self.assertEqual(len(examples), 1)
        self.check_initializers(examples[0].removeprefix('OPENQASM 3.0;'),
                                (('old', 1), ('phases', 1)), (0, 2),
                                ('a', 'bit_index', 'element_index'), (11, 0, -1))
        expected = (1 / math.sqrt(2), cmath.exp(1j * math.pi / 4) / math.sqrt(2))
        for fused in (False, True):
            BackendAuditTests.check(self, self.program(examples[0], fused=fused), expected)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra', type=pathlib.Path)
    arguments = parser.parse_args()
    BackendTests.bra = arguments.bra.resolve() if arguments.bra else None
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(BackendTests))
    print(f'runtime angle bit-selection programs checked: {BackendTests.checked}')
    raise SystemExit(not result.wasSuccessful())
