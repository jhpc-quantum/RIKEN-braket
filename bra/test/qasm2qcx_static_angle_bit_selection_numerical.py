#!/usr/bin/env python3
"""Audit static bit selections on sized angles (#422).

Pass --bra /path/to/bra for the default-double non-MPI backend audit.
"""

import argparse
import cmath
import math
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

    @staticmethod
    def bits(value, indices):
        return [(value >> index) & 1 for index in indices]

    @staticmethod
    def replace_bits(value, indices, bits):
        for index, bit in zip(indices, bits):
            value = (value & ~(1 << index)) | (bit << index)
        return value

    def test_constant_variable_and_array_reads_at_every_width(self):
        native = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        for width in range(1, native + 1):
            mask = (1 << width) - 1
            for raw in sorted({0, 1, 1 << (width - 1), mask}):
                declarations = (f'const angle[{width}] A = angle[{width}]("{raw:0{width}b}"); '
                                f'angle[{width}] value = A; array[angle[{width}], 2] phases = {{A, -A}}; '
                                'int index = 0; ')
                selectors = (('0', (0,)), ('-1', (width - 1,)),
                             (f'0:{width - 1}', tuple(range(width))),
                             (f'{width - 1}:-1:0', tuple(reversed(range(width)))),
                             (f'{{{width - 1}, 0, {width - 1}}}', (width - 1, 0, width - 1)))
                outputs, expected = [], []
                for source_index, source in enumerate(('A', 'value', 'phases[index]')):
                    for index, (selector, indices) in enumerate(selectors):
                        name = f'out{source_index}_{index}'
                        const = 'const ' if source == 'A' else ''
                        declarations += f'{const}bit[{len(indices)}] {name} = {source}[{selector}]; '
                        # Materialize constant registers to inspect their values in QCX.
                        if const:
                            declarations += f'bit[{len(indices)}] copy{index} = {name}; '
                            name = f'copy{index}'
                        outputs.append((name, len(indices)))
                        expected.extend(self.bits(raw, indices))
                with self.subTest(width=width, raw=raw):
                    self.check_initializers(declarations, tuple(outputs) + (('phases', 2),),
                                            expected + [raw, -raw & mask], ('value',), (raw,))

    def test_array_indices_and_singleton_storage(self):
        for index in (-2, -1, 0, 1):
            raw = 3 if index in (-2, 0) else 10
            self.check_initializers('array[angle[4], 2] phases = {angle[4]("0011"), angle[4]("1010")}; '
                f'int index = {index}; bit[4] selected = phases[index][3:-1:0]; '
                'bit[1] one = phases[index][0];',
                (('selected', 4), ('one', 1), ('phases', 2)), self.bits(raw, (3, 2, 1, 0)) + [raw & 1, 3, 10])
        self.check_initializers('array[angle[1], 1] phases = {pi}; int index = -1; '
            'bit[1] one = phases[index][0]; angle[1] roundtrip = angle[1](one);',
            (('one', 1), ('phases', 1)), (1, 1), ('roundtrip',), (1,))

    def test_register_operators_functions_comparisons_and_angle_roundtrips(self):
        self.check_initializers('const angle[4] A = angle[4]("1010"); angle[4] value = A; '
            'bit[4] selected = ~value[3:-1:0] ^ "0011"; '
            'bit[2] pair = rotl(value[0:1], -1); pair <<= 1; '
            'uint count = popcount(value[{3, 0, 3}]); '
            'bool same = value[0:1] == A[0:1]; bool different = value[0:1] != "11"; '
            'angle[4] reversed = angle[4](value[3:-1:0]); '
            'value = angle[4](value[0:3]);',
            (('selected', 4), ('pair', 2)), (1, 0, 0, 1, 0, 1),
            ('count', 'same', 'different', 'reversed', 'value'), (2, 1, 1, 5, 10))
        self.check_initializers('const angle[4] A = angle[4]("1010"); '
            'const bit[2] B = ~A[{3, 0}] ^ "01"; const uint COUNT = popcount(A[0:3]); '
            'const bool SAME = A[0:1] == "10"; const angle[2] C = angle[2](A[2:3]); '
            'bit[2] out = B; uint count = COUNT; bool same = SAME; angle[2] value = C;',
            (('out', 2),), (1, 1), ('count', 'same', 'value'), (2, 1, 2))

    def test_reads_in_loops_reentry_and_iterator_shadowing(self):
        for outer in ('for int i in [0:2]', 'for int i in [0:stop]', 'while (visits < 3)'):
            self.check_initializers('const angle[4] value = pi; '
                'array[angle[4], 2] phases = {angle[4]("0011"), angle[4]("0101")}; '
                'bit[2] out = "00"; uint count = 0; int visits = 0; int stop = 2; ' + outer +
                ' { visits += 1; for angle[4] value in phases { out = value[0:1]; '
                'count += popcount(out); }} count += popcount(value[-1]);',
                (('out', 2), ('phases', 2)), (1, 0, 3, 5), ('count', 'visits'), (10, 3))

    def test_skipped_reads_do_not_execute_bounds_or_index_arithmetic(self):
        prefix = 'array[angle[4], 1] phases = {pi}; int bad = 1; int zero = 0; bit[2] out = "11"; '
        self.check_initializers(prefix + 'if (false) { out = phases[bad][0:1]; } '
            'while (false) { out = phases[1 / zero][0:1]; } '
            'for int i in [1:0] { out = phases[bad][0:1]; } '
            'bool yes = true || phases[bad][0:1] == "00"; '
            'bool no = false && phases[1 / zero][0:1] != "00";',
            (('out', 2), ('phases', 1)), (1, 1, 8), ('yes', 'no'), (1, 0))

    def test_reached_bounds_and_index_arithmetic_fail_without_ubsan(self):
        for expression in ('phases[bad][0:1]', 'phases[1 / zero][0:1]'):
            lines = self.program('array[angle[4], 1] phases = {pi}; int bad = 1; int zero = 0; '
                                 'bit[2] out = ' + expression + ';')
            result = self.execute(lines)
            self.assertNotEqual(result.returncode, 0, expression)
            self.assertNotIn('runtime error:', result.stderr)
            type(self).checked += 1

    def test_writes_preserve_unselected_bits_and_overlaps_at_every_width(self):
        native = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        for width in range(1, native + 1):
            mask = (1 << width) - 1
            for raw in sorted({0, 1, 1 << (width - 1), mask}):
                sparse = (width - 1, 0) if width > 1 else (0,)
                selectors = (('0', (0,)), ('-1', (width - 1,)),
                             (f'0:{width - 1}', tuple(range(width))),
                             (f'{width - 1}:-1:0', tuple(reversed(range(width)))),
                             ('{' + ', '.join(map(str, sparse)) + '}', sparse),
                             (f'0:2:{width - 1}', tuple(range(0, width, 2))))
                source = (f'const angle[{width}] A = angle[{width}]("{raw:0{width}b}"); '
                          f'angle[{width}] value = A; angle[{width}] other = ~A; '
                          f'array[angle[{width}], {4 * len(selectors)}] results; ')
                expected = []
                for number, (selector, indices) in enumerate(selectors):
                    pattern = [index % 2 for index in range(len(indices))]
                    literal = '"' + ''.join(map(str, reversed(pattern))) + '"'
                    reverse = tuple(reversed(indices))
                    reverse_selector = '{' + ', '.join(map(str, reverse)) + '}'
                    source += f'bit[{len(indices)}] input{number} = {literal}; '
                    old_bits = self.bits(raw, reverse)
                    other_bits = self.bits(raw ^ mask, indices)
                    complex_bits = [1 ^ lhs ^ rhs for lhs, rhs in zip(old_bits, other_bits)]
                    for rhs, bits in ((literal, pattern), (f'input{number}', pattern),
                                      (f'value[{reverse_selector}]', old_bits),
                                      (f'~value[{reverse_selector}] ^ other[{selector}]', complex_bits)):
                        source += f'value = A; value[{selector}] = {rhs}; results[{len(expected)}] = value; '
                        expected.append(self.replace_bits(raw, indices, bits))
                with self.subTest(width=width, raw=raw):
                    self.check_initializers(source, (('results', len(expected)),), expected,
                                            ('value', 'other'), (expected[-1], raw ^ mask))

    def test_array_writes_negative_indices_singletons_and_dependent_destinations(self):
        for index in (-2, -1, 0, 1):
            target = index % 2
            values = [5, 10]
            values[target] = self.replace_bits(values[target], (0, 1, 2), self.bits(values[1 - target], (3, 1, 0)))
            self.check_initializers('array[angle[4], 2] phases = {angle[4]("0101"), angle[4]("1010")}; '
                f'int index = {index}; int other = {1 - target}; '
                'phases[index][0:2] = phases[other][{3, 1, 0}];', (('phases', 2),), values)
        self.check_initializers('array[angle[1], 1] phases = {0.0}; int index = -1; '
            'phases[index][0] = "1"; bit[1] out = phases[index][0];',
            (('phases', 1), ('out', 1)), (1, 1))
        self.check_initializers('array[angle[4], 2] phases = {angle[4]("0001"), angle[4]("0011")}; '
            'phases[popcount(phases[0][0:3]) - uint(1)][0:3] = phases[1][0:3];',
            (('phases', 2),), (3, 3))

    def test_overlapping_writes_in_reentered_and_nested_loops(self):
        values = [3, 10]
        for _ in range(3):
            for index in range(2):
                values[index] = self.replace_bits(values[index], (0, 1, 2), self.bits(values[index], (1, 2, 3)))
        self.check_initializers('array[angle[4], 2] phases = {angle[4]("0011"), angle[4]("1010")}; '
            'int visits = 0; while (visits < 3) { visits += 1; '
            'for int index in [0:1] { phases[index][0:2] = phases[index][1:3]; }}',
            (('phases', 2),), values, ('visits',), (3,))
        self.check_initializers('array[angle[4], 3] phases = {0.0, pi / 2, pi}; '
            'int index = 0; for angle[4] value in phases { '
            'if (value == angle[4](0.0)) { index += 1; continue; } '
            'phases[index][0:3] = value[3:-1:0]; index += 1; if (index == 2) { break; }}',
            (('phases', 3),), (0, 2, 8), ('index',), (2,))

    def test_write_checks_execute_only_on_reached_paths(self):
        prefix = 'array[angle[4], 1] phases = {pi}; int bad = 1; int zero = 0; '
        for body in ('phases[bad][0:1] = "11";', 'phases[0][0:1] = phases[bad][0:1];',
                     'phases[1 / zero][0:1] = "11";', 'phases[0][0:1] = phases[1 / zero][0:1];'):
            for wrapper in ('if (false) {{ {} }}', 'while (false) {{ {} }}',
                            'for int outer in [1:0] {{ {} }}'):
                self.check_initializers(prefix + wrapper.format(body), (('phases', 1),), (8,))
            result = self.execute(self.program(prefix + body))
            self.assertNotEqual(result.returncode, 0, body)
            self.assertNotIn('runtime error:', result.stderr)
            type(self).checked += 1

    def test_gate_parameters_capture_values_before_subsequent_selection_writes(self):
        from qasm2qcx_u_power_numerical import BackendAuditTests
        phase = 9 * math.pi / 8
        operations = (
            ('p(value) q[0];', 1, lambda index: bool(index & 1)),
            ('inv @ p(value) q[0];', -1, lambda index: bool(index & 1)),
            ('pow(exponent) @ p(value) q[0];', 2, lambda index: bool(index & 1)),
            ('pow(0.5) @ p(value) q[0];', 0.5, lambda index: bool(index & 1)),
            ('cp(value) q[0], q[1];', 1, lambda index: index == 3),
            ('negctrl @ p(value) q[0], q[1];', 1, lambda index: index == 2),
            ('gphase(value);', 1, lambda index: True),
        )
        for operation, factor, active in operations:
            source = ('include "stdgates.inc"; qubit[2] q; h q; int exponent = 2; '
                      'angle[4] value = angle[4]("0011"); ' + operation +
                      'value[0:3] = "0110"; ' + operation + 'value[0:3] = "0000";')
            expected = tuple(cmath.exp(1j * factor * phase) / 2 if active(index) else 0.5 for index in range(4))
            for fused in (False, True):
                with self.subTest(operation=operation, fused=fused):
                    BackendAuditTests.check(self, self.program(source, fused=fused), expected)

    def test_documented_static_angle_bit_selection_example(self):
        from qasm2qcx_u_power_numerical import BackendAuditTests
        document = (pathlib.Path(__file__).parents[1] / 'qcx/qasm2qcx.md').read_text(encoding='utf-8')
        examples = re.findall(r'<!-- static-angle-bit-selection-example -->\s*```qasm\n(.*?)\n```',
                              document, flags=re.DOTALL)
        self.assertEqual(len(examples), 1)
        source = examples[0].removeprefix('OPENQASM 3.0;')
        self.check_initializers(source, (('phases', 2), ('low', 1), ('upper', 2)), (0, 12, 0, 1, 0),
                                ('a', 'index'), (3, -1))
        expected = (1 / math.sqrt(2), -1j / math.sqrt(2))
        for fused in (False, True):
            BackendAuditTests.check(self, self.program(examples[0], fused=fused), expected)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra', type=pathlib.Path)
    arguments = parser.parse_args()
    BackendTests.bra = arguments.bra.resolve() if arguments.bra else None
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(BackendTests))
    print(f'static angle bit-selection programs checked: {BackendTests.checked}')
    raise SystemExit(not result.wasSuccessful())
