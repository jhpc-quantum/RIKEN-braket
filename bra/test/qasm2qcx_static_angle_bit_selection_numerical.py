#!/usr/bin/env python3
"""Audit static bit selections on sized angles (#422).

Pass --bra /path/to/bra for the default-double non-MPI backend audit.
"""

import argparse
import pathlib
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra', type=pathlib.Path)
    arguments = parser.parse_args()
    BackendTests.bra = arguments.bra.resolve() if arguments.bra else None
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(BackendTests))
    print(f'static angle bit-selection programs checked: {BackendTests.checked}')
    raise SystemExit(not result.wasSuccessful())
