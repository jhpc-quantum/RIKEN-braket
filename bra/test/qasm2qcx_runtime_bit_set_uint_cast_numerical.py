#!/usr/bin/env python3
"""Audit UINT packing from runtime bit-register index sets (#470)."""

import argparse
import pathlib
import resource
import unittest

import qasm2qcx_angle_bit_cast_numerical as casts


class BackendTests(unittest.TestCase):
    bra = None
    checked = 0
    setUp = casts.BackendTests.setUp
    execute = casts.BackendTests.execute
    program = casts.BackendTests.program
    check_exact = casts.BackendTests.check_exact

    def test_every_native_width_preserves_order_repeats_and_high_result_bits(self):
        native = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        for width in range(1, native + 1):
            source_width = native + 3
            positions = [(i * 7 + 1) % source_width for i in range(width)]
            entries = ','.join('index' if i == 0 else str(position)
                               for i, position in enumerate(positions))
            for pattern in (0, (1 << source_width) - 1,
                            int(('10110010' * source_width)[:source_width], 2)):
                expected = sum(((pattern >> position) & 1) << i for i, position in enumerate(positions))
                bits = f'{pattern:0{source_width}b}'
                for constant in (False, True):
                    declaration = ('const ' if constant else '') + f'bit[{source_width}] flags = "{bits}"; '
                    source = declaration + f'int index = {positions[0]}; '
                    source += f'uint[{width}] packed = uint[{width}](flags[{{{entries}}}]); '
                    source += f'bit[{width}] back = bit[{width}](packed); bool same = flags == "{bits}";'
                    self.check_exact(source, (('packed', None), ('back', width), ('same', None)),
                                     (expected, expected, 1))

    def test_unsized_uint_and_singleton_register_sources(self):
        native = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        for bit in (0, 1):
            entries = ','.join(['index'] * native)
            self.check_exact(f'bit[1] flag = "{bit}"; int index = -1; '
                f'uint packed = uint(flag[{{{entries}}}]); uint[1] single = uint[1](flag[{{index}}]); '
                f'bit[{native}] back = bit[{native}](packed);',
                (('packed', None), ('single', None), ('back', native), ('flag', 1)),
                (bit * ((1 << native) - 1), bit, bit * ((1 << native) - 1), bit))

    def test_signed_unsigned_indices_negative_aliases_and_array_expressions(self):
        for constant in (False, True):
            for size in (1, 4, 33, 65, 129):
                bits = ('10110010' * size)[:size]
                positions = (size - 1, 0, size - 1, size - 1, 0)
                expected = sum(int(bits[-1 - position]) << i for i, position in enumerate(positions))
                self.check_exact(('const ' if constant else '') + f'bit[{size}] flags = "{bits}"; '
                    f'int negative = -1; uint positive = {size - 1}; array[int,2] positions = {{-{size},-1}}; '
                    'int slot = 0; uint[5] packed = uint[5](flags[{negative,positions[slot],positive,-1,0}]); '
                    f'bool same = flags == "{bits}";', (('packed', None), ('same', None)), (expected, 1))

    def test_constant_sources_cross_full_and_partial_word_boundaries(self):
        word = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        for size in (word - 1, word, word + 1, 2 * word - 1, 2 * word, 2 * word + 1, 129):
            bits = ('10010110' * size)[:size]
            for position in sorted({0, size - 1, min(word - 1, size - 1), min(word, size - 1)}):
                indices = (position, size - 1, position, 0)
                expected = sum(int(bits[-1 - index]) << i for i, index in enumerate(indices))
                self.check_exact(f'const bit[{size}] C = "{bits}"; uint index = {position}; '
                    'uint[4] packed = uint[4](C[{index,-1,index,0}]); '
                    f'bool same = C == "{bits}";', (('packed', None), ('same', None)), (expected, 1))

    def test_overlapping_source_assignment_and_independent_snapshots(self):
        for bits in ('0000', '0001', '1101', '1111'):
            reverse = int(bits[::-1], 2)
            self.check_exact(f'bit[4] flags = "{bits}"; int index = 3; '
                'uint[4] saved = uint[4](flags[{index,2,1,0}]); '
                'flags = bit[4](uint[4](flags[{index,2,1,0}])); '
                'index = 0; flags[index] = false; '
                'uint[4] after = uint[4](flags[{index,1,2,3}]);',
                (('saved', None), ('after', None), ('flags', 4)), (reverse, reverse & ~1, reverse & ~1))

    def test_all_index_checks_precede_selected_reads_and_destination_writes(self):
        limits = self.module.QASM2QCXConverter
        cases = ((f'int bad = {limits.QCX_INT_MIN};', 'index,bad,0,1', 'assertion failed'),
                 (f'int bad = {limits.QCX_INT_MAX};', 'index,0,1,bad', 'assertion failed'),
                 (f'uint bad = {(1 << limits.QCX_UINT_WIDTH) - 1};', 'index,bad,0,1', 'UINT to INT conversion out of range'),
                 ('int bad = -9;', 'index,bad,0,1', 'assertion failed'),
                 ('int bad = 8;', 'index,0,1,bad', 'assertion failed'),
                 ('int zero = 0;', 'index,0,1,index / zero', 'division by zero'),
                 ('int zero = 0;', 'index,0,1,1 / zero', 'division by zero'),
                 ('', 'index,999,0,1', 'assertion failed'),
                 ('', '-999,index,0,1', 'assertion failed'),
                 ('', 'index,0,1,999999999999999999999999999999', 'assertion failed'))
        for constant in (False, True):
            for declarations, entries, error in cases:
                with self.subTest(constant=constant, entries=entries, declarations=declarations):
                    source = ('const ' if constant else '') + 'bit[8] flags = "11010011"; '
                    source += f'int index = 0; uint[4] packed = 6; {declarations} '
                    program = self.module.openqasm3.parser.parse('OPENQASM 3; ' + source +
                                                               f'packed = uint[4](flags[{{{entries}}}]);')
                    converter = self.module.QASM2QCXConverter(program)
                    converter.visit(program)
                    self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())
                    name = converter._QASM2QCXConverter__capitalize_variable_name('packed')
                    flags = converter._QASM2QCXConverter__capitalize_variable_name('flags')
                    instrumented = []
                    for line in converter:
                        if f' := {flags}:QASM2QCX_' in line or ' := :INT:QASM2QCX_UINT_' in line:
                            instrumented.append('PRINTLN 22222')
                        if line.startswith(f'LET {name} := QASM2QCX_'):
                            instrumented.append('PRINTLN 11111')
                        instrumented.append(line)
                    self.assertIn('PRINTLN 22222', instrumented)
                    self.assertIn('PRINTLN 11111', instrumented)
                    result = self.execute(instrumented)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(error, result.stderr)
                    self.assertNotIn('runtime error:', result.stderr)
                    self.assertNotIn('22222', result.stdout)
                    self.assertNotIn('11111', result.stdout)
                    type(self).checked += 1


def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    args = parser.parse_args()
    BackendTests.bra = args.bra.resolve()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(BackendTests))
    print(f'runtime bit-set UINT cast programs checked: {BackendTests.checked}')
    raise SystemExit(not result.wasSuccessful())


if __name__ == '__main__':
    main()
