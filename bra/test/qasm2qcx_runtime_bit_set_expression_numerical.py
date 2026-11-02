#!/usr/bin/env python3
"""Audit runtime bit-register index-set expression snapshots (#442)."""

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
    program = arrays.BackendTests.program
    execute = arrays.BackendTests.execute
    check_initializers = arrays.BackendTests.check_initializers

    def test_wide_mutable_sources_preserve_selection_order_repeats_and_negative_aliases(self):
        for width in (1, 2, 4, 32, 65, 129):
            for bits in sorted({'0' * width, '1' * width, ('101100' * width)[:width]}):
                for position in sorted({0, width // 2, width - 1}):
                    indices = (position, width - 1, width - 1, position, position)
                    expected = tuple(int(bits[-1 - index]) for index in indices)
                    selected = ''.join(str(bit) for bit in reversed(expected))
                    source = (f'bit[{width}] flags = "{bits}"; int i = {position}; uint j = {width - 1}; '
                              f'int negative = {position - width}; bit[5] selected = flags[{{i, j, -1, negative, i}}]; '
                              f'bool same = selected == "{selected}" && flags == "{bits}";')
                    self.check_initializers(source, (('selected', 5),), expected, ('same',), (1,))

    def test_singletons_keep_register_semantics_and_each_access_has_its_own_snapshot(self):
        for width in (1, 4, 65):
            self.check_initializers(f'bit[{width}] flags = "' + '1' * width + '"; int index = -1; '
                'bit[1] before = flags[{index}]; flags[index] = false; '
                'bit[1] after = flags[{index}]; bool first = before == "1"; bool second = after == "0";',
                (), (), ('first', 'second'), (1, 1))

    def test_constant_sources_at_every_bit_across_full_and_partial_native_words(self):
        word = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        for width in (1, word - 1, word, word + 1, 2 * word - 1, 2 * word, 2 * word + 1, 129):
            for bits in ('0' * width, '1' * width, ('10010110' * width)[:width]):
                checksum = 0
                for index in range(width):
                    positions = (index, width - 1 - index, index, 0, width - 1)
                    values = [int(bits[-1 - position]) for position in positions]
                    checksum = (checksum * 3 + sum(values)) % 997
                last_bits = sum(bit << index for index, bit in enumerate(values))
                self.check_initializers(f'const bit[{width}] C = "{bits}"; int i = 0; uint j = {width - 1}; '
                    f'bit[5] selected; angle[5] phase; uint checksum = 0; while (i < {width}) {{ '
                    'selected = C[{i, j, i, 0, -1}]; checksum = (checksum * 3 + popcount(selected)) % 997; '
                    'phase = angle[5](C[{i, j, i, 0, -1}]); i += 1; if (j > 0) { j -= 1; } } '
                    f'bool same = C == "{bits}";', (), (), ('checksum', 'phase', 'same'), (checksum, last_bits, 1))

    def test_rotation_population_count_and_angle_casts_share_snapshot_consumers(self):
        limits = self.module.QASM2QCXConverter
        for constant in (False, True):
            for width in (1, 4, 65, 129):
                bits = ('101100' * width)[:width]
                positions = (0, width - 1, 0, 0, width - 1)
                raw = sum(int(bits[-1 - position]) << index for index, position in enumerate(positions))
                for distance in (limits.QCX_INT_MIN, -7, 0, 1, 6, limits.QCX_INT_MAX):
                    shift = distance % 5
                    left = ((raw << shift) | (raw >> (5 - shift))) & 31
                    right = ((raw >> shift) | (raw << (5 - shift))) & 31
                    prefix = ('const ' if constant else '') + f'bit[{width}] flags = "{bits}"; int i = 0; '
                    self.check_initializers(prefix + f'int distance = {distance}; '
                        'bit[5] left = rotl(flags[{i, -1, i, 0, -1}], distance); '
                        'bit[5] right = rotr(flags[{i, -1, i, 0, -1}], distance); '
                        'uint ones = popcount(flags[{i, -1, i, 0, -1}]); '
                        'angle[5] phase = angle[5](flags[{i, -1, i, 0, -1}]); '
                        f'bool same = left == "{left:05b}" && right == "{right:05b}";',
                        (), (), ('ones', 'phase', 'same'), (raw.bit_count(), raw, 1))

    def test_nested_constant_and_mutable_selections_preserve_table_and_result_ownership(self):
        self.check_initializers('const bit[65] C = "1' + '0' * 63 + '1"; '
            'const bit[65] D = "0' + '1' * 63 + '0"; bit[4] flags = "1011"; int i = 0; '
            'bit[2] mixed = C[{i, -1}] ^ D[{i, 1}]; '
            'bit[2] nested = C[{popcount(D[{i, 1}]), i}]; '
            'bit[2] from_live = D[{popcount(flags[{i, i}]), i}]; '
            'bool same = mixed == "01" && nested == "10" && from_live == "01";',
            (), (), ('same',), (1,))
        self.check_initializers('const bit[65] C = "1' + '0' * 63 + '1"; int i = 0; bit[2] pair; uint total = 0; '
            'for int i in [0:2] { pair = C[{i, -1}]; total += popcount(pair); } '
            'pair = C[{i, -1}]; total += popcount(pair);', (), (), ('total', 'i'), (6, 0))

    def test_constant_selected_checks_execute_only_on_reached_paths(self):
        limits = self.module.QASM2QCXConverter
        for width in (1, 33, 129):
            for typename, bad in (('int', -width - 1), ('int', width), ('uint', limits.QCX_UINT_MAX)):
                prefix = (f'const bit[{width}] C = "' + '0' * width + '"; int i = 0; '
                          f'{typename} bad = {bad}; int zero = 0; bit[2] pair = "11";')
                for entries in ('i, bad', 'bad, i', 'i, 999', 'i, 1 / zero'):
                    statement = f'pair = C[{{{entries}}}];'
                    for wrapper in ('if (false) {{ {} }}', 'while (false) {{ {} }}', 'for int outer in [1:0] {{ {} }}'):
                        self.check_initializers(prefix + wrapper.format(statement) + 'bool same = pair == "11";',
                                                (), (), ('same',), (1,))
                    result = self.execute(self.program(prefix + statement))
                    self.assertNotEqual(result.returncode, 0)
                    self.assertNotIn('runtime error:', result.stderr)
                    if 'zero' not in entries and (typename == 'int' or 'bad' not in entries):
                        self.assertIn('assertion_error', result.stderr)
                    type(self).checked += 1

    def test_overlapping_whole_static_destination_and_compound_assignments(self):
        self.check_initializers('bit[4] flags = "1010"; int i = 3; uint j = 1; '
            'flags = flags[{i, j, 2, 0}]; bool same = flags == "0011";', (), (), ('same',), (1,))
        self.check_initializers('bit[4] flags = "1010"; int i = 1; '
            'flags[0:2] = flags[{i, 0, i}]; bool same = flags == "1101";', (), (), ('same',), (1,))
        for op in ('&', '|', '^'):
            before = 0b1010
            indices = (3, 1, 2, 0)
            selected = sum(((before >> index) & 1) << target for target, index in enumerate(indices))
            expected = {'&': before & selected, '|': before | selected, '^': before ^ selected}[op]
            self.check_initializers('bit[4] flags = "1010"; int i = 3; uint j = 1; '
                f'flags {op}= flags[{{i, j, 2, 0}}]; bool same = flags == "{expected:04b}";',
                (), (), ('same',), (1,))

    def test_nested_bitwise_shifts_equality_and_short_circuiting(self):
        for raw in range(16):
            for count in (0, 1, 4, 5, self.module.QASM2QCXConverter.QCX_UINT_MAX):
                left = sum(((raw >> index) & 1) << target for target, index in enumerate((3, 0, 3, 1)))
                right = sum(((raw >> index) & 1) << target for target, index in enumerate((1, 2, 0, 3)))
                expected = (((~left & 15) ^ right) << count) & 15 if count < 4 else 0
                self.check_initializers(f'bit[4] flags = "{raw:04b}"; int i = 3; uint shift = {count}; '
                    'bit[4] result = (~flags[{i, 0, i, 1}] ^ flags[{1, 2, 0, i}]) << shift; '
                    f'bool same = result == "{expected:04b}"; int bad = 4; '
                    'bool safe = true || flags[{bad}] == "1"; bool safe_and = false && flags[{bad}] == "1";',
                    (), (), ('same', 'safe', 'safe_and'), (1, 1, 0))

    def test_nested_index_expressions_and_loop_iterator_dependencies(self):
        self.check_initializers('bit[4] flags = "1010"; int i = 0; array[int, 2] positions = {-1, 0}; '
            'bit[3] result = flags[{positions[i], int(flags[i]) + 1, positions[i + 1]}]; '
            'bool same = result == "011";', (), (), ('same',), (1,))
        self.check_initializers('bit[4] flags = "1011"; const int LOW = 0; uint total = 0; '
            'for int LOW in [0:3] { total += popcount(flags[{LOW, 0}]); } '
            'int i = 0; while (i < 4) { total += popcount(flags[{i, i}]); i += 1; }',
            (), (), ('total',), (13,))
        self.check_initializers('bit[4] flags = "1011"; bit[2] result; uint total = 0; int i = 0; '
            'for bit b in flags { result = flags[{int(b), i}]; total += popcount(result); i += 1; }',
            (), (), ('total',), (7,))

    def test_bounds_fail_before_destination_writes_and_only_when_reached(self):
        limits = self.module.QASM2QCXConverter
        for typename, bad in (('int', -5), ('int', 4), ('int', limits.QCX_INT_MIN),
                               ('int', limits.QCX_INT_MAX), ('uint', limits.QCX_UINT_MAX)):
            prefix = f'bit[4] flags = "1011"; int i = 0; {typename} bad = {bad}; int zero = 0; uint visits = 0;'
            for entries in ('i, bad', 'bad, i', 'i, 999', '-999, i'):
                statement = f'flags[0:1] = flags[{{{entries}}}]; visits = 1 / zero;'
                for wrapper in ('if (false) {{ {} }}', 'while (false) {{ {} }}', 'for int outer in [1:0] {{ {} }}'):
                    self.check_initializers(prefix + wrapper.format(statement) + 'bool same = flags == "1011";',
                                            (), (), ('same', 'visits'), (1, 0))
                result = self.execute(self.program(prefix + statement))
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn('runtime error:', result.stderr)
                self.assertNotIn('division', result.stderr.lower())
                if typename == 'int' or 'bad' not in entries:
                    self.assertIn('assertion_error', result.stderr)
                type(self).checked += 1

    def test_skipped_arithmetic_and_nested_array_access_failures(self):
        prefix = 'bit[4] flags = "1011"; int i = 0; int zero = 0; int bad = 1; array[int, 1] positions; bit[2] result;'
        for entries in ('i, 1 / zero', 'positions[bad], i', '1 / 0, i'):
            statement = f'result = flags[{{{entries}}}];'
            self.check_initializers(prefix + 'if (false) {' + statement + '} bool same = flags == "1011";',
                                    (), (), ('same',), (1,))
            result = self.execute(self.program(prefix + statement))
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn('runtime error:', result.stderr)
            type(self).checked += 1

    def test_documented_runtime_bit_set_expression_example(self):
        import cmath
        import math
        from qasm2qcx_u_power_numerical import BackendAuditTests
        document = (pathlib.Path(__file__).parents[1] / 'qcx/qasm2qcx.md').read_text(encoding='utf-8')
        examples = re.findall(r'<!-- runtime-bit-set-expression-example -->\s*```qasm\n(.*?)\n```',
                              document, flags=re.DOTALL)
        self.assertEqual(len(examples), 1)
        source = examples[0].removeprefix('OPENQASM 3.0;')
        source += '\nbool same = flags == "1101" && saved == "111" && after == "011" && pair == "11";'
        source += '\nbool unchanged = C == "1' + '0' * 63 + '1";'
        self.check_initializers(source, (), (), ('i', 'j', 'ones', 'phase', 'same', 'unchanged'), (0, 1, 2, 3, 1, 1))
        expected = (1 / math.sqrt(2), cmath.exp(1j * math.pi / 4) / math.sqrt(2))
        for fused in (False, True):
            BackendAuditTests.check(self, self.program(examples[0], fused=fused), expected)

    def test_selected_snapshots_in_fused_phase_gate_conditions(self):
        import cmath
        import math
        from qasm2qcx_u_power_numerical import BackendAuditTests
        source = ('include "stdgates.inc"; qubit q; h q; bit[4] flags = "1001"; int i = 0; '
                  'bit[2] saved = flags[{i, -1}]; flags[0] = false; i = 1; '
                  'if (saved == "11" && flags[{i, 0}] == "00") { p(pi / 4) q; }')
        expected = (1 / math.sqrt(2), cmath.exp(1j * math.pi / 4) / math.sqrt(2))
        for fused in (False, True):
            BackendAuditTests.check(self, self.program(source, fused=fused), expected)
        source = ('include "stdgates.inc"; qubit q; h q; const bit[65] C = "1' + '0' * 63 + '1"; '
                  'int i = -1; bit[2] saved = C[{i, i}]; i = 0; '
                  'if (saved == "11" && C[{i, 1}] == "01") { p(pi / 4) q; }')
        for fused in (False, True):
            BackendAuditTests.check(self, self.program(source, fused=fused), expected)


def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    args = parser.parse_args()
    BackendTests.bra = args.bra.resolve()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(BackendTests))
    print(f'runtime bit-set expression programs checked: {BackendTests.checked}')
    raise SystemExit(not result.wasSuccessful())


if __name__ == '__main__':
    main()
