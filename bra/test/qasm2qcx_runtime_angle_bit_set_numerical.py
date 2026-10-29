#!/usr/bin/env python3
"""Audit runtime-valued angle bit-set selections (#430)."""

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

    def test_reads_and_iteration_at_every_width(self):
        for width in range(1, self.module.QASM2QCXConverter.QCX_UINT_WIDTH + 1):
            mask = (1 << width) - 1
            for raw in sorted({0, mask, 1 << (width - 1), 0xAAAAAAAA & mask}):
                source = (f'const angle[{width}] C = angle[{width}]("{raw:0{width}b}"); '
                          f'angle[{width}] a = C; array[angle[{width}], 1] phases = {{C}}; '
                          f'int element = -1; int index = -1; uint count = 0; int mid = {width // 2}; ')
                indices = (width - 1, 0, width // 2, width - 1, width - 1)
                bits = [(raw >> index) & 1 for index in indices]
                checksum = 0
                for bit in bits:
                    checksum = checksum * 3 + bit
                outputs, expected, scalars, scalar_values = [], [], [], []
                for number, operand in enumerate(('C', 'a', 'phases[element]')):
                    source += (f'bit[5] selected{number} = {operand}[{{index, count, mid, -1, index}}]; '
                               f'bit[1] single{number} = {operand}[{{index}}]; '
                               f'uint ones{number} = 0; int visits{number} = 0; uint checksum{number} = 0; '
                               f'for bit b in {operand}[{{index, count, mid, -1, index}}] {{ '
                               f'ones{number} += uint(b); visits{number} += 1; '
                               f'checksum{number} = checksum{number} * 3 + uint(b); }}')
                    outputs.extend(((f'selected{number}', 5), (f'single{number}', 1)))
                    expected.extend(bits + [(raw >> (width - 1)) & 1])
                    scalars.extend((f'ones{number}', f'visits{number}', f'checksum{number}'))
                    scalar_values.extend((sum(bits), 5, checksum))
                self.check_initializers(source, tuple(outputs) + (('phases', 1),), expected + [raw],
                                        tuple(scalars) + ('a',), scalar_values + [raw])

    def test_register_consumers_and_nested_integer_index_expressions(self):
        self.check_initializers('angle[4] a = angle[4]("1010"); const angle[4] C = pi; '
            'array[angle[4], 1] phases = {angle[4]("0011")}; int element = -1; int index = -1; uint count = 0; '
            'bit[2] selected = a[{index, count}]; bit[2] inverse = ~a[{index, count}]; '
            'bit[2] combined = a[{index, count}] ^ "11"; bit[2] rotated = rotl(a[{index, count}], 1); '
            'bit[2] shifted = a[{index, count}] << 1; uint ones = popcount(a[{index, count}]); '
            'bool same = a[{index, count}] == "01"; angle[2] roundtrip = angle[2](a[{index, count}]); '
            'bit[2] nested = phases[element][{popcount(a[{index, count}]), count}]; '
            'bool short = true || a[{index, 100}] == "00";',
            (('selected', 2), ('inverse', 2), ('combined', 2), ('rotated', 2), ('shifted', 2), ('nested', 2)),
            (1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1),
            ('ones', 'same', 'roundtrip', 'short'), (1, 1, 1, 1))

    def test_selected_register_can_exceed_native_angle_width(self):
        for width in (1, 33, 65):
            selector = '{' + ', '.join(['index'] * width) + '}'
            self.check_initializers('const angle[4] C = pi; int index = -1; '
                f'bit[{width}] out = C[{selector}]; out >>= 1; uint total = 0; int visits = 0; '
                f'for bit b in C[{selector}] {{ total += uint(b); visits += 1; }}',
                (('out', width),), [1] * (width - 1) + [0], ('total', 'visits'), (width, width))

    def test_loops_capture_indices_but_read_bits_live_including_repetitions(self):
        self.check_initializers('angle[4] a = angle[4]("0001"); int index = 0; uint count = 1; '
            'uint ones = 0; uint checksum = 0; int visits = 0; '
            'for bit b in a[{index, count, index, -1}] { '
            'if (visits == 0) { index = 3; count = 0; a[1] = "1"; a[0] = "0"; } '
            'if (visits == 1) { a[3] = "1"; } '
            'ones += uint(b); checksum = checksum * 3 + uint(b); visits += 1; }',
            (), (), ('a', 'index', 'count', 'ones', 'checksum', 'visits'), (10, 3, 0, 3, 37, 4))
        self.check_initializers('array[angle[4], 2] phases = {angle[4]("0001"), pi}; '
            'int element = 0; int index = 0; uint count = 1; uint checksum = 0; int visits = 0; '
            'for bit b in phases[element][{index, count, -1}] { '
            'if (visits == 0) { element = 1; index = 3; count = 0; phases[0][1] = "1"; phases[0][3] = "1"; } '
            'checksum = checksum * 3 + uint(b); visits += 1; }',
            (('phases', 2),), (11, 8), ('element', 'index', 'count', 'checksum', 'visits'), (1, 3, 0, 13, 3))

    def test_loop_sources_and_selector_names_can_be_shadowed(self):
        for operand, iterator in (('a', 'a'), ('C', 'C'), ('phases[element]', 'phases'),
                                  ('phases[element]', 'element'), ('a', 'index'), ('a', 'count')):
            self.check_initializers('const angle[4] C = pi; angle[4] a = C; array[angle[4], 1] phases = {C}; '
                'int element = -1; int index = -1; uint count = 0; uint total = 0; int visits = 0; '
                f'for bit {iterator} in {operand}[{{index, count, index}}] {{ total += uint({iterator}); visits += 1; }}',
                (), (), ('total', 'visits'), (2, 3))

    def test_nested_loops_reentry_transfers_and_angle_iterators(self):
        self.check_initializers('array[angle[4], 2] phases = {angle[4]("0011"), angle[4]("1010")}; '
            'int passes = 0; uint total = 0; int visits = 0; '
            'while (passes < 2) { passes += 1; for int position in {-1, 0} { '
            'for angle[4] value in phases { for bit b in value[{position, 0}] { total += uint(b); visits += 1; }}}}',
            (), (), ('passes', 'total', 'visits'), (2, 8, 16))
        for transfer, expected in (('break', 1), ('continue', 4)):
            self.check_initializers('angle[4] a = angle[4]("1010"); int index = 0; uint count = 1; '
                'int visits = 0; uint skipped = 0; for bit b in a[{index, count, 0, -1}] { visits += 1; '
                + transfer + '; skipped += uint(b); } skipped += popcount(a[{index, count}]);',
                (), (), ('visits', 'skipped'), (expected, 1))

    def test_bad_entries_and_arithmetic_only_fail_on_reached_paths(self):
        maximum = self.module.QASM2QCXConverter.QCX_UINT_MAX
        minimum = self.module.QASM2QCXConverter.QCX_INT_MIN
        prefix = (f'angle[4] a = pi; array[angle[4], 1] phases = {{pi}}; int element = 0; '
                  f'int bad_element = 1; int index = 0; int bad = -5; uint huge = {maximum}; '
                  f'int minimum = {minimum}; int zero = 0; bit[2] out = "11"; int visits = 0; ')
        for operand in ('a[{index, 100}]', f'a[{{index, {2**200}}}]', f'a[{{index, {minimum}}}]',
                        'a[{index, huge}]', 'a[{bad, index}]', 'a[{index, 1 / zero}]',
                        'a[{1 / 0, index}]', 'phases[bad_element][{index, 0}]',
                        'phases[element][{index, minimum}]'):
            operations = (f'out = {operand};', f'for bit b in {operand} {{ visits += int(b); }}',
                          f'for bit b in {operand} {{ break; }}')
            for operation in operations:
                for wrapper in ('if (false) {{ {} }}', 'while (false) {{ {} }}',
                                'for int outer in [1:0] {{ {} }}'):
                    self.check_initializers(prefix + wrapper.format(operation), (('out', 2),), (1, 1), ('visits',), (0,))
                result = self.execute(self.program(prefix + operation))
                self.assertNotEqual(result.returncode, 0, operand)
                self.assertNotIn('runtime error:', result.stderr)
                type(self).checked += 1
            self.check_initializers(prefix + f'bool yes = true || {operand} == "00"; '
                                    f'bool no = false && {operand} == "00";',
                                    (('out', 2),), (1, 1), ('yes', 'no'), (1, 0))


    @staticmethod
    def replace_bits(raw, indices, bits):
        for index, bit in zip(indices, bits):
            raw = (raw & ~(1 << index)) | (bit << index)
        return raw

    def test_all_assignment_operators_at_every_width(self):
        native = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        for width in range(1, native + 1):
            mask = (1 << width) - 1
            indices = (width - 1, 0) if width > 1 else (0,)
            selector = '{last, first}' if width > 1 else '{last}'
            reverse = '{first, last}' if width > 1 else '{last}'
            repeated = '{last, last}' if width > 1 else '{last}'
            ones = '"' + '1' * len(indices) + '"'
            alternate = '"10"' if width > 1 else '"0"'
            for raw in sorted({0, mask, 0xAAAAAAAA & mask}):
                source = (f'angle[{width}] a = angle[{width}]("{raw:0{width}b}"); '
                          f'array[angle[{width}], 1] phases = {{a}}; int element = -1; '
                          f'int last = -1; uint first = 0; uint one = 1; uint huge = {2**native - 1}; ')
                snapshots, expected = [], []
                for origin, operand in enumerate(('a', 'phases[element]')):
                    current = raw
                    operations = (
                        ('=', ones, [1] * len(indices)),
                        ('&=', alternate, [0, 1] if width > 1 else [0]),
                        ('|=', ones, [1] * len(indices)),
                        ('^=', ones, [1] * len(indices)),
                        ('=', f'{operand}[{reverse}]', tuple(reversed(indices))),
                        ('^=', f'{operand}[{repeated}]', (width - 1,) * len(indices)),
                        ('=', f'~{operand}[{selector}]', 'inverse'),
                        ('<<=', '0', 0), ('>>=', 'one', 1),
                        ('|=', ones, [1] * len(indices)), ('<<=', 'huge', len(indices)),
                        ('|=', ones, [1] * len(indices)), ('>>=', str(2**200), len(indices)),
                    )
                    for step, (operator, rhs, reference) in enumerate(operations):
                        old = [(current >> index) & 1 for index in indices]
                        if operator in ('<<=', '>>='):
                            distance = reference
                            if distance >= len(old):
                                new = [0] * len(old)
                            elif operator == '<<=':
                                new = [0] * distance + old[:len(old) - distance]
                            else:
                                new = old[distance:] + [0] * distance
                        else:
                            if isinstance(reference, tuple):
                                right = [(current >> index) & 1 for index in reference]
                            elif reference == 'inverse':
                                right = [1 - bit for bit in old]
                            else:
                                right = reference
                            if operator == '=':
                                new = right
                            elif operator == '&=':
                                new = [left & right for left, right in zip(old, right)]
                            elif operator == '|=':
                                new = [left | right for left, right in zip(old, right)]
                            else:
                                new = [left ^ right for left, right in zip(old, right)]
                        current = self.replace_bits(current, indices, new)
                        name = f'snapshot{origin}_{step}'
                        source += f'{operand}[{selector}] {operator} {rhs}; angle[{width}] {name} = {operand}; '
                        snapshots.append(name)
                        expected.append(current)
                self.check_initializers(source, (), (), tuple(snapshots), tuple(expected))

    def test_full_width_destination_and_overlapping_rhs(self):
        for width in (1, 4, self.module.QASM2QCXConverter.QCX_UINT_WIDTH):
            mask = (1 << width) - 1
            raw = 0xAAAAAAAA & mask
            target = '{index' + ''.join(f', {i}' for i in range(width - 1)) + '}'
            indices = (width - 1,) + tuple(range(width - 1))
            source = '{' + ', '.join(str(i) for i in range(width)) + '}'
            expected = self.replace_bits(raw, indices, [(raw >> i) & 1 for i in range(width)])
            self.check_initializers(f'angle[{width}] a = angle[{width}]("{raw:0{width}b}"); int index = -1; '
                f'a[{target}] = a[{source}];', (), (), ('a',), (expected,))
        self.check_initializers('angle[4] a = angle[4]("0011"); int index = -1; uint first = 0; '
            'a[{index, first}] = a[{index, index}];', (), (), ('a',), (2,))

    def test_destination_duplicates_fail_before_rhs_even_after_negative_normalization(self):
        prefix = ('angle[4] a = angle[4]("0011"); int last = -1; uint high = 3; int first = 0; int zero = 0; ')
        for selector in ('{last, high}', '{last, -1}', '{first, first}', '{0, last, 0}', '{last, 0, 1, 2, 3}'):
            width = len(selector.strip('{}').split(','))
            for operator, rhs in (('=', '"' + '1' * width + '"'), ('<<=', '1 / zero')):
                statement = f'a[{selector}] {operator} {rhs};'
                for wrapper in ('if (false) {{ {} }}', 'while (false) {{ {} }}',
                                'for int outer in [1:0] {{ {} }}'):
                    self.check_initializers(prefix + wrapper.format(statement), (), (), ('a',), (3,))
                result = self.execute(self.program(prefix + statement))
                self.assertNotEqual(result.returncode, 0, statement)
                self.assertIn('assertion_error', result.stderr)
                self.assertNotIn('integer_zero_divisor_error', result.stderr)
                self.assertNotIn('runtime error:', result.stderr)
                type(self).checked += 1

    def test_write_bounds_rhs_domains_and_counts_fail_only_when_reached(self):
        maximum = self.module.QASM2QCXConverter.QCX_UINT_MAX
        prefix = ('angle[4] a = pi; array[angle[4], 1] phases = {pi}; int element = -1; '
                  f'int bad_element = 1; int index = -1; uint first = 0; uint huge = {maximum}; '
                  'int bad = -5; int high = 4; int zero = 0; int negative = -1; '
                  f'int minimum = {self.module.QASM2QCXConverter.QCX_INT_MIN}; ')
        for statement in ('a[{index, high}] = "11";', 'a[{bad, first}] ^= "11";',
                          'a[{index, huge}] &= "11";', f'a[{{index, {2**200}}}] |= "11";',
                          'a[{index, first}] = phases[bad_element][{index, first}];',
                          'phases[bad_element][{index, first}] = "11";',
                          'a[{index, first}] ^= a[{index, high}];',
                          'a[{index, first}] <<= negative;', 'a[{index, first}] >>= minimum;',
                          'a[{index, first}] >>= 1 / zero;'):
            for wrapper in ('if (false) {{ {} }}', 'while (false) {{ {} }}',
                            'for int outer in [1:0] {{ {} }}'):
                self.check_initializers(prefix + wrapper.format(statement), (('phases', 1),), (8,), ('a',), (8,))
            result = self.execute(self.program(prefix + statement))
            self.assertNotEqual(result.returncode, 0, statement)
            self.assertNotIn('runtime error:', result.stderr)
            type(self).checked += 1

    def test_assignments_in_nested_loops_and_during_live_iteration(self):
        self.check_initializers('angle[4] a; int high = -1; uint low = 0; int visits = 0; '
            'while (visits < 3) { visits += 1; for int i in {0, 1} { a[{high, low}] ^= "11"; }} '
            'a[{high, low}] |= "11"; int observed = 0; '
            'for bit b in a[{low, high}] { observed += int(b); a[{high, low}] = "00"; }',
            (), (), ('a', 'visits', 'observed'), (0, 3, 1))

    def test_phase_parameters_remain_captured_across_set_writes_with_fusion(self):
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
                source = ('include "stdgates.inc"; qubit[2] q; h q; int high = -1; uint low = 0; int exponent = 2; '
                          + declaration + gate + f'{operand}[{{high, low}}] ^= "01"; ' + gate)
                phase = (-2 if factor == 0.5 else 14) * math.pi / 8
                expected = tuple(cmath.exp(1j * factor * phase) / 2 if active(i) else 0.5 for i in range(4))
                for fused in (False, True):
                    BackendAuditTests.check(self, self.program(source, fused=fused), expected)


    def test_documented_runtime_angle_bit_set_example(self):
        import cmath
        import math
        from qasm2qcx_u_power_numerical import BackendAuditTests
        document = (pathlib.Path(__file__).parents[1] / 'qcx/qasm2qcx.md').read_text(encoding='utf-8')
        examples = re.findall(r'<!-- runtime-angle-bit-set-example -->\s*```qasm\n(.*?)\n```',
                              document, flags=re.DOTALL)
        self.assertEqual(len(examples), 1)
        self.check_initializers(examples[0].removeprefix('OPENQASM 3.0;'),
                                (('before', 3), ('phases', 1)), (0, 1, 0, 2),
                                ('a', 'low', 'high', 'element', 'ones', 'visits'), (10, 1, -1, 0, 1, 3))
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
    print(f'runtime angle bit-set programs checked: {BackendTests.checked}')
    raise SystemExit(not result.wasSuccessful())
