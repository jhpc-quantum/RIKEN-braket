#!/usr/bin/env python3
"""Audit angle bit-selection iteration (#428) with a default-double non-MPI bra."""

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

    def test_static_selections_at_every_width_preserve_order_and_sources(self):
        for width in range(1, self.module.QASM2QCXConverter.QCX_UINT_WIDTH + 1):
            mask = (1 << width) - 1
            for raw in sorted({0, mask, 1 << (width - 1), 0xAAAAAAAA & mask}):
                source = (f'const angle[{width}] C = angle[{width}]("{raw:0{width}b}"); '
                          f'angle[{width}] value = C; array[angle[{width}], 1] phases = {{C}}; int element = -1; ')
                outputs, expected = [], []
                selections = (('0', (0,)), (f'0:{width - 1}', tuple(range(width))),
                              (f'-1:-1:{-width}', tuple(reversed(range(width)))),
                              ('{-1, 0, -1}', (width - 1, 0, width - 1)),
                              (f'0:2:{width - 1}', tuple(range(0, width, 2))))
                for origin, operand in enumerate(('C', 'value', 'phases[element]')):
                    for number, (selector, indices) in enumerate(selections):
                        suffix = f'{origin}_{number}'
                        count, visits, checksum = (prefix + suffix for prefix in ('count', 'visits', 'checksum'))
                        source += (f'uint {count} = 0; int {visits} = 0; uint {checksum} = 0; '
                                   f'for bit b in {operand}[{selector}] {{ {count} += uint(b); '
                                   f'{visits} += 1; {checksum} = ({checksum} * 3 + uint(b)) % 997; }}')
                        bits = [(raw >> bit) & 1 for bit in indices]
                        reference = 0
                        for bit in bits:
                            reference = (reference * 3 + bit) % 997
                        outputs.extend((count, visits, checksum))
                        expected.extend((sum(bits), len(bits), reference))
                self.check_initializers(source, (('phases', 1),), (raw,),
                                        tuple(outputs) + ('value',), tuple(expected) + (raw,))

    def test_live_reads_repeated_bits_and_copied_iterator_values(self):
        self.check_initializers('angle[4] a = angle[4]("0001"); int visits = 0; uint count = 0; uint checksum = 0; '
            'for bit b in a[{0, 1, 0, 3}] { '
            'if (visits == 0) { a[1] = "1"; a[0] = "0"; } '
            'if (visits == 1) { a[3] = "1"; } '
            'count += uint(b); checksum = checksum * 3 + uint(b); visits += 1; }',
            (), (), ('a', 'count', 'checksum', 'visits'), (10, 3, 37, 4))

    def test_array_element_address_is_captured_once_but_its_bits_remain_live(self):
        self.check_initializers('array[angle[4], 2] phases = {angle[4]("0001"), pi}; '
            'int element = 0; uint checksum = 0; int visits = 0; '
            'for bit b in phases[element][0:3] { '
            'if (visits == 0) { element = 1; phases[0][3] = "1"; } '
            'checksum = checksum * 3 + uint(b); visits += 1; }',
            (('phases', 2),), (9, 8), ('element', 'checksum', 'visits'), (1, 28, 4))

    def test_source_names_can_be_shadowed_by_bit_iterators(self):
        for declaration, operand, iterator, indices, expected in (
                ('angle[4] a = angle[4]("1010");', 'a', 'a', '0:3', 2),
                ('const angle[4] C = pi;', 'C', 'C', '{3, 0, 3}', 2),
                ('array[angle[4], 1] phases = {pi}; int element = -1;',
                 'phases[element]', 'phases', '0:3', 1),
                ('array[angle[4], 1] phases = {pi}; int element = -1;',
                 'phases[element]', 'element', '0:3', 1)):
            self.check_initializers(declaration + 'uint count = 0; '
                f'for bit {iterator} in {operand}[{indices}] {{ count += uint({iterator}); }}',
                (), (), ('count',), (expected,))

    def test_nested_loops_angle_iterators_and_reentry(self):
        self.check_initializers('array[angle[4], 2] phases = {angle[4]("0011"), angle[4]("1010")}; '
            'uint count = 0; int visits = 0; int passes = 0; int stop = 1; '
            'while (passes < 2) { passes += 1; for int index in [0:stop] { '
            'for angle[4] value in phases { for bit b in value[0:3] { '
            'for bit c in phases[index][{0, 3}] { visits += 1; if (b && c) { count += 1; } }}}}}',
            (), (), ('count', 'visits', 'passes'), (16, 64, 2))

    def test_break_continue_and_declaration_hoisting(self):
        for selection in ('a[0:3]', 'C[0:3]', 'phases[element][0:3]'):
            prefix = ('const angle[4] C = angle[4]("1011"); angle[4] a = C; '
                      'array[angle[4], 1] phases = {C}; int element = 0; uint count = 0; int visits = 0; ')
            self.check_initializers(prefix + f'for bit b in {selection} {{ '
                'visits += 1; if (!b) { continue; } count += uint(b); if (count == 2) { break; }}',
                (), (), ('count', 'visits'), (2, 2))
            for transfer, expected in (('break', 1), ('continue', 4)):
                self.check_initializers(prefix + f'for bit b in {selection} {{ visits += 1; '
                    + transfer + '; count += uint(b); } count += uint(a[0] == "1");',
                    (), (), ('count', 'visits'), (1, expected))

    def test_array_checks_execute_only_on_reached_paths(self):
        prefix = 'array[angle[4], 1] phases = {pi}; int bad = 1; int zero = 0; int visits = 0; '
        for source in ('phases[bad][0:3]', 'phases[1 / zero][{3, 0, 3}]'):
            loop = f'for bit b in {source} {{ visits += int(b); }}'
            for wrapper in ('if (false) {{ {} }}', 'while (false) {{ {} }}',
                            'for int outer in [1:0] {{ {} }}'):
                self.check_initializers(prefix + wrapper.format(loop), (), (), ('visits',), (0,))
            result = self.execute(self.program(prefix + loop))
            self.assertNotEqual(result.returncode, 0, source)
            self.assertNotIn('runtime error:', result.stderr)
            type(self).checked += 1


    def test_runtime_single_indices_at_every_width_execute_once(self):
        for width in range(1, self.module.QASM2QCXConverter.QCX_UINT_WIDTH + 1):
            mask = (1 << width) - 1
            for raw in sorted({0, mask, 1 << (width - 1), 0xAAAAAAAA & mask}):
                source = (f'const angle[{width}] C = angle[{width}]("{raw:0{width}b}"); '
                          f'angle[{width}] value = C; array[angle[{width}], 1] phases = {{C}}; int element = -1; ')
                outputs, expected = [], []
                for bit in sorted({0, width // 2, width - 1}):
                    source += f'int signed{bit} = {bit}; int negative{bit} = {bit - width}; uint unsigned{bit} = {bit}; '
                    for kind in ('signed', 'negative', 'unsigned'):
                        for origin, operand in enumerate(('C', 'value', 'phases[element]')):
                            suffix = f'{kind}{bit}_{origin}'
                            count, visits = 'count' + suffix, 'visits' + suffix
                            source += (f'uint {count} = 0; int {visits} = 0; '
                                       f'for bit b in {operand}[{kind}{bit}] {{ {count} += uint(b); {visits} += 1; }}')
                            outputs.extend((count, visits))
                            expected.extend(((raw >> bit) & 1, 1))
                self.check_initializers(source, (('phases', 1),), (raw,),
                                        tuple(outputs) + ('value',), tuple(expected) + (raw,))

    def test_runtime_indices_and_value_are_captured_before_body_changes(self):
        self.check_initializers('array[angle[4], 2] phases = {angle[4]("0101"), angle[4]("0010")}; '
            'int element = 0; int index = 0; uint count = 0; int visits = 0; uint explicit_read = 0; '
            'for bit b in phases[element][index] { element = 1; index = 3; '
            'phases[0][0] = "0"; count += uint(b); visits += 1; '
            'explicit_read = popcount(phases[element][index]); } '
            'for bit b in phases[element][index] { count += uint(b); visits += 1; }',
            (('phases', 2),), (4, 2), ('count', 'visits', 'explicit_read', 'element', 'index'), (1, 2, 0, 1, 3))

    def test_runtime_headers_resolve_names_before_iterator_shadowing(self):
        for declaration, operand, iterator in (
                ('angle[4] a = pi; int index = -1;', 'a[index]', 'a'),
                ('angle[4] a = pi; int index = -1;', 'a[index]', 'index'),
                ('const angle[4] C = pi; uint index = 3;', 'C[index]', 'C'),
                ('array[angle[4], 1] phases = {pi}; int element = -1; int index = -1;',
                 'phases[element][index]', 'phases'),
                ('array[angle[4], 1] phases = {pi}; int element = -1; int index = -1;',
                 'phases[element][index]', 'element')):
            self.check_initializers(declaration + 'uint count = 0; int visits = 0; '
                f'for bit {iterator} in {operand} {{ count += uint({iterator}); visits += 1; }}',
                (), (), ('count', 'visits'), (1, 1))

    def test_runtime_single_indices_in_nested_loops_and_reentry(self):
        self.check_initializers('angle[4] a = angle[4]("1010"); '
            'array[angle[4], 2] phases = {angle[4]("0011"), a}; uint count = 0; int visits = 0; int passes = 0; '
            'for int index in [0:3] { for bit b in a[index] { '
            'if (!b) { continue; } count += 1; } visits += 1; } '
            'for angle[4] value in phases { for int pos in {-1, 0} { '
            'for bit b in value[pos] { count += uint(b); }}} '
            'while (passes < 2) { passes += 1; '
            'for bit b in a[popcount(a[0:1])] { count += uint(b); break; }}',
            (), (), ('count', 'visits', 'passes'), (6, 4, 2))
        self.check_initializers('angle[4] a = angle[4]("1010"); int stop = 3; uint count = 0; int visits = 0; '
            'for int index in [0:stop] { for bit b in a[index] { '
            'for bit c in a[{0, 3}] { count += uint(b && c); } visits += 1; }}',
            (), (), ('count', 'visits'), (2, 4))
        self.check_initializers('const bit[2] FLAGS = "10"; angle[4] a = angle[4]("0010"); uint count = 0; '
            'for bit index in FLAGS { for bit b in a[int(index)] { count += uint(b); }}',
            (), (), ('count',), (1,))

    def test_runtime_bounds_and_arithmetic_fail_only_on_reached_paths(self):
        prefix = ('angle[4] a = pi; array[angle[4], 1] phases = {pi}; '
                  'int high = 4; int low = -5; int zero = 0; int bad_element = 1; int visits = 0; '
                  f'int minimum = {self.module.QASM2QCXConverter.QCX_INT_MIN}; '
                  f'uint huge = {2**self.module.QASM2QCXConverter.QCX_UINT_WIDTH - 1}; ')
        for operand in ('a[high]', 'a[low]', 'a[minimum]', 'a[huge]', 'a[1 / zero]',
                        'phases[bad_element][zero]', 'phases[zero][high]', 'phases[1 / zero][zero]'):
            loop = f'for bit b in {operand} {{ visits += int(b); }}'
            for wrapper in ('if (false) {{ {} }}', 'while (false) {{ {} }}',
                            'for int outer in [1:0] {{ {} }}'):
                self.check_initializers(prefix + wrapper.format(loop), (), (), ('visits',), (0,))
            for body in ('visits += int(b);', ''):
                result = self.execute(self.program(prefix + f'for bit b in {operand} {{ {body} }}'))
                self.assertNotEqual(result.returncode, 0, operand)
                self.assertNotIn('runtime error:', result.stderr)
                type(self).checked += 1

    def test_runtime_single_bit_transfers_stay_in_the_innermost_loop(self):
        for transfer in ('break', 'continue'):
            self.check_initializers('angle[4] a = pi; int index = -1; int passes = 0; int visits = 0; int skipped = 0; '
                'while (passes < 3) { passes += 1; for bit b in a[index] { visits += int(b); '
                + transfer + '; skipped += 1; }} skipped += popcount(a[0:3]);',
                (), (), ('passes', 'visits', 'skipped'), (3, 3, 1))

    def test_phase_gates_and_modifiers_with_fusion_in_selection_loops(self):
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
            for selector, executions in (('0', 1), ('index', 1), ('0:1', 2), ('{3, 0, 3}', 2)):
                for gate, factor, active in gates:
                    gate = gate.replace('value', operand)
                    source = ('include "stdgates.inc"; qubit[2] q; h q; int index = 0; int exponent = 2; '
                              + declaration + f'for bit b in {operand}[{selector}] {{ if (b) {{ '
                              + gate + f'{operand}[3] ^= "1"; ' + gate + '}}')
                    phase = executions * (-2 if factor == 0.5 else 14) * math.pi / 8
                    expected = tuple(cmath.exp(1j * factor * phase) / 2 if active(i) else 0.5 for i in range(4))
                    for fused in (False, True):
                        BackendAuditTests.check(self, self.program(source, fused=fused), expected)


    def test_documented_angle_bit_selection_loop_example(self):
        import cmath
        import math
        from qasm2qcx_u_power_numerical import BackendAuditTests
        document = (pathlib.Path(__file__).parents[1] / 'qcx/qasm2qcx.md').read_text(encoding='utf-8')
        examples = re.findall(r'<!-- angle-bit-selection-loop-example -->\s*```qasm\n(.*?)\n```',
                              document, flags=re.DOTALL)
        self.assertEqual(len(examples), 1)
        self.check_initializers(examples[0].removeprefix('OPENQASM 3.0;'),
                                (('phases', 1),), (2,),
                                ('a', 'checksum', 'visits', 'element_index', 'bit_index', 'saved'),
                                (10, 37, 4, 0, 0, 1))
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
    print(f'angle bit-selection loop programs checked: {BackendTests.checked}')
    raise SystemExit(not result.wasSuccessful())
