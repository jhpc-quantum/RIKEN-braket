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

    def test_documented_runtime_set_example(self):
        document = (pathlib.Path(__file__).resolve().parents[1] / 'qcx/qasm2qcx.md').read_text()
        marker = '<!-- runtime-bit-set-uint-cast-example -->'
        source = document.split(marker, 1)[1].split('```qasm\n', 1)[1].split('```', 1)[0]
        # check_exact supplies the OpenQASM header.
        source = source.replace('OPENQASM 3.0;', '', 1)
        self.check_exact(source, (('from_constant', None), ('packed', None), ('back', 4),
                                 ('flags', 8), ('fresh', None)), (13, 7, 7, 199, 15))

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


    def test_nested_register_expressions_and_numeric_consumers(self):
        for pattern in range(16):
            entries = 'index,2,1,0'
            value = int(f'{pattern:04b}'[::-1], 2)
            inverted = value ^ 15
            left = ((value << 1) | (value >> 3)) & 15
            right = ((value >> 1) | (value << 3)) & 15
            source = f'bit[4] flags = "{pattern:04b}"; int index = 3; '
            source += f'uint[4] inverted = uint[4](~flags[{{{entries}}}]); '
            source += f'uint[4] left = uint[4](rotl(flags[{{{entries}}}], 1)); '
            source += f'uint[4] right = uint[4](rotr(flags[{{{entries}}}], 1)); '
            source += f'uint[4] sum = uint[4](flags[{{{entries}}}]) + uint[4](1); '
            source += f'angle[4] phase = angle[4](bit[4](uint[4](flags[{{{entries}}}]))); '
            source += f'bit[4] bridge = bit[4](angle[4](flags[{{{entries}}}])); '
            source += 'uint[4] again = uint[4](bridge); '
            source += f'uint[4] mixed = uint[4](flags[{{{entries}}}] ^ "1010"); '
            source += f'array[uint[4],2] values = {{uint[4](flags[{{{entries}}}]), uint[4](~flags[{{{entries}}}])}}; '
            source += 'uint[4] copied = values[0]; values[0] += values[1]; uint[4] combined = values[0]; '
            source += f'bool same = flags == "{pattern:04b}";'
            self.check_exact(source, (('inverted', None), ('left', None), ('right', None), ('sum', None),
                ('phase', None), ('again', None), ('mixed', None), ('copied', None), ('combined', None), ('same', None)),
                (inverted, left, right, (value + 1) & 15, value, value, value ^ 10, value, 15, 1))

    def test_casts_in_index_expressions_and_indexed_destinations(self):
        # The outer read must retain its indices while the inner cast packs.
        for pattern in range(16):
            for index in (0, 1):
                bits = f'{pattern:04b}'
                inner = ((pattern >> index) & 1) | ((pattern & 1) << 1)
                positions = (inner, 0, 1, 2)
                expected = sum(((pattern >> position) & 1) << i for i, position in enumerate(positions))
                source = f'bit[4] flags = "{bits}"; int index = {index}; array[uint[4],2] values = {{0,0}}; '
                source += 'values[uint[1](flags[{index}])] = uint[4](flags[{uint[2](flags[{index,0}]),0,1,2}]); '
                source += 'uint[4] out = values[uint[1](flags[{index}])]; '
                source += f'bool same = flags == "{bits}";'
                self.check_exact(source, (('out', None), ('same', None)), (expected, 1))

    def test_changing_indices_and_sources_in_while_loops_with_transfers(self):
        self.check_exact('bit[4] flags = "1011"; int index = 0; uint total = 0; uint[2] pair; '
            'while (index < 4) { pair = uint[2](flags[{index,-1}]); '
            'index += 1; if (index == 2) { continue; } total += pair; '
            'if (index == 3) { break; } }', (('total', None), ('index', None), ('flags', 4)), (5, 3, 11))
        self.check_exact('bit[4] flags = "0000"; int index = 0; uint[4] packed = 0; int visits = 0; '
            'while (uint[4](flags[{index,1,2,3}]) != uint[4](8)) { '
            'packed = uint[4](flags[{index,1,2,3}]) + uint[4](1); flags = bit[4](packed); '
            'visits += 1; }', (('visits', None), ('packed', None), ('flags', 4)), (8, 8, 8))

    def test_loop_bounds_are_captured_and_iterator_shadows_are_scoped(self):
        self.check_exact('const bit[4] C = "1001"; bit[4] flags = "0101"; int index = 0; uint total = 0; '
            'for int outer in [0:1] { for int i in [uint[1](C[{index}]):uint[3](flags[{index,1,2}])] { '
            'total += uint(i); flags = "0000"; } } '
            'for int index in [-4:-1] { total += uint[1](C[{index}]); } '
            'array[uint[4],2] indices = {0,3}; for uint[4] index in indices { total += uint[1](C[{index}]); } '
            'uint[1] restored = uint[1](C[{index}]);',
            (('total', None), ('restored', None), ('index', None)), (19, 1, 0))
        self.check_exact('const bit[4] C = "1001"; int index = 0; uint total = 0; '
            'for int outer in [0:1] { for int i in {uint[1](C[{outer}]),uint[2](C[{index,outer}])} { total += uint(i); } }',
            (('total', None),), (5,))

    def test_skipped_paths_suppress_index_domains_but_preserve_validated_types(self):
        calls = ('packed = uint[4](flags[{index,bad,0,1}]);',
                 'packed = uint[4](C[{index,999,0,1}]);',
                 'packed = uint[4](flags[{index / zero,0,1,2}]);',
                 'packed = uint[4](flags[{index,positions[bad],0,1}]);')
        wrappers = ('if (false) {{ {} }}', 'while (false) {{ {} }}',
                    'for int outer in [1:0] {{ {} }}', 'for int outer in [first:last] {{ {} }}',
                    'while (visits < 2) {{ visits += 1; continue; {} }}',
                    'while (true) {{ break; {} }}')
        for call in calls:
            for wrapper in wrappers:
                self.check_exact('bit[4] flags = "1010"; const bit[4] C = "1001"; uint[4] packed = 6; '
                    'array[int,2] positions = {0,1}; int index = 0; int bad = -5; int zero = 0; '
                    'int visits = 0; int first = 2; int last = 1; ' + wrapper.format(call),
                    (('packed', None), ('flags', 4)), (6, 10))
        for condition, expected in (('false && bool(uint[4](flags[{index,bad,0,1}]))', 0),
                                    ('true || bool(uint[4](flags[{index / zero,0,1,2}]))', 1)):
            self.check_exact('bit[4] flags = "1010"; int index = 0; int bad = 99; int zero = 0; '
                f'bool result = {condition};', (('result', None), ('flags', 4)), (expected, 10))

    def test_measurement_selection_casts_drive_feedback_from_live_bits(self):
        for anticorrelated in (False, True):
            for _ in range(16):
                self.check_exact('include "stdgates.inc"; qubit[2] q; reset q; h q[0]; cx q[0],q[1]; '
                    + ('x q[1]; ' if anticorrelated else '') +
                    'bit[2] measured = measure q; int index = 1; '
                    'uint[2] packed = uint[2](measured[{index,0}]); bit[2] back = bit[2](packed); '
                    'bool same = back[0] == measured[1] && back[1] == measured[0]; '
                    'if (bool(packed)) { x q[0]; } bit[2] observed = measure q; '
                    'bool feedback = bool(observed[0]) == (bool(measured[0]) != bool(packed)) '
                    '&& observed[1] == measured[1]; '
                    f'bool correlation = (back[0] != back[1]) == {str(anticorrelated).lower()};',
                    (('same', None), ('feedback', None), ('correlation', None)), (1, 1, 1))

    def test_later_iteration_failure_preserves_the_previous_successful_assignment(self):
        for constant in (False, True):
            for entry, update, error in (('index', 'index = 8;', 'assertion failed'),
                                         ('index / divisor', 'divisor = 0;', 'division by zero')):
                program = self.module.openqasm3.parser.parse('OPENQASM 3; ' +
                    ('const ' if constant else '') + 'bit[8] flags = "11010011"; '
                    'int index = 0; int divisor = 1; int iteration = 0; uint[4] packed = 6; '
                    f'while (iteration < 2) {{ packed = uint[4](flags[{{{entry},0,1,2}}]); '
                    f'iteration += 1; {update} }}')
                converter = self.module.QASM2QCXConverter(program)
                converter.visit(program)
                self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())
                destination = converter._QASM2QCXConverter__capitalize_variable_name('packed')
                instrumented = []
                for line in converter:
                    if line.startswith(f'LET {destination} := QASM2QCX_'):
                        instrumented.extend(['PRINTLN 11111', line, f'PRINTLN {destination}'])
                    else:
                        instrumented.append(line)
                self.assertIn('PRINTLN 11111', instrumented)
                result = self.execute(instrumented)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(error, result.stderr)
                self.assertNotIn('runtime error:', result.stderr)
                self.assertEqual(result.stdout.split(), ['11111', '7'])
                type(self).checked += 1

    def test_quantum_parameters_and_integer_powers_with_fusion(self):
        import cmath
        import math
        from qasm2qcx_u_power_numerical import BackendAuditTests
        for fused in (False, True):
            for gate in ('p(angle[4](bit[4](uint[4](flags[{index,2,1,0}])))) q;',
                         'pow(uint[4](flags[{index,2,1,0}])) @ p(pi / 8) q;'):
                source = 'include "stdgates.inc"; qubit q; bit[4] flags = "0110"; int index = 3; h q; ' + gate
                BackendAuditTests.check(self, self.program(source, fused=fused),
                    (1 / math.sqrt(2), cmath.exp(3j * math.pi / 4) / math.sqrt(2)))
            source = ('include "stdgates.inc"; qubit q; bit[4] flags = "0110"; int index = 3; '
                      'rx(float(uint[4](flags[{index,2,1,0}])) / 4.0) q;')
            BackendAuditTests.check(self, self.program(source, fused=fused), (math.cos(0.75), -1j * math.sin(0.75)))


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
