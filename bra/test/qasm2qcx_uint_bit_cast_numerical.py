#!/usr/bin/env python3
"""Audit explicit same-width UINT/bit-register casts (#468)."""

import argparse
import pathlib
import random
import re
import resource
import unittest

import qasm2qcx_angle_bit_cast_numerical as angle_casts


class BackendTests(unittest.TestCase):
    bra = None
    checked = 0
    setUp = angle_casts.BackendTests.setUp
    execute = angle_casts.BackendTests.execute
    program = angle_casts.BackendTests.program
    check_exact = angle_casts.BackendTests.check_exact

    def test_all_widths_constant_and_runtime_round_trips_preserve_sources(self):
        random_source = random.Random(468)
        for width in range(1, self.module.QASM2QCXConverter.QCX_UINT_WIDTH + 1):
            modulus = 1 << width
            patterns = (range(modulus) if width <= 6 else
                        sorted({0, 1, modulus - 1, modulus // 2, modulus // 2 - 1}
                               | {random_source.randrange(modulus) for _ in range(4)}))
            for value in patterns:
                self.check_exact(f'const uint[{width}] F = uint[{width}]("{value:0{width}b}"); '
                    f'const bit[{width}] C = bit[{width}](F); '
                    f'bit[{width}] bits = "{value:0{width}b}"; uint[{width}] packed = uint[{width}](bits); '
                    f'bit[{width}] back = bit[{width}](packed); uint[{width}] number = {value}; '
                    f'bit[{width}] unpacked = bit[{width}](number); uint[{width}] again = uint[{width}](unpacked); '
                    f'bit[{width}] constant_bits = C; uint[{width}] folded = F;',
                    (('bits', width), ('packed', None), ('back', width), ('number', None),
                     ('unpacked', width), ('again', None), ('constant_bits', width), ('folded', None)), (value,) * 8)

    def test_static_selection_order_repeated_indices_and_wide_source_positions(self):
        native = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        for source_width in (8, native + 1, 65, 129):
            pattern = int(('10010110' * source_width)[:source_width], 2)
            for positions in ((source_width - 1,), (source_width - 1, 0, 3, 1), (0, 0, 1, 1),
                              tuple(range(source_width - 1, source_width - min(native, source_width) - 1, -1))):
                width = len(positions)
                expected = sum(((pattern >> position) & 1) << index for index, position in enumerate(positions))
                selector = '{' + ','.join(map(str, positions)) + '}'
                for constant in (False, True):
                    self.check_exact(('const ' if constant else '') + f'bit[{source_width}] source = "{pattern:0{source_width}b}"; '
                        f'uint[{width}] packed = uint[{width}](source[{selector}]); '
                        f'bit[{width}] back = bit[{width}](packed);',
                        (('packed', None), ('back', width)), (expected, expected))

    def test_overlapping_register_assignments_and_narrow_uint_results(self):
        for value in range(16):
            for selector, positions in (('3:-1:0', (3, 2, 1, 0)), ('{1,3,0,2}', (1, 3, 0, 2))):
                expected = sum(((value >> source) & 1) << destination for source, destination in enumerate(positions))
                self.check_exact(f'bit[4] bits = "{value:04b}"; uint[4] number = {value}; '
                    f'bits[{selector}] = bit[4](uint[4](bits)); '
                    'number = uint[4](bit[4](number + uint[4](1)));',
                    (('bits', 4), ('number', None)), (expected, (value + 1) % 16))
        self.check_exact('const uint[4] A = 14; const uint[4] B = 3; const bit[4] C = bit[4](A + B); '
            'uint[4] a = 14; uint[4] b = 3; bit[4] result = bit[4](a + b); '
            'uint[4] nested = uint[4](bit[4](a + b)) + uint[4](1); bit[4] folded = C; '
            'uint[4] rotated = uint[4](rotl(bit[4](a), 1));',
            (('result', 4), ('nested', None), ('folded', 4), ('rotated', None)), (1, 2, 1, 13))

    def test_unsigned_casts_supply_shift_counts_independent_of_shifted_register_width(self):
        native = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        for count_width in (4, native):
            for count in sorted({0, 1, 7, 8, (1 << count_width) - 1, 1 << (count_width - 1)}):
                self.check_exact(f'bit[{count_width}] count_bits = "{count:0{count_width}b}"; '
                    'bit[8] left = "10001111"; bit[8] right = "10001111"; '
                    f'left <<= uint[{count_width}](count_bits); right >>= uint[{count_width}](count_bits);',
                    (('left', 8), ('right', 8), ('count_bits', count_width)),
                    ((143 << count) & 255 if count < 8 else 0, 143 >> count if count < 8 else 0, count))
        self.check_exact(f'uint source = {self.module.QASM2QCXConverter.QCX_UINT_MAX}; '
                         f'bit[{native}] bits = bit[{native}](source); uint packed = uint(bits);',
                         (('source', None), ('bits', native), ('packed', None)),
                         (self.module.QASM2QCXConverter.QCX_UINT_MAX,) * 3)

    def test_nested_expressions_conditions_numeric_conversions_and_narrow_wrap(self):
        for width in (1, 4, self.module.QASM2QCXConverter.QCX_UINT_WIDTH):
            mask = (1 << width) - 1
            for value in sorted({0, 1, mask, 1 << (width - 1)}):
                expected = ((value + 1) & mask) ^ (value ^ mask)
                low_mask = min(mask, 127)
                self.check_exact(f'bit[{width}] bits = "{value:0{width}b}"; '
                    f'uint[{width}] left = uint[{width}](bits); uint[{width}] right = uint[{width}](~bits); '
                    f'bit[{width}] result = bit[{width}]((left + uint[{width}](1)) ^ right); '
                    f'uint[{width}] packed = uint[{width}](result); bool nonzero = bool(uint[{width}](result)); '
                    f'int low = int(uint[{width}](result) & uint[{width}]({low_mask})); '
                    f'float real = float(uint[{width}](result)); bool converted = real == {float(expected)!r}; '
                    f'bool same = result == bit[{width}](packed);',
                    (('result', width), ('packed', None), ('nonzero', None), ('low', None), ('converted', None), ('same', None)),
                    (expected, expected, int(expected != 0), expected & low_mask, 1, 1))

    def test_loop_families_refresh_array_operands_and_both_cast_directions(self):
        loops = (('for int i in [0:3]', 'i', (0,1,2,3)),
                 ('for int i in [first:last]', 'i', (0,1,2,3)),
                 ('for int i in [ufirst:ulast]', 'i', (0,1,2,3)),
                 ('for int i in [first:stride:last]', 'i', (0,1,2,3)),
                 ('for int i in {0,1,2,3}', 'i', (0,1,2,3)),
                 ('for int i in {first,first + 1,last - 1,last}', 'i', (0,1,2,3)),
                 ('for int i in indices', 'i', (0,1,2,3)),
                 ('for uint[4] i in unsigned_indices', 'i', (0,1,2,3)),
                 ('for bit i in flags', 'int(i)', (0,1,0,1)),
                 ('for bit i in C', 'int(i)', (0,1,0,1)))
        values = (0,15,8,3)
        for loop, index, visits in loops:
            results = [(values[i] + 1) % 16 for i in visits]
            self.check_exact('const bit[4] C = "1010"; bit[4] flags = "1010"; bit[4] bits = "0000"; '
                'array[uint[4],4] values = {0,15,8,3}; array[int,4] indices = {0,1,2,3}; '
                'array[uint[4],4] unsigned_indices = {0,1,2,3}; int first = 0; int last = 3; '
                'uint ufirst = 0; uint ulast = 3; int stride = 1; int i = -9; uint total = 0; uint[4] number = 0; '
                + loop + f' {{ number = uint[4](bit[4](values[{index}] + uint[4](1))); '
                'bits = bit[4](number); total += uint[4](bits); }',
                (('number', None), ('bits', 4), ('total', None), ('i', None)),
                (results[-1], results[-1], sum(results), -9))
        self.check_exact('bit[4] bits = "0000"; uint total = 0; '
            'for bit b in bits { total += uint(b); bits = bit[4](uint[4](bits) ^ uint[4](15)); }',
            (('bits', 4), ('total', None)), (0, 2))

    def test_cast_conditions_bounds_transfers_and_nested_constant_register_names(self):
        self.check_exact('bit[4] bits = "0000"; uint[4] number = 0; int visits = 0; '
            'while (uint[4](bits) != uint[4](8)) { number = uint[4](bits) + uint[4](1); '
            'bits = bit[4](number); if (uint[4](bits) == uint[4](3)) { continue; } '
            'visits += 1; if (uint[4](bits) == uint[4](15)) { break; } }',
            (('bits', 4), ('number', None), ('visits', None)), (8, 8, 7))
        self.check_exact('const bit[4] FIRST = "0001"; const bit[4] STEP = "0010"; bit[4] lastbits = "0101"; '
            'int total = 0; for int outer in [0:1] { '
            'for int i in [uint[4](FIRST):uint[4](STEP):uint[4](lastbits)] { total += i; } } '
            'for int outer in [1:0] { for int i in [uint[4](FIRST):uint[4](STEP):uint[4](lastbits)] { total += 99; } } '
            'for int i in {uint[4](FIRST),uint[4](lastbits)} { total += i; }',
            (('total', None), ('lastbits', 4)), (24, 5))
        self.check_exact('bit[4] firstbits = "0001"; bit[4] lastbits = "0101"; int total = 0; '
            'for int i in [uint[4](firstbits):uint[4](lastbits)] { '
            'firstbits = bit[4](uint[4](9)); lastbits = bit[4](uint[4](0)); total += i; }',
            (('total', None), ('firstbits', 4), ('lastbits', 4)), (15, 9, 0))
        self.check_exact('const int N = 4; const uint[4] U = 9; bit[4] b = "1001"; '
            'array[uint[N],3] values = {1,8,15}; uint total = 0; '
            'for uint[N] U in values { b = bit[4](U); total += uint[4](b); } '
            'uint[4] restored = U; for int N in [0:1] { b = bit[4](restored); } '
            'for int b in [0:1] { restored += uint[4](b); }',
            (('b', 4), ('restored', None), ('total', None)), (9, 10, 24))

    def test_promoted_uint_widths_and_unsized_native_array_iterators(self):
        self.check_exact('uint[4] a = 15; uint[8] b = 240; bit[8] promoted = bit[8](a + b); '
            'uint[8] packed = uint[8](promoted); bit[4] narrowed = bit[4](uint[4](packed));',
            (('promoted', 8), ('packed', None), ('narrowed', 4)), (255, 255, 15))
        width = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        top = 1 << (width - 1)
        mask = (1 << width) - 1
        self.check_exact(f'array[uint,3] values = {{0,{top},{mask}}}; bit[{width}] bits = "' + '0' * width +
            f'"; uint packed = 0; uint total = 0; bool same = true; '
            f'for uint value in values {{ bits = bit[{width}](value); packed = uint(bits); '
            'same = same && (packed == value); total += packed; }',
            (('bits', width), ('packed', None), ('same', None), ('total', None)), (mask, mask, 1, top - 1))

    def test_skipped_statements_and_short_circuit_conditions_suppress_operand_domains(self):
        calls = ('packed = uint[4](bits << distance);', 'bits = bit[4](number / zero);',
                 'bits = bit[4](values[bad]);', 'bits = bit[4](uint[4](1) / uint[4](0));',
                 'packed = uint[4]("0011" << -1);')
        wrappers = ('if (false) {{ {} }}', 'while (false) {{ {} }}', 'for int outer in [1:0] {{ {} }}',
                    'for int outer in [first:last] {{ {} }}',
                    'while (visits < 2) {{ visits += 1; continue; {} }}', 'while (true) {{ break; {} }}')
        for call in calls:
            for wrapper in wrappers:
                self.check_exact('bit[4] bits = "1010"; uint[4] packed = 6; uint[4] number = 3; uint[4] zero = 0; '
                    'array[uint[4],2] values = {1,2}; int bad = 2; int distance = -1; int visits = 0; '
                    'int first = 2; int last = 1; ' + wrapper.format(call),
                    (('bits', 4), ('packed', None), ('number', None)), (10, 6, 3))
        for condition, expected in (('false && bool(uint[4](bits << -1))', 0),
                                     ('true || (bit[4](number / zero) == bits)', 1)):
            self.check_exact('bit[4] bits = "1010"; uint[4] number = 3; uint[4] zero = 0; '
                f'bool checked = {condition};', (('checked', None), ('bits', 4)), (expected, 10))

    def test_measurement_patterns_round_trip_and_drive_quantum_feedback(self):
        for anti in (0,1):
            for _ in range(16):
                self.check_exact('include "stdgates.inc"; qubit[2] q; reset q; h q[0]; cx q[0],q[1]; '
                    + ('x q[1]; ' if anti else '') +
                    'bit[2] measured = measure q; uint[2] packed = uint[2](measured); bit[2] back = bit[2](packed); '
                    'bool same = back == measured; uint[2] next_value = uint[2](bit[2](packed + uint[2](1))); '
                    'bool wrapped = next_value == (packed + uint[2](1)); '
                    'if (bool(uint[2](back))) { x q[0]; } bit[2] observed = measure q; '
                    'bool feedback = bool(observed[0]) == (bool(measured[0]) != bool(packed)) '
                    '&& observed[1] == measured[1]; '
                    f'bool correlation = (back[0] != back[1]) == bool({anti});',
                    (('same', None), ('wrapped', None), ('feedback', None), ('correlation', None)), (1,1,1,1))
        width = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        value = 1 << (width - 1)
        self.check_exact('include "stdgates.inc"; qubit q; reset q; x q; '
            f'bit[{width}] flags = "' + '0' * width + f'"; flags[{width - 1}] = measure q; '
            f'uint packed = uint(flags); bit[{width}] back = bit[{width}](packed);',
            (('flags', width), ('packed', None), ('back', width)), (value, value, value))

    def test_gate_parameters_integer_powers_and_exact_angle_patterns_with_fusion(self):
        import cmath
        import math
        from qasm2qcx_u_power_numerical import BackendAuditTests
        phase = 3 * math.pi / 4
        for fused in (False, True):
            for gate in ('p(angle[4](bit[4](uint[4](bits)))) q;', 'pow(uint[4](bits)) @ p(pi / 8) q;'):
                source = 'include "stdgates.inc"; qubit q; bit[4] bits = "0110"; h q; ' + gate
                BackendAuditTests.check(self, self.program(source, fused=fused),
                                        (1 / math.sqrt(2), cmath.exp(1j * phase) / math.sqrt(2)))
            source = ('include "stdgates.inc"; qubit q; bit[4] bits = "0110"; '
                      'rx(float(uint[4](bits)) / 4.0) q;')
            BackendAuditTests.check(self, self.program(source, fused=fused), (math.cos(0.75), -1j * math.sin(0.75)))

    def test_later_operand_failures_preserve_previous_successful_assignments(self):
        cases = (('packed = uint[4](bits << distance);', 'distance = -1;', 'assertion failed', 'PACKED63', 10),
                 ('bits = bit[4](number / divisor);', 'divisor = 0;', 'division by zero', 'BITS15:', 5),
                 ('bits = bit[4](values[index]);', 'index = 2;', 'assertion failed', 'BITS15:', 5))
        for statement, update, error, destination, expected in cases:
            lines = self.program('bit[4] bits = "1010"; uint[4] packed = 0; uint[4] number = 5; uint[4] divisor = 1; '
                'array[uint[4],2] values = {5,9}; int distance = 0; int index = 0; int i = 0; '
                f'while (i < 2) {{ {statement} i += 1; {update} }}')
            writes = [i for i,line in enumerate(lines) if line.startswith(f'LET {destination}') and ' := QASM2QCX_' in line]
            self.assertEqual(len(writes), 4 if destination.startswith('BITS') else 1)
            instrumented = []
            for i,line in enumerate(lines):
                if i in writes:
                    instrumented.append('PRINTLN 12345')
                instrumented.append(line)
                if i == writes[-1]:
                    instrumented.append('PRINTLN ' + (' '.join(f'BITS15:{j}' for j in range(3,-1,-1))
                                                     if destination.startswith('BITS') else 'PACKED63'))
            result = self.execute(instrumented)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(error, result.stderr)
            self.assertNotIn('runtime error:', result.stderr)
            output = [''.join(line.split()) for line in result.stdout.splitlines()]
            self.assertEqual(output, ['12345'] * len(writes) + [f'{expected:04b}' if len(writes) == 4 else str(expected)])
            type(self).checked += 1

    def test_documented_uint_bit_cast_example(self):
        document = (pathlib.Path(__file__).parents[1] / 'qcx/qasm2qcx.md').read_text(encoding='utf-8')
        examples = re.findall(r'<!-- uint-bit-cast-example -->\s*```qasm\n(.*?)\n```', document, flags=re.DOTALL)
        self.assertEqual(len(examples), 1)
        source = examples[0].removeprefix('OPENQASM 3.0;')
        self.check_exact(source, (('bits', 4), ('packed', None), ('wrapped', None), ('flags', 8), ('count_bits', 4), ('total', None)),
                         (13, 11, 2, 30, 1, 15))

    def test_runtime_failures_precede_destination_writes(self):
        cases = (('uint[4] u = 0; bit[4] bits = "1001"; int count = -1; u = uint[4](bits << count);',
                  'U1', 'assertion failed'),
                 ('bit[4] bits = "1001"; array[uint[4],2] values = {1,2}; int index = 2; bits = bit[4](values[index]);',
                  'BITS15:', 'assertion failed'),
                 ('bit[4] bits = "1001"; uint[4] u = 1; uint[4] zero = 0; bits = bit[4](u / zero);',
                  'BITS15:', 'division by zero'))
        for source, destination, error in cases:
            lines = self.program(source)
            instrumented = []
            for line in lines:
                if line.startswith(f'LET {destination}') and ' := QASM2QCX_' in line:
                    instrumented.append('PRINTLN 12345')
                instrumented.append(line)
            self.assertIn('PRINTLN 12345', instrumented)
            result = self.execute(instrumented)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(error, result.stderr)
            self.assertNotIn('runtime error:', result.stderr)
            self.assertNotIn('12345', result.stdout)
            type(self).checked += 1


def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    args = parser.parse_args()
    BackendTests.bra = args.bra.resolve()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(BackendTests))
    print(f'UINT/bit-register cast programs checked: {BackendTests.checked}')
    raise SystemExit(not result.wasSuccessful())


if __name__ == '__main__':
    main()
