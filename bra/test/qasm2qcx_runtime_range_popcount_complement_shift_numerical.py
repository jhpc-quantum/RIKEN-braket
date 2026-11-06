#!/usr/bin/env python3
"""Audit complement chains around one runtime-range popcount shift (#487)."""

import argparse
import cmath
import math
import pathlib
import resource
import unittest
from unittest.mock import patch

import qasm2qcx_angle_bit_cast_numerical as casts


def operand(inner, outer, direction, count, source='flags[first:step:last]'):
    return '~' * outer + '((' + '~' * inner + source + ') ' + direction + ' ' + count + ')'


def reference(bits, inner, outer, direction, distance):
    """Flip, shift with zero fill, then flip individual little-endian bits."""
    assert bits and distance >= 0
    flipped = [bit ^ (inner % 2) for bit in bits]
    if distance >= len(bits):
        shifted = [0] * len(bits)
    elif direction == '<<':
        shifted = [0] * distance + flipped[:len(bits) - distance]
    else:
        shifted = flipped[distance:] + [0] * distance
    return sum(bit ^ (outer % 2) for bit in shifted)


class BackendTests(unittest.TestCase):
    bra = None
    checked = 0
    setUp = casts.BackendTests.setUp
    execute = casts.BackendTests.execute
    program = casts.BackendTests.program
    check_exact = casts.BackendTests.check_exact

    def test_documented_mixed_complement_example(self):
        document = (pathlib.Path(__file__).resolve().parents[1] / 'qcx/qasm2qcx.md').read_text()
        marker = '<!-- runtime-range-popcount-complement-shift-example -->'
        source = document.split(marker, 1)[1].split('```qasm\n', 1)[1].split('```', 1)[0]
        source = source.replace('OPENQASM 3.0;', '', 1)
        self.check_exact(source, tuple((name, None) for name in
            ('plain', 'before', 'after', 'both', 'even', 'full', 'empty_pattern', 'fresh')),
            (2, 1, 2, 3, 2, 4, 0, 3))

    def check_case(self, size, pattern, constant, first, step, last, counts,
                   parities=((0, 0), (1, 0), (0, 1), (1, 1))):
        positions = range(first, last + (1 if step > 0 else -1), step)
        assert positions and all(-size <= index < size for index in positions)
        bits = [(pattern >> (index % size)) & 1 for index in positions]
        source = ('const ' if constant else '') + f'bit[{size}] flags = "{pattern:0{size}b}"; '
        source += f'int first={first}; int step={step}; int last={last}; int distance=0; uint u=0; '
        outputs, expected = [], []
        for case, (distance, literal) in enumerate(counts):
            if literal is None:
                count = 'u' if distance > self.module.QASM2QCXConverter.QCX_INT_MAX else 'distance'
                source += f'{count} = {distance}; '
            else:
                count = literal
            for inner, outer in parities:
                for suffix, direction in (('l', '<<'), ('r', '>>')):
                    name = f'out{case}_{inner}{outer}{suffix}'
                    source += f'uint {name} = popcount({operand(inner, outer, direction, count)}); '
                    outputs.append((name, None))
                    expected.append(reference(bits, inner, outer, direction, distance))
        source += 'uint fresh = popcount(flags[first:step:last]); '
        source += f'bool same = flags == "{pattern:0{size}b}";'
        self.check_exact(source, (*outputs, ('fresh', None), ('same', None)), (*expected, sum(bits), 1))

    def test_every_native_width_patterns_parities_and_both_directions(self):
        native = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        for size in (*range(1, native + 2), 129):
            for pattern in (0, (1 << size) - 1, int(('10110010' * size)[:size], 2)):
                for constant in (False, True):
                    self.check_case(size, pattern, constant, 0, 1, size - 1, ((1, None), (size, None)))

    def test_reverse_strides_negative_aliases_and_varying_selected_lengths(self):
        for size, first, step, last in ((8, -1, -2, -7), (8, -8, 1, 7), (129, 128, -3, 119)):
            pattern = int(('10010110' * size)[:size], 2)
            width = len(range(first, last + (1 if step > 0 else -1), step))
            for constant in (False, True):
                self.check_case(size, pattern, constant, first, step, last,
                                ((0, None), (1, None), (width, None), (width + 1, None)))

    def test_native_endpoints_huge_literals_even_chains_and_zero_branch_reuse(self):
        cls = self.module.QASM2QCXConverter
        counts = ((0, '0'), (1, '1'), (cls.QCX_INT_MAX, None), (cls.QCX_UINT_MAX, None),
                  (cls.QCX_UINT_MAX, 'uint(-1)'), (10 ** 70, str(10 ** 70)))
        for size in (1, 4, 33, 129):
            pattern = int(('10110010' * size)[:size], 2)
            for constant in (False, True):
                self.check_case(size, pattern, constant, 0, 1, size - 1, counts,
                                ((2, 2), (3, 2), (2, 3), (3, 3)))

    def test_native_uint_results_narrow_assignment_and_arithmetic(self):
        maximum = self.module.QASM2QCXConverter.QCX_UINT_MAX
        for size in (8, 33, 129):
            width = 2 * size
            self.check_exact(f'const bit[{size}] C = "' + '1' * size + '"; '
                f'int first=-{size}; int last={size - 1}; int distance=1; uint huge=uint(-1); '
                'uint inner = popcount((~C[first:last]) << distance); '
                'uint outer = popcount(~(C[first:last] << distance)); '
                'uint both = popcount(~((~C[first:last]) << distance)); '
                'uint full = popcount(~((~C[first:last]) >> huge)); '
                'uint[3] narrow = popcount(~((~C[first:last]) >> huge)); '
                'uint complement = ~popcount(~((~C[first:last]) << distance)); '
                'uint power = popcount(~((~C[first:last]) << distance)) ** 3;',
                tuple((name, None) for name in ('inner', 'outer', 'both', 'full', 'narrow', 'complement', 'power')),
                (0, 1, width, width, width % 8, maximum - width, width ** 3))

    def test_nested_counts_bounds_casts_and_array_consumers(self):
        for pattern in range(16):
            bits = [(pattern >> index) & 1 for index in range(4)]
            count = reference(bits, 1, 1, '<<', 1)
            index = bits[0]
            bound = count % 4
            distance = reference(bits, 0, 1, '>>', 1)
            self.check_exact(f'bit[4] flags="{pattern:04b}"; int first=0; int step=1; int last=3; int distance=1; '
                'array[uint,2] values={0,0}; '
                'values[popcount(~((~flags[first:first]) << 0))] = popcount(~((~flags[first:last]) << distance)) ** 2; '
                'uint saved = values[popcount(~((~flags[first:first]) << 0))]; '
                'uint nested = popcount(uint[8](popcount(~((~flags[first:last]) << distance)))); '
                'uint counted = popcount(~((~flags[first:last]) >> popcount(~(flags[first:last] >> distance)))); '
                'uint bounded = popcount(~(flags[first:int(popcount(~((~flags[first:last]) << distance)) % uint(4))] >> distance)); '
                f'uint other=values[{1 - index}]; bool same=flags=="{pattern:04b}";',
                tuple((name, None) for name in ('saved', 'nested', 'counted', 'bounded', 'other', 'same')),
                (count ** 2, count.bit_count(), reference(bits, 1, 1, '>>', distance),
                 reference(bits[:bound + 1], 0, 1, '>>', 1), 0, 1))

    def test_runtime_loops_transfers_and_fresh_sources(self):
        for constant in (False, True):
            pattern = 0b10110101
            def window(first):
                return [(pattern >> index) & 1 for index in range(first, first + 4)]
            expected = sum(reference(window(first), 1, 1, '<<', first % 3) for first in (0, 2, 3))
            source = ('const ' if constant else '') + 'bit[8] flags="10110101"; '
            source += ('int first=0; int last=3; int distance=0; uint total=0; uint count=0; '
                       'while (first<5) { last=first+3; distance=first%3; '
                       'count=popcount(~((~flags[first:last]) << distance)); first+=1; '
                       'if (first==2) { continue; } total+=count; if (first==4) { break; } }')
            self.check_exact(source, (('total', None), ('count', None), ('first', None)),
                             (expected, reference(window(3), 1, 1, '<<', 0), 4))
        self.check_exact('bit[4] flags="1111"; int first=0; int last=3; int distance=1; int visits=0; '
            'while (popcount(~(flags[first:last] >> distance)) < uint(4)) { flags>>=1; visits+=1; }',
            (('visits', None), ('flags', 4)), (3, 1))
        self.check_exact('const bit[4] C="1011"; int first=0; int last=3; uint total=0; '
            'for int i in [popcount((~C[first:first]) << 0):popcount(~(C[first:last] >> 1))] { total+=uint(i); first=99; } '
            'first=0; for int i in {popcount((~C[first:first]) << 0),popcount(~(C[first:last] >> 1))} { total+=uint(i); first=99; } '
            'array[int,2] indices={0,3}; for int first in indices { total+=popcount(~((~C[first:first]) << 0)); }',
            (('total', None), ('first', None)), (7, 99))

    def test_skipped_errors_validate_both_parities_without_execution(self):
        wrappers = ('if (false) {{ {} }}', 'while (false) {{ {} }}', 'for int i in [1:0] {{ {} }}',
                    'for int i in [loop_first:loop_last] {{ {} }}',
                    'while (true) {{ break; {} }}', 'for int i in [0:1] {{ continue; {} }}')
        for inner, outer in ((1, 0), (0, 1), (1, 1), (2, 2)):
            for argument, count in (('flags[first:0:last]', '-1'), ('C[first/zero:last]', 'uint(-1)')):
                call = f'count=popcount({operand(inner, outer, "<<", count, argument)});'
                for wrapper in wrappers:
                    self.check_exact('bit[4] flags="1111"; const bit[4] C="0000"; uint count=6; '
                        'int first=99; int last=0; int zero=0; int loop_first=2; int loop_last=1; ' + wrapper.format(call),
                        (('count', None), ('flags', 4)), (6, 15))
        self.check_exact('bit[4] flags="1010"; int first=99; int last=0; int zero=0; '
            'bool left=false && bool(popcount(~((~flags[first:0:last]) << -1))); '
            'bool right=true || bool(popcount(~~((~~flags[first/zero:last]) >> uint(-1))));',
            (('left', None), ('right', None)), (0, 1))

    def test_range_and_count_errors_precede_counting_and_writes(self):
        cls = self.module.QASM2QCXConverter
        cases = (('last=8;', 'u', 16, 'assertion failed'), ('step=0;', '1/zero', 16, 'assertion failed'),
                 ('first=3; last=0;', 'u', 16, 'assertion failed'), ('', '-1', 16, 'assertion failed'),
                 ('', 'minimum', 16, 'assertion failed'), ('', '1/zero', 16, 'division by zero'),
                 ('', 'u', 2, 'assertion failed'))
        for constant in (False, True):
            for inner, outer in ((1, 0), (0, 1), (1, 1), (2, 2)):
                for update, count, capacity, error in cases:
                    source = ('const ' if constant else '') + 'bit[8] flags="11111111"; '
                    source += ('int first=0; int step=1; int last=3; int zero=0; uint result=6; '
                               f'int minimum={cls.QCX_INT_MIN}; uint u={cls.QCX_UINT_MAX}; '
                               + update + f'result=popcount({operand(inner, outer, ">>", count)});')
                    program = self.module.openqasm3.parser.parse('OPENQASM 3; ' + source)
                    with patch.object(cls, 'MAX_RUNTIME_BIT_RANGE_POSITIONS', capacity):
                        converter = cls(program); converter.visit(program)
                    self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())
                    destination = converter._QASM2QCXConverter__capitalize_variable_name('result')
                    lines = []
                    for line in converter:
                        if line.startswith('LET QASM2QCX_INT_') and ' := QASM2QCX_POSITIONS_' in line:
                            lines.append('PRINTLN 22222')
                        if line.startswith(f'LET {destination} := QASM2QCX_'):
                            lines.append('PRINTLN 11111')
                        lines.append(line)
                    result = self.execute(lines)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(error, result.stderr)
                    self.assertNotIn('runtime error:', result.stderr)
                    self.assertEqual(result.stdout.split(), [])
                    type(self).checked += 1

    def test_measurement_feedback_and_quantum_parameters_with_fusion(self):
        for _ in range(16):
            self.check_exact('include "stdgates.inc"; qubit[2] q; reset q; h q[0]; bit[2] measured=measure q; '
                'int first=0; int last=1; int distance=1; '
                'uint inner=popcount((~measured[first:last]) << distance); '
                'uint outer=popcount(~(measured[first:last] << distance)); '
                'uint both=popcount(~((~measured[first:last]) << distance)); '
                'bool same=inner==uint(!bool(measured[0])) && outer==uint(2)-uint(measured[0]) && both==uint(1)+uint(measured[0]); '
                'if (both==uint(2)) { x q[1]; } bit[2] observed=measure q; '
                'bool feedback=observed[1]==measured[0] && observed[0]==measured[0];',
                (('same', None), ('feedback', None)), (1, 1))
        from qasm2qcx_u_power_numerical import BackendAuditTests
        for inner, outer, count in ((1, 0, 1), (0, 1, 2), (1, 1, 3)):
            phase = cmath.exp(count * 1j * math.pi / 8)
            for fused in (False, True):
                value = 'popcount(' + operand(inner, outer, '<<', 'distance', 'flags[first:last]') + ')'
                for gate in (f'p(float({value}) * pi/8) q;', f'pow({value}) @ p(pi/8) q;'):
                    source = ('include "stdgates.inc"; qubit[2] q; bit[4] flags="1011"; '
                              'int first=0; int last=3; int distance=1; h q; ' + gate)
                    BackendAuditTests.check(self, self.program(source, fused=fused), (0.5, phase/2, phase/2, phase*phase/2))


def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    args = parser.parse_args()
    BackendTests.bra = args.bra.resolve()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(BackendTests))
    print(f'mixed complement/shift runtime range popcount programs checked: {BackendTests.checked}')
    raise SystemExit(not result.wasSuccessful())


if __name__ == '__main__':
    main()
