#!/usr/bin/env python3
"""Audit rotations inside runtime bit-register range population counts (#489)."""

import argparse
import cmath
import math
import pathlib
import resource
import unittest
from unittest.mock import patch

import qasm2qcx_angle_bit_cast_numerical as casts


def operand(inner, outer, function, distance, source='flags[first:step:last]'):
    return '~' * outer + function + '(' + '~' * inner + source + ', ' + distance + ')'


def reference(bits, inner, outer, function, distance):
    """Flip, circularly permute, then flip individual little-endian bits."""
    assert bits
    flipped = [bit ^ (inner % 2) for bit in bits]
    distance %= len(bits)
    if function == 'rotl':
        rotated = flipped[-distance:] + flipped[:-distance] if distance else flipped
    else:
        rotated = flipped[distance:] + flipped[:distance]
    return sum(bit ^ (outer % 2) for bit in rotated)


class BackendTests(unittest.TestCase):
    bra = None
    checked = 0
    setUp = casts.BackendTests.setUp
    execute = casts.BackendTests.execute
    program = casts.BackendTests.program
    check_exact = casts.BackendTests.check_exact

    def check_case(self, size, pattern, constant, first, step, last, distances,
                   parities=((0, 0), (1, 0), (0, 1), (1, 1))):
        positions = list(range(first, last + (1 if step > 0 else -1), step))
        assert positions and all(-size <= index < size for index in positions)
        bits = [(pattern >> (index % size)) & 1 for index in positions]
        source = ('const ' if constant else '') + f'bit[{size}] flags="{pattern:0{size}b}"; '
        source += f'int first={first}; int step={step}; int last={last}; int distance=0; '
        outputs, expected = [], []
        for case, (distance, literal) in enumerate(distances):
            if literal is None:
                source += f'distance={distance}; '
            count = 'distance' if literal is None else literal
            for function in ('rotl', 'rotr'):
                for inner, outer in parities:
                    name = f'out{case}_{function}_{inner}_{outer}'
                    source += f'uint {name}=popcount({operand(inner, outer, function, count)}); '
                    outputs.append((name, None))
                    expected.append(reference(bits, inner, outer, function, distance))
        source += f'bool same=flags=="{pattern:0{size}b}"; uint fresh=popcount(flags[first:step:last]);'
        self.check_exact(source, (*outputs, ('same', None), ('fresh', None)), (*expected, 1, sum(bits)))

    def test_every_native_width_patterns_parities_and_directions(self):
        native = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        for size in (*range(1, native + 2), 129):
            for pattern in (0, (1 << size) - 1, int(('10110010' * size)[:size], 2)):
                for constant in (False, True):
                    self.check_case(size, pattern, constant, 0, 1, size - 1,
                                    ((0, None), (1, None), (size + 1, None)))

    def test_native_endpoints_literals_and_even_complement_chains(self):
        cls = self.module.QASM2QCXConverter
        for size in (1, 4, cls.QCX_UINT_WIDTH + 1, 129):
            pattern = int(('11010010' * size)[:size], 2)
            for constant in (False, True):
                self.check_case(size, pattern, constant, size - 1, -2, 0,
                    ((cls.QCX_INT_MIN, None), (cls.QCX_INT_MAX, None), (-1, '-1'),
                     (cls.QCX_INT_MIN, str(cls.QCX_INT_MIN)), (cls.QCX_INT_MAX, str(cls.QCX_INT_MAX)),
                     (cls.QCX_INT_MAX, f'uint({cls.QCX_INT_MAX})')),
                    ((2, 2), (3, 2), (2, 3), (3, 3)))

    def test_reverse_strides_negative_aliases_and_selected_lengths(self):
        for size, first, step, last in ((8, -1, -2, -7), (8, -8, 1, 7),
                                      (129, 128, -3, 119), (8, 2, 4, 7)):
            pattern = int(('10010110' * size)[:size], 2)
            for constant in (False, True):
                self.check_case(size, pattern, constant, first, step, last,
                                ((-129, None), (-1, None), (8, None)))

    def test_distance_promotions_nested_calls_and_scalar_consumers(self):
        for distance in ('true', 'false', 'condition', 'u', 'uint[1](1)', 'int(u)-2',
                         'indices[popcount(rotl(flags[first:last], -1))]',
                         'popcount(~rotr(~flags[first:last], u))'):
            self.check_exact('bit[4] flags="1011"; int first=0; int last=3; uint u=2; bool condition=true; '
                'array[int,4] indices={-1,0,1,2}; array[uint,4] values={0,0,0,0}; '
                f'uint result=popcount(~rotr(flags[first:last], {distance})); '
                'values[popcount(rotl(flags[first:last], true))] = result; '
                'uint stored=values[3]; '
                'uint[1] narrow=popcount(rotl(flags[first:last], u)); '
                'bit[2] packed=bit[2](uint[2](popcount(rotl(flags[first:last], false)))); '
                'uint nested=popcount(rotl(flags[popcount(rotl(flags[first:first], 0)):last], u)); '
                'float root=sqrt(float(result)); bool valid=root==1.0;',
                (('result', None), ('stored', None), ('narrow', None), ('packed', 2), ('nested', None), ('valid', None)),
                (1, 1, 1, 3, 2, 1))

    def test_fresh_values_loops_transfers_and_captured_bounds(self):
        for constant in (False, True):
            source = ('const ' if constant else '') + 'bit[8] flags="10110110"; '
            total = sum(reference([(0xB6 >> p) & 1 for p in range(first, first + 4)],
                                  1, 0, 'rotl', first - 2) for first in (0, 2, 3))
            self.check_exact(source + 'int first=0; int last=3; int distance=0; uint total=0; uint count=0; '
                'while (first<5) { last=first+3; distance=first-2; '
                'count=popcount(rotl(~flags[first:last], distance)); first+=1; '
                'if (first==2) { continue; } total+=count; if (first==4) { break; } }',
                (('total', None), ('count', None), ('first', None)),
                (total, reference([(0xB6 >> p) & 1 for p in range(3, 7)], 1, 0, 'rotl', 1), 4))
        self.check_exact('bit[4] flags="1111"; int first=0; int last=3; int visits=0; '
            'while (popcount(rotl(flags[first:last], visits))>uint(0)) { flags>>=1; visits+=1; }',
            (('visits', None), ('flags', 4)), (4, 0))
        self.check_exact('const bit[4] C="1011"; int first=0; int last=3; uint total=0; '
            'for int i in [popcount(rotl(C[first:first], -1)):popcount(rotr(C[first:last], 1))] '
            '{ total+=uint(i); first=99; } first=0; '
            'for int i in {popcount(rotl(C[first:first], -1)),popcount(~rotr(C[first:last], 1))} '
            '{ total+=uint(i); first=99; } '
            'array[int,2] indices={0,3}; for int first in indices '
            '{ total+=popcount(~rotr(~C[first:first], first)); }',
            (('total', None), ('first', None)), (10, 99))

    def test_skipped_errors_and_short_circuiting(self):
        wrappers = ('if (false) {{ {} }}', 'while (false) {{ {} }}', 'for int i in [1:0] {{ {} }}',
                    'for int i in [loop_first:loop_last] {{ {} }}',
                    'while (true) {{ break; {} }}', 'for int i in [0:1] {{ continue; {} }}')
        base = ('bit[4] flags="1011"; const bit[4] C="0000"; uint result=7; int first=99; '
                'int step=0; int last=0; int zero=0; uint u=uint(-1); int loop_first=2; int loop_last=1; ')
        for inner, outer in ((1, 0), (0, 1), (1, 1), (2, 2)):
            for source, distance in (('flags[first:step:last]', 'u'), ('C[first/zero:last]', '1/zero')):
                value = f'popcount({operand(inner, outer, "rotl", distance, source)})'
                for wrapper in wrappers:
                    self.check_exact(base + wrapper.format('result=' + value + ';'), (('result', None),), (7,))
                self.check_exact(base + f'bool a=false && bool({value}); bool b=true || bool({value});',
                                 (('a', None), ('b', None)), (0, 1))
        # A valid range must not make skipped distance conversions run.
        for distance in ('uint(-1)', '1/zero', str(10 ** 70)):
            value = f'popcount(rotl(flags[last:last], {distance}))'
            self.check_exact(base + f'if (false) {{ result={value}; }} '
                f'bool a=false && bool({value});', (('result', None), ('a', None)), (7, 0))

    def test_range_and_distance_errors_precede_counting_and_writes(self):
        cls = self.module.QASM2QCXConverter
        cases = (('last=8;', '1/zero', 16, 'assertion failed'),
                 ('step=0;', '1/zero', 16, 'assertion failed'),
                 ('first=3; last=0;', 'u', 16, 'assertion failed'),
                 ('', 'u', 2, 'assertion failed'), ('', '1/zero', 16, 'division by zero'),
                 ('', 'u', 16, None), ('', 'uint(-1)', 16, None),
                 ('', f'uint({cls.QCX_INT_MAX + 1})', 16, None), ('', str(10 ** 70), 16, None))
        for constant in (False, True):
            for inner, outer in ((0, 0), (1, 0), (0, 1), (1, 1)):
                for update, distance, capacity, error in cases:
                    source = ('const ' if constant else '') + 'bit[8] flags="00000000"; '
                    source += ('int first=0; int step=1; int last=3; int zero=0; uint result=6; '
                               'uint u=uint(-1); ' + update
                               + f'result=popcount({operand(inner, outer, "rotr", distance)});')
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
                    if error is not None:
                        self.assertIn(error, result.stderr)
                    self.assertNotIn('runtime error:', result.stderr)
                    self.assertEqual(result.stdout.split(), [])
                    type(self).checked += 1

    def test_measurement_feedback_and_quantum_parameters_with_fusion(self):
        for _ in range(16):
            self.check_exact('include "stdgates.inc"; qubit[2] q; reset q; h q[0]; bit[2] measured=measure q; '
                'int first=0; int last=1; int distance=-3; '
                'uint original=popcount(rotl(measured[first:last], distance)); '
                'uint flipped=popcount(~rotr(measured[first:last], distance)); '
                'bool same=original==uint(measured[0])+uint(measured[1]) && original+flipped==uint(2); '
                'if (original==uint(1)) { x q[1]; } bit[2] observed=measure q; '
                'bool feedback=observed[1]==measured[0] && observed[0]==measured[0];',
                (('same', None), ('feedback', None)), (1, 1))
        from qasm2qcx_u_power_numerical import BackendAuditTests
        for inner, outer, count in ((1, 0, 1), (0, 1, 1), (1, 1, 3)):
            phase = cmath.exp(count * 1j * math.pi / 8)
            for fused in (False, True):
                value = 'popcount(' + operand(inner, outer, 'rotl', 'distance', 'flags[first:last]') + ')'
                for gate in (f'p(float({value}) * pi/8) q;', f'pow({value}) @ p(pi/8) q;'):
                    source = ('include "stdgates.inc"; qubit[2] q; bit[4] flags="1011"; '
                              'int first=0; int last=3; int distance=-1; h q; ' + gate)
                    BackendAuditTests.check(self, self.program(source, fused=fused),
                                            (0.5, phase/2, phase/2, phase*phase/2))


def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    args = parser.parse_args()
    BackendTests.bra = args.bra.resolve()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(BackendTests))
    print(f'rotated runtime range popcount programs checked: {BackendTests.checked}')
    raise SystemExit(not result.wasSuccessful())


if __name__ == '__main__':
    main()
