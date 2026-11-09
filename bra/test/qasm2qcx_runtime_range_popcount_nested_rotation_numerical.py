#!/usr/bin/env python3
"""Audit nested rotations in runtime-range population counts (#495)."""

import argparse
import cmath
import itertools
import math
import pathlib
import resource
import unittest
from unittest.mock import patch

import qasm2qcx_angle_bit_cast_numerical as casts


def operand(functions, distances, complements, source='flags[first:step:last]'):
    assert len(functions) == len(distances) == len(complements) - 1
    value = '~' * complements[0] + source
    for function, distance, depth in zip(functions, distances, complements[1:]):
        value = '~' * depth + f'{function}({value}, {distance})'
    return value


def reference(bits, functions, distances, complements):
    """Rotate and flip each intermediate little-endian bit vector independently."""
    assert bits and len(functions) == len(distances) == len(complements) - 1
    value = [bit ^ (complements[0] % 2) for bit in bits]
    for function, distance, depth in zip(functions, distances, complements[1:]):
        distance %= len(value)
        if function == 'rotl':
            value = value[-distance:] + value[:-distance] if distance else value
        else:
            value = value[distance:] + value[:distance]
        value = [bit ^ (depth % 2) for bit in value]
    return sum(value)


class BackendTests(unittest.TestCase):
    bra = None
    checked = 0
    setUp = casts.BackendTests.setUp
    execute = casts.BackendTests.execute
    program = casts.BackendTests.program
    check_exact = casts.BackendTests.check_exact

    def test_documented_nested_rotation_example(self):
        document = (pathlib.Path(__file__).resolve().parents[1] / 'qcx/qasm2qcx.md').read_text()
        marker = '<!-- runtime-range-popcount-nested-rotation-example -->'
        source = document.split(marker, 1)[1].split('```qasm\n', 1)[1].split('```', 1)[0]
        source = source.replace('OPENQASM 3.0;', '', 1)
        self.check_exact(source, tuple((name, None) for name in
            ('plain', 'middle', 'both', 'inner', 'even', 'promoted', 'longer', 'skipped', 'fresh')),
            (3, 1, 3, 1, 3, 3, 3, 0, 3))

    def check_case(self, size, pattern, constant, first, step, last, chains, cases, modes=None, parities=None):
        depth = len(chains[0])
        modes = modes or ('int',) * depth
        parities = tuple(itertools.product((0, 1), repeat=depth + 1)) if parities is None else parities
        positions = list(range(first, last + (1 if step > 0 else -1), step))
        assert positions and all(-size <= position < size for position in positions)
        bits = [(pattern >> (position % size)) & 1 for position in positions]
        source = ('const ' if constant else '') + f'bit[{size}] flags="{pattern:0{size}b}"; '
        source += f'int first={first}; int step={step}; int last={last}; '
        source += ''.join(f'{mode} distance{i}=0; ' for i, mode in enumerate(modes) if mode != 'literal')
        outputs, expected = [], []
        for case, distances in enumerate(cases):
            counts = []
            for i, (distance, mode) in enumerate(zip(distances, modes)):
                if mode == 'literal':
                    counts.append(str(distance))
                else:
                    source += f'distance{i}={distance}; '
                    counts.append(f'distance{i}')
            for chain, functions in enumerate(chains):
                for mask, complements in enumerate(parities):
                    name = f'out{case}_{chain}_{mask}'
                    source += f'uint {name}=popcount({operand(functions, counts, complements)}); '
                    outputs.append((name, None))
                    expected.append(reference(bits, functions, distances, complements))
        source += f'bool same=flags=="{pattern:0{size}b}"; uint fresh=popcount(flags[first:step:last]);'
        self.check_exact(source, (*outputs, ('same', None), ('fresh', None)), (*expected, 1, sum(bits)))

    def test_every_native_width_patterns_complement_placements_and_directions(self):
        native = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        chains = tuple(itertools.product(('rotl', 'rotr'), repeat=2))
        for size in (*range(1, native + 2), 129):
            for pattern in (0, (1 << size) - 1, int(('10110010' * size)[:size], 2)):
                for constant in (False, True):
                    self.check_case(size, pattern, constant, 0, 1, size - 1, chains, ((0, 0), (1, -1)))

    def test_exhaustive_small_vectors_and_signed_distances(self):
        chains = tuple(itertools.product(('rotl', 'rotr'), repeat=2))
        for size in (1, 2):
            for pattern in range(1 << size):
                for constant in (False, True):
                    self.check_case(size, pattern, constant, 0, 1, size - 1, chains,
                                    tuple(itertools.product((-1, 0, 1), repeat=2)))

    def test_long_chains_even_groups_and_endpoint_distances_without_summing(self):
        cls = self.module.QASM2QCXConverter
        for depth in (3, 4, 8, 16):
            chains = (('rotl',) * depth, ('rotr',) * depth,
                      tuple(('rotl', 'rotr')[i % 2] for i in range(depth)))
            parities = tuple(tuple(2 + ((mask >> (i % 3)) & 1) for i in range(depth + 1)) for mask in range(8))
            for constant in (False, True):
                self.check_case(8, 0xB3, constant, -8, 1, 7, chains,
                    ((cls.QCX_INT_MIN,) * depth, (cls.QCX_INT_MAX,) * depth, (-1,) * depth),
                    modes=('literal',) * depth, parities=parities)

    def test_reverse_strides_negative_aliases_and_selected_lengths(self):
        chains = tuple(itertools.product(('rotl', 'rotr'), repeat=2))
        for size, first, step, last in ((8, -1, -2, -7), (8, -8, 1, 7),
                                      (129, 128, -3, 119), (8, 2, 4, 7)):
            pattern = int(('10010110' * size)[:size], 2)
            for constant in (False, True):
                self.check_case(size, pattern, constant, first, step, last, chains,
                                ((-129, 129), (-1, 0), (0, -1)))

    def test_int_endpoints_and_checked_uint_promotions(self):
        cls = self.module.QASM2QCXConverter
        chains = tuple(itertools.product(('rotl', 'rotr'), repeat=2))
        for size in (1, 4, cls.QCX_UINT_WIDTH + 1, 129):
            for constant in (False, True):
                self.check_case(size, (1 << size) - 1, constant, 0, 1, size - 1, chains,
                    ((cls.QCX_INT_MIN, cls.QCX_INT_MIN), (cls.QCX_INT_MAX, cls.QCX_INT_MAX),
                     (cls.QCX_INT_MIN, cls.QCX_INT_MAX)))
                self.check_case(size, (1 << size) - 1, constant, 0, 1, size - 1, chains,
                    ((cls.QCX_INT_MAX, cls.QCX_INT_MIN), (0, cls.QCX_INT_MAX)), modes=('uint', 'int'))

    def test_nested_calls_bool_distances_arrays_casts_and_bounds(self):
        for distance in ('true', 'false', 'condition', 'u', 'uint[1](1)',
                         'indices[popcount(rotl(rotr(flags[first:last], -1), 1))]',
                         'popcount(~rotl(~rotr(flags[first:last], -1), 1))'):
            self.check_exact('bit[4] flags="1011"; int first=0; int last=3; uint u=2; bool condition=true; '
                'array[int,4] indices={-1,0,1,2}; array[uint,4] values={0,0,0,0}; '
                f'uint result=popcount(~rotl(rotr(flags[first:last], {distance}), -1)); '
                'values[popcount(rotl(rotr(flags[first:last], true), false))]=result; uint stored=values[3]; '
                'uint[1] narrow=popcount(rotl(rotr(flags[first:last], u), -1)); '
                'bit[2] packed=bit[2](uint[2](popcount(rotl(rotr(flags[first:last], false), true)))); '
                'uint nested=popcount(rotl(rotr(flags[popcount(rotl(rotr(flags[first:first], 0), 0)):last], u), -1)); '
                'float root=sqrt(float(result)); bool valid=root==1.0;',
                (('result', None), ('stored', None), ('narrow', None), ('packed', 2), ('nested', None), ('valid', None)),
                (1, 1, 1, 3, 2, 1))

    def test_fresh_loop_values_transfers_and_captured_bounds(self):
        for constant in (False, True):
            bits = [(0xB6 >> p) & 1 for p in range(8)]
            total = sum(reference(bits[first:first+4], ('rotr', 'rotl'), (first - 2, 1), (1, 1, 1))
                        for first in (0, 2, 3))
            source = ('const ' if constant else '') + 'bit[8] flags="10110110"; '
            self.check_exact(source + 'int first=0; int last=3; int a=0; uint count=0; uint total=0; '
                'while (first<5) { last=first+3; a=first-2; count=popcount(~rotl(~rotr(~flags[first:last], a), 1)); '
                'first+=1; if (first==2) { continue; } total+=count; if (first==4) { break; } }',
                (('total', None), ('count', None), ('first', None)),
                (total, reference(bits[3:7], ('rotr', 'rotl'), (1, 1), (1, 1, 1)), 4))
        self.check_exact('bit[4] flags="1111"; int first=0; int last=3; int visits=0; '
            'while (popcount(rotl(rotr(flags[first:last], visits), -1))>uint(0)) { flags>>=1; visits+=1; }',
            (('visits', None), ('flags', 4)), (4, 0))
        self.check_exact('const bit[4] C="1011"; int first=0; int last=3; uint total=0; '
            'for int i in [popcount(rotl(rotr(C[first:first], -1), 1)):popcount(rotl(rotr(C[first:last], -1), 1))] '
            '{ total+=uint(i); first=99; } first=0; '
            'for int i in {popcount(rotl(rotr(C[first:first], -1), 1)),popcount(~rotl(rotr(C[first:last], -1), 1))} '
            '{ total+=uint(i); first=99; } array[int,2] indices={0,3}; '
            'for int first in indices { total+=popcount(rotl(rotr(C[first:first], first), -1)); }',
            (('total', None), ('first', None)), (10, 99))

    def test_skipped_distance_errors_and_short_circuiting(self):
        wrappers = ('if (false) {{ {} }}', 'while (false) {{ {} }}', 'for int i in [1:0] {{ {} }}',
                    'for int i in [loop_first:loop_last] {{ {} }}',
                    'while (true) {{ break; {} }}', 'for int i in [0:1] {{ continue; {} }}')
        base = ('bit[4] flags="1011"; const bit[4] C="0000"; uint result=7; int first=0; int last=3; '
                'int zero=0; uint u=uint(-1); int loop_first=2; int loop_last=1; ')
        for complements in ((0, 1, 0), (1, 1, 1), (2, 2, 2), (3, 3, 3)):
            for source, distances in (('flags[first:last]', ('0', 'u')),
                                      ('C[first:last]', ('-1', '1/zero')),
                                      ('C[first:0:last]', ('u', '1/zero'))):
                value = 'popcount(' + operand(('rotl', 'rotr'), distances, complements, source) + ')'
                for wrapper in wrappers:
                    self.check_exact(base + wrapper.format('result=' + value + ';'), (('result', None),), (7,))
                self.check_exact(base + f'bool a=false && bool({value}); bool b=true || bool({value});',
                                 (('a', None), ('b', None)), (0, 1))

    def test_range_and_distance_failures_precede_main_counting_and_writes(self):
        cls = self.module.QASM2QCXConverter
        cases = (('last=8;', ('1/zero', 'u'), 16, 'assertion failed'),
                 ('step=0;', ('1/zero', 'u'), 16, 'assertion failed'),
                 ('first=3; last=0;', ('u', '1/zero'), 16, 'assertion failed'),
                 ('', ('0', '1/zero'), 2, 'assertion failed'),
                 ('', ('u', '1/zero'), 16, None), ('', ('0', 'u'), 16, None),
                 ('', ('minimum', '1/zero'), 16, 'division by zero'),
                 ('', ('0', f'uint({cls.QCX_INT_MAX + 1})'), 16, None),
                 ('', ('0', str(10 ** 70)), 16, None),
                 ('', ('0', 'minimum', 'u'), 16, None),
                 ('', ('0', 'minimum', '1/zero'), 16, 'division by zero'))
        for constant in (False, True):
            for complements in ((0, 1, 0), (1, 1, 1), (2, 2, 2), (3, 3, 3)):
                for update, distances, capacity, error in cases:
                    functions = ('rotl', 'rotr', 'rotl')[:len(distances)]
                    parities = complements if len(distances) == 2 else complements + (complements[-1],)
                    source = ('const ' if constant else '') + 'bit[8] flags="00000000"; '
                    source += ('int first=0; int step=1; int last=3; int zero=0; uint u=uint(-1); '
                               f'int minimum={cls.QCX_INT_MIN}; uint result=6; ' + update
                               + f'result=popcount({operand(functions, distances, parities)});')
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
                    else:
                        self.assertNotIn('division by zero', result.stderr)
                    self.assertNotIn('runtime error:', result.stderr)
                    self.assertEqual(result.stdout.split(), [])
                    type(self).checked += 1

    def test_measurement_feedback_and_normal_fused_gate_parameters(self):
        for _ in range(16):
            self.check_exact('include "stdgates.inc"; qubit[2] q; reset q; h q[0]; bit[2] measured=measure q; '
                'int first=0; int last=1; int a=-3; uint b=2; '
                'uint count=popcount(~rotl(rotr(measured[first:last], a), b)); '
                'bool valid=count==uint(2)-uint(measured[0])-uint(measured[1]); '
                'if (count==uint(1)) { x q[1]; } bit[2] observed=measure q; '
                'bool feedback=observed[1]==measured[0] && observed[0]==measured[0];',
                (('valid', None), ('feedback', None)), (1, 1))
        from qasm2qcx_u_power_numerical import BackendAuditTests
        for complements, count in (((0, 0, 0), 3), ((0, 1, 0), 1), ((1, 1, 0), 3), ((1, 1, 1), 1)):
            phase = cmath.exp(count * 1j * math.pi / 8)
            for fused in (False, True):
                value = 'popcount(' + operand(('rotl', 'rotr'), ('a', 'b'), complements, 'flags[first:last]') + ')'
                for gate in (f'p(float({value}) * pi/8) q;', f'pow({value}) @ p(pi/8) q;'):
                    source = ('include "stdgates.inc"; qubit[2] q; bit[4] flags="1011"; '
                              'int first=0; int last=3; int a=-1; uint b=2; h q; ' + gate)
                    BackendAuditTests.check(self, self.program(source, fused=fused),
                                            (0.5, phase/2, phase/2, phase*phase/2))


def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    args = parser.parse_args()
    BackendTests.bra = args.bra.resolve()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(BackendTests))
    print(f'nested-rotation runtime range popcount programs checked: {BackendTests.checked}')
    raise SystemExit(not result.wasSuccessful())


if __name__ == '__main__':
    main()
