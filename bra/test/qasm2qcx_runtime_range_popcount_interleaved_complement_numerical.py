#!/usr/bin/env python3
"""Audit interleaved complements/shifts in runtime-range population counts (#493)."""

import argparse
import cmath
import itertools
import math
import pathlib
import resource
import unittest
from unittest.mock import patch

import qasm2qcx_angle_bit_cast_numerical as casts


def operand(directions, distances, complements, source='flags[first:step:last]'):
    assert len(directions) == len(distances) == len(complements) - 1
    value = '~' * complements[0] + source
    for direction, distance, depth in zip(directions, distances, complements[1:]):
        value = '~' * depth + f'(({value}) {direction} ({distance}))'
    return value


def reference(bits, directions, distances, complements):
    """Construct, shift, and complement every intermediate little-endian vector."""
    assert bits and all(distance >= 0 for distance in distances)
    assert len(directions) == len(distances) == len(complements) - 1
    value = [bit ^ (complements[0] % 2) for bit in bits]
    for direction, distance, depth in zip(directions, distances, complements[1:]):
        if distance >= len(value):
            value = [0] * len(value)
        elif direction == '<<':
            value = [0] * distance + value[:len(value) - distance]
        else:
            value = value[distance:] + [0] * distance
        value = [bit ^ (depth % 2) for bit in value]
    return sum(value)


class BackendTests(unittest.TestCase):
    bra = None
    checked = 0
    setUp = casts.BackendTests.setUp
    execute = casts.BackendTests.execute
    program = casts.BackendTests.program
    check_exact = casts.BackendTests.check_exact

    def check_case(self, size, pattern, constant, first, step, last, chains, cases, modes=None, parities=None):
        depth = len(chains[0])
        modes = modes or ('uint',) * depth
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
            for chain, directions in enumerate(chains):
                for mask, complements in enumerate(parities):
                    name = f'out{case}_{chain}_{mask}'
                    source += f'uint {name}=popcount({operand(directions, counts, complements)}); '
                    outputs.append((name, None))
                    expected.append(reference(bits, directions, distances, complements))
        source += f'bool same=flags=="{pattern:0{size}b}"; uint fresh=popcount(flags[first:step:last]);'
        self.check_exact(source, (*outputs, ('same', None), ('fresh', None)), (*expected, 1, sum(bits)))

    def test_every_native_width_patterns_parities_and_directions(self):
        native = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        chains = tuple(itertools.product(('<<', '>>'), repeat=2))
        for size in (*range(1, native + 2), 129):
            for pattern in (0, (1 << size) - 1, int(('10110010' * size)[:size], 2)):
                for constant in (False, True):
                    self.check_case(size, pattern, constant, 0, 1, size - 1, chains, ((1, 1), (size, 1)))

    def test_exhaustive_small_vectors_and_distances(self):
        chains = tuple(itertools.product(('<<', '>>'), repeat=2))
        for size in (1, 2):
            for pattern in range(1 << size):
                for constant in (False, True):
                    self.check_case(size, pattern, constant, 0, 1, size - 1, chains,
                                    tuple(itertools.product(range(size + 2), repeat=2)))

    def test_long_chains_even_groups_and_generated_ones_after_complete_bit_loss(self):
        for depth in (3, 4, 8, 16):
            chains = (('<<',) * depth, ('>>',) * depth,
                      tuple(('<<', '>>')[i % 2] for i in range(depth)))
            parities = tuple(tuple(2 + ((mask >> (i % 3)) & 1) for i in range(depth + 1)) for mask in range(8))
            for constant in (False, True):
                self.check_case(8, 0xB2, constant, -8, 1, 7, chains,
                    ((1,) * depth, (16,) + (0,) * (depth - 1), (0,) * (depth - 1) + (17,)),
                    modes=('literal',) * depth, parities=parities)

    def test_reverse_strides_negative_aliases_and_short_selected_lengths(self):
        chains = tuple(itertools.product(('<<', '>>'), repeat=2))
        for size, first, step, last in ((8, -1, -2, -7), (8, -8, 1, 7),
                                      (129, 128, -3, 119), (8, 2, 4, 7)):
            pattern = int(('10010110' * size)[:size], 2)
            for constant in (False, True):
                self.check_case(size, pattern, constant, first, step, last, chains,
                                ((0, 1), (1, 0), (1, 2), (3, 1)), modes=('int', 'uint'))

    def test_endpoints_huge_literals_and_storage_reuse(self):
        cls = self.module.QASM2QCXConverter
        chains = tuple(itertools.product(('<<', '>>'), repeat=2))
        for size in (1, 4, cls.QCX_UINT_WIDTH + 1, 129):
            for constant in (False, True):
                self.check_case(size, (1 << size) - 1, constant, 0, 1, size - 1, chains,
                    ((cls.QCX_INT_MAX, 0), (0, cls.QCX_UINT_MAX), (cls.QCX_INT_MAX, cls.QCX_UINT_MAX), (0, 0)),
                    modes=('int', 'uint'))
                self.check_case(size, (1 << size) - 1, constant, 0, 1, size - 1, chains,
                    ((10 ** 70, 0), (0, 10 ** 70), (10 ** 70, 10 ** 70), (1, 1)), modes=('literal', 'literal'))

    def test_nested_calls_array_consumers_narrow_results_and_bounds(self):
        self.check_exact('bit[4] flags="0000"; int first=0; int last=3; int a=1; uint b=1; '
            'array[uint,4] values={0,0,0,0}; uint count=popcount((~(flags[first:last] << a)) >> b); '
            'values[popcount((~(flags[first:last] << a)) >> b)]=count; uint stored=values[3]; '
            'uint[1] narrow=popcount((~(flags[first:last] << a)) >> b); '
            'bit[2] packed=bit[2](uint[2](popcount((~(flags[first:last] << a)) >> b))); '
            'uint nested=popcount((~(flags[first:popcount((~(flags[first:last] << a)) >> b)] << a)) >> b); '
            'uint distance=popcount((~(flags[first:last] << popcount((~(flags[first:first] << 1)) >> 0))) >> b); '
            'float root=sqrt(float(count)); bool valid=root>1.0;',
            (('count', None), ('stored', None), ('narrow', None), ('packed', 2),
             ('nested', None), ('distance', None), ('valid', None)), (3, 3, 1, 3, 3, 3, 1))

    def test_fresh_loop_values_transfers_and_captured_bounds(self):
        for constant in (False, True):
            bits = [(0xB6 >> p) & 1 for p in range(8)]
            total = sum(reference(bits[first:first+4], ('<<', '>>'), (first % 3, 1), (1, 1, 1))
                        for first in (0, 2, 3))
            source = ('const ' if constant else '') + 'bit[8] flags="10110110"; '
            self.check_exact(source + 'int first=0; int last=3; int a=0; uint count=0; uint total=0; '
                'while (first<5) { last=first+3; a=first%3; count=popcount(~((~((~flags[first:last]) << a)) >> 1)); '
                'first+=1; if (first==2) { continue; } total+=count; if (first==4) { break; } }',
                (('total', None), ('count', None), ('first', None)),
                (total, reference(bits[3:7], ('<<', '>>'), (0, 1), (1, 1, 1)), 4))
        self.check_exact('bit[4] flags="1111"; int first=0; int last=3; int visits=0; '
            'while (popcount((~~(flags[first:last] << 1)) >> 1)>uint(0)) { flags>>=1; visits+=1; }',
            (('visits', None), ('flags', 4)), (4, 0))
        self.check_exact('const bit[4] C="0000"; int first=0; int last=3; uint total=0; '
            'for int i in [popcount((~(C[first:first] << 1)) >> 0):popcount((~(C[first:last] << 1)) >> 1)] '
            '{ total+=uint(i); first=99; } first=0; '
            'for int i in {popcount((~(C[first:first] << 1)) >> 0),popcount((~(C[first:last] << 1)) >> 1)} '
            '{ total+=uint(i); first=99; } array[int,2] indices={0,3}; '
            'for int first in indices { total+=popcount((~(C[first:first] << 1)) >> 0); }',
            (('total', None), ('first', None)), (12, 99))

    def test_skipped_domains_preserve_original_destination(self):
        wrappers = ('if (false) {{ {} }}', 'while (false) {{ {} }}', 'for int i in [1:0] {{ {} }}',
                    'for int i in [loop_first:loop_last] {{ {} }}',
                    'while (true) {{ break; {} }}', 'for int i in [0:1] {{ continue; {} }}')
        base = ('bit[4] flags="0000"; const bit[4] C="1111"; uint result=7; int first=0; int last=3; '
                'int zero=0; uint huge=uint(-1); int negative=-1; int loop_first=2; int loop_last=1; ')
        for complements in ((0, 1, 0), (1, 1, 1), (2, 2, 2), (3, 3, 3)):
            for source, counts in (('flags[first:last]', ('huge', 'negative')),
                                   ('C[first:last]', (str(10 ** 70), '1/zero')),
                                   ('C[first:0:last]', ('huge', '1/zero'))):
                value = 'popcount(' + operand(('<<', '>>'), counts, complements, source) + ')'
                for wrapper in wrappers:
                    self.check_exact(base + wrapper.format('result=' + value + ';'), (('result', None),), (7,))
                self.check_exact(base + f'bool a=false && bool({value}); bool b=true || bool({value});',
                                 (('a', None), ('b', None)), (0, 1))

    def test_reached_failures_precede_main_source_reads_and_destination_writes(self):
        cls = self.module.QASM2QCXConverter
        cases = (('last=8;', ('1/zero', 'negative'), 16, 'assertion failed'),
                 ('step=0;', ('1/zero', 'negative'), 16, 'assertion failed'),
                 ('first=3; last=0;', ('huge', 'negative'), 16, 'assertion failed'),
                 ('', ('huge', 'negative'), 2, 'assertion failed'),
                 ('', ('negative', '1/zero'), 16, 'assertion failed'),
                 ('', ('huge', '-1'), 16, 'assertion failed'),
                 ('', ('huge', 'minimum'), 16, 'assertion failed'),
                 ('', (str(10 ** 70), '1/zero'), 16, 'division by zero'),
                 ('', ('huge', '0', 'negative'), 16, 'assertion failed'),
                 ('', ('huge', '0', '1/zero'), 16, 'division by zero'))
        for constant in (False, True):
            for complements in ((0, 1, 0), (1, 1, 1), (2, 2, 2), (3, 3, 3)):
                for update, counts, capacity, error in cases:
                    directions = ('<<', '>>', '<<')[:len(counts)]
                    parities = complements if len(counts) == 2 else complements + (complements[-1],)
                    source = ('const ' if constant else '') + 'bit[8] flags="00000000"; '
                    source += ('int first=0; int step=1; int last=3; int zero=0; uint huge=uint(-1); '
                               f'int negative=-1; int minimum={cls.QCX_INT_MIN}; uint result=6; ' + update
                               + f'result=popcount({operand(directions, counts, parities)});')
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

    def test_measurement_feedback_and_normal_fused_gate_parameters(self):
        for _ in range(16):
            self.check_exact('include "stdgates.inc"; qubit[2] q; reset q; h q[0]; bit[2] measured=measure q; '
                'int first=0; int last=1; int a=1; uint b=1; '
                'uint count=popcount((~(measured[first:last] << a)) >> b); '
                'bool valid=count==uint(1)-uint(measured[0]); '
                'if (count==uint(0)) { x q[1]; } bit[2] observed=measure q; '
                'bool feedback=observed[1]==measured[0] && observed[0]==measured[0];',
                (('valid', None), ('feedback', None)), (1, 1))
        from qasm2qcx_u_power_numerical import BackendAuditTests
        for complements, count in (((0, 1, 0), 1), ((0, 1, 1), 3), ((1, 1, 0), 2), ((1, 1, 1), 2)):
            phase = cmath.exp(count * 1j * math.pi / 8)
            for fused in (False, True):
                value = 'popcount(' + operand(('<<', '>>'), ('a', 'b'), complements, 'flags[first:last]') + ')'
                for gate in (f'p(float({value}) * pi/8) q;', f'pow({value}) @ p(pi/8) q;'):
                    source = ('include "stdgates.inc"; qubit[2] q; bit[4] flags="1011"; '
                              'int first=0; int last=3; int a=1; uint b=1; h q; ' + gate)
                    BackendAuditTests.check(self, self.program(source, fused=fused),
                                            (0.5, phase/2, phase/2, phase*phase/2))


def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    args = parser.parse_args()
    BackendTests.bra = args.bra.resolve()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(BackendTests))
    print(f'interleaved-complement runtime range popcount programs checked: {BackendTests.checked}')
    raise SystemExit(not result.wasSuccessful())


if __name__ == '__main__':
    main()
