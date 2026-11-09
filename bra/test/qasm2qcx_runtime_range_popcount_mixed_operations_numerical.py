#!/usr/bin/env python3
"""Audit mixed shifts/rotations in runtime-range population counts (#497).

Run with --bra /path/to/bra; the reference materializes little-endian bit
vectors after every operation, independently of the converter's intervals.
"""

import argparse
import cmath
import itertools
import math
import pathlib
import random
import resource
import unittest
from unittest.mock import patch

import qasm2qcx_angle_bit_cast_numerical as casts


def operand(operations, distances, complements, source='flags[first:step:last]'):
    assert len(operations) == len(distances) == len(complements) - 1
    value = '~' * complements[0] + source
    for operation, distance, depth in zip(operations, distances, complements[1:]):
        if operation in ('<<', '>>'):
            value = f'({value}) {operation} {distance}'
        else:
            value = f'{operation}({value}, {distance})'
        value = '~' * depth + f'({value})'
    return value


def reference(bits, operations, distances, complements):
    """Apply every shift, rotation, and complement to a private bit vector."""
    assert bits and len(operations) == len(distances) == len(complements) - 1
    value = [bit ^ (complements[0] % 2) for bit in bits]
    width = len(value)
    for operation, distance, depth in zip(operations, distances, complements[1:]):
        if operation == '<<':
            assert distance >= 0
            count = min(distance, width)
            value = [0] * count + value[:width - count]
        elif operation == '>>':
            assert distance >= 0
            count = min(distance, width)
            value = value[count:] + [0] * count
        elif operation == 'rotl':
            count = distance % width
            value = value[-count:] + value[:-count] if count else value
        else:
            assert operation == 'rotr'
            count = distance % width
            value = value[count:] + value[:count]
        value = [bit ^ (depth % 2) for bit in value]
    return sum(value)


def mixed_pairs(shift, rotation):
    for direction, function in itertools.product(('<<', '>>'), ('rotl', 'rotr')):
        yield ((direction, shift), (function, rotation))
        yield ((function, rotation), (direction, shift))


class BackendTests(unittest.TestCase):
    bra = None
    checked = 0
    vectors = 0
    setUp = casts.BackendTests.setUp
    execute = casts.BackendTests.execute
    program = casts.BackendTests.program
    check_exact = casts.BackendTests.check_exact

    def check_case(self, size, pattern, constant, first, step, last, chains, *, parities=None, literal=False):
        positions = list(range(first, last + (1 if step > 0 else -1), step))
        assert positions and all(-size <= position < size for position in positions)
        bits = [(pattern >> (position % size)) & 1 for position in positions]
        source = [('const ' if constant else '') + f'bit[{size}] flags="{pattern:0{size}b}"; ',
                  f'int first={first}; int step={step}; int last={last}; ']
        outputs, expected = [], []
        for chain, records in enumerate(chains):
            operations, distances = zip(*records)
            counts = []
            for index, (operation, distance) in enumerate(records):
                if literal:
                    counts.append(str(distance))
                else:
                    name = f'distance{chain}_{index}'
                    if isinstance(distance, bool):
                        kind, value = 'bool', str(distance).lower()
                    else:
                        unsigned = distance > self.module.QASM2QCXConverter.QCX_INT_MAX or (
                            distance >= 0 and operation in ('rotl', 'rotr') and chain % 2 == 1)
                        kind = 'uint' if unsigned else 'int'
                        value = str(distance)
                    source.append(f'{kind} {name}={value}; ')
                    counts.append(name)
            masks = itertools.product((0, 1), repeat=len(records) + 1) if parities is None else parities
            for mask, complements in enumerate(masks):
                name = f'out{chain}_{mask}'
                source.append(f'uint {name}=popcount({operand(operations, counts, complements)}); ')
                outputs.append((name, None))
                expected.append(reference(bits, operations, distances, complements))
        source.append(f'bool same=flags=="{pattern:0{size}b}"; uint fresh=popcount(flags[first:step:last]);')
        self.check_exact(''.join(source), (*outputs, ('same', None), ('fresh', None)), (*expected, 1, sum(bits)))
        type(self).vectors += len(expected)

    def test_exhaustive_small_vectors_orders_directions_and_complement_placements(self):
        for size in (1, 2, 3):
            pairs = ((0, 0), (1, -1), (size, 1), (size + 1, -size - 1), (size - 1, size + 1))
            chains = tuple(chain for shift, rotation in (pairs if size < 3 else ((1, -1),))
                           for chain in mixed_pairs(shift, rotation))
            for pattern in range(1 << size):
                for constant in (False, True):
                    self.check_case(size, pattern, constant, 0, 1, size - 1, chains)

    def test_native_width_boundaries_and_wide_registers(self):
        native = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        parities = ((0, 0, 0), (0, 1, 0), (1, 0, 1), (1, 1, 1))
        for size in (*range(1, native + 2), 129):
            pattern = int(('10110010' * size)[:size], 2)
            self.check_case(size, pattern, bool(size % 2), 0, 1, size - 1,
                            tuple(mixed_pairs(1, -1)), parities=parities)
        for pattern in (0, (1 << 129) - 1):
            for constant in (False, True):
                self.check_case(129, pattern, constant, 0, 1, 128, tuple(mixed_pairs(128, -129)))

    def test_long_alternating_chains_and_int_endpoints_without_overflow(self):
        cls = self.module.QASM2QCXConverter
        rng = random.Random(497)
        for depth in (3, 4, 8, 16):
            # Small shifts keep many bits alive while successive wrap cuts
            # create several fragments; endpoint cases below also discard all.
            chains = [tuple((('<<', 1), ('rotl', 2))[i % 2] for i in range(depth)),
                      tuple((('rotr', -3), ('>>', 1))[i % 2] for i in range(depth)),
                      tuple((('<<', 1), ('rotr', 3), ('>>', 1), ('rotl', -2))[i % 4]
                            for i in range(depth))]
            for _ in range(4):
                chain = [('<<', rng.choice((0, 1, cls.QCX_UINT_MAX))), ('rotl', cls.QCX_INT_MIN)]
                for index in range(2, depth):
                    operation = rng.choice(('<<', '>>', 'rotl', 'rotr'))
                    values = ((0, 1, 15, 16, 17, cls.QCX_UINT_MAX) if operation in ('<<', '>>')
                              else (cls.QCX_INT_MIN, cls.QCX_INT_MAX, -17, -1, 0, 1, 17))
                    chain.append((operation, rng.choice(values)))
                rng.shuffle(chain)
                chains.append(tuple(chain))
            parities = tuple(tuple(2 + ((mask >> (i % 3)) & 1) for i in range(depth + 1)) for mask in range(8))
            for constant in (False, True):
                self.check_case(8, 0xB3, constant, -8, 1, 7, chains, parities=parities, literal=True)

    def test_reverse_strides_aliases_bool_promotions_and_rotations_after_bit_loss(self):
        cls = self.module.QASM2QCXConverter
        chains = (('<<', cls.QCX_UINT_MAX), ('rotl', cls.QCX_INT_MIN), ('>>', 0), ('rotr', True))
        chains = (chains, (('rotr', False), ('<<', 1), ('rotl', True), ('>>', 1)),
                  (('rotl', cls.QCX_INT_MAX), ('>>', 1), ('rotr', -1), ('<<', 1)))
        parities = tuple(itertools.product((0, 1), repeat=5))
        for size, first, step, last in ((8, -1, -2, -7), (8, -8, 1, 7),
                                      (129, 128, -3, 119), (8, 2, 4, 7)):
            for constant in (False, True):
                self.check_case(size, (1 << size) - 3, constant, first, step, last, chains, parities=parities)

    def test_nested_calls_casts_arrays_and_math_consumers(self):
        self.check_exact('bit[4] flags="1011"; int first=0; int last=3; int shift=1; uint rotation=2; '
            'uint result=popcount(rotl(flags[first:last], rotation) << shift); '
            'array[uint,4] values={0,0,0,0}; '
            'values[popcount(rotl(flags[first:last], rotation) << shift)]=result; '
            'uint stored=values[2]; uint[1] narrow=popcount(rotl(flags[first:last], rotation) << shift); '
            'bit[2] packed=bit[2](uint[2](popcount(rotl(flags[first:last], rotation) << shift))); '
            'uint nested=popcount(rotr(flags[first:last] << shift, popcount(rotl(flags[first:last], rotation) << shift))); '
            'uint bounded=popcount(rotl(flags[0:int(result)], rotation) >> shift); '
            'float root=sqrt(float(result)); bool valid=root>1.4 && root<1.5;',
            (('result', None), ('stored', None), ('narrow', None), ('packed', 2), ('nested', None),
             ('bounded', None), ('valid', None), ('flags', 4)), (2, 2, 0, 2, 2, 1, 1, 11))

    def test_fresh_values_control_transfers_and_captured_loop_bounds(self):
        bits = [(0xB6 >> p) & 1 for p in range(8)]
        operations, complements = ('<<', 'rotr', '>>'), (1, 1, 1, 1)
        expected = [reference(bits[first:first + 4], operations, (first % 2, first - 2, 1), complements)
                    for first in range(5)]
        for constant in (False, True):
            source = ('const ' if constant else '') + 'bit[8] flags="10110110"; '
            value = operand(operations, ('first%2', 'first-2', '1'), complements, 'flags[first:last]')
            self.check_exact(source + 'int first=0; int last=3; uint count=0; uint total=0; '
                f'while (first<5) {{ last=first+3; count=popcount({value}); first+=1; '
                'if (first==2) { continue; } total+=count; if (first==4) { break; } }',
                (('total', None), ('count', None), ('first', None)), (sum(expected[i] for i in (0, 2, 3)), expected[3], 4))
        self.check_exact('bit[4] flags="1111"; int first=0; int last=3; int visits=0; '
            'while (popcount(rotl(flags[first:last] << 0, visits))>uint(0)) { flags>>=1; visits+=1; } '
            'uint total=0; flags="1011"; '
            'for int i in [popcount(rotl(flags[first:first] << 0, 0)):popcount(rotl(flags[first:last] << 0, 0))] '
            '{ total+=uint(i); first=99; } first=0; '
            'for int i in {popcount(rotl(flags[first:first] << 0, 0)),popcount(rotl(flags[first:last] << 0, 0))} '
            '{ total+=uint(i); first=99; }', (('visits', None), ('total', None), ('first', None)), (4, 10, 99))

    def test_skipped_invalid_values_and_short_circuiting(self):
        wrappers = ('if (false) {{ {} }}', 'while (false) {{ {} }}', 'for int i in [1:0] {{ {} }}',
                    'for int i in [loop_first:loop_last] {{ {} }}',
                    'while (true) {{ break; {} }}', 'for int i in [0:1] {{ continue; {} }}')
        base = ('bit[4] flags="1011"; uint result=7; int first=0; int last=3; '
                'int zero=0; uint u=uint(-1); int loop_first=2; int loop_last=1; ')
        for value in ('popcount(rotl(flags[first:last] << u, u))',
                      'popcount(~(rotl(flags[first:last], 1/zero) << -1))',
                      'popcount(rotr(flags[first:0:last] << -1, u))'):
            for wrapper in wrappers:
                self.check_exact(base + wrapper.format('result=' + value + ';'), (('result', None),), (7,))
            self.check_exact(base + f'bool a=false && bool({value}); bool b=true || bool({value});',
                             (('a', None), ('b', None)), (0, 1))

    def test_reached_errors_precede_counting_and_destination_writes(self):
        cases = (('last=8;', 'rotl(flags[first:step:last] << (1/zero), u)', 16, 'assertion failed'),
                 ('step=0;', 'rotl(flags[first:step:last] << (1/zero), u)', 16, 'assertion failed'),
                 ('first=3; last=0;', 'rotl(flags[first:step:last] << u, 1/zero)', 16, 'assertion failed'),
                 ('', 'rotl(flags[first:step:last] << (1/zero), u)', 2, 'assertion failed'),
                 ('', 'rotl(flags[first:last] << -1, 1/zero)', 16, 'assertion failed'),
                 ('', 'rotl(flags[first:last] << u, 1/zero)', 16, 'division by zero'),
                 ('', 'rotr(flags[first:last] << u, u)', 16, None),
                 ('', 'rotl(flags[first:last], minimum) >> -1', 16, 'assertion failed'),
                 ('', 'rotl(flags[first:last], u) << (1/zero)', 16, None),
                 ('', 'rotl(rotr(flags[first:last] << u, minimum), 1/zero)', 16, 'division by zero'))
        cls = self.module.QASM2QCXConverter
        for constant in (False, True):
            for update, value, capacity, error in cases:
                source = ('const ' if constant else '') + 'bit[8] flags="00000000"; '
                source += ('int first=0; int step=1; int last=3; int zero=0; uint u=uint(-1); '
                           f'int minimum={cls.QCX_INT_MIN}; uint result=6; {update} result=popcount({value});')
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
        for preparation in ('', 'x q[0]; ', 'h q[0]; ', 'h q[0]; x q[0]; '):
            self.check_exact('include "stdgates.inc"; qubit[2] q; reset q; ' + preparation + 'bit[2] measured=measure q; '
                'int first=0; int last=1; uint count=popcount(~(rotl(measured[first:last], -1) >> 1)); '
                'bool valid=count==uint(2)-uint(measured[0]); '
                'if (count==uint(1)) { x q[1]; } bit[2] observed=measure q; '
                'bool feedback=observed[1]==measured[0];', (('valid', None), ('feedback', None)), (1, 1))
        from qasm2qcx_u_power_numerical import BackendAuditTests
        bits = [1, 1, 0, 1]
        for complements in itertools.product((0, 1), repeat=3):
            count = reference(bits, ('rotl', '<<'), (-1, 1), complements)
            phase = cmath.exp(count * 1j * math.pi / 8)
            value = 'popcount(' + operand(('rotl', '<<'), ('a', 'b'), complements, 'flags[first:last]') + ')'
            for fused in (False, True):
                for gate in (f'p(float({value}) * pi/8) q;', f'pow({value}) @ p(pi/8) q;'):
                    source = ('include "stdgates.inc"; qubit[2] q; bit[4] flags="1011"; '
                              'int first=0; int last=3; int a=-1; uint b=1; h q; ' + gate)
                    BackendAuditTests.check(self, self.program(source, fused=fused),
                                            (0.5, phase/2, phase/2, phase*phase/2))


def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    args = parser.parse_args()
    BackendTests.bra = args.bra.resolve()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(BackendTests))
    print(f'mixed-operation runtime range popcount programs checked: {BackendTests.checked}; '
          f'bit-vector results checked: {BackendTests.vectors}')
    raise SystemExit(not result.wasSuccessful())


if __name__ == '__main__':
    main()
