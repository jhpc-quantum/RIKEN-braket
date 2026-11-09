#!/usr/bin/env python3
"""Audit complemented runtime-range bitwise population counts (#503).

Run with --bra /path/to/bra, including a UBSan build. References explicitly
flip each bit vector for every complement before combining and counting.
"""

import argparse
import cmath
import itertools
import math
import operator
import pathlib
import resource
import unittest
from unittest.mock import patch

import qasm2qcx_angle_bit_cast_numerical as casts


def expression(left, right, operation, depths):
    a, b, outer = depths
    return '~' * outer + '(' + '~' * a + f'({left}) {operation} ' + '~' * b + f'({right}))'


def selected(size, pattern, bounds):
    first, step, last = bounds
    positions = list(range(first, last + (1 if step > 0 else -1), step))
    assert positions and all(-size <= index < size for index in positions)
    return [(pattern >> (index % size)) & 1 for index in positions]


def complemented(bits, depth):
    value = list(bits)
    for _ in range(depth):
        value = [1 - bit for bit in value]
    return value


def reference(left, right, operation, depths):
    assert len(left) == len(right)
    a, b, outer = depths
    combine = {'&': operator.and_, '|': operator.or_, '^': operator.xor}[operation]
    value = [combine(x, y) for x, y in zip(complemented(left, a), complemented(right, b))]
    return sum(complemented(value, outer))


class BackendTests(unittest.TestCase):
    bra = None
    checked = 0
    comparisons = 0
    setUp = casts.BackendTests.setUp
    execute = casts.BackendTests.execute
    program = casts.BackendTests.program
    check_exact = casts.BackendTests.check_exact

    def check_case(self, left, right, constants, depths=None):
        sa, pa, ba = left
        sb, pb, bb = right
        av, bv = selected(*left), selected(*right)
        assert len(av) == len(bv)
        width = len(bv)
        mask = sum(bit << index for index, bit in enumerate(bv))
        source = [('const ' if constants[0] else '') + f'bit[{sa}] a="{pa:0{sa}b}"; ',
                  ('const ' if constants[1] else '') + f'bit[{sb}] b="{pb:0{sb}b}"; ',
                  ('const ' if constants[1] else '') + f'bit[{width}] mask="{mask:0{width}b}"; ']
        for name, value in zip(('first', 'stride', 'last', 'start', 'step', 'stop'), (*ba, *bb)):
            source.append(f'int {name}={value}; ')
        depths = tuple(itertools.product((0, 1), repeat=3)) if depths is None else depths
        forms = (('a[first:stride:last]', 'b[start:step:stop]', av, bv),
                 ('a[first:stride:last]', 'mask', av, bv), ('mask', 'a[first:stride:last]', bv, av))
        outputs, expected = [], []
        for operation, groups, (lhs, rhs, lv, rv) in itertools.product(('&', '|', '^'), depths, forms):
            name = f'out{len(outputs)}'
            source.append(f'uint {name}=popcount({expression(lhs, rhs, operation, groups)}); ')
            outputs.append((name, None))
            expected.append(reference(lv, rv, operation, groups))
        source.append(f'bool same=a=="{pa:0{sa}b}" && b=="{pb:0{sb}b}" && mask=="{mask:0{width}b}"; '
                      'uint fresh=popcount(a[first:stride:last]);')
        self.check_exact(''.join(source), (*outputs, ('same', None), ('fresh', None)), (*expected, 1, sum(av)))
        type(self).comparisons += len(expected)

    def test_exhaustive_small_patterns_parities_operators_and_source_kinds(self):
        for size in (1, 2):
            for a, b, ca, cb in itertools.product(range(1 << size), range(1 << size), (False, True), (False, True)):
                self.check_case((size, a, (0, 1, size - 1)), (size, b, (0, 1, size - 1)), (ca, cb))

    def test_every_native_width_and_wide_sources(self):
        native = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        for size in (*range(1, native + 2), 65, 129):
            a = int(('10110010' * size)[:size], 2)
            b = int(('11001101' * size)[:size], 2)
            self.check_case((size, a, (0, 1, size - 1)), (size, b, (size - 1, -1, 0)),
                            (bool(size % 2), bool((size // 2) % 2)))
        for a, b, ca, cb in itertools.product((0, (1 << 129) - 1), (0,), (False, True), (False, True)):
            self.check_case((129, a, (0, 1, 128)), (129, b, (0, 1, 128)), (ca, cb))

    def test_aliases_reverse_strides_different_widths_and_singletons(self):
        for sa, ba, sb, bb in ((8, (-1, -2, -7), 129, (119, 3, 128)),
                               (4, (-4, 1, 3), 8, (7, -1, 0)),
                               (1, (-1, 1, 0), 129, (128, -128, 0)),
                               (129, (128, -3, 119), 8, (0, 2, 6)),
                               (129, (2, 4, 5), 1, (0, 1, 0))):
            for constants in itertools.product((False, True), repeat=2):
                self.check_case((sa, int(('11010011' * sa)[:sa], 2), ba),
                                (sb, int(('10110110' * sb)[:sb], 2), bb), constants)

    def test_long_even_and_odd_groups_at_every_position(self):
        depths = tuple(itertools.product((0, 1, 2, 3), repeat=3))
        for constants in ((False, True), (True, False)):
            self.check_case((4, 11, (-4, 1, 3)), (8, 0xD3, (7, -1, 0)), constants, depths)

    def test_fixed_computed_operands_keep_their_own_internal_complements(self):
        av = [1, 1, 0, 1]
        operands = (('rotl(~mask, -1)', [1, 0, 0, 0]), ('(~mask) << 1', [0, 0, 1, 0]),
                    ('mask ^ "1010"', [1, 1, 1, 0]), ('mask[{index,0,1,2}]', [1, 1, 0, 1]),
                    ('bit[4](uint[4](mask))', [1, 0, 1, 1]),
                    ('bit[4](angle[4](mask))', [1, 0, 1, 1]))
        source = ['bit[4] a="1011"; bit[4] mask="1101"; int first=0; int last=3; int index=3; ']
        outputs, expected = [], []
        for operation, groups, (fixed, fv), reverse in itertools.product(
                ('&', '|', '^'), tuple(itertools.product((0, 1), repeat=3)), operands, (False, True)):
            lhs, rhs, lv, rv = 'a[first:last]', fixed, av, fv
            if reverse:
                lhs, rhs, lv, rv = rhs, lhs, rv, lv
            name = f'out{len(outputs)}'
            source.append(f'uint {name}=popcount({expression(lhs, rhs, operation, groups)}); ')
            outputs.append((name, None))
            expected.append(reference(lv, rv, operation, groups))
        self.check_exact(''.join(source), outputs, expected)
        type(self).comparisons += len(expected)

    def test_nested_consumers_casts_arrays_and_self_referencing_destinations(self):
        self.check_exact('bit[4] a="1011"; const bit[4] B="1101"; bit[4] mask="1101"; int first=0; int last=3; '
            'uint result=popcount(~((~a[first:last]) & B[first:last])); '
            'array[uint,4] values={0,0,0,0}; values[result]=result; uint stored=values[3]; '
            'uint[1] narrow=popcount(~((~a[first:last]) & B[first:last])); '
            'bit[2] packed=bit[2](uint[2](popcount(~((~a[first:last]) & B[first:last])))); '
            'uint nested=popcount(uint[4](result)); '
            'uint bounded=popcount(~(a[result:last] ^ B[result:last])); '
            'uint legacy=popcount(a[first:last] & ~mask); '
            'uint even=popcount(~~((~~a[first:last]) & ~mask)); '
            'uint bound=3; bound=popcount(~((~a[first:int(bound)]) & (~B[first:int(bound)]))); '
            'bool same=a=="1011" && mask=="1101" && B=="1101";',
            (('result', None), ('stored', None), ('narrow', None), ('packed', 2), ('nested', None),
             ('bounded', None), ('legacy', None), ('even', None), ('bound', None), ('same', None)),
            (3, 3, 1, 3, 2, 1, 1, 1, 4, 1))

    def test_loop_reentry_shrinking_selections_and_transfers(self):
        counts = [reference(selected(8, 0xB6, (i, 1, i + 3)), selected(8, 0xD3, (7 - i, -1, 4 - i)),
                            '^', (1, 1, 1)) for i in range(4)]
        for ca, cb in itertools.product((False, True), repeat=2):
            source = ('const ' if ca else '') + 'bit[8] a="10110110"; '
            source += ('const ' if cb else '') + 'bit[8] b="11010011"; '
            self.check_exact(source + 'int first=0; int last=3; int start=7; int stop=4; uint count=0; uint total=0; '
                'while (first<5) { last=first+3; start=7-first; stop=4-first; '
                'count=popcount(~((~a[first:last]) ^ (~b[start:-1:stop]))); first+=1; '
                'if (first==2) { continue; } total+=count; if (first==4) { break; } }',
                (('total', None), ('count', None), ('first', None)), (sum(counts[i] for i in (0, 2, 3)), counts[3], 4))
        total = sum(reference(selected(4, 11, (0, 1, stop)), selected(4, 13, (0, 1, stop)), '|', (2, 3, 2))
                    for stop in (3, 2, 1, 0))
        self.check_exact('bit[4] a="1011"; bit[4] b="1101"; int first=0; int stop=3; uint total=0; '
            'while (stop>=0) { total+=popcount(~~((~~a[first:stop]) | (~~~b[first:stop]))); stop-=1; } '
            'a="1111"; b="1111"; int last=3; int visits=0; '
            'while (popcount(~~((~~a[first:last]) & (~~b[first:last])))>uint(0)) { a>>=1; visits+=1; }',
            (('total', None), ('visits', None), ('a', 4), ('b', 4)), (total, 4, 0, 15))

    def test_skipped_invalid_operands_and_short_circuiting(self):
        wrappers = ('if (false) {{ {} }}', 'while (false) {{ {} }}', 'for int i in [1:0] {{ {} }}',
                    'for int i in [loop_first:loop_last] {{ {} }}',
                    'while (true) {{ break; {} }}', 'for int i in [0:1] {{ continue; {} }}')
        base = ('bit[4] a="1011"; const bit[8] B="11010011"; bit[4] mask="1101"; uint result=7; '
                'int first=0; int last=3; int start=0; int stop=3; int zero=0; uint huge=uint(-1); '
                'int loop_first=2; int loop_last=1; ')
        for value in ('popcount(~~((~~a[first:0:999]) & (~~B[start/zero:stop])))',
                      'popcount(~~((~~a[first:last]) ^ ~~("11" << (1/zero))))',
                      'popcount(~(~~rotl(mask, huge) | (~a[first/zero:last])))'):
            for wrapper in wrappers:
                self.check_exact(base + wrapper.format('result=' + value + ';'), (('result', None),), (7,))
            self.check_exact(base + f'bool x=false && bool({value}); bool y=true || bool({value});',
                             (('x', None), ('y', None)), (0, 1))

    def test_even_and_odd_groups_preserve_reached_errors_and_write_suppression(self):
        # Each tuple is (left, right, error, left-pair-snapshot-completed, limit).
        cases = (('a[first:0:last]', 'B[start/zero:stop]', 'assertion failed', False, 1000000),
                 ('a[first/zero:last]', 'B[start:0:stop]', 'division by zero', False, 1000000),
                 ('a[first:last]', 'B[start:1]', 'assertion failed', True, 1000000),
                 ('a[first:last]', 'B[start/zero:stop]', 'division by zero', True, 1000000),
                 ('a[first:last]', '("11" << (1/zero))', 'assertion failed', False, 1000000),
                 ('("11" << (1/zero))', 'a[first:last]', 'division by zero', False, 1000000),
                 ('"1111"', 'a[first/zero:last]', 'division by zero', False, 1000000),
                 ('a[first:2]', 'B[start/zero:stop]', 'assertion failed', False, 2))
        cls = self.module.QASM2QCXConverter
        for depths, (left, right, error, left_done, capacity) in itertools.product(
                ((0, 0, 0), (1, 1, 1), (2, 2, 2), (3, 2, 3)), cases):
            source = ('bit[4] a="1011"; const bit[8] B="11010011"; int first=0; int last=3; '
                      'int start=0; int stop=3; int zero=0; uint result=6; '
                      f'result=popcount({expression(left, right, "&", depths)});')
            program = self.module.openqasm3.parser.parse('OPENQASM 3; ' + source)
            with patch.object(cls, 'MAX_RUNTIME_BIT_RANGE_POSITIONS', capacity):
                converter = cls(program); converter.visit(program)
            self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())
            destination = converter._QASM2QCXConverter__capitalize_variable_name('result')
            lines = []
            for line in converter:
                if line == '@QASM2QCX_POPCOUNT_0_RIGHT_SNAPSHOT':
                    lines.extend([line, 'PRINTLN 33333']); continue
                if line.startswith(f'LET {destination} := QASM2QCX_'):
                    lines.append('PRINTLN 11111')
                lines.append(line)
                if line == '@QASM2QCX_POPCOUNT_0_LEFT_SNAPSHOT_DONE':
                    lines.append('PRINTLN 22222')
            result = self.execute(lines)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(error, result.stderr)
            self.assertNotIn('runtime error:', result.stderr)
            self.assertEqual(result.stdout.split(), ['22222'] if left_done else [])
            type(self).checked += 1

    def test_measurement_feedback_and_normal_fused_gate_parameters(self):
        for preparation in ('', 'x q[0]; ', 'h q[0]; ', 'h q[0]; x q[0]; '):
            self.check_exact('include "stdgates.inc"; qubit[2] q; reset q; ' + preparation +
                'bit[2] measured=measure q; const bit[2] MASK="01"; int first=0; int last=1; '
                'uint count=popcount(~((~measured[first:last]) & MASK[first:last])); '
                'bool valid=count==uint(1)+uint(measured[0]); '
                'uint fixed=popcount(~(MASK ^ measured[first:last])); bool same=fixed==count; '
                'if (count==uint(2)) { x q[1]; } bit[2] observed=measure q; '
                'bool feedback=observed[1]==measured[0];',
                (('valid', None), ('same', None), ('feedback', None)), (1, 1, 1))
        from qasm2qcx_u_power_numerical import BackendAuditTests
        forms = (('a[first:last]', 'b[start:stop]'), ('a[first:last]', 'b'), ('b', 'a[first:last]'))
        for operation, depths, (left, right) in itertools.product(
                ('&', '|', '^'), tuple(itertools.product((0, 1), repeat=3)), forms):
            av, bv = [1, 1, 0, 1], [1, 0, 1, 1]
            if left == 'b':
                av, bv = bv, av
            count = reference(av, bv, operation, depths)
            phase = cmath.exp(count * 1j * math.pi / 8)
            value = 'popcount(' + expression(left, right, operation, depths) + ')'
            for fused in (False, True):
                for gate in (f'p(float({value}) * pi/8) q;', f'pow({value}) @ p(pi/8) q;'):
                    source = ('include "stdgates.inc"; qubit[2] q; bit[4] a="1011"; bit[4] b="1101"; '
                              'int first=0; int last=3; int start=0; int stop=3; h q; ' + gate)
                    BackendAuditTests.check(self, self.program(source, fused=fused),
                                            (0.5, phase/2, phase/2, phase*phase/2))


def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    args = parser.parse_args()
    BackendTests.bra = args.bra.resolve()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(BackendTests))
    print(f'complemented runtime-range bitwise popcount programs checked: {BackendTests.checked}; '
          f'bit-vector comparisons: {BackendTests.comparisons}')
    raise SystemExit(not result.wasSuccessful())


if __name__ == '__main__':
    main()
