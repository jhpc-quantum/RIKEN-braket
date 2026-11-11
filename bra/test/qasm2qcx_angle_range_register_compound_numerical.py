#!/usr/bin/env python3
"""Audit whole-register compounds with runtime angle-bit ranges (#543).

Run with --bra /path/to/bra, including a non-MPI UBSan build. Independent
little-endian bit vectors model the original destination and RHS operations.
"""

import argparse
import operator
import pathlib
import random
import resource
import unittest

import qasm2qcx_runtime_range_equality_numerical as registers


OPERATIONS = (('&=', operator.and_), ('|=', operator.or_), ('^=', operator.xor))


class BackendTests(unittest.TestCase):
    bra = None
    checked = 0
    assignments = 0
    setUp = registers.BackendTests.setUp
    execute = registers.BackendTests.execute
    program = registers.BackendTests.program
    check_exact = registers.BackendTests.check_exact

    def check_nodes(self, source, nodes, before):
        statements, outputs, expected = [source], [], []
        width = len(before)
        for symbol, function in OPERATIONS:
            for expression, bits in nodes:
                self.assertEqual(len(bits), width)
                name = f'out{len(outputs)}'
                statements.extend((f'bit[{width}] {name}={registers.literal(before)};',
                                   f'{name}{symbol}{expression};'))
                outputs.append((name, width))
                expected.append(sum(function(a, b) << i for i, (a, b) in enumerate(zip(before, bits))))
        self.check_exact(' '.join(statements), outputs, expected)
        type(self).assignments += len(outputs)

    def test_all_angle_widths_and_mixed_expression_trees(self):
        rng = random.Random(543)
        native = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        for width in range(1, native+1):
            mask = (1 << width)-1
            for pattern in sorted({0, mask, rng.getrandbits(width)}):
                other = rng.getrandbits(width)
                bounds = (width-1, -1, 0) if width%2 else (0, 1, width-1)
                left = ('phases[slot][first:step:last]', registers.selected(pattern, width, bounds))
                right = ('bits[first:step:last]', registers.selected(other, width, bounds))
                constant = ('C[first:step:last]', left[1])
                nodes = [left, registers.unary(left, '~'), registers.unary(constant, '<<', 1),
                         registers.unary(left, 'rotr', -1)]
                nodes.extend(registers.binary(left, right, op) for op in ('&', '|', '^'))
                nodes.append(registers.unary(registers.binary(registers.unary(left, '~'), right, '^'), 'rotl', 1))
                source = (f'const angle[{width}] C=angle[{width}]("{pattern:0{width}b}"); '
                          f'array[angle[{width}],2] phases={{C,C}}; int slot=1; '
                          f'bit[{width}] bits="{other:0{width}b}"; '
                          f'int first={bounds[0]}; int step={bounds[1]}; int last={bounds[2]};')
                self.check_nodes(source, nodes, [int(i%3 == 0) for i in range(width)])

    def test_wide_repeated_source_positions_and_singleton(self):
        native = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        for width in (1, 4, native):
            pattern = (1 << (width-1)) | 1
            bits = registers.selected(pattern, width, (-width, 1, width-1))
            node = registers.unary(('phase[first:last]', bits), '~')
            source = (f'angle[{width}] phase=angle[{width}]("{pattern:0{width}b}"); '
                      f'int first=-{width}; int last={width-1};')
            self.check_nodes(source, (node,), [i%2 for i in range(2*width)])
        self.check_exact('angle[1] phase=pi; int first=0; bit[1] one="1"; '
                         'one^=phase[first:first];', (('one', 1),), (0,))

    def test_overlapping_rhs_reads_original_destination(self):
        for pattern in range(16):
            before = [(pattern >> i)&1 for i in range(4)]
            angle = registers.selected(0b1011, 4, (0, 1, 3))
            bits = [a ^ b for a, b in zip(before, angle)]
            for symbol, function in OPERATIONS:
                expected = sum(function(a, b) << i for i, (a, b) in enumerate(zip(before, bits)))
                self.check_exact(f'angle[4] phase=angle[4]("1011"); int first=0; int last=3; '
                                 f'bit[4] out="{pattern:04b}"; out{symbol}out[first:last]^phase[first:last];',
                                 (('out', 4),), (expected,))

    def test_loop_reentry_and_angle_iterators(self):
        pattern = 180
        for symbol, function in OPERATIONS:
            before = [int(i%2 == 0) for i in range(4)]
            values = []
            for distance, bounds in enumerate(((0,1,3),(1,2,7),(7,-2,1))):
                bits = registers.unary(registers.unary(('', registers.selected(pattern, 8, bounds)), 'rotl', distance), '~')[1]
                before = [function(a, b) for a, b in zip(before, bits)]
                values.append(sum(bit << i for i, bit in enumerate(before)))
            self.check_exact('angle[8] phase=angle[8]("10110100"); bit[4] out="0101"; '
                'int first=0; int step=1; int last=3; int visits=0; array[uint[4],3] values={0,0,0}; '
                'while(visits<3) { out'+symbol+'~rotl(phase[first:step:last],visits); '
                'values[visits]=uint[4](out); '
                'if(visits==0) { first=1; step=2; last=7; } '
                'else { first=7; step=-2; last=1; } visits+=1; } '
                'uint a=values[0]; uint b=values[1]; uint c=values[2];',
                (('a', None), ('b', None), ('c', None)), tuple(values))
            before = [0,0,0,0]
            for value in (11,13):
                before = [function(a, 1-((value >> i)&1)) for i, a in enumerate(before)]
            self.check_exact('array[angle[4],2] phases={angle[4]("1011"),angle[4]("1101")}; '
                'int first=0; int last=3; bit[4] out="0000"; '
                'for angle[4] item in phases { out'+symbol+'~item[first:last]; }',
                (('out', 4),), (sum(bit << i for i, bit in enumerate(before)),))

    def test_rhs_errors_precede_combining_and_destination_writes(self):
        cases = (
            ('rotl(phase[first:first],1/zero)', ('ORIGINAL_DONE', 'VALUE_CHILD_0_SNAPSHOT_DONE'), 'division by zero'),
            ('rotl(phase[first:first],1)', ('ORIGINAL_DONE', 'VALUE_CHILD_0_SNAPSHOT_DONE', 'VALUE_DONE'), 'assertion failed'),
            ('phase[first:first]^bits[first:last]',
             ('ORIGINAL_DONE', 'VALUE_CHILD_0_SNAPSHOT_DONE', 'VALUE_CHILD_1_SNAPSHOT_DONE'), 'assertion failed'),
            ('~phase[first:zero:last]', ('ORIGINAL_DONE',), 'assertion failed'),
            ('~phase[first:99]', ('ORIGINAL_DONE',), 'assertion failed'),
        )
        labels = ('ORIGINAL_DONE', 'VALUE_CHILD_0_SNAPSHOT_DONE', 'VALUE_CHILD_1_SNAPSHOT_DONE', 'VALUE_DONE')
        for symbol, _ in OPERATIONS:
            for rhs, expected, error in cases:
                with self.subTest(operator=symbol, rhs=rhs):
                    source = ('angle[4] phase=angle[4]("1011"); bit[4] bits="1101"; bit[4] out="1010"; '
                              'int first=0; int last=3; int zero=0; '+f'out{symbol}{rhs};')
                    program = self.module.openqasm3.parser.parse('OPENQASM 3; '+source)
                    converter = self.module.QASM2QCXConverter(program); converter.visit(program)
                    target = converter._QASM2QCXConverter__capitalize_variable_name('out')
                    lines, active = [], False
                    for line in converter:
                        active = active or 'QASM2QCX_REGISTER_ASSIGNMENT_0' in line
                        if active and (line.startswith(f'LET {target}:') or line == '@QASM2QCX_REGISTER_ASSIGNMENT_0_BITWISE'):
                            lines.append('PRINTLN 999')
                        lines.append(line)
                        for i, label in enumerate(labels):
                            if line == '@QASM2QCX_REGISTER_ASSIGNMENT_0_'+label:
                                lines.append(f'PRINTLN {i+1}')
                    result = self.execute(lines)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(error, result.stderr)
                    self.assertNotIn('runtime error:', result.stderr)
                    self.assertEqual(result.stdout.split(), [str(labels.index(label)+1) for label in expected])
                    type(self).checked += 1

    def test_skipped_domains_and_later_iteration_failure(self):
        self.check_exact('angle[4] phase=pi; int first=0; int last=3; int zero=0; bit[4] out="1010"; '
            'if(false) { out^=rotl(phase[first:zero:last],1/zero); } '
            'while(false) { out&=~phase[first/zero:last]; } '
            'for int i in [1:0] { out|=~phase[first:zero:last]; }',
            (('out', 4), ('phase', None)), (10, 8))
        lines = self.program('angle[4] phase=angle[4]("1011"); bit[4] out="1010"; '
            'int first=0; int last=3; int visits=0; '
            'while(visits<2) { out^=~phase[first:last]; last-=1; visits+=1; }')
        instrumented = []
        for line in lines:
            instrumented.append(line)
            if line == '@QASM2QCX_REGISTER_ASSIGNMENT_0_END':
                instrumented.append('PRINTLN 111')
        result = self.execute(instrumented)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('assertion failed', result.stderr)
        self.assertNotIn('runtime error:', result.stderr)
        self.assertEqual(result.stdout.split(), ['111'])
        type(self).checked += 1


if __name__ == '__main__':
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    args = parser.parse_args()
    BackendTests.bra = args.bra.resolve()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(BackendTests))
    print(f'angle-range register-compound programs checked: {BackendTests.checked}; '
          f'destination registers checked: {BackendTests.assignments}')
    raise SystemExit(not result.wasSuccessful())
