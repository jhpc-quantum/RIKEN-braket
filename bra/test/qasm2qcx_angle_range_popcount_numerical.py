#!/usr/bin/env python3
"""Audit popcount of runtime angle-bit range expression trees (#551).

Run with --bra /path/to/bra, including a non-MPI UBSan build. Independent
little-endian bit vectors model selections and register operations.
"""

import argparse
import pathlib
import random
import resource
import unittest

import qasm2qcx_runtime_popcount_tree_numerical as trees
import qasm2qcx_runtime_range_equality_numerical as registers


class BackendTests(unittest.TestCase):
    bra = None
    checked = 0
    comparisons = 0
    setUp = registers.BackendTests.setUp
    execute = registers.BackendTests.execute
    program = registers.BackendTests.program
    check_exact = registers.BackendTests.check_exact
    check_nodes = trees.BackendTests.check_nodes

    def test_all_angle_widths_sources_and_nested_trees(self):
        rng = random.Random(551)
        native = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        for width in range(1, native+1):
            mask = (1 << width)-1
            for pattern in sorted({0, mask, rng.getrandbits(width)}):
                other = rng.getrandbits(width)
                bounds = (width-1, -1, 0) if width%2 else (0, 1, width-1)
                left = ('phases[slot][first:step:last]', registers.selected(pattern, width, bounds))
                right = ('bits[first:step:last]', registers.selected(other, width, bounds))
                constant = ('C[first:step:last]', left[1])
                nodes = [left, constant, registers.unary(left, '~'), registers.unary(constant, '<<', 1),
                         registers.unary(left, 'rotr', -1)]
                nodes.extend(registers.binary(left, right, op) for op in ('&', '|', '^'))
                nodes.append(registers.unary(registers.binary(registers.unary(left, '~'), right, '^'), 'rotl', 1))
                source = (f'const angle[{width}] C=angle[{width}]("{pattern:0{width}b}"); '
                          f'array[angle[{width}],2] phases={{C,C}}; int slot=1; '
                          f'bit[{width}] bits="{other:0{width}b}"; '
                          f'int first={bounds[0]}; int step={bounds[1]}; int last={bounds[2]}; ')
                self.check_nodes(source, nodes, preservation=(('phases[slot]==C', 1),))

    def test_repeated_positions_and_counts_beyond_native_width(self):
        native = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        for width in (1, 4, native):
            pattern = (1 << width)-1
            bits = registers.selected(pattern, width, (-width, 1, width-1))
            source = (f'angle[{width}] phase=angle[{width}]("{pattern:0{width}b}"); '
                      f'int first=-{width}; int last={width-1}; ')
            leaf = ('phase[first:last]', bits)
            self.check_nodes(source, (leaf, registers.unary(leaf, '~'), registers.unary(leaf, '>>', 1)))

    def test_scalar_contexts_nested_counts_and_angle_iterators(self):
        self.check_exact('angle[4] phase=angle[4]("1011"); int first=0; int last=3; '
            'uint count=popcount(phase[first:last]); '
            'bool equal=popcount(~phase[first:last])==1; '
            'array[uint[4],2] out={0,0}; out[int(count)-3]=popcount(phase[first:last]); '
            'uint stored=out[0]; int signed_count=int(popcount(phase[first:last]));',
            (('count', None), ('equal', None), ('stored', None), ('signed_count', None)), (3, 1, 3, 3))
        self.check_exact('angle[4] phase=angle[4]("1011"); int first=0; int last=3; '
            'uint count=popcount(rotl(phase[first:last],popcount(phase[first:last])));',
            (('count', None),), (3,))
        self.check_exact('array[angle[4],2] phases={angle[4]("1011"),angle[4]("1101")}; '
            'int first=0; int last=3; uint total=0; '
            'for angle[4] item in phases { total+=popcount(~item[first:last]); }',
            (('total', None),), (2,))

    def test_changing_ranges_and_source_values_in_loops(self):
        source = ('angle[8] phase=angle[8]("10110100"); int first=0; int step=1; int last=3; '
                  'int visits=0; array[uint[8],3] values={0,0,0}; '
                  'while(visits<3) { values[visits]=popcount(~rotl(phase[first:step:last],visits)); '
                  'if(visits==0) { first=1; step=2; last=7; } '
                  'else { first=7; step=-2; last=1; } visits+=1; } '
                  'uint a=values[0]; uint b=values[1]; uint c=values[2];')
        expected = []
        for distance, bounds in enumerate(((0,1,3), (1,2,7), (7,-2,1))):
            bits = registers.selected(180, 8, bounds)
            value = registers.unary(registers.unary(('', bits), 'rotl', distance), '~')
            expected.append(sum(value[1]))
        self.check_exact(source, (('a', None), ('b', None), ('c', None)), tuple(expected))

    def test_failures_complete_children_before_count_or_assignment(self):
        cases = (
            ('rotl(phase[first:first],1/zero)', ('CHILD_0_SNAPSHOT_DONE',), 'division by zero'),
            ('phase[first:first]^bits[first:last]', ('CHILD_0_SNAPSHOT_DONE', 'CHILD_1_SNAPSHOT_DONE'), 'assertion failed'),
            ('~phase[first:zero:last]', (), 'assertion failed'),
            ('~phase[first:99]', (), 'assertion failed'),
        )
        labels = ('CHILD_0_SNAPSHOT_DONE', 'CHILD_1_SNAPSHOT_DONE')
        for expression, expected, error in cases:
            with self.subTest(expression=expression):
                source = ('angle[4] phase=angle[4]("1011"); bit[4] bits="1101"; '
                          'int first=0; int last=3; int zero=0; uint out=99; out=popcount('+expression+');')
                program = self.module.openqasm3.parser.parse('OPENQASM 3; '+source)
                converter = self.module.QASM2QCXConverter(program); converter.visit(program)
                target = converter._QASM2QCXConverter__capitalize_variable_name('out')
                lines, active = [], False
                for line in converter:
                    active = active or 'QASM2QCX_POPCOUNT_0_TREE' in line
                    if active and (line.startswith(f'LET {target} :=') or line == '@QASM2QCX_POPCOUNT_0_TREE_COUNT'):
                        lines.append('PRINTLN 999')
                    lines.append(line)
                    for i, label in enumerate(labels):
                        if line == '@QASM2QCX_POPCOUNT_0_TREE_'+label:
                            lines.append(f'PRINTLN {i+1}')
                result = self.execute(lines)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(error, result.stderr)
                self.assertNotIn('runtime error:', result.stderr)
                self.assertEqual(result.stdout.split(), [str(labels.index(label)+1) for label in expected])
                type(self).checked += 1

    def test_skipped_errors_and_later_iteration_failure(self):
        self.check_exact('angle[4] phase=pi; int first=0; int last=3; int zero=0; uint out=10; '
            'if(false) { out=popcount(rotl(phase[first:zero:last],1/zero)); } '
            'while(false) { out=popcount(~phase[first/zero:last]); } '
            'for int i in [1:0] { out=popcount(~phase[first:zero:last]); }',
            (('out', None), ('phase', None)), (10, 8))
        lines = self.program('angle[4] phase=angle[4]("1011"); int first=0; int last=3; int visits=0; uint out=0; '
            'while(visits<2) { out=popcount(phase[first:last]); last=99; visits+=1; }')
        instrumented = []
        for line in lines:
            instrumented.append(line)
            if line == '@QASM2QCX_POPCOUNT_0_TREE_COUNT_DONE':
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
    print(f'angle-range popcount programs checked: {BackendTests.checked}; '
          f'population counts checked: {BackendTests.comparisons}')
    raise SystemExit(not result.wasSuccessful())
