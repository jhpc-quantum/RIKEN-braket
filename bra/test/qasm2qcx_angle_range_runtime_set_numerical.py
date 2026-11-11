#!/usr/bin/env python3
"""Audit angle-range expressions assigned to runtime bit index sets (#539).

Run with --bra /path/to/bra, including a non-MPI UBSan build. Independent
little-endian bit vectors model expression values and selected writes.
"""

import argparse
import pathlib
import random
import resource
import unittest

import qasm2qcx_runtime_range_equality_numerical as registers
import qasm2qcx_runtime_set_expression_assignment_numerical as sets
from qasm2qcx_static_selection_expression_assignment_numerical import write_selected


class BackendTests(unittest.TestCase):
    bra = None
    checked = 0
    assignments = 0
    setUp = registers.BackendTests.setUp
    execute = registers.BackendTests.execute
    program = registers.BackendTests.program
    check_exact = registers.BackendTests.check_exact
    check_nodes = sets.BackendTests.check_nodes

    def test_documented_example(self):
        document = (pathlib.Path(__file__).resolve().parents[1] / 'qcx/qasm2qcx.md').read_text()
        marker = '<!-- angle-range-runtime-set-assignment-example -->'
        self.assertEqual(document.count(marker), 1)
        source = document.split(marker, 1)[1].split('```qasm\n', 1)[1].split('```', 1)[0]
        self.check_exact(source.replace('OPENQASM 3.0;', '', 1),
                         (('result', 8), ('copied', 8)), (160, 3))

    def test_all_angle_widths_and_mixed_expression_trees(self):
        rng = random.Random(539)
        native = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        for width in range(1, native+1):
            mask = (1 << width)-1
            for pattern in sorted({0, mask, rng.getrandbits(width)}):
                other = rng.getrandbits(width)
                bounds = (width-1, -1, 0) if width % 2 else (0, 1, width-1)
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
                self.check_nodes(source, nodes, [int(i%3 == 0) for i in range(2*width+1)],
                                 (('phases[slot]==C', 1),))

    def test_wide_repeated_source_positions_and_singletons(self):
        native = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        for width in (1, 4, native):
            pattern = (1 << (width-1)) | 1
            bits = registers.selected(pattern, width, (-width, 1, width-1))
            node = registers.unary(('phase[first:last]', bits), '~')
            source = (f'angle[{width}] phase=angle[{width}]("{pattern:0{width}b}"); '
                      f'int first=-{width}; int last={width-1};')
            self.check_nodes(source, (node,), [i%2 for i in range(4*width+1)])
        self.check_exact('angle[1] phase=pi; int first=0; int target=-1; bit[1] one="0"; '
                         'one[{target}]=~phase[first:first];', (('one', 1),), (0,))

    def test_overlap_and_addresses_read_before_destination_writes(self):
        for pattern in range(16):
            before = [(pattern >> i)&1 for i in range(4)]
            angle = registers.selected(0b1011, 4, (0, 1, 3))
            bits = [a ^ b for a, b in zip(before, angle)]
            expected = write_selected(before, (3, 0, 2, 1), bits)
            self.check_exact(f'angle[4] phase=angle[4]("1011"); bit[4] out="{pattern:04b}"; '
                             'int first=0; int last=3; int index=-1; '
                             'out[{index,0,2,1}]=out[first:last]^phase[first:last];',
                             (('out', 4), ('phase', None)),
                             (sum(bit << i for i, bit in enumerate(expected)), 11))
        self.check_exact('angle[2] phase=angle[2]("11"); int first=0; int last=1; '
                         'bit[4] out="0000"; out[{0,int(out[0])+1}]=phase[first:last];',
                         (('out', 4),), (3,))

    def test_loop_reentry_refreshes_sources_bounds_and_destinations(self):
        before = [(0b01010101 >> i)&1 for i in range(8)]
        values = []
        for iteration, bounds in enumerate(((0, 1, 3), (1, 2, 7), (7, -2, 1))):
            bits = registers.unary(registers.unary(('', registers.selected(180, 8, bounds)), 'rotl', iteration), '~')[1]
            before = write_selected(before, (7-iteration, iteration, 4, 3), bits)
            values.append(sum(bit << i for i, bit in enumerate(before)))
        self.check_exact('angle[8] phase=angle[8]("10110100"); bit[8] out="01010101"; '
            'int first=0; int step=1; int last=3; int visits=0; uint high=7; array[uint[8],3] values={0,0,0}; '
            'while(visits<3) { out[{high,visits,4,3}]=~rotl(phase[first:step:last],visits); '
            'values[visits]=uint[8](out); high-=1; '
            'if(visits==0) { first=1; step=2; last=7; } else { first=7; step=-2; last=1; } visits+=1; } '
            'uint a=values[0]; uint b=values[1]; uint c=values[2];',
            (('a', None), ('b', None), ('c', None)), tuple(values))
        self.check_exact('array[angle[4],2] phases={angle[4]("1011"),angle[4]("1101")}; '
            'int first=0; int last=3; int index=0; bit[8] out="00000000"; uint total=0; '
            'for angle[4] item in phases { out[{index,1,2,3}]=~item[first:last]; '
            'total+=uint[8](out); index+=4; }', (('total', None), ('out', 8)), (6, 2))

    def test_destination_and_rhs_failures_precede_writes(self):
        cases = (
            ('first,8', 'rotl(phase[first:last],1/zero)', (), 'assertion failed'),
            ('first,-8', 'rotl(phase[first:last],1/zero)', (), 'assertion failed'),
            ('first,1/zero', '~phase[first:last]', (), 'division by zero'),
            ('first,-8,1/zero', '~phase[first:last]', (), 'division by zero'),
            ('first,1', 'rotl(phase[first:first],1/zero)', ('CHILD_0_SNAPSHOT_DONE',), 'division by zero'),
            ('first,1', 'rotl(phase[first:first],1)', ('CHILD_0_SNAPSHOT_DONE', 'DONE'), 'assertion failed'),
            ('first,1', 'phase[first:first]^bits[first:last]',
             ('CHILD_0_SNAPSHOT_DONE', 'CHILD_1_SNAPSHOT_DONE'), 'assertion failed'),
            ('first,1', '~phase[first:zero:last]', (), 'assertion failed'),
        )
        labels = ('CHILD_0_SNAPSHOT_DONE', 'CHILD_1_SNAPSHOT_DONE', 'DONE')
        for entries, rhs, expected, error in cases:
            with self.subTest(entries=entries, rhs=rhs):
                source = ('angle[4] phase=angle[4]("1011"); bit[4] bits="1101"; bit[8] out="10101010"; '
                          'int first=0; int last=3; int zero=0; '+f'out[{{{entries}}}]={rhs};')
                program = self.module.openqasm3.parser.parse('OPENQASM 3; '+source)
                converter = self.module.QASM2QCXConverter(program); converter.visit(program)
                target = converter._QASM2QCXConverter__capitalize_variable_name('out')
                lines, active = [], False
                for line in converter:
                    active = active or 'QASM2QCX_REGISTER_ASSIGNMENT_0' in line
                    if active and line.startswith(f'LET {target}:'):
                        lines.append('PRINTLN 999')
                    lines.append(line)
                    for i, label in enumerate(labels):
                        if line == '@QASM2QCX_REGISTER_ASSIGNMENT_0_VALUE_'+label:
                            lines.append(f'PRINTLN {i+1}')
                result = self.execute(lines)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(error, result.stderr)
                self.assertNotIn('runtime error:', result.stderr)
                self.assertEqual(result.stdout.split(), [str(labels.index(label)+1) for label in expected])
                type(self).checked += 1

    def test_skipped_domains_and_later_iteration_failure(self):
        self.check_exact('angle[4] phase=pi; int first=0; int last=3; int zero=0; bit[8] out="10101010"; '
            'if(false) { out[{first,1/zero}]=rotl(phase[first:zero:last],1/zero); } '
            'while(false) { out[{first,-8}]=~phase[first/zero:last]; } '
            'for int i in [1:0] { out[{i,0}]=~phase[first:zero:last]; }',
            (('out', 8), ('phase', None)), (170, 8))
        lines = self.program('angle[4] phase=angle[4]("1011"); bit[8] out="10101010"; '
            'int first=0; int last=3; int visits=0; '
            'while(visits<2) { out[{visits,1,5,7}]=~phase[first:last]; last-=1; visits+=1; }')
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
    print(f'angle-range runtime-set programs checked: {BackendTests.checked}; '
          f'destination registers checked: {BackendTests.assignments}')
    raise SystemExit(not result.wasSuccessful())
