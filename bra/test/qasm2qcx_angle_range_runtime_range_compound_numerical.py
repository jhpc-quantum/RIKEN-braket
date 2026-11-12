#!/usr/bin/env python3
"""Audit runtime bit-range compounds with runtime angle ranges (#549).

Run with --bra /path/to/bra, including a non-MPI UBSan build. Independent
bit vectors model expression values, original LHS bits, and selected writes.
"""

import argparse
import pathlib
import resource
import unittest

import qasm2qcx_angle_range_static_compound_numerical as static_compounds
import qasm2qcx_runtime_range_equality_numerical as registers
import qasm2qcx_static_selection_expression_assignment_numerical as selections


OPERATIONS = static_compounds.OPERATIONS


class BackendTests(unittest.TestCase):
    bra = None
    checked = 0
    assignments = 0
    setUp = registers.BackendTests.setUp
    execute = registers.BackendTests.execute
    program = registers.BackendTests.program
    check_exact = registers.BackendTests.check_exact

    # The reference RHS bit vectors are independent of destination handling.
    test_all_angle_widths_and_mixed_expression_trees = (
        static_compounds.BackendTests.test_all_angle_widths_and_mixed_expression_trees)
    test_wide_repeated_source_positions_and_singleton = (
        static_compounds.BackendTests.test_wide_repeated_source_positions_and_singleton)

    def check_nodes(self, source, nodes, before):
        width = len(nodes[0][1])
        size = len(before)
        self.assertEqual(size, 2*width+1)
        declarations = (f'int dstart=0; uint dstop={width-1}; uint high={size-1}; '
                        f'int stride=-2; uint low=1; array[int,3] bounds={{-{size-1},2,-2}};')
        targets = (('dstart:dstop', list(range(width))),
                   ('high:stride:low', list(range(size-1, 1, -2))),
                   ('bounds[0]:bounds[1]:bounds[2]', list(range(1, size-1, 2))))
        for symbol, function in OPERATIONS:
            statements, outputs, expected = [source, declarations], [], []
            for selector, positions in targets:
                for expression, bits in nodes:
                    name = f'out{len(outputs)}'
                    statements.extend((f'bit[{size}] {name}={registers.literal(before)};',
                                       f'{name}[{selector}]{symbol}{expression};'))
                    result = selections.write_selected(
                        before, positions, [function(before[position], bit)
                                            for position, bit in zip(positions, bits)])
                    outputs.append((name, size))
                    expected.append(sum(bit << i for i, bit in enumerate(result)))
            self.check_exact(' '.join(statements), outputs, expected)
            type(self).assignments += len(outputs)

    def test_overlap_and_destination_bounds_read_original_bits(self):
        for pattern in range(16):
            before = [(pattern >> i)&1 for i in range(4)]
            angle = registers.selected(0b1011, 4, (0,1,3))
            bits = [a ^ b for a, b in zip(before, angle)]
            positions = (3,2,1,0)
            for symbol, function in OPERATIONS:
                result = selections.write_selected(before, positions,
                    [function(before[position], bit) for position, bit in zip(positions, bits)])
                self.check_exact(f'angle[4] phase=angle[4]("1011"); int first=0; int last=3; '
                                 f'int high=3; int low=0; bit[4] out="{pattern:04b}"; '
                                 f'out[high:-1:low]{symbol}out[first:last]^phase[first:last];',
                                 (('out', 4), ('phase', None)),
                                 (sum(bit << i for i, bit in enumerate(result)), 11))
        self.check_exact('angle[2] phase=angle[2]("11"); int first=0; int last=1; '
                         'bit[4] out="0000"; out[int(out[0]):int(out[1])+1]|=phase[first:last];',
                         (('out', 4),), (3,))
        self.check_exact('angle[1] phase=pi; int first=0; int index=-1; bit[1] one="1"; '
                         'one[index:index]^=phase[first:first];', (('one', 1),), (0,))

    def test_loop_reentry_and_angle_iterators(self):
        for symbol, function in OPERATIONS:
            before = [(0b01010101 >> i)&1 for i in range(8)]
            values = []
            for distance, (bounds, positions) in enumerate(zip(((0,1,3),(1,2,7),(7,-2,1)),
                                                               ((7,5,3,1),(6,4,2,0),(0,2,4,6)))):
                bits = registers.unary(registers.unary(('', registers.selected(180, 8, bounds)), 'rotl', distance), '~')[1]
                before = selections.write_selected(before, positions,
                    [function(before[position], bit) for position, bit in zip(positions, bits)])
                values.append(sum(bit << i for i, bit in enumerate(before)))
            self.check_exact('angle[8] phase=angle[8]("10110100"); bit[8] out="01010101"; '
                'int first=0; int step=1; int last=3; int visits=0; int high=7; int stride=-2; int low=1; '
                'array[uint[8],3] values={0,0,0}; '
                'while(visits<3) { out[high:stride:low]'+symbol+'~rotl(phase[first:step:last],visits); '
                'values[visits]=uint[8](out); '
                'if(visits==0) { first=1; step=2; last=7; high=6; low=0; } '
                'else { first=7; step=-2; last=1; high=0; low=6; stride=2; } visits+=1; } '
                'uint a=values[0]; uint b=values[1]; uint c=values[2];',
                (('a', None), ('b', None), ('c', None)), tuple(values))
            before = [0]*8
            for start, value in ((0,11), (4,13)):
                positions = tuple(range(start, start+4))
                bits = [1-((value >> i)&1) for i in range(4)]
                before = selections.write_selected(before, positions,
                    [function(before[position], bit) for position, bit in zip(positions, bits)])
            self.check_exact('array[angle[4],2] phases={angle[4]("1011"),angle[4]("1101")}; '
                'int first=0; int last=3; int index=0; bit[8] out="00000000"; '
                'for angle[4] item in phases { out[index:index+3]'+symbol+'~item[first:last]; index+=4; }',
                (('out', 8),), (sum(bit << i for i, bit in enumerate(before)),))

    def test_destination_and_rhs_failures_precede_combining_and_writes(self):
        cases = (
            ('first:8', 'rotl(phase[first:last],1/zero)', (), 'assertion failed'),
            ('first:zero:last', 'rotl(phase[first:last],1/zero)', (), 'assertion failed'),
            ('first/zero:last', '~phase[first:last]', (), 'division by zero'),
            ('-1:8:last+4', 'rotl(phase[first:last],1/zero)', (), 'assertion failed'),
            ('first:1', 'rotl(phase[first:first],1/zero)', ('ORIGINAL_DONE', 'VALUE_CHILD_0_SNAPSHOT_DONE'), 'division by zero'),
            ('first:1', 'rotl(phase[first:first],1)', ('ORIGINAL_DONE', 'VALUE_CHILD_0_SNAPSHOT_DONE', 'VALUE_DONE'), 'assertion failed'),
            ('first:1', 'phase[first:first]^bits[first:last]',
             ('ORIGINAL_DONE', 'VALUE_CHILD_0_SNAPSHOT_DONE', 'VALUE_CHILD_1_SNAPSHOT_DONE'), 'assertion failed'),
            ('first:1', '~phase[first:zero:last]', ('ORIGINAL_DONE',), 'assertion failed'),
            ('first:1', '~phase[first:99]', ('ORIGINAL_DONE',), 'assertion failed'),
        )
        labels = ('ORIGINAL_DONE', 'VALUE_CHILD_0_SNAPSHOT_DONE', 'VALUE_CHILD_1_SNAPSHOT_DONE', 'VALUE_DONE')
        for symbol, _ in OPERATIONS:
            for bounds, rhs, expected, error in cases:
                with self.subTest(operator=symbol, bounds=bounds, rhs=rhs):
                    source = ('angle[4] phase=angle[4]("1011"); bit[4] bits="1101"; bit[8] out="10101010"; '
                              'int first=0; int last=3; int zero=0; '+f'out[{bounds}]{symbol}{rhs};')
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
        self.check_exact('angle[4] phase=pi; int first=0; int last=3; int zero=0; bit[8] out="10101010"; '
            'if(false) { out[first:zero:last]^=rotl(phase[first:zero:last],1/zero); } '
            'while(false) { out[first/zero:last]&=~phase[first/zero:last]; } '
            'for int i in [1:0] { out[i:0:last]|=~phase[first:zero:last]; }',
            (('out', 8), ('phase', None)), (170, 8))
        lines = self.program('angle[4] phase=angle[4]("1011"); bit[8] out="10101010"; '
            'int first=0; int last=3; int visits=0; '
            'while(visits<2) { out[first:3]^=~phase[first:last]; last-=1; visits+=1; }')
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
    print(f'angle-range runtime-range compound programs checked: {BackendTests.checked}; '
          f'destination registers checked: {BackendTests.assignments}')
    raise SystemExit(not result.wasSuccessful())
