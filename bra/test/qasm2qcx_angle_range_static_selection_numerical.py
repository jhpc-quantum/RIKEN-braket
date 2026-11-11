#!/usr/bin/env python3
"""Audit runtime angle-range expressions assigned to static selections (#537).

Run with --bra /path/to/bra, including non-MPI UBSan builds. Independent
little-endian bit vectors model both the expression and the selected writes.
"""

import argparse
import pathlib
import random
import resource
import unittest

import qasm2qcx_runtime_range_equality_numerical as registers
import qasm2qcx_static_selection_expression_assignment_numerical as selections


class BackendTests(unittest.TestCase):
    bra = None
    checked = 0
    assignments = 0
    setUp = registers.BackendTests.setUp
    execute = registers.BackendTests.execute
    program = registers.BackendTests.program
    check_exact = registers.BackendTests.check_exact
    check_nodes = selections.BackendTests.check_nodes

    def test_documented_example(self):
        document = (pathlib.Path(__file__).resolve().parents[1] / 'qcx/qasm2qcx.md').read_text()
        marker = '<!-- angle-range-static-selection-assignment-example -->'
        self.assertEqual(document.count(marker), 1)
        source = document.split(marker, 1)[1].split('```qasm\n', 1)[1].split('```', 1)[0]
        self.check_exact(source.replace('OPENQASM 3.0;', '', 1),
                         (('result', 8), ('copied', 8)), (90, 10))

    def test_all_source_widths_and_destination_orders(self):
        rng = random.Random(537)
        native = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        for width in range(1, native + 1):
            mask = (1 << width) - 1
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
                before = [int(i % 3 == 0) for i in range(2*width+1)]
                source = (f'const angle[{width}] C=angle[{width}]("{pattern:0{width}b}"); '
                          f'array[angle[{width}],2] phases={{C,C}}; int slot=1; '
                          f'bit[{width}] bits="{other:0{width}b}"; '
                          f'int first={bounds[0]}; int step={bounds[1]}; int last={bounds[2]};')
                self.check_nodes(source, nodes, before, selections.targets(width), (('phases[slot]==C', 1),))

    def test_repeated_source_positions_wide_results_and_singletons(self):
        native = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        for width in (1, 4, native):
            pattern = (1 << (width-1)) | 1
            bits = registers.selected(pattern, width, (-width, 1, width-1))
            node = registers.unary(('phase[first:last]', bits), '~')
            before = [i % 2 for i in range(4*width+1)]
            source = (f'angle[{width}] phase=angle[{width}]("{pattern:0{width}b}"); '
                      f'int first=-{width}; int last={width-1};')
            self.check_nodes(source, (node,), before, selections.targets(2*width))
        self.check_exact('angle[1] phase=pi; int first=0; bit[1] one="0"; '
                         'one[{0}]=~phase[first:first];', (('one', 1),), (0,))

    def test_overlapping_bit_operand_and_changing_angle_iterators(self):
        before = [(0b10110101 >> i) & 1 for i in range(8)]
        left = registers.selected(0b10110101, 8, (0, 1, 3))
        right = registers.selected(0b1011, 4, (0, 1, 3))
        result = selections.write_selected(before, (3, 0, 2, 1), [a ^ b for a, b in zip(left, right)])
        self.check_exact('angle[4] phase=angle[4]("1011"); bit[8] out="10110101"; '
                         'int first=0; int last=3; out[{3,0,2,1}]=out[first:last]^phase[first:last];',
                         (('out', 8),), (sum(bit << i for i, bit in enumerate(result)),))
        self.check_exact('array[angle[4],2] phases={angle[4]("1011"),angle[4]("1101")}; '
                         'int first=0; int last=3; bit[8] out="00000000"; uint sum=0; '
                         'for angle[4] item in phases { out[0:3]=~item[first:last]; sum+=uint[8](out); }',
                         (('sum', None), ('out', 8)), (6, 2))

    def test_nonunit_steps_and_source_address_precedes_bounds(self):
        pattern = 0b10110100
        for bounds in ((1, 2, 7), (7, -2, 1), (-1, -2, -7)):
            bits = registers.selected(pattern, 8, bounds)
            before = [i % 2 for i in range(9)]
            source = (f'angle[8] phase=angle[8]("{pattern:08b}"); '
                      f'int first={bounds[0]}; int step={bounds[1]}; int last={bounds[2]};')
            self.check_nodes(source, (('phase[first:step:last]', bits),), before,
                             (('{7,1,5,0}', (7, 1, 5, 0)),))
        source = ('array[angle[4],1] phases={pi}; int zero=0; int first=0; int last=3; '
                  'bit[8] out="10101010"; out[0:3]=~phases[1/zero][first:zero:last];')
        result = self.execute(self.program(source))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('division by zero', result.stderr)
        self.assertNotIn('runtime error:', result.stderr)
        type(self).checked += 1

    def test_reached_failures_preserve_order_and_prevent_selected_writes(self):
        cases = (
            ('rotl(phase[first:first],1/zero)', ('CHILD_0_SNAPSHOT_DONE',), 'division by zero'),
            ('rotl(phase[first:first],1)', ('CHILD_0_SNAPSHOT_DONE', 'DONE'), 'assertion failed'),
            ('phase[first:first]^rotl(bits[first:last],1/zero)',
             ('CHILD_0_SNAPSHOT_DONE', 'CHILD_1_CHILD_0_SNAPSHOT_DONE'), 'division by zero'),
            ('phase[first:first]^bits[first:last]',
             ('CHILD_0_SNAPSHOT_DONE', 'CHILD_1_SNAPSHOT_DONE'), 'assertion failed'),
            ('~phase[first:zero:last]', (), 'assertion failed'),
            ('~phase[first:99]', (), 'assertion failed'),
        )
        labels = ('CHILD_0_SNAPSHOT_DONE', 'CHILD_1_CHILD_0_SNAPSHOT_DONE', 'CHILD_1_SNAPSHOT_DONE', 'DONE')
        for expression, expected, error in cases:
            with self.subTest(expression=expression):
                source = ('angle[4] phase=angle[4]("1011"); bit[4] bits="1101"; '
                          'int first=0; int last=3; int zero=0; bit[8] out="10101010"; '
                          'out[{7,1,5,0}]='+expression+';')
                program = self.module.openqasm3.parser.parse('OPENQASM 3; '+source)
                converter = self.module.QASM2QCXConverter(program); converter.visit(program)
                target = converter._QASM2QCXConverter__capitalize_variable_name('out')
                lines, active = [], False
                for line in converter:
                    active = active or 'QASM2QCX_REGISTER_ASSIGNMENT_0' in line
                    if active and line.startswith(f'LET {target}:'):
                        lines.append('PRINTLN 999')
                    lines.append(line)
                    for index, label in enumerate(labels):
                        if line == '@QASM2QCX_REGISTER_ASSIGNMENT_0_VALUE_'+label:
                            lines.append(f'PRINTLN {index+1}')
                result = self.execute(lines)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(error, result.stderr)
                self.assertNotIn('runtime error:', result.stderr)
                self.assertEqual(result.stdout.split(), [str(labels.index(label)+1) for label in expected])
                type(self).checked += 1

    def test_skipped_domains_and_prior_iteration_preservation(self):
        self.check_exact('angle[4] phase=pi; int first=0; int last=3; int zero=0; bit[8] out="10101010"; '
                         'if(false) { out[0:3]=rotl(phase[first:zero:last],1/zero); } '
                         'while(false) { out[{7,1,5,0}]=~phase[first/zero:last]; } '
                         'for int i in [1:0] { out[0:3]=~phase[first:zero:last]; }',
                         (('out', 8), ('phase', None)), (170, 8))
        lines = self.program('angle[4] phase=angle[4]("1011"); bit[8] out="10101010"; '
                             'int first=0; int last=3; int visits=0; '
                             'while(visits<2) { out[{7,1,5,0}]=~phase[first:last]; last-=1; visits+=1; }')
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
    print(f'angle-range static-selection programs checked: {BackendTests.checked}; '
          f'destination registers checked: {BackendTests.assignments}')
    raise SystemExit(not result.wasSuccessful())
