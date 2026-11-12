#!/usr/bin/env python3
"""Audit runtime angle-bit range equality using independent bit vectors (#553)."""

import argparse
import pathlib
import random
import resource
import unittest

import qasm2qcx_runtime_range_equality_numerical as registers


class BackendTests(unittest.TestCase):
    bra = None
    checked = 0
    comparisons = 0
    setUp = registers.BackendTests.setUp
    execute = registers.BackendTests.execute
    program = registers.BackendTests.program
    check_exact = registers.BackendTests.check_exact
    check_pairs = registers.BackendTests.check_pairs

    def test_documented_example(self):
        document = (pathlib.Path(__file__).resolve().parents[1] / 'qcx/qasm2qcx.md').read_text()
        marker = '<!-- runtime-angle-range-equality-example -->'
        self.assertEqual(document.count(marker), 1)
        source = document.split(marker, 1)[1].split('```qasm\n', 1)[1].split('```', 1)[0]
        self.check_exact(source.replace('OPENQASM 3.0;', '', 1),
                         tuple((name, None) for name in ('same', 'different', 'reversed', 'skipped')),
                         (1, 1, 1, 0))

    def test_widths_sources_and_nested_expressions(self):
        rng = random.Random(553)
        native = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        for width in range(1, native + 1):
            pattern, other = rng.getrandbits(width), rng.getrandbits(width)
            bounds = (0, 1, width-1) if width % 2 else (width-1, -1, 0)
            angle = ('phases[slot][first:step:last]', registers.selected(pattern, width, bounds))
            bit = ('bits[first:step:last]', registers.selected(other, width, bounds))
            constant = ('C[first:step:last]', angle[1])
            source = (f'const angle[{width}] C=angle[{width}]("{pattern:0{width}b}"); '
                      f'array[angle[{width}],2] phases={{C,C}}; int slot=1; '
                      f'bit[{width}] bits="{other:0{width}b}"; '
                      f'int first={bounds[0]}; int step={bounds[1]}; int last={bounds[2]}; ')
            self.preserved = (('phases[slot]', 'C'), ('bits', f'"{other:0{width}b}"'))
            left = registers.binary(registers.unary(angle, '~'), bit, '^')
            right = registers.unary(constant, 'rotl', 1)
            self.check_pairs(source, ((angle, bit), (constant, angle), (bit, angle),
                                      (left, right), (right, left),
                                      (left, (registers.literal(left[1]), left[1]))))

    def test_repeated_indices_and_wide_results(self):
        native = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        for width in (1, 4, native):
            pattern = (1 << width) - 1
            bounds = (-width, 1, width - 1)
            value = registers.selected(pattern, width, bounds)
            source = (f'angle[{width}] phase=angle[{width}]("{pattern:0{width}b}"); '
                      f'int first=-{width}; int last={width-1};')
            self.preserved = (('phase', f'angle[{width}]("{pattern:0{width}b}")'),)
            angle = ('phase[first:last]', value)
            self.check_pairs(source, ((angle, angle), (angle, (registers.literal(value), value)),
                                      (registers.unary(angle, '~'), angle)))

    def test_boolean_contexts_iterators_and_skipped_errors(self):
        self.check_exact('''angle[4] a=angle[4]("1011"); bit[4] b="1011";
            int first=0; int last=3; int zero=0; bool same=a[first:last]==b;
            bit flag=b!=a[first:last]; int visits=0;
            array[angle[4],2] phases={a,angle[4]("1101")};
            for angle[4] item in phases { if(item[first:last]==b) { visits+=1; } }
            bool skipped=false && (a[first:zero:last]==b);
            while(false && a[first/zero:last]==b) { visits+=100; }
            for int i in [1:0] { same=a[i:zero:999]==b; }
            bool nested=rotl(a[first:last],int(same))=="0111";
        ''', tuple((name, None) for name in ('same','flag','visits','skipped','nested')),
            (1,0,1,0,1))

    def test_reached_failures_complete_operands_before_writing(self):
        cases = (
            ('a[first:first] == b[first:last]', ('LEFT_SNAPSHOT_DONE','RIGHT_SNAPSHOT_DONE'), 'assertion failed'),
            ('a[first:last] == (b[first:last] << (1/zero))',
             ('LEFT_SNAPSHOT_DONE','RIGHT_CHILD_0_SNAPSHOT_DONE'), 'division by zero'),
            ('(a[first:first] & b[first:last]) == a[first:last]',
             ('LEFT_CHILD_0_SNAPSHOT_DONE','LEFT_CHILD_1_SNAPSHOT_DONE'), 'assertion failed'),
            ('a[first:zero:last] == b', (), 'assertion failed'),
        )
        labels = ('LEFT_SNAPSHOT_DONE','RIGHT_SNAPSHOT_DONE','RIGHT_CHILD_0_SNAPSHOT_DONE',
                  'LEFT_CHILD_0_SNAPSHOT_DONE','LEFT_CHILD_1_SNAPSHOT_DONE')
        for expression, expected, error in cases:
            with self.subTest(expression=expression):
                source = ('angle[4] a=angle[4]("1011"); bit[4] b="1101"; '
                          'int first=0; int last=3; int zero=0; bool result=true; result='+expression+';')
                program = self.module.openqasm3.parser.parse('OPENQASM 3; '+source)
                converter = self.module.QASM2QCXConverter(program); converter.visit(program)
                destination = converter._QASM2QCXConverter__capitalize_variable_name('result')
                lines = []
                for line in converter:
                    if line.startswith(f'LET {destination} := QASM2QCX_'):
                        lines.append('PRINTLN 999')
                    lines.append(line)
                    for index, label in enumerate(labels):
                        if line == '@QASM2QCX_REGISTER_EQUALITY_0_'+label:
                            lines.append(f'PRINTLN {index+1}')
                result = self.execute(lines)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(error, result.stderr)
                self.assertNotIn('runtime error:', result.stderr)
                self.assertEqual(result.stdout.split(), [str(labels.index(label)+1) for label in expected])
                type(self).checked += 1


if __name__ == '__main__':
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    args = parser.parse_args()
    BackendTests.bra = args.bra.resolve()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(BackendTests))
    print(f'angle-range equality programs checked: {BackendTests.checked}; '
          f'bit-vector comparisons: {BackendTests.comparisons}')
    raise SystemExit(not result.wasSuccessful())
