#!/usr/bin/env python3
"""Check fixed-width bit-register ordering against unsigned integer values (#555)."""

import argparse
import pathlib
import resource
import unittest

import qasm2qcx_runtime_range_equality_numerical as registers


class BackendTests(unittest.TestCase):
    bra = None
    checked = 0
    setUp = registers.BackendTests.setUp
    execute = registers.BackendTests.execute
    program = registers.BackendTests.program
    check_exact = registers.BackendTests.check_exact

    def test_documented_example(self):
        document = (pathlib.Path(__file__).resolve().parents[1] / 'qcx/qasm2qcx.md').read_text()
        marker = '<!-- bit-register-ordering-example -->'
        self.assertEqual(document.count(marker), 1)
        source = document.split(marker, 1)[1].split('```qasm\n', 1)[1].split('```', 1)[0]
        self.check_exact(source.replace('OPENQASM 3.0;', '', 1),
                         tuple((name, None) for name in ('lower', 'reversed', 'equal')), (1, 1, 1))

    def test_small_values_all_operators_and_operand_kinds(self):
        for width in (1, 2, 3):
            for a in range(1 << width):
                for b in range(1 << width):
                    statements = [f'bit[{width}] a="{a:0{width}b}";',
                                  f'bit[{width}] b="{b:0{width}b}";',
                                  f'const bit[{width}] A="{a:0{width}b}";',
                                  f'const bit[{width}] B="{b:0{width}b}";']
                    outputs, expected = [], []
                    for i, (op, value) in enumerate((('<', a < b), ('<=', a <= b),
                                                     ('>', a > b), ('>=', a >= b))):
                        for j, (left, right) in enumerate((('a', 'b'), ('A', 'b'), ('a', 'B'), ('A', 'B'))):
                            name = f'result{i}{j}'
                            if j == 3:
                                statements.append(f'const bool constant{i}={left}{op}{right}; '
                                                  f'bool {name}=constant{i};')
                            else:
                                statements.append(f'bool {name}={left}{op}{right};')
                            outputs.append((name, None)); expected.append(int(value))
                    self.check_exact(' '.join(statements), outputs, expected)

    def test_wide_values_and_selection_significance(self):
        self.check_exact('''bit[65] a="10000000000000000000000000000000000000000000000000000000000000000";
            bit[65] b="01111111111111111111111111111111111111111111111111111111111111111";
            bool high=a>b; bool low=b<a; bool equal=a>=a;
            bit[4] x="1011"; bool reverse=x[3:-1:0]>x;
            bool chosen=x[{3,1,0}]>"010";
        ''', tuple((name, None) for name in ('high','low','equal','reverse','chosen')), (1,1,1,1,1))

    def test_runtime_ranges_against_independent_bit_vectors(self):
        for width in (1, 2, 4, 16, self.module.QASM2QCXConverter.QCX_UINT_WIDTH):
            mask = (1 << width) - 1
            for a, b in ((0, 0), (mask, 0), (0, mask), (mask, mask), (mask // 3, mask // 2)):
                for bounds in ((0, 1, width-1), (width-1, -1, 0), (-width, 1, width-1)):
                    bits_a = registers.selected(a, width, bounds)
                    bits_b = registers.selected(b, width, bounds)
                    value_a = sum(bit << i for i, bit in enumerate(bits_a))
                    value_b = sum(bit << i for i, bit in enumerate(bits_b))
                    source = [f'angle[{width}] angle_a=angle[{width}]("{a:0{width}b}");',
                              f'bit[{width}] bits_b="{b:0{width}b}";',
                              f'int first={bounds[0]}; int step={bounds[1]}; int last={bounds[2]};']
                    outputs, expected = [], []
                    for i, (op, value) in enumerate((('<', value_a < value_b), ('<=', value_a <= value_b),
                                                     ('>', value_a > value_b), ('>=', value_a >= value_b))):
                        source.append(f'bool result{i}=angle_a[first:step:last] {op} bits_b[first:step:last];')
                        outputs.append((f'result{i}', None)); expected.append(int(value))
                    self.check_exact(' '.join(source), outputs, expected)

    def test_runtime_range_checks_and_short_circuit(self):
        self.check_exact('''angle[4] a=angle[4]("1011"); bit[4] b="1101";
            int first=0; int last=3; int zero=0;
            bool nested=(~a[first:last] ^ b[first:last]) > "0000";
            bool skipped=false && (a[first:zero:last] < b);
            bool preserved=true || (a[first:last] >= b[first:first]);
        ''', (('nested', None), ('skipped', None), ('preserved', None)), (1,0,1))
        for expression in ('a[first:first] < b[first:last]', 'a[first:zero:last] >= b'):
            with self.subTest(expression=expression):
                lines = self.program('angle[4] a=angle[4]("1011"); bit[4] b="1101"; '
                                     'int first=0; int last=3; int zero=0; '
                                     'bool result='+expression+';')
                result = self.execute(lines)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('assertion failed', result.stderr)
                self.assertNotIn('runtime error:', result.stderr)

    def test_wide_runtime_bit_ranges_and_array_angle_source(self):
        wide = '1' + '0' * 128
        below = '0' + '1' * 128
        self.check_exact(f'''bit[129] high="{wide}"; bit[129] low="{below}";
            int first=0; int last=128; bool greater=high[first:last]>low[first:last];
            bool less=low[last:-1:first]>high[last:-1:first];
            array[angle[4],2] phases={{angle[4]("1011"),angle[4]("1101")}};
            int slot=1; last=3; bool array_result=phases[slot][first:last]>"1011";
        ''', (('greater', None), ('less', None), ('array_result', None)), (1,1,1))

    def test_both_operands_complete_before_comparison_or_result_write(self):
        cases = (('a[first:first] < b[first:last]', ('LEFT_SNAPSHOT_DONE', 'RIGHT_SNAPSHOT_DONE'),
                  'assertion failed'),
                 ('a[first:last] > (b[first:last] << (1/zero))',
                  ('LEFT_SNAPSHOT_DONE', 'RIGHT_CHILD_0_SNAPSHOT_DONE'), 'division by zero'))
        labels = ('LEFT_SNAPSHOT_DONE', 'RIGHT_SNAPSHOT_DONE', 'RIGHT_CHILD_0_SNAPSHOT_DONE')
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
                    for i, label in enumerate(labels):
                        if line == '@QASM2QCX_REGISTER_EQUALITY_0_'+label:
                            lines.append(f'PRINTLN {i+1}')
                result = self.execute(lines)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(error, result.stderr)
                self.assertNotIn('runtime error:', result.stderr)
                self.assertEqual(result.stdout.split(), [str(labels.index(label)+1) for label in expected])


if __name__ == '__main__':
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    args = parser.parse_args()
    BackendTests.bra = args.bra.resolve()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(BackendTests))
    print(f'fixed-width ordering programs checked: {BackendTests.checked}')
    raise SystemExit(not result.wasSuccessful())
