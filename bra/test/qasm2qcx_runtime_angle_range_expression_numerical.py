#!/usr/bin/env python3
"""Audit whole-register expressions containing runtime angle ranges (#535).

Run with --bra /path/to/bra, including a non-MPI UBSan build. The reference
operations use independent little-endian bit vectors, including wide results.
"""

import argparse
import pathlib
import random
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

    def test_patterns_widths_sources_and_composed_operations(self):
        rng = random.Random(535)
        native = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        for width in range(1, native + 1):
            mask = (1 << width) - 1
            for pattern in sorted({0, mask, rng.getrandbits(width)}):
                other = rng.getrandbits(width)
                bounds = (width - 1, -1, 0) if width % 2 else (0, 1, width - 1)
                left = ('phases[slot][first:step:last]', registers.selected(pattern, width, bounds))
                right = ('bits[first:step:last]', registers.selected(other, width, bounds))
                angle = ('C[first:step:last]', left[1])
                tree = registers.unary(registers.binary(registers.unary(left, '~'), right, '^'), 'rotl', 1)
                mix = registers.binary(angle, right, '&')
                shifts = registers.unary(angle, '>>', 1)
                values = (tree, mix, shifts)
                statements = [f'const angle[{width}] C=angle[{width}]("{pattern:0{width}b}");',
                              f'array[angle[{width}],2] phases={{C,C}}; int slot=1;',
                              f'bit[{width}] bits="{other:0{width}b}";',
                              f'int first={bounds[0]}; int step={bounds[1]}; int last={bounds[2]};']
                outputs, expected = [], []
                for i, (expression, bits) in enumerate(values):
                    statements.extend((f'bit[{width}] init{i}={expression};',
                                       f'bit[{width}] out{i}; out{i}={expression};'))
                    outputs.extend(((f'init{i}', width), (f'out{i}', width)))
                    value = sum(bit << j for j, bit in enumerate(bits))
                    expected.extend((value, value))
                statements.append('bool stable=phases[slot]==C;')
                outputs.append(('stable', None)); expected.append(1)
                self.check_exact(' '.join(statements), outputs, expected)

    def test_repeated_positions_wide_results_and_changing_iterators(self):
        native = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        for width in (1, 4, native):
            pattern = (1 << (width - 1)) | 1
            bits = registers.selected(pattern, width, (-width, 1, width-1))
            result = sum((1-bit) << i for i, bit in enumerate(bits))
            self.check_exact(f'angle[{width}] phase=angle[{width}]("{pattern:0{width}b}"); '
                f'int first=-{width}; int last={width-1}; bit[{2*width}] out=~phase[first:last];',
                (('out', 2*width),), (result,))
        self.check_exact('array[angle[4],2] a={angle[4]("1011"),angle[4]("1101")}; '
            'int first=0; int last=3; uint sum=0; bit[4] out; '
            'for angle[4] item in a { out=~item[first:last]; sum+=uint[4](out); }',
            (('sum', None), ('out', 4)), (6, 2))

    def test_reached_errors_preserve_order_and_prevent_writes(self):
        cases = (
            ('rotl(phase[first:first],1/zero)', ('CHILD_0_SNAPSHOT_DONE',), 'division by zero'),
            ('rotl(phase[first:first],1)', ('CHILD_0_SNAPSHOT_DONE', 'DONE'), 'assertion failed'),
            ('phase[first:first]^rotl(bits[first:last],1/zero)', ('CHILD_0_SNAPSHOT_DONE', 'CHILD_1_CHILD_0_SNAPSHOT_DONE'), 'division by zero'),
            ('phase[first:first]^bits[first:last]', ('CHILD_0_SNAPSHOT_DONE', 'CHILD_1_SNAPSHOT_DONE'), 'assertion failed'),
            ('~phase[first:zero:last]', (), 'assertion failed'),
            ('~phase[first:99]', (), 'assertion failed'),
        )
        labels = ('CHILD_0_SNAPSHOT_DONE', 'CHILD_1_CHILD_0_SNAPSHOT_DONE', 'CHILD_1_SNAPSHOT_DONE', 'DONE')
        for expression, expected, error in cases:
            with self.subTest(expression=expression):
                source = ('angle[4] phase=angle[4]("1011"); bit[4] bits="1101"; '
                          'int first=0; int last=3; int zero=0; bit[4] out="1111"; out='+expression+';')
                program = self.module.openqasm3.parser.parse('OPENQASM 3; '+source)
                converter = self.module.QASM2QCXConverter(program); converter.visit(program)
                target = converter._QASM2QCXConverter__capitalize_variable_name('out')
                lines = []
                for line in converter:
                    if line.startswith(f'LET {target}:QASM2QCX_'):
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

    def test_skipped_errors_and_source_order(self):
        self.check_exact('angle[4] phase=pi; int first=0; int last=3; int zero=0; bit[4] out="1010"; '
            'if(false) { out=rotl(phase[first:zero:last],1/zero); } '
            'while(false) { out=~phase[first/zero:last]; } '
            'for int i in [1:0] { out=~phase[first:zero:last]; }',
            (('out', 4), ('phase', None)), (10, 8))

    def test_destination_alias_and_source_address_failure(self):
        self.check_exact('angle[4] phase=angle[4]("1011"); bit[4] out="1101"; '
            'int first=0; int last=3; out=out[first:last]^phase[first:last];',
            (('out', 4), ('phase', None)), (6, 11))
        source = ('array[angle[4],1] phase={pi}; int zero=0; int first=0; int last=3; '
                  'bit[4] out="1111"; out=~phase[1/zero][first:zero:last];')
        result = self.execute(self.program(source))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('division by zero', result.stderr)
        self.assertNotIn('runtime error:', result.stderr)
        type(self).checked += 1


if __name__ == '__main__':
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    args = parser.parse_args()
    BackendTests.bra = args.bra.resolve()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(BackendTests))
    print(f'runtime angle-range expression programs checked: {BackendTests.checked}')
    raise SystemExit(not result.wasSuccessful())
