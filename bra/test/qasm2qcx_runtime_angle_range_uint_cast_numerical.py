#!/usr/bin/env python3
"""Audit UINT casts from runtime angle-bit ranges (#529).

Run with --bra /path/to/bra, including a non-MPI UBSan build. Expected values
use independently selected little-endian bit vectors.
"""

import argparse
import contextlib
import pathlib
import resource
import unittest
from unittest.mock import patch

import qasm2qcx_angle_bit_cast_numerical as backend
from qasm2qcx_runtime_range_equality_numerical import binary, selected, unary
from qasm2qcx_runtime_range_angle_cast_numerical import packed


class BackendTests(unittest.TestCase):
    bra = None
    checked = 0
    setUp = backend.BackendTests.setUp
    execute = backend.BackendTests.execute
    program = backend.BackendTests.program
    check_exact = backend.BackendTests.check_exact

    def test_documented_runtime_angle_range_uint_cast_example(self):
        document = (pathlib.Path(__file__).resolve().parents[1] / 'qcx/qasm2qcx.md').read_text()
        marker = '<!-- runtime-angle-range-uint-cast-example -->'
        self.assertEqual(document.count(marker), 1)
        source = document.split(marker, 1)[1].split('```qasm\n', 1)[1].split('```', 1)[0]
        source = source.replace('OPENQASM 3.0;', '', 1)
        self.check_exact(source, (('saved', None), ('combined', None), ('copied', 4),
                                  ('element', None), ('fresh', None)), (10, 15, 10, 10, 0))

    def test_every_width_exact_patterns_sources_and_roundtrips(self):
        native = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        mask = (1 << native)-1
        for width in range(1, native+1):
            narrow = (1 << width)-1
            patterns = range(1 << width) if width <= 5 else sorted({0, narrow, 1 << (width-1),
                                                                    int(('10100110'*width)[:width], 2)})
            for pattern in patterns:
                source = pattern | (mask ^ narrow)
                bits = selected(source, native, (0, 1, width-1))
                self.check_exact(f'const angle[{native}] C=angle[{native}]("{source:0{native}b}"); '
                    f'const angle[{native}] D=angle[{native}]("{source^mask:0{native}b}"); '
                    f'angle[{native}] phase=C; array[angle[{native}],2] phases={{C,D}}; '
                    f'int first=0; int last={width-1}; uint slot=1; '
                    f'uint[{width}] value=uint[{width}](phase[first:last]); '
                    f'uint[{width}] constant=uint[{width}](C[first:last]); '
                    f'uint[{width}] element=uint[{width}](phases[slot][first:last]); '
                    f'uint[{width}] reverse=uint[{width}](phase[last:-1:first]); '
                    f'bit[{width}] back=bit[{width}](uint[{width}](phase[first:last])); '
                    'bool stable=phase==C && phases[slot]==D;',
                    (('value', None), ('constant', None), ('element', None), ('reverse', None), ('back', width), ('stable', None)),
                    (packed(bits), packed(bits), packed([1-bit for bit in bits]), packed(bits[::-1]), packed(bits), 1))

    def test_unsized_targets_negative_aliases_and_strides(self):
        width = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        for pattern in (0, 1, 1 << (width-1), (1 << width)-1):
            self.check_exact(f'angle[{width}] phase=angle[{width}]("{pattern:0{width}b}"); '
                f'int first=0; int last={width-1}; uint value=uint(phase[first:last]); '
                f'bit[{width}] back=bit[{width}](value);', (('value', None), ('back', width)), (pattern, pattern))
        for size, bounds in ((1, (-1, -1, -1)), (4, (-1, 1, 3)), (8, (1, 2, 6)), (8, (-1, -2, -7))):
            pattern = int(('10110100'*size)[:size], 2)
            bits = selected(pattern, size, bounds)
            first, step, last = bounds
            self.check_exact(f'angle[{size}] phase=angle[{size}]("{pattern:0{size}b}"); '
                f'int first={first}; int step={step}; int last={last}; '
                f'uint[{len(bits)}] value=uint[{len(bits)}](phase[first:step:last]);',
                (('value', None),), (packed(bits),))

    def test_mixed_expression_trees_and_integer_consumers(self):
        native = self.module.QASM2QCXConverter
        for pattern in range(16):
            lhs = ('phase[first:last]', selected(pattern, 4, (0, 1, 3)))
            rhs = ('wide[last:-1:first]', selected(pattern^15, 4, (3, -1, 0)))
            for count in (0, 1, 4, 5, native.QCX_UINT_MAX):
                turn = min(count, native.QCX_INT_MAX)
                nodes = [binary(lhs, rhs, '&'), binary(rhs, lhs, '|'), binary(lhs, rhs, '^'),
                         unary(binary(unary(lhs, '~'), rhs, '^'), '<<', count, 'distance'),
                         unary(binary(lhs, rhs, '^'), 'rotl', turn, 'turn')]
                source = (f'angle[4] phase=angle[4]("{pattern:04b}"); bit[129] wide="{pattern^15:0129b}"; '
                          f'int first=0; int last=3; uint distance={count}; uint turn={turn}; ')
                for index, (expression, _) in enumerate(nodes):
                    source += f'uint[4] a{index}=uint[4]({expression}); '
                source += 'bool truth=bool(uint[4](phase[first:last])); uint total=popcount(bit[4](uint[4](phase[first:last])));'
                self.check_exact(source, tuple((f'a{i}', None) for i in range(len(nodes)))+(('truth', None), ('total', None)),
                                 tuple(packed(node[1]) for node in nodes)+(int(bool(pattern)), pattern.bit_count()))

    def test_arrays_loops_and_overlapping_roundtrips(self):
        for pattern in range(16):
            reverse = packed(selected(pattern, 4, (3, -1, 0)))
            self.check_exact(f'angle[4] phase=angle[4]("{pattern:04b}"); '
                'array[angle[4],2] phases={0.0,phase}; int slot=1; int first=0; int last=3; '
                'uint[4] saved=uint[4](phase[first:last]); '
                'phase=angle[4](bit[4](uint[4](phase[last:-1:first]))); '
                'phases[slot]=angle[4](bit[4](uint[4](phases[slot][last:-1:first]))); '
                'uint[4] element=uint[4](phases[slot][first:last]);',
                (('saved', None), ('phase', None), ('element', None)), (pattern, reverse, reverse))
        self.check_exact('array[angle[4],2] phases={angle[4]("1011"),angle[4]("1101")}; '
            'int first=0; int last=3; uint total=0; '
            'for angle[4] phase in phases { total+=uint[4](phase[first:last]); } '
            'int slot=0; uint[4] saved; while(slot<2) { saved=uint[4](phases[slot][first:last]); slot+=1; }',
            (('total', None), ('saved', None), ('slot', None)), (24, 13, 2))

    def test_skipped_domains_and_short_circuit(self):
        calls = ('value=uint[4](phase[first:last]);', 'value=uint[4](phase[first:0:last]);',
                 'value=uint[4](phases[1/zero][first:last]);',
                 'value=uint[4](rotl(phase[first:last],1/zero));',
                 'value=uint[4]((phase[first:last]<<-1)^bits[first:last]);')
        wrappers = ('if(false) {{ {} }}', 'while(false) {{ {} }}', 'for int i in [1:0] {{ {} }}',
                    'for int i in [low:high] {{ {} }}', 'while(true) {{ break; {} }}')
        for call in calls:
            for wrapper in wrappers:
                self.check_exact('angle[4] phase=pi; array[angle[4],2] phases={pi,0.0}; bit[4] bits="1011"; '
                    'uint[4] value=6; int first=-5; int last=999; int zero=0; int low=2; int high=1; '
                    +wrapper.format(call), (('value', None), ('phase', None)), (6, 8))
        self.check_exact('angle[4] phase=pi; int first=-5; int last=999; int zero=0; '
            'bool no=false&&bool(uint[4](phase[first:zero:last])); '
            'bool yes=true||bool(uint[4](rotl(phase[first:last],1/zero)));',
            (('no', None), ('yes', None)), (0, 1))

    def test_reached_failures_precede_destination_writes(self):
        cases = (('last=2;', 'phase[first:last]', 'assertion failed'),
                 ('last=4;', 'phase[first:last]', 'assertion failed'),
                 ('first=-5;', 'phase[first:last]', 'assertion failed'),
                 ('', 'phase[first:zero:last]', 'assertion failed'),
                 ('distance=-1;', 'phase[first:last]<<distance', 'assertion failed'),
                 ('', 'rotl(phase[first:last],1/zero)', 'division by zero'),
                 ('', 'phases[1/zero][first:zero:last]', 'division by zero'),
                 ('', 'rotl(phase[first:last],huge)', 'out of range'))
        for constant in (False, True):
            for update, expression, error in cases:
                source = (('const ' if constant else '')+'angle[4] phase=angle[4]("1011"); '
                    'array[angle[4],2] phases={phase,phase}; int first=0; int last=3; int zero=0; int distance=1; '
                    f'uint huge={self.module.QASM2QCXConverter.QCX_UINT_MAX}; uint[4] value; '
                    +update+f'value=uint[4]({expression});')
                program = self.module.openqasm3.parser.parse('OPENQASM 3; '+source)
                c = self.module.QASM2QCXConverter(program)
                c.visit(program)
                self.assertEqual(c._QASM2QCXConverter__used_temporary_variables, set())
                self.assertEqual(c._QASM2QCXConverter__captured_bit_register_values, {})
                name = c._QASM2QCXConverter__capitalize_variable_name('value')
                lines = []
                for line in c:
                    if line.startswith(f'LET {name} := QASM2QCX_'):
                        lines.append('PRINTLN 99999')
                    lines.append(line)
                self.assertIn('PRINTLN 99999', lines)
                result = self.execute(lines)
                self.assertNotEqual(result.returncode, 0, source)
                self.assertIn(error, result.stderr, source)
                self.assertNotIn('runtime error:', result.stderr)
                self.assertEqual(result.stdout, '')
                type(self).checked += 1

    def test_failure_order_and_range_checks_precede_extraction(self):
        left = ['20001']+['20002']*4
        right = ['30001']+['30002']*4
        cases = (('last=2;', 'left[first:last]^right[1/zero:last]', ['20001'], 'assertion failed'),
                 ('last=2;', 'left[first:high]^right[first:last]', left+['30001'], 'assertion failed'),
                 ('distance=-1;', '(left[first:last]<<distance)^right[first:last]', left, 'assertion failed'),
                 ('', 'left[first:last]^(right[first:last]<<(1/zero))', left+right, 'division by zero'))
        for update, expression, expected, error in cases:
            source = ('angle[4] left=angle[4]("1101"); angle[4] right=angle[4]("1010"); '
                'int first=0; int last=3; int high=3; int zero=0; int distance=1; uint[4] value; '
                +update+f'value=uint[4]({expression});')
            program = self.module.openqasm3.parser.parse('OPENQASM 3; '+source)
            c = self.module.QASM2QCXConverter(program)
            original = c._QASM2QCXConverter__runtime_angle_bit_range_value
            @contextlib.contextmanager
            def marked(selection, width):
                marker = {'left': 20000, 'right': 30000}[selection.source.name]
                lines = c._QASM2QCXConverter__qcx_lines
                lines.append(f'PRINTLN {marker+1}')
                start = len(lines)
                with original(selection, width) as bits:
                    region = lines[start:]
                    instrumented = []
                    for index, line in enumerate(region):
                        if index and region[index-1].endswith(' &= 1') and ' := :INT:' in line:
                            instrumented.append(f'PRINTLN {marker+2}')
                        instrumented.append(line)
                    lines[start:] = instrumented
                    yield bits
            with patch.object(c, '_QASM2QCXConverter__runtime_angle_bit_range_value', side_effect=marked):
                c.visit(program)
            self.assertEqual(c._QASM2QCXConverter__used_temporary_variables, set())
            name = c._QASM2QCXConverter__capitalize_variable_name('value')
            lines = []
            for line in c:
                if line.startswith(f'LET {name} := QASM2QCX_'):
                    lines.append('PRINTLN 99999')
                lines.append(line)
            self.assertIn('PRINTLN 99999', lines)
            result = self.execute(lines)
            self.assertNotEqual(result.returncode, 0, source)
            self.assertIn(error, result.stderr, source)
            self.assertNotIn('runtime error:', result.stderr)
            self.assertEqual(result.stdout.split(), expected, source)
            type(self).checked += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra', type=pathlib.Path)
    arguments = parser.parse_args()
    BackendTests.bra = arguments.bra.resolve() if arguments.bra else None
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(BackendTests))
    print(f'runtime angle-bit range UINT cast programs checked: {BackendTests.checked}')
    raise SystemExit(not result.wasSuccessful())
