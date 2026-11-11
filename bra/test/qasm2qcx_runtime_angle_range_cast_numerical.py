#!/usr/bin/env python3
"""Audit exact ANGLE casts from runtime angle-bit ranges (#527).

Run with --bra /path/to/bra, including a non-MPI UBSan build. Expected
patterns come from independent little-endian bit vectors, without converter
folding or floating-point angle conversion. Execution markers check operand
order, extraction after range validation, and writes after complete results.
"""

import argparse
import cmath
import contextlib
import math
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

    def test_documented_runtime_angle_range_cast_example(self):
        document = (pathlib.Path(__file__).resolve().parents[1] / 'qcx/qasm2qcx.md').read_text()
        marker = '<!-- runtime-angle-range-cast-example -->'
        self.assertEqual(document.count(marker), 1)
        source = document.split(marker, 1)[1].split('```qasm\n', 1)[1].split('```', 1)[0]
        source = source.replace('OPENQASM 3.0;', '', 1)
        self.check_exact(source, (('selected', None), ('copied', 4), ('changed', None)), (10, 10, 80))

    def test_all_widths_constant_mutable_array_and_roundtrip_sources(self):
        native = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        native_mask = (1 << native) - 1
        for width in range(1, native + 1):
            mask = (1 << width) - 1
            patterns = (range(1 << width) if width <= 5 else
                        sorted({0, mask, 1 << (width-1), int(('10100110'*width)[:width], 2)}))
            for pattern in patterns:
                # Unselected high bits must not affect the narrower result.
                source_pattern = pattern | (native_mask ^ mask)
                bits = selected(source_pattern, native, (0, 1, width-1))
                self.check_exact(
                    f'const angle[{native}] C=angle[{native}]("{source_pattern:0{native}b}"); '
                    f'const angle[{native}] D=angle[{native}]("{source_pattern^native_mask:0{native}b}"); '
                    f'angle[{native}] phase=C; array[angle[{native}],2] phases={{C,D}}; '
                    f'int first=0; int last={width-1}; uint slot=1; '
                    f'angle[{width}] a=angle[{width}](phase[first:last]); '
                    f'angle[{width}] constant=angle[{width}](C[first:last]); '
                    f'angle[{width}] element=angle[{width}](phases[slot][first:last]); '
                    f'angle[{width}] reverse=angle[{width}](phase[last:-1:first]); '
                    f'bit[{width}] back=bit[{width}](angle[{width}](phase[first:last])); '
                    f'uint[{width}] bridge=uint[{width}](bit[{width}](angle[{width}](phase[first:last]))); '
                    'bool stable=phase==C && phases[slot]==D;',
                    (('a', None), ('constant', None), ('element', None), ('reverse', None),
                     ('back', width), ('bridge', None), ('stable', None)),
                    (packed(bits), packed(bits), packed([1-bit for bit in bits]), packed(bits[::-1]),
                     packed(bits), packed(bits), 1))

    def test_unsized_native_targets_and_angle_width_conversions(self):
        width = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        mask = (1 << width) - 1
        for pattern in (0, 1, 1 << (width-1), mask):
            self.check_exact(f'angle[{width}] phase=angle[{width}]("{pattern:0{width}b}"); '
                f'int first=0; int last={width-1}; '
                'angle a=angle(phase[first:last]); angle sum=a+angle(pi); '
                'angle inverse=~angle(phase[first:last]); '
                f'bit[{width}] back=bit[{width}](a);',
                (('a', None), ('sum', None), ('inverse', None), ('back', width)),
                (pattern, (pattern+(1 << (width-1))) & mask, pattern^mask, pattern))
        self.check_exact('angle[8] phase=angle[8]("00001011"); int first=0; int last=3; '
            'angle[8] widened=angle[4](phase[first:last]); '
            'angle[2] narrowed=angle[4](phase[first:last]); '
            'angle[4] copied=angle[4](phase[first:last]);',
            (('widened', None), ('narrowed', None), ('copied', None)), (176, 2, 11))

    def test_strides_repeated_negative_aliases_and_native_bound_types(self):
        native = self.module.QASM2QCXConverter
        for size, bounds in ((1, (-1, -1, -1)), (4, (-1, 1, 3)), (8, (1, 2, 6)),
                             (8, (-1, -2, -7)), (native.QCX_UINT_WIDTH, (3, -1, 0))):
            pattern = int(('10110100'*size)[:size], 2)
            bits = selected(pattern, size, bounds)
            width = len(bits)
            first, step, last = bounds
            for distance in (-9, -1, 0, 1, native.QCX_INT_MIN, native.QCX_INT_MAX):
                leaf = ('phase[first:step:last]', bits)
                nodes = [leaf, unary(leaf, 'rotl', distance, 'count'),
                         unary(leaf, 'rotr', distance, 'count'), unary(leaf, '~')]
                source = (f'const angle[{size}] C=angle[{size}]("{pattern:0{size}b}"); '
                          f'angle[{size}] phase=C; int first={first}; int step={step}; int last={last}; '
                          f'int count={distance}; ')
                for index, (expression, _) in enumerate(nodes):
                    source += f'angle[{width}] a{index}=angle[{width}]({expression}); '
                self.check_exact(source, tuple((f'a{i}', None) for i in range(len(nodes))),
                                 tuple(packed(node[1]) for node in nodes))
        maximum = native.QCX_UINT_MAX
        self.check_exact(f'angle[1] phase=pi; uint first=0; uint step={maximum}; uint last=0; '
            'angle[1] a=angle[1](phase[first:step:last]);', (('a', None),), (1,))

    def test_nested_register_operations_with_angle_and_bit_range_leaves(self):
        native = self.module.QASM2QCXConverter
        for pattern in range(16):
            angle_bits = selected(pattern, 4, (0, 1, 3))
            bit_bits = selected(pattern^15, 4, (3, -1, 0))
            for count in (0, 1, 4, 5, native.QCX_UINT_MAX):
                turn = min(count, native.QCX_INT_MAX)
                left = ('phase[first:last]', angle_bits)
                right = ('wide[last:-1:first]', bit_bits)
                nodes = [binary(left, right, '&'), binary(right, left, '|'), binary(left, right, '^'),
                         unary(binary(unary(left, '~'), right, '^'), '<<', count, 'distance'),
                         binary(unary(left, '>>', count, 'distance'), unary(right, '~'), '&'),
                         unary(binary(left, right, '^'), 'rotl', turn, 'turn'),
                         unary(binary(unary(left, 'rotr', turn, 'turn'), right, '|'), '~')]
                source = (f'angle[4] phase=angle[4]("{pattern:04b}"); bit[129] wide="{pattern^15:0129b}"; '
                          f'int first=0; int last=3; uint distance={count}; uint turn={turn}; ')
                for index, (expression, _) in enumerate(nodes):
                    source += f'angle[4] a{index}=angle[4]({expression}); '
                self.check_exact(source, tuple((f'a{i}', None) for i in range(len(nodes))),
                                 tuple(packed(node[1]) for node in nodes))

    def test_fixed_leaves_angle_sets_and_nested_scalar_casts(self):
        prefix = ('angle[8] phase=angle[8]("11010011"); angle[4] other=angle[4]("1010"); '
                  'bit[4] fixed="0101"; uint[4] number=3; array[angle[4],2] phases={pi,pi/2}; '
                  'int first=1; int step=2; int last=7; int low=0; int high=3; int slot=1;')
        for operand, value in (('"1100"', 12), ('fixed', 5), ('bit[4](number)', 3),
                                ('other[{low,1,2,high}]', 10), ('phases[slot][{low,1,2,high}]', 4),
                                ('bit[4](angle[4](other[low:high]))', 10)):
            for lhs, rhs in (('phase[first:step:last]', operand), (operand, 'phase[first:step:last]')):
                self.check_exact(prefix+f'angle[4] a=angle[4]({lhs}^{rhs});',
                                 (('a', None),), (9^value,))
        self.check_exact('angle[4] phase=angle[4]("1011"); int low=0; int high=1; '
            'angle[1] a=angle[1](phase[uint[2](bit[2](angle[2](phase[low:high]))):'
            'uint[2](bit[2](angle[2](phase[low:high])))]); '
            'bit[4] shifted="1111"; shifted>>=uint[4](bit[4](angle[4](phase[low:3])));',
            (('a', None), ('shifted', 4)), (1, 0))

    def test_overlapping_scalar_array_and_bit_selection_writes(self):
        for pattern in range(16):
            reverse = packed(selected(pattern, 4, (3, -1, 0)))
            self.check_exact(f'angle[4] phase=angle[4]("{pattern:04b}"); '
                f'array[angle[4],2] phases={{0.0,phase}}; int slot=1; int first=0; int last=3; '
                'angle[4] saved=angle[4](phase[first:last]); phase=angle[4](phase[last:-1:first]); '
                'phases[slot]=angle[4](phases[slot][last:-1:first]); angle[4] element=phases[slot]; '
                'phase[{last,2,1,first}]=bit[4](angle[4](phase[first:last])); '
                'phases[slot][{last,2,1,first}]=bit[4](angle[4](phases[slot][first:last])); '
                'angle[4] restored=phases[slot];',
                (('saved', None), ('phase', None), ('element', None), ('restored', None)),
                (pattern, pattern, reverse, pattern))

    def test_fresh_loop_operands_array_iterators_and_transfers(self):
        self.check_exact('const angle[8] C=angle[8]("10110101"); angle[8] phase=C; '
            'array[angle[4],2] phases; int first=0; int last=3; uint total=0; '
            'while(first<5) { last=first+3; phases[0]=angle[4](phase[first:last]); first+=1; '
            'if(first==2) { continue; } total+=uint[4](bit[4](phases[0])); if(first==4) { break; } } '
            'angle[4] out=phases[0];', (('total', None), ('out', None), ('first', None)),
            (sum(packed(selected(181, 8, (i, 1, i+3))) for i in (0, 2, 3)),
             packed(selected(181, 8, (3, 1, 6))), 4))
        self.check_exact('array[angle[4],2] phases={angle[4]("1011"),angle[4]("1101")}; '
            'int first=0; int last=3; uint total=0; '
            'for angle[4] phase in phases { total+=uint[4](bit[4](angle[4](phase[first:last]))); } '
            'angle[4] phase=angle[4]("1011"); int visits=0; '
            'while(angle[4](phase[first:last])!=angle[4]("1111")) { '
            'phase=angle[4](phase[first:last])+angle[4](pi/8); visits+=1; }',
            (('total', None), ('phase', None), ('visits', None)), (24, 15, 4))
        self.check_exact('array[angle[4],2] phases={angle[4]("1011"),angle[4]("1101")}; '
            'int slot=0; int first=0; int last=3; uint total=0; angle[4] saved; '
            'while(slot<2) { saved=angle[4](phases[slot][first:last]); slot+=1; '
            'total+=uint[4](bit[4](saved)); }',
            (('total', None), ('saved', None), ('slot', None)), (24, 13, 2))

    def test_scalar_angle_arithmetic_comparisons_and_functions(self):
        for pattern in range(16):
            self.check_exact(f'angle[4] phase=angle[4]("{pattern:04b}"); int first=0; int last=3; '
                'angle[4] sum=angle[4](phase[first:last])+angle[4](pi); '
                'angle[4] times=angle[4](phase[first:last])*uint[4](3); '
                'bool truth=bool(angle[4](phase[first:last])); '
                'bool same=angle[4](phase[first:last])==phase; '
                'uint ones=popcount(bit[4](angle[4](phase[first:last])));',
                (('sum', None), ('times', None), ('truth', None), ('same', None), ('ones', None)),
                ((pattern+8)%16, (pattern*3)%16, int(bool(pattern)), 1, pattern.bit_count()))
        self.check_exact('angle[4] phase=pi/2; int first=0; int last=3; '
            'bool sine=sin(angle[4](phase[first:last]))==1.0; '
            'bool cosine=cos(angle[4](phase[first:last]))==0.0;',
            (('sine', None), ('cosine', None)), (1, 1))

    def test_skipped_paths_defer_range_array_and_count_domains(self):
        calls = ('a=angle[4](phase[first:last]);', 'a=angle[4](phase[first:0:last]);',
                 'a=angle[4](phases[1/zero][first:last]);',
                 'a=angle[4](rotl(phase[first:last],1/zero));',
                 'a=angle[4]((phase[first:last]<<-1)^bits[first:last]);')
        wrappers = ('if(false) {{ {} }}', 'while(false) {{ {} }}', 'for int i in [1:0] {{ {} }}',
                    'for int i in [low:high] {{ {} }}', 'while(true) {{ break; {} }}',
                    'while(visits<2) {{ visits+=1; continue; {} }}')
        for call in calls:
            for wrapper in wrappers:
                self.check_exact('angle[4] phase=pi; array[angle[4],2] phases={pi,0.0}; bit[4] bits="1011"; '
                    'angle[4] a=pi/2; int first=-5; int last=999; int zero=0; int visits=0; int low=2; int high=1; '
                    +wrapper.format(call), (('a', None), ('phase', None)), (4, 8))
        self.check_exact('angle[4] phase=pi; int first=-5; int last=999; int zero=0; '
            'bool no=false&&bool(angle[4](phase[first:zero:last])); '
            'bool yes=true||bool(angle[4](rotl(phase[first:last],1/zero)));',
            (('no', None), ('yes', None)), (0, 1))

    def test_failure_order_extraction_guards_and_destination_writes(self):
        # Mark entry into each range leaf and each extracted selected bit.
        # The expected sequence does not depend on temporary variable names.
        left = ['20001']+['20002']*4
        right = ['30001']+['30002']*4
        cases = (('last=2;', 'left[first:last]^right[1/zero:last]', ['20001'], 'assertion failed'),
                 ('distance=-1;', '(left[first:last]<<distance)^right[1/zero:last]', left, 'assertion failed'),
                 ('last=2;', 'left[first:high]^right[first:last]', left+['30001'], 'assertion failed'),
                 ('', 'rotl(left[first:last],1/zero)', left, 'division by zero'),
                 ('', 'left[first:last]^(right[first:last]<<(1/zero))', left+right, 'division by zero'),
                 ('', 'phases[1/zero][first:zero:999]', ['40001'], 'division by zero'),
                 ('', 'left[first:last]^phases[1/zero][first:last]', left+['40001'], 'division by zero'),
                 ('', 'rotl(left[first:last],huge)', left, 'out of range'),
                 ('last=-1;', 'left[high:-1:last]', ['20001'], 'assertion failed'),
                 ('first=2;last=1;', 'left[first:last]', ['20001'], 'assertion failed'),
                 ('', 'left[first:zero:last]', ['20001'], 'assertion failed'),
                 ('first=-5;', 'left[first:last]', ['20001'], 'assertion failed'))
        for constant in (False, True):
            for update, expression, expected, error in cases:
                prefix = 'const ' if constant else ''
                source = (prefix+'angle[4] left=angle[4]("1101"); '+prefix+'angle[4] right=angle[4]("1010"); '
                    'array[angle[4],2] phases={left,right}; int first=0; int last=3; int high=3; int zero=0; '
                    f'uint huge={self.module.QASM2QCXConverter.QCX_UINT_MAX}; int distance=1; angle[4] a; '
                    +update+f'a=angle[4]({expression});')
                program = self.module.openqasm3.parser.parse('OPENQASM 3; '+source)
                c = self.module.QASM2QCXConverter(program)
                original = c._QASM2QCXConverter__runtime_angle_bit_range_value
                @contextlib.contextmanager
                def marked(selection, width):
                    name = (selection.source.name if isinstance(selection.source, self.module.ast.Identifier)
                            else selection.source.collection.name)
                    marker = {'left': 20000, 'right': 30000, 'phases': 40000}[name]
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
                self.assertEqual(c._QASM2QCXConverter__captured_bit_register_values, {})
                target = c._QASM2QCXConverter__capitalize_variable_name('a')
                lines = []
                for line in c:
                    if line.startswith(f'LET {target} := QASM2QCX_'):
                        lines.append('PRINTLN 99999')
                    lines.append(line)
                self.assertIn('PRINTLN 99999', lines)
                result = self.execute(lines)
                self.assertNotEqual(result.returncode, 0, source)
                self.assertIn(error, result.stderr, source)
                self.assertNotIn('runtime error:', result.stderr)
                self.assertEqual(result.stdout.split(), expected, source)
                type(self).checked += 1

    def test_later_failures_preserve_prior_scalar_and_array_writes(self):
        for array in (False, True):
            for update, error in (('last=2;', 'assertion failed'), ('distance=-1;', 'assertion failed'),
                                  ('rlast=2;', 'assertion failed'), ('divisor=0;', 'division by zero')):
                declaration = 'array[angle[4],2] phases; int slot=1;' if array else 'angle[4] a;'
                target = 'phases[slot]' if array else 'a'
                source = ('angle[4] left=angle[4]("1101"); angle[4] right=angle[4]("1010"); '
                    'int first=0; int last=3; int rlast=3; int distance=1; int divisor=1; int iteration=0; '
                    +declaration+' while(iteration<2) { '
                    +target+'=angle[4]((left[first:last]<<distance)^(right[first:rlast]>>(1/divisor))); '
                    'iteration+=1; '+update+' }')
                program = self.module.openqasm3.parser.parse('OPENQASM 3; '+source)
                c = self.module.QASM2QCXConverter(program)
                c.visit(program)
                self.assertEqual(c._QASM2QCXConverter__used_temporary_variables, set())
                name = c._QASM2QCXConverter__capitalize_variable_name('phases' if array else 'a')
                lines = []
                for line in c:
                    if (line.startswith(f'LET {name}:' if array else f'LET {name} := ')
                            and ' := QASM2QCX_' in line):
                        operand = line.split()[1]
                        lines.extend(['PRINTLN 99999', line, f'PRINTLN {operand}'])
                    else:
                        lines.append(line)
                self.assertIn('PRINTLN 99999', lines)
                result = self.execute(lines)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(error, result.stderr)
                self.assertNotIn('runtime error:', result.stderr)
                self.assertEqual(result.stdout.split(), ['99999', '15'])
                type(self).checked += 1

    def test_destination_index_failure_precedes_source_range_checks(self):
        source = ('angle[4] phase=pi; array[angle[4],2] phases; '
                  'int first=0; int last=3; int zero=0; '
                  'phases[1/zero]=angle[4](phase[first:zero:last]);')
        program = self.module.openqasm3.parser.parse('OPENQASM 3; '+source)
        c = self.module.QASM2QCXConverter(program)
        c.visit(program)
        self.assertEqual(c._QASM2QCXConverter__used_temporary_variables, set())
        name = c._QASM2QCXConverter__capitalize_variable_name('phases')
        lines = []
        for line in c:
            if line.startswith(f'LET {name}:') and ' := QASM2QCX_' in line:
                lines.append('PRINTLN 99999')
            lines.append(line)
        self.assertIn('PRINTLN 99999', lines)
        result = self.execute(lines)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('division by zero', result.stderr)
        self.assertNotIn('runtime error:', result.stderr)
        self.assertEqual(result.stdout, '')
        type(self).checked += 1

    def test_gate_parameters_modifiers_and_fusion(self):
        from qasm2qcx_u_power_numerical import BackendAuditTests
        phase = 3*math.pi/4
        prefix = ('include "stdgates.inc"; qubit[2] q; angle[8] phase=angle[8]("00000110"); '
                  'int first=0; int last=3; h q; ')
        for fused in (False, True):
            for gate, expected in (
                    ('p(angle[4](phase[first:last])) q[0];',
                     tuple(cmath.exp(1j*phase)/2 if i & 1 else 0.5 for i in range(4))),
                    ('pow(0.5) @ p(angle[4](phase[first:last])) q[0];',
                     tuple(cmath.exp(0.5j*phase)/2 if i & 1 else 0.5 for i in range(4))),
                    ('ctrl @ p(angle[4](phase[first:last])) q[0],q[1];',
                     (0.5, 0.5, 0.5, cmath.exp(1j*phase)/2)),
                    ('negctrl @ gphase(angle[4](phase[first:last])) q[0];',
                     tuple(0.5 if i & 1 else cmath.exp(1j*phase)/2 for i in range(4)))):
                BackendAuditTests.check(self, self.program(prefix+gate, fused=fused), expected)
            lines = self.program('include "stdgates.inc"; qubit q; angle[4] phase=pi/2; '
                'int first=0; int last=3; rx(sin(angle[4](phase[first:last]))) q;', fused=fused)
            BackendAuditTests.check(self, lines, (math.cos(0.5), -1j*math.sin(0.5)))

    def test_measurement_derived_angle_sources(self):
        for anticorrelated in (False, True):
            for _ in range(8):
                self.check_exact('include "stdgates.inc"; qubit[2] q; reset q; h q[0]; cx q[0],q[1]; '
                    +('x q[1]; ' if anticorrelated else '')+
                    'bit[2] measured=measure q; angle[2] phase=angle[2](measured); int first=0; int last=1; '
                    'angle[2] a=angle[2](phase[first:last]^phase[last:-1:first]); '
                    f'bool same=a==angle[2]("{3 if anticorrelated else 0:02b}"); '
                    'bool preserved=bit[2](angle[2](phase[first:last]))==measured;',
                    (('same', None), ('preserved', None)), (1, 1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra', type=pathlib.Path)
    arguments = parser.parse_args()
    BackendTests.bra = arguments.bra.resolve() if arguments.bra else None
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(BackendTests))
    print(f'runtime angle-bit range ANGLE cast programs checked: {BackendTests.checked}')
    raise SystemExit(not result.wasSuccessful())
