#!/usr/bin/env python3
"""Audit exact ANGLE casts from runtime register ranges (#525).

Run with --bra /path/to/bra, including a non-MPI UBSan build. References
use independent little-endian bit vectors; no converter folding or numeric
float-to-angle conversion supplies an expected bit pattern.
"""

import argparse
import cmath
import math
import pathlib
import resource
import unittest

import qasm2qcx_angle_bit_cast_numerical as backend
from qasm2qcx_runtime_range_equality_numerical import binary, literal, selected, unary


def packed(bits):
    return sum(bit * 2**index for index, bit in enumerate(bits))


class BackendTests(unittest.TestCase):
    bra = None
    checked = 0
    setUp = backend.BackendTests.setUp
    execute = backend.BackendTests.execute
    program = backend.BackendTests.program
    check_exact = backend.BackendTests.check_exact

    def test_all_widths_exact_roundtrips_and_uint_bridge(self):
        native = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        for width in range(1, native + 1):
            # The source parent need not fit in UINT; the selected count does.
            size = 129
            patterns = {0, (1 << width) - 1, 1 << (width - 1),
                        int(('10010110' * width)[:width], 2)}
            for pattern in sorted(patterns):
                bits = selected(pattern, size, (0, 1, width - 1))
                for constant in (False, True):
                    source = ('const ' if constant else '') + f'bit[{size}] b="{pattern:0{size}b}"; '
                    source += (f'int first=0; int last={width - 1}; '
                               f'angle[{width}] a=angle[{width}](b[first:last]); '
                               f'bit[{width}] back=bit[{width}](a); '
                               f'angle[{width}] reverse=angle[{width}](b[last:-1:first]); '
                               f'uint[{width}] bridge=uint[{width}](bit[{width}](angle[{width}](b[first:last]))); '
                               f'bool same=b=="{pattern:0{size}b}";')
                    self.check_exact(source, (('a', None), ('back', width), ('reverse', None),
                                              ('bridge', None), ('same', None)),
                                     (packed(bits), packed(bits), packed(bits[::-1]), packed(bits), 1))

    def test_unsized_native_width_preserves_high_bits_and_angle_arithmetic(self):
        width = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        mask = (1 << width) - 1
        for pattern in (0, 1, 1 << (width - 1), mask):
            self.check_exact(f'bit[{width}] b="{pattern:0{width}b}"; int first=0; int last={width-1}; '
                'angle a=angle(b[first:last]); angle shifted=a+angle(pi); '
                f'bit[{width}] back=bit[{width}](angle(b[first:last])); '
                'angle inverse=~angle(b[first:last]);',
                (('a', None), ('back', width), ('shifted', None), ('inverse', None)),
                (pattern, pattern, (pattern + (1 << (width-1))) & mask, pattern ^ mask))

    def test_nested_bitwise_shift_rotation_trees_against_bit_vectors(self):
        native = self.module.QASM2QCXConverter
        for pattern in range(16):
            bits = selected(pattern, 4, (0, 1, 3))
            reverse = selected(pattern, 4, (3, -1, 0))
            fixed = ('"0101"', [1, 0, 1, 0])
            for distance in (0, 1, 4, 5, native.QCX_UINT_MAX):
                turn = min(distance, native.QCX_INT_MAX)
                left = ('b[first:last]', bits)
                right = ('b[last:-1:first]', reverse)
                nodes = [unary(binary(left, right, '^'), '~'),
                         binary(unary(left, '<<', distance, 'distance'), fixed, '|'),
                         unary(binary(unary(left, '~'), right, '&'), '>>', distance, 'distance'),
                         unary(binary(left, unary(right, '~'), '^'), 'rotl', turn, 'turn'),
                         binary(unary(left, 'rotr', turn, 'turn'), fixed, '&')]
                source = (f'bit[4] b="{pattern:04b}"; int first=0; int last=3; '
                          f'uint distance={distance}; uint turn={turn}; ')
                outputs, expected = [], []
                for index, (expression, value) in enumerate(nodes):
                    source += f'angle[4] a{index}=angle[4]({expression}); '
                    outputs.append((f'a{index}', None))
                    expected.append(packed(value))
                self.check_exact(source, outputs, expected)

    def test_strides_negative_aliases_and_signed_rotation_extremes(self):
        native = self.module.QASM2QCXConverter
        for size, bounds in ((1, (-1, -1, -1)), (8, (1, 2, 7)),
                             (8, (-1, -2, -7)), (129, (128, -3, 119))):
            pattern = int(('10110010' * size)[:size], 2)
            bits = selected(pattern, size, bounds)
            width = len(bits)
            for distance in (-1, -9, native.QCX_INT_MIN, native.QCX_INT_MAX):
                for constant in (False, True):
                    first, step, last = bounds
                    source = ('const ' if constant else '') + f'bit[{size}] b="{pattern:0{size}b}"; '
                    source += f'int first={first}; int step={step}; int last={last}; int count={distance}; '
                    rotated = unary(('b[first:step:last]', bits), 'rotl', distance, 'count')
                    node = unary(rotated, '~')
                    expression, expected = unary(node, 'rotr', distance, 'count')
                    source += (f'angle[{width}] rotated=angle[{width}]({rotated[0]}); '
                               f'angle[{width}] a=angle[{width}]({expression});')
                    self.check_exact(source, (('rotated', None), ('a', None)),
                                     (packed(rotated[1]), packed(expected)))

    def test_fixed_leaves_runtime_angle_bits_arrays_and_nested_cast_boundaries(self):
        prefix = ('bit[8] b="11010011"; const bit[8] C="10100110"; bit[8] wide=C; '
                  'bit[4] fixed="1010"; uint[4] number=3; angle[4] phase=pi; '
                  'array[angle[4],2] phases={pi,pi/2}; array[int,2] positions={0,3}; '
                  'int first=1; int step=2; int last=7; int slot=0; int low=0; int high=3;')
        for operand, bits in (('"0101"', 5), ('fixed', 10), ('C[{7,0,5,1}]', 13),
                               ('wide[{last,0,5,1}]', 13), ('bit[4](number)', 3),
                               ('bit[4](phase)', 8), ('phase[{low,1,2,high}]', 8),
                               ('phases[slot][{low,1,2,high}]', 8),
                               ('bit[4](angle[4](C[low:high]))', 6)):
            for left, right in (('b[first:step:last]', operand), (operand, 'b[first:step:last]')):
                self.check_exact(prefix + f'phases[positions[slot]]=angle[4]({left}^{right}); '
                    'angle[4] out=phases[0]; bool same=b=="11010011";',
                    (('out', None), ('same', None)), (9 ^ bits, 1))
        self.check_exact('bit[4] b="1011"; angle[4] phase=pi; int index=2; '
            'angle[1] a=angle[1](b[index:index]^phase[index]);', (('a', None),), (0,))

    def test_overlapping_roundtrips_and_fresh_loop_evaluation(self):
        for pattern in range(16):
            reverse = int(f'{pattern:04b}'[::-1], 2)
            self.check_exact(f'bit[4] b="{pattern:04b}"; int first=0; int last=3; '
                'angle[4] saved=angle[4](b[first:last]); '
                'b=bit[4](angle[4](b[last:-1:first])); '
                'angle[4] fresh=angle[4](b[first:last]); '
                'b[{3,2,1,0}]=bit[4](angle[4](b[first:last]));',
                (('saved', None), ('fresh', None), ('b', 4)), (pattern, reverse, pattern))
        for constant in (False, True):
            source = ('const ' if constant else '') + 'bit[8] b="10110101"; '
            source += ('int first=0; int last=3; uint total=0; array[angle[4],2] phases={0.0,0.0}; '
                       'while(first<5) { last=first+3; phases[0]=angle[4](b[first:last]); first+=1; '
                       'if(first==2) { continue; } total+=uint[4](bit[4](phases[0])); '
                       'if(first==4) { break; } } angle[4] out=phases[0];')
            expected = sum(packed(selected(181, 8, (i, 1, i+3))) for i in (0, 2, 3))
            self.check_exact(source, (('total', None), ('out', None), ('first', None)),
                             (expected, packed(selected(181, 8, (3, 1, 6))), 4))
        self.check_exact('bit[4] b="0000"; int first=0; int last=3; int visits=0; '
            'while(angle[4](b[first:last])!=angle[4](pi)) { '
            'b=bit[4](angle[4](b[first:last])+angle[4](pi/8)); visits+=1; }',
            (('visits', None), ('b', 4)), (8, 8))

    def test_scalar_consumers_and_angle_type_semantics(self):
        for pattern in range(16):
            self.check_exact(f'bit[4] b="{pattern:04b}"; int first=0; int last=3; '
                'angle[4] added=angle[4](b[first:last])+angle[4](pi); '
                'angle[4] times=angle[4](b[first:last])*uint[4](3); '
                'angle[8] widened=angle[4](b[first:last]); '
                'bool truth=bool(angle[4](b[first:last])); '
                'uint ones=popcount(bit[4](angle[4](b[first:last]))); '
                'bool equal=angle[4](b[first:last])==angle[4](bit[4](angle[4](b[first:last]))); '
                'bit[4] shifted="1111"; shifted>>=uint[4](bit[4](angle[4](b[first:last])));',
                (('added', None), ('times', None), ('widened', None), ('truth', None),
                 ('ones', None), ('equal', None), ('shifted', 4)),
                ((pattern+8) % 16, (pattern*3) % 16, pattern*16, int(bool(pattern)),
                 pattern.bit_count(), 1, 15 >> pattern))

    def test_skipped_paths_defer_domains(self):
        calls = ('a=angle[4](b[first:last]);', 'a=angle[4](b[first:0:last]);',
                 'a=angle[4]((b[first:last]<<-1)^C[1/zero:3]);',
                 'a=angle[4](rotl(b[first:last],1/zero));')
        wrappers = ('if(false) {{ {} }}', 'while(false) {{ {} }}',
                    'for int i in [1:0] {{ {} }}', 'for int i in [low:high] {{ {} }}',
                    'while(true) {{ break; {} }}', 'while(visits<2) {{ visits+=1; continue; {} }}')
        for call in calls:
            for wrapper in wrappers:
                self.check_exact('bit[4] b="1010"; const bit[4] C="1001"; angle[4] a=pi; '
                    'int first=0; int last=999; int zero=0; int low=2; int high=1; int visits=0; '
                    + wrapper.format(call), (('a', None), ('b', 4)), (8, 10))
        self.check_exact('bit[4] b; int first=-1; int last=999; int zero=0; '
            'bool no=false&&bool(angle[4](b[first:last])); '
            'bool yes=true||bool(angle[4](rotl(b[first:last],1/zero)));',
            (('no', None), ('yes', None)), (0, 1))

    def test_failures_preserve_operand_order_and_precede_destination_writes(self):
        cases = (('last=2;', 'left[first:last]^right[1/zero:3]', (0, 0), 'assertion failed'),
                 ('distance=-1;', '(left[first:last]<<distance)^right[1/zero:3]', (4, 0), 'assertion failed'),
                 ('last=2;', 'left[0:high]^right[first:last]', (4, 0), 'assertion failed'),
                 ('', 'rotl(left[first:last],1/zero)^right[first:last]', (4, 0), 'division by zero'),
                 ('', 'left[first:last]^(right[first:last]<<(1/zero))', (4, 4), 'division by zero'),
                 ('last=4;', 'left[first:last]', (0, 0), 'assertion failed'),
                 ('', 'left[first:zero:last]', (0, 0), 'assertion failed'),
                 ('first=-5;', 'left[first:last]', (0, 0), 'assertion failed'),
                 ('', 'rotl(left[first:last],huge)', (4, 0), 'out of range'))
        for constant in (False, True):
            for update, operand, reads, error in cases:
                prefix = 'const ' if constant else ''
                source = (prefix+'bit[4] left="1101"; '+prefix+'bit[4] right="1010"; '
                    'int first=0; int last=3; int high=3; int zero=0; int distance=1; angle[4] a; '
                    f'uint huge={self.module.QASM2QCXConverter.QCX_UINT_MAX}; '
                    + update+f'a=angle[4]({operand});')
                program = self.module.openqasm3.parser.parse('OPENQASM 3; '+source)
                c = self.module.QASM2QCXConverter(program)
                c.visit(program)
                self.assertEqual(c._QASM2QCXConverter__used_temporary_variables, set())
                self.assertEqual(c._QASM2QCXConverter__captured_bit_register_values, {})
                lines = list(c)
                # Mark selected reads for mutable sources. Constant lookup
                # uses different code, so only destination writes are marked.
                if constant:
                    # Mutable sources below additionally expose read order.
                    instrumented = lines
                else:
                    instrumented = []
                    for line in lines:
                        for source_name, marker in (('left', 22222), ('right', 33333)):
                            name = c._QASM2QCXConverter__capitalize_variable_name(source_name)
                            if line.startswith('LET QASM2QCX_') and f' := {name}:' in line:
                                instrumented.append(f'PRINTLN {marker}')
                        instrumented.append(line)
                destination = c._QASM2QCXConverter__capitalize_variable_name('a')
                marked = []
                for line in instrumented:
                    if line.startswith(f'LET {destination} := QASM2QCX_'):
                        marked.append('PRINTLN 11111')
                    marked.append(line)
                self.assertIn('PRINTLN 11111', marked)
                result = self.execute(marked)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(error, result.stderr)
                self.assertNotIn('runtime error:', result.stderr)
                self.assertEqual(result.stdout.split(), [] if constant else
                                 ['22222']*reads[0]+['33333']*reads[1])
                type(self).checked += 1

    def test_later_failures_preserve_prior_scalar_and_array_writes(self):
        for constant in (False, True):
            for array in (False, True):
                for update, error in (('last=2;', 'assertion failed'),
                                      ('distance=-1;', 'assertion failed'),
                                      ('rlast=2;', 'assertion failed'),
                                      ('divisor=0;', 'division by zero')):
                    prefix = 'const ' if constant else ''
                    declaration = 'array[angle[4],2] phases; int slot=1;' if array else 'angle[4] a;'
                    target = 'phases[slot]' if array else 'a'
                    source = (prefix+'bit[4] left="1101"; '+prefix+'bit[4] right="1010"; '
                        'int first=0; int last=3; int rlast=3; int distance=1; int divisor=1; int iteration=0; '
                        + declaration+' while(iteration<2) { '
                        + target+'=angle[4]((left[first:last]<<distance)^(right[first:rlast]>>(1/divisor))); '
                        + 'iteration+=1; '+update+' }')
                    program = self.module.openqasm3.parser.parse('OPENQASM 3; '+source)
                    c = self.module.QASM2QCXConverter(program)
                    c.visit(program)
                    self.assertEqual(c._QASM2QCXConverter__used_temporary_variables, set())
                    name = c._QASM2QCXConverter__capitalize_variable_name('phases' if array else 'a')
                    marked = []
                    for line in c:
                        if (line.startswith(f'LET {name}:' if array else f'LET {name} := ')
                                and ' := QASM2QCX_' in line):
                            operand = line.split()[1]
                            marked.extend(['PRINTLN 11111', line, f'PRINTLN {operand}'])
                        else:
                            marked.append(line)
                    self.assertIn('PRINTLN 11111', marked)
                    result = self.execute(marked)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(error, result.stderr)
                    self.assertNotIn('runtime error:', result.stderr)
                    self.assertEqual(result.stdout.split(), ['11111', '15'])
                    type(self).checked += 1

    def test_direct_gate_arguments_math_and_modifiers_with_fusion(self):
        from qasm2qcx_u_power_numerical import BackendAuditTests
        phase = 3*math.pi/4
        for fused in (False, True):
            prefix = ('include "stdgates.inc"; qubit[2] q; bit[8] b="00000110"; '
                      'int first=0; int last=3; h q; ')
            for gate, expected in (
                    ('p(angle[4](b[first:last])) q[0];',
                     tuple(cmath.exp(1j*phase)/2 if i & 1 else 0.5 for i in range(4))),
                    ('pow(0.5) @ p(angle[4](b[first:last])) q[0];',
                     tuple(cmath.exp(0.5j*phase)/2 if i & 1 else 0.5 for i in range(4))),
                    ('ctrl @ p(angle[4](b[first:last])) q[0],q[1];',
                     (0.5, 0.5, 0.5, cmath.exp(1j*phase)/2)),
                    ('negctrl @ gphase(angle[4](b[first:last])) q[0];',
                     tuple(0.5 if i & 1 else cmath.exp(1j*phase)/2 for i in range(4)))):
                BackendAuditTests.check(self, self.program(prefix+gate, fused=fused), expected)
            lines = self.program('include "stdgates.inc"; qubit q; bit[4] b="0100"; '
                'int first=0; int last=3; rx(sin(angle[4](b[first:last]))) q;', fused=fused)
            BackendAuditTests.check(self, lines, (math.cos(0.5), -1j*math.sin(0.5)))

    def test_measurement_sources_and_classical_feedback(self):
        for anticorrelated in (False, True):
            for _ in range(8):
                self.check_exact('include "stdgates.inc"; qubit[2] q; reset q; h q[0]; cx q[0],q[1]; '
                    + ('x q[1]; ' if anticorrelated else '')+
                    'bit[2] measured=measure q; int first=0; int last=1; '
                    'angle[2] a=angle[2](measured[first:last]^measured[last:-1:first]); '
                    f'bool same=a==angle[2]("{3 if anticorrelated else 0:02b}"); '
                    'bool preserved=bit[2](angle[2](measured[first:last]))==measured;',
                    (('same', None), ('preserved', None)), (1, 1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra', type=pathlib.Path)
    arguments = parser.parse_args()
    BackendTests.bra = arguments.bra.resolve() if arguments.bra else None
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(BackendTests))
    print(f'runtime-range ANGLE cast programs checked: {BackendTests.checked}')
    raise SystemExit(not result.wasSuccessful())
