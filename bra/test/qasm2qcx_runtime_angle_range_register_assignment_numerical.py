#!/usr/bin/env python3
"""Audit direct register copies from runtime angle-bit ranges (#533).

Run with --bra /path/to/bra, including a non-MPI UBSan build. References use
independently selected little-endian bit vectors, without UINT-width packing.
"""

import argparse
import cmath
import math
import pathlib
import resource
import unittest

import qasm2qcx_angle_bit_cast_numerical as backend
from qasm2qcx_runtime_range_equality_numerical import selected
from qasm2qcx_runtime_range_angle_cast_numerical import packed


class BackendTests(unittest.TestCase):
    bra = None
    checked = 0
    setUp = backend.BackendTests.setUp
    execute = backend.BackendTests.execute
    program = backend.BackendTests.program
    check_exact = backend.BackendTests.check_exact

    def test_documented_runtime_angle_range_register_copy_example(self):
        document = (pathlib.Path(__file__).resolve().parents[1] / 'qcx/qasm2qcx.md').read_text()
        marker = '<!-- runtime-angle-range-register-copy-example -->'
        self.assertEqual(document.count(marker), 1)
        source = document.split(marker, 1)[1].split('```qasm\n', 1)[1].split('```', 1)[0]
        source = source.replace('OPENQASM 3.0;', '', 1)
        self.check_exact(source, (('saved', 4), ('element', 4), ('fresh', 4), ('repeated', 8)),
                         (10, 10, 0, 187))

    def test_all_source_widths_patterns_directions_and_sources(self):
        native = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        for width in range(1, native+1):
            mask = (1 << width)-1
            patterns = range(1 << width) if width <= 5 else sorted({0, mask, 1 << (width-1),
                                                                    int(('10100110'*width)[:width], 2)})
            for pattern in patterns:
                other = pattern ^ mask
                for reverse in (False, True):
                    first, step, last = (width-1, -1, 0) if reverse else (0, 1, width-1)
                    bits = selected(pattern, width, (first, step, last))
                    other_bits = selected(other, width, (first, step, last))
                    self.check_exact(f'const angle[{width}] C=angle[{width}]("{pattern:0{width}b}"); '
                        f'const angle[{width}] D=angle[{width}]("{other:0{width}b}"); '
                        f'angle[{width}] phase=C; array[angle[{width}],2] phases={{C,D}}; '
                        f'int first={first}; int step={step}; int last={last}; int slot=1; '
                        f'bit[{width}] initial=phase[first:step:last]; bit[{width}] copy; copy=C[first:step:last]; '
                        f'bit[{width}] element=phases[slot][first:step:last]; bool stable=phase==C && phases[slot]==D;',
                        (('initial', width), ('copy', width), ('element', width), ('stable', None)),
                        (packed(bits), packed(bits), packed(other_bits), 1))

    def test_repeated_aliases_and_destinations_beyond_native_width(self):
        native = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        for width in range(1, native+1):
            mask = (1 << width)-1
            for pattern in sorted({0, mask, 1 << (width-1)}):
                # The raw progression visits each position twice, once
                # negative and once nonnegative. No native UINT stores the result.
                self.check_exact(f'angle[{width}] phase=angle[{width}]("{pattern:0{width}b}"); '
                    f'int first=-{width}; int last={width-1}; bit[{2*width}] twice=phase[first:last];',
                    (('twice', 2*width),), (pattern | (pattern << width),))
        self.check_exact('angle[4] phase=angle[4]("1011"); int first=-1; int last=3; '
                         'bit[5] out=phase[first:last];', (('out', 5),), (23,))

    def test_strides_negative_aliases_and_unsigned_bounds(self):
        native = self.module.QASM2QCXConverter
        for width, bounds in ((1, (-1, -1, -1)), (4, (-1, 1, 3)), (8, (1, 2, 6)),
                              (8, (-1, -2, -7)), (native.QCX_UINT_WIDTH, (3, -1, 0))):
            pattern = int(('10110100'*width)[:width], 2)
            bits = selected(pattern, width, bounds)
            first, step, last = bounds
            self.check_exact(f'angle[{width}] phase=angle[{width}]("{pattern:0{width}b}"); '
                f'int first={first}; int step={step}; int last={last}; bit[{len(bits)}] out; '
                'out=phase[first:step:last];', (('out', len(bits)),), (packed(bits),))
        self.check_exact(f'angle[1] phase=pi; uint first=0; uint step={native.QCX_UINT_MAX}; uint last=0; '
                         'bit[1] out=phase[first:step:last];', (('out', 1),), (1,))

    def test_fresh_sources_array_indices_iterators_and_loop_transfers(self):
        self.check_exact('const angle[8] C=angle[8]("10110101"); angle[8] phase=C; '
            'int first=0; int last=3; uint total=0; bit[4] out; '
            'while(first<5) { last=first+3; out=phase[first:last]; first+=1; '
            'if(first==2) { continue; } total+=uint[4](out); if(first==4) { break; } }',
            (('total', None), ('out', 4), ('first', None)),
            (sum(packed(selected(181, 8, (i, 1, i+3))) for i in (0, 2, 3)),
             packed(selected(181, 8, (3, 1, 6))), 4))
        self.check_exact('array[angle[4],2] phases={angle[4]("1011"),angle[4]("1101")}; '
            'int first=0; int last=3; uint total=0; bit[4] out; '
            'for angle[4] item in phases { out=item[first:last]; total+=uint[4](out); } '
            'int slot=0; while(slot<2) { out=phases[slot][first:last]; slot+=1; }',
            (('total', None), ('out', 4), ('slot', None)), (24, 13, 2))

    def test_skipped_domains_do_not_write(self):
        calls = ('out=phase[first:last];', 'out=phase[first:zero:last];',
                 'out=phases[1/zero][first:last];', 'out=phase[first:1/zero:last];')
        wrappers = ('if(false) {{ {} }}', 'while(false) {{ {} }}', 'for int i in [1:0] {{ {} }}',
                    'for int i in [low:high] {{ {} }}', 'while(true) {{ break; {} }}',
                    'while(visits<2) {{ visits+=1; continue; {} }}')
        for call in calls:
            for wrapper in wrappers:
                self.check_exact('angle[4] phase=pi; array[angle[4],2] phases={pi,0.0}; bit[4] out="1010"; '
                    'int first=-5; int last=999; int zero=0; int low=2; int high=1; int visits=0; '
                    +wrapper.format(call), (('out', 4), ('phase', None)), (10, 8))

    def test_native_endpoint_bounds_and_steps(self):
        native = self.module.QASM2QCXConverter
        for value, kind in ((native.QCX_INT_MIN, 'int'), (native.QCX_UINT_MAX, 'uint')):
            self.check_exact(f'angle[1] phase=pi; int first=0; {kind} step={value}; int last=0; '
                             'bit[1] out=phase[first:step:last];', (('out', 1),), (1,))
            source = (f'angle[4] phase=pi; {kind} first={value}; int last=0; '
                      'bit[1] out=phase[first:last];')
            result = self.execute(self.program(source))
            self.assertNotEqual(result.returncode, 0, source)
            self.assertIn('assertion failed', result.stderr, source)
            self.assertNotIn('runtime error:', result.stderr)
            type(self).checked += 1

    def test_failures_precede_extraction_and_destination_writes(self):
        cases = (('last=2;', 'phase[first:last]', 'assertion failed'),
                 ('last=4;', 'phase[first:last]', 'assertion failed'),
                 ('first=-5;', 'phase[first:last]', 'assertion failed'),
                 ('', 'phase[first:zero:last]', 'assertion failed'),
                 ('', 'phase[first/zero:last]', 'division by zero'),
                 ('', 'phases[1/zero][first:zero:last]', 'division by zero'),
                 ('first=2;last=1;', 'phase[first:last]', 'assertion failed'))
        for constant in (False, True):
            for update, expression, error in cases:
                source = (('const ' if constant else '')+'angle[4] phase=angle[4]("1011"); '
                    'array[angle[4],2] phases={phase,phase}; int first=0; int last=3; int zero=0; bit[4] out="1010"; '
                    +update+f'out={expression};')
                program = self.module.openqasm3.parser.parse('OPENQASM 3; '+source)
                c = self.module.QASM2QCXConverter(program); c.visit(program)
                self.assertEqual(c._QASM2QCXConverter__used_temporary_variables, set())
                target = c._QASM2QCXConverter__capitalize_variable_name('out')
                lines = []
                for line in c:
                    if ' >>= QASM2QCX_INT_' in line:
                        lines.append('PRINTLN 22222')
                    if line.startswith(f'LET {target}:QASM2QCX_'):
                        lines.append('PRINTLN 99999')
                    lines.append(line)
                self.assertIn('PRINTLN 22222', lines)
                self.assertIn('PRINTLN 99999', lines)
                result = self.execute(lines)
                self.assertNotEqual(result.returncode, 0, source)
                self.assertIn(error, result.stderr, source)
                self.assertNotIn('runtime error:', result.stderr)
                self.assertEqual(result.stdout, '', source)
                type(self).checked += 1

    def test_later_failure_preserves_earlier_assignment(self):
        program = self.module.openqasm3.parser.parse('OPENQASM 3; angle[4] phase=angle[4]("1011"); '
            'int first=0; int last=3; int iteration=0; bit[4] out="0000"; '
            'while(iteration<2) { out=phase[first:last]; iteration+=1; last=2; }')
        c = self.module.QASM2QCXConverter(program); c.visit(program)
        name = c._QASM2QCXConverter__capitalize_variable_name('out'); lines = []
        for line in c:
            if line.startswith(f'LET {name}:QASM2QCX_'):
                lines.extend(['PRINTLN 99999', line, f'PRINTLN {line.split()[1]}'])
            else:
                lines.append(line)
        self.assertIn('PRINTLN 99999', lines)
        result = self.execute(lines)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('assertion failed', result.stderr)
        self.assertNotIn('runtime error:', result.stderr)
        self.assertEqual(result.stdout.split(), ['99999', '1', '99999', '1', '99999', '0', '99999', '1'])
        type(self).checked += 1

    def test_impossible_wide_count_fails_before_extraction(self):
        width = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        source = (f'angle[{width}] phase=pi; int first=-{width}; int last={width-1}; '
                  f'bit[{2*width+1}] out; out=phase[first:last];')
        program = self.module.openqasm3.parser.parse('OPENQASM 3; '+source)
        c = self.module.QASM2QCXConverter(program); c.visit(program)
        self.assertEqual(c._QASM2QCXConverter__used_temporary_variables, set())
        target = c._QASM2QCXConverter__capitalize_variable_name('out'); lines = []
        for line in c:
            if ' >>= QASM2QCX_INT_' in line:
                lines.append('PRINTLN 22222')
            if line.startswith(f'LET {target}:QASM2QCX_'):
                lines.append('PRINTLN 99999')
            lines.append(line)
        self.assertIn('PRINTLN 22222', lines)
        self.assertIn('PRINTLN 99999', lines)
        result = self.execute(lines)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('assertion failed', result.stderr)
        self.assertNotIn('runtime error:', result.stderr)
        self.assertEqual(result.stdout, '')
        type(self).checked += 1

    def test_measurement_feedback_and_gate_phases(self):
        for anticorrelated in (False, True):
            for _ in range(8):
                self.check_exact('include "stdgates.inc"; qubit[2] q; reset q; h q[0]; cx q[0],q[1]; '
                    +('x q[1]; ' if anticorrelated else '')+
                    'bit[2] measured=measure q; angle[2] phase=angle[2](measured); '
                    'int first=0; int last=1; bit[2] out=phase[first:last]; '
                    'bool same=out==measured;', (('same', None),), (1,))
        from qasm2qcx_u_power_numerical import BackendAuditTests
        expected = (1/math.sqrt(2), cmath.exp(3j*math.pi/4)/math.sqrt(2))
        source = ('include "stdgates.inc"; qubit q; angle[4] phase=angle[4]("0110"); '
                  'int first=0; int last=3; bit[4] out=phase[first:last]; h q; p(angle[4](out)) q;')
        for fused in (False, True):
            BackendAuditTests.check(self, self.program(source, fused=fused), expected)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra', type=pathlib.Path)
    arguments = parser.parse_args()
    BackendTests.bra = arguments.bra.resolve() if arguments.bra else None
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(BackendTests))
    print(f'runtime angle-bit range register copy programs checked: {BackendTests.checked}')
    raise SystemExit(not result.wasSuccessful())
