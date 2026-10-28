#!/usr/bin/env python3
"""Execute one-dimensional angle-array loops and inspect exact UINT storage (#420).

Pass --bra /path/to/bra for the default-double non-MPI backend audit.
"""

import argparse
import cmath
import math
import pathlib
import resource
import unittest

import qasm2qcx_angle_array_numerical as arrays


class BackendTests(unittest.TestCase):
    bra = None
    checked = 0
    setUp = arrays.BackendTests.setUp
    execute = arrays.BackendTests.execute
    program = arrays.BackendTests.program
    check = arrays.BackendTests.check
    check_initializers = arrays.BackendTests.check_initializers

    @staticmethod
    def angle(width, bits):
        return f'angle[{width}]("{bits:0{width}b}")'

    def test_live_reads_copied_current_value_and_loop_transfers(self):
        native = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        for width in (1, 4, 8, native):
            mask = (1 << width) - 1
            for size in (1, 3):
                initial = [mask, 1, mask - 1][:size]
                for mode in ('all', 'continue', 'break'):
                    values = initial.copy()
                    total = position = 0
                    for index in range(size):
                        copied = values[index]
                        position += 1
                        if mode == 'continue' and position == 2:
                            continue
                        if mode == 'break' and position == 3:
                            break
                        if position < size:
                            values[position] = (values[position] + 1) & mask
                        values[index] = 0
                        total = (total + copied) & mask
                    transfer = ('if (i == 2) { continue; }' if mode == 'continue'
                                else 'if (i == 3) { break; }' if mode == 'break' else '')
                    initializer = ', '.join(self.angle(width, value) for value in initial)
                    self.check_initializers(f'array[angle[{width}], {size}] phases = {{{initializer}}}; '
                        f'angle[{width}] total; int i = 0; for angle[{width}] phase in phases {{ '
                        f'i += 1; {transfer} if (i < {size}) {{ phases[i] += {self.angle(width, 1)}; }} '
                        f'phases[i - 1] = 0.0; total += phase; }}',
                        (('phases', size),), values, ('total', 'i'), (total, position))

    def test_nested_widths_and_iterator_shadowing_restore_outer_values(self):
        self.check_initializers('angle[4] phase = angle[4]("0011"); angle[4] total; '
            'array[angle[4], 2] small = {angle[4]("0001"), angle[4]("0010")}; '
            'array[angle[8], 1] large = {angle[8]("00010000")}; '
            'for angle[4] phase in small { total += phase; '
            'for angle[8] phase in large { total += angle[4](phase); } total += phase; } '
            'total += phase;', (('small', 2), ('large', 1)), (1, 2, 16), ('total', 'phase'), (11, 3))
        self.check_initializers('array[angle[4], 2] phases = {angle[4]("0001"), angle[4]("0010")}; '
            'angle[4] total; for angle[4] phases in phases { total += phases; } total += phases[0];',
            (('phases', 2),), (1, 2), ('total',), (4,))

    def test_reentry_resets_index_and_keeps_live_source_changes(self):
        for outer in ('for int outer in [0:2]', 'for int outer in [0:stop]',
                      'for int outer in {0, stop, 1}', 'while (visits < 3)'):
            self.check_initializers('array[angle[4], 2] phases = {angle[4]("0001"), angle[4]("0010")}; '
                'angle[4] total; int visits = 0; int stop = 2; ' + outer + ' { visits += 1; '
                'for angle[4] phase in phases { phases[1] += angle[4]("0001"); total += phase; }}',
                (('phases', 2),), (1, 8), ('total', 'visits'), (2, 3))

    def test_sized_unsized_matching_at_native_width(self):
        native = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        for element, iterator in (('angle', f'angle[{native}]'), (f'angle[{native}]', 'angle')):
            self.check_initializers(f'array[{element}, 2] phases = {{0.0, pi}}; '
                f'angle total; for {iterator} phase in phases {{ total += phase; }}',
                (('phases', 2),), (0, 1 << (native - 1)), ('total',), (1 << (native - 1),))

    def test_nested_break_and_continue_apply_to_innermost_loop(self):
        self.check_initializers('array[angle[4], 3] phases = {0.0, pi / 2, pi}; '
            'int visits = 0; angle[4] total; '
            'for angle[4] phase in phases { '
            'for angle[4] other in phases { if (other == angle[4](0.0)) { continue; } '
            'if (other == angle[4](pi)) { break; } visits += 1; total += phase + other; }}',
            (('phases', 3),), (0, 4, 8), ('total', 'visits'), (8, 3))

    def test_skipped_loops_and_transfers_do_not_evaluate_unreached_reads(self):
        self.check_initializers('array[angle[4], 2] phases = {0.0, pi}; int bad = 2; angle[4] total; '
            'if (false) { for angle[4] phase in phases { total += phases[bad]; }} '
            'for int outer in [1:0] { for angle[4] phase in phases { total += phases[bad]; }} '
            'for angle[4] phase in phases { continue; total += phases[bad]; } '
            'for angle[4] phase in phases { total += phase; break; total += phases[bad]; }',
            (('phases', 2),), (0, 8), ('total',), (0,))

    def test_arithmetic_bit_functions_and_casts_at_every_width(self):
        native = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        for width in range(1, native + 1):
            mask = (1 << width) - 1
            values = sorted({0, 1, 1 << (width - 1), mask})
            size = len(values)
            one = self.angle(width, 1)
            factor = min(3, mask)
            cases = (
                ('plus', f'phase + {one}', lambda value: (value + 1) & mask),
                ('minus', f'phase - {one}', lambda value: (value - 1) & mask),
                ('negative', '-phase', lambda value: -value & mask),
                ('complement', '~phase', lambda value: value ^ mask),
                ('conjunction', f'phase & {one}', lambda value: value & 1),
                ('disjunction', f'phase | {one}', lambda value: value | 1),
                ('exclusive', f'phase ^ {one}', lambda value: value ^ 1),
                ('times', f'phase * uint[{width}]({factor})', lambda value: value * factor & mask),
                ('divided', f'phase / uint[{width}]({factor})', lambda value: value // factor),
                ('leftshift', 'phase << shift', lambda value: value << 1 & mask),
                ('rightshift', 'phase >> shift', lambda value: value >> 1),
                ('oversized', f'phase << {width}', lambda value: 0),
                ('leftrotate', 'rotl(phase, -1)', lambda value: (value >> 1 | value << (width - 1)) & mask),
                ('rightrotate', 'rotr(phase, -1)', lambda value: (value << 1 | value >> (width - 1)) & mask),
                ('roundtrip', f'angle[{width}](bits)', lambda value: value),
            )
            declarations = ' '.join(f'array[angle[{width}], {size}] {name};' for name, _, _ in cases)
            assignments = ' '.join(f'{name}[i] = {expression};' for name, expression, _ in cases)
            initializer = ', '.join(self.angle(width, value) for value in values)
            source = (f'array[angle[{width}], {size}] phases = {{{initializer}}}; '
                      f'bit[{width}] bits; int i = 0; int shift = 1; '
                      f'array[uint, {size}] counts; array[uint[{width}], {size}] ratios; '
                      f'array[bool, {size}] equal; array[bool, {size}] ordered; {declarations} '
                      f'for angle[{width}] phase in phases {{ bits = bit[{width}](phase); {assignments} '
                      f'counts[i] = popcount(phase); ratios[i] = phase / {one}; '
                      f'equal[i] = phase == {one}; ordered[i] = phase > {one}; i += 1; }}')
            outputs = tuple((name, size) for name, _, _ in cases) + (
                ('counts', size), ('ratios', size), ('equal', size), ('ordered', size), ('phases', size))
            expected = [reference(value) for _, _, reference in cases for value in values]
            expected += [value.bit_count() for value in values] + values
            expected += [int(value == 1) for value in values] + [int(value > 1) for value in values] + values
            with self.subTest(width=width):
                self.check_initializers(source, outputs, expected)

    def test_cross_width_promotion_and_conversion_at_every_width(self):
        native = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        for width in range(1, native + 1):
            wide = min(width + 1, native)
            delta = wide - width
            mask = (1 << wide) - 1
            values = (1, (1 << width) - 1)
            for operation, reference in (
                    ('+', lambda value: ((value << delta) + 1) & mask),
                    ('-', lambda value: ((value << delta) - 1) & mask),
                    ('^', lambda value: (value << delta) ^ 1)):
                expected = [reference(value) for value in values]
                initializer = ', '.join(self.angle(width, value) for value in values)
                self.check_initializers(f'array[angle[{width}], 2] phases = {{{initializer}}}; '
                    f'array[angle[{wide}], 2] promoted; array[angle[{width}], 2] narrowed; int i = 0; '
                    f'for angle[{width}] phase in phases {{ '
                    f'promoted[i] = phase {operation} {self.angle(wide, 1)}; '
                    f'narrowed[i] = promoted[i]; i += 1; }}',
                    (('promoted', 2), ('narrowed', 2)), expected + [value >> delta for value in expected])

    def test_trigonometric_functions_and_nested_calls_at_every_width(self):
        from qasm2qcx_angle_trigonometric_numerical import reference
        native = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        for width in range(1, native + 1):
            values = sorted({0, 1 << (width - 1), (1 << width) - 1})
            # Avoid exact tangent poles; reached poles are audited separately.
            if width >= 2:
                values = [value for value in values if value not in (1 << (width - 2), 3 << (width - 2))]
            initializer = ', '.join(self.angle(width, value) for value in values)
            sine = sum(reference('sin', value, width) for value in values)
            cosine = sum(reference('cos', value, width) for value in values)
            tangent = sum(reference('tan', value, width) for value in values)
            nested = sum(math.cos(reference('sin', value, width)) for value in values)
            self.check(f'array[angle[{width}], {len(values)}] phases = {{{initializer}}}; '
                'float s = 0; float c = 0; float t = 0; float nested = 0; '
                f'for angle[{width}] phase in phases {{ s += sin(phase); c += cos(phase); '
                't += tan(phase); nested += cos(sin(phase)); }',
                (('s', False), ('c', False), ('t', False), ('nested', False)), (sine, cosine, tangent, nested))

    def test_cross_width_comparisons_at_every_width(self):
        native = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        for width in range(1, native + 1):
            wide = min(width + 1, native)
            values = (0, (1 << width) - 1)
            promoted = [value << (wide - width) for value in values]
            cases = (('equal', '==', lambda value: value == 1),
                     ('unequal', '!=', lambda value: value != 1),
                     ('less', '<', lambda value: value < 1),
                     ('greater', '>', lambda value: value > 1),
                     ('atmost', '<=', lambda value: value <= 1),
                     ('atleast', '>=', lambda value: value >= 1))
            initializer = ', '.join(self.angle(width, value) for value in values)
            declarations = ' '.join(f'array[bool, 2] {name};' for name, _, _ in cases)
            body = ' '.join(f'{name}[i] = phase {operator} {self.angle(wide, 1)};'
                            for name, operator, _ in cases)
            self.check_initializers(f'array[angle[{width}], 2] phases = {{{initializer}}}; '
                f'int i = 0; {declarations} for angle[{width}] phase in phases {{ {body} i += 1; }}',
                tuple((name, 2) for name, _, _ in cases),
                [int(reference(value)) for _, _, reference in cases for value in promoted])

    def test_phase_gates_modifiers_and_parameter_capture_with_fusion(self):
        from qasm2qcx_u_power_numerical import BackendAuditTests
        phase = 9 * math.pi / 8
        operations = [(f'{gate}(phase) q[0];', 1, lambda index: bool(index & 1))
                      for gate in ('p', 'phase', 'u1')]
        operations += [(f'{gate}(phase) q[0], q[1];', 1, lambda index: index & 3 == 3)
                       for gate in ('cp', 'cphase')]
        operations += [
            ('gphase(phase);', 1, lambda index: True),
            ('inv @ p(phase) q[0];', -1, lambda index: bool(index & 1)),
            ('pow(2) @ p(phase) q[0];', 2, lambda index: bool(index & 1)),
            ('pow(exponent) @ p(phase) q[0];', 2, lambda index: bool(index & 1)),
            ('pow(popcount(phase)) @ p(phase) q[0];', 2, lambda index: bool(index & 1)),
            ('pow(power) @ p(phase) q[0];', 0.5, lambda index: bool(index & 1)),
            ('inv @ pow(power) @ p(phase) q[0];', -0.5, lambda index: bool(index & 1)),
            ('inv @ cp(phase) q[0], q[1];', -1, lambda index: index & 3 == 3),
            ('pow(power) @ cp(phase) q[0], q[1];', 0.5, lambda index: index & 3 == 3),
            ('inv @ gphase(phase);', -1, lambda index: True),
            ('pow(exponent) @ gphase(phase);', 2, lambda index: True),
            ('pow(power) @ gphase(phase);', 0.5, lambda index: True),
            ('negctrl @ p(phase) q[0], q[1];', 1, lambda index: index & 3 == 2),
            ('ctrl @ gphase(phase) q[0];', 1, lambda index: bool(index & 1)),
            ('negctrl @ gphase(phase) q[0];', 1, lambda index: not index & 1),
            ('ctrl @ negctrl @ p(phase) q[0], q[1], q[2];', 1, lambda index: index & 7 == 5),
        ]
        for operation, factor, active in operations:
            expected = tuple(cmath.exp(1j * factor * phase) / math.sqrt(8) if active(index)
                             else 1 / math.sqrt(8) for index in range(8))
            source = ('include "stdgates.inc"; qubit[3] q; h q; int i = 0; int exponent = 2; const float power = 0.5; '
                      'array[angle[4], 2] phases = {angle[4]("0011"), angle[4]("0110")}; angle[4] scratch; '
                      'for angle[4] phase in phases { ' + operation + 'phases[i] = 0.0; i += 1; } '
                      'scratch = ~phases[0];')
            for fused in (False, True):
                with self.subTest(operation=operation, fused=fused):
                    BackendAuditTests.check(self, self.program(source, fused=fused), expected)

    def test_angle_function_results_as_rotation_parameters(self):
        from qasm2qcx_angle_trigonometric_numerical import reference
        from qasm2qcx_u_power_numerical import BackendAuditTests
        for gate, function, factor, prepare in (('rx', 'sin', 1, ''), ('ry', 'cos', 1, ''),
                                                 ('rx', 'sin', 2, ''), ('rz', 'tan', 1, 'h q;')):
            angle = factor * sum(reference(function, value, 4) for value in (3, 6))
            if gate == 'rx':
                expected = (math.cos(angle / 2), -1j * math.sin(angle / 2))
            elif gate == 'ry':
                expected = (math.cos(angle / 2), math.sin(angle / 2))
            else:
                expected = (cmath.exp(-0.5j * angle) / math.sqrt(2), cmath.exp(0.5j * angle) / math.sqrt(2))
            modifier = 'pow(popcount(phase)) @ ' if factor == 2 else ''
            source = ('include "stdgates.inc"; qubit q; ' + prepare +
                      'array[angle[4], 2] phases = {angle[4]("0011"), angle[4]("0110")}; '
                      f'for angle[4] phase in phases {{ {modifier}{gate}({function}(phase)) q; }}')
            for fused in (False, True):
                BackendAuditTests.check(self, self.program(source, fused=fused), expected)

    def test_integer_loops_depending_on_angle_iterators(self):
        self.check_initializers('array[angle[4], 2] phases = {angle[4]("0011"), angle[4]("0101")}; '
            'int visits = 0; int total = 0; '
            'for angle[4] phase in phases { '
            'for int i in [0:popcount(phase)] { visits += 1; total += i; } '
            'for int i in {phase / angle[4]("0001"), popcount(phase)} { total += i; } }',
            (('phases', 2),), (3, 5), ('total', 'visits'), (18, 6))

    def test_mixed_iterator_types_restore_angle_metadata_and_storage(self):
        self.check_initializers('array[angle[4], 2] phases = {angle[4]("0011"), angle[4]("0101")}; '
            'array[float, 1] floats = {0.5}; array[uint[4], 1] integers = {1}; '
            'array[bool, 1] booleans = {true}; array[complex, 1] complexes = {1im}; bit[2] bits = "01"; '
            'angle[4] total; int visits = 0; '
            'for angle[4] value in phases { total += value; '
            'for float value in floats { if (value > 0.0) { visits += 1; }} '
            'for uint[4] value in integers { visits += int(value); } '
            'for bool value in booleans { if (value) { visits += 1; }} '
            'for complex value in complexes { visits += int(imag(value)); } '
            'for bit value in bits { visits += int(value); } '
            'for int value in [0:1] { visits += value; } total += ~value; }',
            (('phases', 2),), (3, 5), ('total', 'visits'), (14, 12))

    def test_reached_domain_errors_and_skipped_invalid_operations(self):
        for expression in ('phase / zero', 'phase << negative', 'tan(phase)', 'phases[bad]'):
            prefix = ('array[angle[4], 1] phases = {pi / 2}; angle[4] result; float f; '
                      'uint[4] zero = 0; int negative = -1; int bad = 1; ')
            target = 'f' if expression == 'tan(phase)' else 'result'
            body = f'{target} = {expression};'
            for wrapper in ('if (false) {{ {} }}', 'while (false) {{ {} }}',
                            'for int outer in [1:0] {{ {} }}'):
                self.check_initializers(prefix + wrapper.format('for angle[4] phase in phases {' + body + '}'),
                                        (('phases', 1),), (4,), ('result', 'f'), (0, 0))
            result = self.execute(self.program(prefix + 'for angle[4] phase in phases {' + body + '}'))
            self.assertNotEqual(result.returncode, 0, expression)
            self.assertNotIn('runtime error:', result.stderr)
            type(self).checked += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra', type=pathlib.Path)
    arguments = parser.parse_args()
    BackendTests.bra = arguments.bra.resolve() if arguments.bra else None
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(BackendTests))
    print(f'angle-array loop programs checked: {BackendTests.checked}')
    raise SystemExit(not result.wasSuccessful())
