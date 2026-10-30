#!/usr/bin/env python3
"""Audit runtime angle bit-range capture and iteration (#432) with a non-MPI bra."""

import argparse
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
    check_initializers = arrays.BackendTests.check_initializers

    def capture_lines(self, prefix, selection, *, skipped=False, mutate=False):
        program = self.module.openqasm3.parser.parse('OPENQASM 3; ' + prefix)
        converter = self.module.QASM2QCXConverter(program)
        converter.visit(program)
        loop = self.module.openqasm3.parser.parse(
            f'OPENQASM 3; for bit b in {selection} {{}}').statements[0]
        info = converter._QASM2QCXConverter__angle_selection_loop_info(loop.set_declaration, loop.type)
        lines = converter._QASM2QCXConverter__qcx_lines
        previous_temporaries = converter._QASM2QCXConverter__declared_temporary_variables.copy()
        if skipped:
            lines.append('JUMP QASM2QCX_AUDIT_SKIPPED')
        with converter._QASM2QCXConverter__captured_angle_bit_range(info) as (source, positions, count):
            if mutate:
                name = converter._QASM2QCXConverter__capitalize_variable_name
                lines.extend([f'LET {name("element")} := 1', f'LET {name("first")} := 100',
                              f'LET {name("stride")} := 0', f'LET {name("last")} := -100',
                              f'LET {name("phases")}:0 := 9', f'PRINTLN {source}'])
            index = converter._QASM2QCXConverter__add_new_temporary_variable(self.module.ValueType.INT)
            lines.extend([
                f'PRINTLN {count}', f'LET {index} := 0', '@QASM2QCX_AUDIT_PRINT',
                f'JUMPIF QASM2QCX_AUDIT_DONE {index} >= {count}',
                f'PRINTLN {positions}:{index}', f'LET {index} += 1', 'JUMP QASM2QCX_AUDIT_PRINT',
                '@QASM2QCX_AUDIT_DONE',
            ])
        if skipped:
            lines.append('@QASM2QCX_AUDIT_SKIPPED')
        self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())
        converter._QASM2QCXConverter__hoist_temporary_declarations(previous_temporaries)
        return list(converter)

    def check_range(self, width, first, stride, last, kinds=('int', 'int', 'int'), *, default_step=False):
        # Independent mathematical oracle: a monotone progression is in bounds
        # iff its first and last selected positions are. Do not iterate a huge
        # invalid range, nor require the stop itself to be a selected position.
        count = 0
        if stride and (last - first) * stride >= 0:
            count = abs(last - first) // abs(stride) + 1
        final = first + (count - 1) * stride if count else first
        valid = count > 0 and -width <= min(first, final) and max(first, final) < width
        prefix = (f'angle[{width}] a; {kinds[0]} first = {first}; '
                  f'{kinds[1]} stride = {stride}; {kinds[2]} last = {last};')
        selector = 'a[first:last]' if default_step else 'a[first:stride:last]'
        lines = self.capture_lines(prefix, selector)
        result = self.execute(lines)
        self.assertNotIn('runtime error:', result.stderr)
        context = f'width={width}, range={first}:{stride}:{last}, types={kinds}\n{result.stderr}'
        if valid:
            self.assertEqual(result.returncode, 0, context)
            expected = [count] + [(first + index * stride) % width for index in range(count)]
            self.assertEqual(result.stdout.split(), [str(value) for value in expected], context)
        else:
            self.assertNotEqual(result.returncode, 0, context)
            self.assertEqual(result.stdout.strip(), '', context)
        type(self).checked += 1

    def test_signed_ranges_match_mathematical_progression_and_bounds(self):
        for width in (1, 4, self.module.QASM2QCXConverter.QCX_UINT_WIDTH):
            for first in sorted({-width, -1, 0, width - 1}):
                for last in sorted({-width - 1, -width, -1, 0, width - 1, width, width + 1}):
                    for stride in sorted({-2 * width - 1, -width, -2, -1, 0, 1, 2, width, 2 * width + 1}):
                        with self.subTest(width=width, first=first, stride=stride, last=last):
                            self.check_range(width, first, stride, last)

    def test_every_width_preserves_order_aliases_and_default_step(self):
        for width in range(1, self.module.QASM2QCXConverter.QCX_UINT_WIDTH + 1):
            self.check_range(width, -width, 1, width - 1, default_step=True)
            self.check_range(width, width - 1, -1, -width)
            self.check_range(width, 0, width + 1, width)

    def test_native_signed_extremes_never_overflow(self):
        limits = self.module.QASM2QCXConverter
        for width in (1, 4, limits.QCX_UINT_WIDTH):
            for first in sorted({-width, 0, width - 1}):
                for stride in (limits.QCX_INT_MIN, -1, 1, limits.QCX_INT_MAX):
                    for last in (limits.QCX_INT_MIN, limits.QCX_INT_MAX):
                        self.check_range(width, first, stride, last)

    def test_full_width_unsigned_bounds_and_strides_preserve_their_values(self):
        limits = self.module.QASM2QCXConverter
        for first in (-4, 0, 3):
            for stride in (0, 1, 8, limits.QCX_UINT_MAX - 3, limits.QCX_UINT_MAX):
                for last in (0, 3, 7, limits.QCX_INT_MAX, limits.QCX_UINT_MAX):
                    self.check_range(4, first, stride, last, ('int', 'uint', 'uint'))
        for first in (0, 3, limits.QCX_UINT_MAX):
            for stride in (limits.QCX_INT_MIN, -1, 1, limits.QCX_INT_MAX):
                for last in (0, 3, limits.QCX_UINT_MAX):
                    self.check_range(4, first, stride, last, ('uint', 'int', 'uint'))
        for last in (-5, -1, 0, 3, limits.QCX_INT_MAX):
            self.check_range(4, 0, limits.QCX_UINT_MAX, last, ('uint', 'uint', 'int'))

    def test_capture_precedes_bound_mutations_and_preserves_live_element_address(self):
        lines = self.capture_lines('array[angle[4], 2] phases = {pi, pi}; '
            'int element = 0; int first = -4; int stride = 2; int last = 3;',
            'phases[element][first:stride:last]', mutate=True)
        result = self.execute(lines)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn('runtime error:', result.stderr)
        self.assertEqual(result.stdout.split(), ['9', '4', '0', '2', '0', '2'])
        type(self).checked += 1

    def test_invalid_captures_fail_only_when_reached_and_before_consumer(self):
        prefix = ('array[angle[4], 2] phases; angle[4] a; int bad = 2; '
                  'int zero = 0; int first = 3; int stride = 0; int last = 0;')
        for selection in ('a[first:last]', 'a[first:stride:last]', 'a[1 / zero:last]',
                          'a[first:1 / zero:last]', 'a[first:1:1 / zero]',
                          'a[0:1:bad + 3]', 'phases[bad][0:last]',
                          'a[0:stride:999999999999999999999]'):
            for skipped in (False, True):
                result = self.execute(self.capture_lines(prefix, selection, skipped=skipped))
                self.assertNotIn('runtime error:', result.stderr)
                self.assertEqual(result.stdout.strip(), '')
                if skipped:
                    self.assertEqual(result.returncode, 0, selection + '\n' + result.stderr)
                else:
                    self.assertNotEqual(result.returncode, 0, selection)
                type(self).checked += 1

    def test_public_loops_at_every_width_preserve_order_and_repetitions(self):
        limits = self.module.QASM2QCXConverter
        for width in range(1, limits.QCX_UINT_WIDTH + 1):
            raw = 0xAAAAAAAA & ((1 << width) - 1)
            prefix = (f'const angle[{width}] C = angle[{width}]("{raw:0{width}b}"); '
                      f'angle[{width}] a = C; array[angle[{width}], 1] phases = {{C}}; '
                      f'int element = -1; int first = {-width}; int stride = 1; int last = {width - 1}; ')
            source, outputs, expected = prefix, [], []
            for direction in (1, -1):
                positions = list(range(-width, width)) if direction == 1 else list(range(width - 1, -width - 1, -1))
                checksum = 0
                for position in positions:
                    checksum = (checksum * 2 + ((raw >> (position % width)) & 1)) & limits.QCX_UINT_MAX
                for origin, operand in enumerate(('C', 'a', 'phases[element]')):
                    suffix = f'{origin}_{"up" if direction == 1 else "down"}'
                    total, visits = 'total' + suffix, 'visits' + suffix
                    selector = 'first:last' if direction == 1 else 'last:-stride:first'
                    source += (f'uint {total} = 0; int {visits} = 0; '
                               f'for bit b in {operand}[{selector}] {{ '
                               f'{total} = {total} * 2 + uint(b); {visits} += 1; }}')
                    outputs.extend((total, visits))
                    expected.extend((checksum, 2 * width))
            self.check_initializers(source, (('phases', 1),), (raw,),
                                    tuple(outputs) + ('a',), tuple(expected) + (raw,))

    def test_public_loops_keep_full_width_steps_and_unselected_stop_bounds(self):
        limits = self.module.QASM2QCXConverter
        for first, stride, last, kinds, wanted in (
                (0, 3, 4, ('int', 'int', 'int'), (3, 2)),
                (3, -3, -1, ('int', 'int', 'int'), (3, 2)),
                (-1, limits.QCX_UINT_MAX, 3, ('int', 'uint', 'int'), (1, 1)),
                (3, limits.QCX_UINT_MAX, limits.QCX_UINT_MAX, ('int', 'uint', 'uint'), (1, 1)),
                (0, limits.QCX_INT_MIN, -3, ('uint', 'int', 'int'), (1, 1)),
                (3, limits.QCX_INT_MAX, 3, ('uint', 'int', 'uint'), (1, 1))):
            self.check_initializers('angle[4] a = angle[4]("1001"); '
                f'{kinds[0]} first = {first}; {kinds[1]} stride = {stride}; {kinds[2]} last = {last}; '
                'uint total = 0; int visits = 0; for bit b in a[first:stride:last] { '
                'total = total * 2 + uint(b); visits += 1; }',
                (), (), ('total', 'visits'), wanted)

    def test_public_loops_capture_headers_but_read_source_bits_live(self):
        self.check_initializers('array[angle[4], 2] phases = {angle[4]("0011"), pi}; '
            'int element = 0; int first = -4; int stride = 1; int last = 3; '
            'int visits = 0; uint total = 0; bit saved = 0; '
            'for bit b in phases[element][first:stride:last] { if (visits == 0) { '
            'element = 1; first = 100; stride = 0; last = -100; '
            'phases[0][0] = "0"; phases[0][3] = "1"; saved = b; } '
            'visits += 1; total += uint(b); }',
            (('phases', 2),), (10, 8),
            ('element', 'first', 'stride', 'last', 'visits', 'total', 'saved'), (1, 100, 0, -100, 8, 5, 1))

    def test_public_loop_headers_resolve_before_same_name_iterator_binding(self):
        for name in ('a', 'C', 'phases', 'element', 'first', 'stride', 'last'):
            operand = name if name in ('a', 'C') else 'phases[element]'
            self.check_initializers('const angle[4] C = angle[4]("1010"); angle[4] a = C; '
                'array[angle[4], 1] phases = {C}; int element = -1; '
                'int first = 0; int stride = 1; int last = 3; uint total = 0; '
                f'for bit {name} in {operand}[first:stride:last] {{ total += uint({name}); }}',
                (), (), ('total',), (2,))

    def test_public_nested_loops_reentry_and_angle_iterators(self):
        self.check_initializers('array[angle[4], 2] phases = {angle[4]("0011"), angle[4]("1010")}; '
            'int first = 0; int last = 1; int visits = 0; uint total = 0; int passes = 0; '
            'while (passes < 2) { passes += 1; for int element in [0:last] { '
            'for angle[4] a in phases { for bit b in a[first:3] { '
            'for bit c in phases[element][first:2:3] { visits += 1; if (b && c) { total += 1; } }}}}}',
            (), (), ('visits', 'total', 'passes'), (64, 8, 2))
        self.check_initializers('angle[4] a = angle[4]("1011"); int first = 0; int last = 3; '
            'int passes = 0; int visits = 0; uint total = 0; while (passes < 4) { '
            'passes += 1; for bit b in a[first:last] { visits += 1; total += uint(b); } first += 1; }',
            (), (), ('passes', 'visits', 'total'), (4, 10, 7))

    def test_public_break_continue_and_skipped_temporary_initialization(self):
        prefix = ('angle[4] a = angle[4]("1011"); int first = 0; int last = 3; '
                  'int visits = 0; uint total = 0; int passes = 0;')
        self.check_initializers(prefix + 'for bit b in a[first:last] { visits += 1; '
            'if (!b) { continue; } total += uint(b); if (total == 2) { break; }}',
            (), (), ('visits', 'total'), (2, 2))
        for transfer, visits in (('break', 3), ('continue', 12)):
            self.check_initializers(prefix + 'while (passes < 3) { passes += 1; '
                'for bit b in a[first:last] { visits += 1; ' + transfer + '; total += uint(b); }} '
                'total += popcount(a[0:3]);', (), (), ('visits', 'total', 'passes'), (visits, 3, 3))
        self.check_initializers(prefix + 'for bit b in a[first:last] { '
            'for bit c in a[first:last] { if (c) { total += uint(b); break; } } '
            'if (!b) { continue; } visits += 1; }', (), (), ('visits', 'total'), (3, 3))

    def test_public_runtime_checks_execute_before_body_even_with_break(self):
        prefix = ('array[angle[4], 1] phases = {pi}; angle[4] a; int bad = 1; int zero = 0; '
                  'int first = 0; int stride = 1; int last = 4; int visits = 0;')
        for selection in ('a[first:last]', 'a[last:first]', 'a[first:zero:last]',
                          'a[first:1 / zero:last]', 'a[first:stride:1 / zero]',
                          'phases[bad][first:last]', 'a[first:stride:999999999999999999999]'):
            loop = f'for bit b in {selection} {{ visits = 1 / zero; break; }}'
            for wrapper in ('if (false) {{ {} }}', 'while (false) {{ {} }}',
                            'for int outer in [1:0] {{ {} }}'):
                self.check_initializers(prefix + wrapper.format(loop), (), (), ('visits',), (0,))
            result = self.execute(self.program(prefix + loop))
            self.assertNotEqual(result.returncode, 0, selection)
            self.assertNotIn('runtime error:', result.stderr)
            if '1 / zero' not in selection:
                self.assertIn('assertion_error', result.stderr, selection)
            type(self).checked += 1

    def test_public_phase_gates_and_fusion_use_live_values(self):
        import cmath
        import math
        from qasm2qcx_u_power_numerical import BackendAuditTests
        for array in (False, True):
            operand = 'phases[element]' if array else 'a'
            declaration = ('array[angle[4], 1] phases = {angle[4]("0011")}; int element = -1; '
                           if array else 'angle[4] a = angle[4]("0011"); ')
            for gate, factor in (('p', 1), ('inv @ p', -1), ('pow(0.5) @ p', 0.5)):
                # Initial bits 0 and 1 select two visits. Each selected visit
                # applies phases raw 3 then raw 11, toggling bit 3 twice.
                source = ('include "stdgates.inc"; qubit q; h q; int first = 0; int last = 3; '
                          + declaration + f'for bit b in {operand}[first:last] {{ if (b) {{ '
                          + f'{gate}({operand}) q; {operand}[3] ^= "1"; '
                          + f'{gate}({operand}) q; {operand}[3] ^= "1"; ' + '}}')
                phase = 2 * (-2 if factor == 0.5 else 14) * math.pi / 8
                expected = (1 / math.sqrt(2), cmath.exp(1j * factor * phase) / math.sqrt(2))
                for fused in (False, True):
                    BackendAuditTests.check(self, self.program(source, fused=fused), expected)


def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    args = parser.parse_args()
    BackendTests.bra = args.bra.resolve()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(BackendTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print(f'runtime angle bit-range programs checked: {BackendTests.checked}')
    raise SystemExit(not result.wasSuccessful())


if __name__ == '__main__':
    main()
