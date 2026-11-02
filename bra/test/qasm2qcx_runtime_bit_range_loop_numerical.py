#!/usr/bin/env python3
"""Audit runtime bit-register range loops (#440) with a non-MPI bra."""

import argparse
import pathlib
import re
import resource
import unittest
from unittest.mock import patch

import qasm2qcx_angle_array_numerical as arrays


class BackendTests(unittest.TestCase):
    bra = None
    checked = 0
    setUp = arrays.BackendTests.setUp
    program = arrays.BackendTests.program
    execute = arrays.BackendTests.execute
    check_initializers = arrays.BackendTests.check_initializers

    def check_range(self, width, first, stride, last, kinds=('int', 'int', 'int'), *,
                    default=False, constant=False, bits=None):
        # Exact mathematical endpoints avoid iterating a huge invalid range.
        count = abs(last - first) // abs(stride) + 1 if stride and (last - first) * stride >= 0 else 0
        final = first + (count - 1) * stride if count else first
        valid = count > 0 and -width <= min(first, final) and max(first, final) < width
        bits = ('10110010' * width)[:width] if bits is None else bits
        prefix = ('const ' if constant else '') + (f'bit[{width}] flags = "{bits}"; {kinds[0]} first = {first}; '
                  f'{kinds[1]} stride = {stride}; {kinds[2]} last = {last}; '
                  'int visits = 0; uint checksum = 0; int zero = 0;')
        selector = 'flags[first:last]' if default else 'flags[first:stride:last]'
        if valid:
            checksum = 0
            for index in range(count):
                position = (first + index * stride) % width
                checksum = (checksum * 3 + int(bits[-1 - position])) % 997
            self.check_initializers(prefix + f'for bit b in {selector} {{ '
                'checksum = (checksum * 3 + uint(b)) % 997; visits += 1; } '
                f'bool same = flags == "{bits}";', (), (), ('checksum', 'visits', 'same'), (checksum, count, 1))
        else:
            result = self.execute(self.program(prefix + f'for bit b in {selector} {{ visits = 1 / zero; break; }}'))
            context = f'{width=}, {first=}, {stride=}, {last=}, {kinds=}\n{result.stderr}'
            self.assertNotEqual(result.returncode, 0, context)
            self.assertIn('assertion_error', result.stderr, context)
            self.assertNotIn('runtime error:', result.stderr, context)
            self.assertNotIn('division', result.stderr.lower(), context)
            type(self).checked += 1

    def test_signed_progressions_negative_aliases_and_unselected_stops(self):
        for width in (1, 4, 32, 65, 129):
            for first, stride, last in ((-width, 1, width - 1), (width - 1, -1, -width),
                                        (0, width + 1, width), (width - 1, -width - 1, -width),
                                        (-width, width, width - 1), (0, 2, width - 1),
                                        (0, 1, width), (-width - 1, 1, 0), (0, 0, 0),
                                        (1, 1, 0), (0, -1, 1), (0, 1, 0)):
                self.check_range(width, first, stride, last)
            self.check_range(width, -width, 1, width - 1, default=True)

    def test_mixed_native_integer_endpoints_and_full_width_steps_do_not_wrap(self):
        limits = self.module.QASM2QCXConverter
        for width in (1, 4, 65, 129):
            for first in (-width, 0, width - 1):
                for stride in (limits.QCX_INT_MIN, limits.QCX_INT_MAX):
                    for last in (limits.QCX_INT_MIN, limits.QCX_INT_MAX):
                        self.check_range(width, first, stride, last)
                for last in (limits.QCX_UINT_MAX - 1, limits.QCX_UINT_MAX):
                    self.check_range(width, first, limits.QCX_UINT_MAX, last, ('int', 'uint', 'uint'))
            self.check_range(width, 0, limits.QCX_UINT_MAX, 0, ('uint', 'uint', 'uint'))
            self.check_range(width, limits.QCX_UINT_MAX, -1, 0, ('uint', 'int', 'int'))

    def test_capture_precedes_mutation_and_repeated_positions_read_live(self):
        self.check_initializers('bit[4] flags = "1001"; int first = -4; int stride = 3; int last = 3; '
            'int visits = 0; uint checksum = 0; for bit b in flags[first:stride:last] { '
            'if (visits == 0) { first = 999; stride = 0; last = -999; flags[0] = false; flags[3] = false; } '
            'checksum = checksum * 2 + uint(b); visits += 1; } bool same = flags == "0000";',
            (), (), ('checksum', 'visits', 'first', 'stride', 'last', 'same'), (4, 3, 999, 0, -999, 1))
        self.check_initializers('bit[1] flags = "1"; int first = -1; int last = 0; uint checksum = 0; '
            'for bit b in flags[first:last] { flags[0] = false; checksum = checksum * 2 + uint(b); }',
            (), (), ('checksum',), (2,))

    def test_constant_ranges_read_every_bit_across_full_and_partial_words(self):
        word = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        for width in (1, word - 1, word, word + 1, 2 * word - 1, 2 * word, 2 * word + 1, 129):
            for bits in ('0' * width, '1' * width, ('10010110' * width)[:width]):
                for first, stride, last in ((-width, 1, width - 1), (width - 1, -1, -width),
                                            (0, 3, width - 1), (0, width + 1, width)):
                    self.check_range(width, first, stride, last, constant=True, bits=bits)
                self.check_range(width, -width, 1, -1, default=True, constant=True, bits=bits)

    def test_constant_native_endpoints_full_width_steps_and_invalid_later_positions(self):
        limits = self.module.QASM2QCXConverter
        for width in (1, 33, 129):
            for first, stride, last in ((0, limits.QCX_INT_MIN, limits.QCX_INT_MIN + 1),
                                        (0, limits.QCX_INT_MIN, limits.QCX_INT_MIN),
                                        (0, limits.QCX_INT_MAX, limits.QCX_INT_MAX - 1),
                                        (0, limits.QCX_INT_MAX, limits.QCX_INT_MAX),
                                        (0, 1, width), (0, 0, 0), (1, 1, 0), (-width - 1, 1, -1)):
                self.check_range(width, first, stride, last, constant=True)
            for last in (limits.QCX_UINT_MAX - 1, limits.QCX_UINT_MAX):
                self.check_range(width, 0, limits.QCX_UINT_MAX, last, ('uint', 'uint', 'uint'), constant=True)

    def test_constant_capture_shadowing_transfers_and_mixed_nested_reentry(self):
        for name in ('C', 'first', 'last'):
            self.check_initializers('const bit[65] C = "1' + '0' * 63 + '1"; '
                'int first = -65; int last = 64; int visits = 0; uint total = 0; '
                f'for bit {name} in C[first:64:last] {{ visits += 1; total += uint({name}); '
                + ('first = 999;' if name != 'first' else 'last = -999;') + ' }',
                (), (), ('total', 'visits'), (2, 3))
        self.check_initializers('const bit[4] C = "1011"; int first = 0; int last = 3; uint total = 0; int visits = 0; '
            'for bit b in C[first:last] { visits += 1; if (!b) { continue; } total += 1; '
            'if (visits == 2) { break; } }', (), (), ('total', 'visits'), (2, 2))
        self.check_initializers('const bit[65] C = "1' + '0' * 63 + '1"; '
            'const bit[65] D = "0' + '1' * 63 + '0"; bit[1] live = "1"; '
            'int i = 0; int visits = 0; uint total = 0; while (i < 65) { '
            'for bit b in C[i:i] { for bit c in D[0:int(b)] { '
            'for bit d in live[0:i - i] { total += uint(b) + uint(c) + uint(d); visits += 1; } } } i += 64; }',
            (), (), ('total', 'visits'), (10, 4))

    def test_constant_skipped_domains_arithmetic_and_capacity_checks(self):
        prefix = ('const bit[65] C = "' + '0' * 65 + '"; array[int, 1] positions; int bad = 1; '
                  'int first = 0; int stride = 0; int last = 65; int zero = 0; int visits = 0;')
        for selector in ('C[first:stride:last]', 'C[first:last]', 'C[first:1 / zero:last]',
                         'C[positions[bad]:last]', 'C[first:1 / 0:last]', 'C[999:first:0]'):
            loop = f'for bit b in {selector} {{ visits = 1 / zero; break; }}'
            for wrapper in ('if (false) {{ {} }}', 'while (false) {{ {} }}', 'for int outer in [1:0] {{ {} }}'):
                self.check_initializers(prefix + wrapper.format(loop), (), (), ('visits',), (0,))
            result = self.execute(self.program(prefix + loop))
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn('runtime error:', result.stderr)
            type(self).checked += 1
        with patch.object(self.module.QASM2QCXConverter, 'MAX_RUNTIME_BIT_RANGE_POSITIONS', 2):
            self.check_initializers(prefix + 'for bit b in C[first:1] { visits += 1; }', (), (), ('visits',), (2,))
            loop = 'for bit b in C[first:2] { visits = 1 / zero; break; }'
            self.check_initializers(prefix + 'if (false) {' + loop + '}', (), (), ('visits',), (0,))
            result = self.execute(self.program(prefix + loop))
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('assertion_error', result.stderr)
            self.assertNotIn('division', result.stderr.lower())
            self.assertNotIn('runtime error:', result.stderr)
            type(self).checked += 1

    def test_header_shadowing_nested_expressions_and_runtime_reentry(self):
        for name in ('flags', 'first', 'stride', 'last'):
            self.check_initializers('bit[4] flags = "1011"; int first = 0; int stride = 1; int last = 3; uint total = 0; '
                f'for bit {name} in flags[first:stride:last] {{ total += uint({name}); }}',
                (), (), ('total',), (3,))
        self.check_initializers('bit[4] flags = "1011"; array[int, 2] bounds = {0, 3}; int i = 0; uint total = 0; '
            'for bit b in flags[bounds[i]:bounds[i + 1]] { bounds[0] = 100; i = 100; total += uint(b); }',
            (), (), ('total', 'i'), (3, 100))
        self.check_initializers('bit[4] flags = "1011"; int first = 0; int visits = 0; uint total = 0; '
            'while (first < 4) { for bit b in flags[first:first] { total += uint(b); visits += 1; } first += 1; }',
            (), (), ('total', 'visits'), (3, 4))

    def test_nested_tables_and_nearest_loop_transfers(self):
        self.check_initializers('bit[4] flags = "1011"; int first = 0; int last = 3; int visits = 0; uint total = 0; '
            'for bit b in flags[first:last] { for bit c in flags[0:int(b)] { '
            'if (c) { total += 1; continue; } break; } visits += 1; }',
            (), (), ('total', 'visits'), (7, 4))
        self.check_initializers('bit[4] flags = "1011"; int first = 0; int last = 3; int visits = 0; uint total = 0; '
            'for int outer in [0:2] { for bit b in flags[first:last] { visits += 1; '
            'if (!b) { continue; } total += 1; if (visits == 2) { break; } } }',
            (), (), ('total', 'visits'), (8, 10))

    def test_skipped_domains_bounds_and_arithmetic_checks(self):
        prefix = ('bit[4] flags; array[int, 1] positions; int bad = 1; int first = 0; '
                  'int stride = 0; int last = 4; int zero = 0; int visits = 0;')
        for selector in ('flags[first:stride:last]', 'flags[first:last]', 'flags[first:1 / zero:last]',
                         'flags[positions[bad]:last]', 'flags[first:1 / 0:last]', 'flags[999:first:0]'):
            loop = f'for bit b in {selector} {{ visits = 1 / zero; break; }}'
            for wrapper in ('if (false) {{ {} }}', 'while (false) {{ {} }}', 'for int outer in [1:0] {{ {} }}'):
                self.check_initializers(prefix + wrapper.format(loop), (), (), ('visits',), (0,))
            result = self.execute(self.program(prefix + loop))
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn('runtime error:', result.stderr)
            type(self).checked += 1

    def test_position_capacity_limit_is_checked_before_body_and_only_when_reached(self):
        prefix = 'bit[4] flags = "1011"; int first = 0; int last = 2; uint total = 0; int zero = 0;'
        with patch.object(self.module.QASM2QCXConverter, 'MAX_RUNTIME_BIT_RANGE_POSITIONS', 2):
            self.check_initializers(prefix + 'for bit b in flags[first:1] { total += uint(b); }',
                                    (), (), ('total',), (2,))
            loop = 'for bit b in flags[first:last] { total += 1 / zero; break; }'
            self.check_initializers(prefix + 'if (false) {' + loop + '}', (), (), ('total',), (0,))
            result = self.execute(self.program(prefix + loop))
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('assertion_error', result.stderr)
            self.assertNotIn('runtime error:', result.stderr)
            self.assertNotIn('division', result.stderr.lower())
            type(self).checked += 1

    def test_documented_runtime_bit_range_loop_example(self):
        import cmath
        import math
        from qasm2qcx_u_power_numerical import BackendAuditTests
        document = (pathlib.Path(__file__).parents[1] / 'qcx/qasm2qcx.md').read_text(encoding='utf-8')
        examples = re.findall(r'<!-- runtime-bit-range-loop-example -->\s*```qasm\n(.*?)\n```',
                              document, flags=re.DOTALL)
        self.assertEqual(len(examples), 1)
        source = examples[0].removeprefix('OPENQASM 3.0;')
        source += '\nbool same = flags == "1010"; bool unchanged = C == "1' + '0' * 63 + '1";'
        self.check_initializers(source, (), (),
            ('start', 'step', 'stop', 'visits', 'ones', 'saved', 'first', 'constant_ones', 'constant_visits', 'same', 'unchanged'),
            (999, 0, -999, 8, 5, 1, 0, 2, 3, 1, 1))
        expected = (1 / math.sqrt(2), cmath.exp(1j * math.pi / 4) / math.sqrt(2))
        for fused in (False, True):
            BackendAuditTests.check(self, self.program(examples[0], fused=fused), expected)

    def test_measurement_live_reads_and_fused_phase_gates(self):
        self.check_initializers('include "stdgates.inc"; qubit q; x q; bit[1] flags = "0"; '
            'int first = -1; int last = 0; uint checksum = 0; for bit b in flags[first:last] { '
            'flags[0] = measure q; checksum = checksum * 2 + uint(b); }', (), (), ('checksum',), (1,))
        import cmath
        import math
        from qasm2qcx_u_power_numerical import BackendAuditTests
        for gate in ('p(pi / 4) q;', 'gphase(pi / 4);'):
            source = ('include "stdgates.inc"; qubit q; h q; bit[4] flags = "1001"; '
                      'int first = 0; int last = 3; for bit b in flags[first:last] { if (b) { ' + gate + ' }}')
            phase = cmath.exp(1j * math.pi / 2) / math.sqrt(2)
            expected = (phase if gate.startswith('gphase') else 1 / math.sqrt(2), phase)
            for fused in (False, True):
                BackendAuditTests.check(self, self.program(source, fused=fused), expected)
            source = ('include "stdgates.inc"; qubit q; h q; const bit[65] C = "1' + '0' * 63 + '1"; '
                      'int first = 0; int last = 64; for bit b in C[first:last] { '
                      'first = 999; last = -999; if (b) { ' + gate + ' }}')
            for fused in (False, True):
                BackendAuditTests.check(self, self.program(source, fused=fused), expected)


def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    args = parser.parse_args()
    BackendTests.bra = args.bra.resolve()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(BackendTests))
    print(f'runtime bit-range loop programs checked: {BackendTests.checked}')
    raise SystemExit(not result.wasSuccessful())


if __name__ == '__main__':
    main()
