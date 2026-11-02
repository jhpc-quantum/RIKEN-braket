#!/usr/bin/env python3
"""Audit runtime bit-register range loops (#440) with a non-MPI bra."""

import argparse
import pathlib
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

    def check_range(self, width, first, stride, last, kinds=('int', 'int', 'int'), *, default=False):
        # Exact mathematical endpoints avoid iterating a huge invalid range.
        count = abs(last - first) // abs(stride) + 1 if stride and (last - first) * stride >= 0 else 0
        final = first + (count - 1) * stride if count else first
        valid = count > 0 and -width <= min(first, final) and max(first, final) < width
        bits = ('10110010' * width)[:width]
        prefix = (f'bit[{width}] flags = "{bits}"; {kinds[0]} first = {first}; '
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
