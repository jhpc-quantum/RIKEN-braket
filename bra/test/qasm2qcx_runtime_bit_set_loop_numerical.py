#!/usr/bin/env python3
"""Audit runtime bit-register index-set loops (#438) with a non-MPI bra."""

import argparse
import pathlib
import resource
import unittest

import qasm2qcx_angle_array_numerical as arrays


class BackendTests(unittest.TestCase):
    bra = None
    checked = 0
    setUp = arrays.BackendTests.setUp
    program = arrays.BackendTests.program
    execute = arrays.BackendTests.execute
    check_initializers = arrays.BackendTests.check_initializers

    def test_order_repetitions_mixed_types_and_negative_aliases(self):
        for width in (1, 2, 4, 32, 65, 129):
            for bits in sorted({'0' * width, '1' * width, ('101100' * width)[:width]}):
                for position in sorted({0, width // 2, width - 1}):
                    positions = (position, width - 1, position - width, 0, position)
                    expected = 0
                    for index in positions:
                        expected = expected * 2 + int(bits[-1 - (index % width)])
                    self.check_initializers(f'bit[{width}] flags = "{bits}"; int i = {position}; '
                        f'uint j = {width - 1}; int negative = {position - width}; uint checksum = 0; int visits = 0; '
                        'for bit b in flags[{i, j, negative, 0, i}] { '
                        'checksum = checksum * 2 + uint(b); visits += 1; } '
                        f'bool same = flags == "{bits}";', (), (), ('checksum', 'visits', 'same'), (expected, 5, 1))

    def test_positions_are_snapshots_but_bits_are_live_copies(self):
        self.check_initializers('bit[4] flags = "1001"; int i = 0; uint j = 3; uint checksum = 0; int visits = 0; '
            'for bit b in flags[{i, j, i}] { if (visits == 0) { i = 1; j = 2; flags[0] = false; } '
            'checksum = checksum * 2 + uint(b); visits += 1; } bool same = flags == "1000";',
            (), (), ('i', 'j', 'checksum', 'visits', 'same'), (1, 2, 6, 3, 1))
        self.check_initializers('bit[1] flags = "1"; int i = 0; uint checksum = 0; int visits = 0; '
            'for bit b in flags[{i, -1, i}] { flags[0] = false; i = 100; '
            'checksum = checksum * 2 + uint(b); visits += 1; }',
            (), (), ('i', 'checksum', 'visits'), (100, 4, 3))

    def test_constant_sets_at_every_bit_and_word_boundary(self):
        word = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        for width in (1, word - 1, word, word + 1, 2 * word - 1, 2 * word, 2 * word + 1, 129):
            for bits in ('0' * width, '1' * width, ('10010110' * width)[:width]):
                # Each loop mixes signed/unsigned runtime entries, a negative
                # alias, a constant, and a repeat. Inspect every source bit.
                checksum = 0
                for index in range(width):
                    for position in (index, width - 1 - index, index, 0, index):
                        checksum = (checksum * 3 + int(bits[-1 - position])) % 997
                self.check_initializers(f'const bit[{width}] C = "{bits}"; int i = 0; '
                    f'uint j = {width - 1}; uint checksum = 0; int visits = 0; '
                    f'while (i < {width}) {{ for bit b in C[{{i, j, i - {width}, 0, i}}] {{ '
                    'checksum = (checksum * 3 + uint(b)) % 997; visits += 1; } i += 1; '
                    'if (j > 0) { j -= 1; } } '
                    f'bool same = C == "{bits}";', (), (), ('checksum', 'visits', 'same'),
                    (checksum, width * 5, 1))

    def test_constant_shadowing_captured_positions_and_transfers(self):
        for width in (1, 4, 65, 129):
            bits = ('1011' * width)[:width]
            positions = (0, width - 1, 0)
            expected = 0
            for position in positions:
                expected = expected * 2 + int(bits[-1 - position])
            for name in ('C', 'i', 'j'):
                self.check_initializers(f'const bit[{width}] C = "{bits}"; int i = 0; uint j = {width - 1}; '
                    f'uint checksum = 0; for bit {name} in C[{{i, j, i}}] {{ '
                    f'checksum = checksum * 2 + uint({name}); '
                    # Only write the index whose name is not shadowed.
                    + ('i = 100;' if name != 'i' else 'j = 100;') + ' }',
                    (), (), ('checksum',), (expected,))
        self.check_initializers('const bit[4] C = "1011"; int i = 0; int visits = 0; uint total = 0; '
            'for bit b in C[{i, 2, -1, i}] { visits += 1; if (!b) { continue; } '
            'total += uint(b); if (visits == 3) { break; } }',
            (), (), ('visits', 'total'), (3, 2))

    def test_nested_constant_and_mutable_loops_preserve_word_tables_on_reentry(self):
        bits = '1' + '0' * 63 + '1'
        inverse = '0' + '1' * 63 + '0'
        self.check_initializers(f'const bit[65] C = "{bits}"; const bit[65] D = "{inverse}"; '
            'bit[2] live = "11"; int i = 0; int visits = 0; uint total = 0; '
            'while (i < 65) { for bit b in C[{i, 1, i}] { '
            'for bit c in D[{i, 1}] { for bit d in live[{int(c), 0}] { '
            'total += uint(b) + uint(c) + uint(d); visits += 1; } } } i += 64; } '
            'for bit b in D[{1, i - 65}] { total += uint(b); visits += 1; }',
            (), (), ('visits', 'total'), (26, 54))

    def test_constant_invalid_later_indices_fail_before_body_and_only_when_reached(self):
        limits = self.module.QASM2QCXConverter
        for width in (1, limits.QCX_UINT_WIDTH + 1, 129):
            for typename, bad in (('int', -width - 1), ('int', width), ('uint', limits.QCX_UINT_MAX)):
                prefix = (f'const bit[{width}] C = "' + '0' * width + '"; int good = 0; '
                          f'{typename} bad = {bad}; int zero = 0; int visits = 0;')
                for entries in ('good, bad', 'bad, good', 'good, 999', '-999, good', 'good, 1 / zero'):
                    loop = f'for bit b in C[{{{entries}}}] {{ visits = 1 / zero; break; }}'
                    for wrapper in ('if (false) {{ {} }}', 'while (false) {{ {} }}',
                                    'for int outer in [1:0] {{ {} }}'):
                        self.check_initializers(prefix + wrapper.format(loop), (), (), ('visits',), (0,))
                    result = self.execute(self.program(prefix + loop))
                    self.assertNotEqual(result.returncode, 0)
                    self.assertNotIn('runtime error:', result.stderr)
                    if 'zero' not in entries and (typename == 'int' or 'bad' not in entries):
                        self.assertIn('assertion_error', result.stderr)
                    type(self).checked += 1

    def test_header_shadowing_and_nested_array_expressions(self):
        for name in ('flags', 'i', 'j'):
            self.check_initializers('bit[4] flags = "1011"; int i = -1; uint j = 0; uint checksum = 0; '
                f'for bit {name} in flags[{{i, j, i}}] {{ checksum = checksum * 2 + uint({name}); }}',
                (), (), ('i', 'j', 'checksum'), (-1, 0, 7))
        self.check_initializers('bit[4] flags = "1010"; array[int, 2] positions = {-1, 0}; '
            'int i = 0; uint checksum = 0; for bit b in flags[{positions[i], positions[i + 1], i + 1}] { '
            'positions[0] = 0; i = 100; checksum = checksum * 2 + uint(b); }',
            (), (), ('i', 'checksum'), (100, 5))

    def test_runtime_reentry_and_nested_transfers_keep_tables_independent(self):
        self.check_initializers('bit[4] flags = "1011"; int i = 0; uint total = 0; int visits = 0; '
            'while (i < 4) { for bit b in flags[{i, 0}] { '
            'for bit c in flags[{int(b), i}] { total += uint(c); continue; total += 100; } '
            'visits += 1; } i += 1; }', (), (), ('total', 'visits', 'i'), (14, 8, 4))
        self.check_initializers('bit[4] flags = "1011"; int i = 0; uint total = 0; int visits = 0; '
            'for int outer in [0:2] { for bit b in flags[{i, -1, 2}] { '
            'visits += 1; if (!b) { continue; } total += 1; if (visits == 2) { break; } } }',
            (), (), ('total', 'visits'), (6, 8))
        self.check_initializers('bit[4] flags = "1011"; int i = 0; int visits = 0; '
            'for bit b in flags[{i, 3}] { for bit c in flags[{int(b), i}] { break; } visits += 1; }',
            (), (), ('visits',), (2,))

    def test_all_indices_are_checked_before_break_or_body_failure(self):
        limits = self.module.QASM2QCXConverter
        for width in (1, 4, 129):
            for typename, bad in (('int', -width - 1), ('int', width),
                                   ('int', limits.QCX_INT_MIN), ('int', limits.QCX_INT_MAX),
                                   ('uint', limits.QCX_UINT_MAX)):
                prefix = (f'bit[{width}] flags; int good = 0; {typename} bad = {bad}; '
                          'int zero = 0; int visits = 0;')
                for entries in ('good, bad', 'bad, good', 'good, 999', '-999, good'):
                    loop = f'for bit b in flags[{{{entries}}}] {{ visits = 1 / zero; break; }}'
                    for wrapper in ('if (false) {{ {} }}', 'while (false) {{ {} }}',
                                    'for int outer in [1:0] {{ {} }}'):
                        self.check_initializers(prefix + wrapper.format(loop), (), (), ('visits',), (0,))
                    result = self.execute(self.program(prefix + loop))
                    self.assertNotEqual(result.returncode, 0)
                    self.assertNotIn('runtime error:', result.stderr)
                    self.assertNotIn('division', result.stderr.lower())
                    if typename == 'int' or 'bad' not in entries:
                        self.assertIn('assertion_error', result.stderr)
                    type(self).checked += 1

    def test_skipped_runtime_arithmetic_and_nested_index_failures(self):
        prefix = 'bit[4] flags; array[int, 1] positions; int bad = 1; int zero = 0; int visits = 0;'
        for entries in ('zero, 1 / zero', 'positions[bad], zero', 'zero, 1 / 0'):
            loop = f'for bit b in flags[{{{entries}}}] {{ visits += 1; }}'
            for wrapper in ('if (false) {{ {} }}', 'while (false) {{ {} }}',
                            'for int outer in [1:0] {{ {} }}'):
                self.check_initializers(prefix + wrapper.format(loop), (), (), ('visits',), (0,))
            result = self.execute(self.program(prefix + loop))
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn('runtime error:', result.stderr)
            type(self).checked += 1

    def test_measurement_preserves_copy_and_changes_future_visits(self):
        self.check_initializers('include "stdgates.inc"; qubit q; x q; bit[4] flags = "0000"; '
            'int i = 3; uint checksum = 0; for bit b in flags[{i, i}] { '
            'flags[3] = measure q; checksum = checksum * 2 + uint(b); } bool same = flags == "1000";',
            (), (), ('checksum', 'same'), (1, 1))

    def test_gate_conditions_and_fusion_follow_live_selected_bits(self):
        import cmath
        import math
        from qasm2qcx_u_power_numerical import BackendAuditTests
        for gate in ('p(pi/4) q;', 'gphase(pi/4);'):
            source = ('include "stdgates.inc"; qubit q; h q; bit[4] flags = "1001"; int i = 0; '
                      'int visits = 0; for bit b in flags[{i, -1, i}] { '
                      'if (visits == 0) { flags[0] = false; i = 2; } '
                      'if (b) { ' + gate + ' } visits += 1; }')
            phase = cmath.exp(1j * math.pi / 2) / math.sqrt(2)
            expected = (phase if gate.startswith('gphase') else 1 / math.sqrt(2), phase)
            for fused in (False, True):
                BackendAuditTests.check(self, self.program(source, fused=fused), expected)
            source = ('include "stdgates.inc"; qubit q; h q; const bit[65] C = "1' + '0' * 63 + '1"; '
                      'int i = 0; for bit b in C[{i, 1, -1}] { i = 100; if (b) { ' + gate + ' }}')
            for fused in (False, True):
                BackendAuditTests.check(self, self.program(source, fused=fused), expected)


def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    args = parser.parse_args()
    BackendTests.bra = args.bra.resolve()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(BackendTests))
    print(f'runtime bit-set loop programs checked: {BackendTests.checked}')
    raise SystemExit(not result.wasSuccessful())


if __name__ == '__main__':
    main()
