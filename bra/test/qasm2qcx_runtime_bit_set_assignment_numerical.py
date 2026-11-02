#!/usr/bin/env python3
"""Audit runtime bit-register index-set assignment destinations (#444)."""

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

    def test_plain_assignments_preserve_order_and_unselected_bits_at_wide_widths(self):
        for width in (1, 2, 4, 32, 65, 129):
            positions = sorted({0, width // 2, width - 1}, reverse=True)
            declarations, names = [], []
            for entry, position in enumerate(positions):
                kind, value = ('int', position - width) if entry % 2 == 0 else ('uint', position)
                name = f'index{entry}'
                names.append(name)
                declarations.append(f'{kind} {name} = {value};')
            selector = '{' + ','.join(names) + '}'
            count = len(positions)
            for bits in sorted({'0' * width, '1' * width, ('101100' * width)[:width]}):
                for payload in sorted({'0' * count, '1' * count, ('10' * count)[:count]}):
                    before = [int(bit) for bit in reversed(bits)]
                    values = [int(bit) for bit in reversed(payload)]
                    for position, value in zip(positions, values):
                        before[position] = value
                    expected = ''.join(str(bit) for bit in reversed(before))
                    for rhs in (f'"{payload}"', 'source', f'source[0:{count - 1}]',
                                'source[{' + ','.join(str(i) for i in range(count - 1)) +
                                (',' if count > 1 else '') + 'rhs_index}]'):
                        self.check_initializers(f'bit[{width}] flags = "{bits}"; ' + ''.join(declarations) +
                            f'bit[{count}] source = "{payload}"; int rhs_index = {count - 1}; '
                            f'flags[{selector}] = {rhs}; bool same = flags == "{expected}";',
                            (), (), ('same',), (1,))

    def test_overlapping_runtime_and_static_rhs_use_original_bits(self):
        for raw in range(16):
            for positions in ((3, 1, 2, 0), (1, 0, 3, 2), (0, 3), (3, 0)):
                source_positions = tuple(reversed(positions))
                bits = [(raw >> i) & 1 for i in range(4)]
                expected = bits.copy()
                for destination, source in zip(positions, source_positions):
                    expected[destination] = bits[source]
                result = ''.join(str(bit) for bit in reversed(expected))
                target = '{index,' + ','.join(str(i) for i in positions[1:]) + '}'
                source = '{' + ','.join('index' if i == positions[0] else str(i) for i in source_positions) + '}'
                self.check_initializers(f'bit[4] flags = "{raw:04b}"; int index = {positions[0]}; '
                    f'flags[{target}] = flags[{source}]; bool same = flags == "{result}";',
                    (), (), ('same',), (1,))
            reversed_bits = f'{raw:04b}'[::-1]
            self.check_initializers(f'bit[4] flags = "{raw:04b}"; int index = 3; '
                f'flags[{{index, 2, 1, 0}}] = flags; bool same = flags == "{reversed_bits}";',
                (), (), ('same',), (1,))

    def test_rhs_consumers_and_destination_indices_from_original_register(self):
        self.check_initializers('bit[4] flags = "1001"; flags[{0, int(flags[0]) + 1}] = "10"; '
            'bool same = flags == "1100";', (), (), ('same',), (1,))
        self.check_initializers('bit[4] flags = "1010"; int i = 0; uint j = 3; bit[2] pair = "01"; '
            'flags[{i, j}] = rotl(pair ^ ~flags[{j, i}], 1); bool same = flags == "1011";',
            (), (), ('same',), (1,))
        self.check_initializers('const bit[65] C = "1' + '0' * 63 + '1"; bit[4] flags = "0000"; '
            'int i = 0; int high = -1; flags[{i, 3}] = C[{high, i}]; '
            'bool same = flags == "1001";', (), (), ('same',), (1,))

    def test_loops_reentry_shadowing_and_live_source_reads(self):
        self.check_initializers('bit[4] flags = "1011"; int i = 0; uint total = 0; '
            'for bit b in flags { flags[{i}] = "0"; total += uint(b); i += 1; } '
            'bool same = flags == "0000";', (), (), ('total', 'same'), (3, 1))
        self.check_initializers('bit[4] flags = "0001"; int i = 0; uint total = 0; '
            'for bit b in flags[{i, i}] { flags[{i}] = "0"; total += uint(b); } '
            'bool same = flags == "0000";', (), (), ('total', 'same'), (1, 1))
        self.check_initializers('const int LOW = 0; bit[4] flags = "0000"; '
            'for int LOW in [0:3] { flags[{LOW}] = "1"; } int i = 0; '
            'while (i < 4) { flags[{i}] = "0"; i += 1; } bool same = flags == "0000";',
            (), (), ('same',), (1,))

    def test_duplicate_destinations_fail_before_rhs_even_at_wide_positions(self):
        for width in (1, 4, 33, 129):
            for first, second in ((0, 0), (width - 1, -1), (-width, 0)):
                prefix = (f'bit[{width}] flags; int first = {first}; int second = {second}; '
                          'int zero = 0; bit[2] source = "11"; int visits = 0;')
                statement = 'flags[{first, second}] = source[{1 / zero, 0}]; visits += 1;'
                for wrapper in ('if (false) {{ {} }}', 'while (false) {{ {} }}', 'for int outer in [1:0] {{ {} }}'):
                    self.check_initializers(prefix + wrapper.format(statement), (), (), ('visits',), (0,))
                result = self.execute(self.program(prefix + statement))
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('assertion_error', result.stderr)
                self.assertNotIn('division', result.stderr.lower())
                self.assertNotIn('runtime error:', result.stderr)
                type(self).checked += 1

    def test_destination_bounds_arithmetic_and_rhs_failures_only_when_reached(self):
        limits = self.module.QASM2QCXConverter
        for typename, value in (('int', -5), ('int', 4), ('int', limits.QCX_INT_MIN),
                                ('int', limits.QCX_INT_MAX), ('uint', limits.QCX_UINT_MAX)):
            prefix = f'bit[4] flags = "1011"; int i = 0; {typename} bad = {value}; int zero = 0; bit[2] source;'
            statement = 'flags[{i, bad}] = source[{1 / zero, 0}];'
            for wrapper in ('if (false) {{ {} }}', 'while (false) {{ {} }}', 'for int outer in [1:0] {{ {} }}'):
                self.check_initializers(prefix + wrapper.format(statement) + 'bool same = flags == "1011";',
                                        (), (), ('same',), (1,))
            result = self.execute(self.program(prefix + statement))
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn('runtime error:', result.stderr)
            self.assertNotIn('division', result.stderr.lower())
            type(self).checked += 1
        prefix = 'bit[4] flags = "1011"; int i = 0; int zero = 0; int bad = 4; bit[2] source;'
        for statement in ('flags[{i, 1 / zero}] = source;', 'flags[{i, 3}] = flags[{i, bad}];',
                          'flags[{1 / 0, i}] = source;'):
            self.check_initializers(prefix + 'if (false) {' + statement + '} bool same = flags == "1011";',
                                    (), (), ('same',), (1,))
            result = self.execute(self.program(prefix + statement))
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn('runtime error:', result.stderr)
            type(self).checked += 1

    def test_updated_registers_feed_phase_gates_and_fusion(self):
        import cmath
        import math
        from qasm2qcx_u_power_numerical import BackendAuditTests
        source = ('include "stdgates.inc"; qubit q; h q; bit[4] flags = "1010"; int i = 0; int j = 3; '
                  'flags[{i, j}] = flags[{j, i}]; p(angle[4](flags)) q;')
        expected = (1 / math.sqrt(2), cmath.exp(1j * 3 * math.pi / 8) / math.sqrt(2))
        for fused in (False, True):
            BackendAuditTests.check(self, self.program(source, fused=fused), expected)


def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    args = parser.parse_args()
    BackendTests.bra = args.bra.resolve()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(BackendTests))
    print(f'runtime bit-set assignment programs checked: {BackendTests.checked}')
    raise SystemExit(not result.wasSuccessful())


if __name__ == '__main__':
    main()
