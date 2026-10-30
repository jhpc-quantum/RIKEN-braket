#!/usr/bin/env python3
"""Audit runtime singleton bit-register selection loops (#436) with a non-MPI bra."""

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

    def check_indices(self, width, raw, *, constant=False):
        bits = f'{raw:0{width}b}'
        source = ('const ' if constant else '') + f'bit[{width}] flags = "{bits}";'
        outputs, expected = [], []
        for position in sorted({0, width // 2, width - 1}):
            for kind, value, typename in (('positive', position, 'int'),
                                           ('negative', position - width, 'int'),
                                           ('unsigned', position, 'uint')):
                suffix = kind + str(position)
                index, result, visits = 'index' + suffix, 'result' + suffix, 'visits' + suffix
                source += (f'{typename} {index} = {value}; bit {result} = 0; int {visits} = 0; '
                           f'for bit b in flags[{{{index}}}] {{ {result} = b; {visits} += 1; }}')
                outputs.extend((result, visits))
                expected.extend((int(bits[-1 - position]), 1))
        source += f'bool same = flags == "{bits}";'
        self.check_initializers(source, (), (), tuple(outputs) + ('same',), tuple(expected) + (1,))

    def test_mutable_signed_unsigned_and_negative_indices_at_wide_widths(self):
        for width in (1, 2, 4, 8, 32, 65, 129):
            for raw in sorted({0, (1 << width) - 1, int(('101100' * width)[:width], 2)}):
                self.check_indices(width, raw)

    def test_constant_signed_unsigned_and_negative_indices_at_word_boundaries(self):
        word = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        for width in (1, word, word + 1, 2 * word, 2 * word + 1, 129):
            for raw in sorted({0, (1 << width) - 1, int(('101100' * width)[:width], 2)}):
                self.check_indices(width, raw, constant=True)

    def test_every_constant_bit_across_native_word_boundaries(self):
        word = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        for width in (1, word - 1, word, word + 1, 2 * word - 1, 2 * word, 2 * word + 1, 129):
            for bits in ('0' * width, '1' * width, ('10010110' * width)[:width]):
                checksum = 0
                for index in range(-width, width):
                    checksum = (checksum * 3 + int(bits[-1 - (index % width)])) % 997
                self.check_initializers(f'const bit[{width}] C = "{bits}"; int index = {-width}; '
                    'uint checksum = 0; int visits = 0; '
                    f'while (index < {width}) {{ for bit b in C[{{index}}] {{ '
                    'checksum = (checksum * 3 + uint(b)) % 997; visits += 1; } index += 1; } '
                    f'bool same = C == "{bits}";', (), (), ('checksum', 'visits', 'same'), (checksum, 2 * width, 1))

    def test_index_and_value_are_captured_before_body_changes(self):
        self.check_initializers('bit[4] flags = "1010"; int index = -1; bit saved = 0; '
            'bit explicit_read = 0; bit second = 1; int visits = 0; '
            'for bit b in flags[{index}] { index = 0; flags[3] = false; saved = b; '
            'explicit_read = flags[index]; visits += 1; } '
            'for bit b in flags[{index}] { second = b; visits += 1; } bool same = flags == "0010";',
            (), (), ('index', 'saved', 'explicit_read', 'second', 'visits', 'same'), (0, 1, 0, 0, 2, 1))
        self.check_initializers('bit[1] flags = "1"; uint index = 0; bit saved = 0; int visits = 0; '
            'for bit b in flags[{index}] { index = 100; flags[0] = false; saved = b; visits += 1; } '
            'bool same = flags == "0";', (), (), ('index', 'saved', 'visits', 'same'), (100, 1, 1, 1))

    def test_header_resolves_source_and_index_before_shadowing(self):
        for name in ('flags', 'index'):
            self.check_initializers('bit[4] flags = "1010"; int index = -1; uint total = 0; '
                f'for bit {name} in flags[{{index}}] {{ total += uint({name}); }} '
                'bool same = flags == "1010";', (), (), ('index', 'total', 'same'), (-1, 1, 1))

    def test_nested_loops_and_runtime_reentry_capture_fresh_indices(self):
        self.check_initializers('bit[4] flags = "1011"; int index = 0; int visits = 0; uint total = 0; '
            'while (index < 4) { for bit b in flags[{index}] { visits += 1; '
            'for bit c in flags[{int(b)}] { total += uint(b && c); } } index += 1; }',
            (), (), ('index', 'visits', 'total'), (4, 4, 3))
        self.check_initializers('bit[4] flags = "1011"; uint total = 0; int visits = 0; '
            'for int index in [0:3] { for bit b in flags[{index}] { '
            'if (!b) { continue; } for bit c in flags[0:1] { total += uint(c); visits += 1; } }}',
            (), (), ('visits', 'total'), (6, 6))
        self.check_initializers('bit[4] flags = "1011"; array[uint[2], 2] positions = {0, 3}; '
            'uint total = 0; int visits = 0; for uint[2] index in positions { '
            'for bit b in flags[{index}] { total += uint(b); visits += 1; }}',
            (), (), ('visits', 'total'), (2, 2))

    def test_transfers_exit_only_the_singleton_loop(self):
        prefix = 'bit[4] flags = "1011"; int index = 0; int passes = 0; int visits = 0; uint total = 0; int zero = 0;'
        for transfer in ('break', 'continue'):
            self.check_initializers(prefix + 'while (passes < 3) { passes += 1; '
                'for bit b in flags[{index}] { visits += 1; ' + transfer + '; total += 1 / zero; } '
                'total += 1; index += 1; }', (), (), ('passes', 'visits', 'total', 'index'), (3, 3, 3, 3))
        self.check_initializers(prefix + 'while (index < 4) { for bit b in flags[{index}] { '
            'for bit c in flags[{index}] { break; } visits += 1; total += uint(b); '
            'continue; total += 100; } index += 1; }', (), (), ('index', 'visits', 'total'), (4, 4, 3))

    def test_nested_index_expressions_and_measurements_preserve_the_copy(self):
        self.check_initializers('bit[4] flags = "1010"; array[int, 2] positions = {-1, 0}; '
            'int index = 0; uint total = 0; int visits = 0; '
            'for bit b in flags[{positions[index % 2]}] { index += 1; total += uint(b); visits += 1; } '
            'for bit b in flags[{int(flags[index]) + 2}] { total += uint(b); visits += 1; }',
            (), (), ('index', 'total', 'visits'), (1, 2, 2))
        self.check_initializers('include "stdgates.inc"; qubit q; x q; bit[4] flags = "0000"; '
            'int index = 3; bit saved = 1; int visits = 0; for bit b in flags[{index}] { '
            'flags[3] = measure q; saved = b; visits += 1; } bool same = flags == "1000";',
            (), (), ('saved', 'visits', 'same'), (0, 1, 1))

    def test_constant_headers_shadowing_and_nested_word_table_reuse(self):
        for name in ('C', 'index'):
            self.check_initializers('const bit[65] C = "1' + '0' * 64 + '"; int index = -1; uint total = 0; '
                f'for bit {name} in C[{{index}}] {{ total += uint({name}); }} '
                'bool same = C == "1' + '0' * 64 + '";', (), (), ('index', 'total', 'same'), (-1, 1, 1))
        self.check_initializers('const bit[65] C = "1' + '0' * 63 + '1"; '
            'const bit[65] D = "0' + '1' * 63 + '0"; bit[1] live = "1"; '
            'uint checksum = 0; int index = 0; int visits = 0; while (index < 65) { '
            'for bit b in C[{index}] { for bit c in D[{index}] { '
            'for bit d in live[{int(c)}] { checksum += uint(b || c) + uint(d); visits += 1; } } } '
            'index += 64; }', (), (), ('checksum', 'visits'), (4, 2))

    def test_constant_transfers_and_changed_index_preserve_the_copy(self):
        prefix = 'const bit[4] C = "1011"; int index = 0; int visits = 0; uint total = 0; int zero = 0;'
        for transfer in ('break', 'continue'):
            self.check_initializers(prefix + 'while (index < 4) { for bit b in C[{index}] { '
                'visits += 1; total += uint(b); ' + transfer + '; total += 1 / zero; } index += 1; }',
                (), (), ('index', 'visits', 'total'), (4, 4, 3))
        self.check_initializers('const bit[129] C = "1' + '0' * 128 + '"; int index = -1; bit saved = 0; '
            'for bit b in C[{index}] { index = 0; saved = b; }', (), (), ('index', 'saved'), (0, 1))

    def test_invalid_constant_indices_and_skipped_arithmetic(self):
        limits = self.module.QASM2QCXConverter
        for width in (1, limits.QCX_UINT_WIDTH + 1, 129):
            prefix = f'const bit[{width}] C = "' + '0' * width + '"; int zero = 0; int visits = 0;'
            for typename, index in (('int', -width - 1), ('int', width), ('int', limits.QCX_INT_MIN),
                                    ('int', limits.QCX_INT_MAX), ('uint', limits.QCX_UINT_MAX)):
                declaration = f'{typename} index = {index};'
                loop = 'for bit b in C[{index}] { visits = 1 / zero; break; }'
                for wrapper in ('if (false) {{ {} }}', 'while (false) {{ {} }}',
                                'for int outer in [1:0] {{ {} }}'):
                    self.check_initializers(prefix + declaration + wrapper.format(loop), (), (), ('visits',), (0,))
                result = self.execute(self.program(prefix + declaration + loop))
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn('runtime error:', result.stderr)
                if typename == 'int':
                    self.assertIn('assertion_error', result.stderr)
                type(self).checked += 1
        prefix = 'const bit[4] C = "1011"; int zero = 0; int visits = 0;'
        loop = 'for bit b in C[{1 / zero}] { visits += 1; }'
        self.check_initializers(prefix + 'if (false) {' + loop + '}', (), (), ('visits',), (0,))
        result = self.execute(self.program(prefix + loop))
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn('runtime error:', result.stderr)
        type(self).checked += 1

    def test_invalid_indices_fail_before_body_and_only_when_reached(self):
        limits = self.module.QASM2QCXConverter
        for width in (1, 4, 129):
            for typename, index in (('int', -width - 1), ('int', width),
                                    ('int', limits.QCX_INT_MIN), ('int', limits.QCX_INT_MAX),
                                    ('uint', limits.QCX_UINT_MAX)):
                prefix = (f'bit[{width}] flags = "' + '0' * width + f'"; {typename} index = {index}; '
                          'int zero = 0; int visits = 0;')
                loop = 'for bit b in flags[{index}] { visits = 1 / zero; break; }'
                for wrapper in ('if (false) {{ {} }}', 'while (false) {{ {} }}',
                                'for int outer in [1:0] {{ {} }}'):
                    self.check_initializers(prefix + wrapper.format(loop), (), (), ('visits',), (0,))
                result = self.execute(self.program(prefix + loop))
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn('runtime error:', result.stderr)
                if typename == 'int':
                    self.assertIn('assertion_error', result.stderr)
                type(self).checked += 1
        prefix = 'bit[4] flags = "0000"; array[int, 1] positions; int zero = 0; int bad = 1; int visits = 0;'
        for index in ('1 / zero', 'positions[bad]'):
            loop = f'for bit b in flags[{{{index}}}] {{ visits += 1; }}'
            for wrapper in ('if (false) {{ {} }}', 'while (false) {{ {} }}',
                            'for int outer in [1:0] {{ {} }}'):
                self.check_initializers(prefix + wrapper.format(loop), (), (), ('visits',), (0,))
            result = self.execute(self.program(prefix + loop))
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn('runtime error:', result.stderr)
            type(self).checked += 1

    def test_phase_gates_modifiers_and_fusion_use_the_captured_bit(self):
        import cmath
        import math
        from qasm2qcx_u_power_numerical import BackendAuditTests
        gates = (
            ('p(pi/4) q[0];', 1, lambda i: bool(i & 1)),
            ('inv @ p(pi/4) q[0];', -1, lambda i: bool(i & 1)),
            ('pow(0.5) @ p(pi/4) q[0];', 0.5, lambda i: bool(i & 1)),
            ('negctrl @ p(pi/4) q[0], q[1];', 1, lambda i: i == 2),
            ('gphase(pi/4);', 1, lambda i: True),
        )
        for constant in (False, True):
            operand = 'C' if constant else 'flags'
            for gate, factor, active in gates:
                source = ('include "stdgates.inc"; qubit[2] q; h q; bit[4] flags = "1010"; '
                          'const bit[65] C = "1' + '0' * 64 + '"; int index = -1; '
                          f'for bit b in {operand}[{{index}}] {{ index = 0; flags[3] = false; if (b) {{ ' + gate + '}}')
                expected = tuple(cmath.exp(1j * factor * math.pi / 4) / 2 if active(i) else 0.5 for i in range(4))
                for fused in (False, True):
                    BackendAuditTests.check(self, self.program(source, fused=fused), expected)


def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    args = parser.parse_args()
    BackendTests.bra = args.bra.resolve()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(BackendTests))
    print(f'runtime singleton bit-loop programs checked: {BackendTests.checked}')
    raise SystemExit(not result.wasSuccessful())


if __name__ == '__main__':
    main()
