#!/usr/bin/env python3
"""Audit whole/static bitwise assignments with runtime range RHS values (#464)."""

import argparse
import operator
import pathlib
import re
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
    operations = (('&=', operator.and_), ('|=', operator.or_), ('^=', operator.xor))

    def check_assignment(self, source_bits, original, positions, selector, raw, constant, operation):
        symbol, function = operation
        expected = original.copy()
        for destination, index in zip(positions, raw):
            expected[destination] = function(original[destination], source_bits[index % len(source_bits)])
        source_text = ''.join(map(str, reversed(source_bits)))
        before = ''.join(map(str, reversed(original)))
        after = ''.join(map(str, reversed(expected)))
        step = raw[1] - raw[0] if len(raw) > 1 else 1
        declarations = ''.join(f'{"uint" if value >= 0 else "int"} {name} = {value}; '
                               for name, value in (('start', raw[0]), ('step', step), ('stop', raw[-1])))
        self.check_initializers(f'{"const " if constant else ""}bit[{len(source_bits)}] source = "{source_text}"; '
            f'bit[{len(original)}] result = "{before}"; ' + declarations +
            f'result{selector} {symbol} source[start:step:stop]; '
            f'bool same = result == "{after}"; bool unchanged = source == "{source_text}";',
            (), (), ('same', 'unchanged'), (1, 1))

    def test_whole_static_ranges_and_sets(self):
        for width in (1, 2, 4, 33, 65, 129):
            source = [int(i % 3 == 0) for i in range(width)]
            for raw in (list(range(width)), list(range(-1, -width - 1, -1)),
                        list(range(0, width, 2)), [-1, width - 1]):
                count = len(raw)
                positions = tuple(range(2 * count, 0, -2))
                targets = ((list(range(count)), '', count),
                           (positions, f'[{2 * count}:-2:2]', 2 * count + 1),
                           (positions[1:] + positions[:1],
                            '[{' + ','.join(map(str, positions[1:] + positions[:1])) + '}]', 2 * count + 1))
                for selected, selector, target_width in targets:
                    original = [i % 2 for i in range(target_width)]
                    for constant in (False, True):
                        for operation in self.operations:
                            self.check_assignment(source, original, selected, selector, raw, constant, operation)

    def test_self_overlap_uses_original_values(self):
        for pattern in range(16):
            bits = [(pattern >> i) & 1 for i in range(4)]
            for raw, positions, selector in (([3, 2, 1, 0], (0, 1, 2, 3), ''),
                                              ([0, 1, 2], (3, 0, 1), '[{3,0,1}]'),
                                              ([-1, 3], (3, 0), '[3:-3:0]')):
                step = raw[1] - raw[0]
                for symbol, function in self.operations:
                    expected = bits.copy()
                    for destination, index in zip(positions, raw):
                        expected[destination] = function(bits[destination], bits[index % 4])
                    after = ''.join(map(str, reversed(expected)))
                    self.check_initializers(f'bit[4] flags = "{pattern:04b}"; '
                        f'int start = {raw[0]}; int step = {step}; int stop = {raw[-1]}; '
                        f'flags{selector} {symbol} flags[start:step:stop]; bool same = flags == "{after}";',
                        (), (), ('same',), (1,))

    def test_singletons_default_steps_and_native_extremes(self):
        limits = self.module.QASM2QCXConverter
        for start, step, stop, kind in ((0, limits.QCX_UINT_MAX, limits.QCX_UINT_MAX - 1, 'uint'),
                                         (0, limits.QCX_INT_MIN, -4, 'int'),
                                         (-1, limits.QCX_INT_MAX, 3, 'int')):
            for symbol, function in self.operations:
                value = function(1, int(start == -1))
                self.check_initializers('const bit[4] C = "1000"; bit[1] one = "1"; '
                    f'int start = {start}; {kind} step = {step}; {kind} stop = {stop}; '
                    f'one[{{-1}}] {symbol} C[start:step:stop]; bool same = one == "{value}";',
                    (), (), ('same',), (1,))
        for constant in (False, True):
            for symbol, function in self.operations:
                for target in ('one', 'one[0:0]', 'one[{-1}]'):
                    self.check_initializers(f'{"const " if constant else ""}bit[1] source = "1"; '
                        'bit[1] one = "1"; int index = -1; '
                        f'{target} {symbol} source[index:index]; bool same = one == "{function(1, 1)}";',
                        (), (), ('same',), (1,))

    def test_loop_families_refresh_original_and_rhs_snapshots(self):
        loops = (('for int i in [0:3]', 'i', (0, 1, 2, 3)),
                 ('for int i in [first:last]', 'i', (0, 1, 2, 3)),
                 ('for int i in [ufirst:ulast]', 'i', (0, 1, 2, 3)),
                 ('for int i in [first:stride:last]', 'i', (0, 1, 2, 3)),
                 ('for int i in {0, 1, 2, 3}', 'i', (0, 1, 2, 3)),
                 ('for int i in {first, first + 1, last - 1, last}', 'i', (0, 1, 2, 3)),
                 ('for int i in indices', 'i', (0, 1, 2, 3)),
                 ('for uint i in unsigned_indices', 'i', (0, 1, 2, 3)),
                 ('for bit i in flags', 'int(i)', (0, 1, 0, 1)),
                 ('for bit i in C', 'int(i)', (0, 1, 0, 1)))
        source_bits = [0, 1, 0, 0, 1, 1, 0, 1]
        for loop, index, visits in loops:
            for constant in (False, True):
                for selector, positions, before in (('', tuple(range(4)), '1100'),
                                                     ('[7:-2:1]', (7, 5, 3, 1), '11001100'),
                                                     ('[{7,1,5,3}]', (7, 1, 5, 3), '11001100')):
                    for symbol, function in self.operations:
                        expected = list(map(int, reversed(before)))
                        total = 0
                        for start in visits:
                            for destination, bit in zip(positions, source_bits[start:start + 4]):
                                expected[destination] = function(expected[destination], bit)
                            total += expected[positions[0]]
                        after = ''.join(map(str, reversed(expected)))
                        self.check_initializers(('const ' if constant else '') + 'bit[8] source = "10110010"; '
                            f'bit[{len(before)}] result = "{before}"; bit[4] flags = "1010"; const bit[4] C = "1010"; '
                            'array[int,4] indices = {0,1,2,3}; array[uint,4] unsigned_indices = {0,1,2,3}; '
                            'int first = 0; int last = 3; uint ufirst = 0; uint ulast = 3; int stride = 1; '
                            'int i = -9; int total = 0; ' + loop +
                            f' {{ result{selector} {symbol} source[{index}:{index} + 3]; total += int(result[{positions[0]}]); }} '
                            f'bool same = result == "{after}"; bool unchanged = source == "10110010";',
                            (), (), ('same', 'unchanged', 'total', 'i'), (1, 1, total, -9))

    def test_while_reentry_changes_source_direction_and_repeated_positions(self):
        for constant in (False, True):
            for symbol, function in self.operations:
                for selector, positions in (('[5:-2:1]', (5, 3, 1)), ('[{1,7,3}]', (1, 7, 3))):
                    source_bits = [0, 1, 0, 1]
                    expected = list(map(int, reversed('11001100')))
                    for raw in ((0, 1, 2), (3, 2, 1), (-1, 1, 3)):
                        for destination, index in zip(positions, raw):
                            expected[destination] = function(expected[destination], source_bits[index % 4])
                        if not constant:
                            source_bits = [bit ^ 1 for bit in source_bits]
                    after = ''.join(map(str, reversed(expected)))
                    self.check_initializers(('const ' if constant else '') + 'bit[4] source = "1010"; '
                        'bit[8] result = "11001100"; int i = 0; int start = 0; int step = 1; int stop = 2; '
                        f'while (i < 3) {{ result{selector} {symbol} source[start:step:stop]; '
                        + ('' if constant else 'source ^= "1111"; ') +
                        'if (i == 0) { start = 3; step = -1; stop = 1; } '
                        'else { start = -1; step = 2; stop = 3; } i += 1; } '
                        f'bool same = result == "{after}";', (), (), ('same', 'i'), (1, 3))

    def test_reached_branches_refresh_overlapping_self_references(self):
        for pattern in range(16):
            for symbol, function in self.operations:
                expected = [(pattern >> i) & 1 for i in range(4)]
                for iteration in range(4):
                    original = expected.copy()
                    positions = (3, 2, 1, 0) if iteration % 2 == 0 else (0, 1, 2, 3)
                    operation = function if iteration % 2 == 0 else operator.xor
                    expected = [operation(a, original[b]) for a, b in zip(original, positions)]
                after = ''.join(map(str, reversed(expected)))
                self.check_initializers(f'bit[4] flags = "{pattern:04b}"; '
                    'int start = 3; int step = -1; int stop = 0; int i = 0; '
                    f'while (i < 4) {{ if (i % 2 == 0) {{ flags {symbol} flags[start:step:stop]; }} '
                    'else { flags[0:3] ^= flags[start:step:stop]; } '
                    'start = 3 - start; stop = 3 - stop; step = -step; i += 1; } '
                    f'bool same = flags == "{after}";', (), (), ('same', 'i'), (1, 4))

    def test_nested_transfers_and_shadowing_restore_names(self):
        for symbol, function in self.operations:
            expected = [0, 0, 1, 1, 0, 0, 1, 1]
            total = 0
            source = [0, 1, 0, 1]
            for _ in range(3):
                for i in (1, 2):
                    expected[7] = function(expected[7], source[i])
                    expected[1] = function(expected[1], source[i + 1])
                    total += expected[7]
            after = ''.join(map(str, reversed(expected)))
            self.check_initializers('const int LAST = 7; const bit[4] C = "1010"; bit[8] result = "11001100"; '
                'int first = 0; int last = 2; int i = -1; int visits = 0; int total = 0; '
                'for int LAST in [first:last] { for int i in [0:3] { '
                f'if (i == 0) {{ continue; }} result[{{7,1}}] {symbol} C[i:i + 1]; '
                'visits += 1; total += int(result[7]); if (i == 2) { break; } } } '
                f'bool same = result == "{after}"; int restored = LAST;',
                (), (), ('same', 'visits', 'total', 'restored', 'i'), (1, 6, total, 7, -1))

    def test_skipped_invalid_ranges_never_evaluate_bounds(self):
        bounds = ('start:0:stop', 'bad:stop', 'start:1 / zero:stop', 'start:start',
                  'start:stop + 1', 'bounds[bad]:stop')
        wrappers = ('if (false) {{ {} }}', 'while (false) {{ {} }}', 'for int outer in [1:0] {{ {} }}',
                    'for int outer in [first:last] {{ if (outer < 0) {{ {} }} }}',
                    'while (visits < 2) {{ visits += 1; continue; {} }}', 'while (true) {{ break; {} }}')
        for constant in (False, True):
            for target, before in (('result', '10'), ('result[{7,1}]', '11001100')):
                for symbol, _ in self.operations:
                    for expression in bounds:
                        for wrapper in wrappers:
                            self.check_initializers(('const ' if constant else '') + 'bit[4] source = "1010"; '
                                f'bit[{len(before)}] result = "{before}"; '
                                'int start = 0; int stop = 1; int bad = 4; int zero = 0; '
                                'int first = 0; int last = 1; int visits = 0; array[int,2] bounds = {0,1}; '
                                + wrapper.format(f'{target} {symbol} source[{expression}];') +
                                f'bool same = result == "{before}";', (), (), ('same',), (1,))

    def test_measured_sources_and_bitwise_results_drive_feedback(self):
        for anti in (0, 1):
            for symbol, _ in self.operations:
                # With original bits 1 and 0, AND, OR and XOR each have a
                # separately checkable relationship to the two measured bits.
                expected = {'&=': ('measured[start]', 'false'),
                            '|=': ('true', 'measured[stop]'),
                            '^=': ('!measured[start]', 'measured[stop]')}[symbol]
                for _ in range(8):
                    self.check_initializers('include "stdgates.inc"; qubit[2] q; reset q; h q[0]; cx q[0], q[1]; '
                        + ('x q[1]; ' if anti else '') +
                        'bit[2] measured = measure q; bit[8] result = "11001100"; '
                        'int start = 0; int stop = 1; int step = 1; '
                        'if (measured[0]) { start = 1; stop = 0; step = -1; } '
                        f'result[{{7,1}}] {symbol} measured[start:step:stop]; '
                        f'bool ordered = bool(result[7]) == ({expected[0]}) && bool(result[1]) == ({expected[1]}); '
                        'bool unchanged = (result & "01111101") == "01001100"; '
                        'if (result[1]) { if (stop == 0) { x q[0]; } else { x q[1]; } } bit[2] observed = measure q; '
                        'bool feedback = bool(observed[stop]) == (bool(measured[stop]) != bool(result[1])) '
                        '&& observed[start] == measured[start];',
                        (), (), ('ordered', 'unchanged', 'feedback'), (1, 1, 1))

    def test_later_failures_precede_new_rhs_reads_and_user_writes(self):
        cases = (('start:step:stop', 'step = 0;', 'assertion failed'),
                 ('start:step:stop', 'stop = 0;', 'assertion failed'),
                 ('start:step:stop', 'stop = 2;', 'assertion failed'),
                 ('start:step:stop', 'start = 3; stop = 4;', 'assertion failed'),
                 ('start:step / divisor:stop', 'divisor = 0;', 'division by zero'))
        for constant in (False, True):
            for symbol, function in self.operations:
                for expression, update, error in cases:
                    after = list(map(int, reversed('11001100')))
                    after[7] = function(after[7], 0)
                    after[1] = function(after[1], 1)
                    text = ''.join(map(str, reversed(after)))
                    lines = self.program(('const ' if constant else '') + 'bit[4] source = "1010"; '
                        'bit[8] result = "11001100"; int start = 0; int stop = 1; int step = 1; '
                        'int divisor = 1; int i = 0; while (i < 2) { '
                        f'result[{{7,1}}] {symbol} source[{expression}]; i += 1; {update} }}')
                    writes = [i for i, line in enumerate(lines) if line.startswith('LET RESULT63:')
                              and ' := QASM2QCX_RANGE_BITWISE_ORIGINAL_' in line]
                    self.assertEqual(len(writes), 2)
                    instrumented = []
                    for i, line in enumerate(lines):
                        if line.startswith('LET QASM2QCX_RANGE_COPY_SOURCE_'):
                            instrumented.append('PRINTLN 12345')
                        if i in writes:
                            instrumented.append('PRINTLN 67890')
                        instrumented.append(line)
                        if i == writes[-1]:
                            instrumented.append('PRINTLN ' + ' '.join(f'RESULT63:{bit}' for bit in reversed(range(8))))
                    result = self.execute(instrumented)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(error, result.stderr)
                    self.assertNotIn('runtime error:', result.stderr)
                    self.assertEqual([''.join(line.split()) for line in result.stdout.splitlines()],
                                     ['12345', '12345', '67890', '67890', text])
                    type(self).checked += 1

    def test_bitwise_results_feed_angle_phase_parameters_and_fusion(self):
        import cmath
        import math
        from qasm2qcx_u_power_numerical import BackendAuditTests
        for symbol, function in self.operations:
            value = function(12, 10)
            source = ('include "stdgates.inc"; qubit q; h q; '
                      'const bit[4] C = "1010"; bit[4] result = "1100"; '
                      'int start = 0; int stop = 3; int visits = 0; '
                      f'while (visits < 1) {{ result {symbol} C[start:stop]; visits += 1; }} '
                      'p(angle[4](result)) q;')
            expected = (1 / math.sqrt(2), cmath.exp(1j * value * math.pi / 8) / math.sqrt(2))
            for fused in (False, True):
                BackendAuditTests.check(self, self.program(source, fused=fused), expected)

    def test_documented_runtime_range_bitwise_rhs_example(self):
        document = (pathlib.Path(__file__).parents[1] / 'qcx/qasm2qcx.md').read_text(encoding='utf-8')
        examples = re.findall(r'<!-- runtime-range-bitwise-rhs-example -->\s*```qasm\n(.*?)\n```',
                              document, flags=re.DOTALL)
        self.assertEqual(len(examples), 1)
        source = examples[0].removeprefix('OPENQASM 3.0;')
        self.check_initializers(source + '\nbool same = result == "0011" && flags == "11101100" && one == "0";',
                                (), (), ('same', 'one', 'start', 'step', 'stop', 'index'), (1, 0, 7, -2, 1, -1))

    def test_invalid_ranges_fail_before_rhs_reads_or_writes(self):
        cases = (('0', '0', '1'), ('1', '1', '0'), ('-5', '1', '-4'),
                 ('3', '1', '4'), ('0', '1', '0'), ('0', '1', '2'))
        for start, step, stop in cases:
            for target in ('result', 'result[{1,0}]'):
                for symbol, _ in self.operations:
                    lines = self.program('bit[4] source = "1010"; bit[2] result = "10"; '
                        f'int start = {start}; int step = {step}; int stop = {stop}; '
                        f'{target} {symbol} source[start:step:stop];')
                    instrumented = []
                    for line in lines:
                        if (line.startswith('LET QASM2QCX_RANGE_COPY_SOURCE_') or
                                (line.startswith('LET RESULT63:') and ' := QASM2QCX_RANGE_BITWISE_ORIGINAL_' in line)):
                            instrumented.append('PRINTLN 12345')
                        instrumented.append(line)
                    self.assertIn('PRINTLN 12345', instrumented)
                    result = self.execute(instrumented)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn('assertion failed', result.stderr)
                    self.assertNotIn('runtime error:', result.stderr)
                    self.assertNotIn('12345', result.stdout)
                    type(self).checked += 1


def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    args = parser.parse_args()
    BackendTests.bra = args.bra.resolve()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(BackendTests))
    print(f'runtime range bitwise RHS programs checked: {BackendTests.checked}')
    raise SystemExit(not result.wasSuccessful())


if __name__ == '__main__':
    main()
