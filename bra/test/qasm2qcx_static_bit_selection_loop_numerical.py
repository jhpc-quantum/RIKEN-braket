#!/usr/bin/env python3
"""Audit static bit-register selection iteration (#434) with a non-MPI bra."""

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

    def test_mutable_selections_at_singleton_native_and_wide_widths(self):
        for width in (1, 2, 3, 4, 8, 32, 65, 129):
            for raw in sorted({0, (1 << width) - 1, int(('101100' * width)[:width], 2)}):
                bits = f'{raw:0{width}b}'
                source = f'bit[{width}] flags = "{bits}";'
                outputs, expected = [], []
                selectors = (
                    (f'0:{width - 1}', tuple(range(width))),
                    (f'{width - 1}:-1:0', tuple(reversed(range(width)))),
                    (f'{{{width - 1}, 0, -1}}', (width - 1, 0, width - 1)),
                    ('{0}', (0,)),
                    (f'{-width}:-1', tuple(range(width))),
                    (f'0:2:{width - 1}', tuple(range(0, width, 2))),
                )
                for index, (selector, positions) in enumerate(selectors):
                    checksum = 0
                    selected = [int(bits[-1 - position]) for position in positions]
                    for bit in selected:
                        checksum = (checksum * 3 + bit) % 997
                    source += (f'uint sum{index} = 0; uint checksum{index} = 0; int visits{index} = 0; '
                               f'for bit b in flags[{selector}] {{ sum{index} += uint(b); '
                               f'checksum{index} = (checksum{index} * 3 + uint(b)) % 997; visits{index} += 1; }}')
                    outputs.extend((f'sum{index}', f'checksum{index}', f'visits{index}'))
                    expected.extend((sum(selected), checksum, len(selected)))
                source += f'bool same = flags == "{bits}";'
                self.check_initializers(source, (), (), tuple(outputs) + ('same',), tuple(expected) + (1,))

    def test_source_mutations_affect_future_visits_but_not_copied_iterator(self):
        self.check_initializers('bit[4] flags = "0001"; uint checksum = 0; int visits = 0; bit saved = 0; '
            'for bit b in flags[{0, 1, 0, 3}] { '
            'if (visits == 0) { flags[1] = true; flags[0] = false; saved = b; } '
            'if (visits == 1) { flags[3] = true; } '
            'checksum = checksum * 3 + uint(b); visits += 1; } '
            'bool same = flags == "1010";', (), (), ('checksum', 'visits', 'saved', 'same'), (37, 4, 1, 1))
        self.check_initializers('bit[1] flags = "1"; uint checksum = 0; int visits = 0; '
            'for bit b in flags[{0, 0, 0, 0}] { flags[0] = !flags[0]; '
            'checksum = checksum * 2 + uint(b); visits += 1; } bool same = flags == "1";',
            (), (), ('checksum', 'visits', 'same'), (10, 4, 1))

    def test_header_names_resolve_before_iterator_shadowing(self):
        for name in ('flags', 'LOW', 'HIGH'):
            self.check_initializers('const int LOW = 0; const uint HIGH = 3; '
                'bit[4] flags = "1011"; uint total = 0; bit saved = 0; '
                f'for bit {name} in flags[LOW:HIGH] {{ total += uint({name}); }} '
                'saved = flags[LOW];', (), (), ('total', 'saved'), (3, 1))

    def test_nested_loops_reentry_and_mixed_collection_sources(self):
        self.check_initializers('bit[4] flags = "1011"; int passes = 0; int visits = 0; uint total = 0; '
            'while (passes < 2) { passes += 1; for int outer in [0:1] { '
            'for bit b in flags[0:3] { for bit c in flags[{3, 0, 3}] { '
            'visits += 1; if (b && c) { total += 1; } }}}}',
            (), (), ('passes', 'visits', 'total'), (2, 48, 36))
        self.check_initializers('const bit[2] C = "10"; bit[4] flags = "1011"; '
            'array[angle[4], 1] phases = {pi}; uint total = 0; int visits = 0; '
            'for bit c in C { for angle[4] a in phases { for bit b in flags[3:-1:0] { '
            'for bit d in a[0:3] { visits += 1; if (b && c && d) { total += 1; } }}}}',
            (), (), ('visits', 'total'), (32, 3))

    def test_break_continue_and_skipped_temporary_initialization(self):
        prefix = 'bit[4] flags = "1011"; int visits = 0; uint total = 0; int passes = 0;'
        self.check_initializers(prefix + 'for bit b in flags[0:3] { visits += 1; '
            'if (!b) { continue; } total += uint(b); if (total == 2) { break; }}',
            (), (), ('visits', 'total'), (2, 2))
        for transfer, visits in (('break', 3), ('continue', 12)):
            self.check_initializers(prefix + 'while (passes < 3) { passes += 1; '
                'for bit b in flags[0:3] { visits += 1; ' + transfer + '; total += uint(b); }} '
                'total += popcount(flags);', (), (), ('visits', 'total', 'passes'), (visits, 3, 3))
        self.check_initializers(prefix + 'for bit b in flags[0:3] { '
            'for bit c in flags[0:3] { if (c) { total += uint(b); break; } } '
            'if (!b) { continue; } visits += 1; }', (), (), ('visits', 'total'), (3, 3))

    def test_body_arithmetic_executes_only_on_reached_paths(self):
        prefix = 'bit[4] flags = "1011"; int zero = 0; int value = 0;'
        loop = 'for bit b in flags[0:3] { value = 1 / zero; }'
        for wrapper in ('if (false) {{ {} }}', 'while (false) {{ {} }}',
                        'for int outer in [1:0] {{ {} }}'):
            self.check_initializers(prefix + wrapper.format(loop), (), (), ('value',), (0,))
        for transfer in ('break', 'continue'):
            self.check_initializers(prefix + 'for bit b in flags[0:3] { '
                + transfer + '; value = 1 / zero; }', (), (), ('value',), (0,))
        result = self.execute(self.program(prefix + loop))
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn('runtime error:', result.stderr)
        type(self).checked += 1

    def test_phase_gates_modifiers_and_fusion_use_the_copied_bit(self):
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
        for gate, factor, active in gates:
            source = ('include "stdgates.inc"; qubit[2] q; h q; bit[4] flags = "0011"; '
                      'for bit b in flags[{0, 0, 3}] { flags[0] = !flags[0]; if (b) { '
                      + gate + '}}')
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
    print(f'static bit-selection loop programs checked: {BackendTests.checked}')
    raise SystemExit(not result.wasSuccessful())


if __name__ == '__main__':
    main()
