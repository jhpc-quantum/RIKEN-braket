#!/usr/bin/env python3
"""Execute one-dimensional angle-array loops and inspect exact UINT storage (#420).

Pass --bra /path/to/bra for the default-double non-MPI backend audit.
"""

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra', type=pathlib.Path)
    arguments = parser.parse_args()
    BackendTests.bra = arguments.bra.resolve() if arguments.bra else None
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(BackendTests))
    print(f'angle-array loop programs checked: {BackendTests.checked}')
    raise SystemExit(not result.wasSuccessful())
