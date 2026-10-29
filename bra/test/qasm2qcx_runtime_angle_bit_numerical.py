#!/usr/bin/env python3
"""Audit runtime single-bit reads from sized angles (#426).

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
    program = arrays.BackendTests.program
    check_initializers = arrays.BackendTests.check_initializers

    def test_reads_at_every_width_with_signed_unsigned_and_negative_indices(self):
        for width in range(1, self.module.QASM2QCXConverter.QCX_UINT_WIDTH + 1):
            mask = (1 << width) - 1
            for raw in sorted({0, 1 << (width - 1), mask, 0xAAAAAAAA & mask}):
                source = (f'const angle[{width}] C = angle[{width}]("{raw:0{width}b}"); '
                          f'angle[{width}] value = C; array[angle[{width}], 1] phases = {{C}}; '
                          'int element = -1; ')
                outputs, expected = [], []
                for bit in sorted({0, width // 2, width - 1}):
                    source += f'int signed{bit} = {bit}; int negative{bit} = {bit - width}; uint unsigned{bit} = {bit}; '
                    for kind in ('signed', 'negative', 'unsigned'):
                        for origin, operand in enumerate(('C', 'value', 'phases[element]')):
                            name = f'out{kind}{bit}_{origin}'
                            source += f'bit[1] {name} = {operand}[{kind}{bit}]; '
                            outputs.append((name, 1))
                            expected.append((raw >> bit) & 1)
                self.check_initializers(source, tuple(outputs) + (('phases', 1),),
                                        expected + [raw], ('value',), (raw,))

    def test_register_consumers_and_nested_index_expressions(self):
        self.check_initializers('angle[4] a = angle[4]("1010"); int i = -1; uint u = 0; '
            'array[angle[4], 2] phases = {a, angle[4]("0011")}; '
            'array[int, 2] indices = {-1, 0}; int element = 1; '
            'bit[1] direct = a[i]; bit[1] inverse = ~a[u]; '
            'bit[1] combined = a[i] ^ phases[element][u]; '
            'bit[1] rotated = rotl(a[i], i); bit[1] shifted = a[i] >> u; '
            'uint count = popcount(a[i]); bool same = a[i] == "1"; '
            'bool different = a[u] != a[i]; angle[1] roundtrip = angle[1](a[i]); '
            'bit[1] nested = phases[popcount(a[i])][indices[element]]; '
            'bit[1] recursive = phases[element][popcount(a[i])]; '
            'bit[1] arithmetic = a[indices[element] + 3];',
            (('direct', 1), ('inverse', 1), ('combined', 1), ('rotated', 1), ('shifted', 1),
             ('nested', 1), ('recursive', 1), ('arithmetic', 1)),
            (1, 1, 0, 1, 1, 1, 1, 1), ('count', 'same', 'different', 'roundtrip'), (1, 1, 1, 1))

    def test_reentry_shadowing_and_angle_iterators(self):
        self.check_initializers('const int index = 100; const angle[4] value = pi; '
            'array[angle[4], 2] phases = {angle[4]("0011"), angle[4]("1010")}; '
            'uint total = 0; bit[1] out; int stop = 3; int visits = 0; '
            'for int index in [0:3] { total += popcount(value[index]); } '
            'for int index in [0:stop] { for angle[4] value in phases { '
            'out = value[index]; total += popcount(out); }} '
            'while (visits < 2) { visits += 1; for int index in {-1, 0} { '
            'total += popcount(phases[-1][index]); }}',
            (('out', 1),), (1,), ('total', 'visits'), (7, 2))

    def test_bounds_and_domain_failures_only_execute_on_reached_paths(self):
        native = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        maximum = (1 << native) - 1
        prefix = ('angle[4] value = pi; array[angle[4], 1] phases = {pi}; '
                  'int high = 4; int low = -5; int zero = 0; int element = 1; '
                  f'uint huge = {maximum}; int minimum = {self.module.QASM2QCXConverter.QCX_INT_MIN}; '
                  'bit[1] out = "1"; ')
        for operand in ('value[high]', 'value[low]', 'value[minimum]', 'value[huge]',
                        'value[1 / zero]', 'phases[element][zero]',
                        'phases[zero][high]', 'phases[1 / zero][high]'):
            for wrapper in ('if (false) {{ out = {}; }}', 'while (false) {{ out = {}; }}',
                            'for int outer in [1:0] {{ out = {}; }}'):
                self.check_initializers(prefix + wrapper.format(operand), (('out', 1),), (1,))
            self.check_initializers(prefix + f'bool yes = true || {operand} == "0"; '
                                    f'bool no = false && {operand} == "0";',
                                    (('out', 1),), (1,), ('yes', 'no'), (1, 0))
            result = self.execute(self.program(prefix + f'out = {operand};'))
            self.assertNotEqual(result.returncode, 0, operand)
            self.assertNotIn('runtime error:', result.stderr)
            type(self).checked += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra', type=pathlib.Path)
    arguments = parser.parse_args()
    BackendTests.bra = arguments.bra.resolve() if arguments.bra else None
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(BackendTests))
    print(f'runtime angle bit-read programs checked: {BackendTests.checked}')
    raise SystemExit(not result.wasSuccessful())
