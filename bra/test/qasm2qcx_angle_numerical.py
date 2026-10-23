#!/usr/bin/env python3

"""Check scalar angle storage with a default-double non-MPI bra.

Stage 1 injects valid raw UINT bit patterns into generated QCX to test storage
and copying independently of the later floating-point conversion kernel.

  python3 bra/test/qasm2qcx_angle_numerical.py --bra /path/to/bra
"""

import argparse
import pathlib
import resource
import unittest

import qasm2qcx_math_function_numerical as functions


class BackendTests(unittest.TestCase):
    bra = None
    checked = 0
    setUp = functions.BackendTests.setUp
    program = functions.BackendTests.program
    execute = functions.BackendTests.execute

    def check_raw_copy(self, source, raw):
        lines = self.program(source, (('origin', False), ('out', False)))
        program = self.module.openqasm3.parser.parse('OPENQASM 3;')
        converter = self.module.QASM2QCXConverter(program)
        origin = converter._QASM2QCXConverter__capitalize_variable_name('origin')
        declaration = lines.index(f'VAR {origin} UINT')
        lines.insert(declaration + 1, f'LET {origin} := {raw}')
        result = self.execute(lines)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.split(), [str(raw), str(raw)])
        type(self).checked += 1

    def test_explicit_width_storage_and_copies_preserve_exact_bits(self):
        native = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        for width in sorted({1, 4, 8, native}):
            mask = (1 << width) - 1
            for raw in sorted({0, 1, mask, mask // 3}):
                with self.subTest(width=width, raw=raw):
                    self.check_raw_copy(f'angle[{width}] origin; angle[{width}] out = origin; out = origin;', raw)

    def test_default_width_matches_native_uint_storage(self):
        native = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        for raw in (0, 1, (1 << native) - 1):
            self.check_raw_copy(f'angle origin; angle[{native}] out = origin;', raw)

    def test_width_constant_shadowing_does_not_change_copy_representation(self):
        self.check_raw_copy('''const int WIDTH = 4; angle[WIDTH] origin; angle[WIDTH] out;
            for int WIDTH in [0:1] { out = origin; }''', 13)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    BackendTests.bra = parser.parse_args().bra.resolve()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__)))
    print(f'angle programs checked: {BackendTests.checked}')
    raise SystemExit(not result.wasSuccessful())


if __name__ == '__main__':
    main()
