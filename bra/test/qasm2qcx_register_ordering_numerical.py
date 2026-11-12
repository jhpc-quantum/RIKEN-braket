#!/usr/bin/env python3
"""Check fixed-width bit-register ordering against unsigned integer values (#555)."""

import argparse
import pathlib
import resource
import unittest

import qasm2qcx_runtime_range_equality_numerical as registers


class BackendTests(unittest.TestCase):
    bra = None
    checked = 0
    setUp = registers.BackendTests.setUp
    execute = registers.BackendTests.execute
    program = registers.BackendTests.program
    check_exact = registers.BackendTests.check_exact

    def test_small_values_all_operators_and_operand_kinds(self):
        for width in (1, 2, 3):
            for a in range(1 << width):
                for b in range(1 << width):
                    statements = [f'bit[{width}] a="{a:0{width}b}";',
                                  f'bit[{width}] b="{b:0{width}b}";',
                                  f'const bit[{width}] A="{a:0{width}b}";',
                                  f'const bit[{width}] B="{b:0{width}b}";']
                    outputs, expected = [], []
                    for i, (op, value) in enumerate((('<', a < b), ('<=', a <= b),
                                                     ('>', a > b), ('>=', a >= b))):
                        for j, (left, right) in enumerate((('a', 'b'), ('A', 'b'), ('a', 'B'), ('A', 'B'))):
                            name = f'result{i}{j}'
                            if j == 3:
                                statements.append(f'const bool constant{i}={left}{op}{right}; '
                                                  f'bool {name}=constant{i};')
                            else:
                                statements.append(f'bool {name}={left}{op}{right};')
                            outputs.append((name, None)); expected.append(int(value))
                    self.check_exact(' '.join(statements), outputs, expected)

    def test_wide_values_and_selection_significance(self):
        self.check_exact('''bit[65] a="10000000000000000000000000000000000000000000000000000000000000000";
            bit[65] b="01111111111111111111111111111111111111111111111111111111111111111";
            bool high=a>b; bool low=b<a; bool equal=a>=a;
            bit[4] x="1011"; bool reverse=x[3:-1:0]>x;
            bool chosen=x[{3,1,0}]>"010";
        ''', tuple((name, None) for name in ('high','low','equal','reverse','chosen')), (1,1,1,1,1))


if __name__ == '__main__':
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    args = parser.parse_args()
    BackendTests.bra = args.bra.resolve()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(BackendTests))
    print(f'fixed-width ordering programs checked: {BackendTests.checked}')
    raise SystemExit(not result.wasSuccessful())
