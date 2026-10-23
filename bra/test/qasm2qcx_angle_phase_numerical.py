#!/usr/bin/env python3

"""Compare angle-valued phase gates with independent diagonal matrices (#392).

  python3 bra/test/qasm2qcx_angle_phase_numerical.py --bra /path/to/bra
"""

import argparse
import cmath
import math
import pathlib
import re
import resource
import unittest

import qasm2qcx_math_function_numerical as functions


class BackendTests(unittest.TestCase):
    bra = None
    checked = 0
    setUp = functions.BackendTests.setUp
    program = functions.BackendTests.program
    execute = functions.BackendTests.execute

    def check_phase(self, source, width, bits, conditions, *, basis=None, fused=False):
        # Reference the whole diagonal phase, not a product of the emitted R
        # gates. Use a signed representative to resolve phases near one turn.
        signed = bits if bits <= (1 << (width - 1)) else bits - (1 << width)
        phase = cmath.exp(1j * math.ldexp(math.tau * signed, -width))
        if basis is None:
            preparation = 'h q; s q[0]; t q[2]; '
            initial = [(1j if index & 1 else 1) * (cmath.exp(1j * math.pi / 4) if index & 4 else 1)
                       / math.sqrt(8) for index in range(8)]
        else:
            preparation = ''.join(f'x q[{bit}]; ' for bit in range(3) if basis & (1 << bit))
            initial = [complex(index == basis) for index in range(8)]
        expected = initial[:]
        for wires in conditions:
            mask = sum(1 << wire for wire in wires)
            expected = [value * phase if index & mask == mask else value
                        for index, value in enumerate(expected)]
        lines = self.program('include "stdgates.inc"; qubit[3] q; ' + preparation + source, fused=fused)
        lines.append('DO AMPLITUDES')
        result = self.execute(lines)
        self.assertEqual(result.returncode, 0, result.stderr + '\n' + source)
        actual = {int(index, 2): complex(float(real), float(imaginary)) for index, real, imaginary in
                  re.findall(r'^([01]+) => (\S+) \+ (\S+) i$', result.stdout, re.MULTILINE)}
        self.assertEqual(set(actual), set(range(8)), result.stdout)
        for index, value in enumerate(expected):
            self.assertTrue(cmath.isclose(actual[index], value, rel_tol=1e-5, abs_tol=2e-6),
                            (source, index, actual[index], value))
        type(self).checked += 1

    def test_constants_all_aliases_widths_patterns_and_basis_states(self):
        native = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        for gate in ('p', 'phase', 'u1', 'cp', 'cphase'):
            controlled = gate in ('cp', 'cphase')
            wires = (2, 0) if controlled else (1,)
            operands = ', '.join(f'q[{wire}]' for wire in wires)
            for width in sorted({1, 4, 8, native}):
                for bits in sorted({0, 1, 1 << (width - 1), (1 << width) // 3, (1 << width) - 1}):
                    radians = math.ldexp(math.tau * bits, -width)
                    source = f'const angle[{width}] A = {radians!r}; {gate}(A) {operands};'
                    for basis in (*range(8), None):
                        with self.subTest(gate=gate, width=width, bits=bits, basis=basis):
                            self.check_phase(source, width, bits, (wires,), basis=basis)

    def test_constant_width_casts_and_wrapped_inputs(self):
        for fused in (False, True):
            for expression, width, bits in (('angle[8](A)', 8, 80), ('angle[2](A)', 2, 1),
                                             ('angle(A)', self.module.QASM2QCXConverter.QCX_UINT_WIDTH,
                                              5 << (self.module.QASM2QCXConverter.QCX_UINT_WIDTH - 4))):
                self.check_phase('const angle[4] A = 5 * pi / 8; p(' + expression + ') q[1];',
                                 width, bits, ((1,),), fused=fused)
            self.check_phase('const angle[4] A = -pi / 2; cp(A) q[2], q[0];',
                             4, 12, ((2, 0),), fused=fused)
            self.check_phase('const angle[4] A = tau; p(A) q;', 4, 0, (), fused=fused)

    def test_broadcasting_and_conditioned_control_flow(self):
        for fused in (False, True):
            self.check_phase('const angle[4] A = 5 * pi / 8; p(A) q;',
                             4, 5, ((0,), (1,), (2,)), fused=fused)
            self.check_phase('const angle[4] A = 5 * pi / 8; cp(A) q[2], q[0:1];',
                             4, 5, ((2, 0), (2, 1)), fused=fused)
            self.check_phase('''const angle[4] A = 5 * pi / 8; bool flag = true;
                if (flag) { cp(A) q[0], q[2]; }
                for int i in [1:0] { p(A) q[1]; }
                while (false) { p(A) q[0]; }''', 4, 5, ((0, 2),), fused=fused)
            self.check_phase('const angle[4] A = 5 * pi / 8; for int i in [0:1] { p(A) q[i]; }',
                             4, 5, ((0,), (1,)), fused=fused)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    BackendTests.bra = parser.parse_args().bra.resolve()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__)))
    print(f'angle-phase programs checked: {BackendTests.checked}')
    raise SystemExit(not result.wasSuccessful())


if __name__ == '__main__':
    main()
