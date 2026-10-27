#!/usr/bin/env python3
"""Audit floating powers of angle-valued phase gates against exact turns (#396).

  python3 bra/test/qasm2qcx_angle_power_numerical.py --bra /path/to/bra

Compare full diagonal matrices, including absolute global phases. The oracle
uses independent rational turn arithmetic, with no converter reduction helpers.
"""

import argparse
import cmath
from fractions import Fraction
import math
import pathlib
import re
import resource
import unittest

from anglebits2phase_reference import decode_chain, evaluate_turns
import qasm2qcx_math_function_numerical as functions


def powered_turns(bits, width, descriptors):
    return evaluate_turns(bits, width, decode_chain(descriptors))


def modifiers(descriptors):
    operations = []
    position = 0
    while position < len(descriptors):
        if descriptors[position] == 'INV':
            operations.append('inv')
            position += 1
        else:
            operations.append(f'pow({descriptors[position + 1]})')
            position += 2
    return ' @ '.join(reversed(operations)) + ' @ '


class BackendTests(unittest.TestCase):
    bra = None
    checked = 0
    setUp = functions.BackendTests.setUp
    program = functions.BackendTests.program
    execute = functions.BackendTests.execute

    def check(self, source, phases, *, fused=False, qubits=3, preserved=(), basis=None):
        if not qubits:
            preparation, initial = '', [1 + 0j]
        elif basis is None:
            preparation = f'qubit[{qubits}] q; h q; s q[0]; t q[{qubits - 1}]; '
            initial = [(1j if index & 1 else 1)
                       * (cmath.exp(1j * math.pi / 4) if index & (1 << (qubits - 1)) else 1)
                       / math.sqrt(1 << qubits) for index in range(1 << qubits)]
        else:
            preparation = f'qubit[{qubits}] q; ' + ''.join(
                f'x q[{wire}]; ' for wire in range(qubits) if basis & (1 << wire))
            initial = [complex(index == basis) for index in range(1 << qubits)]
        expected = initial[:]
        for turns, wires in phases:
            # Round the exact product with the stored double TWO_PI only at
            # the final boundary. No R decomposition or EXP/ARG reconstruction.
            phase = cmath.exp(1j * float(turns * Fraction(math.tau)))
            conditions = [(wire, 1) if isinstance(wire, int) else wire for wire in wires]
            mask = sum(1 << wire for wire, _ in conditions)
            required = sum(value << wire for wire, value in conditions)
            expected = [value * phase if index & mask == required else value
                        for index, value in enumerate(expected)]
        lines = self.program('include "stdgates.inc"; ' + preparation + source,
                             tuple((name, False) for name, _ in preserved), fused=fused) + ['DO AMPLITUDES']
        result = self.execute(lines)
        self.assertEqual(result.returncode, 0, result.stderr + '\n' + '\n'.join(lines))
        if preserved:
            self.assertEqual(re.findall(r'^([0-9]+)$', result.stdout, re.MULTILINE),
                             [str(value) for _, value in preserved])
        actual = {int(index, 2) if index else 0: complex(float(real), float(imaginary))
                  for index, real, imaginary in re.findall(r'^([01]*) => (\S+) \+ (\S+) i$', result.stdout, re.MULTILINE)}
        self.assertEqual(set(actual), set(range(1 << qubits)), result.stdout)
        for index, wanted in enumerate(expected):
            self.assertTrue(cmath.isclose(actual[index], wanted, rel_tol=1e-5, abs_tol=2e-6),
                            (source, index, actual[index], wanted))
        type(self).checked += 1

    def test_all_six_aliases_constants_runtime_inputs_and_branch_neighbors(self):
        chains = [('POW', '0.5'), ('POW', '0.5', 'INV'), ('INV', 'POW', '0.5'),
                  ('POW', '2', 'POW', '0.5'), ('POW', '4', 'INV', 'POW', '0.5'),
                  ('POW', '-1.5'), ('POW', '0.5', 'POW', '0.0')]
        for gate in ('p', 'phase', 'u1', 'cp', 'cphase', 'gphase'):
            wires = (2, 0) if gate in ('cp', 'cphase') else () if gate == 'gphase' else (1,)
            operands = ', '.join(f'q[{wire}]' for wire in wires)
            for width in sorted({1, 4, 8, self.module.QASM2QCXConverter.QCX_UINT_WIDTH}):
                half = 1 << (width - 1)
                for bits in sorted({0, 1, half - 1, half, half + 1, (1 << width) - 1} - {1 << width}):
                    declaration = f'const angle[{width}] A = {math.ldexp(math.tau * bits, -width)!r}; '
                    for runtime in (False, True):
                        source = declaration + (f'angle[{width}] live = A; ' if runtime else '')
                        expression = 'live' if runtime else 'A'
                        for chain in chains:
                            turns = powered_turns(bits, width, chain)
                            for fused in (False, True):
                                self.check(source + modifiers(chain) + f'{gate}({expression}) {operands};',
                                           [(turns, wires)], fused=fused,
                                           preserved=(('live', bits),) if runtime else ())

    def test_exact_integer_and_binary64_factors_do_not_collapse(self):
        for factor in ('9007199254740993', '9007199254740993.0', str(10 ** 400 + 1),
                       '0.1', '5e-324', '1.7976931348623157e308'):
            chain = ('POW', factor, 'POW', '0.5')
            turns = powered_turns(1, 1, chain)
            for fused in (False, True):
                self.check('const angle[1] A = pi; ' + modifiers(chain) + 'gphase(A);',
                           [(turns, ())], fused=fused, qubits=0)
                self.check('const angle[1] A = pi; angle[1] live = A; '
                           + modifiers(chain) + 'p(live) q[0];', [(turns, (0,))], fused=fused,
                           preserved=(('live', 1),))

    def test_mixed_control_polarities_broadcasting_and_basis_states(self):
        chain = ('POW', '0.5')
        turns = powered_turns(12, 4, chain)
        declaration = 'const angle[4] A = 3 * pi / 2; angle[4] live = A; '
        for fused in (False, True):
            self.check(declaration + 'pow(0.5) @ p(live) q;',
                       [(turns, (wire,)) for wire in range(3)], fused=fused, preserved=(('live', 12),))
            self.check(declaration + 'pow(0.5) @ cp(live) q[2], q[0:1];',
                       [(turns, (2, 0)), (turns, (2, 1))], fused=fused)
            self.check(declaration + 'negctrl @ pow(0.5) @ gphase(live) q;',
                       [(turns, ((wire, 0),)) for wire in range(3)], fused=fused)
            for gate, operands, wires in (('p', 'q[2], q[0]', ((2, 0), 0)),
                                          ('cp', 'q[2], q[1], q[0]', ((2, 0), 1, 0)),
                                          ('gphase', 'q[2]', ((2, 0),))):
                for basis in (None, *range(8)):
                    self.check(declaration + f'negctrl @ pow(0.5) @ {gate}(live) {operands};',
                               [(turns, wires)], fused=fused, basis=basis)
            self.check(declaration + 'negctrl @ pow(0.5) @ ctrl @ cp(live) q[3], q[2], q[1], q[0];',
                       [(turns, ((3, 0), 2, 1, 0))], fused=fused, qubits=4)

    def test_casts_capture_executable_expressions_and_preserve_source_values(self):
        for fused in (False, True):
            for expression, width, bits in (('angle[8](live)', 8, 80), ('angle[2](live)', 2, 1)):
                self.check('const angle[4] A = 5 * pi / 8; angle[4] live = A; '
                           + f'pow(0.5) @ p({expression}) q[1];',
                           [(powered_turns(bits, width, ('POW', '0.5')), (1,))],
                           fused=fused, preserved=(('live', 5),))
            # PI's exact half-turn is created by a reached native conversion,
            # then resized before the complete modifier chain is evaluated.
            self.check('array[float, 2] data = {0, pi}; int index = 1; '
                       'pow(0.5) @ gphase(angle[8](angle[4](data[index])));',
                       [(Fraction(1, 4), ())], fused=fused, qubits=0)

    def test_classical_control_flow_and_repeated_entries_capture_live_values(self):
        for fused in (False, True):
            self.check('''const angle[4] A = pi; const angle[4] B = 3 * pi / 2;
                angle[4] live = A; int count = 0;
                while (count < 2) { pow(0.5) @ p(live) q[1]; live = B; count += 1; }''',
                       [(Fraction(1, 4), (1,)), (Fraction(-1, 8), (1,))], fused=fused, preserved=(('live', 12),))
            self.check('''const angle[4] A = pi; const angle[4] B = 3 * pi / 2; angle[4] live = A;
                for int i in [0:1] { if (i == 0) { pow(0.5) @ gphase(live); }
                    else { pow(0.5) @ p(live) q[i]; } live = B; }''',
                       [(Fraction(1, 4), ()), (Fraction(-1, 8), (1,))], fused=fused)
            self.check('''const angle[4] A = pi; int stop = 1;
                for int A in [0:stop] { pow(0.5) @ p(A) q[0]; }
                pow(0.5) @ gphase(A);''',
                       [(Fraction.from_float(0.5) / Fraction(math.tau), (0,)), (Fraction(1, 4), ())], fused=fused)

    def test_runtime_validation_skipped_paths_zero_factors_and_exact_resource_limits(self):
        deep_chain = 'pow(0.0) @ ' + 'pow(5e-324) @ ' * 62
        for fused in (False, True):
            for call in ('pow(0.0) @ p(angle[4](1e309)) q[0];',
                         'ctrl @ pow(0.0) @ p(angle[4](1e309)) q[1], q[0];',
                         'pow(0.0) @ gphase(angle[4](1e309));',
                         'pow(0.5) @ p(angle[4](data[index])) q[0];',
                         deep_chain + 'gphase(A);', deep_chain + 'p(live) q[0];'):
                source = 'include "stdgates.inc"; qubit[2] q; const angle[4] A = pi; angle[4] live = A; '
                source += 'array[float, 1] data = {0}; int index = 2; ' + call
                result = self.execute(self.program(source, fused=fused) + ['DO AMPLITUDES'])
                self.assertNotEqual(result.returncode, 0, call)
                self.assertNotIn(' => ', result.stdout)
                type(self).checked += 1
            for wrapper in ('if (flag) {{ {} }}', 'for int i in [1:0] {{ {} }}', 'while (flag) {{ {} }}'):
                source = 'const angle[4] A = pi; bool flag = false; '
                source += wrapper.format('pow(0.0) @ gphase(angle[4](1e309)); ' + deep_chain + 'gphase(A);')
                self.check(source, [], fused=fused, qubits=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    BackendTests.bra = parser.parse_args().bra.resolve()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(BackendTests))
    print(f'angle floating-power programs checked: {BackendTests.checked}')
    raise SystemExit(not result.wasSuccessful())


if __name__ == '__main__':
    main()
