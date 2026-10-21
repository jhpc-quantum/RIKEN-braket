#!/usr/bin/env python3

import importlib.util
import ctypes
import math
from fractions import Fraction
import re
from pathlib import Path
import unittest
from unittest.mock import mock_open, patch


CONVERTER_PATH = Path(__file__).parents[1] / "qcx" / "qasm2qcx.py"
SPEC = importlib.util.spec_from_file_location("qasm2qcx", CONVERTER_PATH)
qasm2qcx = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(qasm2qcx)


def convert(source: str) -> list[str]:
    return qasm2qcx.convert(source)


class IntegerPowerModifierFoundationTests(unittest.TestCase):
    @staticmethod
    def converter(source: str = ''):
        program = qasm2qcx.openqasm3.parser.parse('OPENQASM 3; ' + source)
        converter = qasm2qcx.QASM2QCXConverter(program)
        converter.visit(program)
        return converter

    @staticmethod
    def modifiers(source: str):
        return qasm2qcx.openqasm3.parser.parse(
            'OPENQASM 3; ' + source + ' @ x q;').statements[0].modifiers

    def analyze(self, converter, source: str, iterators=None):
        return converter._QASM2QCXConverter__analyze_gate_modifiers(
            self.modifiers(source), iterators)

    def test_signed_powers_nested_powers_and_inverse_parity(self) -> None:
        converter = self.converter()
        self.assertEqual(converter._QASM2QCXConverter__analyze_gate_modifiers([]),
                         qasm2qcx._GateModifierInfo(1, ()))
        for source, expected in (
                ('pow(0)', 0), ('pow(1)', 1), ('pow(3)', 3), ('pow(-3)', -3),
                ('inv', -1), ('inv @ inv', 1), ('inv @ pow(3)', -3),
                ('pow(-3) @ inv', 3), ('pow(2) @ pow(-3)', -6),
                ('inv @ pow(-2) @ inv @ pow(-3)', 6), ('pow(0) @ inv @ pow(-3)', 0)):
            with self.subTest(source=source):
                self.assertEqual(self.analyze(converter, source), qasm2qcx._GateModifierInfo(expected, ()))

    def test_constant_integer_expressions_and_explicit_integer_casts(self) -> None:
        converter = self.converter('const int N = 2; const uint M = 3;')
        for expression, expected in (
                ('N', 2), ('N + M', 5), ('N * 2 - 1', 3), ('-N', -2),
                ('int(2.5)', 2), ('int(true)', 1), ('uint[2](5)', 1), ('2 ** 3', 8)):
            with self.subTest(expression=expression):
                self.assertEqual(self.analyze(converter, f'pow({expression})').power, expected)

    def test_control_order_is_preserved_across_power_and_inverse_modifiers(self) -> None:
        converter = self.converter()
        for source in ('ctrl(2) @ pow(-3) @ negctrl @ inv @ ctrl',
                       'pow(-3) @ inv @ ctrl(2) @ negctrl @ ctrl',
                       'ctrl(2) @ negctrl @ ctrl @ inv @ pow(-3)'):
            with self.subTest(source=source):
                modifiers = self.modifiers(source)
                before = repr(modifiers)
                self.assertEqual(converter._QASM2QCXConverter__analyze_gate_modifiers(modifiers),
                                 qasm2qcx._GateModifierInfo(3, ((1, 2), (0, 1), (1, 1))))
                self.assertEqual(repr(modifiers), before)

    def test_large_nested_powers_remain_exact_without_emitting_or_expanding(self) -> None:
        converter = self.converter(f'const uint N = {qasm2qcx.QASM2QCXConverter.QCX_UINT_MAX};')
        maximum = converter.QCX_UINT_MAX
        lines = converter._QASM2QCXConverter__qcx_lines.copy()
        self.assertEqual(self.analyze(converter, 'pow(N) @ pow(N)').power, maximum * maximum)
        self.assertEqual(self.analyze(converter, f'pow({converter.QCX_INT_MIN})').power,
                         converter.QCX_INT_MIN)
        self.assertEqual(self.analyze(converter, f'ctrl({converter.QCX_INT_MAX}) @ pow(0)'),
                         qasm2qcx._GateModifierInfo(0, ((1, converter.QCX_INT_MAX),)))
        with self.assertRaisesRegex(qasm2qcx.UnsupportedOpenQASMError, 'total control count'):
            self.analyze(converter, f'pow(0) @ ctrl({converter.QCX_INT_MAX}) @ ctrl')
        self.assertEqual(converter._QASM2QCXConverter__qcx_lines, lines)

    def test_runtime_noninteger_unknown_and_array_exponents_are_rejected(self) -> None:
        converter = self.converter('int n = 2; uint u = 2; float f = 2; bool b = true; '
            'bit bit_value; array[int, 1] values = {2}; const float F = 2;')
        for expression in ('n', 'u', 'f', 'b', 'bit_value', 'values[0]', 'missing',
                           '2.0', 'F', 'true', '1.0im', 'n / 0'):
            with self.subTest(expression=expression), self.assertRaises(qasm2qcx.QASM2QCXError):
                self.analyze(converter, f'pow({expression})')

    def test_loop_iterator_dependencies_are_rejected_even_when_unrolled(self) -> None:
        converter = self.converter('const int N = 2;')
        for value_type in qasm2qcx.ValueType:
            with self.subTest(value_type=value_type), self.assertRaisesRegex(
                    qasm2qcx.UnsupportedOpenQASMError, 'loop iterator'):
                self.analyze(converter, 'pow(N)', {'N': value_type})
        with converter._QASM2QCXConverter__iterator_binding('N', 2):
            with self.assertRaisesRegex(qasm2qcx.UnsupportedOpenQASMError, 'loop iterator'):
                self.analyze(converter, 'pow(N)')

    def test_forward_constant_reference_is_rejected(self) -> None:
        converter = qasm2qcx.QASM2QCXConverter(
            qasm2qcx.openqasm3.parser.parse('OPENQASM 3; const int N = 2;'))
        with self.assertRaises(qasm2qcx.NoVariableNameException):
            self.analyze(converter, 'pow(N)')

    def test_zero_power_does_not_hide_invalid_exponents_or_control_counts(self) -> None:
        converter = self.converter('int n = 2;')
        for source in ('pow(0) @ pow(n)', 'pow(n) @ pow(0)', 'pow(0) @ pow(2.0)',
                       'pow(0) @ pow(1 / 0)', 'pow(1 / 0) @ pow(0)',
                       'pow(0) @ ctrl(0)', 'negctrl(0) @ pow(0)',
                       'pow(0) @ ctrl(n)', 'inv @ inv @ pow(0) @ pow(missing)'):
            with self.subTest(source=source), self.assertRaises(qasm2qcx.QASM2QCXError):
                self.analyze(converter, source)

    def test_missing_power_argument_and_invalid_inverse_arguments_are_rejected(self) -> None:
        converter = self.converter()
        missing = qasm2qcx.ast.QuantumGateModifier(qasm2qcx.ast.GateModifierName.pow)
        invalid = qasm2qcx.ast.QuantumGateModifier(
            qasm2qcx.ast.GateModifierName.inv, qasm2qcx.ast.IntegerLiteral(1))
        zero = self.modifiers('pow(0)')
        for modifier in (missing, invalid):
            with self.subTest(modifier=modifier), self.assertRaises(qasm2qcx.UnsupportedOpenQASMError):
                converter._QASM2QCXConverter__analyze_gate_modifiers(zero + [modifier])

    def test_analysis_preserves_expression_state_caller_temporaries_and_output(self) -> None:
        converter = self.converter('const int N = 2;')
        caller = converter._QASM2QCXConverter__add_new_temporary_variable(qasm2qcx.ValueType.FLOAT)
        fields = ('__expression_kind', '__value', '__value_type', '__value_kind', '__evaluate_constant')
        converter._QASM2QCXConverter__evaluate_constant = False
        previous = [getattr(converter, '_QASM2QCXConverter' + name) for name in fields]
        lines = converter._QASM2QCXConverter__qcx_lines.copy()
        for source in ('pow(N + 1) @ inv', 'pow(1 / 0)', 'pow(0) @ pow(true)'):
            with self.subTest(source=source):
                if source == 'pow(N + 1) @ inv':
                    self.assertEqual(self.analyze(converter, source).power, -3)
                else:
                    with self.assertRaises(qasm2qcx.QASM2QCXError):
                        self.analyze(converter, source)
                self.assertEqual([getattr(converter, '_QASM2QCXConverter' + name) for name in fields], previous)
                self.assertEqual(converter._QASM2QCXConverter__qcx_lines, lines)
                self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, {caller})

    def test_conversion_accepts_integer_power_and_validates_empty_loop_bodies(self) -> None:
        for exponent in (0, 1, -1, 2):
            for operation in ('x q', 'gphase(0.5)'):
                for wrapper in ('{}', 'for int i in [1:0] {{ {} }}'):
                    with self.subTest(exponent=exponent, operation=operation, wrapper=wrapper):
                        lines = convert('OPENQASM 3; include "stdgates.inc"; qubit q; '
                                        + wrapper.format(f'pow({exponent}) @ {operation};'))
                        if wrapper != '{}' or exponent == 0:
                            self.assertEqual(lines, ['QUBITS 1'])


class FloatingPowerModifierFoundationTests(unittest.TestCase):
    converter = staticmethod(IntegerPowerModifierFoundationTests.converter)
    modifiers = staticmethod(IntegerPowerModifierFoundationTests.modifiers)

    def exponent(self, converter, source, iterators=None):
        return converter._QASM2QCXConverter__power_modifier_exponent(
            self.modifiers(source)[0], iterators, allow_runtime=True, allow_float=True)

    def test_finite_constant_float_expressions_keep_their_type(self):
        converter = self.converter('const float F = 0.25; const int N = 2;')
        for expression, expected in (('0.5', 0.5), ('-0.5', -0.5), ('2.0', 2.0),
                                     ('F + 0.25', 0.5), ('1.0 / N', 0.5),
                                     ('float(N)', 2.0), ('float(true)', 1.0),
                                     ('pi / 2', math.pi / 2), ('1e308', 1e308)):
            with self.subTest(expression=expression):
                value = self.exponent(converter, f'pow({expression})')
                self.assertIs(type(value), float)
                self.assertEqual(value, expected)
        self.assertEqual(math.copysign(1, self.exponent(converter, 'pow(-0.0)')), -1)

    def test_nan_infinity_and_overflow_are_rejected(self):
        converter = self.converter()
        for value in (math.nan, math.inf, -math.inf):
            modifier = qasm2qcx.ast.QuantumGateModifier(
                qasm2qcx.ast.GateModifierName.pow, qasm2qcx.ast.FloatLiteral(value))
            with self.subTest(value=value), self.assertRaisesRegex(qasm2qcx.UnsupportedOpenQASMError, 'finite'):
                converter._QASM2QCXConverter__power_modifier_exponent(modifier, allow_float=True)
        for expression in ('1e309', '-1e309', '1e308 * 1e308', '1e309 - 1e309', '1.0 / 0.0'):
            with self.subTest(expression=expression), self.assertRaises(qasm2qcx.QASM2QCXError):
                self.exponent(converter, f'pow({expression})')

    def test_runtime_float_expressions_are_rejected_without_evaluation(self):
        converter = self.converter('float f = 0.5; int n = 2; array[float, 1] values = {0.5};')
        lines = converter._QASM2QCXConverter__qcx_lines.copy()
        for expression in ('f', 'f / 0.0', 'float(n)', 'values[0]'):
            with self.subTest(expression=expression), self.assertRaisesRegex(
                    qasm2qcx.UnsupportedOpenQASMError, 'compile-time constant'):
                self.exponent(converter, f'pow({expression})')
        self.assertEqual(converter._QASM2QCXConverter__qcx_lines, lines)

    def test_iterator_dependent_float_is_not_a_constant_even_when_unrolled(self):
        converter = self.converter('const float F = 0.5;')
        for source, iterators in (('pow(F)', {'F': qasm2qcx.ValueType.FLOAT}),
                                  ('pow(float(i))', {'i': qasm2qcx.ValueType.INT})):
            with self.subTest(source=source), self.assertRaisesRegex(
                    qasm2qcx.UnsupportedOpenQASMError, 'compile-time constant'):
                self.exponent(converter, source, iterators)
        with converter._QASM2QCXConverter__iterator_binding('i', 1):
            with self.assertRaisesRegex(qasm2qcx.UnsupportedOpenQASMError, 'compile-time constant'):
                self.exponent(converter, 'pow(float(i))')

    def test_invalid_types_names_and_missing_arguments_still_fail(self):
        converter = self.converter()
        for expression in ('true', '1.0im', 'missing'):
            with self.subTest(expression=expression), self.assertRaises(qasm2qcx.QASM2QCXError):
                self.exponent(converter, f'pow({expression})')
        missing = qasm2qcx.ast.QuantumGateModifier(qasm2qcx.ast.GateModifierName.pow)
        with self.assertRaisesRegex(qasm2qcx.UnsupportedOpenQASMError, 'requires an exponent'):
            converter._QASM2QCXConverter__power_modifier_exponent(missing, allow_float=True)

    def test_analysis_restores_expression_state_and_preserves_caller_storage(self):
        converter = self.converter('const float F = 0.5;')
        caller = converter._QASM2QCXConverter__add_new_temporary_variable(qasm2qcx.ValueType.FLOAT)
        fields = ('__expression_kind', '__value', '__value_type', '__value_kind', '__evaluate_constant')
        converter._QASM2QCXConverter__evaluate_constant = False
        previous = [getattr(converter, '_QASM2QCXConverter' + name) for name in fields]
        lines = converter._QASM2QCXConverter__qcx_lines.copy()
        self.assertEqual(self.exponent(converter, 'pow(F + 0.5)'), 1.0)
        for expression in ('1.0 / 0.0', '1e309', 'missing'):
            with self.assertRaises(qasm2qcx.QASM2QCXError):
                self.exponent(converter, f'pow({expression})')
            self.assertEqual([getattr(converter, '_QASM2QCXConverter' + name) for name in fields], previous)
            self.assertEqual(converter._QASM2QCXConverter__qcx_lines, lines)
            self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, {caller})

    def test_integer_analysis_is_not_promoted_to_float(self):
        converter = self.converter('int n = 2;')
        value = self.exponent(converter, f'pow({2 ** 100})')
        self.assertIs(type(value), int)
        self.assertEqual(value, 2 ** 100)
        self.assertIsInstance(self.exponent(converter, 'pow(n)'), qasm2qcx._RuntimeGateExponent)

    def test_public_conversion_accepts_fixed_family_float_powers(self):
        for wrapper in ('{}', 'for int i in [1:0] {{ {} }}'):
            lines = convert('OPENQASM 3; include "stdgates.inc"; qubit q; '
                            + wrapper.format('pow(0.5) @ x q;'))
            if wrapper == '{}':
                self.assertEqual(lines, ['QUBITS 1', 'H 0', f'U1 0 {math.pi / 2}', 'H 0'])
            else:
                self.assertEqual(lines, ['QUBITS 1'])


class ParameterizedPhasePowerFoundationTests(unittest.TestCase):
    converter = staticmethod(IntegerPowerModifierFoundationTests.converter)

    def test_angle_is_snapshotted_once_and_source_storage_is_not_modified(self):
        converter = self.converter()
        lines = converter._QASM2QCXConverter__qcx_lines
        with converter._QASM2QCXConverter__captured_phase_angle(
                ('SOURCE', qasm2qcx.ValueType.FLOAT, qasm2qcx.ValueKind.LVALUE)) as (angle, coefficient):
            self.assertNotEqual(angle, 'SOURCE')
            self.assertEqual(lines.count(f'LET {angle} := SOURCE'), 1)
            self.assertEqual(lines[-3:], [f'LET {coefficient} := :I',
                f'LET {coefficient} *= :COMPLEX:{angle}', f'APPLY {coefficient} <- EXP {coefficient}'])
            self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, {angle, coefficient})
        self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())
        self.assertFalse(any(line.startswith('LET SOURCE ') for line in lines))

    def test_integer_inputs_are_explicitly_promoted_and_literals_are_real(self):
        for value_type, value, kind, expected in (
                (qasm2qcx.ValueType.INT, 'N', qasm2qcx.ValueKind.LVALUE, ':REAL:N'),
                (qasm2qcx.ValueType.UINT, 'U', qasm2qcx.ValueKind.LVALUE, ':REAL:U'),
                (qasm2qcx.ValueType.INT, 2, qasm2qcx.ValueKind.LITERAL, '2.0'),
                (qasm2qcx.ValueType.FLOAT, -0.0, qasm2qcx.ValueKind.LITERAL, '-0.0')):
            converter = self.converter()
            with self.subTest(value_type=value_type, value=value):
                with converter._QASM2QCXConverter__captured_phase_angle((value, value_type, kind)) as (angle, _):
                    self.assertIn(f'LET {angle} := {expected}', converter._QASM2QCXConverter__qcx_lines)

    def test_wrong_types_are_rejected_without_emitting_code(self):
        converter = self.converter()
        before = converter._QASM2QCXConverter__qcx_lines.copy()
        for value_type in (qasm2qcx.ValueType.BIT, qasm2qcx.ValueType.BOOL, qasm2qcx.ValueType.COMPLEX):
            with self.subTest(value_type=value_type), self.assertRaises(qasm2qcx.WrongParameterTypeException):
                with converter._QASM2QCXConverter__captured_phase_angle(('BAD', value_type, qasm2qcx.ValueKind.LVALUE)):
                    self.fail('invalid type accepted')
        self.assertEqual(converter._QASM2QCXConverter__qcx_lines, before)

    def test_principal_phase_uses_components_not_an_angle_tolerance(self):
        converter = self.converter()
        with converter._QASM2QCXConverter__captured_phase_angle(
                (0.7, qasm2qcx.ValueType.FLOAT, qasm2qcx.ValueKind.LITERAL)) as (angle, coefficient):
            reserved = converter._QASM2QCXConverter__used_temporary_variables.copy()
            for _ in range(2):
                converter._QASM2QCXConverter__emit_principal_phase(angle, coefficient)
                self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, reserved)
            lines = converter._QASM2QCXConverter__qcx_lines
            self.assertEqual(lines.count(f'APPLY {angle} <- ARG {coefficient}'), 2)
            self.assertEqual(lines.count(f'LET {angle} := :PI'), 2)
            self.assertEqual(sum(line.endswith(f':IMAG:{coefficient}') for line in lines), 2)
            self.assertEqual(sum(line.endswith(f':REAL:{coefficient}') for line in lines), 2)
            jumps = [line for line in lines if line.startswith('JUMPIF ')]
            self.assertTrue(all(line.endswith(('\\= 0.0', '>= 0.0')) for line in jumps))
            labels = [line for line in lines if line.startswith('@')]
            self.assertEqual(len(labels), len(set(labels)))

    def test_capture_preserves_caller_temporaries_on_normal_and_exceptional_exit(self):
        converter = self.converter()
        caller = converter._QASM2QCXConverter__add_new_temporary_variable(qasm2qcx.ValueType.FLOAT)
        for failure in (False, True):
            try:
                with converter._QASM2QCXConverter__captured_phase_angle(
                        (caller, qasm2qcx.ValueType.FLOAT, qasm2qcx.ValueKind.TEMPORARY)) as (angle, coefficient):
                    converter._QASM2QCXConverter__emit_principal_phase(angle, coefficient)
                    self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, {caller, angle, coefficient})
                    if failure:
                        raise RuntimeError('caller failed')
            except RuntimeError:
                if not failure:
                    raise
            self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, {caller})


class RotationPowerFoundationTests(unittest.TestCase):
    converter = staticmethod(IntegerPowerModifierFoundationTests.converter)

    @staticmethod
    def expression(source):
        return qasm2qcx.openqasm3.parser.parse('OPENQASM 3; rx(' + source + ') q;').statements[0].arguments[0]

    @staticmethod
    def info(converter, source='pow(0.5)'):
        return converter._QASM2QCXConverter__analyze_gate_modifiers(
            IntegerPowerModifierFoundationTests.modifiers(source), allow_runtime=True, allow_float=True)

    def test_source_expression_is_evaluated_once_and_caller_state_is_restored(self):
        converter = self.converter('float theta = 0.7;')
        caller = converter._QASM2QCXConverter__add_new_temporary_variable(qasm2qcx.ValueType.FLOAT)
        info = self.info(converter)
        expression = self.expression('theta + 0.1')
        converter._QASM2QCXConverter__evaluate_constant = False
        fields = ('__expression_kind', '__value', '__value_type', '__value_kind', '__evaluate_constant')
        previous = [getattr(converter, '_QASM2QCXConverter' + field) for field in fields]
        start = len(converter._QASM2QCXConverter__qcx_lines)
        with patch.object(converter, 'visit', wraps=converter.visit) as visit:
            with converter._QASM2QCXConverter__powered_rotation_angles(expression, info) as angles:
                self.assertEqual(len(set(angles)), 2)
                self.assertNotIn(caller, angles)
                used = converter._QASM2QCXConverter__used_temporary_variables
                self.assertTrue({caller, *angles} <= used)
                extra = converter._QASM2QCXConverter__add_new_temporary_variable(qasm2qcx.ValueType.FLOAT)
                self.assertNotIn(extra, angles)
                converter._QASM2QCXConverter__release_temporary_variable(extra)
            self.assertEqual(sum(call.args[0] is expression for call in visit.call_args_list), 1)
        self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, {caller})
        self.assertEqual([getattr(converter, '_QASM2QCXConverter' + field) for field in fields], previous)
        lines = converter._QASM2QCXConverter__qcx_lines[start:]
        self.assertEqual(sum(line.endswith('+= 0.1') for line in lines), 1)
        source = converter._QASM2QCXConverter__capitalize_variable_name('theta')
        self.assertFalse(any(line.startswith(f'LET {source} ') for line in lines))

    def test_each_modifier_is_applied_to_both_independent_eigenphases(self):
        converter = self.converter()
        info = self.info(converter, 'pow(0.5) @ inv @ pow(6)')
        with converter._QASM2QCXConverter__powered_rotation_angles(self.expression('0.7'), info) as angles:
            plus, minus = angles
            lines = converter._QASM2QCXConverter__qcx_lines
            factors = [(line.split()[1], line.split()[-1]) for line in lines
                       if line.startswith('LET ') and ' *= ' in line and line.split()[1] in angles]
            self.assertEqual(factors, [(plus, '-0.5'), (minus, '-1.0'),
                (plus, '6.0'), (minus, '6.0'), (plus, '-1.0'), (minus, '-1.0'),
                (plus, '0.5'), (minus, '0.5')])
            self.assertEqual(sum(line.startswith(f'APPLY {plus} <- ARG ') for line in lines), 2)
            self.assertEqual(sum(line.startswith(f'APPLY {minus} <- ARG ') for line in lines), 2)
            self.assertEqual(sum(' <- EXP ' in line for line in lines), 9)
            self.assertFalse(any(' -= ' in line for line in lines))
            ForLoopControlLoweringTests.assert_resolved_jumps(lines)
        self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())

    def test_integer_and_unsigned_sources_are_promoted_before_halving(self):
        for name in ('n', 'u'):
            converter = self.converter(f'int n = {qasm2qcx.QASM2QCXConverter.QCX_INT_MIN}; uint u = uint(-1);')
            info = self.info(converter)
            with converter._QASM2QCXConverter__powered_rotation_angles(self.expression(name), info):
                source = converter._QASM2QCXConverter__capitalize_variable_name(name)
                self.assertEqual(sum(line.endswith(f':= :REAL:{source}')
                                     for line in converter._QASM2QCXConverter__qcx_lines), 1)

    def test_runtime_array_index_is_captured_and_checked_once(self):
        converter = self.converter('array[float, 2] values = {0.7, 1.2}; int index = 0;')
        info = self.info(converter)
        start = len(converter._QASM2QCXConverter__qcx_lines)
        with converter._QASM2QCXConverter__powered_rotation_angles(self.expression('values[index]'), info):
            lines = converter._QASM2QCXConverter__qcx_lines[start:]
            self.assertEqual(sum(line.startswith('ASSERT ') for line in lines), 2)
        self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())

    def test_invalid_factors_fail_before_angle_evaluation(self):
        converter = self.converter('int n = 2;')
        info = self.info(converter, 'pow(0.0) @ pow(0.5) @ pow(n)')
        before = converter._QASM2QCXConverter__qcx_lines.copy()
        with patch.object(converter, 'visit', wraps=converter.visit) as visit:
            with self.assertRaisesRegex(qasm2qcx.UnsupportedOpenQASMError, 'runtime integer factors'):
                with converter._QASM2QCXConverter__powered_rotation_angles(self.expression('missing'), info):
                    self.fail('runtime factor accepted')
            visit.assert_not_called()
        self.assertEqual(converter._QASM2QCXConverter__qcx_lines, before)
        self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())

    def test_failures_do_not_release_caller_temporaries_or_leak_expression_state(self):
        converter = self.converter()
        caller = converter._QASM2QCXConverter__add_new_temporary_variable(qasm2qcx.ValueType.COMPLEX)
        info = self.info(converter)
        fields = ('__expression_kind', '__value', '__value_type', '__value_kind', '__evaluate_constant')
        previous = [getattr(converter, '_QASM2QCXConverter' + field) for field in fields]
        for source in ('true', '1im', 'missing'):
            with self.subTest(source=source), self.assertRaises(qasm2qcx.QASM2QCXError):
                with converter._QASM2QCXConverter__powered_rotation_angles(self.expression(source), info):
                    self.fail('invalid angle accepted')
            self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, {caller})
            self.assertEqual([getattr(converter, '_QASM2QCXConverter' + field) for field in fields], previous)
        with self.assertRaisesRegex(RuntimeError, 'caller failed'):
            with converter._QASM2QCXConverter__powered_rotation_angles(self.expression('0.7'), info):
                raise RuntimeError('caller failed')
        self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, {caller})
        self.assertEqual([getattr(converter, '_QASM2QCXConverter' + field) for field in fields], previous)


class UPowerFoundationTests(unittest.TestCase):
    converter = staticmethod(IntegerPowerModifierFoundationTests.converter)
    info = staticmethod(RotationPowerFoundationTests.info)

    @staticmethod
    def gate(name='U', arguments='theta, phi, lam'):
        return qasm2qcx.openqasm3.parser.parse(f'OPENQASM 3; {name}({arguments}) q;').statements[0]

    def test_parameters_are_captured_once_left_to_right_and_caller_state_is_restored(self):
        for name, arguments in (('u2', 'phi, lam'), ('u3', 'theta, phi, lam'), ('U', 'theta, phi, lam')):
            converter = self.converter('float theta = 0.7; float phi = 0.4; float lam = -0.2;')
            caller = converter._QASM2QCXConverter__add_new_temporary_variable(qasm2qcx.ValueType.COMPLEX)
            converter._QASM2QCXConverter__evaluate_constant = False
            fields = ('__expression_kind', '__value', '__value_type', '__value_kind', '__evaluate_constant')
            before = [getattr(converter, '_QASM2QCXConverter' + field) for field in fields]
            gate = self.gate(name, arguments)
            start = len(converter._QASM2QCXConverter__qcx_lines)
            with patch.object(converter, 'visit', wraps=converter.visit) as visit:
                with converter._QASM2QCXConverter__powered_u_gate_spectrum(gate, self.info(converter)) as prepared:
                    self.assertEqual(len(set(prepared)), 4)
                    self.assertNotIn(caller, prepared)
                    self.assertTrue({caller, *prepared} <= converter._QASM2QCXConverter__used_temporary_variables)
                    extra = converter._QASM2QCXConverter__add_new_temporary_variable(qasm2qcx.ValueType.FLOAT)
                    self.assertNotIn(extra, prepared)
                    converter._QASM2QCXConverter__release_temporary_variable(extra)
                visited = [call.args[0] for call in visit.call_args_list
                           if any(call.args[0] is argument for argument in gate.arguments)]
                self.assertEqual([id(argument) for argument in visited], [id(argument) for argument in gate.arguments])
            self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, {caller})
            self.assertEqual([getattr(converter, '_QASM2QCXConverter' + field) for field in fields], before)
            lines = converter._QASM2QCXConverter__qcx_lines[start:]
            self.assertTrue(any(' <- ABS ' in line for line in lines))
            self.assertFalse(any(line.startswith(('U3 ', 'U3+ ', 'U1 ', 'PHASE ')) for line in lines))
            for source in ('theta', 'phi', 'lam'):
                storage = converter._QASM2QCXConverter__capitalize_variable_name(source)
                self.assertFalse(any(line.startswith(f'LET {storage} ') for line in lines))
            ForLoopControlLoweringTests.assert_resolved_jumps(lines)

    def test_indexed_parameters_are_checked_once_and_snapshots_survive_later_parameters(self):
        converter = self.converter('array[float, 3] values = {0.7, 0.4, -0.2}; int index = 0;')
        start = len(converter._QASM2QCXConverter__qcx_lines)
        gate = self.gate(arguments='values[index], values[index + 1], values[index + 2]')
        with converter._QASM2QCXConverter__powered_u_gate_spectrum(gate, self.info(converter)):
            lines = converter._QASM2QCXConverter__qcx_lines[start:]
            self.assertEqual(sum(line.startswith('ASSERT ') for line in lines), 6)
        self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())

    def test_int_and_uint_parameters_are_cast_to_real_before_halving(self):
        converter = self.converter('int n = -2; uint u = uint(-1);')
        start = len(converter._QASM2QCXConverter__qcx_lines)
        with converter._QASM2QCXConverter__powered_u_gate_spectrum(self.gate(arguments='n, u, 0'), self.info(converter)):
            lines = converter._QASM2QCXConverter__qcx_lines[start:]
            for source in ('n', 'u'):
                storage = converter._QASM2QCXConverter__capitalize_variable_name(source)
                self.assertEqual(sum(line.endswith(f':= :REAL:{storage}') for line in lines), 1)

    def test_invalid_factors_fail_before_parameter_evaluation(self):
        converter = self.converter('int n = 2;')
        info = self.info(converter, 'pow(0.0) @ pow(0.5) @ pow(n)')
        before = converter._QASM2QCXConverter__qcx_lines.copy()
        with patch.object(converter, 'visit', wraps=converter.visit) as visit:
            with self.assertRaisesRegex(qasm2qcx.UnsupportedOpenQASMError, 'runtime integer factors'):
                with converter._QASM2QCXConverter__powered_u_gate_spectrum(self.gate(arguments='missing, 0, 0'), info):
                    self.fail('runtime factor accepted')
            visit.assert_not_called()
        self.assertEqual(converter._QASM2QCXConverter__qcx_lines, before)

    def test_errors_restore_expression_state_and_preserve_caller_temporaries(self):
        for name, arguments in (('U', '0.7, true, 0'), ('u2', 'b, 0'), ('u3', '0, 0, z'),
                                ('U', '0.7, values, 0'), ('U', '0, missing, 0'),
                                ('U', '0, 0'), ('bogus', '0, 0, 0, 0')):
            converter = self.converter('bit b; complex z; array[float, 2] values = {0.7, 0.4};')
            caller = converter._QASM2QCXConverter__add_new_temporary_variable(qasm2qcx.ValueType.COMPLEX)
            fields = ('__expression_kind', '__value', '__value_type', '__value_kind', '__evaluate_constant')
            before = [getattr(converter, '_QASM2QCXConverter' + field) for field in fields]
            with self.assertRaises(qasm2qcx.QASM2QCXError):
                with converter._QASM2QCXConverter__powered_u_gate_spectrum(self.gate(name, arguments), self.info(converter)):
                    self.fail('invalid parameters accepted')
            self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, {caller})
            self.assertEqual([getattr(converter, '_QASM2QCXConverter' + field) for field in fields], before)
        converter = self.converter()
        caller = converter._QASM2QCXConverter__add_new_temporary_variable(qasm2qcx.ValueType.FLOAT)
        with self.assertRaisesRegex(RuntimeError, 'caller failed'):
            with converter._QASM2QCXConverter__powered_u_gate_spectrum(self.gate(arguments='0.7, 0.4, -0.2'), self.info(converter)):
                raise RuntimeError('caller failed')
        self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, {caller})

    def test_parameter_validation_precedes_all_half_angle_work(self):
        converter = self.converter()
        with converter._QASM2QCXConverter__powered_u_gate_spectrum(self.gate(arguments='0.7, 0.4, -0.2'), self.info(converter)):
            lines = converter._QASM2QCXConverter__qcx_lines
            first_half = next(index for index, line in enumerate(lines) if line.endswith('*= 0.5'))
            self.assertEqual(sum(' <- EXP ' in line for line in lines[:first_half]), 3)


class CUPowerFoundationTests(unittest.TestCase):
    converter = staticmethod(IntegerPowerModifierFoundationTests.converter)
    info = staticmethod(RotationPowerFoundationTests.info)

    @staticmethod
    def gate(arguments='theta, phi, lam, gamma'):
        return qasm2qcx.openqasm3.parser.parse(f'OPENQASM 3; cu({arguments}) q[1], q[0];').statements[0]

    def test_four_parameters_are_captured_once_in_order_with_private_output_lifetime(self):
        converter = self.converter('float theta = 0.7; float phi = 0.4; float lam = -0.2; float gamma = 0.9;')
        caller = converter._QASM2QCXConverter__add_new_temporary_variable(qasm2qcx.ValueType.COMPLEX)
        converter._QASM2QCXConverter__evaluate_constant = False
        fields = ('__expression_kind', '__value', '__value_type', '__value_kind', '__evaluate_constant')
        before = [getattr(converter, '_QASM2QCXConverter' + field) for field in fields]
        statement = self.gate()
        start = len(converter._QASM2QCXConverter__qcx_lines)
        with patch.object(converter, 'visit', wraps=converter.visit) as visit:
            with converter._QASM2QCXConverter__powered_u_gate_spectrum(statement, self.info(converter)) as prepared:
                self.assertEqual(len(set(prepared)), 4)
                self.assertNotIn(caller, prepared)
                self.assertTrue({caller, *prepared} <= converter._QASM2QCXConverter__used_temporary_variables)
                extra = converter._QASM2QCXConverter__add_new_temporary_variable(qasm2qcx.ValueType.FLOAT)
                self.assertNotIn(extra, prepared)
                converter._QASM2QCXConverter__release_temporary_variable(extra)
            visited = [call.args[0] for call in visit.call_args_list
                       if any(call.args[0] is argument for argument in statement.arguments)]
            self.assertEqual([id(argument) for argument in visited], [id(argument) for argument in statement.arguments])
        self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, {caller})
        self.assertEqual([getattr(converter, '_QASM2QCXConverter' + field) for field in fields], before)
        lines = converter._QASM2QCXConverter__qcx_lines[start:]
        for source in ('theta', 'phi', 'lam', 'gamma'):
            storage = converter._QASM2QCXConverter__capitalize_variable_name(source)
            self.assertFalse(any(line.startswith(f'LET {storage} ') for line in lines))
        self.assertFalse(any(line.startswith(('U3 ', 'U3+ ', 'U1 ', 'PHASE ')) for line in lines))
        ForLoopControlLoweringTests.assert_resolved_jumps(lines)

    def test_gamma_is_validated_before_core_halving_and_uses_its_full_angle_factor_twice(self):
        converter = self.converter('float theta = 0.7; float phi = 0.4; float lam = -0.2; float gamma = 0.9;')
        start = len(converter._QASM2QCXConverter__qcx_lines)
        with converter._QASM2QCXConverter__powered_u_gate_spectrum(self.gate(), self.info(converter)) as prepared:
            lines = converter._QASM2QCXConverter__qcx_lines[start:]
            first_half = next(index for index, line in enumerate(lines) if line.endswith('*= 0.5'))
            original_factors = [line.split()[1] for line in lines[:first_half] if ' <- EXP ' in line]
            self.assertEqual(len(original_factors), 4)
            gamma_factor = original_factors[-1]
            storage = converter._QASM2QCXConverter__capitalize_variable_name('gamma')
            snapshot = [line.split()[1] for line in lines if line.endswith(f':= {storage}')]
            self.assertEqual(len(snapshot), 1)
            self.assertNotIn(f'LET {snapshot[0]} *= 0.5', lines)
            gamma_products = [index for index, line in enumerate(lines) if line.endswith(f'*= {gamma_factor}')]
            self.assertEqual(len(gamma_products), 2)
            first_principal = next(index for index, line in enumerate(lines)
                                   if line.startswith(f'APPLY {prepared[2]} <- ARG '))
            self.assertLess(max(gamma_products), first_principal)

    def test_four_indexed_parameters_are_checked_once_and_gamma_accepts_numeric_types(self):
        converter = self.converter('array[float, 4] values = {0.7, 0.4, -0.2, 0.9}; int index = 0;')
        statement = self.gate(', '.join(f'values[index + {offset}]' for offset in range(4)))
        start = len(converter._QASM2QCXConverter__qcx_lines)
        with converter._QASM2QCXConverter__powered_u_gate_spectrum(statement, self.info(converter)):
            self.assertEqual(sum(line.startswith('ASSERT ') for line in converter._QASM2QCXConverter__qcx_lines[start:]), 8)
        for name, declaration in (('n', 'int n = -2;'), ('u', 'uint u = uint(-1);')):
            converter = self.converter(declaration)
            with converter._QASM2QCXConverter__powered_u_gate_spectrum(self.gate('0.7, 0.4, -0.2, ' + name), self.info(converter)):
                storage = converter._QASM2QCXConverter__capitalize_variable_name(name)
                self.assertEqual(sum(line.endswith(f':= :REAL:{storage}')
                                     for line in converter._QASM2QCXConverter__qcx_lines), 1)

    def test_invalid_factors_fail_before_any_parameter_evaluation(self):
        converter = self.converter('int n = 2;')
        info = self.info(converter, 'pow(0.0) @ pow(0.5) @ pow(n)')
        before = converter._QASM2QCXConverter__qcx_lines.copy()
        with patch.object(converter, 'visit', wraps=converter.visit) as visit:
            with self.assertRaisesRegex(qasm2qcx.UnsupportedOpenQASMError, 'runtime integer factors'):
                with converter._QASM2QCXConverter__powered_u_gate_spectrum(self.gate('missing, 0, 0, 0'), info):
                    self.fail('runtime factor accepted')
            visit.assert_not_called()
        self.assertEqual(converter._QASM2QCXConverter__qcx_lines, before)

    def test_errors_and_caller_exceptions_restore_state_and_temporary_ownership(self):
        for arguments in ('0.7, 0.4, -0.2, true', '0.7, 0.4, -0.2, b', '0.7, 0.4, -0.2, z',
                          '0.7, 0.4, -0.2, values', '0.7, 0.4, -0.2, missing', '0, 0, 0'):
            converter = self.converter('bit b; complex z; array[float, 2] values = {0.7, 0.9};')
            caller = converter._QASM2QCXConverter__add_new_temporary_variable(qasm2qcx.ValueType.COMPLEX)
            fields = ('__expression_kind', '__value', '__value_type', '__value_kind', '__evaluate_constant')
            before = [getattr(converter, '_QASM2QCXConverter' + field) for field in fields]
            with self.assertRaises(qasm2qcx.QASM2QCXError):
                with converter._QASM2QCXConverter__powered_u_gate_spectrum(self.gate(arguments), self.info(converter)):
                    self.fail('invalid parameters accepted')
            self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, {caller})
            self.assertEqual([getattr(converter, '_QASM2QCXConverter' + field) for field in fields], before)
        converter = self.converter()
        caller = converter._QASM2QCXConverter__add_new_temporary_variable(qasm2qcx.ValueType.FLOAT)
        with self.assertRaisesRegex(RuntimeError, 'caller failed'):
            with converter._QASM2QCXConverter__powered_u_gate_spectrum(self.gate('0.7, 0.4, -0.2, 0.9'), self.info(converter)):
                raise RuntimeError('caller failed')
        self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, {caller})


class CUPowerLoweringTests(unittest.TestCase):
    def clean(self, source):
        return InverseModifierLoweringTests.assert_clean_conversion(self, source)

    def test_documented_cu_power_example(self):
        document = CONVERTER_PATH.with_suffix('.md').read_text(encoding='utf-8')
        examples = re.findall(r'<!-- cu-power-example -->\s*```qasm\n(.*?)\n```',
                              document, flags=re.DOTALL)
        self.assertEqual(len(examples), 1)
        self.clean(examples[0].split('\n', 1)[1])

    def test_intrinsic_and_mixed_controls_condition_both_projectors(self):
        for prefix, operands, mnemonic, conditions in (
                ('', 'q[1], q[0]', 'CU1', ['1']),
                ('negctrl @ ctrl @ ', 'q[3], q[2], q[1], q[0]', 'C3U1', ['3#0', '2', '1'])):
            lines = self.clean('include "stdgates.inc"; qubit[4] q; ' + prefix
                               + 'pow(0.5) @ cu(0.7, 0.4, -0.2, 0.9) ' + operands + ';')
            gates = [line.split() for line in lines if line.startswith(('U3+ ', 'U3 ', mnemonic + ' '))]
            self.assertEqual([gate[0] for gate in gates], ['U3+', mnemonic, mnemonic, 'U3'])
            self.assertEqual(gates[0][1:], gates[-1][1:])
            self.assertEqual(gates[0][1], '0')
            self.assertEqual(gates[1][1:-1], conditions + ['0#0'])
            self.assertEqual(gates[2][1:-1], conditions + ['0'])

    def test_broadcast_captures_four_parameters_only_once(self):
        source = ('include "stdgates.inc"; qubit[2] a; float gamma = 0.9; '
                  'pow(0.5) @ cu(0.7, 0.4, -0.2, gamma + 0.1) a, b;')
        single = self.clean('qubit b; ' + source)
        broadcast = self.clean('qubit[2] b; ' + source)
        self.assertEqual(sum(' <- ARG ' in line for line in single), sum(' <- ARG ' in line for line in broadcast))
        self.assertEqual(sum(line.endswith('+= 0.1') for line in broadcast), 1)
        phases = [line.split() for line in broadcast if line.startswith('CU1 ')]
        self.assertEqual([line[1:3] for line in phases], [['2', '0#0'], ['2', '0'], ['3', '1#0'], ['3', '1']])
        self.assertEqual(len({line[-1] for line in phases}), 2)

    def test_invalid_calls_cannot_hide_at_zero_or_in_empty_loops(self):
        calls = ('cu(0, 0, 0, true) q[1], q[0]', 'cu(0, 0, 0, b) q[1], q[0]',
                 'cu(0, 0, 0, z) q[1], q[0]', 'cu(0, 0, 0, values) q[1], q[0]',
                 'cu(0, 0, 0, missing) q[1], q[0]', 'cu(0, 0, 0) q[1], q[0]',
                 'cu(0, 0, 0, 0) q[0]', 'cu(0, 0, 0, 0) q[0], q[0]',
                 'cu(0, 0, 0, 0) q[2], q[0]', 'cu(0, 0, 0, 0) missing, q[0]',
                 'cu(0, 0, 0, 0) q, a')
        for call in calls:
            for wrapper in ('{}', 'for int i in [1:0] {{ {} }}'):
                with self.subTest(call=call, wrapper=wrapper), self.assertRaises(qasm2qcx.QASM2QCXError):
                    convert('OPENQASM 3; include "stdgates.inc"; qubit[2] q; qubit[3] a; '
                            'bit b; complex z; array[float, 2] values = {0, 0}; '
                            + wrapper.format('pow(0.0) @ ' + call + ';'))
        for wrapper in ('{}', 'for int i in [1:0] {{ {} }}'):
            with self.assertRaises(qasm2qcx.UnsupportedOpenQASMError):
                convert('OPENQASM 3; qubit[2] q; ' + wrapper.format('pow(0.5) @ cu(0, 0, 0, 0) q[1], q[0];'))

    def test_runtime_loop_and_static_indices_and_output_budget(self):
        self.clean('include "stdgates.inc"; qubit[3] q; int stop = 2; '
                   'for int i in [0:stop] { pow(0.5) @ cu(float(i), 0, 0, float(i)) q[2], q[0]; }')
        lines = self.clean('include "stdgates.inc"; qubit[3] q; '
                           'for int i in [0:1] { pow(0.5) @ cu(0, 0, 0, 0) q[2], q[i]; }')
        self.assertEqual(sum(line.startswith('CU1 ') for line in lines), 4)
        source = 'OPENQASM 3; include "stdgates.inc"; qubit[2] q; pow(0.5) @ cu(0.7, 0.4, -0.2, 0.9) q[1], q[0];'
        count = len(convert(source)) - 1
        with patch.object(qasm2qcx.QASM2QCXConverter, 'MAX_GATE_POWER_OUTPUT_LINES', count):
            convert(source)
        with patch.object(qasm2qcx.QASM2QCXConverter, 'MAX_GATE_POWER_OUTPUT_LINES', count - 1):
            with self.assertRaises(qasm2qcx.UnsupportedOpenQASMError):
                convert(source)

    def test_conversion_failures_release_temporaries_and_cli_prints_no_partial_output(self):
        for call in ('cu(0.7, 0.4, -0.2, true) q[1], q[0]',
                     'cu(0.7, 0.4, -0.2, 0.9) missing, q[0]'):
            source = 'OPENQASM 3; include "stdgates.inc"; qubit[2] q; pow(0.5) @ ' + call + ';'
            program = qasm2qcx.openqasm3.parser.parse(source)
            converter = qasm2qcx.QASM2QCXConverter(program)
            with self.assertRaises(qasm2qcx.QASM2QCXError):
                converter.visit(program)
            self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())
            with patch('builtins.open', mock_open(read_data=source)), patch('builtins.print') as output:
                with self.assertRaises(SystemExit):
                    qasm2qcx.main(['cu-power.qasm'])
                output.assert_not_called()


class UPowerLoweringTests(unittest.TestCase):
    def clean(self, source):
        return InverseModifierLoweringTests.assert_clean_conversion(self, source)

    def test_documented_u_family_power_example(self):
        document = CONVERTER_PATH.with_suffix('.md').read_text(encoding='utf-8')
        examples = re.findall(r'<!-- u-family-power-example -->\s*```qasm\n(.*?)\n```',
                              document, flags=re.DOTALL)
        self.assertEqual(len(examples), 1)
        self.clean(examples[0].split('\n', 1)[1])

    def test_all_three_families_use_conventional_basis_changes_and_two_full_phases(self):
        for call in ('u2(phi, lam)', 'u3(theta, phi, lam)', 'U(theta, phi, lam)'):
            lines = self.clean('include "stdgates.inc"; qubit q; float theta = 0.7; float phi = 0.4; '
                               'float lam = -0.2; pow(0.5) @ ' + call + ' q;')
            gates = [line.split() for line in lines if line.startswith(('U3+ ', 'U3 ', 'U1 ', 'PHASE '))]
            self.assertEqual([line[0] for line in gates], ['U3+', 'U1', 'U1', 'U3'])
            self.assertEqual(gates[0][1:], gates[-1][1:])
            self.assertEqual(gates[1][1], '0#0')
            self.assertEqual(gates[2][1], '0')
            self.assertNotEqual(gates[1][2], gates[2][2])
            self.assertTrue(any(' <- ABS ' in line for line in lines))

    def test_builtin_u_needs_no_stdgates_include_but_aliases_do(self):
        self.clean('qubit q; pow(0.5) @ U(0.7, 0.4, -0.2) q;')
        for call in ('u2(0.4, -0.2)', 'u3(0.7, 0.4, -0.2)'):
            for wrapper in ('{}', 'for int i in [1:0] {{ {} }}'):
                with self.assertRaises(qasm2qcx.QASM2QCXError):
                    convert('OPENQASM 3; qubit q; ' + wrapper.format('pow(0.0) @ ' + call + ' q;'))

    def test_mixed_controls_condition_both_phases_but_not_basis_changes(self):
        lines = self.clean('qubit[4] q; negctrl @ pow(0.5) @ ctrl(2) @ U(0.7, 0.4, -0.2) '
                           'q[3], q[2], q[1], q[0];')
        gates = [line.split() for line in lines if line.startswith(('U3+ ', 'U3 ', 'C3U1 '))]
        self.assertEqual([line[0] for line in gates], ['U3+', 'C3U1', 'C3U1', 'U3'])
        self.assertEqual(gates[1][1:-1], ['3#0', '2', '1', '0#0'])
        self.assertEqual(gates[2][1:-1], ['3#0', '2', '1', '0'])

    def test_broadcasting_reuses_one_parameter_capture_and_spectrum(self):
        source = ('include "stdgates.inc"; float theta = 0.7; float phi = 0.4; float lam = -0.2; '
                  'pow(0.5) @ u3(theta + 0.1, phi, lam) q;')
        single = self.clean('qubit q; ' + source)
        broadcast = self.clean('qubit[3] q; ' + source)
        self.assertEqual(sum(' <- ARG ' in line for line in broadcast), sum(' <- ARG ' in line for line in single))
        self.assertEqual(sum(line.endswith('+= 0.1') for line in broadcast), 1)
        diagonals = [line.split() for line in broadcast if line.startswith('U1 ')]
        self.assertEqual([line[1] for line in diagonals], ['0#0', '0', '1#0', '1', '2#0', '2'])
        self.assertEqual(len({line[-1] for line in diagonals}), 2)

    def test_invalid_types_names_arity_and_qubits_cannot_hide_at_zero_or_in_empty_loops(self):
        calls = ('u2(true, 0) q[0]', 'u3(0, b, 0) q[0]', 'U(0, 0, z) q[0]',
                 'U(values, 0, 0) q[0]', 'u3(0, missing, 0) q[0]', 'u2(0) q[0]',
                 'U(0, 0, 0, 0) q[0]', 'u3(0, 0, 0) missing', 'U(0, 0, 0) q[2]',
                 'ctrl @ U(0, 0, 0) q[0], q[0]', 'ctrl @ u2(0, 0) q, a')
        for call in calls:
            for wrapper in ('{}', 'for int i in [1:0] {{ {} }}'):
                with self.subTest(call=call, wrapper=wrapper), self.assertRaises(qasm2qcx.QASM2QCXError):
                    convert('OPENQASM 3; include "stdgates.inc"; qubit[2] q; qubit[3] a; bit b; '
                            'complex z; array[float, 2] values = {0.7, 0.4}; '
                            + wrapper.format('pow(0.0) @ ' + call + ';'))

    def test_runtime_and_nonfinite_factors_remain_rejected(self):
        for call in ('u2(0, 0) q[0]', 'u3(0, 0, 0) q[0]', 'U(0, 0, 0) q[0]', 'cu(0, 0, 0, 0) q[1], q[0]'):
            for modifiers in ('pow(f)', 'pow(0.0) @ pow(0.5) @ pow(n)',
                              'pow(0.0) @ pow(1e309)', 'pow(0.5) @ pow(' + str(10 ** 400) + ')'):
                for wrapper in ('{}', 'for int i in [1:0] {{ {} }}'):
                    with self.assertRaises(qasm2qcx.QASM2QCXError):
                        convert('OPENQASM 3; include "stdgates.inc"; qubit[2] q; int n = 2; float f = 0.5; '
                                + wrapper.format(modifiers + ' @ ' + call + ';'))

    def test_static_qubit_indices_and_runtime_loop_parameters_are_supported(self):
        lines = self.clean('qubit[2] q; for int i in [0:1] { pow(0.5) @ U(float(i), 0, 0) q[i]; }')
        self.assertEqual(sum(line.startswith('U3+ ') for line in lines), 2)
        self.assertEqual(self.clean('qubit q; for int i in [1:0] { pow(0.5) @ U(0, 0, 0) q; }'), ['QUBITS 1'])
        self.clean('qubit q; int stop = 2; for int i in [0:stop] { pow(0.5) @ U(float(i), 0, 0) q; }')

    def test_output_budget_counts_preparation_and_all_broadcast_gates(self):
        with patch.object(qasm2qcx.QASM2QCXConverter, 'MAX_GATE_POWER_OUTPUT_LINES', 1):
            with self.assertRaises(qasm2qcx.UnsupportedOpenQASMError):
                convert('OPENQASM 3; qubit q; pow(0.5) @ U(0.7, 0.4, -0.2) q;')
        source = 'OPENQASM 3; qubit q; pow(0.5) @ U(0.7, 0.4, -0.2) q;'
        lines = convert(source)
        count = len(lines) - 1  # Includes private declarations allocated by preparation.
        with patch.object(qasm2qcx.QASM2QCXConverter, 'MAX_GATE_POWER_OUTPUT_LINES', count):
            convert(source)
            with self.assertRaises(qasm2qcx.UnsupportedOpenQASMError):
                convert(source.replace('qubit q;', 'qubit[2] q;'))

    def test_integer_only_powers_keep_existing_u_family_lowering(self):
        lines = self.clean('include "stdgates.inc"; qubit q; pow(2) @ u3(0.7, 0.4, -0.2) q;')
        self.assertEqual(lines, ['QUBITS 1', 'U3 0 0.7 0.4 -0.2', 'U3 0 0.7 0.4 -0.2'])
        floating = self.clean('include "stdgates.inc"; qubit q; pow(2.0) @ u3(0.7, 0.4, -0.2) q;')
        self.assertTrue(any(' <- ABS ' in line for line in floating))

    def test_conversion_failures_release_temporaries_and_cli_emits_no_partial_output(self):
        for source in ('OPENQASM 3; qubit q; pow(0.5) @ U(0.7, 0.4, -0.2) missing;',
                       'OPENQASM 3; qubit q; pow(0.5) @ U(0.7, 0.4, true) q;'):
            program = qasm2qcx.openqasm3.parser.parse(source)
            converter = qasm2qcx.QASM2QCXConverter(program)
            with self.assertRaises(qasm2qcx.QASM2QCXError):
                converter.visit(program)
            self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())
            with patch('builtins.open', mock_open(read_data=source)), patch('builtins.print') as output:
                with self.assertRaises(SystemExit):
                    qasm2qcx.main(['u-power.qasm'])
                output.assert_not_called()
        with patch.object(qasm2qcx.QASM2QCXConverter, 'MAX_GATE_POWER_OUTPUT_LINES', 1):
            with patch('builtins.open', mock_open(read_data='OPENQASM 3; qubit q; pow(0.5) @ U(0, 0, 0) q;')):
                with patch('builtins.print') as output, self.assertRaises(SystemExit):
                    qasm2qcx.main(['u-power.qasm'])
                output.assert_not_called()


class RotationPowerLoweringTests(unittest.TestCase):
    def clean(self, source):
        return InverseModifierLoweringTests.assert_clean_conversion(self, source)

    def test_documented_rotation_power_example(self):
        document = CONVERTER_PATH.with_suffix('.md').read_text(encoding='utf-8')
        examples = re.findall(r'<!-- rotation-power-example -->\s*```qasm\n(.*?)\n```',
                              document, flags=re.DOTALL)
        self.assertEqual(len(examples), 1)
        self.clean(examples[0].split('\n', 1)[1])

    def test_rotation_families_emit_both_projector_phases(self):
        for gate in ('rx', 'ry', 'rz', 'crx', 'cry', 'crz'):
            operands = 'q[1], q[0]' if gate.startswith('cr') else 'q[0]'
            for expression in ('0.7', 'theta', 'theta + 0.1', 'n', 'u'):
                with self.subTest(gate=gate, expression=expression):
                    lines = self.clean('include "stdgates.inc"; qubit[2] q; '
                        'float theta = 0.7; int n = 1; uint u = 2; '
                        + f'pow(0.5) @ {gate}({expression}) {operands};')
                    self.assertEqual(sum(' <- ARG ' in line for line in lines), 2)
                    self.assertEqual(sum(' <- EXP ' in line for line in lines), 5)
                    diagonal = [line.split() for line in lines if line.startswith(('U1 ', 'CU1 '))]
                    prefix = ['CU1', '1'] if gate.startswith('cr') else ['U1']
                    self.assertEqual(diagonal[0][:-1], prefix + ['0#0'])
                    self.assertEqual(diagonal[1][:-1], prefix + ['0'])
                    self.assertNotEqual(diagonal[0][-1], diagonal[1][-1])
                    basis = [line for line in lines if line.startswith(('H ', 'S ', 'S+ '))]
                    expected = {'x': ['H 0', 'H 0'], 'y': ['S+ 0', 'H 0', 'H 0', 'S 0'], 'z': []}
                    self.assertEqual(basis, expected[gate[-1]])

    def test_mixed_controls_condition_both_projectors_not_basis_changes(self):
        lines = self.clean('include "stdgates.inc"; qubit[4] q; '
            'negctrl @ pow(0.5) @ ctrl @ crx(0.7) q[3], q[2], q[1], q[0];')
        gates = [line for line in lines if line.startswith(('H ', 'C3U1 '))]
        self.assertEqual(gates[0], 'H 0')
        self.assertTrue(gates[1].startswith('C3U1 3#0 2 1 0#0 '))
        self.assertTrue(gates[2].startswith('C3U1 3#0 2 1 0 '))
        self.assertEqual(gates[3], 'H 0')

    def test_broadcasting_shares_preparation_and_preserves_distinct_phase_storage(self):
        lines = self.clean('include "stdgates.inc"; qubit[3] q; float theta = 0.7; '
                           'pow(0.5) @ ry(theta + 0.1) q;')
        diagonal = [line.split() for line in lines if line.startswith('U1 ')]
        self.assertEqual(len(diagonal), 6)
        self.assertEqual([line[1] for line in diagonal], ['0#0', '0', '1#0', '1', '2#0', '2'])
        self.assertEqual(len({line[-1] for line in diagonal}), 2)
        self.assertEqual(len({line[-1] for line in diagonal[::2]}), 1)
        self.assertEqual(sum(' <- ARG ' in line for line in lines), 2)
        self.assertEqual(sum(line.endswith('+= 0.1') for line in lines), 1)

    def test_nested_modifiers_are_not_collapsed_or_emitted_as_scaled_rotations(self):
        lines = self.clean('include "stdgates.inc"; qubit q; '
                          'pow(0.5) @ inv @ pow(6) @ rz(0.7) q;')
        self.assertEqual(sum(' <- ARG ' in line for line in lines), 4)
        self.assertEqual(sum(' <- EXP ' in line for line in lines), 9)
        self.assertFalse(any(line.startswith(('EX ', 'EX+ ', 'EY ', 'EY+ ', 'EZ ', 'EZ+ ', 'PHASE ')) for line in lines))
        self.assertFalse(any(' -= ' in line for line in lines))

    def test_zero_powers_and_empty_loops_preserve_validation(self):
        for gate in ('rx', 'ry', 'rz', 'crx', 'cry', 'crz'):
            operands = 'q[1], q[0]' if gate.startswith('cr') else 'q[0]'
            lines = self.clean('include "stdgates.inc"; qubit[2] q; ' + f'pow(0.0) @ {gate}(0.7) {operands};')
            self.assertEqual(sum(' <- EXP ' in line for line in lines), 5)
            self.assertEqual(self.clean('include "stdgates.inc"; qubit[2] q; '
                + f'for int i in [1:0] {{ pow(0.5) @ {gate}(0.7) {operands}; }}'), ['QUBITS 2'])

    def test_invalid_parameters_arity_operands_and_includes_are_rejected(self):
        calls = ('rx(true) q[0]', 'ry(b) q[0]', 'crz(z) q[1], q[0]', 'rx(values) q[0]',
                 'rx(missing) q[0]', 'ry(0.7) missing', 'crx(0.7) q[0], q[0]',
                 'negctrl @ ry(0.7) q[0], q[0]', 'rz(0.7) q[2]', 'crz(0.7) q[0]',
                 'rx(0.7, 1) q[0]', 'crx(0.7) a, q', 'ctrl @ rz(0.7) q[0]')
        for call in calls:
            for wrapper in ('{}', 'for int i in [1:0] {{ {} }}'):
                with self.subTest(call=call, wrapper=wrapper), self.assertRaises(qasm2qcx.QASM2QCXError):
                    convert('OPENQASM 3; include "stdgates.inc"; qubit[2] q; qubit[3] a; '
                            'bit b; complex z; array[float, 2] values = {0.7, 1.2}; '
                            + wrapper.format(f'pow(0.0) @ {call};'))
        for wrapper in ('{}', 'for int i in [1:0] {{ {} }}'):
            with self.assertRaises(qasm2qcx.QASM2QCXError):
                convert('OPENQASM 3; qubit q; ' + wrapper.format('pow(0.5) @ rx(0.7) q;'))

    def test_invalid_factors_cannot_hide_behind_zero_or_empty_loops(self):
        for modifiers in ('pow(f)', 'pow(0.0) @ pow(0.5) @ pow(n)', 'pow(0.0) @ pow(1e309)',
                          'pow(0.5) @ pow(' + str(10 ** 400) + ')'):
            for wrapper in ('{}', 'for int i in [1:0] {{ {} }}'):
                with self.subTest(modifiers=modifiers, wrapper=wrapper), self.assertRaises(qasm2qcx.QASM2QCXError):
                    convert('OPENQASM 3; include "stdgates.inc"; qubit q; float f = 0.5; int n = 2; '
                            + wrapper.format(f'{modifiers} @ rx(1.0 / 0.0) q;'))

    def test_static_qubit_selectors_and_loop_angle_iterators(self):
        lines = self.clean('include "stdgates.inc"; qubit[2] q; '
                          'for int i in [0:1] { pow(0.5) @ ry(float(i)) q[i]; }')
        self.assertEqual(sum(' <- ARG ' in line for line in lines), 4)

    def test_output_budget_and_exceptional_cleanup(self):
        with patch.object(qasm2qcx.QASM2QCXConverter, 'MAX_GATE_POWER_OUTPUT_LINES', 1):
            with self.assertRaises(qasm2qcx.UnsupportedOpenQASMError):
                convert('OPENQASM 3; include "stdgates.inc"; qubit q; pow(0.5) @ rx(0.7) q;')
        program = qasm2qcx.openqasm3.parser.parse('OPENQASM 3; include "stdgates.inc"; qubit q; '
                                                'pow(0.5) @ rx(0.7) missing;')
        converter = qasm2qcx.QASM2QCXConverter(program)
        with self.assertRaises(qasm2qcx.QASM2QCXError):
            converter.visit(program)
        self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())

    def test_integer_only_rotation_powers_keep_the_existing_lowering(self):
        self.assertEqual(self.clean('include "stdgates.inc"; qubit q; pow(2) @ rx(0.7) q;'),
                         ['QUBITS 1', 'EX 0 -0.7'])


class ParameterizedPhasePowerLoweringTests(unittest.TestCase):
    def clean(self, source):
        return InverseModifierLoweringTests.assert_clean_conversion(self, source)

    def test_documented_parameterized_phase_power_example(self):
        document = CONVERTER_PATH.with_suffix('.md').read_text(encoding='utf-8')
        examples = re.findall(r'<!-- parameterized-phase-power-example -->\s*```qasm\n(.*?)\n```',
                              document, flags=re.DOTALL)
        self.assertEqual(len(examples), 1)
        self.clean(examples[0].split('\n', 1)[1])

    def test_builtin_global_phase_needs_neither_stdgates_nor_declared_qubits(self):
        lines = self.clean('pow(0.5) @ gphase(0.7);')
        self.assertEqual(lines[0], 'QUBITS 0')
        self.assertTrue(lines[-1].startswith('PHASE '))

    def test_phase_family_aliases_use_checked_runtime_lowering(self):
        for gate, operands, mnemonic in (('p', 'q[0]', 'U1'), ('phase', 'q[0]', 'U1'),
                ('u1', 'q[0]', 'U1'), ('cp', 'q[1], q[0]', 'CU1'), ('cphase', 'q[1], q[0]', 'CU1')):
            for expression in ('0.7', 'theta', 'theta + 0.1', 'n', 'u'):
                with self.subTest(gate=gate, expression=expression):
                    lines = self.clean('include "stdgates.inc"; qubit[2] q; '
                        'float theta = 0.7; int n = 1; uint u = 2; '
                        + f'pow(0.5) @ {gate}({expression}) {operands};')
                    self.assertEqual(sum(' <- ARG ' in line for line in lines), 1)
                    self.assertEqual(sum(' <- EXP ' in line for line in lines), 2)
                    self.assertTrue(lines[-1].startswith(mnemonic + ' '))

    def test_capture_and_checks_are_shared_across_broadcast_targets(self):
        lines = self.clean('include "stdgates.inc"; qubit[3] q; float theta = 0.7; '
                           'pow(0.5) @ p(theta + 0.1) q;')
        gates = [line for line in lines if line.startswith('U1 ')]
        self.assertEqual(len(gates), 3)
        self.assertEqual(len({line.split()[-1] for line in gates}), 1)
        self.assertEqual(sum(line.endswith('+= 0.1') for line in lines), 1)
        self.assertEqual(sum(' <- ARG ' in line for line in lines), 1)

    def test_nested_powers_and_inverse_order_are_not_collapsed(self):
        lines = self.clean('include "stdgates.inc"; qubit q; '
                          'pow(0.5) @ inv @ pow(6) @ p(0.7) q;')
        self.assertEqual(sum(' <- ARG ' in line for line in lines), 2)
        self.assertEqual(sum(' <- EXP ' in line for line in lines), 4)
        factors = [line.split()[-1] for line in lines if line.startswith('LET QASM2QCX_REAL_') and ' *= ' in line]
        self.assertEqual(factors, ['6.0', '-1.0', '0.5'])

    def test_quantum_phase_and_mixed_controls_keep_operand_order(self):
        for statement, expected in (
                ('pow(0.5) @ gphase(0.7);', 'PHASE '),
                ('negctrl @ pow(0.5) @ gphase(0.7) q[2];', 'U1 2#0 '),
                ('negctrl @ pow(0.5) @ ctrl @ gphase(0.7) q[2], q[1];', 'CU1 2#0 1 '),
                ('negctrl @ pow(0.5) @ cp(0.7) q[2], q[1], q[0];', 'C2U1 2#0 1 0 ')):
            with self.subTest(statement=statement):
                lines = self.clean('include "stdgates.inc"; qubit[3] q; ' + statement)
                self.assertTrue(lines[-1].startswith(expected), lines)

    def test_zero_powers_do_not_drop_runtime_checks(self):
        for gate in ('p(0.7) q', 'gphase(0.7)'):
            lines = self.clean('include "stdgates.inc"; qubit q; ' + f'pow(0.0) @ {gate};')
            self.assertEqual(sum(' <- EXP ' in line for line in lines), 2)
            self.assertEqual(sum(' <- ARG ' in line for line in lines), 1)

    def test_parameter_types_and_operands_are_validated_in_empty_loops(self):
        calls = ('p(true) q[0]', 'gphase(true)', 'p(1im) q[0]', 'gphase(1im)',
                 'p(missing) q[0]', 'gphase(missing)', 'p(1) missing',
                 'cp(1) q[0], q[0]', 'p(1) q[2]', 'p(1, 2) q[0]',
                 'gphase(1) q[0]', 'ctrl @ gphase(1) q[0], q[1]',
                 'ctrl(2) @ gphase(1) q[0], q[0]', 'p(1) a, q[0]')
        for call in calls:
            for wrapper in ('{}', 'for int i in [1:0] {{ {} }}'):
                with self.subTest(call=call, wrapper=wrapper), self.assertRaises(qasm2qcx.QASM2QCXError):
                    convert('OPENQASM 3; include "stdgates.inc"; qubit[2] q; qubit[3] a; '
                            + wrapper.format(f'pow(0.0) @ {call};'))

    def test_runtime_or_nonfinite_factors_cannot_hide_behind_zero_or_empty_loops(self):
        for factors in ('pow(0.0) @ pow(n) @ pow(0.5)', 'pow(0.0) @ pow(f)',
                        'pow(0.0) @ pow(1e309)', 'pow(0.5) @ pow(' + str(10 ** 400) + ')'):
            for gate in ('p(1) q', 'gphase(1)'):
                for wrapper in ('{}', 'for int i in [1:0] {{ {} }}'):
                    with self.subTest(factors=factors, gate=gate, wrapper=wrapper), self.assertRaises(qasm2qcx.QASM2QCXError):
                        convert('OPENQASM 3; include "stdgates.inc"; qubit q; int n = 2; float f = 0.5; '
                                + wrapper.format(f'{factors} @ {gate};'))

    def test_nonfinite_literal_angle_is_materialized_only_on_reached_path(self):
        lines = self.clean('include "stdgates.inc"; qubit q; '
                          'pow(0.0) @ p(1e308 * 1e308) q;')
        self.assertTrue(any(line.endswith('/= 0.0') for line in lines))
        self.assertFalse(any(' inf' in line.lower() or ' nan' in line.lower() for line in lines))

    def test_output_limit_and_caller_state_are_preserved(self):
        with patch.object(qasm2qcx.QASM2QCXConverter, 'MAX_GATE_POWER_OUTPUT_LINES', 1):
            with self.assertRaises(qasm2qcx.UnsupportedOpenQASMError):
                convert('OPENQASM 3; include "stdgates.inc"; qubit q; pow(0.5) @ p(0.7) q;')
        program = qasm2qcx.openqasm3.parser.parse('OPENQASM 3; include "stdgates.inc"; qubit q; '
                                                'pow(0.5) @ p(true) q;')
        converter = qasm2qcx.QASM2QCXConverter(program)
        with self.assertRaises(qasm2qcx.WrongParameterTypeException):
            converter.visit(program)
        self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())

    def test_static_loop_qubit_selectors_and_runtime_angle_iterators(self):
        for gate in ('p(float(i)) q[i]', 'ctrl @ gphase(float(i)) q[i]'):
            lines = self.clean('include "stdgates.inc"; qubit[2] q; '
                               + f'for int i in [0:1] {{ pow(0.5) @ {gate}; }}')
            self.assertEqual(sum(' <- ARG ' in line for line in lines), 2)


class FloatingPowerModifierOrderingTests(unittest.TestCase):
    converter = staticmethod(IntegerPowerModifierFoundationTests.converter)
    modifiers = staticmethod(IntegerPowerModifierFoundationTests.modifiers)

    def analyze(self, converter, source, iterators=None):
        return converter._QASM2QCXConverter__analyze_gate_modifiers(
            self.modifiers(source), iterators, allow_runtime=True, allow_float=True)

    def phases(self, source, initial=(Fraction(0), Fraction(1))):
        converter = self.converter()
        info = self.analyze(converter, source)
        self.assertIsInstance(info, qasm2qcx._OrderedGateModifierInfo)
        return converter._QASM2QCXConverter__ordered_gate_eigenphases(initial, info)

    def test_representation_preserves_every_modifier_and_argument_in_source_order(self):
        converter = self.converter()
        modifiers = self.modifiers('negctrl @ pow(0.5) @ inv @ ctrl(2) @ pow(-3) @ pow(2.0)')
        original = repr(modifiers)
        info = converter._QASM2QCXConverter__analyze_gate_modifiers(
            modifiers, allow_runtime=True, allow_float=True)
        self.assertIsInstance(info, qasm2qcx._OrderedGateModifierInfo)
        self.assertEqual([modifier.kind for modifier in info.modifiers], [modifier.modifier for modifier in modifiers])
        self.assertEqual([modifier.argument for modifier in info.modifiers], [1, 0.5, None, 2, -3, 2.0])
        self.assertIs(type(info.modifiers[-1].argument), float)
        self.assertIs(type(info.modifiers[-2].argument), int)
        self.assertEqual(info.controls, ((0, 1), (1, 2)))
        self.assertEqual(repr(modifiers), original)
        self.assertFalse(hasattr(info, 'power'))

    def test_float_typed_whole_numbers_still_select_the_ordered_path(self):
        converter = self.converter('const float F = 2.0;')
        for source in ('pow(2.0)', 'pow(F)', 'pow(float(2))', 'pow(-0.0)'):
            with self.subTest(source=source):
                self.assertIsInstance(self.analyze(converter, source), qasm2qcx._OrderedGateModifierInfo)
        self.assertIsInstance(self.analyze(converter, 'pow(2)'), qasm2qcx._GateModifierInfo)

    def test_integer_only_analysis_returns_the_unchanged_legacy_info(self):
        converter = self.converter('int n = 2; uint u = 3;')
        for source in ('inv @ pow(-3) @ ctrl @ pow(2)',
                       'negctrl @ pow(n) @ inv @ pow(u) @ pow(-3)',
                       'pow(0) @ pow(n) @ ctrl(2)'):
            with self.subTest(source=source):
                expected = converter._QASM2QCXConverter__analyze_gate_modifiers(
                    self.modifiers(source), allow_runtime=True)
                self.assertEqual(self.analyze(converter, source), expected)

    def test_nested_powers_are_applied_from_the_inside(self):
        self.assertEqual(self.phases('pow(0.5) @ pow(2)'), (0, 0))
        self.assertEqual(self.phases('pow(2) @ pow(0.5)'), (0, 1))
        self.assertEqual(self.phases('pow(0.5) @ pow(2.0)'), (0, 0))

    def test_inverse_order_is_not_replaced_by_exponent_sign(self):
        self.assertEqual(self.phases('pow(0.5) @ inv'), (0, Fraction(1, 2)))
        self.assertEqual(self.phases('inv @ pow(0.5)'), (0, Fraction(-1, 2)))
        self.assertEqual(self.phases('pow(-0.5)'), (0, Fraction(-1, 2)))
        self.assertEqual(self.phases('inv @ pow(0.5) @ inv'), (0, Fraction(-1, 2)))

    def test_each_power_reselects_the_principal_phase(self):
        self.assertEqual(self.phases('pow(0.5) @ pow(6)', (Fraction(0), Fraction(1, 4))),
                         (0, Fraction(-1, 4)))
        self.assertEqual(self.phases('pow(0.5) @ pow(3)', (Fraction(0), Fraction(-1, 2))),
                         (0, Fraction(1, 4)))

    def test_principal_branch_endpoint_and_nearby_values_are_exact(self):
        wrap = qasm2qcx.QASM2QCXConverter._QASM2QCXConverter__principal_gate_phase
        epsilon = Fraction(1, 2 ** 55)
        for phase, expected in ((-1, 1), (1, 1), (3, 1), (-3, 1), (2, 0),
                                (1 - epsilon, 1 - epsilon), (1 + epsilon, -1 + epsilon),
                                (-1 - epsilon, 1 - epsilon), (-1 + epsilon, -1 + epsilon)):
            with self.subTest(phase=phase):
                self.assertEqual(wrap(Fraction(phase)), expected)

    def test_exact_float_ratio_and_large_exponents_do_not_overflow(self):
        self.assertEqual(self.phases('pow(0.1)'), (0, Fraction(0.1)))
        self.assertEqual(self.phases('pow(1e308)'), (0, 0))
        self.assertEqual(self.phases('pow(1e-308)'), (0, Fraction(1e-308)))
        # Integer factors are not rounded through float while constructing info.
        self.assertEqual(self.phases(f'pow(0.5) @ pow({2 ** 100 + 1})'), (0, Fraction(1, 2)))

    def test_controls_keep_order_and_do_not_change_active_eigenphases(self):
        converter = self.converter()
        for source in ('negctrl @ pow(0.5) @ ctrl(2)', 'pow(0.5) @ negctrl @ ctrl(2)',
                       'negctrl @ ctrl(2) @ pow(0.5)'):
            with self.subTest(source=source):
                info = self.analyze(converter, source)
                self.assertEqual(info.controls, ((0, 1), (1, 2)))
                self.assertEqual(converter._QASM2QCXConverter__ordered_gate_eigenphases(
                    (Fraction(0), Fraction(1)), info), (0, Fraction(1, 2)))

    def test_runtime_integer_factors_are_preserved_but_not_evaluated_as_constants(self):
        converter = self.converter('int n = 3;')
        info = self.analyze(converter, 'pow(0.5) @ inv @ pow(n)')
        self.assertIsInstance(info.modifiers[-1].argument, qasm2qcx._RuntimeGateExponent)
        with self.assertRaisesRegex(qasm2qcx.UnsupportedOpenQASMError, 'runtime integer factors'):
            converter._QASM2QCXConverter__ordered_gate_eigenphases((Fraction(0), Fraction(1)), info)

    def test_zero_factors_do_not_hide_invalid_modifiers_or_dependencies(self):
        converter = self.converter('int n = 1; float f = 0.5;')
        for source in ('pow(0.0) @ pow(missing)', 'pow(0.0) @ pow(1e309)',
                       'pow(0.0) @ pow(f)', 'pow(0.0) @ ctrl(0)',
                       'pow(0.0) @ ctrl(n)', 'pow(0.0) @ pow(true)'):
            with self.subTest(source=source), self.assertRaises(qasm2qcx.QASM2QCXError):
                self.analyze(converter, source)
        with self.assertRaisesRegex(qasm2qcx.UnsupportedOpenQASMError, 'compile-time constant'):
            self.analyze(converter, 'pow(0.0) @ pow(float(i))', {'i': qasm2qcx.ValueType.INT})

    def test_analysis_emits_nothing_and_preserves_expression_state_and_storage(self):
        converter = self.converter('const float F = 0.5;')
        caller = converter._QASM2QCXConverter__add_new_temporary_variable(qasm2qcx.ValueType.FLOAT)
        fields = ('__expression_kind', '__value', '__value_type', '__value_kind', '__evaluate_constant')
        converter._QASM2QCXConverter__evaluate_constant = False
        previous = [getattr(converter, '_QASM2QCXConverter' + name) for name in fields]
        lines = converter._QASM2QCXConverter__qcx_lines.copy()
        self.analyze(converter, 'negctrl @ pow(F) @ inv @ pow(2)')
        with self.assertRaises(qasm2qcx.QASM2QCXError):
            self.analyze(converter, 'pow(F) @ pow(1.0 / 0.0)')
        self.assertEqual([getattr(converter, '_QASM2QCXConverter' + name) for name in fields], previous)
        self.assertEqual(converter._QASM2QCXConverter__qcx_lines, lines)
        self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, {caller})


class FloatingPowerModifierLoweringTests(unittest.TestCase):
    # Resolve this later-defined helper at call time, as this class appears
    # before the existing inverse-lowering tests in the module.
    def assert_clean_conversion(self, source):
        return InverseModifierLoweringTests.assert_clean_conversion(self, source)

    def test_documented_floating_power_example(self):
        document = CONVERTER_PATH.with_suffix('.md').read_text(encoding='utf-8')
        examples = re.findall(r'<!-- floating-power-example -->\s*```qasm\n(.*?)\n```',
                              document, flags=re.DOTALL)
        self.assertEqual(len(examples), 1)
        self.assert_clean_conversion(examples[0].split('\n', 1)[1])

    def test_fixed_basis_changes_use_existing_qcx_gates(self):
        for gate, expected in (
                ('x', ['H 0', f'U1 0 {math.pi / 2}', 'H 0']),
                ('y', ['S+ 0', 'H 0', f'U1 0 {math.pi / 2}', 'H 0', 'S 0']),
                ('h', [f'EY 0 {math.pi / 8}', f'U1 0 {math.pi / 2}', f'EY 0 {-math.pi / 8}']),
                ('z', [f'U1 0 {math.pi / 2}']), ('s', [f'U1 0 {math.pi / 4}']),
                ('sdg', [f'U1 0 {-math.pi / 4}']), ('t', [f'U1 0 {math.pi / 8}']),
                ('tdg', [f'U1 0 {-math.pi / 8}']),
                ('sx', ['H 0', f'U1 0 {math.pi / 4}', 'H 0']), ('id', [])):
            with self.subTest(gate=gate):
                self.assertEqual(self.assert_clean_conversion('include "stdgates.inc"; qubit q; '
                    + f'pow(0.5) @ {gate} q;'), ['QUBITS 1', *expected])

    def test_swap_basis_change_and_conditioned_phase(self):
        self.assertEqual(self.assert_clean_conversion('include "stdgates.inc"; qubit[3] q; '
            'negctrl @ pow(0.5) @ swap q[2], q[0], q[1];'),
            ['QUBITS 3', 'CX 0 1', 'H 0', f'C2U1 2#0 0 1 {math.pi / 2}', 'H 0', 'CX 0 1'])
        self.assertEqual(self.assert_clean_conversion('include "stdgates.inc"; qubit[3] q; '
            'pow(0.5) @ cswap q[2], q[0], q[1];'),
            ['QUBITS 3', 'CX 0 1', 'H 0', f'C2U1 2 0 1 {math.pi / 2}', 'H 0', 'CX 0 1'])

    def test_only_diagonal_phase_is_conditioned_and_control_order_is_preserved(self):
        self.assertEqual(self.assert_clean_conversion('include "stdgates.inc"; qubit[4] q; '
            'negctrl @ pow(0.5) @ ctrl @ cx q[3], q[0], q[2], q[1];'),
            ['QUBITS 4', 'H 1', f'C3U1 3#0 0 2 1 {math.pi / 2}', 'H 1'])

    def test_inverse_order_and_nested_powers_reach_lowering_unchanged(self):
        for modifiers, expected in (('pow(0.5) @ inv', [f'U1 0 {math.pi / 2}']),
                                     ('inv @ pow(0.5)', [f'U1 0 {-math.pi / 2}']),
                                     ('pow(0.5) @ pow(2)', []), ('pow(2) @ pow(0.5)', [f'U1 0 {math.pi}'])):
            with self.subTest(modifiers=modifiers):
                self.assertEqual(self.assert_clean_conversion('include "stdgates.inc"; qubit q; '
                    + f'{modifiers} @ z q;'), ['QUBITS 1', *expected])
        self.assertEqual(self.assert_clean_conversion('include "stdgates.inc"; qubit q; '
            'pow(0.5) @ pow(6) @ t q;'), ['QUBITS 1', f'U1 0 {-math.pi / 4}'])

    def test_tiny_phase_is_not_rounded_to_zero_before_multiplication_by_pi(self):
        self.assertEqual(self.assert_clean_conversion('include "stdgates.inc"; qubit q; '
            'pow(5e-324) @ t q;'), ['QUBITS 1', 'U1 0 5e-324'])

    def test_large_exponents_and_constant_zero_emit_no_repetition(self):
        for gate, operands in (('x', 'q[0]'), ('t', 'q[0]'), ('swap', 'q[0], q[1]')):
            with self.subTest(gate=gate):
                self.assertEqual(self.assert_clean_conversion('include "stdgates.inc"; qubit[2] q; '
                    + f'pow(1e308) @ {gate} {operands};'), ['QUBITS 2'])
        self.assertEqual(self.assert_clean_conversion('include "stdgates.inc"; qubit q; '
            'pow(0.0) @ pow(0.5) @ x q;'), ['QUBITS 1'])

    def test_identity_powers_still_validate_arity_operands_and_broadcasting(self):
        for call in ('pow(2.0) @ cx q[0], q[0]', 'pow(0.0) @ swap q[0], q[0]',
                     'pow(0.0) @ negctrl @ id q[0], q[0]', 'pow(0.0) @ x missing',
                     'pow(0.0) @ x q[2]', 'pow(0.0) @ x(1) q[0]',
                     'pow(0.0) @ swap q[0:1], q[0:0]'):
            for wrapper in ('{}', 'for int i in [1:0] {{ {} }}'):
                with self.subTest(call=call, wrapper=wrapper), self.assertRaises(qasm2qcx.QASM2QCXError):
                    convert('OPENQASM 3; include "stdgates.inc"; qubit[2] q; ' + wrapper.format(call + ';'))

    def test_cu_zero_power_and_empty_loop_are_supported(self):
        self.assert_clean_conversion('include "stdgates.inc"; qubit q; qubit r; '
                                     'pow(0.0) @ cu(0.2, 0.3, 0.4, 0.5) q, r;')
        self.assertEqual(self.assert_clean_conversion('include "stdgates.inc"; qubit q; qubit r; '
            'for int i in [1:0] { pow(0.0) @ cu(0.2, 0.3, 0.4, 0.5) q, r; }'), ['QUBITS 2'])

    def test_runtime_exponents_and_nonfinite_constants_remain_explicit_errors(self):
        for modifiers in ('pow(f)', 'pow(0.5) @ pow(n)', 'pow(n) @ pow(0.5)',
                          'pow(0.0) @ pow(1e309)', 'pow(0.0) @ pow(f)'):
            for wrapper in ('{}', 'for int i in [1:0] {{ {} }}'):
                with self.subTest(modifiers=modifiers, wrapper=wrapper), self.assertRaises(qasm2qcx.QASM2QCXError):
                    convert('OPENQASM 3; include "stdgates.inc"; qubit q; int n = 2; float f = 0.5; '
                            + wrapper.format(f'{modifiers} @ x q;'))

    def test_include_and_duration_checks_are_not_bypassed(self):
        for source in ('qubit q; pow(0.0) @ x q;',
                       'include "stdgates.inc"; qubit q; pow(0.5) @ x[1ns] q;'):
            with self.subTest(source=source), self.assertRaises(qasm2qcx.QASM2QCXError):
                convert('OPENQASM 3; ' + source)

    def test_loops_branches_and_broadcasting_use_no_classical_scratch(self):
        lines = self.assert_clean_conversion('include "stdgates.inc"; qubit[2] q; int stop = 1; '
            'if (true) { pow(0.5) @ x q; } for int i in [0:stop] { pow(0.5) @ h q; } '
            'while (false) { pow(0.5) @ swap q[0], q[1]; }')
        self.assertFalse(any(line.startswith('VAR QASM2QCX_REAL_') for line in lines))

    def test_constant_loop_iterator_qubit_indices_are_validated_at_expansion(self):
        lines = self.assert_clean_conversion('include "stdgates.inc"; qubit[2] q; '
            'for int i in [0:1] { pow(0.5) @ x q[i]; }')
        self.assertEqual(lines, ['QUBITS 2', 'H 0', f'U1 0 {math.pi / 2}', 'H 0',
                                'H 1', f'U1 1 {math.pi / 2}', 'H 1'])
        self.assertEqual(self.assert_clean_conversion('include "stdgates.inc"; qubit[2] q; '
            'for int i in [1:0] { pow(0.5) @ x q[i]; }'), ['QUBITS 2'])
        for index in ('I', '0 + 0'):
            with self.subTest(index=index):
                self.assertEqual(self.assert_clean_conversion('include "stdgates.inc"; const int I = 0; '
                    f'qubit q; for int i in [0:0] {{ pow(0.5) @ x q[{index}]; }}'),
                    ['QUBITS 1', 'H 0', f'U1 0 {math.pi / 2}', 'H 0'])
        for source in ('for int i in [0:2] { pow(0.5) @ x q[i]; }',
                       'int stop = 1; for int i in [0:stop] { pow(0.5) @ x q[i]; }'):
            with self.subTest(source=source), self.assertRaises(qasm2qcx.QASM2QCXError):
                convert('OPENQASM 3; include "stdgates.inc"; qubit[2] q; ' + source)

    def test_broadcast_budget_is_checked_and_cli_emits_no_partial_output(self):
        source = 'OPENQASM 3; include "stdgates.inc"; qubit[2] q; pow(0.5) @ x q;'
        with patch.object(qasm2qcx.QASM2QCXConverter, 'MAX_GATE_POWER_OUTPUT_LINES', 5):
            with self.assertRaisesRegex(qasm2qcx.UnsupportedOpenQASMError, 'gate power expansion'):
                convert(source)
            with patch('builtins.open', mock_open(read_data=source)), patch('builtins.print') as output:
                with self.assertRaises(SystemExit):
                    qasm2qcx.main(['float-power.qasm'])
                output.assert_not_called()


class RuntimeIntegerPowerModifierFoundationTests(unittest.TestCase):
    converter = staticmethod(IntegerPowerModifierFoundationTests.converter)
    modifiers = staticmethod(IntegerPowerModifierFoundationTests.modifiers)

    def analyze(self, converter, source, iterators=None):
        return converter._QASM2QCXConverter__analyze_gate_modifiers(
            self.modifiers(source), iterators, allow_runtime=True)

    def test_runtime_factors_remain_separate_and_in_source_order(self) -> None:
        converter = self.converter('int n = -2; uint[3] u = 3;')
        modifiers = self.modifiers('negctrl @ pow(n) @ inv @ pow(-3) @ ctrl(2) @ pow(u) @ pow(n)')
        original = repr(modifiers)
        info = converter._QASM2QCXConverter__analyze_gate_modifiers(modifiers, allow_runtime=True)
        self.assertEqual(info.power, 3)
        self.assertEqual(info.controls, ((0, 1), (1, 2)))
        self.assertEqual([factor.value_type for factor in info.runtime_exponents],
                         [qasm2qcx.ValueType.INT, qasm2qcx.ValueType.UINT, qasm2qcx.ValueType.INT])
        self.assertEqual([factor.expression.name for factor in info.runtime_exponents], ['n', 'u', 'n'])
        self.assertEqual(repr(modifiers), original)

    def test_constant_path_is_unchanged_and_zero_does_not_discard_runtime_factors(self) -> None:
        converter = self.converter('int n = 2; const int N = 3;')
        self.assertEqual(self.analyze(converter, 'inv @ pow(N) @ pow(-2)'),
                         qasm2qcx._GateModifierInfo(6, ()))
        info = self.analyze(converter, 'pow(0) @ pow(n)')
        self.assertEqual(info.power, 0)
        self.assertEqual(len(info.runtime_exponents), 1)
        for source in ('pow(0) @ pow(true)', 'pow(0) @ pow(missing)',
                       'pow(n) @ ctrl(0)', 'pow(n) @ pow(1 / 0)'):
            with self.subTest(source=source), self.assertRaises(qasm2qcx.QASM2QCXError):
                self.analyze(converter, source)

    def test_integer_expression_types_are_classified_without_execution(self) -> None:
        converter = self.converter('int n = 2; uint[3] u = 3; int index = 0; bool flag = true; '
            'array[int, 1] values = {2}; array[uint[3], 1] unsigneds = {3};')
        lines = converter._QASM2QCXConverter__qcx_lines.copy()
        for expression, expected in (('n + 1', qasm2qcx.ValueType.INT),
                                     ('u + uint[3](1)', qasm2qcx.ValueType.UINT),
                                     ('values[index]', qasm2qcx.ValueType.INT),
                                     ('unsigneds[index]', qasm2qcx.ValueType.UINT),
                                     ('int(flag)', qasm2qcx.ValueType.INT),
                                     ('n / 0', qasm2qcx.ValueType.INT)):
            with self.subTest(expression=expression):
                info = self.analyze(converter, f'pow({expression})')
                self.assertEqual(info.runtime_exponents[0].value_type, expected)
                self.assertEqual(converter._QASM2QCXConverter__qcx_lines, lines)
        for expression in ('true', '2.0', '1.0im', 'values', 'missing'):
            with self.subTest(expression=expression), self.assertRaises(qasm2qcx.QASM2QCXError):
                self.analyze(converter, f'pow({expression})')

    def test_integer_iterator_types_and_shadowed_constants_are_classified_without_values(self) -> None:
        converter = self.converter('const int N = 3;')
        for value_type in qasm2qcx.ValueType:
            with self.subTest(value_type=value_type):
                if value_type in (qasm2qcx.ValueType.INT, qasm2qcx.ValueType.UINT):
                    info = self.analyze(converter, 'pow(N)', {'N': value_type})
                    self.assertEqual(info.runtime_exponents[0].value_type, value_type)
                else:
                    with self.assertRaises(qasm2qcx.UnsupportedOpenQASMError):
                        self.analyze(converter, 'pow(N)', {'N': value_type})
        with converter._QASM2QCXConverter__iterator_binding('N', -2):
            info = self.analyze(converter, 'pow(N)')
            self.assertEqual(len(info.runtime_exponents), 1)
            with converter._QASM2QCXConverter__captured_gate_exponents(info.runtime_exponents) as captured:
                self.assertIn(f'LET {captured[0].storage} := -2', converter._QASM2QCXConverter__qcx_lines)

    def test_analysis_preserves_expression_state_and_caller_storage(self) -> None:
        converter = self.converter('int n = 2;')
        caller = converter._QASM2QCXConverter__add_new_temporary_variable(qasm2qcx.ValueType.FLOAT)
        fields = ('__expression_kind', '__value', '__value_type', '__value_kind', '__evaluate_constant')
        converter._QASM2QCXConverter__evaluate_constant = False
        previous = [getattr(converter, '_QASM2QCXConverter' + field) for field in fields]
        lines = converter._QASM2QCXConverter__qcx_lines.copy()
        self.analyze(converter, 'pow(n / 0) @ inv')
        with self.assertRaises(qasm2qcx.QASM2QCXError):
            self.analyze(converter, 'pow(n) @ pow(missing)')
        self.assertEqual([getattr(converter, '_QASM2QCXConverter' + field) for field in fields], previous)
        self.assertEqual(converter._QASM2QCXConverter__qcx_lines, lines)
        self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, {caller})

    def test_capture_reserves_distinct_native_snapshots_and_evaluates_each_factor_once(self) -> None:
        converter = self.converter('int n = -2; uint[3] u = 7;')
        info = self.analyze(converter, 'pow(n) @ pow(u + uint[3](1)) @ pow(n + 1)')
        expressions = [factor.expression for factor in info.runtime_exponents]
        visited = []
        visit = converter.visit

        def record(node, context=None):
            visited.append(node)
            return visit(node, context)

        with patch.object(converter, 'visit', side_effect=record):
            with converter._QASM2QCXConverter__captured_gate_exponents(info.runtime_exponents) as captured:
                self.assertEqual(len({factor.storage for factor in captured}), 3)
                self.assertEqual([factor.value_type for factor in captured], [factor.value_type for factor in info.runtime_exponents])
                self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables,
                                 {factor.storage for factor in captured})
                self.assertEqual(converter._QASM2QCXConverter__temporary_uint_types[captured[1].storage].width,
                                 converter.QCX_UINT_WIDTH)
        for expression in expressions:
            self.assertEqual(sum(node is expression for node in visited), 1)
        self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())

    def test_capture_preserves_full_signed_and_unsigned_boundaries_without_conversion(self) -> None:
        converter = self.converter(f'int n = {qasm2qcx.QASM2QCXConverter.QCX_INT_MIN}; '
            f'uint u = {qasm2qcx.QASM2QCXConverter.QCX_UINT_MAX};')
        info = self.analyze(converter, 'pow(n) @ pow(u)')
        start = len(converter._QASM2QCXConverter__qcx_lines)
        with converter._QASM2QCXConverter__captured_gate_exponents(info.runtime_exponents) as captured:
            lines = converter._QASM2QCXConverter__qcx_lines[start:]
            self.assertEqual(lines, [f'VAR {captured[0].storage} INT', f'LET {captured[0].storage} := N1',
                                     f'VAR {captured[1].storage} UINT', f'LET {captured[1].storage} := U1'])

    def test_capture_reads_indexed_expressions_once_and_cleans_index_temporaries(self) -> None:
        converter = self.converter('array[int, 2] values = {-2, 3}; int i = 0;')
        info = self.analyze(converter, 'pow(values[i]) @ pow(values[i + 1])')
        start = len(converter._QASM2QCXConverter__qcx_lines)
        with converter._QASM2QCXConverter__captured_gate_exponents(info.runtime_exponents) as captured:
            lines = converter._QASM2QCXConverter__qcx_lines[start:]
            self.assertEqual(sum(line.endswith(' := I1') for line in lines), 2)
            self.assertEqual(sum(line.startswith('ASSERT ') for line in lines), 4)
            self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables,
                             {factor.storage for factor in captured})

    def test_capture_restores_expression_state_on_success_and_failure(self) -> None:
        converter = self.converter('int n = 2;')
        caller = converter._QASM2QCXConverter__add_new_temporary_variable(qasm2qcx.ValueType.INT)
        fields = ('__expression_kind', '__value', '__value_type', '__value_kind', '__evaluate_constant')
        converter._QASM2QCXConverter__evaluate_constant = False
        previous = [getattr(converter, '_QASM2QCXConverter' + field) for field in fields]
        info = self.analyze(converter, 'pow(n + 1)')
        with converter._QASM2QCXConverter__captured_gate_exponents(info.runtime_exponents) as captured:
            self.assertEqual([getattr(converter, '_QASM2QCXConverter' + field) for field in fields], previous)
            self.assertNotEqual(captured[0].storage, caller)
        bad = qasm2qcx._RuntimeGateExponent(qasm2qcx.ast.Identifier('missing'), qasm2qcx.ValueType.INT)
        with self.assertRaises(qasm2qcx.QASM2QCXError):
            with converter._QASM2QCXConverter__captured_gate_exponents(info.runtime_exponents + (bad,)):
                self.fail('Failed capture must not yield')
        self.assertEqual([getattr(converter, '_QASM2QCXConverter' + field) for field in fields], previous)
        self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, {caller})

    def test_capture_cleans_up_on_body_failure_and_preserves_enclosing_captures(self) -> None:
        converter = self.converter('int n = 2;')
        info = self.analyze(converter, 'pow(n)')
        with converter._QASM2QCXConverter__captured_gate_exponents(info.runtime_exponents) as outer:
            with self.assertRaisesRegex(RuntimeError, 'emission failure'):
                with converter._QASM2QCXConverter__captured_gate_exponents(info.runtime_exponents) as inner:
                    self.assertNotEqual(outer[0].storage, inner[0].storage)
                    raise RuntimeError('emission failure')
            self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, {outer[0].storage})
        self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())

    def test_unrolled_iterator_captures_reject_out_of_native_range_values(self) -> None:
        converter = self.converter()
        for value in (converter.QCX_INT_MIN - 1, converter.QCX_INT_MAX + 1):
            with self.subTest(value=value), converter._QASM2QCXConverter__iterator_binding('i', value):
                info = self.analyze(converter, 'pow(i)')
                with self.assertRaisesRegex(qasm2qcx.UnsupportedOpenQASMError, 'native integer range'):
                    with converter._QASM2QCXConverter__captured_gate_exponents(info.runtime_exponents):
                        self.fail('Out-of-range capture must not yield')
                self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())

    def test_capture_type_failure_releases_partial_expression_storage(self) -> None:
        converter = self.converter('int n = 2;')
        expression = self.modifiers('pow(n + 1.0)')[0].argument
        exponent = qasm2qcx._RuntimeGateExponent(expression, qasm2qcx.ValueType.INT)
        with self.assertRaises(qasm2qcx.UnsupportedOpenQASMError):
            with converter._QASM2QCXConverter__captured_gate_exponents((exponent,)):
                self.fail('Mismatched capture type must not yield')
        self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())
        self.assertIsNone(converter._QASM2QCXConverter__expression_kind)

    def test_empty_capture_context_emits_nothing(self) -> None:
        converter = self.converter()
        before = converter._QASM2QCXConverter__qcx_lines.copy()
        with converter._QASM2QCXConverter__captured_gate_exponents(()) as captured:
            self.assertEqual(captured, ())
        self.assertEqual(converter._QASM2QCXConverter__qcx_lines, before)
        self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())

    def test_runtime_conversion_accepts_single_factors_and_validates_empty_loops(self) -> None:
        for call in ('pow(n) @ x q;', 'pow(n) @ gphase(0.5);', 'pow(0) @ pow(n) @ x q;'):
            for wrapper in ('{}', 'for int i in [1:0] {{ {} }}'):
                with self.subTest(call=call, wrapper=wrapper):
                    convert('OPENQASM 3; include "stdgates.inc"; qubit q; int n = 2; ' + wrapper.format(call))


class ControlModifierFoundationTests(unittest.TestCase):
    @staticmethod
    def control_count(converter, modifiers, iterators=None):
        return sum(count for _, count in converter._QASM2QCXConverter__control_modifiers(modifiers, iterators))

    @staticmethod
    def converter(source: str = ''):
        program = qasm2qcx.openqasm3.parser.parse('OPENQASM 3; ' + source)
        converter = qasm2qcx.QASM2QCXConverter(program)
        converter.visit(program)
        return converter

    @staticmethod
    def modifiers(text: str):
        return qasm2qcx.openqasm3.parser.parse('OPENQASM 3; ' + text + ' @ x q;').statements[0].modifiers

    def test_default_explicit_and_repeated_control_counts(self) -> None:
        converter = self.converter()
        helper = lambda modifiers: self.control_count(converter, modifiers)
        self.assertEqual(helper([]), 0)
        for modifiers, count in (('ctrl', 1), ('ctrl(2)', 2), ('ctrl @ ctrl(2)', 3),
                                 ('ctrl(2) @ ctrl(3) @ ctrl', 6)):
            with self.subTest(modifiers=modifiers):
                parsed = self.modifiers(modifiers)
                original = parsed.copy()
                self.assertEqual(helper(parsed), count)
                self.assertEqual(parsed, original)

    def test_count_accepts_constant_integer_expressions_and_casts(self) -> None:
        converter = self.converter('const int N = 2; const uint M = 3;')
        for expression, expected in (('N', 2), ('N + M', 5), ('N * 2 - 1', 3),
                                     ('int(2.5)', 2), ('int(true)', 1), ('uint[2](5)', 1), ('2 ** 3', 8)):
            with self.subTest(expression=expression):
                self.assertEqual(self.control_count(converter,
                    self.modifiers(f'ctrl({expression})')), expected)

    def test_invalid_count_types_values_and_runtime_expressions_rejected(self) -> None:
        converter = self.converter('int n = 2; uint u = 2; float f = 2; bool flag = true; '
            'array[int, 1] values = {2};')
        for expression in ('0', '-1', 'true', '2.0', '1.0im', 'n', 'u', 'f', 'flag', 'values[0]',
                           'missing', 'uint(-1)', str(qasm2qcx.QASM2QCXConverter.QCX_INT_MAX + 1)):
            with self.subTest(expression=expression), self.assertRaises(qasm2qcx.QASM2QCXError):
                self.control_count(converter, self.modifiers(f'ctrl({expression})'))

    def test_control_counts_reject_all_loop_iterator_dependencies(self) -> None:
        converter = self.converter('const int N = 2;')
        for value_type in qasm2qcx.ValueType:
            with self.subTest(value_type=value_type), self.assertRaises(qasm2qcx.UnsupportedOpenQASMError):
                self.control_count(converter, self.modifiers('ctrl(N)'), {'N': value_type})
        with converter._QASM2QCXConverter__iterator_binding('N', 2):
            with self.assertRaises(qasm2qcx.UnsupportedOpenQASMError):
                self.control_count(converter, self.modifiers('ctrl(N)'))

    def test_mixed_and_unsupported_modifier_kinds_rejected_without_evaluating_arguments(self) -> None:
        converter = self.converter()
        for modifiers in ('inv @ ctrl', 'ctrl @ inv @ inv',
                          'pow(1 / 0) @ ctrl', 'ctrl @ pow(1 / 0)', 'ctrl(1 / 0) @ inv'):
            with self.subTest(modifiers=modifiers), self.assertRaises(qasm2qcx.UnsupportedOpenQASMError):
                self.control_count(converter, self.modifiers(modifiers))

    def test_forward_constant_references_rejected_before_declaration_is_visited(self) -> None:
        program = qasm2qcx.openqasm3.parser.parse('OPENQASM 3; const int N = 2;')
        converter = qasm2qcx.QASM2QCXConverter(program)
        with self.assertRaises(qasm2qcx.NoVariableNameException):
            self.control_count(converter, self.modifiers('ctrl(N)'))

    def test_count_limits_and_repeated_count_overflow(self) -> None:
        converter = self.converter()
        maximum = converter.QCX_INT_MAX
        self.assertEqual(self.control_count(converter,
            self.modifiers(f'ctrl({maximum})')), maximum)
        with self.assertRaisesRegex(qasm2qcx.UnsupportedOpenQASMError, 'total control count'):
            self.control_count(converter, self.modifiers(f'ctrl({maximum}) @ ctrl'))

    def test_count_evaluation_preserves_expression_state_and_emits_no_code(self) -> None:
        converter = self.converter('const int N = 2;')
        fields = ('__expression_kind', '__value', '__value_type', '__value_kind', '__evaluate_constant')
        converter._QASM2QCXConverter__evaluate_constant = False
        previous = [getattr(converter, '_QASM2QCXConverter' + name) for name in fields]
        lines = converter._QASM2QCXConverter__qcx_lines.copy()
        self.assertEqual(self.control_count(converter, self.modifiers('ctrl(N + 1)')), 3)
        with self.assertRaises(qasm2qcx.QASM2QCXError):
            self.control_count(converter, self.modifiers('ctrl(1 / 0)'))
        self.assertEqual([getattr(converter, '_QASM2QCXConverter' + name) for name in fields], previous)
        self.assertEqual(converter._QASM2QCXConverter__qcx_lines, lines)
        self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())

    def test_control_mapping_covers_all_existing_gate_and_adjoint_lowerings(self) -> None:
        converter = qasm2qcx.QASM2QCXConverter
        self.assertEqual(set(converter.qcx_controlled_gate_families), set(converter.qcx_adjoint_gate_map))
        for gate, (root, existing) in converter.qcx_controlled_gate_families.items():
            for added in (1, 2, 4):
                with self.subTest(gate=gate, added=added):
                    count = existing + added
                    expected = ('C' if count == 1 else f'C{count}') + root
                    self.assertEqual(converter._QASM2QCXConverter__controlled_gate_name(gate, added), expected)
        for gate, expected in (('I', 'CI1'), ('CX', 'C2X'), ('CCX', 'C3X'), ('CU3', 'C2U3'),
                               ('CSWAP', 'C2SWAP'), ('S+', 'CS+')):
            self.assertEqual(converter._QASM2QCXConverter__controlled_gate_name(gate, 1), expected)

    def test_controlled_phase_uses_only_the_condition_qubits(self) -> None:
        helper = qasm2qcx.QASM2QCXConverter._QASM2QCXConverter__controlled_phase_gate_name
        for count, expected in ((1, 'U1'), (2, 'CU1'), (3, 'C2U1'), (5, 'C4U1')):
            self.assertEqual(helper(count), expected)

    def test_invalid_mappings_and_counts_rejected_without_large_mnemonic_allocation(self) -> None:
        converter = qasm2qcx.QASM2QCXConverter
        maximum = converter.QCX_INT_MAX
        self.assertEqual(converter._QASM2QCXConverter__controlled_gate_name('X', maximum), f'C{maximum}X')
        for count in (0, -1, maximum + 1):
            with self.subTest(count=count), self.assertRaises(qasm2qcx.UnsupportedOpenQASMError):
                converter._QASM2QCXConverter__controlled_gate_name('X', count)
            with self.assertRaises(qasm2qcx.UnsupportedOpenQASMError):
                converter._QASM2QCXConverter__controlled_phase_gate_name(count)
        for gate, count in (('PHASE', 1), ('RESET', 1), ('UNKNOWN', 1), ('CX', maximum), ('CCX', maximum)):
            with self.subTest(gate=gate, count=count), self.assertRaises(qasm2qcx.UnsupportedOpenQASMError):
                converter._QASM2QCXConverter__controlled_gate_name(gate, count)


class ControlModifierLoweringTests(unittest.TestCase):
    def assert_clean_conversion(self, source: str):
        converter = ControlModifierFoundationTests.converter(source)
        self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())
        self.assertEqual(converter._QASM2QCXConverter__loop_bindings, [])
        self.assertEqual(converter._QASM2QCXConverter__loop_contexts, [])
        ForLoopControlLoweringTests.assert_resolved_jumps(converter._QASM2QCXConverter__qcx_lines)
        return converter._QASM2QCXConverter__qcx_lines

    def test_controlled_gate_mnemonics_and_operand_order(self) -> None:
        for source, expected in (
                ('ctrl @ h q[0], q[1];', 'CH 0 1'),
                ('ctrl @ id q[0], q[1];', 'CI1 0 1'),
                ('ctrl @ sdg q[0], q[1];', 'CS+ 0 1'),
                ('ctrl(2) @ sx q[0], q[1], q[2];', 'C2SX 0 1 2'),
                ('ctrl @ cx q[0], q[1], q[2];', 'C2X 0 1 2'),
                ('ctrl @ ccx q[0], q[1], q[2], q[3];', 'C3X 0 1 2 3'),
                ('ctrl @ cswap q[0], q[1], q[2], q[3];', 'C2SWAP 0 1 2 3'),
                ('ctrl @ u2(1, 2) q[0], q[1];', 'CU2 0 1 1 2'),
                ('ctrl @ crx(2) q[0], q[1], q[2];', 'C2EX 0 1 2 -1.0')):
            with self.subTest(source=source):
                self.assertEqual(self.assert_clean_conversion('include "stdgates.inc"; qubit[4] q; ' + source),
                                 ['QUBITS 4', expected])

    def test_constant_and_repeated_controls_change_arity_and_preserve_source_scope(self) -> None:
        lines = self.assert_clean_conversion('include "stdgates.inc"; const int N = 2; qubit[4] q; '
            'ctrl(N) @ ctrl @ x q[0], q[1], q[2], q[3];')
        self.assertEqual(lines, ['QUBITS 4', 'C3X 0 1 2 3'])
        with self.assertRaises(qasm2qcx.QASM2QCXError):
            convert('OPENQASM 3; include "stdgates.inc"; qubit[3] q; '
                    'ctrl(N) @ x q[0], q[1], q[2]; const int N = 2;')

    def test_builtin_u_conditions_global_phase_on_only_the_added_controls(self) -> None:
        self.assertEqual(self.assert_clean_conversion('qubit[3] q; ctrl(2) @ U(1, 2, 3) q[0], q[1], q[2];'),
                         ['QUBITS 3', 'CU1 0 1 0.5', 'C2U3 0 1 2 1 2 3'])

    def test_cu_conditions_control_phase_on_added_and_original_controls(self) -> None:
        self.assertEqual(self.assert_clean_conversion('include "stdgates.inc"; qubit[4] q; '
            'ctrl(2) @ cu(1, 2, 3, 4) q[0], q[1], q[2], q[3];'),
                         ['QUBITS 4', 'C2U1 0 1 2 4.5', 'C3U3 0 1 2 3 1 2 3'])

    def test_controlled_global_phase_uses_no_additional_target(self) -> None:
        for controls, operands, expected in ((1, 'q[0]', 'U1 0 1'),
                                              (2, 'q[0], q[1]', 'CU1 0 1 1'),
                                              (3, 'q[0], q[1], q[2]', 'C2U1 0 1 2 1')):
            with self.subTest(controls=controls):
                self.assertEqual(self.assert_clean_conversion(f'qubit[3] q; ctrl({controls}) @ gphase(1) {operands};'),
                                 ['QUBITS 3', expected])

    def test_runtime_parameters_are_evaluated_once_and_retained_through_broadcast(self) -> None:
        lines = self.assert_clean_conversion('qubit c; qubit[2] targets; array[float, 1] values = {0.2}; '
            'int i = 0; ctrl @ U(values[i], 2, 3) c, targets;')
        self.assertEqual(sum(line.endswith(' := I1') for line in lines), 1)
        operations = [line for line in lines if line.startswith(('U1 ', 'CU3 '))]
        self.assertEqual(len(operations), 4)
        self.assertEqual(operations[0], operations[2])
        self.assertTrue(operations[0].startswith('U1 0 QASM2QCX_REAL_'))
        self.assertTrue(operations[1].startswith('CU3 0 1 '))
        self.assertTrue(operations[3].startswith('CU3 0 2 '))

    def test_control_register_broadcast_and_global_phase_broadcast(self) -> None:
        self.assertEqual(self.assert_clean_conversion('include "stdgates.inc"; qubit[2] c; qubit target; '
            'ctrl @ x c, target; ctrl @ gphase(1) c;'),
                         ['QUBITS 3', 'CX 0 2', 'CX 1 2', 'U1 0 1', 'U1 1 1'])
        with self.assertRaises(qasm2qcx.WrongBroadcastingException):
            convert('OPENQASM 3; qubit[2] c; qubit[3] d; ctrl(2) @ gphase(1) c, d;')

    def test_arity_and_duplicate_qubits_are_rejected(self) -> None:
        for statement in ('ctrl @ x q[0];', 'ctrl(2) @ x q[0], q[1];',
                          'ctrl @ gphase(1);', 'ctrl(2) @ gphase(1) q[0];',
                          'ctrl @ x q[0], q[0];', 'ctrl @ cx q[0], q[1], q[0];',
                          'ctrl(2) @ gphase(1) q[0], q[0];'):
            with self.subTest(statement=statement), self.assertRaises(qasm2qcx.QASM2QCXError):
                convert('OPENQASM 3; include "stdgates.inc"; qubit[3] q; ' + statement)

    def test_invalid_counts_and_modifier_combinations_rejected_even_in_empty_loops(self) -> None:
        for modifiers in ('ctrl(0)', 'ctrl(-1)', 'ctrl(true)', 'ctrl(2.0)', 'ctrl(n)',
                          'negctrl(0)', 'inv @ negctrl(n)', 'ctrl @ pow(1 / 0)'):
            for wrapper in ('{}', 'for int i in [1:0] {{ {} }}'):
                with self.subTest(modifiers=modifiers, wrapper=wrapper), self.assertRaises(qasm2qcx.QASM2QCXError):
                    convert('OPENQASM 3; include "stdgates.inc"; int n = 1; qubit[2] q; '
                            + wrapper.format(f'{modifiers} @ x q[0], q[1];'))
        for count in ('i', 'int(i)'):
            with self.assertRaises(qasm2qcx.UnsupportedOpenQASMError):
                convert('OPENQASM 3; include "stdgates.inc"; qubit[2] q; '
                        f'for int i in [1:1] {{ ctrl({count}) @ x q[0], q[1]; }}')

    def test_controlled_gates_work_in_nested_runtime_and_empty_loops(self) -> None:
        self.assert_clean_conversion('include "stdgates.inc"; qubit[2] q; int stop = 1; '
            'const int N = 1; float theta = 0.3; for int i in [0:stop] { '
            'ctrl(N) @ rx(theta + 1) q[0], q[1]; ctrl @ gphase(theta) q[0]; '
            'for int j in [1:0] { ctrl @ U(theta, 2, 3) q[0], q[1]; } }')

    def test_failed_parameter_or_operand_lowering_releases_only_gate_owned_temporaries(self) -> None:
        for statement in ('ctrl @ U(theta + 1, missing, 0) q[0], q[1];',
                          'ctrl @ U(theta + 1, true, 0) q[0], q[1];',
                          'ctrl @ U(theta + 1, 2, 3) q[0], q[0];',
                          'ctrl @ gphase(theta + 1) missing;', 'ctrl @ gphase(complex(theta)) q[0];'):
            converter = ControlModifierFoundationTests.converter('qubit[2] q; float theta = 0.2;')
            caller = converter._QASM2QCXConverter__add_new_temporary_variable(qasm2qcx.ValueType.FLOAT)
            child = qasm2qcx.openqasm3.parser.parse('OPENQASM 3; ' + statement).statements[0]
            with self.subTest(statement=statement), self.assertRaises(qasm2qcx.QASM2QCXError):
                converter.visit(child)
            self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, {caller})
            self.assertIsNone(converter._QASM2QCXConverter__expression_kind)


class ControlModifierIntegrationTests(unittest.TestCase):
    assert_clean_conversion = ControlModifierLoweringTests.assert_clean_conversion

    def test_runtime_cu_parameters_and_control_phase_have_distinct_snapshots(self) -> None:
        lines = self.assert_clean_conversion('include "stdgates.inc"; qubit[4] q; '
            'array[float, 4] angles = {0.2, 0.4, 0.6, 0.8}; int i = 0; '
            'ctrl(2) @ cu(angles[i], angles[i + 1], angles[i + 2], angles[i + 3]) q[0], q[1], q[2], q[3];')
        phase = next(line for line in lines if line.startswith('C2U1 ')).split()
        gate = next(line for line in lines if line.startswith('C3U3 ')).split()
        self.assertEqual(phase[1:4], ['0', '1', '2'])
        self.assertEqual(gate[1:5], ['0', '1', '2', '3'])
        self.assertEqual(len(set(gate[5:])), 3)
        self.assertNotIn(phase[-1], gate[5:])
        self.assertEqual(sum(line.endswith(' := I1') for line in lines), 4)
        self.assertEqual(sum(line.startswith('ASSERT ') for line in lines), 8)

    def test_runtime_loops_hoist_controlled_parameter_and_phase_temporaries(self) -> None:
        for header in ('for int i in [0:stop]', 'for int i in {0, stop}', 'while (false)'):
            with self.subTest(header=header):
                lines = self.assert_clean_conversion('include "stdgates.inc"; qubit[3] q; '
                    'float theta = 0.2; int stop = 1; ' + header + '{ '
                    'ctrl @ U(theta + 1, theta * 2, 3) q[0], q[1]; '
                    'ctrl @ cu(theta, 1, 2, theta + 3) q[0], q[1], q[2]; '
                    'ctrl(2) @ gphase(theta + 1) q[0], q[1]; }')
                first_label = next(i for i, line in enumerate(lines) if line.startswith('@'))
                self.assertFalse(any(line.startswith('VAR QASM2QCX_') for line in lines[first_label:]))
                self.assertTrue(any(line.startswith('C2U3 ') for line in lines))

    def test_count_shadowing_rejected_even_in_empty_nested_loops(self) -> None:
        for nested in ('for int N in [1:0]', 'for int N in [0:stop]', 'for float N in values'):
            with self.subTest(nested=nested), self.assertRaisesRegex(
                    qasm2qcx.UnsupportedOpenQASMError, 'loop iterator'):
                convert('OPENQASM 3; const int N = 1; int stop = 1; array[float, 1] values = {1}; qubit q; '
                        'for int outer in [1:0] { ' + nested + ' { ctrl(N) @ gphase(1) q; } }')

    def test_global_phase_counts_and_arity_checked_inside_empty_loops(self) -> None:
        for statement in ('ctrl(2) @ gphase(1) q;', 'ctrl @ gphase(1);',
                          'ctrl(0) @ gphase(1) q;', 'ctrl(2.0) @ gphase(1) q;',
                          'ctrl @ inv @ negctrl @ gphase(1) q;', 'ctrl(n) @ gphase(1) q;'):
            with self.subTest(statement=statement), self.assertRaises(qasm2qcx.QASM2QCXError):
                convert('OPENQASM 3; int n = 1; qubit q; for int outer in [1:0] { '
                        + statement + ' }')

    def test_inactive_quantum_controls_do_not_suppress_runtime_parameter_checks(self) -> None:
        lines = self.assert_clean_conversion('include "stdgates.inc"; qubit[2] q; '
            'array[float, 1] angles; int bad = 1; ctrl @ rx(angles[bad]) q[0], q[1];')
        assertion = next(i for i, line in enumerate(lines) if line.startswith('ASSERT '))
        gate = next(i for i, line in enumerate(lines) if line.startswith('CEX '))
        self.assertLess(assertion, gate)

    def test_successful_gate_and_phase_preserve_caller_owned_storage(self) -> None:
        for source in ('ctrl @ U(theta, theta + 1, theta * 2) q[0], q[1];',
                       'ctrl(2) @ gphase(theta + 1) q[0], q[1];',
                       'negctrl @ inv @ U(theta, theta + 1, theta * 2) q[0], q[1];',
                       'ctrl @ inv @ negctrl @ gphase(theta + 1) q[0], q[1];'):
            converter = ControlModifierFoundationTests.converter('qubit[2] q; float theta = 0.2;')
            caller = converter._QASM2QCXConverter__add_new_temporary_variable(qasm2qcx.ValueType.FLOAT)
            child = qasm2qcx.openqasm3.parser.parse('OPENQASM 3; ' + source).statements[0]
            start = len(converter._QASM2QCXConverter__qcx_lines)
            with self.subTest(source=source):
                converter.visit(child)
                self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, {caller})
                self.assertFalse(any(line.startswith(f'LET {caller} ') for line in
                                     converter._QASM2QCXConverter__qcx_lines[start:]))

    def test_cli_control_errors_emit_no_partial_output(self) -> None:
        for instruction in ('ctrl(n) @ x q[0], q[1];', 'ctrl @ x q[0], q[0];',
                            'ctrl @ pow(1 / 0) @ gphase(1) q[0];'):
            source = 'OPENQASM 3; include "stdgates.inc"; int n = 1; qubit[2] q; h q; ' + instruction
            with self.subTest(instruction=instruction), patch('builtins.open', mock_open(read_data=source)), \
                    patch('builtins.print') as output:
                with self.assertRaises(SystemExit):
                    qasm2qcx.main(['control.qasm'])
                output.assert_not_called()


class NegativeControlModifierTests(unittest.TestCase):
    assert_clean_conversion = ControlModifierLoweringTests.assert_clean_conversion

    def test_modifier_groups_preserve_source_order_without_expansion(self) -> None:
        converter = ControlModifierFoundationTests.converter('const int N = 2;')
        for modifiers, expected in (
                ('negctrl', (False, [(0, 1)])),
                ('negctrl(N) @ ctrl @ negctrl', (False, [(0, 2), (1, 1), (0, 1)])),
                ('inv @ ctrl @ inv @ negctrl(2) @ inv', (True, [(1, 1), (0, 2)])),
                (f'negctrl({converter.QCX_INT_MAX})', (False, [(0, converter.QCX_INT_MAX)]))):
            parsed = ControlModifierFoundationTests.modifiers(modifiers)
            original = parsed.copy()
            with self.subTest(modifiers=modifiers):
                self.assertEqual(converter._QASM2QCXConverter__analyze_gate_modifiers(parsed),
                                 qasm2qcx._GateModifierInfo(-1 if expected[0] else 1, tuple(expected[1])))
                self.assertEqual(parsed, original)
        with self.assertRaisesRegex(qasm2qcx.UnsupportedOpenQASMError, 'total control count'):
            converter._QASM2QCXConverter__analyze_gate_modifiers(
                ControlModifierFoundationTests.modifiers(f'ctrl({converter.QCX_INT_MAX}) @ negctrl'))

    def test_negative_and_mixed_controls_lower_natively_in_operand_order(self) -> None:
        for source, expected in (
                ('negctrl @ h q[4], q[1];', ['CH 4#0 1']),
                ('negctrl(2) @ x q[4], q[0], q[2];', ['C2X 4#0 0#0 2']),
                ('ctrl @ negctrl @ x q[4], q[0], q[2];', ['C2X 4 0#0 2']),
                ('negctrl @ ctrl @ x q[4], q[0], q[2];', ['C2X 4#0 0 2']),
                ('ctrl @ negctrl(2) @ inv @ ctrl @ x q[0], q[1], q[2], q[3], q[4];',
                 ['C4X 0 1#0 2#0 3 4']),
                ('negctrl @ cx q[4], q[0], q[2];', ['C2X 4#0 0 2']),
                ('negctrl @ ccx q[4], q[0], q[3], q[2];', ['C3X 4#0 0 3 2']),
                ('negctrl @ cswap q[4], q[0], q[3], q[2];', ['C2SWAP 4#0 0 3 2']),
                ('negctrl @ inv @ sdg q[4], q[1];', ['CS 4#0 1']),
                ('inv @ negctrl @ inv @ sx q[4], q[1];', ['CSX 4#0 1']),
                ('negctrl @ id q[4], q[1];', ['CI1 4#0 1'])):
            with self.subTest(source=source):
                self.assertEqual(self.assert_clean_conversion('include "stdgates.inc"; qubit[5] q; ' + source),
                                 ['QUBITS 5', *expected])

    def test_inverted_builtin_u_and_cu_keep_conditioned_phase_corrections(self) -> None:
        self.assertEqual(self.assert_clean_conversion('qubit[2] q; ctrl @ inv @ U(1, 2, 3) q[0], q[1];'),
                         ['QUBITS 2', 'CU3+ 0 1 1 2 3', 'U1 0 -0.5'])
        for modifiers in ('inv @ negctrl', 'negctrl @ inv', 'inv @ inv @ negctrl @ inv'):
            with self.subTest(modifiers=modifiers):
                self.assertEqual(self.assert_clean_conversion('qubit[2] q; '
                    + modifiers + ' @ U(1, 2, 3) q[0], q[1];'),
                    ['QUBITS 2', 'CU3+ 0#0 1 1 2 3', 'U1 0#0 -0.5'])
                self.assertEqual(self.assert_clean_conversion('include "stdgates.inc"; qubit[3] q; '
                    + modifiers + ' @ cu(1, 2, 3, 4) q[0], q[1], q[2];'),
                    ['QUBITS 3', 'C2U3+ 0#0 1 2 1 2 3', 'CU1+ 0#0 1 4.5'])

    def test_negative_global_phase_and_broadcasting(self) -> None:
        self.assertEqual(self.assert_clean_conversion('qubit[2] c; qubit t; '
            'negctrl @ U(1, 2, 3) c, t; negctrl @ inv @ gphase(2) c;'),
            ['QUBITS 3', 'U1 0#0 0.5', 'CU3 0#0 2 1 2 3',
             'U1 1#0 0.5', 'CU3 1#0 2 1 2 3', 'U1 0#0 -2.0', 'U1 1#0 -2.0'])
        self.assertEqual(self.assert_clean_conversion('qubit[3] q; '
            'negctrl @ ctrl @ inv @ gphase(2) q[2], q[0];'),
            ['QUBITS 3', 'CU1 2#0 0 -2.0'])

    def test_invalid_counts_types_and_shadowing_rejected_even_in_empty_loops(self) -> None:
        for expression in ('0', '-1', 'true', '2.0', '1.0im', 'n', 'u', 'values[0]',
                           'i', 'int(i)', 'missing', 'uint(-1)', str(qasm2qcx.QASM2QCXConverter.QCX_INT_MAX + 1)):
            for wrapper in ('{}', 'for int i in [1:0] {{ {} }}'):
                with self.subTest(expression=expression, wrapper=wrapper), self.assertRaises(qasm2qcx.QASM2QCXError):
                    convert('OPENQASM 3; include "stdgates.inc"; qubit[2] q; int n = 1; uint u = 1; '
                            'array[int, 1] values; ' + wrapper.format(f'negctrl({expression}) @ x q[0], q[1];'))
        with self.assertRaisesRegex(qasm2qcx.UnsupportedOpenQASMError, 'loop iterator'):
            convert('OPENQASM 3; const int N = 1; qubit q; for int N in [1:0] { '
                    'negctrl(N) @ gphase(1) q; }')

    def test_valid_constant_count_expressions(self) -> None:
        for expression in ('N', 'N + 0', 'int(1.5)', 'uint[2](5)'):
            with self.subTest(expression=expression):
                self.assertEqual(self.assert_clean_conversion('include "stdgates.inc"; const uint N = 1; qubit[2] q; '
                    f'negctrl({expression}) @ x q[0], q[1];'), ['QUBITS 2', 'CX 0#0 1'])

    def test_invalid_modifier_arguments_report_their_errors(self) -> None:
        converter = ControlModifierFoundationTests.converter()
        for text in ('negctrl(1 / 0) @ pow(1 / 0)', 'inv @ negctrl @ pow(1 / 0) @ inv'):
            with self.subTest(text=text), self.assertRaises(qasm2qcx.ZeroDivisorException):
                converter._QASM2QCXConverter__analyze_gate_modifiers(ControlModifierFoundationTests.modifiers(text))
        modifiers = ControlModifierFoundationTests.modifiers('negctrl @ inv')
        modifiers[1].argument = qasm2qcx.ast.IntegerLiteral(2)
        with self.assertRaisesRegex(qasm2qcx.UnsupportedOpenQASMError, 'inv modifier with an argument'):
            converter._QASM2QCXConverter__analyze_gate_modifiers(modifiers)

    def test_arity_duplicates_and_broadcast_mismatches(self) -> None:
        for statement in ('negctrl @ x q[0];', 'negctrl(2) @ x q[0], q[1];',
                          'negctrl @ ctrl @ gphase(1) q[0];',
                          'negctrl @ x q[0], q[0];', 'ctrl @ negctrl @ x q[0], q[0], q[1];',
                          'negctrl(2) @ gphase(1) q[0], q[0];'):
            with self.subTest(statement=statement), self.assertRaises(qasm2qcx.QASM2QCXError):
                convert('OPENQASM 3; include "stdgates.inc"; qubit[3] q; ' + statement)
        with self.assertRaises(qasm2qcx.WrongBroadcastingException):
            convert('OPENQASM 3; qubit[2] a; qubit[3] b; negctrl @ ctrl @ gphase(1) a, b;')

    def test_runtime_parameter_storage_and_empty_loop_validation(self) -> None:
        for header in ('for int i in [0:stop]', 'for int i in {0, stop}', 'while (false)',
                       'for int i in [1:0]'):
            with self.subTest(header=header):
                lines = self.assert_clean_conversion('include "stdgates.inc"; qubit[4] q; '
                    'array[float, 4] angles = {0.2, 0.4, 0.6, 0.8}; int stop = 1; int i = 0; '
                    + header + '{ negctrl @ inv @ U(angles[i], 2, 3) q[3], q[1]; '
                    'ctrl @ negctrl @ inv @ cu(angles[i], 1, 2, 3) q[3], q[0], q[2], q[1]; '
                    'negctrl @ inv @ gphase(angles[i]) q[2]; }')
                first_label = next((i for i, line in enumerate(lines) if line.startswith('@')), len(lines))
                self.assertFalse(any(line.startswith('VAR QASM2QCX_') for line in lines[first_label:]))

    def test_failure_preserves_caller_temporaries_and_cli_has_no_partial_output(self) -> None:
        converter = ControlModifierFoundationTests.converter('qubit[2] q; float theta = 0.2;')
        caller = converter._QASM2QCXConverter__add_new_temporary_variable(qasm2qcx.ValueType.FLOAT)
        child = qasm2qcx.openqasm3.parser.parse('OPENQASM 3; '
            'negctrl @ inv @ U(theta + 1, 2, 3) q[0], q[0];').statements[0]
        with self.assertRaises(qasm2qcx.QASM2QCXError):
            converter.visit(child)
        self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, {caller})
        self.assertIsNone(converter._QASM2QCXConverter__expression_kind)
        source = 'OPENQASM 3; qubit q; U(0, 0, 0) q; negctrl @ pow(1 / 0) @ gphase(1) q;'
        with patch('builtins.open', mock_open(read_data=source)), patch('builtins.print') as output:
            with self.assertRaises(SystemExit):
                qasm2qcx.main(['negative.qasm'])
            output.assert_not_called()


class InverseModifierFoundationTests(unittest.TestCase):
    def test_mapping_covers_all_existing_gate_lowerings_and_is_involutive(self) -> None:
        converter = qasm2qcx.QASM2QCXConverter
        mappings = converter.qcx_adjoint_gate_map
        forward = set(converter.default_gates_qcx_map.values()) | set(converter.stdgates_qcx_map.values())
        self.assertTrue(forward <= mappings.keys())
        for gate, adjoint in mappings.items():
            with self.subTest(gate=gate):
                self.assertEqual(converter._QASM2QCXConverter__adjoint_gate_name(gate), adjoint)
                self.assertEqual(converter._QASM2QCXConverter__adjoint_gate_name(adjoint), gate)

    def test_self_adjoint_gates_keep_accepted_qcx_spelling(self) -> None:
        for gate in ('I', 'H', 'X', 'Y', 'Z', 'CX', 'CY', 'CZ', 'CH', 'SWAP', 'CCX', 'CSWAP'):
            with self.subTest(gate=gate):
                self.assertEqual(qasm2qcx.QASM2QCXConverter._QASM2QCXConverter__adjoint_gate_name(gate), gate)

    def test_nonselfadjoint_gates_toggle_existing_adjoint_mnemonics(self) -> None:
        for gate in ('S', 'T', 'SX', 'U1', 'U2', 'U3', 'EX', 'EY', 'EZ',
                     'CU1', 'CU3', 'CEX', 'CEY', 'CEZ'):
            with self.subTest(gate=gate):
                helper = qasm2qcx.QASM2QCXConverter._QASM2QCXConverter__adjoint_gate_name
                self.assertEqual(helper(gate), gate + '+')
                self.assertEqual(helper(gate + '+'), gate)

    def test_unknown_instructions_and_phase_require_explicit_handling(self) -> None:
        for gate in ('H+', 'X+', 'PHASE', 'RESET', 'M', 'UNKNOWN'):
            with self.subTest(gate=gate), self.assertRaises(qasm2qcx.UnsupportedOpenQASMError):
                qasm2qcx.QASM2QCXConverter._QASM2QCXConverter__adjoint_gate_name(gate)

    def test_inverse_parity_for_gates_and_global_phase(self) -> None:
        helper = qasm2qcx.QASM2QCXConverter._QASM2QCXConverter__inverse_modifier_parity
        for count in range(7):
            for instruction in ('h q', 'gphase(0.5)'):
                with self.subTest(count=count, instruction=instruction):
                    statement = qasm2qcx.openqasm3.parser.parse(
                        'OPENQASM 3; ' + 'inv @ ' * count + instruction + ';').statements[0]
                    original = statement.modifiers.copy()
                    self.assertEqual(helper(statement.modifiers), count % 2 == 1)
                    self.assertEqual(statement.modifiers, original)

    def test_unsupported_modifiers_rejected_even_among_cancelling_inverses(self) -> None:
        helper = qasm2qcx.QASM2QCXConverter._QASM2QCXConverter__inverse_modifier_parity
        for unsupported in ('ctrl', 'negctrl', 'ctrl(2)', 'negctrl(2)', 'pow(2)', 'pow(1 / 0)'):
            for prefix, suffix in (('', ''), ('inv @ ', ' @ inv'), ('inv @ inv @ ', '')):
                with self.subTest(unsupported=unsupported, prefix=prefix):
                    statement = qasm2qcx.openqasm3.parser.parse(
                        f'OPENQASM 3; {prefix}{unsupported}{suffix} @ x q;').statements[0]
                    with self.assertRaises(qasm2qcx.UnsupportedOpenQASMError):
                        helper(statement.modifiers)

    def test_malformed_inverse_argument_is_rejected_without_evaluation(self) -> None:
        helper = qasm2qcx.QASM2QCXConverter._QASM2QCXConverter__inverse_modifier_parity
        argument = qasm2qcx.ast.BinaryExpression(qasm2qcx.ast.BinaryOperator['/'],
                                               qasm2qcx.ast.IntegerLiteral(1), qasm2qcx.ast.IntegerLiteral(0))
        modifier = qasm2qcx.ast.QuantumGateModifier(qasm2qcx.ast.GateModifierName.inv, argument)
        with self.assertRaisesRegex(qasm2qcx.UnsupportedOpenQASMError, 'inv modifier with an argument'):
            helper([modifier, modifier])


class InverseModifierLoweringTests(unittest.TestCase):
    def assert_clean_conversion(self, source: str):
        program = qasm2qcx.openqasm3.parser.parse('OPENQASM 3; ' + source)
        converter = qasm2qcx.QASM2QCXConverter(program)
        converter.visit(program)
        self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())
        self.assertEqual(converter._QASM2QCXConverter__loop_bindings, [])
        self.assertEqual(converter._QASM2QCXConverter__loop_contexts, [])
        ForLoopControlLoweringTests.assert_resolved_jumps(converter._QASM2QCXConverter__qcx_lines)
        return converter._QASM2QCXConverter__qcx_lines

    def test_all_single_instruction_gates_emit_the_explicit_adjoint(self) -> None:
        converter = qasm2qcx.QASM2QCXConverter
        for gate, (parameters, qubits) in converter.gate_signatures.items():
            if gate in ('U', 'cu'):
                continue
            arguments = '(' + ', '.join(str(i + 1) for i in range(parameters)) + ')' if parameters else ''
            operands = ', '.join(f'q[{i}]' for i in range(qubits))
            prefix = 'OPENQASM 3; include "stdgates.inc"; qubit[3] q; '
            forward = convert(prefix + f'{gate}{arguments} {operands};')
            inverse = convert(prefix + f'inv @ {gate}{arguments} {operands};')
            with self.subTest(gate=gate):
                mnemonic, rest = forward[-1].split(' ', 1)
                self.assertEqual(inverse, forward[:-1] + [converter.qcx_adjoint_gate_map[mnemonic] + ' ' + rest])

    def test_repeated_inverse_parity_preserves_forward_output(self) -> None:
        for gate in ('h q', 'sdg q', 'rx(1) q', 'U(1, 2, 3) q', 'cu(1, 2, 3, 4) q, r', 'gphase(0.5)'):
            prefix = 'OPENQASM 3; include "stdgates.inc"; qubit q; qubit r; '
            for count in (2, 4):
                with self.subTest(gate=gate, count=count):
                    self.assertEqual(convert(prefix + 'inv @ ' * count + gate + ';'), convert(prefix + gate + ';'))
            self.assertEqual(convert(prefix + 'inv @ inv @ inv @ ' + gate + ';'),
                             convert(prefix + 'inv @ ' + gate + ';'))

    def test_builtin_u_inverts_full_sequence_and_global_phase(self) -> None:
        lines = self.assert_clean_conversion('qubit q; inv @ U(1, 2, 3) q;')
        self.assertEqual(lines, ['QUBITS 1', 'U3+ 0 1 2 3', 'PHASE -0.5'])

    def test_cu_inverts_sequence_and_control_phase(self) -> None:
        lines = self.assert_clean_conversion('include "stdgates.inc"; qubit[2] q; '
            'inv @ cu(1, 2, 3, 4) q[0], q[1];')
        self.assertEqual(lines, ['QUBITS 2', 'CU3+ 0 1 1 2 3', 'U1+ 0 4.5'])

    def test_inverse_rotations_keep_existing_half_angle_conversion(self) -> None:
        lines = self.assert_clean_conversion('include "stdgates.inc"; qubit[2] q; float theta = 2; '
            'inv @ rx(theta) q[0]; inv @ crz(theta) q[0], q[1];')
        self.assertIn('EX+ 0 QASM2QCX_REAL_0', lines)
        self.assertIn('CEZ+ 0 1 QASM2QCX_REAL_0', lines)
        self.assertEqual(lines.count('LET QASM2QCX_REAL_0 := THETA31'), 2)
        self.assertEqual(lines.count('LET QASM2QCX_REAL_0 *= -0.5'), 2)

    def test_runtime_phase_negation_uses_private_float_without_mutating_source(self) -> None:
        for declaration, operand in (('float theta = 2;', 'THETA31'),
                                     (f'int theta = {qasm2qcx.QASM2QCXConverter.QCX_INT_MIN};', ':REAL:THETA31'),
                                     ('uint theta = 2;', ':REAL:THETA31')):
            with self.subTest(declaration=declaration):
                lines = self.assert_clean_conversion(declaration + ' inv @ gphase(theta);')
                self.assertIn(f'LET QASM2QCX_REAL_0 := {operand}', lines)
                self.assertIn('LET QASM2QCX_REAL_0 *= -1.0', lines)
                self.assertEqual(lines[-1], 'PHASE QASM2QCX_REAL_0')
                self.assertEqual(sum(line.startswith('LET THETA31 ') for line in lines), 1)

    def test_runtime_u_phase_is_computed_once_before_broadcast_and_reversed_per_element(self) -> None:
        lines = self.assert_clean_conversion('qubit[2] q; float theta = 1; '
            'inv @ U(theta, theta + 1, theta * 2) q;')
        operations = [line for line in lines if line.startswith(('U3+', 'PHASE'))]
        self.assertEqual(len(operations), 4)
        self.assertTrue(operations[0].startswith('U3+ 0 '))
        self.assertTrue(operations[1].startswith('PHASE QASM2QCX_REAL_'))
        self.assertTrue(operations[2].startswith('U3+ 1 '))
        self.assertEqual(operations[1], operations[3])
        self.assertEqual(sum(line.endswith(' *= -1.0') for line in lines), 1)

    def test_cu_broadcast_and_runtime_phase_preserve_control_operands(self) -> None:
        lines = self.assert_clean_conversion('include "stdgates.inc"; qubit control; qubit[2] targets; '
            'float theta = 1; inv @ cu(theta, 2, 3, theta + 1) control, targets;')
        operations = [line for line in lines if line.startswith(('CU3+', 'U1+'))]
        self.assertEqual(len(operations), 4)
        self.assertTrue(operations[0].startswith('CU3+ 0 1 '))
        self.assertTrue(operations[2].startswith('CU3+ 0 2 '))
        self.assertEqual(operations[1], operations[3])
        self.assertTrue(operations[1].startswith('U1+ 0 QASM2QCX_REAL_'))

    def test_inverse_supported_in_branches_and_empty_nested_loops(self) -> None:
        self.assert_clean_conversion('include "stdgates.inc"; qubit q; float theta = 1; bool ready = true; '
            'if (ready) { inv @ rx(theta + 1) q; } '
            'for int i in [0:1] { inv @ U(theta, 2, 3) q; inv @ gphase(theta); } '
            'for int i in [1:0] { inv @ h q; for int j in {1, 2} { inv @ gphase(theta); } } '
            'while (false) { inv @ rx(theta) q; }')

    def test_unsupported_modifier_combinations_rejected_even_in_empty_loops(self) -> None:
        for modifier in ('negctrl(0)', 'inv @ pow(1 / 0) @ inv', 'inv @ inv @ pow(1 / 0)'):
            for instruction in ('x q', 'gphase(1)'):
                for wrapper in ('{}', 'for int i in [1:0] {{ {} }}'):
                    with self.subTest(modifier=modifier, instruction=instruction, wrapper=wrapper), self.assertRaises(
                            qasm2qcx.QASM2QCXError):
                        convert('OPENQASM 3; include "stdgates.inc"; qubit q; '
                                + wrapper.format(f'{modifier} @ {instruction};'))

    def test_inverse_still_checks_include_arity_parameter_types_and_phase_qubits(self) -> None:
        for source in ('qubit q; inv @ h q;',
                       'include "stdgates.inc"; qubit q; inv @ rx() q;',
                       'include "stdgates.inc"; qubit q; inv @ rx(true) q;',
                       'qubit q; inv @ gphase(1) q;', 'inv @ gphase(true);'):
            with self.subTest(source=source), self.assertRaises(qasm2qcx.QASM2QCXError):
                convert('OPENQASM 3; ' + source)

    def test_failed_gate_and_phase_parameters_cleanup_without_releasing_caller_storage(self) -> None:
        for statement in ('inv @ U(theta + 1, missing, 0) q;',
                          'inv @ U(theta + 1, true, 0) q;',
                          'inv @ gphase(theta + missing);', 'inv @ gphase(complex(theta + 1));'):
            program = qasm2qcx.openqasm3.parser.parse('OPENQASM 3; qubit q; float theta = 1;')
            converter = qasm2qcx.QASM2QCXConverter(program)
            converter.visit(program)
            caller = converter._QASM2QCXConverter__add_new_temporary_variable(qasm2qcx.ValueType.FLOAT)
            child = qasm2qcx.openqasm3.parser.parse('OPENQASM 3; ' + statement).statements[0]
            with self.subTest(statement=statement), self.assertRaises(qasm2qcx.QASM2QCXError):
                converter.visit(child)
            self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, {caller})
            self.assertIsNone(converter._QASM2QCXConverter__expression_kind)


class InverseModifierIntegrationTests(unittest.TestCase):
    assert_clean_conversion = InverseModifierLoweringTests.assert_clean_conversion

    def test_runtime_indexed_parameters_and_phase_snapshots_do_not_alias(self) -> None:
        lines = self.assert_clean_conversion('qubit q; array[float, 3] values = {0.2, 0.4, 0.6}; int i = 0; '
            'inv @ U(values[i], values[i + 1], values[i + 2]) q;')
        gate = next(line for line in lines if line.startswith('U3+ ')).split()
        phase = next(line for line in lines if line.startswith('PHASE ')).split()[1]
        self.assertEqual(len(set(gate[2:])), 3)
        self.assertNotIn(phase, gate[2:])
        self.assertEqual(sum(line.endswith(' := I1') for line in lines), 3)  # One capture per index expression.
        self.assertEqual(sum(line.startswith('ASSERT ') for line in lines), 6)

    def test_each_loop_kind_preserves_inverse_gates_and_hoists_temporaries(self) -> None:
        declarations = 'include "stdgates.inc"; qubit q; float theta = 0.2; int stop = 1; '
        for header in ('for int i in [0:1]', 'for int i in [0:stop]',
                       'for int i in {0, stop}', 'while (false)'):
            with self.subTest(header=header):
                lines = self.assert_clean_conversion(declarations + header + '{ inv @ rx(theta + 1) q; '
                    'inv @ U(theta, theta + 1, theta * 2) q; inv @ gphase(theta); }')
                self.assertTrue(any(line.startswith('EX+ ') for line in lines))
                self.assertTrue(any(line.startswith('U3+ ') for line in lines))
                first_label = next((i for i, line in enumerate(lines) if line.startswith('@')), len(lines))
                self.assertFalse(any(line.startswith('VAR QASM2QCX_') for line in lines[first_label:]))

    def test_unsupported_modifiers_rejected_inside_nested_empty_collection_loops(self) -> None:
        for instruction in ('negctrl(0) @ x q;', 'pow(1 / 0) @ gphase(1);'):
            with self.subTest(instruction=instruction), self.assertRaises(qasm2qcx.QASM2QCXError):
                convert('OPENQASM 3; include "stdgates.inc"; qubit q; array[float, 1] values; '
                        'for int outer in [1:0] { for float value in values { while (false) { '
                        + instruction + ' } } }')

    def test_successful_inverse_gate_preserves_caller_owned_temporary(self) -> None:
        program = qasm2qcx.openqasm3.parser.parse('OPENQASM 3; qubit q; float theta = 0.2;')
        converter = qasm2qcx.QASM2QCXConverter(program)
        converter.visit(program)
        caller = converter._QASM2QCXConverter__add_new_temporary_variable(qasm2qcx.ValueType.FLOAT)
        statement = qasm2qcx.openqasm3.parser.parse('OPENQASM 3; '
            'inv @ U(theta, theta + 1, theta * 2) q;').statements[0]
        start = len(converter._QASM2QCXConverter__qcx_lines)
        converter.visit(statement)
        self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, {caller})
        self.assertFalse(any(line.startswith(f'LET {caller} ') for line in
                             converter._QASM2QCXConverter__qcx_lines[start:]))

    def test_inverse_duration_stays_unsupported(self) -> None:
        program = qasm2qcx.openqasm3.parser.parse('OPENQASM 3; qubit q;')
        converter = qasm2qcx.QASM2QCXConverter(program)
        converter.visit(program)
        statement = qasm2qcx.openqasm3.parser.parse('OPENQASM 3; inv @ U(0, 0, 0) q;').statements[0]
        statement.duration = qasm2qcx.ast.IntegerLiteral(1)
        with self.assertRaisesRegex(qasm2qcx.UnsupportedOpenQASMError, 'duration'):
            converter.visit(statement)
        self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())

    def test_cli_unsupported_inverse_combination_has_no_partial_output(self) -> None:
        source = 'OPENQASM 3; include "stdgates.inc"; qubit q; h q; inv @ pow(1 / 0) @ inv @ x q;'
        with patch('builtins.open', mock_open(read_data=source)), patch('builtins.print') as output:
            with self.assertRaises(SystemExit) as error:
                qasm2qcx.main(['inverse.qasm'])
        self.assertIn('divisor must not be zero', str(error.exception))
        output.assert_not_called()


class IntegerPowerModifierLoweringTests(unittest.TestCase):
    assert_clean_conversion = InverseModifierLoweringTests.assert_clean_conversion

    def test_unit_and_negative_unit_powers_match_existing_lowering_for_every_gate(self) -> None:
        for gate, (parameters, qubits) in qasm2qcx.QASM2QCXConverter.gate_signatures.items():
            args = '(' + ', '.join(str(i + 1) for i in range(parameters)) + ')' if parameters else ''
            operands = ', '.join(f'q[{i}]' for i in range(qubits))
            prefix = 'OPENQASM 3; include "stdgates.inc"; qubit[3] q; '
            call = f'{gate}{args} {operands};'
            with self.subTest(gate=gate):
                if gate == 'id':
                    self.assertEqual(convert(prefix + 'pow(1) @ ' + call), ['QUBITS 3'])
                    self.assertEqual(convert(prefix + 'pow(-1) @ ' + call), ['QUBITS 3'])
                    continue
                self.assertEqual(convert(prefix + 'pow(1) @ ' + call), convert(prefix + call))
                self.assertEqual(convert(prefix + 'pow(-1) @ ' + call), convert(prefix + 'inv @ ' + call))

    def test_general_positive_and_negative_powers_repeat_complete_lowered_sequences(self) -> None:
        for gate, forward, inverse in (
                ('u3(1, 2, 3) q[0]', ['U3 0 1 2 3'], ['U3+ 0 1 2 3']),
                ('U(1, 2, 3) q[0]', ['PHASE 0.5', 'U3 0 1 2 3'], ['U3+ 0 1 2 3', 'PHASE -0.5']),
                ('cu(1, 2, 3, 4) q[0], q[1]', ['U1 0 4.5', 'CU3 0 1 1 2 3'],
                 ['CU3+ 0 1 1 2 3', 'U1+ 0 4.5'])):
            for exponent in (2, -3):
                with self.subTest(gate=gate, exponent=exponent):
                    lines = self.assert_clean_conversion('include "stdgates.inc"; qubit[2] q; '
                        f'pow({exponent}) @ {gate};')
                    self.assertEqual(lines, ['QUBITS 2'] + (forward if exponent > 0 else inverse) * abs(exponent))

    def test_nested_powers_and_inversions_have_signed_product_semantics(self) -> None:
        prefix = 'OPENQASM 3; include "stdgates.inc"; qubit q; '
        for modifiers, exponent in (('pow(2) @ pow(-3)', -6), ('inv @ pow(-2) @ inv @ pow(-3)', 6),
                                    ('pow(2) @ inv @ pow(3)', -6), ('pow(-2) @ inv @ pow(-3)', -6)):
            with self.subTest(modifiers=modifiers):
                self.assertEqual(convert(prefix + modifiers + ' @ u3(1, 2, 3) q;'),
                                 convert(prefix + f'pow({exponent}) @ u3(1, 2, 3) q;'))

    def test_controls_apply_to_every_instruction_in_each_repetition(self) -> None:
        lines = self.assert_clean_conversion('qubit[3] q; '
            'negctrl @ pow(-2) @ ctrl @ U(1, 2, 3) q[2], q[0], q[1];')
        self.assertEqual(lines, ['QUBITS 3'] + ['C2U3+ 2#0 0 1 1 2 3', 'CU1 2#0 0 -0.5'] * 2)
        lines = self.assert_clean_conversion('include "stdgates.inc"; qubit[3] q; '
            'pow(-2) @ negctrl @ cu(1, 2, 3, 4) q[2], q[0], q[1];')
        self.assertEqual(lines, ['QUBITS 3'] + ['C2U3+ 2#0 0 1 1 2 3', 'CU1+ 2#0 0 4.5'] * 2)

    def test_runtime_parameter_expressions_are_evaluated_once_and_storage_is_retained(self) -> None:
        lines = self.assert_clean_conversion('qubit q; float theta = 0.2; '
            'pow(-3) @ U(theta + 1, theta + 2, theta + 3) q;')
        gates = [line for line in lines if line.startswith('U3+')]
        phases = [line for line in lines if line.startswith('PHASE')]
        self.assertEqual(len(gates), 3)
        self.assertEqual(len(set(gates)), 1)
        self.assertEqual(len(phases), 3)
        self.assertEqual(len(set(phases)), 1)
        for value in (1, 2, 3):
            self.assertEqual(sum(line.endswith(f' += {value}.0') for line in lines), 1)
        operands = gates[0].split()[2:]
        self.assertEqual(len(set(operands)), 3)
        self.assertNotIn(phases[0].split()[1], operands)

    def test_runtime_indexed_parameters_are_captured_once_per_source_application(self) -> None:
        lines = self.assert_clean_conversion('include "stdgates.inc"; qubit q; '
            'array[float, 3] values = {0.2, 0.4, 0.6}; int i = 0; '
            'pow(3) @ u3(values[i], values[i + 1], values[i + 2]) q;')
        gates = [line for line in lines if line.startswith('U3 ')]
        self.assertEqual(len(gates), 3)
        self.assertEqual(len(set(gates)), 1)
        self.assertEqual(len(set(gates[0].split()[2:])), 3)
        self.assertEqual(sum(line.endswith(' := I1') for line in lines), 3)
        self.assertEqual(sum(line.startswith('ASSERT ') for line in lines), 6)

    def test_broadcast_repeats_each_complete_gate_without_recomputing_parameters(self) -> None:
        lines = self.assert_clean_conversion('qubit[2] q; float theta = 1; '
            'pow(2) @ U(theta + 1, 2, 3) q;')
        gates = [line for line in lines if line.startswith('U3 ')]
        self.assertEqual([line.split()[1] for line in gates], ['0', '0', '1', '1'])
        self.assertEqual(len([line for line in lines if line.startswith('PHASE ')]), 4)
        self.assertEqual(sum(line.endswith(' += 1.0') for line in lines), 1)

    def test_phase_powers_support_zero_negative_runtime_and_mixed_controls(self) -> None:
        for exponent in (0, 2, -3):
            with self.subTest(exponent=exponent):
                lines = self.assert_clean_conversion(f'pow({exponent}) @ gphase(0.5);')
                self.assertEqual(lines, ['QUBITS 0'] + ([f'PHASE {0.5 * exponent}'] if exponent else []))
        lines = self.assert_clean_conversion('qubit[2] q; float theta = 1; '
            'negctrl @ pow(-3) @ ctrl @ gphase(theta + 1) q[1], q[0];')
        gates = [line for line in lines if line.startswith('CU1 ')]
        self.assertEqual(len(gates), 3)
        self.assertEqual(len(set(gates)), 1)
        self.assertTrue(gates[0].startswith('CU1 1#0 0 '))
        self.assertEqual(sum(line.endswith(' *= -1.0') for line in lines), 1)

    def test_zero_power_still_checks_gate_parameters_operands_and_broadcasts(self) -> None:
        for source in ('qubit q; pow(0) @ h q;',  # Missing include.
                       'include "stdgates.inc"; qubit q; pow(0) @ unknown q;',
                       'include "stdgates.inc"; qubit q; pow(0) @ rx() q;',
                       'include "stdgates.inc"; qubit q; pow(0) @ rx(true) q;',
                       'qubit q; pow(0) @ U(missing, 0, 0) q;',
                       'qubit q; pow(0) @ U(0, 0, 0) missing;',
                       'qubit[2] q; pow(0) @ U(0, 0, 0) q[2];',
                       'include "stdgates.inc"; qubit q; pow(0) @ ctrl @ x q, q;',
                       'qubit[2] a; qubit[3] b; pow(0) @ ctrl @ U(0, 0, 0) a, b;',
                       'pow(0) @ gphase(true);', 'qubit q; pow(0) @ gphase(1) q;'):
            with self.subTest(source=source), self.assertRaises(qasm2qcx.QASM2QCXError):
                convert('OPENQASM 3; ' + source)

    def test_powered_gates_work_in_all_loop_kinds_without_late_temp_declarations(self) -> None:
        for header in ('for int i in [0:1]', 'for int i in [0:stop]',
                       'for int i in {0, stop}', 'while (false)'):
            with self.subTest(header=header):
                lines = self.assert_clean_conversion('qubit q; float theta = 0.2; int stop = 1; '
                    + header + ' { pow(-2) @ U(theta + 1, 2, 3) q; }')
                self.assertTrue(any(line.startswith('U3+ ') for line in lines))
                first_label = next((i for i, line in enumerate(lines) if line.startswith('@')), len(lines))
                self.assertFalse(any(line.startswith('VAR QASM2QCX_') for line in lines[first_label:]))

    def test_zero_power_does_not_hide_invalid_parameters_or_qubit_names_in_empty_loops(self) -> None:
        for call in ('rx(true) q', 'U(missing, 0, 0) q', 'gphase(true)', 'x missing', 'x i'):
            with self.subTest(call=call), self.assertRaises(qasm2qcx.QASM2QCXError):
                convert('OPENQASM 3; include "stdgates.inc"; qubit q; '
                        f'for int i in [1:0] {{ pow(0) @ {call}; }}')

    def test_loop_output_budget_is_checked_before_repetition(self) -> None:
        with patch.object(qasm2qcx.QASM2QCXConverter, 'MAX_LOOP_OUTPUT_LINES', 4):
            with self.assertRaisesRegex(qasm2qcx.InvalidLoopRangeException, 'For-loop output'):
                convert('OPENQASM 3; qubit q; for int i in [0:0] { pow(2) @ U(1, 2, 3) q; }')

    def test_expansion_limit_counts_full_sequences_and_all_broadcast_elements(self) -> None:
        with patch.object(qasm2qcx.QASM2QCXConverter, 'MAX_GATE_POWER_OUTPUT_LINES', 4):
            self.assertEqual(len(convert('OPENQASM 3; qubit q; pow(2) @ U(1, 2, 3) q;')), 5)
            for call in ('qubit q; pow(3) @ U(1, 2, 3) q;',
                         'qubit[2] q; pow(2) @ U(1, 2, 3) q;',
                         'include "stdgates.inc"; qubit q; pow(5) @ u3(1, 2, 3) q;',
                         'include "stdgates.inc"; qubit q; pow(2147483647) @ pow(2147483647) @ u3(1, 2, 3) q;'):
                with self.subTest(call=call), self.assertRaisesRegex(qasm2qcx.UnsupportedOpenQASMError, 'gate power expansion'):
                    convert('OPENQASM 3; ' + call)

    def test_expansion_failure_cleans_temporaries_and_cli_emits_no_partial_output(self) -> None:
        converter = ControlModifierFoundationTests.converter('qubit q; float theta = 1;')
        caller = converter._QASM2QCXConverter__add_new_temporary_variable(qasm2qcx.ValueType.FLOAT)
        child = qasm2qcx.openqasm3.parser.parse('OPENQASM 3; pow(3) @ U(theta + 1, 2, 3) q;').statements[0]
        with patch.object(qasm2qcx.QASM2QCXConverter, 'MAX_GATE_POWER_OUTPUT_LINES', 4):
            with self.assertRaises(qasm2qcx.UnsupportedOpenQASMError):
                converter.visit(child)
            self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, {caller})
            self.assertIsNone(converter._QASM2QCXConverter__expression_kind)
            source = 'OPENQASM 3; qubit q; U(0, 0, 0) q; pow(3) @ U(1, 2, 3) q;'
            with patch('builtins.open', mock_open(read_data=source)), patch('builtins.print') as output:
                with self.assertRaises(SystemExit):
                    qasm2qcx.main(['power.qasm'])
                output.assert_not_called()


class IntegerPowerModifierSimplificationTests(unittest.TestCase):
    assert_clean_conversion = InverseModifierLoweringTests.assert_clean_conversion

    def test_self_inverse_and_identity_gates_reduce_before_emission(self) -> None:
        converter = qasm2qcx.QASM2QCXConverter
        prefix = 'OPENQASM 3; include "stdgates.inc"; qubit[3] q; '
        for gate in ('id', 'x', 'y', 'z', 'h', 'swap', 'cx', 'cy', 'cz', 'ch', 'ccx', 'cswap'):
            operands = ', '.join(f'q[{i}]' for i in range(converter.gate_signatures[gate][1]))
            call = f'{gate} {operands};'
            for power in (-8, -3, -2, -1, 0, 1, 2, 3, 8):
                with self.subTest(gate=gate, power=power):
                    expected = ['QUBITS 3'] if gate == 'id' or power % 2 == 0 else convert(prefix + call)
                    self.assertEqual(convert(prefix + f'pow({power}) @ ' + call), expected)

    def test_phase_and_square_root_cycles_use_existing_exact_qcx_gates(self) -> None:
        for gate, cycle in (
                ('s', [[], ['S'], ['Z'], ['S+']]),
                ('sdg', [[], ['S+'], ['Z'], ['S']]),
                ('t', [[], ['T'], ['S'], ['S', 'T'], ['Z'], ['Z', 'T'], ['S+'], ['T+']]),
                ('tdg', [[], ['T+'], ['S+'], ['Z', 'T'], ['Z'], ['S', 'T'], ['S'], ['T']]),
                ('sx', [[], ['SX'], ['X'], ['SX+']])):
            for power in range(2 * len(cycle)):
                with self.subTest(gate=gate, power=power):
                    expected = ['QUBITS 1'] + [name + ' 0' for name in cycle[power % len(cycle)]]
                    self.assertEqual(self.assert_clean_conversion(
                        f'include "stdgates.inc"; qubit q; pow({power}) @ {gate} q;'), expected)

    def test_reductions_preserve_intrinsic_and_mixed_control_conditions(self) -> None:
        for call, expected in (
                ('negctrl @ pow(2) @ sx q[2], q[0]', ['CX 2#0 0']),
                ('ctrl @ pow(2) @ negctrl @ s q[2], q[0], q[1]', ['C2Z 2 0#0 1']),
                ('pow(-2) @ negctrl @ t q[2], q[0]', ['CS+ 2#0 0']),
                ('negctrl @ inv @ pow(2) @ tdg q[2], q[0]', ['CS 2#0 0']),
                ('pow(3) @ negctrl @ cx q[2], q[0], q[1]', ['C2X 2#0 0 1']),
                ('pow(2) @ negctrl @ cx q[2], q[0], q[1]', [])):
            with self.subTest(call=call):
                self.assertEqual(self.assert_clean_conversion('include "stdgates.inc"; qubit[3] q; '
                    + call + ';'), ['QUBITS 3'] + expected)

    def test_reductions_happen_before_expansion_limits_even_for_large_nested_powers(self) -> None:
        with patch.object(qasm2qcx.QASM2QCXConverter, 'MAX_GATE_POWER_OUTPUT_LINES', 1):
            for gate, power, expected in (('x', 2147483646, []), ('x', 2147483647, ['X 0']),
                                          ('id', 2147483647, []), ('t', 2147483646, ['Z 0']),
                                          ('sx', 2147483646, [])):
                with self.subTest(gate=gate, power=power):
                    self.assertEqual(convert('OPENQASM 3; include "stdgates.inc"; qubit q; '
                        f'pow({power}) @ pow({power}) @ {gate} q;'),
                        ['QUBITS 1'] + expected)
            self.assertEqual(convert('OPENQASM 3; pow(1000000) @ gphase(0.25);'),
                             ['QUBITS 0', 'PHASE 250000.0'])
            with self.assertRaisesRegex(qasm2qcx.UnsupportedOpenQASMError, 'gate power expansion'):
                convert('OPENQASM 3; include "stdgates.inc"; qubit q; pow(3) @ t q;')

    def test_literal_phase_and_rotation_parameters_are_scaled_exactly_as_gate_families(self) -> None:
        converter = qasm2qcx.QASM2QCXConverter
        for gate in ('p', 'phase', 'u1', 'rx', 'ry', 'rz', 'cp', 'cphase', 'crx', 'cry', 'crz'):
            operands = 'q[0], q[1]' if converter.gate_signatures[gate][1] == 2 else 'q[0]'
            for power in (3, -3):
                with self.subTest(gate=gate, power=power):
                    lines = self.assert_clean_conversion('include "stdgates.inc"; qubit[2] q; '
                        f'pow({power}) @ {gate}(0.5) {operands};')
                    mnemonic = converter.stdgates_qcx_map[gate]
                    angle = -0.75 if 'r' in gate else 1.5
                    if power < 0:
                        mnemonic = converter.qcx_adjoint_gate_map[mnemonic]
                    self.assertEqual(lines, ['QUBITS 2', f'{mnemonic} {"0 1" if "," in operands else "0"} {angle}'])

    def test_literal_scaling_keeps_negative_controls_and_global_phase(self) -> None:
        self.assertEqual(self.assert_clean_conversion('include "stdgates.inc"; qubit[3] q; '
            'negctrl @ pow(-3) @ ctrl @ rz(0.5) q[2], q[0], q[1];'),
            ['QUBITS 3', 'C2EZ+ 2#0 0 1 -0.75'])
        self.assertEqual(self.assert_clean_conversion('qubit[2] q; '
            'negctrl @ pow(-3) @ ctrl @ gphase(0.5) q[1], q[0];'),
            ['QUBITS 2', 'CU1 1#0 0 -1.5'])

    def test_runtime_angles_keep_bounded_repetition_and_private_rotation_storage(self) -> None:
        for gate, mnemonic in (('p', 'U1'), ('rx', 'EX'), ('gphase', 'PHASE')):
            with self.subTest(gate=gate):
                call = f'{gate}(theta)' + ('' if gate == 'gphase' else ' q')
                lines = self.assert_clean_conversion('include "stdgates.inc"; qubit q; float theta = 0.2; '
                    f'pow(3) @ {call};')
                gates = [line for line in lines if line.startswith(mnemonic + ' ')]
                self.assertEqual(len(gates), 3)
                self.assertEqual(len(set(gates)), 1)
                self.assertFalse(any(line.endswith(' *= 3.0') for line in lines))

    def test_nonfinite_scaled_literals_fall_back_without_emitting_infinity(self) -> None:
        for call, mnemonic in (('p(1e308) q', 'U1'), ('gphase(1e308)', 'PHASE')):
            with self.subTest(call=call):
                lines = self.assert_clean_conversion('include "stdgates.inc"; qubit q; '
                    f'pow(2) @ {call};')
                self.assertEqual(len([line for line in lines if line.startswith(mnemonic + ' ')]), 2)
                self.assertFalse(any('inf' in line.lower() for line in lines))

    def test_inexact_or_overflowing_scaling_coefficients_do_not_silently_round(self) -> None:
        for power in (2 ** 53 + 1, 10 ** 400):
            with self.subTest(power=power), self.assertRaisesRegex(
                    qasm2qcx.UnsupportedOpenQASMError, 'gate power expansion'):
                convert(f'OPENQASM 3; pow({power}) @ gphase(0.5);')

    def test_rounded_angle_products_and_integer_narrowing_keep_original_operands(self) -> None:
        for angle in ('1e100', '0.1', '9007199254740993'):
            for gate, mnemonic in (('p', 'U1'), ('gphase', 'PHASE')):
                call = f'{gate}({angle})' + (' q' if gate == 'p' else '')
                with self.subTest(angle=angle, gate=gate):
                    lines = self.assert_clean_conversion('include "stdgates.inc"; qubit q; '
                        f'pow(3) @ {call};')
                    gates = [line for line in lines if line.startswith(mnemonic + ' ')]
                    self.assertEqual(len(gates), 3)
                    self.assertEqual(len(set(gates)), 1)
                    self.assertEqual(gates[0].split()[-1], str(float(angle)) if 'e' in angle else angle)

    def test_identity_reductions_do_not_hide_invalid_operations(self) -> None:
        for call in ('pow(2) @ swap q[0], q[0]', 'pow(2) @ cx q[0], q[0]',
                     'pow(2) @ x missing', 'pow(4) @ s q[2]', 'pow(8) @ t(true) q[0]',
                     'pow(2) @ negctrl @ h q[0], q[0]'):
            with self.subTest(call=call), self.assertRaises(qasm2qcx.QASM2QCXError):
                convert('OPENQASM 3; include "stdgates.inc"; qubit[2] q; ' + call + ';')

    def test_documented_integer_power_example(self) -> None:
        document = CONVERTER_PATH.with_suffix('.md').read_text(encoding='utf-8')
        examples = re.findall(r'<!-- integer-power-example -->\s*```qasm\n(.*?)\n```',
                              document, flags=re.DOTALL)
        self.assertEqual(len(examples), 1)
        self.assertEqual(convert(examples[0]), ['QUBITS 3', 'Z 0', 'S 0', 'T 0',
                                              'CX 1#0 2', 'EZ+ 2 -0.75', 'PHASE 0.5'])


class RuntimeIntegerPowerModifierLoweringTests(unittest.TestCase):
    assert_clean_conversion = InverseModifierLoweringTests.assert_clean_conversion

    def test_documented_runtime_integer_power_example(self) -> None:
        document = CONVERTER_PATH.with_suffix('.md').read_text(encoding='utf-8')
        examples = re.findall(r'<!-- runtime-integer-power-example -->\s*```qasm\n(.*?)\n```',
                              document, flags=re.DOTALL)
        self.assertEqual(len(examples), 1)
        lines = self.assert_clean_conversion(examples[0].split('\n', 1)[1])
        self.assertIn('CU3 2#0 0 THETA31 0.4 0.6', lines)
        self.assertIn('CU3+ 2#0 0 THETA31 0.4 0.6', lines)

    def test_signed_runtime_loop_contains_complete_forward_and_inverse_sequences(self) -> None:
        lines = self.assert_clean_conversion('qubit q; int n = -2; pow(n) @ U(1, 2, 3) q;')
        phase = lines.index('PHASE 0.5')
        self.assertEqual(lines[phase:phase + 2], ['PHASE 0.5', 'U3 0 1 2 3'])
        inverse = lines.index('U3+ 0 1 2 3')
        self.assertEqual(lines[inverse:inverse + 2], ['U3+ 0 1 2 3', 'PHASE -0.5'])
        self.assertTrue(any(line.endswith(' < 0') and line.startswith('JUMPIF ') for line in lines))
        self.assertTrue(any(line.endswith(' == 0') and line.startswith('JUMPIF ') for line in lines))
        self.assertTrue(any(line.startswith('LET QASM2QCX_UINT_') and line.endswith(' -= 1') for line in lines))

    def test_unsigned_runtime_exponents_are_not_narrowed_through_int(self) -> None:
        lines = self.assert_clean_conversion(f'qubit q; uint n = {qasm2qcx.QASM2QCXConverter.QCX_UINT_MAX}; '
            'pow(n) @ U(1, 2, 3) q;')
        self.assertFalse(any(':INT:' in line for line in lines))
        self.assertFalse(any('_INVERSE' in line for line in lines))
        self.assertIn('U3 0 1 2 3', lines)
        self.assertNotIn('U3+ 0 1 2 3', lines)
        self.assertEqual(sum(line.endswith(' := N1') for line in lines), 1)

    def test_signed_magnitude_uses_unsigned_subtraction_including_int_min(self) -> None:
        lines = self.assert_clean_conversion(f'qubit q; int n = {qasm2qcx.QASM2QCXConverter.QCX_INT_MIN}; '
            'pow(n) @ U(1, 2, 3) q;')
        self.assertTrue(any(line.startswith('LET QASM2QCX_UINT_') and ' -= :UINT:QASM2QCX_INT_' in line for line in lines))
        self.assertFalse(any(line.startswith('LET QASM2QCX_INT_') and ' *= ' in line for line in lines))
        self.assertEqual(sum(line.endswith(' := N1') for line in lines), 1)

    def test_generated_code_size_does_not_depend_on_runtime_exponent_value(self) -> None:
        sizes = []
        for n in (-2147483648, -3, 0, 1, 2147483647):
            lines = self.assert_clean_conversion(f'qubit q; int n = {n}; pow(n) @ U(1, 2, 3) q;')
            sizes.append(len(lines))
        self.assertEqual(len(set(sizes)), 1)

    def test_indexed_exponent_and_gate_parameters_are_each_captured_once(self) -> None:
        lines = self.assert_clean_conversion('qubit q; int i = 0; array[int, 1] counts = {-2}; '
            'array[float, 3] angles = {0.2, 0.4, 0.6}; '
            'pow(counts[i]) @ U(angles[i], angles[i + 1], angles[i + 2]) q;')
        self.assertEqual(sum(line.endswith(' := I1') for line in lines), 4)
        self.assertEqual(sum(line.startswith('ASSERT ') for line in lines), 8)
        forward = next(line for line in lines if line.startswith('U3 ')).split()[2:]
        inverse = next(line for line in lines if line.startswith('U3+ ')).split()[2:]
        self.assertEqual(forward, inverse)
        self.assertEqual(len(set(forward)), 3)
        loop = next(i for i, line in enumerate(lines) if line.startswith('@QASM2QCX_GATE_POWER_') and line.endswith('_LOOP'))
        self.assertFalse(any(line.startswith(('LET QASM2QCX_REAL_', 'ASSERT ', 'VAR ')) for line in lines[loop:]))

    def test_controlled_cu_preserves_both_phase_corrected_directions(self) -> None:
        lines = self.assert_clean_conversion('include "stdgates.inc"; qubit[3] q; int n = -2; '
            'negctrl @ pow(n) @ cu(1, 2, 3, 4) q[2], q[0], q[1];')
        forward = lines.index('CU1 2#0 0 4.5')
        self.assertEqual(lines[forward:forward + 2], ['CU1 2#0 0 4.5', 'C2U3 2#0 0 1 1 2 3'])
        inverse = lines.index('C2U3+ 2#0 0 1 1 2 3')
        self.assertEqual(lines[inverse:inverse + 2], ['C2U3+ 2#0 0 1 1 2 3', 'CU1+ 2#0 0 4.5'])

    def test_constant_factor_and_inverse_signs_are_combined_without_runtime_products(self) -> None:
        lines = self.assert_clean_conversion('qubit q; int n = 2; inv @ pow(-2) @ pow(n) @ U(1, 2, 3) q;')
        self.assertEqual(lines.count('U3 0 1 2 3'), 1)
        self.assertEqual(lines.count('U3+ 0 1 2 3'), 1)
        self.assertEqual(sum(line.startswith('@QASM2QCX_GATE_POWER_') and line.endswith('_LOOP') for line in lines), 2)
        self.assertFalse(any(line.startswith('LET QASM2QCX_INT_') and ' *= ' in line for line in lines))

    def test_phase_and_broadcast_loops_reuse_one_exponent_snapshot(self) -> None:
        lines = self.assert_clean_conversion('qubit[2] q; int n = -2; float theta = 0.3; '
            'negctrl @ pow(n) @ gphase(theta) q;')
        self.assertEqual(sum(line.endswith(' := N1') for line in lines), 1)
        self.assertEqual(sum(line.startswith('@QASM2QCX_GATE_POWER_') and line.endswith('_LOOP') for line in lines), 2)
        self.assertEqual(sum(line.startswith('U1 ') for line in lines), 4)
        self.assertEqual(sum(line.endswith(' *= -1.0') for line in lines), 1)

    def test_runtime_powers_in_loops_hoist_declarations_and_keep_unique_labels(self) -> None:
        for header in ('for int i in [-1:1]', 'for int i in [first:last]',
                       'for int i in {first, last}', 'while (false)'):
            with self.subTest(header=header):
                lines = self.assert_clean_conversion('qubit q; int first = -1; int last = 1; int n = 2; float theta = 0.3; '
                    + header + ' { pow(n) @ U(theta + 1, 2, 3) q; pow(n) @ gphase(theta); }')
                if header == 'for int i in [-1:1]':
                    # Unrolled bodies are emitted sequentially. Declarations
                    # may follow a preceding power's END, but not enter its loop.
                    for i, line in enumerate(lines):
                        if line.startswith('@QASM2QCX_GATE_POWER_') and line.endswith('_LOOP'):
                            end = lines.index(line[:-5] + '_END', i)
                            self.assertFalse(any(code.startswith('VAR ') for code in lines[i:end]))
                else:
                    first_label = next(i for i, line in enumerate(lines) if line.startswith('@'))
                    self.assertFalse(any(line.startswith('VAR QASM2QCX_') for line in lines[first_label:]))

    def test_iterator_exponents_support_unrolled_runtime_and_unsigned_collection_bindings(self) -> None:
        for header, declarations in (('for int n in [-1:1]', ''),
                                     ('for int n in [first:last]', 'int first = -1; int last = 1;'),
                                     ('for uint[3] n in values', 'array[uint[3], 2] values = {1, 2};')):
            with self.subTest(header=header):
                self.assert_clean_conversion('qubit q; ' + declarations + header + ' { pow(n) @ U(1, 2, 3) q; }')

    def test_invalid_runtime_types_are_rejected_even_in_empty_loops(self) -> None:
        for modifier in ('pow(flag)', 'pow(f)', 'pow(missing)'):
            for wrapper in ('{}', 'for int i in [1:0] {{ {} }}'):
                with self.subTest(modifier=modifier, wrapper=wrapper), self.assertRaises(qasm2qcx.QASM2QCXError):
                    convert('OPENQASM 3; qubit q; int n = 2; bool flag = true; float f = 2.0; '
                            + wrapper.format(modifier + ' @ U(1, 2, 3) q;'))

    def test_runtime_zero_does_not_hide_invalid_operations_and_constant_zero_emits_no_gate(self) -> None:
        for call in ('pow(n) @ U(true, 0, 0) q;', 'pow(n) @ U(0, 0, 0) missing;',
                     'pow(n) @ gphase(true);', 'pow(n) @ ctrl @ U(1, 2, 3) q, q;'):
            with self.subTest(call=call), self.assertRaises(qasm2qcx.QASM2QCXError):
                convert('OPENQASM 3; qubit q; int n = 0; ' + call)
        lines = self.assert_clean_conversion('qubit q; int n = 2; pow(0) @ pow(n) @ U(1, 2, 3) q;')
        self.assertFalse(any(line.startswith(('U3 ', 'U3+ ', 'PHASE ')) for line in lines))
        self.assertEqual(sum(line.endswith(' := N1') for line in lines), 1)

    def test_runtime_failure_cleans_up_and_cli_does_not_emit_partial_output(self) -> None:
        converter = RuntimeIntegerPowerModifierFoundationTests.converter('qubit q; int n = 2; float theta = 0.3;')
        caller = converter._QASM2QCXConverter__add_new_temporary_variable(qasm2qcx.ValueType.FLOAT)
        statement = qasm2qcx.openqasm3.parser.parse('OPENQASM 3; pow(n) @ U(theta + 1, 0, 0) q;').statements[0]
        with patch.object(qasm2qcx.QASM2QCXConverter, 'MAX_GATE_POWER_OUTPUT_LINES', 5):
            with self.assertRaises(qasm2qcx.UnsupportedOpenQASMError):
                converter.visit(statement)
            self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, {caller})
            self.assertIsNone(converter._QASM2QCXConverter__expression_kind)
            with patch('builtins.open', mock_open(read_data='OPENQASM 3; qubit q; int n = 2; pow(n) @ U(1, 2, 3) q;')), \
                    patch('builtins.print') as output:
                with self.assertRaises(SystemExit):
                    qasm2qcx.main(['runtime-power.qasm'])
                output.assert_not_called()


class RuntimeIntegerPowerModifierNestingTests(unittest.TestCase):
    assert_clean_conversion = InverseModifierLoweringTests.assert_clean_conversion

    def test_nested_runtime_counts_have_distinct_counters_and_reset_inner_counts(self) -> None:
        lines = self.assert_clean_conversion('qubit q; int n = -2; uint m = 3; '
            'pow(n) @ pow(m) @ U(1, 2, 3) q;')
        starts = [i for i, line in enumerate(lines) if line.startswith('@QASM2QCX_GATE_POWER_') and line.endswith('_LOOP')]
        self.assertEqual(len(starts), 2)
        self.assertTrue(lines[starts[0] - 1].startswith('LET QASM2QCX_UINT_'))
        self.assertTrue(lines[starts[1] - 1].startswith('LET QASM2QCX_UINT_'))
        self.assertNotEqual(lines[starts[0] - 1].split()[1], lines[starts[1] - 1].split()[1])
        self.assertFalse(any(line.startswith('VAR ') for line in lines[starts[0]:]))
        self.assertEqual(sum(line.endswith(' := N1') for line in lines), 1)
        self.assertEqual(sum(line.endswith(' := M1') for line in lines), 1)
        self.assertEqual(lines.count('U3 0 1 2 3'), 1)

    def test_zero_factor_guards_precede_every_repetition_loop(self) -> None:
        lines = self.assert_clean_conversion(f'qubit q; uint n = {qasm2qcx.QASM2QCXConverter.QCX_UINT_MAX}; int m = 0; '
            'pow(n) @ pow(m) @ U(1, 2, 3) q;')
        first_loop = next(i for i, line in enumerate(lines) if line.startswith('@QASM2QCX_GATE_POWER_') and line.endswith('_LOOP'))
        guards = [i for i, line in enumerate(lines) if line.startswith('JUMPIF ') and '_DONE ' in line]
        self.assertEqual(len(guards), 2)
        self.assertTrue(all(i < first_loop for i in guards))
        self.assertTrue(all(lines[i].endswith(' == 0') for i in guards))

    def test_nested_signed_counts_combine_only_sign_parity(self) -> None:
        lines = self.assert_clean_conversion('qubit q; int n = -2; int m = -3; '
            'inv @ pow(n) @ pow(m) @ U(1, 2, 3) q;')
        toggles = [line for line in lines if line.endswith(' ^= -1')]
        self.assertEqual(len(toggles), 2)
        self.assertEqual(len({line.split()[1] for line in toggles}), 1)
        self.assertFalse(any(line.startswith(('LET QASM2QCX_INT_', 'LET QASM2QCX_UINT_')) and ' *= ' in line for line in lines))
        self.assertTrue(any(toggles[0].split()[1] + ' < 0' in line for line in lines if line.startswith('JUMPIF ')))

    def test_periodic_counts_reduce_before_multiplication_and_use_one_bounded_loop(self) -> None:
        self.assertTrue(all(len(cycle) in (1, 2, 4, 8)
                            for cycle in qasm2qcx.QASM2QCXConverter.qcx_gate_power_cycles.values()))
        for gate, operands, period in (('x', 'q[0]', 2), ('s', 'q[0]', 4), ('t', 'q[0]', 8),
                                       ('sx', 'q[0]', 4), ('ccx', 'q[0], q[1], q[2]', 2)):
            with self.subTest(gate=gate):
                lines = self.assert_clean_conversion('include "stdgates.inc"; qubit[3] q; int n = -2; uint m = 3; '
                    f'pow(n) @ pow(m) @ {gate} {operands};')
                self.assertEqual(sum(line.startswith('@QASM2QCX_GATE_POWER_') and line.endswith('_LOOP') for line in lines), 1)
                residues = [line for line in lines if line.endswith(f' &= {period - 1}')]
                self.assertEqual(len(residues), 4)
                for i, line in enumerate(lines):
                    if line.startswith('LET QASM2QCX_UINT_') and ' *= ' in line:
                        self.assertTrue(lines[i - 1].endswith(f' &= {period - 1}'))
                        self.assertTrue(lines[i + 1].endswith(f' &= {period - 1}'))

    def test_period_uses_the_constant_powered_operation_and_keeps_control_values(self) -> None:
        lines = self.assert_clean_conversion('include "stdgates.inc"; qubit[3] q; int n = -2; '
            'negctrl @ pow(2) @ pow(n) @ ctrl @ s q[2], q[0], q[1];')
        self.assertTrue(any(line.endswith(' &= 1') for line in lines))
        self.assertFalse(any(line.endswith(' &= 3') for line in lines))
        self.assertIn('C2Z 2#0 0 1', lines)

    def test_large_constant_factor_products_use_separate_native_counters(self) -> None:
        factor = 1 << (qasm2qcx.QASM2QCXConverter.QCX_UINT_WIDTH // 2)
        lines = self.assert_clean_conversion(f'qubit q; int n = 0; pow({factor}) @ pow(n) @ pow({factor}) @ U(1, 2, 3) q;')
        self.assertEqual(sum(line.endswith(f' := {factor}') for line in lines), 2)
        self.assertEqual(sum(line.startswith('@QASM2QCX_GATE_POWER_') and line.endswith('_LOOP') for line in lines), 3)
        self.assertEqual(lines.count('U3 0 1 2 3'), 1)
        self.assertFalse(any(line.startswith('LET QASM2QCX_UINT_') and ' *= ' in line for line in lines))

    def test_scaled_constant_angles_are_not_counted_twice(self) -> None:
        lines = self.assert_clean_conversion('include "stdgates.inc"; qubit q; int n = 3; '
            'pow(2) @ pow(n) @ p(0.5) q;')
        self.assertEqual(sum(line.startswith('@QASM2QCX_GATE_POWER_') and line.endswith('_LOOP') for line in lines), 1)
        self.assertIn('U1 0 1.0', lines)
        self.assertIn('U1+ 0 1.0', lines)

    def test_individual_constant_counts_outside_native_range_are_explicitly_rejected(self) -> None:
        too_large = qasm2qcx.QASM2QCXConverter.QCX_UINT_MAX + 1
        with self.assertRaisesRegex(qasm2qcx.UnsupportedOpenQASMError, 'constant factor exceeds native UINT'):
            self.assert_clean_conversion(f'qubit q; int n = 0; pow({too_large}) @ pow(n) @ U(1, 2, 3) q;')
        # Periodic identities need no native counter for the constant factor.
        self.assert_clean_conversion(f'include "stdgates.inc"; qubit q; int n = 1; pow({too_large}) @ pow(n) @ x q;')

    def test_nested_exponents_and_parameters_are_captured_once_before_broadcast(self) -> None:
        lines = self.assert_clean_conversion('qubit[2] q; int i = 0; array[int, 2] counts = {-2, 3}; '
            'array[float, 3] angles = {0.2, 0.4, 0.6}; '
            'pow(counts[i]) @ pow(counts[i + 1]) @ U(angles[i], angles[i + 1], angles[i + 2]) q;')
        self.assertEqual(sum(line.endswith(' := I1') for line in lines), 5)
        self.assertEqual(sum(line.startswith('ASSERT ') for line in lines), 10)
        self.assertEqual(sum(line.startswith('@QASM2QCX_GATE_POWER_') and line.endswith('_LOOP') for line in lines), 4)

    def test_nested_runtime_factors_validate_all_types_even_after_zero(self) -> None:
        for wrapper in ('{}', 'for int i in [1:0] {{ {} }}'):
            self.assert_clean_conversion('qubit q; int n = 0; int m = -2; '
                + wrapper.format('pow(n) @ pow(m) @ U(1, 2, 3) q;'))
            with self.assertRaises(qasm2qcx.QASM2QCXError):
                self.assert_clean_conversion('qubit q; int n = 0; bool m = true; '
                    + wrapper.format('pow(n) @ pow(m) @ U(1, 2, 3) q;'))


class UnsignedIntegerFoundationTests(unittest.TestCase):
    def converter(self, source: str):
        program = qasm2qcx.openqasm3.parser.parse('OPENQASM 3.0; ' + source)
        converter = qasm2qcx.QASM2QCXConverter(program)
        converter.visit(program)
        return converter

    def test_native_unsigned_bounds_and_enum(self) -> None:
        converter = qasm2qcx.QASM2QCXConverter
        self.assertEqual(converter.QCX_UINT_WIDTH, ctypes.sizeof(ctypes.c_uint) * 8)
        self.assertEqual(converter.QCX_UINT_MAX, (1 << converter.QCX_UINT_WIDTH) - 1)
        self.assertNotEqual(qasm2qcx.ValueType.INT, qasm2qcx.ValueType.UINT)
        self.assertEqual(qasm2qcx.ValueType.BOOL.value, 5)

    def test_unsized_and_explicit_native_types_remain_distinct(self) -> None:
        width = qasm2qcx.QASM2QCXConverter.QCX_UINT_WIDTH
        converter = self.converter(f'uint a; uint[{width}] b; uint[1] c;')
        types = converter._QASM2QCXConverter__uint_variable_types
        self.assertEqual(types['A1'], qasm2qcx._UnsignedIntegerType(width, False))
        self.assertEqual(types['B1'], qasm2qcx._UnsignedIntegerType(width, True))
        self.assertEqual(types['C1'], qasm2qcx._UnsignedIntegerType(1, True))
        self.assertNotEqual(types['A1'], types['B1'])

    def test_every_supported_width_and_modulo_normalization(self) -> None:
        native_width = qasm2qcx.QASM2QCXConverter.QCX_UINT_WIDTH
        for width in range(1, native_width + 1):
            with self.subTest(width=width):
                converter = self.converter(f'uint[{width}] a;')
                info = converter._QASM2QCXConverter__uint_variable_types['A1']
                self.assertEqual(info.mask, (1 << width) - 1)
                for value in (-1, -(1 << (width + 2)), 0, 1, info.mask,
                              info.mask + 1, (1 << (width + 3)) + 7):
                    self.assertEqual(info.normalize(value), value % (1 << width))
        for width in (0, -1):
            with self.assertRaises(ValueError):
                qasm2qcx._UnsignedIntegerType(width, True)

    def test_constant_widths_and_separate_unsigned_bookkeeping(self) -> None:
        converter = self.converter('''
            const int N = 4; const uint M = 2;
            const uint[N + M] mask = 3;
            uint[N * M] value = mask; int signed_value = N;
            for int i in {M} { signed_value += i; }
        ''')
        self.assertEqual(converter._QASM2QCXConverter__const_uint_types['mask'].width, 6)
        self.assertEqual(converter._QASM2QCXConverter__uint_variable_types['VALUE31'].width, 8)
        self.assertEqual(converter._QASM2QCXConverter__const_uint_variable_name_values_map,
                         {'M': [2], 'mask': [3]})
        self.assertEqual(converter._QASM2QCXConverter__const_int_variable_name_values_map, {'N': [4]})
        self.assertEqual(converter._QASM2QCXConverter__uint_variable_name_size_map, {'VALUE31': 1})
        self.assertNotIn('VALUE31', converter._QASM2QCXConverter__int_variable_name_size_map)

    def test_width_is_resolved_in_declaration_scope(self) -> None:
        converter = self.converter('''const int N = 8; uint[N] value;
            for int N in [1:2] { value += N; }''')
        self.assertEqual(converter._QASM2QCXConverter__uint_variable_types['VALUE31'].width, 8)

    def test_rejects_nonpositive_and_oversized_declaration_widths(self) -> None:
        native_width = qasm2qcx.QASM2QCXConverter.QCX_UINT_WIDTH
        for declaration in ('uint[{width}] a;', 'const uint[{width}] a = 1;'):
            for width in (0, -1, native_width + 1, native_width * 2):
                error = (qasm2qcx.InvalidDeclarationException if width <= 0
                         else qasm2qcx.UnsupportedOpenQASMError)
                with self.subTest(declaration=declaration, width=width), self.assertRaises(error):
                    # A named width reaches converter validation even when
                    # the parser rejects a nonpositive literal designator.
                    convert(f'OPENQASM 3.0; const int SIZE = {width}; '
                            + declaration.format(width='SIZE'))

    def test_rejects_noninteger_and_runtime_widths(self) -> None:
        for width in ('true', '2.5', 'R', 'N', 'float(8)', 'N + 1'):
            with self.subTest(width=width), self.assertRaises(qasm2qcx.InvalidDeclarationException):
                convert('OPENQASM 3.0; int N = 8; float R = 8.0; '
                        f'uint[{width}] value;')
        for source in ('uint[missing] value;', 'uint[N] value; const int N = 8;',
                       'const uint[N] N = 8;'):
            with self.subTest(source=source), self.assertRaises(qasm2qcx.NoVariableNameException):
                convert('OPENQASM 3.0; ' + source)

    def test_cast_width_validation(self) -> None:
        width = qasm2qcx.QASM2QCXConverter.QCX_UINT_WIDTH
        for size in ('1', '8', str(width), 'N + 1'):
            convert('OPENQASM 3.0; const int N = 7; int value = 3; '
                    f'int result = uint[{size}](value);')
        for size in ('N - N', 'N - N - 1', str(width + 1), 'true', '1.5', 'value'):
            with self.subTest(size=size), self.assertRaises(qasm2qcx.QASM2QCXError):
                convert('OPENQASM 3.0; const int N = 7; int value = 3; '
                        f'int result = uint[{size}](value);')

    def test_cast_width_checked_during_loop_analysis(self) -> None:
        with self.assertRaisesRegex(qasm2qcx.InvalidDeclarationException, 'constant integer'):
            convert('OPENQASM 3.0; int width = 8; '
                    'for int i in [0:uint[width](3)] {}')
        for loop in ('for int N in [1:2]', 'for int N in [2:1]'):
            with self.subTest(loop=loop), self.assertRaisesRegex(
                    qasm2qcx.InvalidDeclarationException, 'loop iterator'):
                convert('OPENQASM 3.0; const int N = 8; '
                        + loop + ' { for int i in [0:uint[N](1)] {} }')

    def test_width_evaluation_restores_expression_state(self) -> None:
        converter = self.converter('const int N = 8; uint[N] a;')
        converter._QASM2QCXConverter__expression_kind = qasm2qcx.ExpressionKind.ARITHMETIC
        converter._QASM2QCXConverter__value = 'SENTINEL'
        converter._QASM2QCXConverter__value_type = qasm2qcx.ValueType.FLOAT
        converter._QASM2QCXConverter__value_kind = qasm2qcx.ValueKind.TEMPORARY
        converter._QASM2QCXConverter__evaluate_constant = False
        for size in (qasm2qcx.ast.IntegerLiteral(8), qasm2qcx.ast.IntegerLiteral(0)):
            try:
                converter._QASM2QCXConverter__unsigned_integer_type(qasm2qcx.ast.UintType(size))
            except qasm2qcx.InvalidDeclarationException:
                pass
            self.assertEqual((converter._QASM2QCXConverter__expression_kind,
                              converter._QASM2QCXConverter__value,
                              converter._QASM2QCXConverter__value_type,
                              converter._QASM2QCXConverter__value_kind,
                              converter._QASM2QCXConverter__evaluate_constant),
                             (qasm2qcx.ExpressionKind.ARITHMETIC, 'SENTINEL',
                              qasm2qcx.ValueType.FLOAT, qasm2qcx.ValueKind.TEMPORARY, False))

    def test_uint_temporary_reuse_updates_width_and_emits_masks(self) -> None:
        converter = self.converter('')
        narrow = qasm2qcx._UnsignedIntegerType(8, True)
        native = qasm2qcx._UnsignedIntegerType(converter.QCX_UINT_WIDTH, False)
        name = converter._QASM2QCXConverter__add_new_temporary_variable(qasm2qcx.ValueType.UINT, narrow)
        self.assertEqual(converter._QASM2QCXConverter__temporary_uint_types[name], narrow)
        converter._QASM2QCXConverter__normalize_uint_storage(name, narrow)
        converter._QASM2QCXConverter__release_temporary_variable(name)
        reused = converter._QASM2QCXConverter__add_new_temporary_variable(qasm2qcx.ValueType.UINT)
        self.assertEqual(name, reused)
        self.assertEqual(converter._QASM2QCXConverter__temporary_uint_types[name], native)
        converter._QASM2QCXConverter__normalize_uint_storage(name, native)
        self.assertEqual(list(converter), ['QUBITS 0', f'VAR {name} UINT', f'LET {name} &= 255'])
        with self.assertRaises(ValueError):
            converter._QASM2QCXConverter__add_new_temporary_variable(qasm2qcx.ValueType.INT, narrow)


class NativeUnsignedExpressionTests(unittest.TestCase):
    def test_native_constants_keep_full_unsigned_range(self) -> None:
        maximum = qasm2qcx.QASM2QCXConverter.QCX_UINT_MAX
        lines = convert(f'''OPENQASM 3.0; const uint U = uint(-1);
            uint a = U; uint b = U + uint(1); uint c = ~U; uint d = -uint(1);''')
        for name, value in (('A1', maximum), ('B1', 0), ('C1', 0), ('D1', maximum)):
            self.assertIn(f'VAR {name} UINT', lines)
            self.assertIn(f'LET {name} := {value}', lines)

    def test_narrow_constant_operations_mask_intermediate_results(self) -> None:
        lines = convert('''OPENQASM 3.0; const uint[8] U = 255;
            uint[8] a = (U + uint[8](1)) / uint[8](2);
            uint[8] b = (U * U) % uint[8](7);
            uint[8] c = ~uint[8](0); uint[8] d = -uint[8](1);''')
        for name, value in (('A1', 0), ('B1', 1), ('C1', 255), ('D1', 255)):
            self.assertIn(f'LET {name} := {value}', lines)
            self.assertIn(f'LET {name} &= 255', lines)

    def test_runtime_narrow_results_use_uint_temporaries_and_masks(self) -> None:
        lines = convert('''OPENQASM 3.0; uint[8] a = 255; uint[8] b = 1;
            uint[8] c = (a + b) / b; c = ~a; c = -b; c %= b;''')
        self.assertTrue(any(line.startswith('VAR QASM2QCX_UINT_') for line in lines))
        self.assertFalse(any(line.startswith('VAR QASM2QCX_INT_') for line in lines))
        self.assertTrue(any(line.startswith('LET QASM2QCX_UINT_') and line.endswith(' &= 255') for line in lines))
        self.assertTrue(any(line.endswith(' ^= 255') for line in lines))
        self.assertTrue(any(line.endswith(' *= :UINT:-1') for line in lines))

    def test_mixed_integer_promotions_depend_on_unsigned_width(self) -> None:
        narrow = convert('OPENQASM 3.0; int a = -1; uint[8] b = 1; int c = a + b;')
        self.assertIn('LET QASM2QCX_INT_0 += :INT:B1', narrow)
        native = convert('OPENQASM 3.0; int a = -1; uint b = 1; uint c = a + b;')
        self.assertIn('LET QASM2QCX_UINT_0 := :UINT:A1', native)
        self.assertIn('LET QASM2QCX_UINT_0 += B1', native)
        lines = convert('OPENQASM 3.0; uint[4] a = 15; uint[8] b = 1; uint[8] c = a + b;')
        self.assertIn('LET QASM2QCX_UINT_0 &= 255', lines)

    def test_constant_and_runtime_comparisons_use_the_same_promotions(self) -> None:
        maximum = qasm2qcx.QASM2QCXConverter.QCX_UINT_MAX
        lines = convert(f'''OPENQASM 3.0; const uint U = {maximum};
            const bool YES = U == -1; bool result = YES;
            uint a = U; if (a == -1) {{ result = true; }}''')
        self.assertIn('LET RESULT63 := 1', lines)
        self.assertIn(f'JUMPIF QASM2QCX_IF_0 A1 == {maximum}', lines)
        narrow = convert('''OPENQASM 3.0; uint[8] a = 255; bool result = a > -1;''')
        self.assertIn('LET QASM2QCX_INT_1 := :INT:A1', narrow)

    def test_mixed_non_native_signed_widths_are_diagnosed(self) -> None:
        for source in (
                'int[8] a = -1; uint[8] b = 1; int c = a + b;',
                'const int[8] A = -1; const uint[8] B = 1; int c = A + B;',
                'int[8] a = -1; uint b = 1; a += b;',
                'array[int[8], 1] a = {-1}; uint b = 1; uint c = a[0] + b;'):
            with self.subTest(source=source), self.assertRaises(qasm2qcx.UnsupportedOpenQASMError):
                convert('OPENQASM 3.0; ' + source)
        width = qasm2qcx.QASM2QCXConverter.QCX_UINT_WIDTH
        lines = convert(f'OPENQASM 3.0; int[{width}] a = -1; uint b = 1; uint c = a + b;')
        self.assertIn('LET QASM2QCX_UINT_0 := :UINT:A1', lines)

    def test_unsigned_boolean_literals_keep_scalar_compatibility(self) -> None:
        lines = convert('OPENQASM 3.0; bool a = uint(0); bool b = uint[1](1);')
        self.assertIn('LET A1 := 0', lines)
        self.assertIn('LET B1 := 1', lines)

    def test_casts_apply_target_width_and_preserve_integer_precision(self) -> None:
        maximum = qasm2qcx.QASM2QCXConverter.QCX_UINT_MAX
        lines = convert(f'''OPENQASM 3.0; uint a = uint({maximum});
            uint[8] b = uint[8](a); uint[1] c = uint[1](b);
            uint[8] d = uint[8](257.9); uint[8] e = uint[8](-1);''')
        self.assertIn(f'LET A1 := {maximum}', lines)
        self.assertIn('LET QASM2QCX_UINT_0 := :UINT:A1', lines)
        self.assertIn('LET QASM2QCX_UINT_0 &= 255', lines)
        self.assertIn('LET QASM2QCX_UINT_0 &= 1', lines)
        self.assertIn('LET D1 := 1', lines)
        self.assertIn('LET E1 := 255', lines)

    def test_constant_invalid_conversions_are_diagnosed(self) -> None:
        for value in ('-1.0', '1e999', str(qasm2qcx.QASM2QCXConverter.QCX_UINT_MAX + 1) + '.0'):
            with self.subTest(value=value), self.assertRaisesRegex(qasm2qcx.UnsupportedOpenQASMError, 'UINT range'):
                convert(f'OPENQASM 3.0; const uint a = uint({value});')
        with self.assertRaisesRegex(qasm2qcx.UnsupportedOpenQASMError, 'INT range'):
            convert('OPENQASM 3.0; const int a = int(uint(-1));')
        with self.assertRaisesRegex(qasm2qcx.UnsupportedOpenQASMError, 'fixed-width signed cast'):
            convert('OPENQASM 3.0; uint[8] a = 255; int b = int[8](a);')

    def test_runtime_invalid_literal_casts_stay_in_control_flow(self) -> None:
        lines = convert('''OPENQASM 3.0; uint a = 7; int b = 3;
            if (false) { a = uint(-1.0); b = int(uint(-1)); }''')
        self.assertGreater(lines.index('LET QASM2QCX_UINT_0 := :UINT::REAL:-1.0'), lines.index('@QASM2QCX_IF_0'))
        self.assertGreater(lines.index(f'LET QASM2QCX_INT_0 := :INT::UINT:{qasm2qcx.QASM2QCXConverter.QCX_UINT_MAX}'),
                           lines.index('@QASM2QCX_IF_0'))
        self.assertIn('LET OUT7 := 0', convert('''OPENQASM 3.0;
            const bool OFF = false && bool(uint(-1.0)); bool out = OFF;'''))


class UnsignedIntegrationTests(unittest.TestCase):
    def test_runtime_loop_operands_use_checked_signed_captures(self) -> None:
        for loop in ('for int i in [first:step:last] { total += i; }',
                     'for int i in {first, last, first} { total += i; }'):
            lines = convert('OPENQASM 3.0; uint[8] first = 1; uint last = 3; '
                            'uint step = 1; int total = 0; ' + loop)
            for name in ('FIRST31', 'LAST15'):
                self.assertTrue(any(line.startswith('LET QASM2QCX_INT_')
                                    and line.endswith(f':= :INT:{name}') for line in lines))
            if '[' in loop:
                self.assertTrue(any(line.endswith(':= :INT:STEP15') for line in lines))

    def test_constant_unsigned_loop_values_must_fit_signed_iterator(self) -> None:
        for loop in ('for int i in [U:U] {}', 'for int i in [0:U] {}',
                     'for int i in [0:U:0] {}', 'for int i in {U} {}',
                     'for int i in [1:0] { for int j in {U} {} }'):
            with self.subTest(loop=loop), self.assertRaisesRegex(
                    qasm2qcx.InvalidLoopRangeException, 'UINT value must fit QCX INT'):
                convert('OPENQASM 3.0; const uint U = uint(-1); ' + loop)
        maximum = qasm2qcx.QASM2QCXConverter.QCX_INT_MAX
        lines = convert(f'OPENQASM 3.0; const uint U = {maximum}; int a = 0; '
                        'for int i in {U} { a = i; }')
        self.assertIn(f'LET A1 := {maximum}', lines)

    def test_unsigned_runtime_indices_are_converted_before_bounds_checks(self) -> None:
        lines = convert('''OPENQASM 3.0; uint i = 1; array[int, 2] a = {4, 5};
            bit[2] flags = "01"; int out = a[i]; flags[i] = !flags[i];''')
        captures = [index for index, line in enumerate(lines) if line.endswith(':= :INT:I1')]
        self.assertEqual(len(captures), 3)
        for index in captures:
            self.assertTrue(lines[index + 1].startswith('ASSERT '))

    def test_static_unsigned_bit_indices_and_selections(self) -> None:
        lines = convert('''OPENQASM 3.0; const uint[8] I = 1; bit[3] flags = "101";
            bit out = flags[I]; flags[I] = true;
            bit[2] selected = flags[{uint[8](0), I + 1}];''')
        self.assertIn('LET OUT7 := FLAGS31:1', lines)
        self.assertIn('LET FLAGS31:1 := 1', lines)
        self.assertIn('LET SELECTED255:0 := FLAGS31:0', lines)
        self.assertIn('LET SELECTED255:1 := FLAGS31:2', lines)
        with self.assertRaises(qasm2qcx.InvalidBitOperandException):
            convert('OPENQASM 3.0; bit[2] flags; bit out = flags[uint(-1)];')

    def test_complex_to_uint_cast_in_runtime_index(self) -> None:
        lines = convert('''OPENQASM 3.0; complex z = 1.5 + 2im;
            array[int, 2] a = {4, 5}; int out = a[uint(z)];''')
        self.assertTrue(any(':= :UINT:Z1' in line for line in lines))

    def test_unsigned_quantum_angles_use_real_conversion(self) -> None:
        lines = convert('''OPENQASM 3.0; include "stdgates.inc"; qubit q;
            uint theta = 1; rx(theta) q; gphase(theta);''')
        self.assertIn('LET QASM2QCX_REAL_0 := :REAL:THETA31', lines)
        self.assertIn('EX 0 QASM2QCX_REAL_0', lines)
        self.assertIn('PHASE :REAL:THETA31', lines)

    def test_uint_iterators_and_dynamic_qubits_remain_unsupported(self) -> None:
        for source in ('for uint i in [0:1] {}',
                       'qubit[2] q; uint i = 1; reset q[i];'):
            with self.subTest(source=source), self.assertRaises(qasm2qcx.UnsupportedOpenQASMError):
                convert('OPENQASM 3.0; ' + source)


class CheckedNegationFoundationTests(unittest.TestCase):
    def setUp(self) -> None:
        converter = qasm2qcx.QASM2QCXConverter
        self.converter = UnsignedIntegerFoundationTests().converter(f'''
            const int N = {converter.QCX_UINT_WIDTH}; const int MINIMUM = {converter.QCX_INT_MIN};
            const int[8] SMALL = 1; int a; int[N] native; int[8] small;
            array[int, 2] values; array[int[8], 1] smalls;
            uint u; uint[4] narrow; bool ready; bit flag; bit[2] bits;
            float f; complex z;''')

    @staticmethod
    def expression(source: str):
        return qasm2qcx.openqasm3.parse('OPENQASM 3.0; int out = ' + source + ';').statements[0].init_expression

    def info(self, source: str, iterators=None):
        return self.converter._QASM2QCXConverter__negation_expression_info(self.expression(source), iterators)

    def test_numeric_types_promotions_and_dependencies(self) -> None:
        for source, expected, runtime in (
                ('-1', qasm2qcx.ValueType.INT, False), ('-MINIMUM', qasm2qcx.ValueType.INT, False),
                ('-a', qasm2qcx.ValueType.INT, True), ('-native', qasm2qcx.ValueType.INT, True),
                ('-values[a]', qasm2qcx.ValueType.INT, True),
                ('-u', qasm2qcx.ValueType.UINT, True), ('-narrow', qasm2qcx.ValueType.UINT, True),
                ('-uint[4](1)', qasm2qcx.ValueType.UINT, False),
                ('-(a + narrow)', qasm2qcx.ValueType.INT, True),
                ('-(a + u)', qasm2qcx.ValueType.UINT, True),
                ('-f', qasm2qcx.ValueType.FLOAT, True), ('-1.5', qasm2qcx.ValueType.FLOAT, False),
                ('-z', qasm2qcx.ValueType.COMPLEX, True), ('-1im', qasm2qcx.ValueType.COMPLEX, False)):
            with self.subTest(source=source):
                self.assertEqual(self.info(source), (expected, runtime))

    def test_minimum_literal_recognition_uses_ast_shape_not_folded_value(self) -> None:
        magnitude = -self.converter.QCX_INT_MIN
        check = self.converter._QASM2QCXConverter__is_minimum_int_literal
        for source in (f'-{magnitude}', f'-({magnitude})', f'-0x{magnitude:x}'):
            with self.subTest(source=source):
                self.assertTrue(check(self.expression(source)))
                self.assertEqual(self.info(source), (qasm2qcx.ValueType.INT, False))
        for source in ('-MINIMUM', f'-(-{magnitude})', f'-({magnitude - 1} + 1)',
                       f'-int({magnitude})', '-1', f'-{magnitude + 1}'):
            with self.subTest(source=source):
                self.assertFalse(check(self.expression(source)))

    def test_boundary_validation_accepts_representable_results_and_minimum_spelling(self) -> None:
        validate = self.converter._QASM2QCXConverter__validate_native_negation_value
        minimum, maximum = self.converter.QCX_INT_MIN, self.converter.QCX_INT_MAX
        for value in (minimum + 1, -1, 0, 1, maximum):
            validate(self.expression('-a'), value)
        validate(self.expression(str(minimum)), -minimum)
        for source in ('-a', '-MINIMUM', f'-({minimum})'):
            with self.subTest(source=source), self.assertRaisesRegex(
                    qasm2qcx.InvalidNegationException, 'result outside QCX INT range'):
                validate(self.expression(source), minimum)
        for value in (minimum - 1, maximum + 1, maximum + 2):
            with self.subTest(value=value), self.assertRaisesRegex(
                    qasm2qcx.InvalidNegationException, 'operand outside QCX INT range'):
                validate(self.expression('-a'), value)

    def test_invalid_types_and_non_native_signed_widths(self) -> None:
        for source in ('-ready', '-flag', '-bits', '-bits[0]', '-bits[0:0]',
                       '-values', '-small', '-SMALL', '-smalls[0]', '-int[8](a)',
                       '-values[f]'):
            with self.subTest(source=source), self.assertRaises(qasm2qcx.QASM2QCXError):
                self.info(source)
        with self.assertRaises(qasm2qcx.NoVariableNameException):
            self.info('-missing')

    def test_metadata_does_not_evaluate_values_arithmetic_or_indices(self) -> None:
        converter = self.converter
        before = (list(converter._QASM2QCXConverter__qcx_lines),
                  converter._QASM2QCXConverter__value, converter._QASM2QCXConverter__value_type,
                  converter._QASM2QCXConverter__value_kind, converter._QASM2QCXConverter__expression_kind,
                  converter._QASM2QCXConverter__evaluate_constant,
                  set(converter._QASM2QCXConverter__used_temporary_variables))
        for source in ('-MINIMUM', f'-({converter.QCX_INT_MIN})', '-(a / 0)', '-(a ** -1)',
                       '-values[999]', '-values[1 / 0]', '-(1 << -1)'):
            self.info(source)
        self.assertEqual(before, (list(converter._QASM2QCXConverter__qcx_lines),
                         converter._QASM2QCXConverter__value, converter._QASM2QCXConverter__value_type,
                         converter._QASM2QCXConverter__value_kind, converter._QASM2QCXConverter__expression_kind,
                         converter._QASM2QCXConverter__evaluate_constant,
                         set(converter._QASM2QCXConverter__used_temporary_variables)))

    def test_empty_loops_validate_types_without_evaluating_negation(self) -> None:
        convert(f'''OPENQASM 3.0; int a; array[int, 1] values;
            const int MINIMUM = {self.converter.QCX_INT_MIN};
            for int i in [1:0] {{ a = -MINIMUM; a = -values[999]; a = -(1 / 0); }}''')
        for declaration, source in (('int[8] small;', '-small'), ('bool b;', '-b'),
                                    ('bit[2] b;', '-b[0]'), ('int[8] small;', '-int[8](small)')):
            with self.subTest(source=source), self.assertRaises(qasm2qcx.QASM2QCXError):
                convert('OPENQASM 3.0; int a; ' + declaration
                        + ' for int i in [1:0] { a = ' + source + '; }')

    def test_iterator_shadowing_and_declaration_scope_widths(self) -> None:
        self.assertEqual(self.info('-small', {'small': qasm2qcx.ValueType.INT}),
                         (qasm2qcx.ValueType.INT, False))
        with self.assertRaises(qasm2qcx.UnsupportedOpenQASMError):
            self.info('-small', {'small': qasm2qcx.ValueType.BIT})
        self.converter._QASM2QCXConverter__loop_bindings.append({'N': 1})
        try:
            self.assertEqual(self.info('-native'), (qasm2qcx.ValueType.INT, True))
        finally:
            self.converter._QASM2QCXConverter__loop_bindings.pop()
        with self.assertRaises(qasm2qcx.InvalidDeclarationException):
            self.info('-int[N](a)', {'N': qasm2qcx.ValueType.INT})

    def test_public_paths_preserve_valid_values_and_reject_unsupported_widths(self) -> None:
        minimum = self.converter.QCX_INT_MIN
        lines = convert(f'''OPENQASM 3.0; const int C = {minimum}; int a = C;
            uint[4] u = -uint[4](1); float f = -1.5; complex z = -1im;''')
        self.assertIn(f'LET A1 := {minimum}', lines)
        self.assertIn('LET U1 := 15', lines)
        # Do not impose the stricter loop-metadata cast subset on ordinary
        # float/complex negation expressions that the converter already accepts.
        convert('''OPENQASM 3.0; complex z = 1 + 2im;
            int a = -int(z); float f = -float(z); complex w = -z;''')
        for wrapper in ('{}', 'if (false) {{ {} }}', 'for int i in [1:0] {{ {} }}'):
            with self.subTest(wrapper=wrapper), self.assertRaisesRegex(
                    qasm2qcx.UnsupportedOpenQASMError, 'non-native signed width'):
                convert('OPENQASM 3.0; int[8] small = 1; int out; ' + wrapper.format('out = -small;'))

    def test_failed_native_width_validation_releases_computed_operands(self) -> None:
        program = qasm2qcx.openqasm3.parse('OPENQASM 3.0; int[8] small = 1; int out = -(small + 1);')
        converter = qasm2qcx.QASM2QCXConverter(program)
        with self.assertRaisesRegex(qasm2qcx.UnsupportedOpenQASMError, 'non-native signed width'):
            converter.visit(program)
        self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())


class CheckedNegationExpressionTests(unittest.TestCase):
    def test_minimum_literal_spellings_and_valid_constant_boundaries(self) -> None:
        minimum = qasm2qcx.QASM2QCXConverter.QCX_INT_MIN
        maximum = qasm2qcx.QASM2QCXConverter.QCX_INT_MAX
        for source, expected in ((str(minimum), minimum), (f'-({-minimum})', minimum),
                                 (f'-0x{-minimum:x}', minimum), ('-0', 0), ('-1', -1),
                                 (f'-({minimum + 1})', maximum), (f'-({maximum})', -maximum)):
            with self.subTest(source=source):
                lines = convert(f'OPENQASM 3.0; const int C = {source}; int out = C;')
                self.assertIn(f'LET OUT7 := {expected}', lines)
                self.assertFalse(any(line.startswith('ASSERT ') for line in lines))

    def test_constant_minimum_value_and_nested_negation_overflow(self) -> None:
        minimum = qasm2qcx.QASM2QCXConverter.QCX_INT_MIN
        for source in ('-MINIMUM', f'-({minimum})', f'-(-({minimum}))'):
            with self.subTest(source=source), self.assertRaisesRegex(
                    qasm2qcx.InvalidNegationException, 'result outside QCX INT range'):
                convert(f'OPENQASM 3.0; const int MINIMUM = {minimum}; const int BAD = {source};')

    def test_unrepresentable_inputs_are_conversion_errors_not_wrapped_literals(self) -> None:
        magnitude = -qasm2qcx.QASM2QCXConverter.QCX_INT_MIN
        for source in (f'-{magnitude + 1}', f'-({magnitude - 1} + 1)', f'-int({magnitude})'):
            for wrapper in ('int out = {};', 'const int C = {};', 'int out; if (false) {{ out = {}; }}'):
                with self.subTest(source=source, wrapper=wrapper), self.assertRaisesRegex(
                        qasm2qcx.InvalidNegationException, 'operand outside QCX INT range'):
                    convert('OPENQASM 3.0; ' + wrapper.format(source))

    def test_constant_short_circuit_suppresses_value_checks_not_type_checks(self) -> None:
        minimum = qasm2qcx.QASM2QCXConverter.QCX_INT_MIN
        lines = convert(f'''OPENQASM 3.0; const int MINIMUM = {minimum};
            const bool OFF = false && bool(-MINIMUM);
            const bool ON = true || bool(-({minimum}));
            bool a = OFF; bool b = ON;''')
        self.assertIn('LET A1 := 0', lines)
        self.assertIn('LET B1 := 1', lines)
        with self.assertRaises(qasm2qcx.UnsupportedOpenQASMError):
            convert('OPENQASM 3.0; const bool OFF = false && bool(-true);')

    def test_runtime_operand_is_captured_and_guarded_before_multiplication(self) -> None:
        minimum = qasm2qcx.QASM2QCXConverter.QCX_INT_MIN
        lines = convert('OPENQASM 3.0; int a = 2; int out = -a;')
        capture = next(index for index, line in enumerate(lines) if line.endswith(':= A1'))
        temporary = lines[capture].split()[1]
        self.assertEqual(lines[capture + 1], f'ASSERT {temporary} \\= {minimum}')
        self.assertEqual(lines[capture + 2], f'LET {temporary} *= -1')
        self.assertEqual(lines[-1], f'LET OUT7 := {temporary}')

    def test_literal_runtime_result_overflow_is_deferred_without_bad_int_literal(self) -> None:
        minimum = qasm2qcx.QASM2QCXConverter.QCX_INT_MIN
        lines = convert(f'OPENQASM 3.0; int out = -({minimum});')
        self.assertTrue(any(line.startswith('ASSERT ') for line in lines))
        self.assertFalse(any(line.endswith(f':= {-minimum}') for line in lines))
        convert(f'''OPENQASM 3.0; int out = 0;
            if (false) {{ out = -({minimum}); }}
            bool off = false && bool(-({minimum}));''')

    def test_indexed_reads_and_computed_operands_release_temporaries(self) -> None:
        program = qasm2qcx.openqasm3.parse('''OPENQASM 3.0; array[int, 2] v = {1, 2};
            int i = 0; int out = -v[i + 1]; out = -(out + v[i]);''')
        converter = qasm2qcx.QASM2QCXConverter(program)
        converter.visit(program)
        lines = list(converter)
        self.assertEqual(sum(line.endswith(' += 1') for line in lines), 1)
        self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())
        ForLoopControlLoweringTests.assert_resolved_jumps(lines)

    def test_unsigned_float_and_complex_lowering_do_not_get_int_guards(self) -> None:
        lines = convert('''OPENQASM 3.0; uint[4] u = 1; uint[4] v = -u;
            float f = 1.5; float g = -f; complex z = 1 + 2im; complex w = -z;''')
        self.assertFalse(any(line.startswith('ASSERT ') for line in lines))
        self.assertTrue(any(' *= :UINT:-1' in line for line in lines))
        self.assertTrue(any(' *= :COMPLEX:-1.0' in line for line in lines))


class CheckedNegationIntegrationTests(unittest.TestCase):
    def test_rejected_bit_and_boolean_operands_release_computed_values(self) -> None:
        for source in ('bit a = 1; bit b = 0; int out = -(a & b);',
                       'int a = 1; int out = -bool(a);'):
            with self.subTest(source=source):
                program = qasm2qcx.openqasm3.parse('OPENQASM 3.0; ' + source)
                converter = qasm2qcx.QASM2QCXConverter(program)
                with self.assertRaises(qasm2qcx.UnsupportedOpenQASMError):
                    converter.visit(program)
                self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())

    def test_cli_constant_negation_errors_have_no_partial_output(self) -> None:
        minimum = qasm2qcx.QASM2QCXConverter.QCX_INT_MIN
        for expression, diagnostic in ((f'-({minimum})', 'result outside QCX INT range'),
                                       (f'-({-minimum - 1} + 1)', 'operand outside QCX INT range')):
            source = f'OPENQASM 3.0; qubit q; const int BAD = {expression};'
            with self.subTest(expression=expression), patch('builtins.open', mock_open(read_data=source)), \
                    patch('builtins.print') as output:
                with self.assertRaises(SystemExit) as error:
                    qasm2qcx.main(['negation.qasm'])
                self.assertTrue(str(error.exception).startswith('qasm2qcx.py:'))
                self.assertIn(diagnostic, str(error.exception))
                output.assert_not_called()

    def test_failed_nested_operand_releases_only_its_own_temporaries(self) -> None:
        program = qasm2qcx.openqasm3.parse('''OPENQASM 3.0; int a = 2;
            int out = -(later + -a); int later;''')
        converter = qasm2qcx.QASM2QCXConverter(program)
        reserved = converter._QASM2QCXConverter__add_new_temporary_variable(qasm2qcx.ValueType.INT)
        with self.assertRaises(qasm2qcx.NoVariableNameException):
            converter.visit(program)
        self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, {reserved})

    def test_failed_condition_releases_guarded_rhs_without_releasing_caller_storage(self) -> None:
        for statement in ('bool out = -later == -a;', 'if (-later == -a) {}',
                          'while (-later == -a) {}'):
            with self.subTest(statement=statement):
                program = qasm2qcx.openqasm3.parse('OPENQASM 3.0; int a = 2; ' + statement + ' int later;')
                converter = qasm2qcx.QASM2QCXConverter(program)
                reserved = converter._QASM2QCXConverter__add_new_temporary_variable(qasm2qcx.ValueType.INT)
                with self.assertRaises(qasm2qcx.NoVariableNameException):
                    converter.visit(program)
                self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, {reserved})

    def test_cast_order_controls_whether_int_negation_is_checked(self) -> None:
        for declaration, checked, widened in (
                ('float out;', 'out = float(-a);', 'out = -float(a);'),
                ('uint out;', 'out = uint(-a);', 'out = -uint(a);'),
                ('complex out;', 'out = complex(-a);', 'out = -complex(a);')):
            prefix = 'OPENQASM 3.0; int a = 2; ' + declaration
            with self.subTest(declaration=declaration):
                self.assertEqual(sum(line.startswith('ASSERT ') for line in convert(prefix + checked)), 1)
                self.assertFalse(any(line.startswith('ASSERT ') for line in convert(prefix + widened)))

    def test_indexed_assignment_captures_destination_and_releases_storage(self) -> None:
        program = qasm2qcx.openqasm3.parse('''OPENQASM 3.0; array[int, 2] v = {2, 3};
            int i = 0; v[i + 1] = -v[i]; v[i] += -v[i + 1];''')
        converter = qasm2qcx.QASM2QCXConverter(program)
        converter.visit(program)
        lines = list(converter)
        self.assertEqual(sum(line.endswith(' += 1') for line in lines), 2)
        self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())

    def test_nested_values_conditions_and_loop_headers_have_clean_storage(self) -> None:
        width = qasm2qcx.QASM2QCXConverter.QCX_UINT_WIDTH
        program = qasm2qcx.openqasm3.parse(f'''OPENQASM 3.0; const int N = {width};
            int[N] a = -2; int step = -1; int total = 0; array[int, 2] v = {{1, 2}};
            bool same = -(-a) == a; same = !bool(-a) || (-a > 0 && same);
            for int N in [0:1] {{ total += -a; }}
            for int i in [0:-step:2] {{ total += -i; }}
            for int j in {{-a, -step}} {{ total += -j; }}
            while (-step > 0) {{ step = -step; if (-a == 2) {{ break; }} }}
            total += v[-step + 1];''')
        converter = qasm2qcx.QASM2QCXConverter(program)
        converter.visit(program)
        self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())
        self.assertEqual(converter._QASM2QCXConverter__loop_bindings, [])
        ForLoopControlLoweringTests.assert_resolved_jumps(list(converter))

    def test_storage_is_declared_before_a_skipped_first_use(self) -> None:
        minimum = qasm2qcx.QASM2QCXConverter.QCX_INT_MIN
        lines = convert(f'''OPENQASM 3.0; int a = {minimum}; bool off = false && bool(-a);
            int out = 0; if (false) {{ out = -a; }} out = -out;''')
        first_jump = next(index for index, line in enumerate(lines) if line.startswith('JUMP '))
        self.assertTrue(all(index < first_jump for index, line in enumerate(lines)
                            if line.startswith('VAR QASM2QCX_INT_')))
        ForLoopControlLoweringTests.assert_resolved_jumps(lines)

    def test_empty_loop_nested_validation_does_not_evaluate_bad_values(self) -> None:
        minimum = qasm2qcx.QASM2QCXConverter.QCX_INT_MIN
        convert(f'''OPENQASM 3.0; int a = {minimum}; array[int, 1] v;
            for int i in [1:0] {{
                while (bool(-a)) {{ a = -v[999]; }}
                if (false && bool(-(1 / 0))) {{ a = -(-a); }}
            }}''')
        with self.assertRaises(qasm2qcx.UnsupportedOpenQASMError):
            convert('''OPENQASM 3.0; int[8] small; int a;
                for int i in [1:0] { while (false) { a = -small; } }''')

    def test_bit_iterators_need_explicit_numeric_cast_and_remain_read_only(self) -> None:
        convert('OPENQASM 3.0; bit[2] flags; int a; for bit b in flags { a = -int(b); }')
        for body in ('a = -b;', 'b = -int(b);'):
            with self.subTest(body=body), self.assertRaises(qasm2qcx.QASM2QCXError):
                convert('OPENQASM 3.0; bit[2] flags; int a; for bit b in flags { ' + body + ' }')


class IntegerPowerFoundationTests(unittest.TestCase):
    def setUp(self) -> None:
        width = qasm2qcx.QASM2QCXConverter.QCX_UINT_WIDTH
        self.converter = UnsignedIntegerFoundationTests().converter(f'''
            const int C = 3; const uint[4] U = 2;
            int n; int m; int[8] small; int[{width}] native;
            uint u; uint[4] narrow; uint[8] wider;
            bool ready; bit flag; bit[2] bits; float f; complex z;
            array[int, 2] values; array[int[8], 1] smalls;''')

    @staticmethod
    def expression(source: str):
        return qasm2qcx.openqasm3.parse('OPENQASM 3.0; int out = ' + source + ';').statements[0].init_expression

    def info(self, source: str, iterators=None):
        return self.converter._QASM2QCXConverter__power_expression_info(self.expression(source), iterators)

    def test_parser_precedence_associativity_and_compound_assignment(self) -> None:
        expression = self.expression('2 ** 3 ** 2')
        self.assertEqual(expression.op.name, '**')
        self.assertEqual(expression.rhs.op.name, '**')
        expression = self.expression('-2 ** 2')
        self.assertEqual(expression.op.name, '-')
        self.assertEqual(expression.expression.op.name, '**')
        statement = qasm2qcx.openqasm3.parse('OPENQASM 3.0; int n; n **= 2;').statements[1]
        self.assertEqual(statement.op.name, '**=')

    def test_signed_result_types_and_runtime_dependencies(self) -> None:
        for source, runtime in (('2 ** 3', False), ('C ** 0', False),
                                ('0 ** 0', False), ('(-2) ** -1', False),
                                ('n ** C', True), ('C ** m', True),
                                ('values[999] ** n', True), ('native ** 2', True),
                                ('int(narrow) ** m', True)):
            with self.subTest(source=source):
                self.assertEqual(self.info(source), (qasm2qcx.ValueType.INT, None, runtime))

    def test_unsigned_widths_and_mixed_promotion_match_arithmetic(self) -> None:
        width = self.converter.QCX_UINT_WIDTH
        for source, result_type, uint_type, runtime in (
                ('narrow ** wider', qasm2qcx.ValueType.UINT, qasm2qcx._UnsignedIntegerType(8, True), True),
                ('uint[4](2) ** U', qasm2qcx.ValueType.UINT, qasm2qcx._UnsignedIntegerType(4, True), False),
                ('u ** n', qasm2qcx.ValueType.UINT, qasm2qcx._UnsignedIntegerType(width, False), True),
                ('n ** u', qasm2qcx.ValueType.UINT, qasm2qcx._UnsignedIntegerType(width, False), True),
                ('narrow ** n', qasm2qcx.ValueType.INT, None, True),
                ('n ** narrow', qasm2qcx.ValueType.INT, None, True),
                ('(n + narrow) ** 2', qasm2qcx.ValueType.INT, None, True),
                ('(narrow << n) ** uint[4](2)', qasm2qcx.ValueType.UINT,
                 qasm2qcx._UnsignedIntegerType(4, True), True),
                ('(narrow ** wider) ** U', qasm2qcx.ValueType.UINT,
                 qasm2qcx._UnsignedIntegerType(8, True), True)):
            with self.subTest(source=source):
                self.assertEqual(self.info(source), (result_type, uint_type, runtime))

    def test_invalid_types_registers_and_signed_widths_rejected(self) -> None:
        for source in ('ready ** 2', 'n ** ready', 'flag ** 2', 'bits[0] ** 2',
                       'bits ** 2', '"01" ** 2', 'n ** bits[0:0]', 'f ** n',
                       'n ** z', 'values ** n', 'small ** 2', '2 ** small',
                       'smalls[0] ** 2', 'int[8](2) ** 2', 'n ** uint[1 - 1](2)',
                       'n ** uint[1000](2)', 'values[f] ** 2'):
            with self.subTest(source=source), self.assertRaises(qasm2qcx.QASM2QCXError):
                self.info(source)
        with self.assertRaises(qasm2qcx.NoVariableNameException):
            self.info('n ** missing')

    def test_metadata_never_evaluates_power_count_or_index_arithmetic(self) -> None:
        converter = self.converter
        before = (list(converter._QASM2QCXConverter__qcx_lines),
                  converter._QASM2QCXConverter__value, converter._QASM2QCXConverter__value_type,
                  converter._QASM2QCXConverter__value_kind, converter._QASM2QCXConverter__expression_kind,
                  converter._QASM2QCXConverter__evaluate_constant,
                  set(converter._QASM2QCXConverter__used_temporary_variables))
        for source in ('2 ** -1', '2 ** (1 / 0)', 'n ** values[999]',
                       'values[1 / 0] ** n', '2 ** 999999999999999999999999999999',
                       '(n << -1) ** 2', 'uint[4](2) ** uint[4](3)'):
            self.info(source)
        self.assertEqual(before, (list(converter._QASM2QCXConverter__qcx_lines),
                         converter._QASM2QCXConverter__value, converter._QASM2QCXConverter__value_type,
                         converter._QASM2QCXConverter__value_kind, converter._QASM2QCXConverter__expression_kind,
                         converter._QASM2QCXConverter__evaluate_constant,
                         set(converter._QASM2QCXConverter__used_temporary_variables)))

    def test_iterator_shadowing_uses_typed_scalar_metadata(self) -> None:
        self.assertEqual(self.info('narrow ** 2', {'narrow': qasm2qcx.ValueType.INT}),
                         (qasm2qcx.ValueType.INT, None, False))
        with self.assertRaises(qasm2qcx.UnsupportedOpenQASMError):
            self.info('narrow ** 2', {'narrow': qasm2qcx.ValueType.BIT})
        self.assertEqual(self.info('small ** 2', {'small': qasm2qcx.ValueType.INT}),
                         (qasm2qcx.ValueType.INT, None, False))

    def test_empty_loop_validation_defers_values_but_checks_types(self) -> None:
        for expression in ('i ** -1', '2 ** (1 / 0)', 'values[999] ** i', 'n ** 999999999999999999'):
            convert('OPENQASM 3.0; int n; array[int, 1] values; '
                    'for int i in [1:0] { n = ' + expression + '; }')
        for expression in ('i ** true', '1.0 ** i', 'missing ** i'):
            with self.subTest(expression=expression), self.assertRaises(qasm2qcx.QASM2QCXError):
                convert('OPENQASM 3.0; int n; for int i in [1:0] { n = ' + expression + '; }')

    def test_loop_metadata_and_reached_paths_validate_integer_operands(self) -> None:
        expression = self.expression('n ** m')
        self.assertEqual(self.converter._QASM2QCXConverter__loop_expression_info(expression),
                         (qasm2qcx.ValueType.INT, True))
        self.assertIn('LET N1 := 8', convert('OPENQASM 3.0; int n = 2 ** 3;'))
        with self.assertRaisesRegex(qasm2qcx.UnsupportedOpenQASMError, 'requires scalar int or uint'):
            convert('OPENQASM 3.0; int n = int(true ** 3);')


class IntegerPowerExpressionTests(unittest.TestCase):
    def test_constant_folding_precedence_and_endpoints(self) -> None:
        converter = qasm2qcx.QASM2QCXConverter
        for expression, expected in (('0 ** 0', 1), ('0 ** 7', 0), ('(-2) ** 3', -8),
                                     ('-2 ** 2', -4), ('2 ** 3 ** 2', 512),
                                     (f'(-2) ** {converter.QCX_UINT_WIDTH - 1}', converter.QCX_INT_MIN),
                                     (f'{converter.QCX_INT_MAX} ** 1', converter.QCX_INT_MAX),
                                     (f'({converter.QCX_INT_MIN}) ** 1', converter.QCX_INT_MIN),
                                     (f'(-1) ** {converter.QCX_INT_MAX}', -1)):
            with self.subTest(expression=expression):
                lines = convert(f'OPENQASM 3.0; const int C = {expression}; int out = C;')
                self.assertIn(f'LET OUT7 := {expected}', lines)

    def test_unsigned_folding_and_promoted_result_width(self) -> None:
        maximum = qasm2qcx.QASM2QCXConverter.QCX_UINT_MAX
        for expression, expected in (('uint[4](3) ** uint[4](4)', 1),
                                     ('uint[4](3) ** 4', 81),
                                     ('(-2) ** uint[4](3)', -8),
                                     ('uint[4](3) ** uint[8](4)', 81),
                                     (f'uint({maximum}) ** uint(2)', 1),
                                     ('uint(2) ** uint(-1)', 0)):
            with self.subTest(expression=expression):
                self.assertIn(f'LET OUT7 := {expected}', convert(
                    f'OPENQASM 3.0; const int C = int({expression}); int out = C;'))

    def test_constant_domain_and_range_errors(self) -> None:
        width = qasm2qcx.QASM2QCXConverter.QCX_UINT_WIDTH
        maximum = qasm2qcx.QASM2QCXConverter.QCX_INT_MAX
        for expression in ('2 ** -1', 'uint(2) ** -1', '0 ** -1',
                           f'2 ** {width - 1}', f'(-2) ** {width}', f'{maximum} ** 2'):
            with self.subTest(expression=expression), self.assertRaises(qasm2qcx.InvalidPowerException):
                convert(f'OPENQASM 3.0; const int C = int({expression});')

    def test_skipped_constant_errors_are_not_evaluated_but_types_are_checked(self) -> None:
        lines = convert('''OPENQASM 3.0;
            const bool OFF = false && bool(uint(2) ** -1);
            const bool ON = true || bool(2147483647 ** 2);
            bool a = OFF; bool b = ON;''')
        self.assertIn('LET A1 := 0', lines)
        self.assertIn('LET B1 := 1', lines)
        with self.assertRaises(qasm2qcx.UnsupportedOpenQASMError):
            convert('OPENQASM 3.0; const bool BAD = false && bool(true ** 2);')

    def test_runtime_domain_check_precedes_unsigned_conversion(self) -> None:
        lines = convert('OPENQASM 3.0; uint a = 2; int n = -1; uint out = a ** n;')
        check = lines.index('ASSERT N1 >= 0')
        conversion = next(index for index, line in enumerate(lines) if line.endswith(':= :UINT:N1'))
        self.assertLess(check, conversion)
        self.assertTrue(any(' >>= 1' in line for line in lines))

    def test_literal_runtime_failures_are_deferred_to_reached_paths(self) -> None:
        lines = convert('''OPENQASM 3.0; int out = 7;
            if (false) { out = 2 ** -1; out = 2147483647 ** 2; }
            bool off = false && bool(2 ** -1);''')
        self.assertTrue(any(line.startswith('ASSERT ') for line in lines))
        ForLoopControlLoweringTests.assert_resolved_jumps(lines)

    def test_runtime_uint_width_is_preserved_and_each_product_is_masked(self) -> None:
        lines = convert('OPENQASM 3.0; uint[4] a = 3; uint[4] n = 4; uint[4] out = a ** n;')
        for index, line in enumerate(lines):
            if ' *= ' in line:
                self.assertEqual(lines[index + 1], f'LET {line.split()[1]} &= 15')
        self.assertEqual(sum(' *= ' in line for line in lines), 2)

    def test_operands_are_evaluated_left_to_right_and_storage_is_released(self) -> None:
        program = qasm2qcx.openqasm3.parse('''OPENQASM 3.0; int i = 0; int j = 1;
            array[int, 2] a = {2, 3}; int out = a[i] ** a[j];
            out = (out ** a[j]) ** (a[i] ** 1);''')
        converter = qasm2qcx.QASM2QCXConverter(program)
        converter.visit(program)
        lines = list(converter)
        left = next(index for index, line in enumerate(lines) if line.endswith(':= I1'))
        right = next(index for index, line in enumerate(lines) if line.endswith(':= J1'))
        self.assertLess(left, right)
        self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())
        ForLoopControlLoweringTests.assert_resolved_jumps(lines)

    def test_failed_rhs_releases_lhs_temporaries(self) -> None:
        program = qasm2qcx.openqasm3.parse('OPENQASM 3.0; int n = 2; int out = (n ** 2) ** later; int later;')
        converter = qasm2qcx.QASM2QCXConverter(program)
        with self.assertRaises(qasm2qcx.QASM2QCXError):
            converter.visit(program)
        self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())

    def test_signed_literal_inputs_must_fit_even_when_the_result_would_fit(self) -> None:
        maximum = qasm2qcx.QASM2QCXConverter.QCX_INT_MAX
        for expression in (f'{maximum + 1} ** 0', f'1 ** {maximum + 1}'):
            with self.subTest(expression=expression), self.assertRaisesRegex(
                    qasm2qcx.UnsupportedOpenQASMError, 'operand outside QCX INT range'):
                convert(f'OPENQASM 3.0; int x = {expression};')


class IntegerPowerAssignmentTests(unittest.TestCase):
    def test_scalar_compound_assignments_produce_completed_results(self) -> None:
        for value_type in ('int', 'uint', 'uint[4]'):
            for rhs in ('2', 'n', 'a', 'a ** n', 'uint[4](n)'):
                with self.subTest(value_type=value_type, rhs=rhs):
                    lines = convert(f'OPENQASM 3.0; {value_type} a = 2; int n = 2; a **= {rhs};')
                    self.assertTrue(lines[-1].startswith('LET A1 '))
                    self.assertFalse(any('**' in line for line in lines))
                    ForLoopControlLoweringTests.assert_resolved_jumps(lines)

    def test_mixed_compound_types_match_expression_promotion(self) -> None:
        signed = convert('OPENQASM 3.0; int a = -2; uint[4] n = 3; a **= n;')
        self.assertTrue(signed[-1].startswith('LET A1 := QASM2QCX_INT_'))
        unsigned = convert('OPENQASM 3.0; int a = -2; uint n = 3; a **= n;')
        self.assertTrue(unsigned[-1].startswith('LET A1 := :INT:QASM2QCX_UINT_'))
        self.assertFalse(any(' /= ' in line for line in unsigned))

    def test_index_is_captured_once_and_reserved_until_final_write(self) -> None:
        lines = convert('OPENQASM 3.0; array[int, 2] a = {2, 3}; int i = -1; a[i + 1] **= 2;')
        self.assertEqual(sum(line.endswith(' += 1') for line in lines), 1)
        checks = [line for line in lines if line.startswith('ASSERT ')]
        index = checks[0].split()[1]
        self.assertEqual(checks[1], f'ASSERT {index} < 2')
        self.assertTrue(lines[-1].startswith(f'LET A1:{index} := QASM2QCX_INT_'))
        snapshots = [line for line in lines if line.endswith(f':= A1:{index}')]
        self.assertEqual(len(snapshots), 1)
        self.assertNotEqual(snapshots[0].split()[1], index)

    def test_target_snapshot_precedes_rhs_reads_and_checks(self) -> None:
        lines = convert('OPENQASM 3.0; int a = 2; array[int, 1] v = {3}; int i = 0; a **= v[i];')
        snapshot = next(index for index, line in enumerate(lines) if line.endswith(':= A1'))
        rhs = next(index for index, line in enumerate(lines) if line.endswith(':= I1'))
        self.assertLess(snapshot, rhs)
        self.assertTrue(lines[-1].startswith('LET A1 := QASM2QCX_INT_'))

    def test_self_references_static_and_runtime_indices_release_temporaries(self) -> None:
        program = qasm2qcx.openqasm3.parse('''OPENQASM 3.0; int a = 2; int n = 2;
            uint[4] u = 3; array[int, 2] v = {2, 3}; int i = 0;
            a **= a; u **= uint[4](u); v[1] **= v[0];
            v[i] **= v[v[i] - 2] ** n;''')
        converter = qasm2qcx.QASM2QCXConverter(program)
        converter.visit(program)
        self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())
        ForLoopControlLoweringTests.assert_resolved_jumps(list(converter))

    def test_failed_rhs_and_unrepresentable_literals_release_snapshots(self) -> None:
        maximum = qasm2qcx.QASM2QCXConverter.QCX_INT_MAX
        for source in ('int a = 2; a **= later; int later;',
                       f'int a = 2; a **= {maximum + 1};'):
            program = qasm2qcx.openqasm3.parse('OPENQASM 3.0; ' + source)
            converter = qasm2qcx.QASM2QCXConverter(program)
            with self.assertRaises(qasm2qcx.QASM2QCXError):
                converter.visit(program)
            self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())

    def test_invalid_types_and_widths_rejected_even_on_skipped_paths(self) -> None:
        for declaration, body in (
                ('bool a;', 'a **= 2;'), ('bit a;', 'a **= 2;'),
                ('bit[2] a;', 'a **= 2;'), ('float a;', 'a **= 2;'),
                ('complex a;', 'a **= 2;'), ('int a;', 'a **= true;'),
                ('int a;', 'a **= 2.0;'), ('int[8] a;', 'a **= 2;'),
                ('int a; int[8] n;', 'a **= n;'),
                ('array[int[8], 1] a;', 'a[0] **= 2;'),
                ('array[int, 1] a; float f;', 'a[f] **= 2;'),
                ('array[int, 1] a;', 'a **= 2;')):
            for wrapper in ('{}', 'if (false) {{ {} }}', 'for int i in [1:0] {{ {} }}'):
                with self.subTest(declaration=declaration, wrapper=wrapper), self.assertRaises(qasm2qcx.QASM2QCXError):
                    convert('OPENQASM 3.0; ' + declaration + wrapper.format(body))

    def test_empty_loops_check_types_without_evaluating_values(self) -> None:
        convert('''OPENQASM 3.0; int a; array[int, 1] v;
            for int i in [1:0] { v[999] **= -1; a **= 2 ** (1 / 0); }''')
        for body in ('i **= 2;', 'while (false) { i **= 2; }'):
            with self.assertRaises(qasm2qcx.UnsupportedOpenQASMError):
                convert('OPENQASM 3.0; for int i in [1:0] { ' + body + ' }')

    def test_nested_control_flow_has_resolved_labels_and_clean_storage(self) -> None:
        program = qasm2qcx.openqasm3.parse('''OPENQASM 3.0; int a = 2; int n = 2;
            uint[4] u = 3; int total = 0; array[int, 4] v = {0, 1, 2, 3};
            if (false && bool(a ** -1)) { a **= -1; }
            for int i in [0:n ** 2] {
                while (bool(a ** n) && a < 100) {
                    a **= n; if (a ** 0 == 1) { break; }
                }
                total += v[(i ** 2) % 4];
                for int j in {i ** 0, n ** i} { u **= uint[4](j); }
            }''')
        converter = qasm2qcx.QASM2QCXConverter(program)
        converter.visit(program)
        lines = list(converter)
        self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())
        self.assertEqual(converter._QASM2QCXConverter__loop_bindings, [])
        ForLoopControlLoweringTests.assert_resolved_jumps(lines)

    def test_unsigned_declaration_width_survives_iterator_shadowing(self) -> None:
        for bounds in ('1:0', '1:2'):
            lines = convert('''OPENQASM 3.0; const int N = 4; uint[N] a = 3;
                const uint[N] E = 2; uint[N] b = 0;
                for int N in [''' + bounds + '''] { a **= E; b = a ** E; }''')
            ForLoopControlLoweringTests.assert_resolved_jumps(lines)
        with self.assertRaises(qasm2qcx.InvalidDeclarationException):
            convert('''OPENQASM 3.0; const int N = 4; uint[N] a = 3;
                for int N in [1:0] { a **= uint[N](2); }''')

    def test_signed_declaration_width_survives_iterator_shadowing(self) -> None:
        width = qasm2qcx.QASM2QCXConverter.QCX_UINT_WIDTH
        for bounds in ('1:0', '1:2'):
            program = qasm2qcx.openqasm3.parse(f'''OPENQASM 3.0; const int N = {width};
                int[N] a = 2; array[int[N], 1] v = {{2}};
                for int N in [{bounds}] {{ a **= 2; v[0] **= 2; }}''')
            converter = qasm2qcx.QASM2QCXConverter(program)
            converter.visit(program)
            self.assertEqual(converter._QASM2QCXConverter__loop_bindings, [])
            self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())
        with self.assertRaises(qasm2qcx.InvalidDeclarationException):
            convert(f'''OPENQASM 3.0; const int N = {width}; int a = 2;
                for int N in [1:0] {{ a **= int[N](2); }}''')


class ScalarShiftExpressionTests(unittest.TestCase):
    def test_signed_constant_folding_and_endpoints(self) -> None:
        width = qasm2qcx.QASM2QCXConverter.QCX_UINT_WIDTH
        minimum = qasm2qcx.QASM2QCXConverter.QCX_INT_MIN
        lines = convert(f'''OPENQASM 3.0; const int A = -3 >> 1;
            int a = A; int b = -1 << {width - 1}; int c = {minimum} >> {width - 1};
            int d = 3 << 1 + 1; int e = 32 >> 1 >> 2;''')
        for name, value in (('A1', -2), ('B1', minimum), ('C1', -1), ('D1', 12), ('E1', 4)):
            self.assertIn(f'LET {name} := {value}', lines)

    def test_unsigned_folding_retains_left_width(self) -> None:
        lines = convert('''OPENQASM 3.0; const uint[8] A = uint[8](255) << 1;
            uint[8] a = A; uint[8] b = uint[8](128) >> uint(7);
            uint[4] c = (uint[4](15) << 1) >> 1;''')
        for name, value in (('A1', 254), ('B1', 1), ('C1', 7)):
            self.assertIn(f'LET {name} := {value}', lines)

    def test_runtime_signed_shift_count_does_not_promote_result(self) -> None:
        lines = convert('OPENQASM 3.0; int a = -3; uint count = 1; int out = a >> count;')
        self.assertIn('LET QASM2QCX_INT_0 := A1', lines)
        self.assertIn('LET QASM2QCX_INT_0 >>= COUNT31', lines)
        self.assertFalse(any(line.startswith('VAR QASM2QCX_UINT_') for line in lines))

    def test_runtime_narrow_shift_has_count_guards_and_result_mask(self) -> None:
        lines = convert('OPENQASM 3.0; uint[8] a = 255; int n = 1; uint[8] out = a << n;')
        self.assertIn('ASSERT N1 >= 0', lines)
        self.assertIn('ASSERT N1 < 8', lines)
        self.assertIn('LET QASM2QCX_UINT_0 <<= N1', lines)
        self.assertIn('LET QASM2QCX_UINT_0 &= 255', lines)
        self.assertFalse(any(line.startswith('VAR QASM2QCX_INT_') for line in lines))
        literal = convert('OPENQASM 3.0; uint[4] a = 15; uint[4] out = a >> 1;')
        self.assertIn('ASSERT QASM2QCX_INT_0 < 4', literal)

    def test_chained_shift_ignores_count_signed_width(self) -> None:
        lines = convert('''OPENQASM 3.0; int a = 2; int[8] count = 1;
            int out = (a << count) << 1; uint mixed = (a << count) + uint(1);
            for int i in [1:0] { out = (i << count) >> count; }''')
        self.assertIn('LET QASM2QCX_INT_0 <<= COUNT31', lines)
        self.assertIn('LET OUT7 := QASM2QCX_INT_1', lines)
        self.assertIn('LET MIXED31 := QASM2QCX_UINT_0', lines)
        folded = convert('''OPENQASM 3.0; const int[8] COUNT = 1;
            const int OUT = (2 << COUNT) << 1; int out = OUT;''')
        self.assertIn('LET OUT7 := 8', folded)

    def test_indexed_operands_evaluate_in_source_order_and_release_temporaries(self) -> None:
        program = qasm2qcx.openqasm3.parse('''OPENQASM 3.0;
            array[int, 2] a = {4, 1}; int i = 0; int out = a[i] << a[i + 1];''')
        converter = qasm2qcx.QASM2QCXConverter(program)
        converter.visit(program)
        lines = list(converter)
        reads = [index for index, line in enumerate(lines) if ':= A1:QASM2QCX_INT_' in line]
        self.assertEqual(len(reads), 2)
        shift = next(index for index, line in enumerate(lines) if ' <<= ' in line)
        self.assertLess(reads[0], reads[1])
        self.assertLess(reads[1], shift)
        self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())

    def test_invalid_constant_counts_and_signed_overflow_are_converter_errors(self) -> None:
        width = qasm2qcx.QASM2QCXConverter.QCX_UINT_WIDTH
        maximum = qasm2qcx.QASM2QCXConverter.QCX_INT_MAX
        for expression in ('1 << -1', f'1 >> {width}', 'uint[4](1) << 4', f'{maximum} << 1'):
            with self.subTest(expression=expression), self.assertRaises(qasm2qcx.InvalidShiftException):
                convert('OPENQASM 3.0; const int A = int(' + expression + ');')

    def test_invalid_literal_runtime_shifts_are_deferred(self) -> None:
        maximum = qasm2qcx.QASM2QCXConverter.QCX_INT_MAX
        lines = convert(f'''OPENQASM 3.0; int out = 7;
            if (false) {{ out = 1 << -1; out = {maximum} << 1; }}
            uint[4] small = 3; if (false) {{ small = uint[4](1) >> 4; }}''')
        self.assertIn('LET QASM2QCX_INT_0 <<= -1', lines)
        self.assertIn('ASSERT QASM2QCX_INT_0 < 4', lines)
        self.assertGreater(lines.index('LET QASM2QCX_INT_0 <<= -1'), lines.index('@QASM2QCX_IF_0'))
        self.assertIn('LET OUT7 := 0', convert('''OPENQASM 3.0;
            const bool OFF = false && bool(1 << -1); bool out = OFF;'''))

    def test_non_integer_operands_and_non_native_signed_values_rejected(self) -> None:
        for expression in ('true << 1', 'bit(true) >> 1', '1 << false', '1 >> 1.5', '1.0im << 1'):
            with self.subTest(expression=expression), self.assertRaises(qasm2qcx.UnsupportedOpenQASMError):
                convert('OPENQASM 3.0; int out = int(' + expression + ');')
        for prefix, expression in (('int[8] a = 3;', 'a << 1'),
                                   ('const int[8] A = 3;', 'A >> 1'),
                                   ('array[int[8], 1] a = {3};', 'a[0] << 1')):
            with self.subTest(expression=expression), self.assertRaisesRegex(
                    qasm2qcx.UnsupportedOpenQASMError, 'non-native signed width'):
                convert('OPENQASM 3.0; ' + prefix + ' int out = ' + expression + ';')

    def test_empty_loops_validate_types_without_evaluating_counts(self) -> None:
        convert('OPENQASM 3.0; int out = 0; for int i in [1:0] { out = i << -1; }')
        for expression in ('true << i', 'i >> false', 'bit(true) << i'):
            with self.subTest(expression=expression), self.assertRaises(qasm2qcx.UnsupportedOpenQASMError):
                convert('OPENQASM 3.0; int out = 0; for int i in [1:0] { out = int(' + expression + '); }')


class ScalarShiftAssignmentTests(unittest.TestCase):
    def test_scalar_compounds_match_explicit_expressions(self) -> None:
        for value_type in ('int', 'uint', 'uint[4]'):
            for operator in ('<<', '>>'):
                for rhs in ('1', 'n', 'a', '(a >> n) + 1', 'uint(n)'):
                    with self.subTest(value_type=value_type, operator=operator, rhs=rhs):
                        prefix = f'OPENQASM 3.0; {value_type} a = 3; int n = 1; '
                        self.assertEqual(convert(prefix + f'a {operator}= {rhs};'),
                                         convert(prefix + f'a = a {operator} ({rhs});'))

    def test_signed_target_with_uint_count_stays_signed(self) -> None:
        lines = convert('OPENQASM 3.0; int a = -3; uint n = 1; a >>= n;')
        self.assertIn('LET QASM2QCX_INT_0 >>= N1', lines)
        self.assertEqual(lines[-1], 'LET A1 := QASM2QCX_INT_0')
        self.assertFalse(any(line.startswith('VAR QASM2QCX_UINT_') for line in lines))

    def test_runtime_destination_is_captured_once_and_reserved(self) -> None:
        lines = convert('''OPENQASM 3.0; array[int, 2] a = {4, 1}; int i = -2;
            a[i + 1] <<= a[i] >> 2;''')
        checks = [line for line in lines if line.startswith('ASSERT ')]
        self.assertEqual(len(checks), 4)
        destination, source = checks[0].split()[1], checks[2].split()[1]
        self.assertNotEqual(destination, source)
        self.assertEqual(sum(line.endswith(' += 1') for line in lines), 1)
        self.assertTrue(lines[-1].startswith(f'LET A1:{destination} := QASM2QCX_INT_'))

    def test_static_elements_and_self_references_release_temporaries(self) -> None:
        program = qasm2qcx.openqasm3.parse('''OPENQASM 3.0;
            int n = 1; uint[4] u = 15; array[int, 2] a = {0, 1}; int i = 0;
            n <<= n; u >>= u >> 3; a[1] <<= n; a[a[i]] <<= a[i + 1];''')
        converter = qasm2qcx.QASM2QCXConverter(program)
        converter.visit(program)
        lines = list(converter)
        self.assertTrue(any(line.startswith('LET A1:1 := QASM2QCX_INT_') for line in lines))
        self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())

    def test_invalid_types_and_signed_widths_rejected_even_when_skipped(self) -> None:
        for declaration, body in (
                ('bool a = true;', 'a <<= 1;'), ('bit a = 1;', 'a >>= 1;'),
                ('bit[2] a = "01";', 'a[0] <<= 1;'), ('float a = 1.0;', 'a <<= 1;'),
                ('complex a = 1.0im;', 'a >>= 1;'), ('int a = 1;', 'a <<= true;'),
                ('int a = 1;', 'a >>= 1.0;'), ('int[8] a = 1;', 'a <<= 1;'),
                ('array[int[8], 1] a = {1};', 'a[0] >>= 1;'),
                ('array[int, 1] a = {1}; float f = 0.5;', 'a[f] <<= 1;')):
            for wrapper in ('{}', 'if (false) {{ {} }}', 'for int j in [1:0] {{ {} }}'):
                with self.subTest(declaration=declaration, wrapper=wrapper), self.assertRaises(qasm2qcx.QASM2QCXError):
                    convert('OPENQASM 3.0; ' + declaration + wrapper.format(body))

    def test_empty_loops_do_not_evaluate_counts_or_indices(self) -> None:
        convert('''OPENQASM 3.0; array[int, 1] a = {1}; uint[4] u = 3;
            for int i in [1:0] { a[999] <<= -1; u >>= 4; }''')
        for body in ('i <<= 1;', 'while (false) { i >>= 1; }'):
            with self.subTest(body=body), self.assertRaises(qasm2qcx.UnsupportedOpenQASMError):
                convert('OPENQASM 3.0; for int i in [1:0] { ' + body + ' }')

    def test_runtime_loop_bounds_sets_and_indices_use_shift_metadata(self) -> None:
        program = qasm2qcx.openqasm3.parse('''OPENQASM 3.0;
            int n = 2; int count = 1; int total = 0; array[int, 2] a = {1, 2};
            for int i in [0:n >> count] { a[i] <<= count; }
            for int j in {n >> count, n << count} { total += j; }
            while (count < 2) { total >>= count; count += 1; }
            total += a[n >> count];''')
        converter = qasm2qcx.QASM2QCXConverter(program)
        converter.visit(program)
        self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())
        self.assertEqual(converter._QASM2QCXConverter__loop_bindings, [])
        self.assertEqual(converter._QASM2QCXConverter__loop_contexts, [])


class BitRegisterBitwiseFoundationTests(unittest.TestCase):
    def converter(self, declarations: str):
        program = qasm2qcx.openqasm3.parse('OPENQASM 3.0; ' + declarations)
        converter = qasm2qcx.QASM2QCXConverter(program)
        converter.visit(program)
        return converter

    def expression(self, source: str):
        program = qasm2qcx.openqasm3.parse('OPENQASM 3.0; bit[4] result = ' + source + ';')
        return program.statements[0].init_expression

    def width(self, converter, source: str, iterators=None):
        return converter._QASM2QCXConverter__bit_register_expression_width(
            self.expression(source), iterators)

    def test_registers_literals_and_nested_expression_widths(self) -> None:
        converter = self.converter('bit[4] a; bit[4] b; bit[1] one; bit scalar;')
        for expression, width in (
                ('a', 4), ('"0001"', 4), ('one', 1), ('"0"', 1),
                ('~a', 4), ('~~"0011"', 4), ('a & b', 4),
                ('a | "0011"', 4), ('~(a ^ b) & ("0001" | a)', 4),
                ('one ^ "1"', 1), ('scalar', None), ('~scalar', None)):
            with self.subTest(expression=expression):
                self.assertEqual(self.width(converter, expression), width)

    def test_different_widths_and_scalar_register_mixing_rejected(self) -> None:
        converter = self.converter('bit[4] a; bit[2] b; bit[1] one; bit scalar; int n; bool ready;')
        for expression in ('a & b', 'a | "01"', 'one ^ "00"', '~(a & (b | "01"))'):
            with self.subTest(expression=expression), self.assertRaisesRegex(
                    qasm2qcx.InvalidBitOperandException, 'widths .* do not match'):
                self.width(converter, expression)
        for expression in ('one & scalar', 'scalar | "1"', 'a ^ n', 'ready & a',
                           'a[0] & one', 'one | bit(true)'):
            with self.subTest(expression=expression), self.assertRaisesRegex(
                    qasm2qcx.InvalidBitOperandException, 'both be scalars or both be registers'):
                self.width(converter, expression)

    def test_width_checks_do_not_emit_or_evaluate_index_arithmetic(self) -> None:
        converter = self.converter('bit[4] a; bit[4] b; bit scalar; int i;')
        before = (list(converter._QASM2QCXConverter__qcx_lines), converter._QASM2QCXConverter__value,
                  converter._QASM2QCXConverter__value_type, converter._QASM2QCXConverter__value_kind,
                  converter._QASM2QCXConverter__expression_kind,
                  set(converter._QASM2QCXConverter__used_temporary_variables))
        self.assertEqual(self.width(converter, '~(a & b)'), 4)
        for expression in ('a[999] & scalar', 'a[1 / 0] ^ scalar', 'a[i + 1] | scalar'):
            self.assertIsNone(self.width(converter, expression))
        self.assertEqual(before, (list(converter._QASM2QCXConverter__qcx_lines), converter._QASM2QCXConverter__value,
                         converter._QASM2QCXConverter__value_type, converter._QASM2QCXConverter__value_kind,
                         converter._QASM2QCXConverter__expression_kind,
                         set(converter._QASM2QCXConverter__used_temporary_variables)))

    def test_register_widths_remain_in_declaration_scope(self) -> None:
        converter = self.converter('const int N = 4; bit[N] a; bit[4] b;')
        self.assertEqual(self.width(converter, 'a & b', {'N': qasm2qcx.ValueType.INT}), 4)
        self.assertIsNone(self.width(converter, 'a', {'a': qasm2qcx.ValueType.BIT}))
        self.assertIsNone(self.width(converter, '~a', {'a': qasm2qcx.ValueType.INT}))
        with self.assertRaises(qasm2qcx.InvalidBitOperandException):
            self.width(converter, 'a & b', {'a': qasm2qcx.ValueType.BIT})
        converter._QASM2QCXConverter__loop_bindings.append({'a': 2})
        try:
            self.assertIsNone(self.width(converter, '~a'))
        finally:
            converter._QASM2QCXConverter__loop_bindings.pop()

    def test_unsupported_operations_and_selections_rejected(self) -> None:
        converter = self.converter('bit[4] a; bit[4] b; float f; array[int, 2] values;')
        for expression in ('a << true', 'a + b', '!a', 'a < b', 'int(a)', 'a[0:1] & b',
                           'a[{0, 1}] | b', 'a[f] & a[0]', 'values & a', 'sin(1.0)'):
            with self.subTest(expression=expression), self.assertRaises(qasm2qcx.QASM2QCXError):
                self.width(converter, expression)
        with self.assertRaises(qasm2qcx.NoVariableNameException):
            self.width(converter, 'a & missing')

    def test_existing_scalar_type_rules_are_preserved(self) -> None:
        converter = self.converter('int a; uint b; bit scalar; bool ready;')
        for expression in ('a & b', '~a', 'scalar ^ bit(true)', 'a << 1'):
            with self.subTest(expression=expression):
                self.assertIsNone(self.width(converter, expression))
        for expression in ('scalar & a', '~ready', 'ready | ready'):
            with self.subTest(expression=expression), self.assertRaises(qasm2qcx.QASM2QCXError):
                self.width(converter, expression)

    def test_register_assignment_reports_operand_and_target_width_errors(self) -> None:
        for source, diagnostic in (
                ('bit[4] a; bit[2] b; bit[4] c = a & b;', 'widths 4 and 2 do not match'),
                ('bit[4] a; bit[2] c = ~a;', 'width 4 does not match target size 2'),
                ('bit scalar; bit[1] c = ~scalar;', 'requires a register value'),
                ('bit[1] one; bit scalar; one = one & scalar;', 'both be scalars or both be registers')):
            with self.subTest(source=source), self.assertRaisesRegex(qasm2qcx.InvalidBitOperandException, diagnostic):
                convert('OPENQASM 3.0; ' + source)

    def test_compound_selection_width_mismatch_rejected(self) -> None:
        with self.assertRaises(qasm2qcx.InvalidBitOperandException):
            convert('OPENQASM 3.0; bit[4] a; bit[4] b; a[0:2] &= b;')
        self.assertEqual(convert('OPENQASM 3.0; bit[1] a = "1"; bit[1] b = a;'),
                         ['QUBITS 0', 'VAR A1 INT', 'LET A1 := 1', 'VAR B1 INT', 'LET B1 := A1'])


class BitRegisterEqualityFoundationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.foundation = BitRegisterBitwiseFoundationTests()
        self.converter = self.foundation.converter('''bit[4] a; bit[4] b; bit[1] one;
            bit scalar; bit other; int n; uint u; float f; bool ready; bit[65] wide;
            array[int, 2] counts;''')

    def comparison(self, source: str):
        return qasm2qcx.openqasm3.parse('OPENQASM 3.0; bool result = ' + source + ';').statements[0].init_expression

    def width(self, source: str, iterators=None):
        return self.converter._QASM2QCXConverter__register_comparison_width(self.comparison(source), iterators)

    def test_equal_width_register_operands_literals_and_expressions(self) -> None:
        for expression, expected in (('a == b', 4), ('a != "0000"', 4),
                                     ('"0000" == "0001"', 4), ('~a == (b ^ "0011")', 4),
                                     ('(a << n) != (b >> u)', 4), ('a[0:1] == a[{3, 1}]', 2),
                                     ('one == a[0:0]', 1), ('one != "1"', 1), ('wide == ~wide', 65)):
            with self.subTest(expression=expression):
                self.assertEqual(self.width(expression), expected)

    def test_width_mismatches_and_scalar_register_mixing_rejected(self) -> None:
        for expression in ('a == "01"', 'a != one', 'a[0:1] == a[0:2]', '"0" != "00"'):
            with self.subTest(expression=expression), self.assertRaisesRegex(
                    qasm2qcx.InvalidBitOperandException, 'register widths .* do not match'):
                self.width(expression)
        for expression in ('a == n', 'a != u', 'a == ready', 'one == scalar',
                           'one != a[0]', 'true == one', 'a[0:0] == 1'):
            with self.subTest(expression=expression), self.assertRaisesRegex(
                    qasm2qcx.InvalidBitOperandException, 'both be scalars or both be registers'):
                self.width(expression)

    def test_scalar_comparisons_keep_existing_type_path(self) -> None:
        for expression in ('n == u', 'n < f', 'scalar == other', 'scalar != ready',
                           'a[n] == scalar', 'one[0] != true', '1 == 1'):
            with self.subTest(expression=expression):
                self.assertIsNone(self.width(expression))
        lines = convert('OPENQASM 3.0; int n = 1; bool same = n == 1; bit flag = 1; if (flag != false) {}')
        self.assertTrue(any(line.startswith('JUMPIF ') for line in lines))

    def test_ordered_register_comparisons_remain_unsupported(self) -> None:
        for operator in ('<', '<=', '>', '>='):
            with self.subTest(operator=operator), self.assertRaisesRegex(
                    qasm2qcx.UnsupportedOpenQASMError, 'ordered comparisons'):
                self.width('a ' + operator + ' b')

    def test_validation_does_not_emit_or_evaluate_runtime_arithmetic(self) -> None:
        before = (list(self.converter._QASM2QCXConverter__qcx_lines),
                  self.converter._QASM2QCXConverter__value, self.converter._QASM2QCXConverter__value_type,
                  self.converter._QASM2QCXConverter__value_kind, self.converter._QASM2QCXConverter__expression_kind)
        for expression in ('(a << -1) == b', '(a >> (1 / 0)) != b',
                           '(a << counts[999]) == b', '(a << n) == ~b'):
            self.assertEqual(self.width(expression), 4)
        self.assertEqual(before, (list(self.converter._QASM2QCXConverter__qcx_lines),
                         self.converter._QASM2QCXConverter__value, self.converter._QASM2QCXConverter__value_type,
                         self.converter._QASM2QCXConverter__value_kind, self.converter._QASM2QCXConverter__expression_kind))
        self.assertEqual(self.converter._QASM2QCXConverter__used_temporary_variables, set())

    def test_bad_selection_shapes_names_and_bounds_rejected(self) -> None:
        for expression in ('a[n:3] == b', 'a[0:4] == b', 'a[1:0] != "1"',
                           '(a | b)[0:1] == "11"', 'int(a) == n',
                           'a[{0, 1}] == a[{0, 1, 2}]'):
            with self.subTest(expression=expression), self.assertRaises(qasm2qcx.QASM2QCXError):
                self.width(expression)
        with self.assertRaises(qasm2qcx.NoVariableNameException):
            self.width('a == missing')

    def test_iterator_shadowing_preserves_scalar_identity(self) -> None:
        with self.assertRaises(qasm2qcx.InvalidBitOperandException):
            self.width('a == b', {'a': qasm2qcx.ValueType.INT})
        self.assertIsNone(self.width('a == n', {'a': qasm2qcx.ValueType.INT}))
        self.assertIsNone(self.width('a == true', {'a': qasm2qcx.ValueType.BIT}))
        self.converter._QASM2QCXConverter__loop_bindings.append({'a': 1})
        try:
            self.assertIsNone(self.width('a == n'))
        finally:
            self.converter._QASM2QCXConverter__loop_bindings.pop()

    def test_public_runtime_and_constant_paths_validate_widths(self) -> None:
        for source in ('bit[4] a; bool same = a == "01";',
                       'bit[4] a; if (false && a == "01") {}',
                       'const bool SAME = "0000" == "00";',
                       'const bool OFF = false && ("0000" != "00");',
                       'bit[1] one; bit scalar; bool same = one == scalar;'):
            with self.subTest(source=source), self.assertRaises(qasm2qcx.InvalidBitOperandException):
                convert('OPENQASM 3.0; ' + source)
        for source in ('bit[4] a; bit[4] b; bool same = a == b;',
                       'const bool SAME = "0000" == "0000";'):
            with self.subTest(source=source):
                convert('OPENQASM 3.0; ' + source)


class BitRegisterEqualityExpressionTests(unittest.TestCase):
    def test_runtime_equality_and_inequality_use_mismatch_branches(self) -> None:
        for operator, mismatch, match in (('==', 'FALSE', 'TRUE'), ('!=', 'TRUE', 'FALSE')):
            with self.subTest(operator=operator):
                lines = convert(f'OPENQASM 3.0; bit[2] a; bit[2] b; bool same = a {operator} b;')
                self.assertIn(f'JUMPIF QASM2QCX_BOOL_{mismatch}_0 A1:0 \\= B1:0', lines)
                self.assertIn(f'JUMPIF QASM2QCX_BOOL_{mismatch}_0 A1:1 \\= B1:1', lines)
                self.assertIn(f'JUMP QASM2QCX_BOOL_{match}_0', lines)

    def test_literal_lhs_uses_register_variable_for_jumpif(self) -> None:
        lines = convert('OPENQASM 3.0; bit[2] a; bool same = "01" == a;')
        self.assertIn('JUMPIF QASM2QCX_BOOL_FALSE_0 A1:0 \\= 1', lines)
        self.assertIn('JUMPIF QASM2QCX_BOOL_FALSE_0 A1:1 \\= 0', lines)

    def test_constant_literals_nested_operations_and_shifts_fold(self) -> None:
        lines = convert('''OPENQASM 3.0; const int COUNT = 1;
            const bool SAME = (~"1010" ^ "1100") == "1001";
            const bool DIFF = ("1011" << COUNT) != "1011";
            const bool ZERO = ("1011" >> uint(-1)) == "0000";
            bool a = SAME; bool b = DIFF; bool c = ZERO;''')
        for name in ('A1', 'B1', 'C1'):
            self.assertIn(f'LET {name} := 1', lines)
        self.assertFalse(any(line.startswith('VAR QASM2QCX_') for line in lines))
        runtime = convert('OPENQASM 3.0; bool a = "01" == "01";')
        self.assertFalse(any(line.startswith('JUMPIF ') for line in runtime))

    def test_constant_short_circuit_suppresses_shift_errors_not_type_checks(self) -> None:
        lines = convert('''OPENQASM 3.0;
            const bool OFF = false && (("01" << -1) == "00");
            const bool ON = true || (("01" >> (1 / 0)) != "11");
            bool a = OFF; bool b = ON;''')
        self.assertIn('LET A1 := 0', lines)
        self.assertIn('LET B1 := 1', lines)
        with self.assertRaises(qasm2qcx.InvalidShiftException):
            convert('OPENQASM 3.0; const bool BAD = ("01" << -1) == "00";')
        with self.assertRaises(qasm2qcx.QASM2QCXError):
            convert('OPENQASM 3.0; bit[2] a; const bool BAD = a == "00";')

    def test_both_operands_complete_left_to_right_before_comparison(self) -> None:
        lines = convert('''OPENQASM 3.0; bit[4] a; bit[4] b; int n = 1;
            bool same = (a << n) == (b >> n);''')
        comparison = next(index for index, line in enumerate(lines)
                          if line.startswith('JUMPIF QASM2QCX_BOOL_'))
        self.assertLess(lines.index('@QASM2QCX_REGISTER_SHIFT_0_END'),
                        lines.index('@QASM2QCX_REGISTER_SHIFT_1_END'))
        self.assertLess(lines.index('@QASM2QCX_REGISTER_SHIFT_1_END'), comparison)
        self.assertEqual(sum(line.startswith('VAR QASM2QCX_BITS_4_') for line in lines), 4)

    def test_singleton_and_reordered_selections_keep_identity_and_order(self) -> None:
        lines = convert('''OPENQASM 3.0; bit[4] a; bit[1] one;
            bool same = a[{3, 0}] == a[1:2]; bool single = a[{0}] != one;''')
        self.assertIn('JUMPIF QASM2QCX_BOOL_FALSE_0 A1:3 \\= A1:1', lines)
        self.assertIn('JUMPIF QASM2QCX_BOOL_FALSE_0 A1:0 \\= A1:2', lines)
        self.assertIn('JUMPIF QASM2QCX_BOOL_TRUE_1 A1:0 \\= ONE7', lines)

    def test_results_reuse_and_release_storage(self) -> None:
        program = qasm2qcx.openqasm3.parse('''OPENQASM 3.0; bit[4] a; bit[4] b;
            bool same = (~a ^ b) == (a | b); same = (~a ^ b) != (a | b);''')
        converter = qasm2qcx.QASM2QCXConverter(program)
        converter.visit(program)
        self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())
        ForLoopControlLoweringTests.assert_resolved_jumps(list(converter))

    def test_failed_rhs_lowering_releases_both_operands_and_boolean_result(self) -> None:
        program = qasm2qcx.openqasm3.parse('''OPENQASM 3.0; bit[2] a;
            bool same = ~a == later; bit[2] later;''')
        converter = qasm2qcx.QASM2QCXConverter(program)
        with self.assertRaises(qasm2qcx.QASM2QCXError):
            converter.visit(program)
        self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())


class BitRegisterEqualityControlFlowTests(unittest.TestCase):
    def test_metadata_tracks_boolean_type_and_runtime_dependency_without_evaluation(self) -> None:
        foundation = BitRegisterEqualityFoundationTests()
        foundation.setUp()
        converter = foundation.converter
        before = list(converter._QASM2QCXConverter__qcx_lines)
        for source, runtime in (('"01" == ~"10"', False),
                                ('("01" << -1) == "00"', False),
                                ('("01" >> (1 / 0)) != "11"', False),
                                ('a == b', True), ('a[0:1] == "01"', True),
                                ('("0000" << n) == "0000"', True),
                                ('!(a == b) || ((a != b) == true)', True)):
            with self.subTest(source=source):
                self.assertEqual(converter._QASM2QCXConverter__loop_expression_info(
                    foundation.comparison(source)), (qasm2qcx.ValueType.BOOL, runtime))
        self.assertEqual(converter._QASM2QCXConverter__qcx_lines, before)
        self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())

    def test_nested_boolean_comparisons_casts_and_shift_counts(self) -> None:
        lines = convert('''OPENQASM 3.0; bit[2] a; bit[2] b;
            bool same = (a == b) == true; int count = int(a != b);
            bool shifted = (a << int(a == b)) == b;
            if (bool(a == b) && !(a != b)) { same = true; }''')
        ForLoopControlLoweringTests.assert_resolved_jumps(lines)

    def test_nested_if_while_and_for_release_temporaries(self) -> None:
        program = qasm2qcx.openqasm3.parse('''OPENQASM 3.0; bit[4] a; bit[4] b;
            int n = 1; bool ready = true;
            for int i in [0:1] {
                while (ready && ((a << n) != ~b)) {
                    if ((~a == b) || !(a[{0, 2}] == b[1:2])) { break; }
                    a = b; continue;
                }
            }''')
        converter = qasm2qcx.QASM2QCXConverter(program)
        converter.visit(program)
        self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())
        ForLoopControlLoweringTests.assert_resolved_jumps(list(converter))

    def test_empty_loops_validate_comparisons_without_evaluating_counts(self) -> None:
        for condition in ('(a << -1) == b', '(a >> (1 / 0)) != b',
                          '(a << counts[999]) == b', '(~a == b) == true'):
            with self.subTest(condition=condition):
                convert('OPENQASM 3.0; bit[4] a; bit[4] b; array[int, 2] counts; '
                        'for int i in [1:0] { while (' + condition + ') { break; } }')
        for condition in ('a == "01"', 'a == missing', 'a < b', 'a[n:3] == b',
                          '(a << 1.5) == b', 'a[0:4] == b'):
            with self.subTest(condition=condition), self.assertRaises(qasm2qcx.QASM2QCXError):
                convert('OPENQASM 3.0; bit[4] a; bit[4] b; int n; '
                        'for int i in [1:0] { if (false && (' + condition + ')) {} }')

    def test_iterator_shadowing_applies_in_empty_and_executed_loops(self) -> None:
        for bounds in ('1:0', '0:1'):
            convert('OPENQASM 3.0; bit[2] a; int n; for int a in [' + bounds + '] { if (a == n) {} }')
            with self.assertRaises(qasm2qcx.InvalidBitOperandException):
                convert('OPENQASM 3.0; bit[2] a; bit[2] b; '
                        'for int a in [' + bounds + '] { if (a == b) {} }')

    def test_short_circuit_branches_precede_rhs_operand_evaluation(self) -> None:
        for operator, lhs, destination in (('&&', 'false', 'FALSE'), ('||', 'true', 'TRUE')):
            lines = convert(f'''OPENQASM 3.0; bit[2] a; int n = -1;
                bool ready = {lhs} {operator} ((a << n) == "00");''')
            jump = lines.index(f'JUMP QASM2QCX_BOOL_{destination}_0')
            assertion = next(index for index, line in enumerate(lines) if line.startswith('ASSERT '))
            self.assertLess(jump, assertion)
            ForLoopControlLoweringTests.assert_resolved_jumps(lines)


class StaticBitSelectionFoundationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.foundation = BitRegisterBitwiseFoundationTests()
        self.converter = self.foundation.converter('''const int LOW = 1; const uint HIGH = 3;
            bit[4] a; bit[1] one; bit scalar; int n; float f; array[int, 2] values;''')

    def selection(self, source: str, *, destination: bool = False, iterators=None):
        return self.converter._QASM2QCXConverter__static_bit_selection(
            self.foundation.expression(source), iterators, destination=destination)

    def test_order_ranges_sets_negative_indices_and_constants(self) -> None:
        for source, indices in (
                ('a[0:2]', (0, 1, 2)), ('a[3:-1:0]', (3, 2, 1, 0)),
                ('a[0:2:3]', (0, 2)), ('a[-2:-1]', (2, 3)),
                ('a[-1:-1:-4]', (3, 2, 1, 0)), ('a[{3, 0, 2}]', (3, 0, 2)),
                ('a[{LOW, HIGH}]', (1, 3)), ('a[LOW:HIGH]', (1, 2, 3)),
                ('a[{HIGH - LOW, uint(0)}]', (2, 0))):
            with self.subTest(source=source):
                selection = self.selection(source)
                self.assertEqual(selection.source_name, 'a')
                self.assertEqual(selection.indices, indices)
                self.assertEqual(selection.width, len(indices))
                self.assertTrue(selection.is_register)

    def test_singleton_register_selection_differs_from_scalar_element(self) -> None:
        for source, is_register in (('a[0]', False), ('a[0:0]', True), ('a[{0}]', True),
                                    ('one', True), ('one[0]', False), ('one[0:0]', True),
                                    ('scalar', False), ('scalar[0:0]', True)):
            with self.subTest(source=source):
                selection = self.selection(source)
                self.assertEqual(selection.width, 1)
                self.assertEqual(selection.is_register, is_register)

    def test_indexed_identifier_and_expression_have_same_description(self) -> None:
        statement = qasm2qcx.openqasm3.parse('OPENQASM 3.0; a[{3, 0, 2}] = "101";').statements[0]
        self.assertEqual(self.converter._QASM2QCXConverter__static_bit_selection(statement.lvalue),
                         self.selection('a[{3, 0, 2}]'))

    def test_repeated_sources_preserved_and_destinations_rejected(self) -> None:
        for source, indices in (('a[{1, 1}]', (1, 1)), ('a[{3, -1}]', (3, 3)),
                                ('a[-4:3]', (0, 1, 2, 3, 0, 1, 2, 3))):
            with self.subTest(source=source):
                self.assertEqual(self.selection(source).indices, indices)
                with self.assertRaisesRegex(qasm2qcx.InvalidBitOperandException, 'must not repeat'):
                    self.selection(source, destination=True)
        self.assertEqual(self.selection('a[{3, 0, 2}]', destination=True).indices, (3, 0, 2))

    def test_empty_bad_bounds_and_unsupported_shapes_rejected(self) -> None:
        for source in ('a[1:0]', 'a[0:0:3]', 'a[0:4]', 'a[-5:-1]',
                       'a[0:4294967295]', 'a[0:]', 'a[:3]', 'a[f:3]',
                       'a[0, 1]', 'a[0][1]', '(a | a)[0:1]', 'values[0:1]'):
            with self.subTest(source=source), self.assertRaises(qasm2qcx.QASM2QCXError):
                self.selection(source)
        empty = qasm2qcx.ast.IndexExpression(qasm2qcx.ast.Identifier('a'), qasm2qcx.ast.DiscreteSet([]))
        with self.assertRaises(qasm2qcx.InvalidBitOperandException):
            self.converter._QASM2QCXConverter__static_bit_selection(empty)

    def test_runtime_and_iterator_dependent_bounds_rejected(self) -> None:
        for source in ('a[n:3]', 'a[{0, n}]', 'a[n]'):
            with self.subTest(source=source), self.assertRaisesRegex(
                    qasm2qcx.UnsupportedOpenQASMError, 'runtime-dependent'):
                self.selection(source)
        for source in ('a[LOW:3]', 'a[{LOW, 0}]'):
            with self.subTest(source=source), self.assertRaisesRegex(
                    qasm2qcx.UnsupportedOpenQASMError, 'iterator-dependent'):
                self.selection(source, iterators={'LOW': qasm2qcx.ValueType.INT})
        self.converter._QASM2QCXConverter__loop_bindings.append({'LOW': 1})
        try:
            with self.assertRaises(qasm2qcx.UnsupportedOpenQASMError):
                self.selection('a[LOW:3]')
        finally:
            self.converter._QASM2QCXConverter__loop_bindings.pop()
        with self.assertRaises(qasm2qcx.UnsupportedOpenQASMError):
            self.selection('a[0:1]', iterators={'a': qasm2qcx.ValueType.INT})

    def test_nested_register_widths_and_scalar_type_behavior(self) -> None:
        for source, width in (('~a[0:1]', 2), ('a[{3, 1}] & "01"', 2),
                              ('(a[0:1] ^ a[2:3]) << n', 2), ('a[0:0] | one', 1)):
            with self.subTest(source=source):
                self.assertEqual(self.foundation.width(self.converter, source), width)
        for source in ('a[0:1] & a', 'a[0:0] | scalar', 'a[{0}] & a[0]'):
            with self.subTest(source=source), self.assertRaises(qasm2qcx.InvalidBitOperandException):
                self.foundation.width(self.converter, source)
        self.assertIsNone(self.foundation.width(self.converter, 'a[n] & scalar'))

    def test_metadata_is_immutable_and_validation_does_not_emit(self) -> None:
        before = (list(self.converter._QASM2QCXConverter__qcx_lines),
                  self.converter._QASM2QCXConverter__value,
                  self.converter._QASM2QCXConverter__value_type,
                  self.converter._QASM2QCXConverter__value_kind,
                  self.converter._QASM2QCXConverter__expression_kind)
        selection = self.selection('a[LOW:HIGH]')
        with self.assertRaises(AttributeError):
            selection.indices = (0,)
        self.assertEqual(before, (list(self.converter._QASM2QCXConverter__qcx_lines),
                         self.converter._QASM2QCXConverter__value,
                         self.converter._QASM2QCXConverter__value_type,
                         self.converter._QASM2QCXConverter__value_kind,
                         self.converter._QASM2QCXConverter__expression_kind))
        self.assertEqual(self.converter._QASM2QCXConverter__used_temporary_variables, set())
        self.assertIn('LET QASM2QCX_INT_0 := A1:0',
                      convert('OPENQASM 3.0; bit[4] a; bit[2] b = ~a[0:1];'))


class StaticBitSelectionOperandTests(unittest.TestCase):
    def test_source_order_is_preserved_in_generated_reads(self) -> None:
        for selection, indices in (('a[3:-1:0]', (3, 2, 1, 0)),
                                    ('a[{3, 0, 2}]', (3, 0, 2)), ('a[{1, 1}]', (1, 1))):
            with self.subTest(selection=selection):
                lines = convert(f'OPENQASM 3.0; bit[4] a; bit[{len(indices)}] b = ~{selection};')
                reads = [line.split(' := ')[1] for line in lines if ' := A1:' in line]
                self.assertEqual(reads, [f'A1:{index}' for index in indices])

    def test_nested_selection_expressions_mix_registers_and_literals(self) -> None:
        lines = convert('''OPENQASM 3.0; bit[4] a; bit[2] b = "10";
            bit[2] c = ~(a[0:1] ^ a[{3, 1}]) & (b | "01");''')
        self.assertIn('LET QASM2QCX_INT_0 := A1:0', lines)
        self.assertIn('LET QASM2QCX_INT_0 ^= A1:3', lines)
        self.assertTrue(lines[-1].startswith('LET C1:1 := QASM2QCX_INT_'))

    def test_singleton_selection_keeps_register_identity(self) -> None:
        lines = convert('''OPENQASM 3.0; bit[4] a; bit[1] b = ~a[{2}];
            bit[1] one = "1"; b = one[0:0] | "0";''')
        self.assertIn('LET QASM2QCX_INT_0 := A1:2', lines)
        self.assertIn('LET QASM2QCX_INT_0 := ONE7', lines)
        for source in ('bit[4] a; bit b = ~a[{0}];',
                       'bit[4] a; bit[1] b = a[0] | "1";'):
            with self.subTest(source=source), self.assertRaises(qasm2qcx.QASM2QCXError):
                convert('OPENQASM 3.0; ' + source)

    def test_shift_selection_width_not_container_width(self) -> None:
        lines = convert('OPENQASM 3.0; bit[4] a; uint n; bit[2] b = a[{3, 0}] << n;')
        self.assertTrue(any(line.startswith('JUMPIF QASM2QCX_REGISTER_SHIFT_') and line.endswith(' >= 2') for line in lines))
        self.assertTrue(any(line.startswith('VAR QASM2QCX_BITS_2_') for line in lines))
        self.assertFalse(any(line.startswith('VAR QASM2QCX_BITS_4_') for line in lines))

    def test_self_reference_consumes_all_sources_before_whole_register_write(self) -> None:
        lines = convert('OPENQASM 3.0; bit[4] a = "1010"; a = ~a[3:-1:0];')
        reads = [index for index, line in enumerate(lines) if ':= A1:' in line]
        writes = [index for index, line in enumerate(lines) if line.startswith('LET A1:')]
        self.assertLess(max(reads), writes[4])

    def test_nested_buffers_release_and_reuse_storage(self) -> None:
        program = qasm2qcx.openqasm3.parse('''OPENQASM 3.0; bit[4] a; uint n = 1;
            bit[2] b = (a[0:1] << n) ^ (a[{3, 0}] >> n);
            b = (a[0:1] << n) ^ (a[{3, 0}] >> n);''')
        converter = qasm2qcx.QASM2QCXConverter(program)
        converter.visit(program)
        self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())
        self.assertEqual(sum(line.startswith('VAR QASM2QCX_BITS_2_') for line in list(converter)), 4)

    def test_invalid_operands_are_rejected_even_when_skipped(self) -> None:
        prefix = 'OPENQASM 3.0; bit[4] a; bit[2] b; int n; '
        for body in ('b = a[0:2] & "11";', 'b = ~a[n:3];', 'b = ~a[3:0];',
                     'b = ~a[0:4];', 'b = (a | a)[0:1] ^ "11";',
                     'b = a[0:1] << a[0:0];'):
            for wrapper in ('{}', 'if (false) {{ {} }}', 'for int i in [1:0] {{ {} }}'):
                with self.subTest(body=body, wrapper=wrapper), self.assertRaises(qasm2qcx.QASM2QCXError):
                    convert(prefix + wrapper.format(body))
        with self.assertRaises(qasm2qcx.InvalidBitOperandException):
            convert(prefix + 'a[0:2] = ~b;')

    def test_forward_reference_failure_releases_partial_result(self) -> None:
        program = qasm2qcx.openqasm3.parse('''OPENQASM 3.0; bit[4] a;
            bit[2] b = ~a[0:1] | later[0:1]; bit[4] later;''')
        converter = qasm2qcx.QASM2QCXConverter(program)
        with self.assertRaises(qasm2qcx.QASM2QCXError):
            converter.visit(program)
        self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())


class StaticBitSelectionAssignmentTests(unittest.TestCase):
    def test_compounds_match_explicit_expression_assignments(self) -> None:
        for target in ('a[1:2]', 'a[{3, 0}]', 'a[2:2]'):
            width = 1 if target == 'a[2:2]' else 2
            for operator in ('&', '|', '^', '<<', '>>'):
                rhs = 'n' if operator in ('<<', '>>') else ('a[{1}]' if width == 1 else 'a[0:1]')
                prefix = 'OPENQASM 3.0; bit[4] a = "1010"; uint n = 1; '
                with self.subTest(target=target, operator=operator):
                    self.assertEqual(convert(prefix + f'{target} {operator}= {rhs};'),
                                     convert(prefix + f'{target} = {target} {operator} ({rhs});'))

    def test_direct_overlapping_copies_snapshot_before_writes(self) -> None:
        for target, source in (('a[1:2]', 'a[0:1]'), ('a[{0, 3}]', 'a[{3, 0}]'),
                                ('a', 'a[3:-1:0]'), ('a[{0, 1}]', 'a[{0, 0}]')):
            with self.subTest(target=target, source=source):
                lines = convert(f'OPENQASM 3.0; bit[4] a; {target} = {source};')
                reads = [index for index, line in enumerate(lines) if ':= A1:' in line]
                writes = [index for index, line in enumerate(lines) if line.startswith('LET A1:')]
                self.assertLess(max(reads), min(writes))

    def test_singleton_destinations_remain_registers(self) -> None:
        lines = convert('OPENQASM 3.0; bit[1] one = "1"; one[{0}] ^= "1"; one[0:0] <<= 1;')
        self.assertIn('LET ONE7 := 0', lines)
        self.assertFalse(any('ONE7:' in line for line in lines))
        for statement in ('a[{0}] = bit(true);', 'a[0] = "1";', 'a[{0}] ^= a[1];'):
            with self.subTest(statement=statement), self.assertRaises(qasm2qcx.QASM2QCXError):
                convert('OPENQASM 3.0; bit[4] a; ' + statement)

    def test_repeated_destinations_rejected_even_when_skipped(self) -> None:
        for target in ('a[{1, 1}]', 'a[{3, -1}]', 'a[-4:3]'):
            for operator in ('=', '&=', '|=', '^=', '<<=', '>>='):
                width = 8 if target == 'a[-4:3]' else 2
                rhs = '1' if operator in ('<<=', '>>=') else '"' + '1' * width + '"'
                for wrapper in ('{}', 'if (false) {{ {} }}', 'for int i in [1:0] {{ {} }}'):
                    with self.subTest(target=target, operator=operator, wrapper=wrapper), self.assertRaisesRegex(
                            qasm2qcx.InvalidBitOperandException, 'must not repeat'):
                        convert('OPENQASM 3.0; bit[4] a; ' + wrapper.format(f'{target} {operator} {rhs};'))

    def test_invalid_widths_bounds_and_dynamic_selections_rejected(self) -> None:
        for body in ('a[0:2] = "11";', 'a[{0, 1}] &= a;', 'a[0:1] = a[0];',
                     'a[n:3] <<= 1;', 'a[{n, 0}] = "11";', 'a[1:0] = "1";',
                     'a[0:4] >>= 1;', 'a[0:1] <<= false;', 'a[0:1] = ~missing;'):
            for wrapper in ('{}', 'while (false) {{ {} }}', 'for int i in [1:0] {{ {} }}'):
                with self.subTest(body=body, wrapper=wrapper), self.assertRaises(qasm2qcx.QASM2QCXError):
                    convert('OPENQASM 3.0; bit[4] a; int n; ' + wrapper.format(body))
        convert('OPENQASM 3.0; bit[4] a; for int i in [1:0] { a[0:1] >>= -1; }')

    def test_nested_loops_and_temporaries_release_all_storage(self) -> None:
        program = qasm2qcx.openqasm3.parse('''OPENQASM 3.0; bit[4] a; uint n = 1;
            for int i in [0:1] { a[{3, 0}] ^= a[1:2];
                while (n > 0) { a[1:2] = ~a[0:1]; n -= 1; break; }
                a[0:1] <<= uint(i); if (i == 0) { continue; } a[2:3] >>= n; }
            a = a[3:-1:0];''')
        converter = qasm2qcx.QASM2QCXConverter(program)
        converter.visit(program)
        ForLoopControlLoweringTests.assert_resolved_jumps(list(converter))
        self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())
        self.assertEqual(converter._QASM2QCXConverter__loop_bindings, [])
        self.assertEqual(converter._QASM2QCXConverter__loop_contexts, [])

    def test_iterator_dependent_bounds_and_iterator_targets_remain_unsupported(self) -> None:
        for body in ('a[i:3] ^= "1111";', 'a[{0, i}] = "11";', 'i[0:0] <<= 1;'):
            with self.subTest(body=body), self.assertRaises(qasm2qcx.QASM2QCXError):
                convert('OPENQASM 3.0; bit[4] a; for int i in [1:0] { ' + body + ' }')

    def test_empty_loops_reject_unknown_and_nonbit_selection_destinations(self) -> None:
        for destination in ('missing[0:1]', 'n[0:1]', 'values[0:1]'):
            for operator in ('=', '&=', '<<='):
                rhs = '1' if operator == '<<=' else '"11"'
                with self.subTest(destination=destination, operator=operator), self.assertRaises(qasm2qcx.QASM2QCXError):
                    convert('OPENQASM 3.0; int n; array[int, 2] values; '
                            f'for int i in [1:0] {{ {destination} {operator} {rhs}; }}')


class BitRegisterShiftFoundationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.foundation = BitRegisterBitwiseFoundationTests()
        self.converter = self.foundation.converter('''bit[4] a; bit[4] b; bit[1] one;
            bit scalar; int n; uint u; uint[2] small; int[8] narrow; float f; bool ready;
            complex z; array[int, 2] counts;''')

    def width(self, expression: str, iterators=None):
        return self.foundation.width(self.converter, expression, iterators)

    def test_width_depends_only_on_shifted_register(self) -> None:
        for expression, expected in (
                ('a << n', 4), ('a >> u', 4), ('a << small', 4), ('a >> narrow', 4),
                ('one << 1', 1), ('"0001" >> u', 4), ('~(a << n) & (b >> u)', 4),
                ('(a ^ b) << counts[n]', 4), ('a << int(a[0])', 4),
                ('(a << n) >> u', 4)):
            with self.subTest(expression=expression):
                self.assertEqual(self.width(expression), expected)

    def test_noninteger_register_and_invalid_index_counts_rejected(self) -> None:
        for count in ('ready', 'true', 'scalar', 'a[0]', 'f', '1.5', 'z',
                      'a', '"0001"', '(a | b)', 'counts[f]', 'counts[0:1]'):
            with self.subTest(count=count), self.assertRaises(qasm2qcx.QASM2QCXError):
                self.width('a << (' + count + ')')
        with self.assertRaises(qasm2qcx.NoVariableNameException):
            self.width('a >> missing')
        for expression in ('a[0:4] << 1', 'a[{0,4}] >> 1', 'scalar << 1'):
            with self.subTest(expression=expression), self.assertRaises(qasm2qcx.QASM2QCXError):
                self.width(expression)

    def test_constant_policy_zero_fill_order_and_boundary_counts(self) -> None:
        fold = self.converter._QASM2QCXConverter__fold_bit_register_shift
        self.assertEqual(fold(0b1011, 4, 1, '<<'), 0b0110)
        self.assertEqual(fold(0b1011, 4, 1, '>>'), 0b0101)
        self.assertEqual(fold(0b1011, 4, 0, '<<'), 0b1011)
        for width in (1, 4, self.converter.QCX_UINT_WIDTH + 1, 65):
            mask = (1 << width) - 1
            for operator in ('<<', '>>'):
                with self.subTest(width=width, operator=operator):
                    self.assertEqual(fold(mask, width, 0, operator), mask)
                    self.assertEqual(fold(mask, width, width - 1, operator),
                                     1 << (width - 1) if operator == '<<' else 1)
                    for count in (width, width + 1, self.converter.QCX_UINT_MAX, 1 << 1000):
                        self.assertEqual(fold(mask, width, count, operator), 0)
                    self.assertEqual(fold(0, width, width, operator), 0)

    def test_negative_counts_rejected_even_for_zero_values(self) -> None:
        fold = self.converter._QASM2QCXConverter__fold_bit_register_shift
        for operator in ('<<', '>>'):
            for value in (0, 15):
                with self.subTest(operator=operator, value=value), self.assertRaisesRegex(
                        qasm2qcx.InvalidShiftException, 'must not be negative'):
                    fold(value, 4, -1, operator)
        with self.assertRaises(ValueError):
            fold(0, 0, 0, '<<')
        with self.assertRaises(ValueError):
            fold(0, 4, 0, '^')

    def test_type_validation_does_not_evaluate_counts(self) -> None:
        before = list(self.converter._QASM2QCXConverter__qcx_lines)
        for expression in ('a << -1', 'a >> 4', 'a << (1 / 0)',
                           'a >> counts[999]', 'a << counts[n + 1]'):
            with self.subTest(expression=expression):
                self.assertEqual(self.width(expression), 4)
        self.assertEqual(self.converter._QASM2QCXConverter__qcx_lines, before)
        self.assertEqual(self.converter._QASM2QCXConverter__used_temporary_variables, set())

    def test_iterator_shadowing_and_scalar_shift_rules_unchanged(self) -> None:
        self.assertIsNone(self.width('a << 1', {'a': qasm2qcx.ValueType.INT}))
        self.assertEqual(self.width('a << n', {'n': qasm2qcx.ValueType.INT}), 4)
        with self.assertRaises(qasm2qcx.QASM2QCXError):
            self.width('a >> n', {'n': qasm2qcx.ValueType.BIT})
        self.assertIsNone(self.width('n << u'))
        with self.assertRaises(qasm2qcx.InvalidShiftException):
            convert(f'OPENQASM 3.0; const int A = 1 << {self.converter.QCX_UINT_WIDTH};')
        self.assertEqual(self.converter._QASM2QCXConverter__fold_bit_register_shift(
            1, 4, self.converter.QCX_UINT_WIDTH, '<<'), 0)

    def test_public_validation_reports_width_and_count_types(self) -> None:
        for source in ('bit[4] a; bit[2] b = a << 1;',
                       'bit[4] a; a = a >> true;', 'bit[4] a; a = a << a[0];'):
            with self.subTest(source=source), self.assertRaises(qasm2qcx.InvalidBitOperandException):
                convert('OPENQASM 3.0; ' + source)
        self.assertIn('LET A1:0 := 0', convert('OPENQASM 3.0; bit[4] a; a = a << 1;'))


class BitRegisterShiftExpressionTests(unittest.TestCase):
    def test_literal_folding_zero_fill_and_oversized_counts(self) -> None:
        for expression, expected in (('"1011" << 1', (0, 1, 1, 0)),
                                     ('"1011" >> 1', (1, 0, 1, 0)),
                                     ('"1011" << 4', (0, 0, 0, 0)),
                                     ('"1011" >> uint(-1)', (0, 0, 0, 0)),
                                     ('~("1011" >> 1) & "1111"', (0, 1, 0, 1))):
            with self.subTest(expression=expression):
                lines = convert('OPENQASM 3.0; bit[4] a = ' + expression + ';')
                self.assertEqual(lines[2:], [f'LET A1:{index} := {value}' for index, value in enumerate(expected)])

    def test_constant_shifts_snapshot_all_source_reads_before_writing(self) -> None:
        for operator in ('<<', '>>'):
            lines = convert(f'OPENQASM 3.0; bit[4] a = "1011"; a = a {operator} 1;')
            reads = [index for index, line in enumerate(lines) if ':= A1:' in line]
            writes = [index for index, line in enumerate(lines) if line.startswith('LET A1:')]
            self.assertLess(max(reads), writes[4])

    def test_runtime_count_captured_once_and_uint_guard_precedes_cast(self) -> None:
        lines = convert('OPENQASM 3.0; bit[4] a; uint n; bit[4] b = a << n;')
        self.assertEqual(sum(line.endswith(' := N1') for line in lines), 1)
        guard = next(index for index, line in enumerate(lines) if line.startswith('JUMPIF ') and ' >= 4' in line)
        cast = next(index for index, line in enumerate(lines) if ':= :INT:QASM2QCX_UINT_' in line)
        self.assertLess(guard, cast)
        self.assertFalse(any(line.startswith('ASSERT ') and 'QASM2QCX_UINT_' in line for line in lines))
        signed = convert('OPENQASM 3.0; bit[4] a; int n; a = a >> n;')
        self.assertTrue(any(line.startswith('ASSERT QASM2QCX_INT_') and line.endswith(' >= 0') for line in signed))

    def test_invalid_literal_count_is_deferred_to_runtime(self) -> None:
        lines = convert('OPENQASM 3.0; bit[4] a; if (false) { a = "1011" << -1; }')
        assertion = next(index for index, line in enumerate(lines) if line.startswith('ASSERT '))
        self.assertGreater(assertion, lines.index('@QASM2QCX_IF_0'))
        self.assertIn('LET QASM2QCX_INT_0 := -1', lines)

    def test_nested_runtime_results_reuse_buffers_and_release_storage(self) -> None:
        program = qasm2qcx.openqasm3.parse('''OPENQASM 3.0; bit[4] a; bit[4] b; int n = 1;
            a = ~(a << n) ^ (b >> n); b = ~(a << n) ^ (b >> n);
            bit[1] one = "1"; one = one >> n;''')
        converter = qasm2qcx.QASM2QCXConverter(program)
        converter.visit(program)
        lines = list(converter)
        self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())
        self.assertEqual(sum(line.startswith('VAR QASM2QCX_BITS_4_') for line in lines), 2)
        ForLoopControlLoweringTests.assert_resolved_jumps(lines)

    def test_self_referencing_count_and_index_evaluated_once(self) -> None:
        program = qasm2qcx.openqasm3.parse('''OPENQASM 3.0;
            bit[4] a = "1011"; int i = 0; array[int, 2] counts = {1, 2};
            a = a << int(a[i]); a = a >> counts[i + 1];''')
        converter = qasm2qcx.QASM2QCXConverter(program)
        converter.visit(program)
        lines = list(converter)
        count_evaluation = lines[lines.index('@QASM2QCX_REGISTER_SHIFT_0_END') + 1:
                                 lines.index('@QASM2QCX_REGISTER_SHIFT_1_COPY')]
        self.assertEqual(sum(line.endswith(' += 1') for line in count_evaluation), 1)
        self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())

    def test_unsupported_targets_counts_and_scalar_contexts_still_rejected(self) -> None:
        prefix = 'OPENQASM 3.0; bit[4] a; int n; '
        for statement in ('a[0:2] = a << n;', 'a = a << true;',
                          'a = a[0:4] >> n;', 'int out = int(a << n);',
                          'if (a >> n) {}', 'bit scalar = "1" << n;', 'a <<= true;'):
            with self.subTest(statement=statement), self.assertRaises(qasm2qcx.QASM2QCXError):
                convert(prefix + statement)


class BitRegisterShiftIntegrationTests(unittest.TestCase):
    def test_compounds_match_explicit_assignment(self) -> None:
        for width in (1, 4):
            for operator in ('<<', '>>'):
                for count in ('0', '1', str(width), 'n', 'u', 'int(a[0])', 'n + 1'):
                    with self.subTest(width=width, operator=operator, count=count):
                        prefix = f'OPENQASM 3.0; bit[{width}] a; int n = 1; uint u = 1; '
                        self.assertEqual(convert(prefix + f'a {operator}= {count};'),
                                         convert(prefix + f'a = a {operator} ({count});'))

    def test_empty_and_false_loops_validate_types_but_not_counts(self) -> None:
        for body in ('a <<= -1;', 'a >>= 4;', 'a <<= counts[k];', 'a = a >> (1 / 0);'):
            for wrapper in ('for int i in [1:0] {{ {} }}', 'while (false) {{ {} }}'):
                with self.subTest(body=body, wrapper=wrapper):
                    convert('OPENQASM 3.0; bit[4] a; array[int, 1] counts; int k = 999; ' + wrapper.format(body))
        convert('OPENQASM 3.0; bit[4] a; array[int, 1] counts; for int i in [1:0] { a <<= counts[999]; }')
        prefix = 'OPENQASM 3.0; bit[4] a; bit scalar; bool ready; float f; '
        for body in ('a <<= ready;', 'a >>= a;', 'a <<= scalar;', 'a >>= f;',
                     'a[0:3] <<= true;', 'a[{0,1,2,3}] >>= scalar;', 'scalar <<= 1;',
                     'a[0] >>= 1;', 'a <<= missing;'):
            for wrapper in ('{}', 'if (false) {{ {} }}', 'for int i in [1:0] {{ {} }}'):
                with self.subTest(body=body, wrapper=wrapper), self.assertRaises(qasm2qcx.QASM2QCXError):
                    convert(prefix + wrapper.format(body))

    def test_nested_loops_and_buffers_release_storage(self) -> None:
        program = qasm2qcx.openqasm3.parse('''OPENQASM 3.0; bit[4] a = "1011"; int n = 1;
            array[int, 2] counts = {0, 1};
            for int count in counts {
                a <<= count;
                for int j in [0:n] { if (j == 1) { continue; } a >>= uint(j); }
                while (n > 0) { a = ~(a << n); n -= 1; if (n == 0) { break; } }
            }
            for bit value in a { if (value) { a >>= int(value); } }
            for int j in {n, n + 1} { a <<= j; }''')
        converter = qasm2qcx.QASM2QCXConverter(program)
        converter.visit(program)
        ForLoopControlLoweringTests.assert_resolved_jumps(list(converter))
        self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())
        self.assertEqual(converter._QASM2QCXConverter__loop_bindings, [])
        self.assertEqual(converter._QASM2QCXConverter__loop_contexts, [])

    def test_shadowed_iterators_remain_scalar_and_read_only(self) -> None:
        prefix = 'OPENQASM 3.0; bit[4] a; bit[4] b; '
        for loop in ('for int a in [1:0]', 'for bit a in b'):
            with self.subTest(loop=loop), self.assertRaises(qasm2qcx.QASM2QCXError):
                convert(prefix + loop + ' { a <<= 1; }')
        convert(prefix + 'for int a in [1:0] { b >>= a; }')
        with self.assertRaises(qasm2qcx.QASM2QCXError):
            convert(prefix + 'for bit a in b { b >>= a; }')

    def test_array_count_evaluated_once_for_compound_assignment(self) -> None:
        program = qasm2qcx.openqasm3.parse('''OPENQASM 3.0; bit[4] a; int i = 0;
            array[int, 2] counts = {0, 1}; a <<= counts[i + 1]; a >>= int(a[i]);''')
        converter = qasm2qcx.QASM2QCXConverter(program)
        converter.visit(program)
        lines = list(converter)
        before_copy = lines[:lines.index('@QASM2QCX_REGISTER_SHIFT_0_COPY')]
        self.assertEqual(sum(line.endswith(' += 1') for line in before_copy), 1)
        self.assertEqual(sum(':= COUNTS63:QASM2QCX_INT_' in line for line in lines), 1)
        self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())

    def test_register_results_not_accepted_as_scalar_loop_expressions(self) -> None:
        prefix = 'OPENQASM 3.0; bit[4] a; int n; '
        for body in ('if (a << n) {}', 'n = int(a >> n);', 'a[a << n] = bit(true);',
                     'for int j in [0:a << n] {}', 'for int j in {a >> n} {}'):
            with self.subTest(body=body), self.assertRaises(qasm2qcx.QASM2QCXError):
                convert(prefix + 'for int i in [1:0] { ' + body + ' }')

    def test_failed_count_lowering_releases_left_expression_storage(self) -> None:
        program = qasm2qcx.openqasm3.parse('OPENQASM 3.0; bit[4] a; a = (~a) << n; int n;')
        converter = qasm2qcx.QASM2QCXConverter(program)
        with self.assertRaises(qasm2qcx.QASM2QCXError):
            converter.visit(program)
        self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())


class BitRegisterBitwiseExpressionTests(unittest.TestCase):
    def test_binary_initializers_and_assignment_output(self) -> None:
        for operator in ('&', '|', '^'):
            with self.subTest(operator=operator):
                lines = convert(f'''OPENQASM 3.0; bit[2] a = "01"; bit[2] b = "10";
                    bit[2] c = a {operator} b; c = a {operator} "11";''')
                self.assertIn(f'LET QASM2QCX_INT_0 {operator}= B1:0', lines)
                self.assertIn(f'LET QASM2QCX_INT_1 {operator}= B1:1', lines)
                self.assertEqual(lines[-2:], ['LET C1:0 := QASM2QCX_INT_0', 'LET C1:1 := QASM2QCX_INT_1'])
                self.assertEqual(sum(line.startswith('VAR QASM2QCX_') for line in lines), 2)

    def test_literal_expressions_fold_with_leading_zeroes(self) -> None:
        lines = convert('''OPENQASM 3.0; bit[4] a = ~("0011" ^ "0101") & "1010";
            bit[1] b = ~"1";''')
        self.assertEqual([line for line in lines if line.startswith('LET A1:')],
                         ['LET A1:0 := 0', 'LET A1:1 := 0', 'LET A1:2 := 0', 'LET A1:3 := 1'])
        self.assertIn('LET B1 := 0', lines)
        self.assertFalse(any(line.startswith('VAR QASM2QCX_') for line in lines))

    def test_self_references_write_only_after_complete_result(self) -> None:
        lines = convert('OPENQASM 3.0; bit[3] a = "101"; bit[3] b = "011"; a = ~(a & b) ^ a;')
        writes = [index for index, line in enumerate(lines) if line.startswith('LET A1:')]
        reads = [index for index, line in enumerate(lines) if ':= A1:' in line or '^= A1:' in line]
        self.assertEqual(len(writes), 6)
        self.assertGreater(writes[3], max(reads))

    def test_singleton_registers_remain_distinct_from_scalars(self) -> None:
        lines = convert('OPENQASM 3.0; bit[1] a = "1"; bit[1] b = ~a; a = a ^ b;')
        self.assertIn('LET QASM2QCX_INT_0 := A1', lines)
        self.assertIn('LET QASM2QCX_INT_0 ^= 1', lines)
        self.assertFalse(any('A1:' in line or 'B1:' in line for line in lines))
        for source in ('bit[1] a; bit b = ~a;', 'bit b; bit[1] a = ~b;',
                       'bit[1] a; bit b; a = a | b;'):
            with self.subTest(source=source), self.assertRaises(qasm2qcx.QASM2QCXError):
                convert('OPENQASM 3.0; ' + source)

    def test_nested_results_release_and_reuse_storage(self) -> None:
        program = qasm2qcx.openqasm3.parse('''OPENQASM 3.0; bit[4] a; bit[4] b;
            a = ~(a ^ b) & (a | "0011"); b = ~(a ^ b) & (a | "0011");''')
        converter = qasm2qcx.QASM2QCXConverter(program)
        converter.visit(program)
        lines = list(converter)
        self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())
        # Three simultaneously live four-bit buffers, reused for the second assignment.
        self.assertEqual(sum(line.startswith('VAR QASM2QCX_') for line in lines), 12)

    def test_unsupported_selections_casts_and_conditions_stay_rejected(self) -> None:
        prefix = 'OPENQASM 3.0; bit[4] a; bit[4] b; '
        for statement in ('a[0:2] = ~b;', 'a[{0, 1, 2}] = b & a;',
                          'a = a[0:4] ^ b;', 'a = a << true;', 'a = bit[4](1);',
                          'int n = int(a & b);', 'if (a | b) {}', 'bool ok = a < b;',
                          'if (false) { a = a ^ "01"; }'):
            with self.subTest(statement=statement), self.assertRaises(qasm2qcx.QASM2QCXError):
                convert(prefix + statement)

    def test_failure_releases_partial_expression_storage(self) -> None:
        # The source type pass knows later registers, but lowering must still
        # reject reading their storage before the actual declaration.
        program = qasm2qcx.openqasm3.parse('''OPENQASM 3.0; bit[2] a;
            a = ~a | b; bit[2] b;''')
        converter = qasm2qcx.QASM2QCXConverter(program)
        with self.assertRaises(qasm2qcx.QASM2QCXError):
            converter.visit(program)
        self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())


class BitRegisterBitwiseIntegrationTests(unittest.TestCase):
    def test_compounds_match_explicit_assignment(self) -> None:
        for width in (1, 4):
            for operator in ('&', '|', '^'):
                for rhs in ('a', 'b', '~a', '~(a ^ b)', '"' + '1' * width + '"'):
                    with self.subTest(width=width, operator=operator, rhs=rhs):
                        prefix = f'OPENQASM 3.0; bit[{width}] a; bit[{width}] b; '
                        self.assertEqual(convert(prefix + f'a {operator}= {rhs};'),
                                         convert(prefix + f'a = a {operator} ({rhs});'))

    def test_nested_loop_assignments_release_storage(self) -> None:
        program = qasm2qcx.openqasm3.parse('''OPENQASM 3.0;
            bit[4] a = "1010"; bit[4] b = "1100"; int n = 2;
            for int i in [0:1] {
                a ^= a & b;
                for int j in [0:n] { if (j == 1) { continue; } a = ~a; }
                while (n > 0) { b |= a; n -= 1; if (n == 0) { break; } }
            }
            for bit value in a { if (value) { b &= "1010"; } }
            for int j in {n, n + 1} { a = a | b; }''')
        converter = qasm2qcx.QASM2QCXConverter(program)
        converter.visit(program)
        ForLoopControlLoweringTests.assert_resolved_jumps(list(converter))
        self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())
        self.assertEqual(converter._QASM2QCXConverter__loop_bindings, [])
        self.assertEqual(converter._QASM2QCXConverter__loop_contexts, [])

    def test_empty_and_false_loops_accept_register_assignments(self) -> None:
        for body in ('a = ~(a ^ b);', 'a &= b;', 'a |= ~b;', 'a ^= "0011";'):
            for wrapper in ('for int i in [1:0] {{ {} }}', 'while (false) {{ {} }}',
                            'for int i in [1:0] {{ if (false) {{ {} }} }}'):
                with self.subTest(body=body, wrapper=wrapper):
                    convert('OPENQASM 3.0; bit[4] a; bit[4] b; ' + wrapper.format(body))

    def test_invalid_assignments_rejected_even_in_empty_loops(self) -> None:
        prefix = 'OPENQASM 3.0; bit[4] a; bit[2] b; bit scalar; int n; float f; '
        for body in ('a &= b;', 'a ^= "01";', 'a |= scalar;', 'a &= n;',
                     'a = ~b;', 'a = a | scalar;', 'scalar ^= ~a;', 'n &= a;',
                     'a[0:2] ^= a;', 'a[{0,1,2}] &= a;',
                     'a ^= a[f];', 'a &= a[0:2];', 'a = ~missing;'):
            for wrapper in ('{}', 'if (false) {{ {} }}', 'for int i in [1:0] {{ {} }}',
                            'while (false) {{ {} }}'):
                with self.subTest(body=body, wrapper=wrapper), self.assertRaises(qasm2qcx.QASM2QCXError):
                    convert(prefix + wrapper.format(body))

    def test_iterators_shadow_registers_and_remain_read_only(self) -> None:
        prefix = 'OPENQASM 3.0; bit[4] a; bit[4] b; '
        for loop in ('for int a in [1:0]', 'for bit a in b'):
            for body in ('b ^= a;', 'a &= b;', 'b = ~a;', 'if (false) { a ^= a; }'):
                with self.subTest(loop=loop, body=body), self.assertRaises(qasm2qcx.QASM2QCXError):
                    convert(prefix + loop + ' { ' + body + ' }')
        convert(prefix + 'for int a in [1:0] { b ^= b; }')

    def test_register_results_do_not_leak_into_scalar_loop_contexts(self) -> None:
        prefix = 'OPENQASM 3.0; bit[4] a; bit[4] b; int n; array[int, 1] values; '
        for body in ('if (a & b) {}', 'while (~a) {}', 'n = int(a | b);',
                     'for int j in {a ^ b} {}', 'for int j in [0:~a] {}',
                     'values[a & b] = 1;', 'a[a ^ b] = bit(true);'):
            with self.subTest(body=body), self.assertRaises(qasm2qcx.QASM2QCXError):
                convert(prefix + 'for int i in [1:0] { ' + body + ' }')

    def test_loop_validation_does_not_emit_or_reserve_storage(self) -> None:
        program = qasm2qcx.openqasm3.parse('''OPENQASM 3.0; bit[4] a; bit[4] b;
            for int i in [1:0] { a ^= ~(a & b); }''')
        converter = qasm2qcx.QASM2QCXConverter(program)
        before = list(converter._QASM2QCXConverter__qcx_lines)
        converter._QASM2QCXConverter__validate_loop_body(program.statements[-1].block,
                                                       {'i': qasm2qcx.ValueType.INT})
        self.assertEqual(converter._QASM2QCXConverter__qcx_lines, before)
        self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())


class ScalarBitwiseExpressionTests(unittest.TestCase):
    def test_integer_and_bit_constant_folding(self) -> None:
        self.assertEqual(convert('''OPENQASM 3.0;
            const int MASK = 7 & 3; const int INVERSE = ~MASK;
            int a = (7 | 8) ^ 3; int b = INVERSE; bit out = ~bit(false);'''),
                         ['QUBITS 0', 'VAR A1 INT', 'LET A1 := 12',
                          'VAR B1 INT', 'LET B1 := -4', 'VAR OUT7 INT', 'LET OUT7 := 1'])
        for op, expected in (('&', 0), ('|', 1), ('^', 1)):
            self.assertIn(f'LET OUT7 := {expected}', convert(
                f'OPENQASM 3.0; bit out = bit(true) {op} bit(false);'))

    def test_runtime_integer_lowering_preserves_operands(self) -> None:
        lines = convert('OPENQASM 3.0; int a = -3; int b = 7; int c = ~(a & b) | (a ^ b);')
        self.assertTrue(any(' &= B1' in line for line in lines))
        self.assertTrue(any(' ^= -1' in line for line in lines))
        self.assertTrue(any(' ^= B1' in line for line in lines))
        self.assertTrue(any(' |= QASM2QCX_INT_' in line for line in lines))
        self.assertEqual([line for line in lines if line.startswith('LET A1 ')], ['LET A1 := -3'])
        self.assertEqual([line for line in lines if line.startswith('LET B1 ')], ['LET B1 := 7'])

    def test_scalar_bit_results_use_one_bit_complement_and_keep_type(self) -> None:
        lines = convert('OPENQASM 3.0; bit a = 1; bit b = 0; bit out = ~a ^ b; bool ready = a | b;')
        self.assertTrue(any(line.endswith(' ^= 1') for line in lines))
        self.assertFalse(any(line.endswith(' ^= -1') for line in lines))
        for expression in ('~a', 'a & b', 'a | b', 'a ^ b'):
            with self.subTest(expression=expression), self.assertRaises(qasm2qcx.NoImplicitCastException):
                convert('OPENQASM 3.0; bit a = 1; bit b = 0; int c = ' + expression + ';')

    def test_integer_operand_limits_and_suppressed_constant_arithmetic(self) -> None:
        converter = qasm2qcx.QASM2QCXConverter
        for value in (converter.QCX_INT_MIN, converter.QCX_INT_MAX):
            self.assertIn(f'LET A1 := {~value}', convert(f'OPENQASM 3.0; int a = ~({value});'))
        for value in (converter.QCX_INT_MIN - 1, converter.QCX_INT_MAX + 1):
            for expression in (f'~({value})', f'({value}) & 0'):
                with self.subTest(expression=expression), self.assertRaises(qasm2qcx.QASM2QCXError):
                    convert('OPENQASM 3.0; int a = ' + expression + ';')
        self.assertIn('LET A1 := 0', convert('''OPENQASM 3.0;
            const bool SKIP = false && bool(~(1 / 0)); int a = int(SKIP);'''))

    def test_indexed_operands_source_order_and_temporary_cleanup(self) -> None:
        program = qasm2qcx.openqasm3.parser.parse('''OPENQASM 3.0;
            array[int, 2] a = {0, 1}; bit[2] flags = "01"; int i = 0;
            int c = a[i] ^ a[i + 1]; bit out = ~flags[i] | flags[i + 1];
            flags[i & 1] = flags[i] ^ flags[i + 1];''')
        converter = qasm2qcx.QASM2QCXConverter(program)
        converter.visit(program)
        lines = list(converter)
        reads = [line for line in lines if ':= A1:QASM2QCX_INT_' in line]
        self.assertEqual(len(reads), 2)
        self.assertLess(lines.index(reads[0]), lines.index(reads[1]))
        # The first access captures i, not i + 1.
        first_check = next(line for line in lines if line.startswith('ASSERT '))
        self.assertIn(f'LET {first_check.split()[1]} := I1', lines)
        self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())

    def test_conditions_loop_headers_and_typed_iterators(self) -> None:
        lines = convert('''OPENQASM 3.0; bit[2] flags = "01"; bit a = 1; bit b = 0;
            int n = 1; int total = 0;
            if (~a | b) { total = 1; }
            for int i in [0:n & 1] { total += i ^ n; }
            for int i in {~n, n | 2} { total += i; }
            for bit value in flags { a = value ^ b; if (~value) { b = a; } }
            while (a & b) { break; }
            for int i in [1:0] { total = ~i; }''')
        ForLoopControlLoweringTests.assert_resolved_jumps(lines)
        for expression in ('~n', 'n & n', 'n | n', 'n ^ n'):
            with self.subTest(expression=expression), self.assertRaises(qasm2qcx.QASM2QCXError):
                convert('OPENQASM 3.0; int n = 1; if (' + expression + ') {}')

    def test_unsupported_types_and_mixing_registers_rejected(self) -> None:
        prefix = 'OPENQASM 3.0; int n = 1; float f = 1.5; complex z = 1.0im; bool b = true; bit a = 1; bit[1] flags = "1"; '
        for expression in ('n & f', 'n | z', 'a ^ n', 'a & 1', '~f', '~z', '~b',
                           'b & b', 'a | b', '~flags', 'flags & flags'):
            with self.subTest(expression=expression), self.assertRaises(qasm2qcx.QASM2QCXError):
                convert(prefix + 'int out = int(' + expression + ');')
        for statement in ('if (false) { n = n & f; }', 'bool out = false && bool(n & f);'):
            with self.subTest(statement=statement), self.assertRaises(qasm2qcx.QASM2QCXError):
                convert(prefix + statement)
        for statement in ('for int i in [1:0] { n = ~f; }',
                          'for int i in [1:0] { n = i & a; }',
                          'for int i in [1:0] { if (false) { n = n ^ b; } }'):
            with self.subTest(statement=statement), self.assertRaises(qasm2qcx.QASM2QCXError):
                convert(prefix + statement)


class ScalarBitwiseAssignmentTests(unittest.TestCase):
    def test_scalar_integer_and_bit_compound_output(self) -> None:
        for op in ('&=', '|=', '^='):
            with self.subTest(op=op):
                self.assertEqual(convert(f'OPENQASM 3.0; int a = 7; uint b = 3; a {op} b;'),
                                 ['QUBITS 0', 'VAR A1 INT', 'LET A1 := 7',
                                  'VAR B1 UINT', 'LET B1 := 3', 'VAR QASM2QCX_UINT_0 UINT',
                                  'LET QASM2QCX_UINT_0 := :UINT:A1', f'LET QASM2QCX_UINT_0 {op} B1',
                                  'LET A1 := :INT:QASM2QCX_UINT_0'])
                self.assertEqual(convert(f'OPENQASM 3.0; bit a = 1; bit b = 0; a {op} b;'),
                                 ['QUBITS 0', 'VAR A1 INT', 'LET A1 := 1',
                                  'VAR B1 INT', 'LET B1 := 0', f'LET A1 {op} B1'])

    def test_runtime_destinations_are_captured_once_and_reserved(self) -> None:
        for declaration, assignment, target in (
                ('array[int, 2] a = {7, 3};', 'a[i + 1] ^= a[i];', 'A1'),
                ('bit[2] a = "01";', 'a[i + 1] ^= a[i];', 'A1')):
            with self.subTest(declaration=declaration):
                lines = convert('OPENQASM 3.0; int i = 0; ' + declaration + assignment)
                checks = [line for line in lines if line.startswith('ASSERT ')]
                self.assertEqual(len(checks), 4)
                destination, source = checks[0].split()[1], checks[2].split()[1]
                self.assertNotEqual(destination, source)
                self.assertTrue(lines[-1].startswith(f'LET {target}:{destination} ^= QASM2QCX_INT_'))
                self.assertEqual(sum(line.endswith(' += 1') for line in lines), 1)

    def test_static_and_singleton_element_targets(self) -> None:
        lines = convert('''OPENQASM 3.0; bit[1] flags = "1"; int i = -1;
            flags[0] &= bit(true); flags[i] ^= bit(true);
            array[int, 1] a = {7}; a[i] |= a[0];''')
        self.assertIn('LET FLAGS31 &= 1', lines)
        self.assertIn('LET FLAGS31 ^= 1', lines)
        self.assertFalse(any('FLAGS31:' in line for line in lines))
        self.assertTrue(any(line.startswith('LET A1:QASM2QCX_INT_') and line.endswith(' |= A1:0')
                            for line in lines))

    def test_self_references_and_nested_assignments_release_temporaries(self) -> None:
        program = qasm2qcx.openqasm3.parser.parse('''OPENQASM 3.0;
            int n = 3; int i = 0; bit[2] flags = "01"; array[int, 2] a = {0, 1};
            n &= ~n | (n ^ 7); a[a[i]] ^= a[i] & ~a[i];
            flags[a[i]] |= ~flags[i] & flags[i + 1];
            for int j in a { n ^= j; flags[j] &= flags[1 - j]; }
            for bit value in flags { flags[int(value)] ^= value; }
            while (false) { n |= n; }
            for int j in [1:0] { n &= j; }''')
        converter = qasm2qcx.QASM2QCXConverter(program)
        converter.visit(program)
        self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())
        ForLoopControlLoweringTests.assert_resolved_jumps(list(converter))

    def test_invalid_types_register_selections_and_iterator_writes(self) -> None:
        prefix = 'OPENQASM 3.0; int i = 0; float f = 1.5; bool b = true; complex z = 1.0im; bit a = 1; bit[2] flags = "01"; array[int, 2] values = {0, 1}; '
        for statement in ('i &= a;', 'i |= f;', 'i ^= b;', 'f &= i;', 'z ^= i;',
                          'b |= b;', 'a ^= 1;', 'a &= b;', 'flags ^= a;',
                          'flags[{0}] |= a;', 'flags[0:0] &= a;',
                          'values &= 1;', 'values[0:0] ^= 1;',
                          'values[i] |= a;', 'flags[i] &= values[i];',
                          'for int j in [1:0] { j ^= 1; }',
                          'for bit value in flags { value &= a; }',
                          'for int j in [1:0] { flags[0:0] &= a; }',
                          'for int j in [1:0] { i ^= a; }',
                          'for int j in [1:0] { flags[f] ^= a; }',
                          'for int j in [1:0] { a ^= flags[f]; }',
                          'for int j in [1:0] { a = ~flags[f]; }',
                          'for int j in [1:0] { for int k in {int(~flags[f])} {} }',
                          'if (false) { flags[i] |= 2; }'):
            with self.subTest(statement=statement), self.assertRaises(qasm2qcx.QASM2QCXError):
                convert(prefix + statement)
        maximum = qasm2qcx.QASM2QCXConverter.QCX_INT_MAX
        with self.assertRaises(qasm2qcx.QASM2QCXError):
            convert(f'OPENQASM 3.0; int i = 0; i &= {maximum + 1};')

    def test_rhs_type_failure_releases_destination_and_operand_temporaries(self) -> None:
        program = qasm2qcx.openqasm3.parser.parse('''OPENQASM 3.0;
            int i = 0; bit[2] flags = "01"; float f = 0.5;
            flags[i] ^= f + 1.0;''')
        converter = qasm2qcx.QASM2QCXConverter(program)
        with self.assertRaises(qasm2qcx.QASM2QCXError):
            converter.visit(program)
        self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())
        self.assertFalse(any(line.startswith('LET FLAGS31:') and ' ^=' in line
                             for line in converter._QASM2QCXConverter__qcx_lines))


class RuntimeBitReadTests(unittest.TestCase):
    def test_bounds_capture_and_bit_semantics(self) -> None:
        lines = convert('OPENQASM 3.0; bit[3] flags = "101"; int i = -1; '
                        'bit out = flags[i]; bool ready = flags[i];')
        checks = [line for line in lines if line.startswith('ASSERT ')]
        self.assertEqual(len(checks), 4)
        self.assertTrue(checks[0].endswith(' >= -3'))
        self.assertTrue(checks[1].endswith(' < 3'))
        index = checks[0].split()[1]
        self.assertIn(f'LET {index} := I1', lines)
        read = next(line for line in lines if line.endswith(f':= FLAGS31:{index}'))
        self.assertLess(lines.index(checks[1]), lines.index(read))
        self.assertNotIn('LET I1 += 3', lines)
        for statement in ('int n = flags[i];', 'float n = flags[i];'):
            with self.subTest(statement=statement), self.assertRaises(qasm2qcx.QASM2QCXError):
                convert('OPENQASM 3.0; bit[2] flags; int i = 0; ' + statement)

    def test_singleton_uses_scalar_storage_after_bounds_checks(self) -> None:
        lines = convert('OPENQASM 3.0; bit[1] flags = "1"; int i = -1; bit out = flags[i];')
        self.assertTrue(any(line.endswith(':= FLAGS31') for line in lines))
        self.assertFalse(any('FLAGS31:' in line for line in lines))
        self.assertTrue(any(line.endswith(' >= -1') for line in lines if line.startswith('ASSERT ')))

    def test_static_reads_keep_existing_output(self) -> None:
        self.assertEqual(convert('OPENQASM 3.0; bit[2] flags = "01"; bit out = flags[0];'),
                         ['QUBITS 0', 'VAR FLAGS31 INT 2', 'LET FLAGS31:0 := 1',
                          'LET FLAGS31:1 := 0', 'VAR OUT7 INT', 'LET OUT7 := FLAGS31:0'])

    def test_nested_reads_loop_conditions_and_casts_release_temporaries(self) -> None:
        program = qasm2qcx.openqasm3.parser.parse('''OPENQASM 3.0;
            bit[2] flags = "01"; int i = 0; bit out = flags[int(flags[i])];
            bool ready = bool(flags[i]); array[int, 2] a = {0, 1};
            out = flags[a[i]];
            for int j in a { if (flags[j]) { out = flags[j]; } }
            for bit value in flags { out = flags[int(value)]; }
            while (false) { if (flags[i]) { break; } }
            ready = false && flags[i];''')
        converter = qasm2qcx.QASM2QCXConverter(program)
        converter.visit(program)
        self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())

    def test_invalid_types_scalar_indexing_and_register_scalar_mismatch_are_rejected(self) -> None:
        prefix = 'OPENQASM 3.0; qubit q; bit scalar; bit[2] flags; int i = 0; float f = 0.5; bool b = true; '
        for statement in ('bit out = flags[f];', 'bit out = flags[b];',
                          'bit out = flags[scalar];', 'bit out = scalar[i];',
                          'bit[1] out = flags[i];', 'if (false) { if (flags[f]) {} }'):
            with self.subTest(statement=statement), self.assertRaises(qasm2qcx.QASM2QCXError):
                convert(prefix + statement)
        maximum = qasm2qcx.QASM2QCXConverter.QCX_INT_MAX
        with self.assertRaises(qasm2qcx.InvalidBitOperandException):
            convert(f'OPENQASM 3.0; bit[{maximum + 1}] flags; int i = 0; if (flags[i]) {{}}')


class RuntimeBitTargetTests(unittest.TestCase):
    def test_destination_is_captured_once_and_reserved_across_rhs(self) -> None:
        lines = convert('OPENQASM 3.0; bit[3] flags = "101"; int i = 0; '
                        'flags[i + 1] = flags[i];')
        checks = [line for line in lines if line.startswith('ASSERT ')]
        self.assertEqual(len(checks), 4)
        target_index, source_index = checks[0].split()[1], checks[2].split()[1]
        self.assertNotEqual(target_index, source_index)
        self.assertTrue(lines[-1].startswith(f'LET FLAGS31:{target_index} := QASM2QCX_INT_'))
        additions = [line for line in lines if line.endswith(' += 1')]
        self.assertEqual(len(additions), 1)

    def test_measurement_checks_target_before_m_and_copies_outcome_immediately(self) -> None:
        for statement in ('flags[i] = measure q;', 'measure q -> flags[i];'):
            with self.subTest(statement=statement):
                lines = convert('OPENQASM 3.0; qubit q; bit[2] flags; int i = -1; ' + statement)
                checks = [line for line in lines if line.startswith('ASSERT ')]
                self.assertEqual(len(checks), 2)
                index = checks[0].split()[1]
                measurement = lines.index('M 0')
                self.assertLess(lines.index(checks[1]), measurement)
                self.assertEqual(lines[measurement + 1], f'LET FLAGS31:{index} := :OUTCOME')

    def test_singleton_assignment_and_measurement_use_scalar_storage(self) -> None:
        lines = convert('OPENQASM 3.0; qubit q; bit[1] flags; int i = -1; '
                        'flags[i] = true; flags[i] = measure q;')
        self.assertFalse(any('FLAGS31:' in line for line in lines))
        self.assertEqual(lines[-2:], ['M 0', 'LET FLAGS31 := :OUTCOME'])

    def test_runtime_targets_preserve_bit_rules_and_static_qubit_restrictions(self) -> None:
        prefix = 'OPENQASM 3.0; qubit[2] q; bit[2] flags; bit scalar; int i = 0; float f = 0.5; '
        for statement in ('flags[i] = 2;', 'flags[i] = i;', 'flags[i] = f;',
                          'flags[i] = "1";', 'flags[i] = flags;', 'flags[i] += 1;',
                          'flags[f] = 1;', 'scalar[i] = 1;',
                          'flags[i] = measure q;', 'flags[i] = measure q[0:0];',
                          'flags[i] = measure q[i];', 'flags[f] = measure q[0];',
                          'for bit value in flags { value = 1; }',
                          'if (false) { flags[i] = 2; }'):
            with self.subTest(statement=statement), self.assertRaises(qasm2qcx.QASM2QCXError):
                convert(prefix + statement)

    def test_nested_targets_boolean_rhs_and_measurement_release_temporaries(self) -> None:
        program = qasm2qcx.openqasm3.parser.parse('''OPENQASM 3.0;
            qubit q; bit[2] flags = "01"; array[int, 2] a = {0, 1}; int i = 0;
            flags[a[i]] = flags[i] && !flags[i + 1];
            flags[int(flags[i])] = measure q;
            for int j in a { flags[j] = flags[1 - j]; }
            for bit value in flags { flags[int(value)] = value; }''')
        converter = qasm2qcx.QASM2QCXConverter(program)
        converter.visit(program)
        self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())


class RuntimeBitIndexRegressionTests(unittest.TestCase):
    def test_runtime_bit_access_enforces_representable_register_sizes(self) -> None:
        maximum = qasm2qcx.QASM2QCXConverter.QCX_INT_MAX
        for statement in ('bit out = flags[i];', 'flags[i] = true;', 'flags[i] = measure q;'):
            with self.subTest(statement=statement):
                lines = convert(f'OPENQASM 3.0; qubit q; bit[{maximum}] flags; int i = -1; ' + statement)
                checks = [line for line in lines if line.startswith('ASSERT ')]
                self.assertEqual(len(checks), 2)
                self.assertTrue(checks[0].endswith(f' >= {-maximum}'))
                self.assertTrue(checks[1].endswith(f' < {maximum}'))
                with self.assertRaises(qasm2qcx.InvalidBitOperandException):
                    convert(f'OPENQASM 3.0; qubit q; bit[{maximum + 1}] flags; int i = 0; ' + statement)

    def test_nested_bit_and_integer_array_indices_capture_distinct_destinations(self) -> None:
        lines = convert('''OPENQASM 3.0; bit[3] flags = "101";
            array[int, 3] a = {0, 1, 2}; int i = -1;
            flags[a[int(flags[i])]] = flags[a[i]] && !flags[int(flags[i])];''')
        checks = [line for line in lines if line.startswith('ASSERT ')]
        self.assertEqual(len(checks), 14)
        # The third capture is the outer destination, after its nested reads.
        destination = checks[4].split()[1]
        self.assertTrue(lines[-1].startswith(f'LET FLAGS31:{destination} := '))
        self.assertNotIn(destination, {line.split()[1] for line in checks[6:]})
        ForLoopControlLoweringTests.assert_resolved_jumps(lines)

    def test_nested_measurement_index_finishes_all_checks_before_m(self) -> None:
        lines = convert('''OPENQASM 3.0; qubit q; bit[2] flags = "01";
            array[int, 2] a = {0, 1}; int i = -1;
            flags[a[int(flags[i])]] = measure q;''')
        measurement = lines.index('M 0')
        checks = [position for position, line in enumerate(lines) if line.startswith('ASSERT ')]
        self.assertEqual(len(checks), 6)
        self.assertTrue(all(position < measurement for position in checks))
        self.assertTrue(lines[measurement + 1].endswith(' := :OUTCOME'))

    def test_static_assignments_and_measurements_do_not_add_runtime_checks(self) -> None:
        lines = convert('''OPENQASM 3.0; qubit[2] q; bit[2] flags;
            flags[0] = 1; flags[-1] = flags[0]; flags = measure q;''')
        self.assertEqual(lines, ['QUBITS 2', 'VAR FLAGS31 INT 2',
                                'LET FLAGS31:0 := 1', 'LET FLAGS31:1 := FLAGS31:0',
                                'M 0', 'LET FLAGS31:0 := :OUTCOME',
                                'M 1', 'LET FLAGS31:1 := :OUTCOME'])

    def test_runtime_slices_sets_multidimensional_indices_and_compound_writes_rejected(self) -> None:
        prefix = 'OPENQASM 3.0; bit[2] flags; qubit q; int i = 0; bool b = true; '
        for statement in ('bit out = flags[i:1];', 'bit out = flags[{i}];',
                          'bit out = flags[i, 0];', 'flags[i:1] = "01";',
                          'flags[{i}] = "1";', 'flags[i, 0] = 1;',
                          'flags[i][0] = 1;', 'flags[b] = measure q;',
                          'flags[{i}] = measure q;', 'flags[i:1] = measure q;',
                          'flags[i] &= 1;', 'flags[i] ^= 1;', 'flags[i] <<= 1;',
                          'for bit i in flags { flags[i] = 1; }',
                          'for bit flags in flags { flags[0] = 1; }'):
            with self.subTest(statement=statement), self.assertRaises(qasm2qcx.QASM2QCXError):
                convert(prefix + statement)

    def test_invalid_rhs_releases_reserved_destination_and_invalid_measurement_emits_no_m(self) -> None:
        for statement in ('flags[i] = 2;', 'flags[i] = measure q;', 'flags[true] = measure q[0];'):
            with self.subTest(statement=statement):
                program = qasm2qcx.openqasm3.parser.parse(
                    'OPENQASM 3.0; bit[2] flags; qubit[2] q; int i = 0; ' + statement)
                converter = qasm2qcx.QASM2QCXConverter(program)
                with self.assertRaises(qasm2qcx.QASM2QCXError):
                    converter.visit(program)
                self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())
                self.assertFalse(any(line.startswith('M ') for line in converter._QASM2QCXConverter__qcx_lines))

    def test_repeated_nested_accesses_reuse_temporaries(self) -> None:
        statement = 'flags[int(flags[i])] = flags[i] || flags[1 - i];'
        prefix = 'OPENQASM 3.0; bit[2] flags = "01"; int i = 0; '
        once = convert(prefix + statement)
        program = qasm2qcx.openqasm3.parser.parse(prefix + statement * 100)
        converter = qasm2qcx.QASM2QCXConverter(program)
        converter.visit(program)
        private_declarations = lambda lines: [line for line in lines if line.startswith('VAR QASM2QCX_')]
        self.assertEqual(private_declarations(list(converter)), private_declarations(once))
        self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())

    def test_cli_rejects_invalid_unreachable_access_without_partial_output(self) -> None:
        for statement in ('if (false) { flags[bad] = 1; }',
                          'while (false) { flags[bad] = measure q; }',
                          'bool ready = true || flags[bad];'):
            source = 'OPENQASM 3.0; qubit q; bit[2] flags; float bad = 0.5; ' + statement
            with self.subTest(statement=statement), \
                    patch('builtins.open', mock_open(read_data=source)), patch('builtins.print') as output:
                with self.assertRaises(SystemExit) as error:
                    qasm2qcx.main(['runtime-bit.qasm'])
                self.assertTrue(str(error.exception).startswith('qasm2qcx.py:'))
                output.assert_not_called()


class IntegerArrayDeclarationTests(unittest.TestCase):
    def test_declarations_and_initializers_use_qcx_integer_arrays(self) -> None:
        self.assertEqual(convert('OPENQASM 3.0; array[int, 3] values = {1, 3, 5};'),
                         ['QUBITS 0', 'VAR VALUES63 INT 3',
                          'LET VALUES63:0 := 1', 'LET VALUES63:1 := 3', 'LET VALUES63:2 := 5'])
        self.assertEqual(convert('OPENQASM 3.0; array[int[8], 1] a = {-2};'),
                         ['QUBITS 0', 'VAR A1 INT', 'LET A1:0 := -2'])
        self.assertEqual(convert('OPENQASM 3.0; array[int, 2] values;'),
                         ['QUBITS 0', 'VAR VALUES63 INT 2'])

    def test_size_uses_constants_integer_arithmetic_and_explicit_casts(self) -> None:
        for size in ('N', 'N - 1 + 1', 'int(3.5)', 'uint(3.5)'):
            with self.subTest(size=size):
                lines = convert('OPENQASM 3.0; const uint N = 3; '
                                f'array[int, {size}] a = {{1, 2, 3}};')
                self.assertEqual(lines, ['QUBITS 0', 'VAR A1 INT 3',
                                         'LET A1:0 := 1', 'LET A1:1 := 2', 'LET A1:2 := 3'])

    def test_runtime_values_and_casted_initializers_are_evaluated_in_element_order(self) -> None:
        lines = convert('OPENQASM 3.0; int seed = 3; float f = 2.5; '
                        'array[int, 3] a = {seed, seed + 1, int(f)};')
        assignments = [line for line in lines if line.startswith('LET A1:')]
        self.assertEqual(len(assignments), 3)
        self.assertEqual(assignments[0], 'LET A1:0 := SEED15')
        self.assertTrue(assignments[1].startswith('LET A1:1 := QASM2QCX_INT_'))
        self.assertTrue(assignments[2].startswith('LET A1:2 := QASM2QCX_INT_'))
        self.assertLess(lines.index(assignments[0]), lines.index(assignments[1]))
        self.assertLess(lines.index(assignments[1]), lines.index(assignments[2]))

    def test_size_rejects_zero_negative_runtime_noninteger_and_unrepresentable_values(self) -> None:
        prefix = 'OPENQASM 3.0; const int NEGATIVE = -1; int n = 3; float f = 3.5; '
        for size in ('0', 'NEGATIVE', 'n', 'n + 1', 'f', 'true', '3.5',
                     str(qasm2qcx.QASM2QCXConverter.QCX_INT_MAX + 1)):
            with self.subTest(size=size), self.assertRaises(qasm2qcx.InvalidDeclarationException):
                convert(prefix + f'array[int, {size}] a;')
        for size in ('missing', '1 / 0', '1 % 0'):
            with self.subTest(size=size), self.assertRaises(qasm2qcx.QASM2QCXError):
                convert(prefix + f'array[int, {size}] a;')

    def test_unsupported_dimensions_and_base_types_are_rejected(self) -> None:
        for declaration in ('array[int, 2, 2] a;',
                            'array[bit, 2] a;', 'array[angle, 2] a;'):
            with self.subTest(declaration=declaration), self.assertRaises(qasm2qcx.UnsupportedOpenQASMError):
                convert('OPENQASM 3.0; ' + declaration)

    def test_initializer_requires_matching_flat_array_literal_of_integers(self) -> None:
        for initializer in ('{}', '{1}', '{1, 2, 3}', '{1.5, 2}', '{true, 2}', '{1.0im, 2}'):
            with self.subTest(initializer=initializer), self.assertRaises(qasm2qcx.InvalidDeclarationException):
                convert('OPENQASM 3.0; array[int, 2] a = ' + initializer + ';')
        for initializer in ('{{1}, {2}}', '2', 'other'):
            with self.subTest(initializer=initializer), self.assertRaises(qasm2qcx.UnsupportedOpenQASMError):
                convert('OPENQASM 3.0; int other = 2; array[int, 2] a = ' + initializer + ';')

    def test_whole_arrays_cannot_be_used_as_scalars_even_when_size_is_one(self) -> None:
        for size in (1, 2):
            prefix = f'OPENQASM 3.0; array[int, {size}] a; '
            for statement in ('int value = a;', 'int value = a + 1;', 'int value = int(a);',
                              'if (a) {}', 'a = 1;', 'a += 1;', 'array[int, 1] b = {a};',
                              'for int i in [0:a] {}', 'for int i in {a} {}'):
                with self.subTest(size=size, statement=statement), self.assertRaises(qasm2qcx.UnsupportedOpenQASMError):
                    convert(prefix + statement)

    def test_sliced_writes_and_iteration_remain_unsupported(self) -> None:
        prefix = 'OPENQASM 3.0; array[int, 2] a = {1, 2}; '
        for statement in ('a[0:1] = 3;', 'for int value in a[0:1] {}'):
            with self.subTest(statement=statement), self.assertRaises(qasm2qcx.QASM2QCXError):
                convert(prefix + statement)

    def test_duplicates_forward_references_self_references_and_block_local_arrays_rejected(self) -> None:
        for source in ('array[int, 1] a; int a;',
                       'int a; array[int, 1] a;',
                       'array[int, 1] a = {seed}; int seed = 1;',
                       'array[int, N] a; const int N = 1;',
                       'array[int, 1] a = {a};',
                       'if (false) { array[int, 1] a; }',
                       'for int i in [1:0] { array[int, 1] a; }',
                       'while (false) { array[int, 1] a; }'):
            with self.subTest(source=source), self.assertRaises(
                    (qasm2qcx.QASM2QCXError, qasm2qcx.openqasm3.parser.QASM3ParsingError)):
                convert('OPENQASM 3.0; ' + source)

    def test_size_evaluation_restores_expression_state_on_success_and_failure(self) -> None:
        converter = qasm2qcx.QASM2QCXConverter(qasm2qcx.openqasm3.parser.parse('OPENQASM 3.0;'))
        converter._QASM2QCXConverter__expression_kind = qasm2qcx.ExpressionKind.ARITHMETIC
        converter._QASM2QCXConverter__value = 17
        converter._QASM2QCXConverter__value_type = qasm2qcx.ValueType.INT
        converter._QASM2QCXConverter__value_kind = qasm2qcx.ValueKind.LITERAL
        converter._QASM2QCXConverter__evaluate_constant = False
        fields = ('expression_kind', 'value', 'value_type', 'value_kind', 'evaluate_constant')
        before = [getattr(converter, '_QASM2QCXConverter__' + field) for field in fields]
        for size in ('2', '0', 'missing'):
            declaration = qasm2qcx.openqasm3.parser.parse(f'OPENQASM 3.0; array[int, {size}] a;').statements[0]
            if size == '2':
                self.assertEqual(converter._QASM2QCXConverter__array_size(declaration.type, 'a'), 2)
            else:
                with self.assertRaises(qasm2qcx.QASM2QCXError):
                    converter._QASM2QCXConverter__array_size(declaration.type, 'a')
            self.assertEqual([getattr(converter, '_QASM2QCXConverter__' + field) for field in fields], before)
        self.assertEqual(converter._QASM2QCXConverter__qcx_lines, ['QUBITS 0'])


class BooleanArrayDeclarationTests(unittest.TestCase):
    @staticmethod
    def converter(source: str):
        program = qasm2qcx.openqasm3.parser.parse('OPENQASM 3; ' + source)
        converter = qasm2qcx.QASM2QCXConverter(program)
        converter.visit(program)
        return converter

    def test_declarations_use_sized_int_storage_but_keep_boolean_identity(self) -> None:
        self.assertEqual(convert('OPENQASM 3; array[bool, 3] flags = {true, false, true};'),
                         ['QUBITS 0', 'VAR FLAGS31 INT 3', 'LET FLAGS31:0 := 1',
                          'LET FLAGS31:1 := 0', 'LET FLAGS31:2 := 1'])
        self.assertEqual(convert('OPENQASM 3; array[bool, 1] a = {false};'),
                         ['QUBITS 0', 'VAR A1 INT', 'LET A1:0 := 0'])
        self.assertEqual(convert('OPENQASM 3; array[bool, 2] flags;'),
                         ['QUBITS 0', 'VAR FLAGS31 INT 2'])
        converter = self.converter('array[bool, 2] a; bool b;')
        self.assertEqual(converter._QASM2QCXConverter__array_element_types, {'A1': qasm2qcx.ValueType.BOOL})
        self.assertEqual(converter._QASM2QCXConverter__array_source_sizes, {'a': 2})
        self.assertEqual(converter._QASM2QCXConverter__bool_variable_names, {'A1', 'B1'})
        self.assertEqual(converter._QASM2QCXConverter__int_variable_name_size_map, {})
        self.assertEqual(converter._QASM2QCXConverter__type_of('A1'), qasm2qcx.ValueType.BOOL)

    def test_scalar_boolean_storage_is_unchanged(self) -> None:
        self.assertEqual(convert('OPENQASM 3; bool a = true; bool b = false;'),
                         ['QUBITS 0', 'VAR A1 INT', 'LET A1 := 1', 'VAR B1 INT', 'LET B1 := 0'])

    def test_initializer_uses_existing_scalar_boolean_literal_conversions(self) -> None:
        self.assertEqual(convert('OPENQASM 3; const bool YES = true; const int ONE = 1; '
                                'array[bool, 6] a = {YES, ONE, 0, uint(1), false, 1 - 1};'),
                         ['QUBITS 0', 'VAR A1 INT 6', 'LET A1:0 := 1', 'LET A1:1 := 1',
                          'LET A1:2 := 0', 'LET A1:3 := 1', 'LET A1:4 := 0', 'LET A1:5 := 0'])

    def test_runtime_boolean_and_bit_initializers_keep_order_and_release_temporaries(self) -> None:
        converter = self.converter('bool ready = true; bit b = 1; int n = 2; '
            'array[bool, 5] a = {ready, b, !ready, n > 0, bool(n)};')
        lines = converter._QASM2QCXConverter__qcx_lines
        assignments = [line for line in lines if line.startswith('LET A1:')]
        self.assertEqual(assignments[:2], ['LET A1:0 := READY31', 'LET A1:1 := B1'])
        self.assertEqual(len(assignments), 5)
        for index, assignment in enumerate(assignments):
            self.assertTrue(assignment.startswith(f'LET A1:{index} := '))
        self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())

    def test_short_circuit_initializers_lower_checks_inside_their_paths(self) -> None:
        converter = self.converter('array[int, 1] values = {1}; int bad = 1; int zero = 0; '
            'array[bool, 3] flags = {false && values[bad] > 0, true || bool(1 / zero), bool(0)};')
        self.assertIn('VAR FLAGS31 INT 3', converter._QASM2QCXConverter__qcx_lines)
        self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())
        ForLoopControlLoweringTests.assert_resolved_jumps(converter._QASM2QCXConverter__qcx_lines)

    def test_size_uses_constant_integer_expressions_in_declaration_scope(self) -> None:
        for size in ('N', 'N - 1 + 1', 'int(2.5)', 'uint(2)', 'int(true) + 1'):
            with self.subTest(size=size):
                converter = self.converter('const uint N = 2; '
                    f'array[bool, {size}] a; for int N in [0:1] {{}}')
                self.assertEqual(converter._QASM2QCXConverter__array_source_sizes['a'], 2)
                self.assertIn('VAR A1 INT 2', converter._QASM2QCXConverter__qcx_lines)

    def test_invalid_dimensions_are_rejected(self) -> None:
        for size in ('0', 'NEGATIVE', 'n', 'true', '2.5', 'missing',
                     str(qasm2qcx.QASM2QCXConverter.QCX_INT_MAX + 1)):
            with self.subTest(size=size), self.assertRaises(qasm2qcx.QASM2QCXError):
                convert('OPENQASM 3; const int NEGATIVE = -1; int n = 2; '
                        f'array[bool, {size}] a;')
        with self.assertRaises(qasm2qcx.UnsupportedOpenQASMError):
            convert('OPENQASM 3; array[bool, 2, 2] a;')

    def test_initializer_requires_a_matching_flat_array_literal(self) -> None:
        for initializer in ('{}', '{true}', '{true, false, true}'):
            with self.subTest(initializer=initializer), self.assertRaises(qasm2qcx.InvalidDeclarationException):
                convert(f'OPENQASM 3; array[bool, 2] a = {initializer};')
        for initializer in ('{{true}, {false}}', 'true', 'other'):
            with self.subTest(initializer=initializer), self.assertRaises(qasm2qcx.UnsupportedOpenQASMError):
                convert(f'OPENQASM 3; bool other = true; array[bool, 2] a = {initializer};')

    def test_invalid_boolean_conversions_release_completed_initializer_temporaries(self) -> None:
        for initializer in ('2', '-1', '1.0', 'n', 'u', 'f', '1.0im', 'complex(ready)', 'n + 1'):
            with self.subTest(initializer=initializer):
                program = qasm2qcx.openqasm3.parser.parse('OPENQASM 3; bool ready = true; '
                    'int n = 1; uint u = 1; float f = 1; '
                    f'array[bool, 1] a = {{{initializer}}};')
                converter = qasm2qcx.QASM2QCXConverter(program)
                with self.assertRaises(qasm2qcx.NoImplicitCastException):
                    converter.visit(program)
                self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())

    def test_whole_arrays_are_not_scalars_even_when_size_is_one(self) -> None:
        for size in (1, 2):
            for statement in ('bool b = a;', 'bool b = bool(a);', 'bool b = !a;',
                              'if (a) {}', 'a = true;', 'bit b = a;', 'array[bool, 1] b = {a};'):
                with self.subTest(size=size, statement=statement), self.assertRaises(qasm2qcx.QASM2QCXError):
                    convert(f'OPENQASM 3; array[bool, {size}] a; {statement}')

    def test_iteration_requires_a_matching_boolean_iterator(self) -> None:
        for statement in ('for bit b in a {}', 'for int b in a {}'):
            with self.subTest(statement=statement), self.assertRaises(qasm2qcx.UnsupportedOpenQASMError):
                convert('OPENQASM 3; array[bool, 2] a; ' + statement)

    def test_duplicates_forward_references_self_references_and_local_arrays_are_rejected(self) -> None:
        for source in ('array[bool, 1] a; bool a;', 'bool a; array[bool, 1] a;',
                       'array[bool, 1] a = {ready}; bool ready = true;',
                       'array[bool, N] a; const int N = 1;', 'array[bool, 1] a = {a};',
                       'if (false) { array[bool, 1] a; }', 'for int i in [1:0] { array[bool, 1] a; }'):
            with self.subTest(source=source), self.assertRaises(
                    (qasm2qcx.QASM2QCXError, qasm2qcx.openqasm3.parser.QASM3ParsingError)):
                convert('OPENQASM 3; ' + source)


class BooleanArrayAccessTests(unittest.TestCase):
    def assert_clean_conversion(self, source: str):
        converter = BooleanArrayDeclarationTests.converter(source)
        self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())
        ForLoopControlLoweringTests.assert_resolved_jumps(converter._QASM2QCXConverter__qcx_lines)
        return converter

    def test_static_reads_and_writes_keep_boolean_type_and_singleton_identity(self) -> None:
        converter = self.assert_clean_conversion('array[bool, 1] flags = {true}; '
            'bool ready = flags[-1]; flags[0] = ready;')
        lines = converter._QASM2QCXConverter__qcx_lines
        self.assertIn('LET READY31 := FLAGS31:0', lines)
        self.assertIn('LET FLAGS31:0 := READY31', lines)
        expression = qasm2qcx.openqasm3.parser.parse('OPENQASM 3; bool b = flags[0];').statements[0].init_expression
        self.assertEqual(converter._QASM2QCXConverter__condition_operand(expression),
                         ('FLAGS31:0', qasm2qcx.ValueType.BOOL, qasm2qcx.ValueKind.LVALUE))

    def test_runtime_reads_use_boolean_snapshots_and_private_checked_indices(self) -> None:
        converter = self.assert_clean_conversion('array[bool, 2] flags = {true, false}; int i = -1; '
            'bool ready = flags[i];')
        lines = converter._QASM2QCXConverter__qcx_lines
        self.assertIn('LET QASM2QCX_INT_0 := I1', lines)
        self.assertIn('ASSERT QASM2QCX_INT_0 >= -2', lines)
        self.assertIn('ASSERT QASM2QCX_INT_0 < 2', lines)
        self.assertIn('LET QASM2QCX_INT_1 := FLAGS31:QASM2QCX_INT_0', lines)
        self.assertIn('LET READY31 := QASM2QCX_INT_1', lines)

    def test_assignments_capture_destination_once_and_reserve_it_during_rhs(self) -> None:
        converter = self.assert_clean_conversion('array[bool, 2] flags = {true, false}; int i = 0; int j = 1; '
            'flags[i] = flags[j] && !flags[i];')
        lines = converter._QASM2QCXConverter__qcx_lines
        self.assertEqual(sum(line.endswith(' := I1') for line in lines), 2)  # Target and explicit RHS read.
        self.assertEqual(sum(line.endswith(' := J1') for line in lines), 1)
        self.assertTrue(any(line.startswith('LET FLAGS31:QASM2QCX_INT_0 := ') for line in lines))

    def test_elements_work_in_conditions_logical_values_equality_and_casts(self) -> None:
        self.assert_clean_conversion('array[bool, 2] flags = {true, false}; int i = 0; bit b = 1; '
            'bool ready = flags[i] || flags[1]; ready = flags[0] == b; '
            'if (flags[i] && !flags[1]) { flags[1] = true; } '
            'while (flags[1]) { flags[1] = false; } '
            'int n = int(flags[i]); uint u = uint[4](flags[i]); '
            'float f = float(flags[0]); complex z = complex(flags[i]); '
            'array[bool, 2] copy = {flags[0], !flags[i]};')

    def test_boolean_elements_support_scalar_bit_initializers_and_indexed_bit_writes(self) -> None:
        self.assert_clean_conversion('array[bool, 2] flags = {true, false}; int i = 0; int j = 1; '
            'bit b = flags[0]; b = flags[j]; bit[2] bits = "00"; bits[i] = flags[j]; '
            'flags[i] = bits[j];')

    def test_nested_integer_indices_and_explicit_bool_index_casts(self) -> None:
        self.assert_clean_conversion('array[int, 2] indices = {1, 0}; array[bool, 2] flags = {true, false}; '
            'int i = 0; bool b = flags[indices[i]]; flags[int(flags[i])] = !b;')

    def test_compound_assignments_are_rejected_even_in_empty_loops(self) -> None:
        for operator in ('+=', '-=', '*=', '/=', '%=', '**=', '&=', '|=', '^=', '<<=', '>>='):
            for wrapper in ('{}', 'for int i in [1:0] {{ {} }}'):
                with self.subTest(operator=operator, wrapper=wrapper), self.assertRaises(qasm2qcx.UnsupportedOpenQASMError):
                    convert('OPENQASM 3; array[bool, 2] flags; '
                            + wrapper.format(f'flags[0] {operator} true;'))

    def test_invalid_indices_selections_and_implicit_boolean_indices_are_rejected(self) -> None:
        for access in ('flags[-3]', 'flags[2]', 'flags[true]', 'flags[0.5]',
                       'flags[flags[0]]', 'flags[0:1]', 'flags[{0, 1}]', 'flags[0, 1]'):
            for statement in (f'bool b = {access};', f'{access} = true;'):
                with self.subTest(statement=statement), self.assertRaises(qasm2qcx.QASM2QCXError):
                    convert('OPENQASM 3; array[bool, 2] flags; ' + statement)

    def test_scalar_boolean_operator_and_assignment_restrictions_are_preserved(self) -> None:
        for statement in ('int n = flags[0] + 1;', 'bool b = ~flags[0];',
                          'int n = -flags[0];', 'bool b = flags[0] & true;',
                          'bool b = flags[0] < flags[1];', 'bool b = flags[0] == 1;',
                          'flags[i] = n;', 'flags[i] = 2;'):
            with self.subTest(statement=statement), self.assertRaises(qasm2qcx.QASM2QCXError):
                convert('OPENQASM 3; array[bool, 2] flags; int i = 0; int n = 1; ' + statement)

    def test_elements_cannot_be_measurement_targets_or_bare_gate_parameters(self) -> None:
        for statement in ('flags[0] = measure q;', 'rx(flags[0]) q;', 'gphase(flags[0]);'):
            with self.subTest(statement=statement), self.assertRaises(qasm2qcx.QASM2QCXError):
                convert('OPENQASM 3; include "stdgates.inc"; qubit q; array[bool, 2] flags; ' + statement)

    def test_invalid_rhs_releases_destination_index_and_completed_temporaries(self) -> None:
        program = qasm2qcx.openqasm3.parser.parse('OPENQASM 3; array[bool, 2] flags; int i = 0; '
            'flags[i] = complex(flags[i]);')
        converter = qasm2qcx.QASM2QCXConverter(program)
        with self.assertRaises(qasm2qcx.NoImplicitCastException):
            converter.visit(program)
        self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())


class BooleanArrayLoopTests(unittest.TestCase):
    def assert_clean_conversion(self, source: str):
        converter = BooleanArrayDeclarationTests.converter(source)
        self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())
        self.assertEqual(converter._QASM2QCXConverter__loop_bindings, [])
        self.assertEqual(converter._QASM2QCXConverter__loop_contexts, [])
        ForLoopControlLoweringTests.assert_resolved_jumps(converter._QASM2QCXConverter__qcx_lines)
        return converter

    def test_loop_uses_int_storage_but_boolean_binding_and_live_read(self) -> None:
        converter = self.assert_clean_conversion('array[bool, 3] flags = {true, false, true}; int count = 0; '
            'for bool flag in flags { if (flag) { count += 1; } flags[1] = true; }')
        lines = converter._QASM2QCXConverter__qcx_lines
        body = lines.index('@QASM2QCX_LOOP_0_BODY')
        self.assertIn('VAR QASM2QCX_INT_0 INT', lines[:body])
        self.assertIn('VAR QASM2QCX_INT_1 INT', lines[:body])
        self.assertEqual(lines[body + 1], 'LET QASM2QCX_INT_0 := FLAGS31:QASM2QCX_INT_1')
        self.assertEqual(lines.count('LET COUNT31 += 1'), 1)
        self.assertEqual(lines.count('LET FLAGS31:1 := 1'), 1)
        self.assertTrue(any(line.startswith('JUMPIF ') and ' QASM2QCX_INT_0 \\= 0' in line for line in lines))

    def test_singleton_loop_keeps_explicit_index_and_endpoint_guard(self) -> None:
        converter = self.assert_clean_conversion('array[bool, 1] flags = {true}; '
            'for bool flag in flags { continue; }')
        lines = converter._QASM2QCXConverter__qcx_lines
        self.assertIn('LET QASM2QCX_INT_0 := FLAGS31:QASM2QCX_INT_1', lines)
        self.assertIn('JUMPIF QASM2QCX_LOOP_0_END QASM2QCX_INT_1 == 0', lines)

    def test_loop_supports_logical_values_comparisons_casts_and_scalar_bit_copy(self) -> None:
        self.assert_clean_conversion('array[bool, 2] flags = {true, false}; bool ready = false; '
            'bit b = 0; bit[2] bits = "00"; int i = 0; uint u; float f; complex z; '
            'for bool flag in flags { ready = !flag || flag == b; b = flag; bits[i] = flag; '
            'u = uint[4](flag); f = float(flag); z = complex(flag); '
            'if (flag && ready) { flags[i] = flag; } i += 1; }')

    def test_typed_runtime_metadata_is_preserved_in_nested_and_empty_loops(self) -> None:
        self.assert_clean_conversion('array[bool, 2] flags; int count = 0; '
            'for int outer in [1:0] { for bool flag in flags { '
            'for int j in [0:int(flag)] { count += j; } '
            'for int j in {1 / int(flag)} { count += j; } } }')
        converter = BooleanArrayDeclarationTests.converter('int flag = 1;')
        binding = qasm2qcx._RuntimeIteratorBinding('PRIVATE_BOOL', qasm2qcx.ValueType.BOOL)
        expression = qasm2qcx.ast.Identifier('flag')
        with converter._QASM2QCXConverter__iterator_binding('flag', binding):
            self.assertEqual(converter._QASM2QCXConverter__condition_operand(expression),
                             ('PRIVATE_BOOL', qasm2qcx.ValueType.BOOL, qasm2qcx.ValueKind.LVALUE))
            self.assertEqual(converter._QASM2QCXConverter__loop_expression_info(
                expression, {'flag': qasm2qcx.ValueType.BOOL}), (qasm2qcx.ValueType.BOOL, True))

    def test_iterators_can_shadow_source_arrays_and_mixed_type_outer_iterators(self) -> None:
        self.assert_clean_conversion('array[bool, 2] flags = {true, false}; '
            'array[uint[4], 1] values = {1}; bit[1] bits; bool ready = false; int count = 0; '
            'for bool flags in flags { ready = flags; } ready = flags[0]; '
            'for bool flag in flags { for int flag in [0:1] { count += flag; } '
            'for uint[4] flag in values { count += int(flag); } '
            'for bit flag in bits { ready = flag; } ready = flag; }')

    def test_read_only_boolean_iterators_rejected_even_in_empty_outer_loop(self) -> None:
        for body in ('flag = true;', 'flag += 1;', 'flag[0] = false;', 'flag = measure q;',
                     'if (false) { flag = true; }', 'while (false) { flag = false; }'):
            for wrapper in ('{}', 'for int outer in [1:0] {{ {} }}'):
                with self.subTest(body=body, wrapper=wrapper), self.assertRaises(qasm2qcx.QASM2QCXError):
                    convert('OPENQASM 3; qubit q; array[bool, 2] flags; '
                            + wrapper.format(f'for bool flag in flags {{ {body} }}'))

    def test_unsupported_operators_and_implicit_integer_bounds_are_rejected(self) -> None:
        for body in ('count = ~flag;', 'count = -flag;', 'count = flag & true;',
                     'count = flag << 1;', 'count = flag ** 2;', 'if (flag < true) {}',
                     'for int j in [0:flag] {}', 'for int j in {flag} {}'):
            with self.subTest(body=body), self.assertRaises(qasm2qcx.QASM2QCXError):
                convert('OPENQASM 3; array[bool, 2] flags; int count = 0; '
                        f'for bool flag in flags {{ {body} }}')

    def test_mismatched_types_and_non_array_iterables_remain_unsupported(self) -> None:
        for source in ('array[int, 2] values;', 'array[uint, 2] values;',
                       'array[float, 2] values;', 'bit[2] values;', 'bool values;'):
            with self.subTest(source=source), self.assertRaises(qasm2qcx.UnsupportedOpenQASMError):
                convert('OPENQASM 3; ' + source + 'for bool flag in values {}')
        for source in ('[0:1]', '{true, false}', 'flags[0:1]', 'flags[0]'):
            with self.subTest(source=source), self.assertRaises(qasm2qcx.UnsupportedOpenQASMError):
                convert('OPENQASM 3; array[bool, 2] flags; '
                        f'for int outer in [1:0] {{ for bool flag in {source} {{}} }}')

    def test_failed_nested_loop_cleans_up_without_releasing_caller_storage(self) -> None:
        converter = BooleanArrayDeclarationTests.converter('array[bool, 2] flags; bool ready;')
        caller = converter._QASM2QCXConverter__add_new_temporary_variable(qasm2qcx.ValueType.BOOL)
        statement = qasm2qcx.openqasm3.parser.parse('OPENQASM 3; for bool flag in flags { '
            'for bool other in flags { ready = complex(flag || other); } }').statements[0]
        with self.assertRaises(qasm2qcx.NoImplicitCastException):
            converter.visit(statement)
        self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, {caller})
        self.assertEqual(converter._QASM2QCXConverter__loop_bindings, [])
        self.assertEqual(converter._QASM2QCXConverter__loop_contexts, [])


class UnsignedArrayDeclarationTests(unittest.TestCase):
    @staticmethod
    def converter(source: str):
        program = qasm2qcx.openqasm3.parser.parse('OPENQASM 3; ' + source)
        converter = qasm2qcx.QASM2QCXConverter(program)
        converter.visit(program)
        return converter

    def test_native_and_sized_declarations_use_uint_storage(self) -> None:
        native = qasm2qcx.QASM2QCXConverter.QCX_UINT_WIDTH
        for base in ('uint', f'uint[{native}]', 'uint[4]'):
            with self.subTest(base=base):
                self.assertEqual(convert(f'OPENQASM 3; array[{base}, 2] a;'),
                                 ['QUBITS 0', 'VAR A1 UINT 2'])
                self.assertEqual(convert(f'OPENQASM 3; array[{base}, 1] a;'),
                                 ['QUBITS 0', 'VAR A1 UINT'])

    def test_element_metadata_preserves_native_and_explicit_widths(self) -> None:
        native = qasm2qcx.QASM2QCXConverter.QCX_UINT_WIDTH
        converter = self.converter(f'array[uint, 2] a; array[uint[{native}], 3] b; array[uint[4], 1] c;')
        self.assertEqual(converter._QASM2QCXConverter__array_source_sizes, {'a': 2, 'b': 3, 'c': 1})
        self.assertEqual(converter._QASM2QCXConverter__array_element_types,
                         {name: qasm2qcx.ValueType.UINT for name in ('A1', 'B1', 'C1')})
        self.assertEqual(converter._QASM2QCXConverter__uint_variable_types,
                         {'A1': qasm2qcx._UnsignedIntegerType(native, False),
                          'B1': qasm2qcx._UnsignedIntegerType(native, True),
                          'C1': qasm2qcx._UnsignedIntegerType(4, True)})
        self.assertEqual(converter._QASM2QCXConverter__uint_variable_name_size_map,
                         {'A1': 2, 'B1': 3, 'C1': 1})
        self.assertEqual(converter._QASM2QCXConverter__int_variable_name_size_map, {})

    def test_narrow_initializer_masks_each_element_using_scalar_rules(self) -> None:
        maximum = qasm2qcx.QASM2QCXConverter.QCX_UINT_MAX
        self.assertEqual(convert('OPENQASM 3; array[uint[4], 4] a = {-1, 16, 3, true};'),
                         ['QUBITS 0', 'VAR A1 UINT 4', f'LET A1:0 := {maximum}', 'LET A1:0 &= 15',
                          'LET A1:1 := 16', 'LET A1:1 &= 15', 'LET A1:2 := 3', 'LET A1:2 &= 15',
                          'LET A1:3 := 1', 'LET A1:3 &= 15'])
        self.assertEqual(convert('OPENQASM 3; array[uint[1], 1] a = {2};'),
                         ['QUBITS 0', 'VAR A1 UINT', 'LET A1:0 := 2', 'LET A1:0 &= 1'])

    def test_native_initializers_do_not_add_narrow_masks(self) -> None:
        maximum = qasm2qcx.QASM2QCXConverter.QCX_UINT_MAX
        lines = convert(f'OPENQASM 3; array[uint, 3] a = {{0, -1, uint({maximum})}};')
        self.assertEqual(lines, ['QUBITS 0', 'VAR A1 UINT 3', 'LET A1:0 := 0',
                                 f'LET A1:1 := {maximum}', f'LET A1:2 := {maximum}'])

    def test_every_supported_width_uses_its_own_mask(self) -> None:
        native = qasm2qcx.QASM2QCXConverter.QCX_UINT_WIDTH
        for width in range(1, native + 1):
            with self.subTest(width=width):
                lines = convert(f'OPENQASM 3; array[uint[{width}], 1] a = {{-1}};')
                masks = [line for line in lines if line.startswith('LET A1:0 &=')]
                self.assertEqual(masks, [f'LET A1:0 &= {(1 << width) - 1}'] if width < native else [])

    def test_runtime_initializers_keep_order_casts_and_intermediate_widths(self) -> None:
        converter = self.converter('int n = -1; uint[4] seed = 15; float f = 2.5; '
            'array[uint[8], 4] a = {n, seed + uint[4](1), seed, uint(f)};')
        lines = list(converter)
        assignments = [line for line in lines if line.startswith('LET A1:') and ' := ' in line]
        self.assertEqual(len(assignments), 4)
        self.assertEqual(assignments[0], 'LET A1:0 := :UINT:N1')
        self.assertEqual(assignments[2], 'LET A1:2 := SEED15')
        for index, assignment in enumerate(assignments):
            self.assertTrue(assignment.startswith(f'LET A1:{index} := '))
            self.assertEqual(lines[lines.index(assignment) + 1], f'LET A1:{index} &= 255')
        self.assertTrue(any(line.startswith('LET QASM2QCX_UINT_') and line.endswith(' &= 15') for line in lines))
        self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())

    def test_width_and_size_constants_are_resolved_before_iterator_shadowing(self) -> None:
        converter = self.converter('const int W = 4; const uint N = 2; '
            'array[uint[W], N] a = {1, 17}; for int W in [0:1] {} for int N in [0:1] {}')
        self.assertEqual(converter._QASM2QCXConverter__uint_variable_types['A1'],
                         qasm2qcx._UnsignedIntegerType(4, True))
        self.assertEqual(converter._QASM2QCXConverter__array_source_sizes['a'], 2)
        self.assertIn('LET A1:1 &= 15', list(converter))
        converter = self.converter('const uint W = 2; array[uint[W + 2], W + 1] a;')
        self.assertEqual(converter._QASM2QCXConverter__uint_variable_types['A1'].width, 4)
        self.assertEqual(converter._QASM2QCXConverter__array_source_sizes['a'], 3)

    def test_invalid_widths_and_sizes_are_rejected(self) -> None:
        native = qasm2qcx.QASM2QCXConverter.QCX_UINT_WIDTH
        for width in ('ZERO', 'NEGATIVE', str(native + 1), 'n', 'true', '2.5', 'missing'):
            with self.subTest(width=width), self.assertRaises(qasm2qcx.QASM2QCXError):
                convert('OPENQASM 3; const int ZERO = 0; const int NEGATIVE = -1; int n = 2; '
                        f'array[uint[{width}], 2] a;')
        for size in ('0', 'NEGATIVE', 'n', 'true', '2.5', 'missing',
                     str(qasm2qcx.QASM2QCXConverter.QCX_INT_MAX + 1)):
            with self.subTest(size=size), self.assertRaises(qasm2qcx.QASM2QCXError):
                convert(f'OPENQASM 3; const int NEGATIVE = -1; int n = 2; array[uint[4], {size}] a;')
        with self.assertRaises(qasm2qcx.UnsupportedOpenQASMError):
            convert('OPENQASM 3; array[uint, 2, 2] a;')

    def test_initializer_must_be_a_flat_literal_with_matching_length(self) -> None:
        for initializer in ('{}', '{1}', '{1, 2, 3}'):
            with self.subTest(initializer=initializer), self.assertRaises(qasm2qcx.InvalidDeclarationException):
                convert(f'OPENQASM 3; array[uint, 2] a = {initializer};')
        for initializer in ('{{1}, {2}}', '2', 'other'):
            with self.subTest(initializer=initializer), self.assertRaises(qasm2qcx.UnsupportedOpenQASMError):
                convert(f'OPENQASM 3; uint other = 2; array[uint, 2] a = {initializer};')

    def test_noninteger_values_require_explicit_uint_casts(self) -> None:
        for value in ('1.5', 'f', '1.0im', 'complex(f)'):
            with self.subTest(value=value):
                program = qasm2qcx.openqasm3.parser.parse(
                    f'OPENQASM 3; float f = 2.5; array[uint[4], 1] a = {{{value}}};')
                converter = qasm2qcx.QASM2QCXConverter(program)
                with self.assertRaises(qasm2qcx.NoImplicitCastException):
                    converter.visit(program)
                self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())
        self.assertIn('LET A1:0 := 2', convert('OPENQASM 3; array[uint[4], 1] a = {uint(2.5)};'))

    def test_whole_arrays_are_not_scalars_and_iteration_requires_matching_types(self) -> None:
        for size in (1, 2):
            for statement in ('uint x = a;', 'uint x = uint(a);', 'a = 1;',
                              'for float x in a {}', 'for int x in a {}'):
                with self.subTest(size=size, statement=statement), self.assertRaises(qasm2qcx.UnsupportedOpenQASMError):
                    convert(f'OPENQASM 3; array[uint[4], {size}] a; {statement}')

    def test_forward_references_duplicates_self_references_and_local_arrays_are_rejected(self) -> None:
        for source in ('array[uint[W], 2] a; const int W = 4;',
                       'array[uint, N] a; const int N = 2;',
                       'array[uint, 1] a = {seed}; uint seed = 1;',
                       'array[uint, 1] a = {a};', 'array[uint, 1] a; uint a;',
                       'uint a; array[uint, 1] a;', 'if (false) { array[uint, 1] a; }',
                       'for int i in [1:0] { array[uint, 1] a; }'):
            with self.subTest(source=source), self.assertRaises(
                    (qasm2qcx.QASM2QCXError, qasm2qcx.openqasm3.parser.QASM3ParsingError)):
                convert('OPENQASM 3; ' + source)


class UnsignedArrayAccessTests(unittest.TestCase):
    def test_static_and_runtime_reads_retain_width_metadata(self) -> None:
        for index in ('0', '-1', 'i'):
            with self.subTest(index=index):
                program = qasm2qcx.openqasm3.parser.parse(
                    f'OPENQASM 3; array[uint[4], 2] a = {{15, 3}}; int i = 1; uint x = a[{index}];')
                converter = qasm2qcx.QASM2QCXConverter(program)
                converter.visit(program)
                expression = program.statements[-1].init_expression
                self.assertEqual(converter._QASM2QCXConverter__uint_expression_type(expression),
                                 qasm2qcx._UnsignedIntegerType(4, True))
                if index == 'i':
                    self.assertEqual(converter._QASM2QCXConverter__temporary_uint_types['QASM2QCX_UINT_0'],
                                     qasm2qcx._UnsignedIntegerType(4, True))
                    self.assertIn('LET QASM2QCX_UINT_0 := A1:QASM2QCX_INT_0', list(converter))
                self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())

    def test_singleton_and_negative_indices_keep_array_identity(self) -> None:
        lines = convert('OPENQASM 3; array[uint[4], 1] a = {15}; uint x = a[-1]; a[0] = 17;')
        self.assertIn('VAR A1 UINT', lines)
        self.assertIn('LET X1 := A1:0', lines)
        self.assertIn('LET A1:0 := 17', lines)
        self.assertEqual(lines[-1], 'LET A1:0 &= 15')

    def test_array_width_metadata_is_available_without_evaluating_indices(self) -> None:
        converter = UnsignedArrayDeclarationTests.converter(
            'const int W = 4; array[uint[W], 2] a; int i = 0;')
        expression = qasm2qcx.openqasm3.parser.parse('OPENQASM 3; uint x = a[1 / i];').statements[0].init_expression
        before = converter._QASM2QCXConverter__qcx_lines.copy()
        with converter._QASM2QCXConverter__iterator_binding('W', 0):
            self.assertEqual(converter._QASM2QCXConverter__integer_operand_info(expression, None),
                             (qasm2qcx.ValueType.UINT, qasm2qcx._UnsignedIntegerType(4, True), True))
        self.assertEqual(converter._QASM2QCXConverter__qcx_lines, before)

    def test_all_compound_assignments_capture_destination_index_once(self) -> None:
        for operator in ('+=', '-=', '*=', '/=', '%=', '&=', '|=', '^=', '<<=', '>>=', '**='):
            with self.subTest(operator=operator):
                converter = UnsignedArrayDeclarationTests.converter('array[uint[4], 2] a = {3, 5}; '
                    f'int i = -1; uint[4] rhs = 1; a[i] {operator} rhs;')
                lines = list(converter)
                self.assertEqual(lines.count('LET QASM2QCX_INT_0 := I1'), 1)
                self.assertEqual(sum(line.endswith(' := I1') for line in lines), 1)
                self.assertEqual(sum('ASSERT ' in line and ' >= -2' in line for line in lines), 1)
                self.assertTrue(any(line.startswith('LET A1:QASM2QCX_INT_0 := ') for line in lines))
                self.assertIn('LET A1:QASM2QCX_INT_0 &= 15', lines)
                self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())

    def test_nested_indices_and_rhs_reads_use_distinct_private_storage(self) -> None:
        converter = UnsignedArrayDeclarationTests.converter('array[uint[4], 2] a = {1, 3}; '
            'int i = 0; a[a[i]] += a[i] + uint[4](1);')
        lines = list(converter)
        checks = [line for line in lines if line.startswith('ASSERT ') and ' < 2' in line]
        self.assertEqual(len(checks), 3)
        self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())

    def test_reads_support_arithmetic_bitwise_shifts_power_and_comparisons(self) -> None:
        converter = UnsignedArrayDeclarationTests.converter('array[uint[4], 2] a = {15, 1}; int i = 0; '
            'uint x = (a[i] + a[1]) >> 1; x = ~a[i]; x = -a[i]; '
            'x = a[i] ** uint[4](2); bool b = a[i] > 1; float f = float(a[i]); '
            'array[uint[8], 2] copy = {a[0], a[i]};')
        self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())

    def test_invalid_indices_and_selections_are_rejected(self) -> None:
        for access in ('a[-3]', 'a[2]', 'a[true]', 'a[0.5]', 'a[0:1]', 'a[{0, 1}]', 'a[0, 1]'):
            for statement in (f'uint x = {access};', f'{access} = 1;'):
                with self.subTest(statement=statement), self.assertRaises(qasm2qcx.QASM2QCXError):
                    convert('OPENQASM 3; array[uint[4], 2] a; ' + statement)

    def test_invalid_rhs_releases_destination_and_expression_temporaries(self) -> None:
        for operator in ('=', '+=', '%=', '&=', '<<=', '**='):
            program = qasm2qcx.openqasm3.parser.parse('OPENQASM 3; array[uint[4], 2] a; int i = 0; '
                f'a[i] {operator} complex(a[i]);')
            converter = qasm2qcx.QASM2QCXConverter(program)
            with self.subTest(operator=operator), self.assertRaises(qasm2qcx.QASM2QCXError):
                converter.visit(program)
            self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())

    def test_non_native_signed_mixed_compound_assignments_remain_rejected(self) -> None:
        for operator in ('+=', '%=', '&=', '**='):
            with self.subTest(operator=operator), self.assertRaises(qasm2qcx.UnsupportedOpenQASMError):
                convert('OPENQASM 3; array[uint[4], 2] a; int[8] n = 1; int i = 0; '
                        f'a[i] {operator} n;')

    def test_source_width_controls_shift_count_guards(self) -> None:
        lines = convert('OPENQASM 3; array[uint[4], 2] a = {1, 2}; int i = 0; int count = 1; '
                        'a[i] <<= count;')
        self.assertTrue(any(line.startswith('ASSERT ') and line.endswith(' < 4') for line in lines))
        self.assertIn('LET A1:QASM2QCX_INT_0 &= 15', lines)


class UnsignedArrayLoopTests(unittest.TestCase):
    def assert_clean_conversion(self, source: str):
        converter = UnsignedArrayDeclarationTests.converter(source)
        self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())
        self.assertEqual(converter._QASM2QCXConverter__loop_bindings, [])
        self.assertEqual(converter._QASM2QCXConverter__loop_contexts, [])
        ForLoopControlLoweringTests.assert_resolved_jumps(list(converter))
        return converter

    def test_binding_width_metadata_is_required_only_for_uint(self) -> None:
        with self.assertRaises(ValueError):
            qasm2qcx._RuntimeIteratorBinding('PRIVATE_UINT', qasm2qcx.ValueType.UINT)
        with self.assertRaises(ValueError):
            qasm2qcx._RuntimeIteratorBinding('PRIVATE_INT', qasm2qcx.ValueType.INT,
                                           qasm2qcx._UnsignedIntegerType(4, True))

    def test_uint_loop_storage_and_live_read_body_are_generated_once(self) -> None:
        converter = self.assert_clean_conversion('array[uint[4], 3] values = {1, 2, 15}; uint total = 0; '
            'for uint[4] value in values { total += value; values[1] = 7; }')
        lines = converter._QASM2QCXConverter__qcx_lines
        body = lines.index('@QASM2QCX_LOOP_0_BODY')
        self.assertIn('VAR QASM2QCX_UINT_0 UINT', lines[:body])
        self.assertIn('VAR QASM2QCX_INT_0 INT', lines[:body])
        self.assertEqual(lines[body + 1], 'LET QASM2QCX_UINT_0 := VALUES63:QASM2QCX_INT_0')
        self.assertEqual(converter._QASM2QCXConverter__temporary_uint_types['QASM2QCX_UINT_0'],
                         qasm2qcx._UnsignedIntegerType(4, True))
        self.assertEqual(lines.count('LET VALUES63:1 := 7'), 1)

    def test_effective_width_compatibility_retains_iterator_sizedness(self) -> None:
        native = qasm2qcx.QASM2QCXConverter.QCX_UINT_WIDTH
        for base, iterator, sized in (('uint', 'uint', False), ('uint', f'uint[{native}]', True),
                                      (f'uint[{native}]', 'uint', False), ('uint[4]', 'uint[4]', True)):
            with self.subTest(base=base, iterator=iterator):
                converter = self.assert_clean_conversion(f'array[{base}, 1] values = {{1}}; '
                    f'for {iterator} value in values {{ continue; }}')
                info = converter._QASM2QCXConverter__temporary_uint_types['QASM2QCX_UINT_0']
                self.assertEqual(info, qasm2qcx._UnsignedIntegerType(4 if base == 'uint[4]' else native, sized))

    def test_mismatched_widths_and_base_types_rejected_even_in_empty_outer_loop(self) -> None:
        for iterator in ('uint', 'uint[1]', 'uint[8]', 'int', 'float', 'bit'):
            for wrapper in ('{}', 'for int outer in [1:0] {{ {} }}'):
                with self.subTest(iterator=iterator, wrapper=wrapper), self.assertRaises(qasm2qcx.UnsupportedOpenQASMError):
                    convert('OPENQASM 3; array[uint[4], 2] values; '
                            + wrapper.format(f'for {iterator} value in values {{}}'))
        for source in ('array[int, 2] values;', 'array[float, 2] values;', 'bit[2] values;'):
            with self.subTest(source=source), self.assertRaises(qasm2qcx.UnsupportedOpenQASMError):
                convert('OPENQASM 3; ' + source + 'for uint value in values {}')

    def test_runtime_and_unrepresentable_iterator_widths_are_rejected(self) -> None:
        native = qasm2qcx.QASM2QCXConverter.QCX_UINT_WIDTH
        for width in ('n', 'ZERO', 'NEGATIVE', 'true', '4.5', str(native + 1)):
            with self.subTest(width=width), self.assertRaises(qasm2qcx.QASM2QCXError):
                convert('OPENQASM 3; const int ZERO = 0; const int NEGATIVE = -1; int n = 4; '
                        f'array[uint[4], 2] values; for uint[{width}] value in values {{}}')
        with self.assertRaises(qasm2qcx.InvalidDeclarationException):
            convert('OPENQASM 3; const int W = 4; array[uint[W], 2] values; '
                    'for int W in [1:0] { for uint[W] value in values {} }')
        self.assert_clean_conversion('const int W = 4; array[uint[W], 2] values; '
            'for int W in [0:0] { for uint[4] value in values {} }')

    def test_nested_width_shadowing_and_global_source_width_are_restored(self) -> None:
        self.assert_clean_conversion('uint[8] value = 3; array[uint[4], 2] small = {15, 1}; '
            'array[uint[8], 2] large = {255, 1}; uint total = 0; '
            'for uint[4] value in small { total += value + uint[4](1); '
            'for uint[8] value in large { total += value + uint[8](1); } '
            'total += value ** uint[4](2); } total += value;')
        self.assert_clean_conversion('array[uint[4], 2] values; uint total; '
            'for uint[4] values in values { total = ~values; } total += values[0];')

    def test_iteration_supports_expressions_bounds_indices_and_gate_parameters(self) -> None:
        self.assert_clean_conversion('include "stdgates.inc"; qubit q; array[uint[4], 2] values = {0, 1}; '
            'uint total = 0; bool flag = false; float f; complex z; '
            'for uint[4] value in values { total += (~value + value) ^ uint[4](2); '
            'total += value ** uint[4](2); f = float(value); z = complex(value); '
            'flag = bool(value) && value > 0; values[value] += uint[4](1); '
            'for int j in [0:value] { total += j; } '
            'for int j in {value, value + uint[4](1)} { total += j; } '
            'rx(value) q; gphase(value); }')

    def test_nested_float_bit_and_signed_iterators_restore_unsigned_metadata(self) -> None:
        self.assert_clean_conversion('array[uint[4], 2] values; array[float, 1] floats; bit[1] flags; uint total; '
            'for uint[4] value in values { for float value in floats { if (bool(value)) {} } '
            'for bit value in flags { if (value) {} } '
            'for int value in [0:1] { total += uint(value); } total += ~value; }')

    def test_empty_outer_loop_keeps_uint_metadata_without_evaluating_values(self) -> None:
        self.assert_clean_conversion('array[uint[4], 2] values; uint total; '
            'for int outer in [1:0] { for uint[4] value in values { '
            'total = value ** uint[4](2); '
            'for int j in {1 / int(value)} { total += j; } } }')
        for body in ('if (value) {}', 'while (value) {}', 'value = 0;',
                     'value += 1;', 'value[0] = 1;', 'for uint[4] next in value {}'):
            with self.subTest(body=body), self.assertRaises(qasm2qcx.QASM2QCXError):
                convert('OPENQASM 3; array[uint[4], 2] values; '
                        f'for int outer in [1:0] {{ for uint[4] value in values {{ {body} }} }}')

    def test_runtime_uint_binding_overrides_shadowed_declaration_width(self) -> None:
        converter = UnsignedArrayDeclarationTests.converter('uint[8] value = 1;')
        expression = qasm2qcx.ast.Identifier('value')
        info = qasm2qcx._UnsignedIntegerType(4, True)
        binding = qasm2qcx._RuntimeIteratorBinding('PRIVATE_UINT', qasm2qcx.ValueType.UINT, info)
        with converter._QASM2QCXConverter__iterator_binding('value', binding):
            operand = converter._QASM2QCXConverter__condition_operand(expression)
            self.assertEqual(operand, ('PRIVATE_UINT', qasm2qcx.ValueType.UINT, qasm2qcx.ValueKind.LVALUE))
            self.assertEqual(converter._QASM2QCXConverter__uint_expression_type(expression), info)
            self.assertEqual(converter._QASM2QCXConverter__integer_operand_info(
                expression, {'value': qasm2qcx.ValueType.UINT}), (qasm2qcx.ValueType.UINT, info, True))
        self.assertEqual(converter._QASM2QCXConverter__identifier_uint_type('value').width, 8)

    def test_failed_nested_loop_preserves_caller_storage_and_cleans_bindings(self) -> None:
        converter = UnsignedArrayDeclarationTests.converter('array[uint[4], 2] values; uint total;')
        caller = converter._QASM2QCXConverter__add_new_temporary_variable(qasm2qcx.ValueType.UINT)
        statement = qasm2qcx.openqasm3.parser.parse('OPENQASM 3; for uint[4] value in values { '
            'for uint[4] other in values { total = complex(value + other); } }').statements[0]
        with self.assertRaises(qasm2qcx.NoImplicitCastException):
            converter.visit(statement)
        self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, {caller})
        self.assertEqual(converter._QASM2QCXConverter__loop_bindings, [])
        self.assertEqual(converter._QASM2QCXConverter__loop_contexts, [])


class ComplexArrayDeclarationTests(unittest.TestCase):
    @staticmethod
    def converter(source: str):
        program = qasm2qcx.openqasm3.parser.parse('OPENQASM 3; ' + source)
        converter = qasm2qcx.QASM2QCXConverter(program)
        converter.visit(program)
        return converter

    def test_declarations_use_complex_storage_and_existing_precision_policy(self) -> None:
        for base in ('complex', 'complex[float[32]]', 'complex[float[64]]'):
            with self.subTest(base=base):
                self.assertEqual(convert(f'OPENQASM 3; array[{base}, 2] a;'),
                                 ['QUBITS 0', 'VAR A1 COMPLEX 2'])
                self.assertEqual(convert(f'OPENQASM 3; array[{base}, 1] a = {{1.0im}};'),
                                 ['QUBITS 0', 'VAR A1 COMPLEX', 'LET A1:0 := :I'])

    def test_initializer_conversions_match_scalar_complex_assignments(self) -> None:
        prefix = 'OPENQASM 3; int n = 2; uint u = 3; float f = 0.5; bool ready = true; bit b = 1; '
        for value in ('0', 'uint(2)', 'true', '0.5', '1.0im', '-1.0im', '2 + 3.0im',
                      'n', 'u', 'f', 'ready', 'b', 'complex(b)', 'complex(f)',
                      'n + 2.0im', 'const_z'):
            with self.subTest(value=value):
                source = prefix + 'const complex const_z = 2 - 3.0im; '
                scalar = convert(source + f'complex a = {value};')
                array = convert(source + f'array[complex, 1] a = {{{value}}};')
                self.assertEqual(array, [line.replace('LET A1 := ', 'LET A1:0 := ') for line in scalar])

    def test_runtime_initializers_keep_order_and_release_complex_temporaries(self) -> None:
        converter = self.converter('complex z = 1.0im; int n = 2; '
            'array[complex, 4] a = {z, z + n, -z, 2 + 3.0im};')
        assignments = [line for line in converter._QASM2QCXConverter__qcx_lines
                       if line.startswith('LET A1:')]
        self.assertEqual(len(assignments), 4)
        self.assertEqual(assignments[0], 'LET A1:0 := Z1')
        for index, assignment in enumerate(assignments):
            self.assertTrue(assignment.startswith(f'LET A1:{index} := '))
        self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())

    def test_metadata_preserves_complex_element_type_and_array_identity(self) -> None:
        converter = self.converter('array[complex, 1] a; array[complex, 3] b;')
        self.assertEqual(converter._QASM2QCXConverter__array_source_sizes, {'a': 1, 'b': 3})
        self.assertEqual(converter._QASM2QCXConverter__complex_variable_name_size_map, {'A1': 1, 'B1': 3})
        self.assertEqual(converter._QASM2QCXConverter__array_element_types,
                         {'A1': qasm2qcx.ValueType.COMPLEX, 'B1': qasm2qcx.ValueType.COMPLEX})

    def test_sizes_are_constant_in_declaration_scope(self) -> None:
        for size in ('N', 'N - 1 + 1', 'int(2.5)', 'uint(2)', 'int(true) + 1'):
            with self.subTest(size=size):
                converter = self.converter(f'const int N = 2; array[complex, {size}] a; '
                                           'for int N in [0:1] {}')
                self.assertEqual(converter._QASM2QCXConverter__array_source_sizes['a'], 2)

    def test_invalid_sizes_and_multidimensional_arrays_are_rejected(self) -> None:
        for size in ('0', 'NEGATIVE', 'n', '2.5', 'true', '1.0im', 'missing',
                     str(qasm2qcx.QASM2QCXConverter.QCX_INT_MAX + 1)):
            with self.subTest(size=size), self.assertRaises(qasm2qcx.QASM2QCXError):
                convert(f'OPENQASM 3; const int NEGATIVE = -1; int n = 2; array[complex, {size}] a;')
        with self.assertRaises(qasm2qcx.UnsupportedOpenQASMError):
            convert('OPENQASM 3; array[complex, 2, 2] a;')

    def test_initializer_requires_a_matching_flat_literal(self) -> None:
        for initializer in ('{}', '{1.0im}', '{1, 2, 3}'):
            with self.subTest(initializer=initializer), self.assertRaises(qasm2qcx.InvalidDeclarationException):
                convert(f'OPENQASM 3; array[complex, 2] a = {initializer};')
        for initializer in ('{{1}, {2}}', '1.0im', 'z'):
            with self.subTest(initializer=initializer), self.assertRaises(qasm2qcx.UnsupportedOpenQASMError):
                convert(f'OPENQASM 3; complex z; array[complex, 2] a = {initializer};')

    def test_whole_arrays_are_not_scalars_even_for_singletons(self) -> None:
        for size in (1, 2):
            for statement in ('complex z = a;', 'complex z = a + 1;', 'complex z = complex(a);',
                              'if (a) {}', 'a = 1;', 'a += 1;', 'array[complex, 1] b = {a};'):
                with self.subTest(size=size, statement=statement), self.assertRaises(qasm2qcx.QASM2QCXError):
                    convert(f'OPENQASM 3; array[complex, {size}] a; {statement}')

    def test_iteration_requires_matching_complex_type(self) -> None:
        for iterator in ('int', 'uint', 'float', 'bool', 'bit'):
            with self.subTest(iterator=iterator), self.assertRaises(qasm2qcx.UnsupportedOpenQASMError):
                convert(f'OPENQASM 3; array[complex, 2] a; for {iterator} z in a {{}}')

    def test_duplicates_forward_self_references_and_local_arrays_are_rejected(self) -> None:
        for source in ('array[complex, 1] a; complex a;', 'complex a; array[complex, 1] a;',
                       'array[complex, 1] a = {z}; complex z = 1.0im;',
                       'array[complex, N] a; const int N = 1;', 'array[complex, 1] a = {a};',
                       'if (false) { array[complex, 1] a; }', 'while (false) { array[complex, 1] a; }',
                       'for int i in [1:0] { array[complex, 1] a; }'):
            with self.subTest(source=source), self.assertRaises(
                    (qasm2qcx.QASM2QCXError, qasm2qcx.openqasm3.parser.QASM3ParsingError)):
                convert('OPENQASM 3; ' + source)

    def test_failed_initializer_releases_completed_temporaries(self) -> None:
        program = qasm2qcx.openqasm3.parser.parse('OPENQASM 3; complex z = 1.0im; '
            'array[complex, 2] a = {z + 1, bool(z)};')
        converter = qasm2qcx.QASM2QCXConverter(program)
        with self.assertRaises(qasm2qcx.QASM2QCXError):
            converter.visit(program)
        self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())


class ComplexArrayAccessTests(unittest.TestCase):
    def assert_clean_conversion(self, source: str):
        converter = ComplexArrayDeclarationTests.converter(source)
        self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())
        ForLoopControlLoweringTests.assert_resolved_jumps(converter._QASM2QCXConverter__qcx_lines)
        return converter._QASM2QCXConverter__qcx_lines

    def test_static_reads_writes_and_singleton_indexing(self) -> None:
        lines = self.assert_clean_conversion('array[complex, 1] a = {1.0im}; '
            'complex z = a[-1]; a[0] = -z;')
        self.assertIn('LET Z1 := A1:0', lines)
        self.assertTrue(any(line.startswith('LET A1:0 := QASM2QCX_COMPLEX_') for line in lines))
        lines = self.assert_clean_conversion('array[complex, 3] a; complex z = a[-3]; a[-1] = z;')
        self.assertIn('LET Z1 := A1:0', lines)
        self.assertIn('LET A1:2 := Z1', lines)

    def test_runtime_reads_use_complex_snapshot_and_private_checked_int_index(self) -> None:
        lines = self.assert_clean_conversion('array[complex, 2] a; int i = -1; complex z = a[i];')
        self.assertIn('LET QASM2QCX_INT_0 := I1', lines)
        self.assertIn('ASSERT QASM2QCX_INT_0 >= -2', lines)
        self.assertIn('ASSERT QASM2QCX_INT_0 < 2', lines)
        self.assertIn('LET QASM2QCX_INT_0 += 2', lines)
        self.assertIn('LET QASM2QCX_COMPLEX_0 := A1:QASM2QCX_INT_0', lines)
        self.assertIn('LET Z1 := QASM2QCX_COMPLEX_0', lines)

    def test_supported_assignments_promote_numeric_rhs(self) -> None:
        for operator, qcx in (('=', ':='), ('+=', '+='), ('-=', '-='), ('*=', '*='), ('/=', '/=')):
            for index in ('0', 'i'):
                with self.subTest(operator=operator, index=index):
                    lines = self.assert_clean_conversion('array[complex, 2] a; int i = 0; uint u = 2; '
                        f'a[{index}] {operator} u;')
                    self.assertTrue(any(line.startswith('LET A1:') and line.endswith(f' {qcx} :COMPLEX:U1')
                                        for line in lines))

    def test_runtime_destination_captured_once_and_reserved_during_complex_rhs(self) -> None:
        lines = self.assert_clean_conversion('array[complex, 2] a = {1.0im, 2.0im}; int i = 0; int j = 1; '
            'a[i] += a[j] * (a[i] + 2.0im);')
        self.assertEqual(lines.count('LET QASM2QCX_INT_0 := I1'), 1)
        self.assertTrue(any(line.startswith('LET A1:QASM2QCX_INT_0 += QASM2QCX_COMPLEX_') for line in lines))
        self.assertEqual(sum(line.endswith(' := I1') for line in lines), 2)
        self.assertEqual(sum(line.endswith(' := J1') for line in lines), 1)

    def test_nested_indices_casts_and_array_initializers(self) -> None:
        self.assert_clean_conversion('array[int, 2] indices = {1, 0}; int i = 0; '
            'array[complex, 2] a = {1 + 1.0im, 2 + 2.0im}; '
            'array[complex, 2] b = {a[0], a[indices[i]]}; '
            'complex z = complex(a[i]); uint u = uint[4](a[i]); float f = float(a[i]); '
            'complex other = a[uint(a[i])] + b[i];')

    def test_remainder_power_bitwise_and_shift_assignments_rejected_in_empty_loops(self) -> None:
        for operator in ('%=', '**=', '&=', '|=', '^=', '<<=', '>>='):
            for index in ('0', 'i'):
                for wrapper in ('{}', 'for int outer in [1:0] {{ {} }}'):
                    with self.subTest(operator=operator, index=index, wrapper=wrapper), self.assertRaises(
                            qasm2qcx.UnsupportedOpenQASMError):
                        convert('OPENQASM 3; int i = 0; array[complex, 2] a; '
                                + wrapper.format(f'a[{index}] {operator} 1;'))

    def test_unsupported_scalar_complex_expressions_remain_rejected(self) -> None:
        for expression in ('a[i] % 2', 'a[i] ** 2', 'a[i] & 1', 'a[i] | 1', 'a[i] ^ 1',
                           'a[i] << 1', 'a[i] >> 1', '~a[i]', '!a[i]', 'bool(a[i])', 'bit(a[i])',
                           'a[i] && true', 'false || a[i]', 'a[i] == a[0]', 'a[i] != 1.0im', 'a[i] < 1'):
            for wrapper in ('{}', 'for int outer in [1:0] {{ {} }}'):
                with self.subTest(expression=expression, wrapper=wrapper), self.assertRaises(qasm2qcx.QASM2QCXError):
                    convert('OPENQASM 3; array[complex, 2] a; int i = 0; complex z; '
                            + wrapper.format(f'z = {expression};'))

    def test_complex_comparisons_conditions_and_gate_parameters_rejected(self) -> None:
        for statement in ('if (a[i]) {}', 'while (a[i]) {}',
                          'if (a[i] == a[0]) {}', 'if (a[i] != 1.0im) {}', 'if (a[i] < 1) {}',
                          'rx(a[i]) q;', 'gphase(a[i]);', 'a[i] = measure q;'):
            with self.subTest(statement=statement), self.assertRaises(qasm2qcx.QASM2QCXError):
                convert('OPENQASM 3; include "stdgates.inc"; qubit q; array[complex, 2] a; int i = 0; '
                        + statement)

    def test_invalid_indices_and_selections_rejected_for_reads_and_writes(self) -> None:
        for access in ('a[-3]', 'a[2]', 'a[true]', 'a[0.5]', 'a[1.0im]', 'a[a[0]]',
                       'a[0:1]', 'a[{0, 1}]', 'a[0, 1]', 'a[0][0]'):
            for statement in (f'complex z = {access};', f'{access} = 1.0im;'):
                with self.subTest(statement=statement), self.assertRaises(qasm2qcx.QASM2QCXError):
                    convert('OPENQASM 3; array[complex, 2] a; ' + statement)

    def test_implicit_narrowing_assignments_rejected(self) -> None:
        for target in ('int', 'uint', 'float', 'bool', 'bit'):
            for index in ('0', 'i'):
                with self.subTest(target=target, index=index), self.assertRaises(qasm2qcx.QASM2QCXError):
                    convert(f'OPENQASM 3; array[complex, 2] a; int i = 0; {target} v = a[{index}];')

    def test_failed_rhs_releases_destination_and_complex_temporaries(self) -> None:
        for rhs in ('a[i] + missing', 'bool(a[i] + 1)', 'a[i] % 2'):
            program = qasm2qcx.openqasm3.parser.parse('OPENQASM 3; array[complex, 2] a; int i = 0; '
                f'a[i] = {rhs};')
            converter = qasm2qcx.QASM2QCXConverter(program)
            with self.subTest(rhs=rhs), self.assertRaises(qasm2qcx.QASM2QCXError):
                converter.visit(program)
            self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())


class ComplexArrayLoopTests(unittest.TestCase):
    def assert_clean_conversion(self, source: str):
        converter = ComplexArrayDeclarationTests.converter(source)
        self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())
        self.assertEqual(converter._QASM2QCXConverter__loop_bindings, [])
        self.assertEqual(converter._QASM2QCXConverter__loop_contexts, [])
        ForLoopControlLoweringTests.assert_resolved_jumps(converter._QASM2QCXConverter__qcx_lines)
        return converter._QASM2QCXConverter__qcx_lines

    def test_complex_iterator_uses_complex_storage_and_body_emitted_once(self) -> None:
        lines = self.assert_clean_conversion('array[complex, 2] a = {1.0im, 2.0im}; complex total = 0; '
            'for complex z in a { total += z; a[1] += 1; }')
        body = lines.index('@QASM2QCX_LOOP_0_BODY')
        self.assertIn('VAR QASM2QCX_COMPLEX_0 COMPLEX', lines[:body])
        self.assertIn('VAR QASM2QCX_INT_0 INT', lines[:body])
        self.assertEqual(lines[body + 1], 'LET QASM2QCX_COMPLEX_0 := A1:QASM2QCX_INT_0')
        self.assertEqual(lines.count('LET TOTAL31 += QASM2QCX_COMPLEX_0'), 1)

    def test_singleton_has_explicit_index_and_endpoint_guard(self) -> None:
        lines = self.assert_clean_conversion('array[complex, 1] a = {1.0im}; for complex z in a { continue; }')
        self.assertIn('LET QASM2QCX_COMPLEX_0 := A1:QASM2QCX_INT_0', lines)
        self.assertIn('JUMPIF QASM2QCX_LOOP_0_END QASM2QCX_INT_0 == 0', lines)

    def test_matching_precision_policy_reuses_scalar_native_complex(self) -> None:
        for base in ('complex', 'complex[float[32]]', 'complex[float[64]]'):
            for iterator in ('complex', 'complex[float[32]]', 'complex[float[64]]'):
                with self.subTest(base=base, iterator=iterator):
                    self.assert_clean_conversion(f'array[{base}, 1] a = {{1.0im}}; complex total = 0; '
                        f'for {iterator} z in a {{ total += z; }}')

    def test_shadowing_source_and_mixed_type_iterators_restores_bindings(self) -> None:
        self.assert_clean_conversion('array[complex, 2] a; array[uint[4], 1] u; array[float, 1] f; '
            'array[bool, 1] flags; bit[1] bits; complex total = 0; complex z = 1.0im; '
            'for complex a in a { total += a; } total += a[0]; '
            'for complex z in a { for int z in [0:1] { total += z; } '
            'for uint[4] z in u { total += z; } for float z in f { total += z; } '
            'for bool z in flags { total += z; } for bit z in bits { total += complex(z); } total += z; } '
            'total += z;')

    def test_casts_and_nested_integer_ranges_keep_runtime_metadata_in_empty_loops(self) -> None:
        self.assert_clean_conversion('array[complex, 2] a; complex total; '
            'for int outer in [1:0] { for complex z in a { '
            'for int j in [0:uint(z)] { total += z; } '
            'for int j in {1 / uint(z)} { total += j; } } }')
        converter = ComplexArrayDeclarationTests.converter('array[complex, 1] a;')
        expression = qasm2qcx.ast.Identifier('z')
        self.assertEqual(converter._QASM2QCXConverter__loop_expression_info(
            expression, {'z': qasm2qcx.ValueType.COMPLEX}), (qasm2qcx.ValueType.COMPLEX, True))

    def test_readonly_iterators_rejected_including_empty_outer_loops(self) -> None:
        for body in ('z = 1;', 'z += 1;', 'z[0] = 1;', 'z = measure q;',
                     'if (false) { z = 1; }', 'while (false) { z = 1; }'):
            for wrapper in ('{}', 'for int outer in [1:0] {{ {} }}'):
                with self.subTest(body=body, wrapper=wrapper), self.assertRaises(qasm2qcx.QASM2QCXError):
                    convert('OPENQASM 3; qubit q; array[complex, 2] a; '
                            + wrapper.format(f'for complex z in a {{ {body} }}'))

    def test_complex_type_restrictions_remain_in_nested_and_empty_loops(self) -> None:
        for body in ('if (z) {}', 'if (z == 1) {}', 'total = !z;', 'total = bool(z);',
                     'total = z % 2;', 'total = z ** 2;', 'total = z & 1;', 'total = z << 1;',
                     'for int j in [0:z] {}', 'for int j in {z} {}'):
            for wrapper in ('{}', 'for int outer in [1:0] {{ {} }}'):
                with self.subTest(body=body, wrapper=wrapper), self.assertRaises(qasm2qcx.QASM2QCXError):
                    convert('OPENQASM 3; array[complex, 2] a; complex total; '
                            + wrapper.format(f'for complex z in a {{ {body} }}'))

    def test_unsupported_sources_rejected_even_in_empty_outer_loops(self) -> None:
        for declaration in ('array[int, 2] a;', 'array[uint, 2] a;', 'array[float, 2] a;',
                            'array[bool, 2] a;', 'bit[2] a;', 'complex a;'):
            with self.subTest(declaration=declaration), self.assertRaises(qasm2qcx.UnsupportedOpenQASMError):
                convert('OPENQASM 3; ' + declaration + 'for int outer in [1:0] { for complex z in a {} }')
        for source in ('[0:1]', '{1.0im, 2.0im}', 'a[0:1]', 'a[0]'):
            with self.subTest(source=source), self.assertRaises(qasm2qcx.UnsupportedOpenQASMError):
                convert(f'OPENQASM 3; array[complex, 2] a; for complex z in {source} {{}}')

    def test_failed_body_preserves_caller_storage_and_releases_loop_temporaries(self) -> None:
        converter = ComplexArrayDeclarationTests.converter('array[complex, 2] a; complex total;')
        caller = converter._QASM2QCXConverter__add_new_temporary_variable(qasm2qcx.ValueType.COMPLEX)
        statement = qasm2qcx.openqasm3.parser.parse('OPENQASM 3; for complex z in a { '
            'for complex other in a { total = (z + other) + missing; } }').statements[0]
        with self.assertRaises(qasm2qcx.QASM2QCXError):
            converter.visit(statement)
        self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, {caller})
        self.assertEqual(converter._QASM2QCXConverter__loop_bindings, [])
        self.assertEqual(converter._QASM2QCXConverter__loop_contexts, [])


class FloatArrayDeclarationTests(unittest.TestCase):
    def test_declarations_use_real_storage_including_singletons_and_sized_floats(self) -> None:
        for base in ('float', 'float[32]', 'float[64]'):
            with self.subTest(base=base):
                self.assertEqual(convert(f'OPENQASM 3; array[{base}, 2] a;'),
                                 ['QUBITS 0', 'VAR A1 REAL 2'])
                self.assertEqual(convert(f'OPENQASM 3; array[{base}, 1] a = {{0.5}};'),
                                 ['QUBITS 0', 'VAR A1 REAL', 'LET A1:0 := 0.5'])

    def test_literal_initializers_follow_scalar_float_promotion(self) -> None:
        self.assertEqual(convert('OPENQASM 3; array[float, 5] a = {1, uint(2), 0.5, -0.25, true};'),
                         ['QUBITS 0', 'VAR A1 REAL 5',
                          'LET A1:0 := 1.0', 'LET A1:1 := 2.0', 'LET A1:2 := 0.5',
                          'LET A1:3 := -0.25', 'LET A1:4 := 1.0'])

    def test_runtime_initializers_preserve_order_and_release_temporaries(self) -> None:
        program = qasm2qcx.openqasm3.parser.parse(
            'OPENQASM 3; int n = 2; uint u = 3; float f = 0.25; bool b = true; '
            'array[float, 6] a = {n, u, f, f + 0.5, b, float(n)};')
        converter = qasm2qcx.QASM2QCXConverter(program)
        converter.visit(program)
        lines = converter._QASM2QCXConverter__qcx_lines
        assignments = [line for line in lines if line.startswith('LET A1:')]
        self.assertEqual(len(assignments), 6)
        for index, assignment in enumerate(assignments):
            self.assertTrue(assignment.startswith(f'LET A1:{index} := '))
        self.assertEqual(assignments[:3], ['LET A1:0 := :REAL:N1',
                                           'LET A1:1 := :REAL:U1', 'LET A1:2 := F1'])
        self.assertEqual(assignments[4], 'LET A1:4 := :REAL:B1')
        self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())

    def test_array_metadata_retains_element_types_and_source_sizes(self) -> None:
        program = qasm2qcx.openqasm3.parser.parse(
            'OPENQASM 3; array[int, 2] a; array[float, 3] b;')
        converter = qasm2qcx.QASM2QCXConverter(program)
        converter.visit(program)
        self.assertEqual(converter._QASM2QCXConverter__array_source_sizes, {'a': 2, 'b': 3})
        self.assertEqual(converter._QASM2QCXConverter__array_element_types,
                         {'A1': qasm2qcx.ValueType.INT, 'B1': qasm2qcx.ValueType.FLOAT})

    def test_constant_dimensions_are_evaluated_in_declaration_scope(self) -> None:
        for size in ('N', 'N + 1 - 1', 'int(2.5)', 'uint(2)'):
            with self.subTest(size=size):
                lines = convert(f'OPENQASM 3; const int N = 2; array[float, {size}] a; '
                                'for int N in [0:0] {}')
                self.assertIn('VAR A1 REAL 2', lines)

    def test_invalid_dimensions_are_rejected(self) -> None:
        for size in ('0', 'NEGATIVE', 'n', 'n + 1', '2.5', 'true',
                     str(qasm2qcx.QASM2QCXConverter.QCX_INT_MAX + 1)):
            with self.subTest(size=size), self.assertRaises(qasm2qcx.InvalidDeclarationException):
                convert(f'OPENQASM 3; const int NEGATIVE = -1; int n = 2; array[float, {size}] a;')
        with self.assertRaises(qasm2qcx.UnsupportedOpenQASMError):
            convert('OPENQASM 3; array[float, 2, 2] a;')

    def test_initializer_must_be_a_matching_flat_literal(self) -> None:
        for initializer in ('{}', '{1}', '{1, 2, 3}'):
            with self.subTest(initializer=initializer), self.assertRaises(qasm2qcx.InvalidDeclarationException):
                convert(f'OPENQASM 3; array[float, 2] a = {initializer};')
        for initializer in ('{{1}, {2}}', '2', 'other'):
            with self.subTest(initializer=initializer), self.assertRaises(qasm2qcx.UnsupportedOpenQASMError):
                convert(f'OPENQASM 3; float other = 2; array[float, 2] a = {initializer};')

    def test_complex_initializers_require_explicit_conversion(self) -> None:
        for initializer in ('1.0im', 'z', 'z + 1.0im'):
            with self.subTest(initializer=initializer):
                program = qasm2qcx.openqasm3.parser.parse(
                    f'OPENQASM 3; complex z = 1.0im; array[float, 1] a = {{{initializer}}};')
                converter = qasm2qcx.QASM2QCXConverter(program)
                with self.assertRaises(qasm2qcx.NoImplicitCastException):
                    converter.visit(program)
                self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())
        self.assertIn('LET A1:0 := 2.0', convert(
            'OPENQASM 3; array[float, 1] a = {float(2.0 + 1.0im)};'))

    def test_whole_arrays_cannot_be_used_as_scalars(self) -> None:
        for size in (1, 2):
            for statement in ('float x = a;', 'float x = a + 1;', 'float x = float(a);',
                              'if (a) {}', 'a = 1;', 'a += 1;', 'array[float, 1] b = {a};'):
                with self.subTest(size=size, statement=statement), self.assertRaises(qasm2qcx.QASM2QCXError):
                    convert(f'OPENQASM 3; array[float, {size}] a; {statement}')

    def test_float_array_iteration_requires_a_float_iterator(self) -> None:
        for statement in ('for int x in a {}', 'for bit x in a {}'):
            with self.subTest(statement=statement), self.assertRaises(qasm2qcx.UnsupportedOpenQASMError):
                convert('OPENQASM 3; array[float, 2] a; ' + statement)

    def test_duplicate_forward_self_references_and_local_arrays_are_rejected(self) -> None:
        for source in ('array[float, 1] a; float a;', 'float a; array[float, 1] a;',
                       'array[float, 1] a = {seed}; float seed = 1;',
                       'array[float, N] a; const int N = 1;',
                       'array[float, 1] a = {a};',
                       'if (false) { array[float, 1] a; }',
                       'while (false) { array[float, 1] a; }'):
            with self.subTest(source=source), self.assertRaises(
                    (qasm2qcx.QASM2QCXError, qasm2qcx.openqasm3.parser.QASM3ParsingError)):
                convert('OPENQASM 3; ' + source)


class FloatArrayAccessTests(unittest.TestCase):
    def test_static_reads_and_writes_normalize_negative_indices(self) -> None:
        lines = convert('OPENQASM 3; const int LAST = -1; '
                        'array[float, 3] a = {0.25, 0.5, 0.75}; '
                        'float x = a[LAST]; a[-3] = x; a[1 + 1] += 0.25;')
        self.assertIn('LET X1 := A1:2', lines)
        self.assertIn('LET A1:0 := X1', lines)
        self.assertIn('LET A1:2 += 0.25', lines)

    def test_singleton_access_keeps_explicit_index(self) -> None:
        lines = convert('OPENQASM 3; array[float, 1] a = {0.25}; '
                        'float x = a[-1]; a[0] *= 2;')
        self.assertIn('VAR A1 REAL', lines)
        self.assertIn('LET X1 := A1:0', lines)
        self.assertIn('LET A1:0 *= 2.0', lines)

    def test_compound_assignments_use_real_target_and_scalar_promotion(self) -> None:
        for operator, qcx_operator in (('=', ':='), ('+=', '+='), ('-=', '-='),
                                       ('*=', '*='), ('/=', '/=')):
            with self.subTest(operator=operator):
                lines = convert(f'OPENQASM 3; uint u = 2; array[float, 2] a; a[0] {operator} u;')
                self.assertIn(f'LET A1:0 {qcx_operator} :REAL:U1', lines)

    def test_runtime_read_uses_real_snapshot_and_private_integer_index(self) -> None:
        program = qasm2qcx.openqasm3.parser.parse(
            'OPENQASM 3; array[float, 2] a = {0.25, 0.75}; int i = -1; float x = a[i];')
        converter = qasm2qcx.QASM2QCXConverter(program)
        converter.visit(program)
        lines = converter._QASM2QCXConverter__qcx_lines
        self.assertIn('LET QASM2QCX_INT_0 := I1', lines)
        self.assertIn('ASSERT QASM2QCX_INT_0 >= -2', lines)
        self.assertIn('ASSERT QASM2QCX_INT_0 < 2', lines)
        self.assertIn('LET QASM2QCX_INT_0 += 2', lines)
        self.assertIn('VAR QASM2QCX_REAL_0 REAL', lines)
        self.assertIn('LET QASM2QCX_REAL_0 := A1:QASM2QCX_INT_0', lines)
        self.assertIn('LET X1 := QASM2QCX_REAL_0', lines)
        self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())

    def test_runtime_destination_index_is_reserved_during_rhs(self) -> None:
        program = qasm2qcx.openqasm3.parser.parse(
            'OPENQASM 3; array[float, 2] a = {0.25, 0.75}; int i = 0; int j = 1; '
            'a[i] += a[j] * 2;')
        converter = qasm2qcx.QASM2QCXConverter(program)
        converter.visit(program)
        lines = converter._QASM2QCXConverter__qcx_lines
        self.assertIn('LET QASM2QCX_INT_0 := I1', lines)
        self.assertIn('LET QASM2QCX_INT_1 := J1', lines)
        self.assertTrue(any(line.startswith('LET A1:QASM2QCX_INT_0 += QASM2QCX_REAL_')
                            for line in lines))
        self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())

    def test_nested_integer_array_indices_and_explicit_index_casts(self) -> None:
        for index in ('indices[i]', 'int(f)', 'uint(1)', 'int(true)'):
            with self.subTest(index=index):
                lines = convert('OPENQASM 3; array[int, 2] indices = {0, 1}; int i = 1; '
                                'float f = 1.5; array[float, 2] a = {0.25, 0.75}; '
                                f'float x = a[{index}]; a[{index}] = x;')
                self.assertTrue(any(line.startswith('LET X1 := ') for line in lines))

    def test_float_element_can_initialize_another_array(self) -> None:
        lines = convert('OPENQASM 3; array[float, 2] a = {0.25, 0.75}; int i = 1; '
                        'array[float, 2] b = {a[0], a[i]};')
        self.assertIn('LET B1:0 := A1:0', lines)
        self.assertTrue(any(line.startswith('LET B1:1 := QASM2QCX_REAL_') for line in lines))

    def test_invalid_indices_and_selections_are_rejected_for_reads_and_writes(self) -> None:
        for access in ('a[-3]', 'a[2]', 'a[true]', 'a[0.5]', 'a[f]', 'a[0:1]',
                       'a[{0, 1}]', 'a[0, 1]', 'a[0][0]'):
            for statement in (f'float x = {access};', f'{access} = 0.5;'):
                with self.subTest(statement=statement), self.assertRaises(qasm2qcx.QASM2QCXError):
                    convert('OPENQASM 3; float f = 1; array[float, 2] a; ' + statement)

    def test_integer_only_compound_operators_are_rejected(self) -> None:
        for operator in ('%=', '**=', '&=', '|=', '^=', '<<=', '>>='):
            for index in ('0', 'i'):
                with self.subTest(operator=operator, index=index), self.assertRaises(
                        qasm2qcx.UnsupportedOpenQASMError):
                    convert(f'OPENQASM 3; int i = 0; array[float, 2] a; a[{index}] {operator} 1;')

    def test_float_reads_are_not_implicitly_narrowed_to_integers(self) -> None:
        for index in ('0', 'i'):
            with self.subTest(index=index), self.assertRaises(qasm2qcx.NoImplicitCastException):
                convert(f'OPENQASM 3; int i = 0; array[float, 2] a; int x = a[{index}];')

    def test_runtime_index_is_released_after_invalid_rhs(self) -> None:
        for rhs in ('1.0im', 'a[i] + 1.0im'):
            with self.subTest(rhs=rhs):
                program = qasm2qcx.openqasm3.parser.parse(
                    f'OPENQASM 3; int i = 0; array[float, 2] a; a[i] = {rhs};')
                converter = qasm2qcx.QASM2QCXConverter(program)
                with self.assertRaises(qasm2qcx.NoImplicitCastException):
                    converter.visit(program)
                self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())


class FloatArrayIntegrationTests(unittest.TestCase):
    def assert_converts_without_temporary_leaks(self, source: str) -> list[str]:
        program = qasm2qcx.openqasm3.parser.parse('OPENQASM 3; ' + source)
        converter = qasm2qcx.QASM2QCXConverter(program)
        converter.visit(program)
        self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())
        lines = converter._QASM2QCXConverter__qcx_lines
        ForLoopControlLoweringTests.assert_resolved_jumps(lines)
        return lines

    def test_arithmetic_unary_negation_and_numeric_casts(self) -> None:
        self.assert_converts_without_temporary_leaks(
            'array[float, 2] a = {0.5, 1.5}; int i = 1; '
            'float x = -a[i] + a[0] * 2 - a[i] / 2; '
            'int n = int(a[i]); uint u = uint[4](a[i]); '
            'complex z = complex(a[i]) + 1.0im; float f = float(a[0]);')

    def test_all_comparisons_with_integer_unsigned_and_float_operands(self) -> None:
        for operator in ('==', '!=', '<', '<=', '>', '>='):
            for rhs in ('1', 'u', 'a[0]'):
                with self.subTest(operator=operator, rhs=rhs):
                    self.assert_converts_without_temporary_leaks(
                        'array[float, 2] a = {0.5, 1.5}; int i = 1; uint u = 1; '
                        f'bool b = a[i] {operator} {rhs}; '
                        f'if (a[i] {operator} {rhs}) {{ a[0] += 0.25; }} else {{ a[0] -= 0.25; }}')

    def test_boolean_casts_and_short_circuit_conditions(self) -> None:
        self.assert_converts_without_temporary_leaks(
            'array[float, 2] a = {0, 0.5}; int i = 1; int bad = 2; '
            'bool b = bool(a[i]); b = !bool(a[0]); '
            'b = false && a[bad] > 0; b = true || bool(a[bad]); '
            'if (bool(a[i]) && a[0] == 0) { a[i] += 0.25; }')

    def test_elements_work_in_constant_runtime_and_while_loops(self) -> None:
        self.assert_converts_without_temporary_leaks(
            'array[float, 3] a = {0.25, 0.5, 0.75}; int stop = 2; '
            'for int i in [0:2] { a[i] = -a[i]; } '
            'for int i in [0:stop] { if (a[i] < 0) { a[i] *= -1; } } '
            'int j = 0; while (a[j] < 1) { a[j] += 0.25; '
            'if (a[j] == 1) { break; } continue; } '
            'for int i in {0, stop} { a[i] += 0.25; }')

    def test_explicit_integer_casts_work_in_runtime_loop_bounds(self) -> None:
        self.assert_converts_without_temporary_leaks(
            'array[float, 2] a = {0, 2}; int i = 1; int total = 0; '
            'for int j in [int(a[0]):int(a[i])] { total += j; } '
            'for int j in {int(a[0]), int(a[i])} { total += j; }')

    def test_gate_parameters_and_global_phase_keep_real_elements(self) -> None:
        lines = self.assert_converts_without_temporary_leaks(
            'include "stdgates.inc"; qubit q; int i = 1; array[float, 2] a = {0, pi}; '
            'rx(a[i]) q; ry(a[0] + a[i] / 2) q; rz(-a[i]) q; '
            'U(a[0], a[i], a[i] / 2) q; gphase(a[i]);')
        self.assertTrue(any(line.startswith('EX ') for line in lines))

    def test_bare_float_conditions_and_implicit_integer_loop_bounds_are_rejected(self) -> None:
        for statement in ('if (a[i]) {}', 'while (a[i]) {}',
                          'for int j in [0:a[i]] {}', 'for int j in {a[i]} {}'):
            with self.subTest(statement=statement), self.assertRaises(qasm2qcx.QASM2QCXError):
                convert('OPENQASM 3; int i = 0; array[float, 2] a; ' + statement)

    def test_integer_only_operators_rejected_even_in_empty_loops(self) -> None:
        for expression in ('~a[i]', 'a[i] & 1', 'a[i] << 1', 'a[i] ** 2', 'a[i] % 2'):
            with self.subTest(expression=expression), self.assertRaises(qasm2qcx.QASM2QCXError):
                convert('OPENQASM 3; array[float, 2] a; float x; '
                        f'for int i in [1:0] {{ x = {expression}; }}')
        for operator in ('%=', '**=', '&=', '|=', '^=', '<<=', '>>='):
            with self.subTest(operator=operator), self.assertRaises(qasm2qcx.QASM2QCXError):
                convert('OPENQASM 3; array[float, 2] a; '
                        f'for int i in [1:0] {{ a[i] {operator} 1; }}')

    def test_iterator_shadowing_rejects_indexing_and_restores_array_after_loop(self) -> None:
        lines = self.assert_converts_without_temporary_leaks(
            'array[float, 2] a = {0.25, 0.5}; float x = 0; int stop = 1; '
            'for int a in [0:stop] { x += float(a); } x += a[0];')
        self.assertIn('LET X1 += A1:0', lines)
        for bounds in ('[0:1]', '[1:0]'):
            with self.subTest(bounds=bounds), self.assertRaises(qasm2qcx.QASM2QCXError):
                convert('OPENQASM 3; array[float, 2] a; float x; '
                        f'for int a in {bounds} {{ x = -a[0]; }}')


class FloatArrayLoopValidationTests(unittest.TestCase):
    @staticmethod
    def converter(source: str = 'array[float, 2] values;'):
        program = qasm2qcx.openqasm3.parser.parse('OPENQASM 3; ' + source)
        converter = qasm2qcx.QASM2QCXConverter(program)
        converter.visit(program)
        return converter

    @staticmethod
    def expression(source: str):
        return qasm2qcx.openqasm3.parser.parse(
            f'OPENQASM 3; float result = {source};').statements[0].init_expression

    def test_collection_metadata_preserves_float_type_and_dimension(self) -> None:
        for base in ('float', 'float[32]', 'float[64]'):
            for iterator in ('float', 'float[32]', 'float[64]'):
                with self.subTest(base=base, iterator=iterator):
                    converter = self.converter(f'const int N = 3; array[{base}, N] values;')
                    statement = qasm2qcx.openqasm3.parser.parse(
                        f'OPENQASM 3; for {iterator} value in values {{}}').statements[0]
                    converter._QASM2QCXConverter__validate_loop_header(statement)
                    self.assertEqual(converter._QASM2QCXConverter__collection_loop_info(
                        statement.set_declaration, statement.type),
                        ('VALUES63', 3, qasm2qcx.ValueType.FLOAT, None))

    def test_initialization_pass_validates_without_emitting_loop_storage(self) -> None:
        program = qasm2qcx.openqasm3.parser.parse('OPENQASM 3; array[float, 1] values; '
            'float result; for float value in values { result = -value; }')
        converter = qasm2qcx.QASM2QCXConverter(program)
        self.assertEqual(converter._QASM2QCXConverter__qcx_lines, ['QUBITS 0'])
        self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())
        self.assertEqual(converter._QASM2QCXConverter__loop_bindings, [])
        self.assertEqual(converter._QASM2QCXConverter__loop_contexts, [])

    def test_typed_placeholders_preserve_runtime_dependencies(self) -> None:
        converter = self.converter('int value = 0;')
        placeholders = {'value': qasm2qcx.ValueType.FLOAT}
        for expression, expected in (
                ('value', qasm2qcx.ValueType.FLOAT), ('-value', qasm2qcx.ValueType.FLOAT),
                ('value + 1', qasm2qcx.ValueType.FLOAT), ('int(value)', qasm2qcx.ValueType.INT),
                ('bool(value)', qasm2qcx.ValueType.BOOL), ('value > 0', qasm2qcx.ValueType.BOOL)):
            with self.subTest(expression=expression):
                self.assertEqual(converter._QASM2QCXConverter__loop_expression_info(
                    self.expression(expression), placeholders), (expected, True))
        self.assertEqual(converter._QASM2QCXConverter__loop_expression_info(
            self.expression('value'), {'value': qasm2qcx.ValueType.INT}),
            (qasm2qcx.ValueType.INT, False))

    def test_valid_float_loops_inside_empty_integer_loop_need_no_runtime_lowering(self) -> None:
        lines = convert('OPENQASM 3; array[float, 2] values; float result; '
            'for int outer in [1:0] { for float value in values { '
            'result = -value; if (value > 0) { continue; } break; } }')
        self.assertEqual(lines, ['QUBITS 0', 'VAR VALUES63 REAL 2', 'VAR RESULT63 REAL'])

    def test_integer_casts_in_nested_bounds_do_not_fabricate_float_values(self) -> None:
        convert('OPENQASM 3; array[float, 2] values; int total; '
            'for int outer in [1:0] { for float value in values { '
            'for int j in [0:int(value)] { total += j; } '
            'for int j in {1 / int(value)} { total += j; } } }')
        for bounds in ('[0:value]', '[0:value:2]', '{value}'):
            with self.subTest(bounds=bounds), self.assertRaises(qasm2qcx.InvalidLoopRangeException):
                convert('OPENQASM 3; array[float, 2] values; '
                    'for int outer in [1:0] { for float value in values { '
                    f'for int j in {bounds} {{}} }} }}')

    def test_mismatched_collection_and_iterator_types_are_rejected(self) -> None:
        declarations = 'array[float, 2] floats; array[int, 2] integers; bit[2] flags; '
        for iterator, source in (('int', 'floats'), ('bit', 'floats'),
                                 ('float', 'integers'), ('float', 'flags'),
                                 ('uint', 'floats'), ('bool', 'floats')):
            for wrapper in ('{}', 'for int outer in [1:0] {{ {} }}'):
                with self.subTest(iterator=iterator, source=source, wrapper=wrapper), self.assertRaises(
                        qasm2qcx.UnsupportedOpenQASMError):
                    convert('OPENQASM 3; ' + declarations
                            + wrapper.format(f'for {iterator} value in {source} {{}}'))

    def test_float_ranges_sets_slices_scalars_and_unknown_sources_are_rejected(self) -> None:
        for source in ('[0:2]', '{0.5, 1.5}', 'values[0:1]', 'values[0]', 'scalar', 'missing'):
            with self.subTest(source=source), self.assertRaises(qasm2qcx.QASM2QCXError):
                convert('OPENQASM 3; array[float, 2] values; float scalar; '
                    f'for int outer in [1:0] {{ for float value in {source} {{}} }}')

    def test_float_iterator_is_read_only_even_in_empty_enclosing_loop(self) -> None:
        for body in ('value = 1;', 'value += 1;', 'value[0] = 1;',
                     'while (false) { value = 1; }', 'value = measure q;'):
            with self.subTest(body=body), self.assertRaises(qasm2qcx.UnsupportedOpenQASMError):
                convert('OPENQASM 3; qubit q; array[float, 2] values; '
                    f'for int outer in [1:0] {{ for float value in values {{ {body} }} }}')

    def test_float_iterator_rejects_integer_only_operations_in_empty_enclosing_loop(self) -> None:
        for expression in ('~value', 'value & 1', 'value << 1', 'value ** 2', 'value % 2'):
            with self.subTest(expression=expression), self.assertRaises(qasm2qcx.UnsupportedOpenQASMError):
                convert('OPENQASM 3; array[float, 2] values; float result; '
                    'for int outer in [1:0] { for float value in values { '
                    f'result = {expression}; }} }}')

    def test_float_conditions_require_explicit_conversion_even_in_empty_enclosing_loop(self) -> None:
        prefix = 'OPENQASM 3; array[float, 2] values; '
        for condition in ('value', '-value', 'value + 1', '!value', 'value && true'):
            for statement in ('if (' + condition + ') {}', 'while (' + condition + ') {}'):
                with self.subTest(statement=statement), self.assertRaises(qasm2qcx.QASM2QCXError):
                    convert(prefix + 'for int outer in [1:0] { for float value in values { '
                            + statement + ' } }')
        for condition in ('bool(value)', 'value > 0', '!bool(value)', 'bool(value) && value > 0',
                          'bool(value) && bool(1 / int(value))'):
            with self.subTest(condition=condition):
                convert(prefix + 'for int outer in [1:0] { for float value in values { '
                        f'if ({condition}) {{}} while ({condition}) {{ break; }} }} }}')

    def test_typed_shadowing_restores_float_validation_after_inner_integer_loop(self) -> None:
        prefix = 'OPENQASM 3; array[float, 2] values; float result; '
        convert(prefix + 'for int outer in [1:0] { for float value in values { '
            'for int value in [0:1] { result = ~value; } result = -value; } }')
        with self.assertRaises(qasm2qcx.UnsupportedOpenQASMError):
            convert(prefix + 'for int outer in [1:0] { for float value in values { '
                'for int value in [0:1] {} result = ~value; } }')

    def test_scalar_iterator_shadowing_a_collection_cannot_be_iterated_or_indexed(self) -> None:
        for statement in ('for float next in values {}', 'result = -values[0];'):
            with self.subTest(statement=statement), self.assertRaises(qasm2qcx.UnsupportedOpenQASMError):
                convert('OPENQASM 3; array[float, 2] values; float result; '
                    'for int outer in [1:0] { for float values in values { '
                    + statement + ' } }')


class FloatArrayLoopLoweringTests(unittest.TestCase):
    @staticmethod
    def converter(source: str):
        program = qasm2qcx.openqasm3.parser.parse('OPENQASM 3; ' + source)
        converter = qasm2qcx.QASM2QCXConverter(program)
        converter.visit(program)
        return converter

    def test_loop_uses_real_iterator_and_integer_index_and_emits_body_once(self) -> None:
        lines = convert('OPENQASM 3; array[float, 3] values = {0.25, 0.5, 0.75}; '
            'float total = 0; for float value in values { total += value; values[1] = 1.5; }')
        body = lines.index('@QASM2QCX_LOOP_0_BODY')
        self.assertIn('VAR QASM2QCX_REAL_0 REAL', lines[:body])
        self.assertIn('VAR QASM2QCX_INT_0 INT', lines[:body])
        self.assertEqual(lines[body - 1], 'LET QASM2QCX_INT_0 := 0')
        self.assertEqual(lines[body + 1], 'LET QASM2QCX_REAL_0 := VALUES63:QASM2QCX_INT_0')
        self.assertEqual(lines.count('LET TOTAL31 += QASM2QCX_REAL_0'), 1)
        self.assertEqual(lines.count('LET VALUES63:1 := 1.5'), 1)
        self.assertIn('JUMPIF QASM2QCX_LOOP_0_END QASM2QCX_INT_0 == 2', lines)
        ForLoopControlLoweringTests.assert_resolved_jumps(lines)

    def test_singleton_retains_indexed_array_identity_and_endpoint_guard(self) -> None:
        lines = convert('OPENQASM 3; array[float, 1] values = {0.5}; '
            'for float value in values { continue; }')
        self.assertIn('VAR VALUES63 REAL', lines)
        self.assertIn('LET QASM2QCX_REAL_0 := VALUES63:QASM2QCX_INT_0', lines)
        next_label = lines.index('@QASM2QCX_LOOP_0_NEXT')
        self.assertEqual(lines[next_label + 1], 'JUMPIF QASM2QCX_LOOP_0_END QASM2QCX_INT_0 == 0')
        self.assertIn('LET QASM2QCX_INT_0 += 1', lines[next_label + 2:])

    def test_sized_float_forms_use_native_real_iterator_storage(self) -> None:
        for base in ('float', 'float[32]', 'float[64]'):
            for iterator in ('float', 'float[32]', 'float[64]'):
                with self.subTest(base=base, iterator=iterator):
                    lines = convert(f'OPENQASM 3; array[{base}, 1] values = {{0.5}}; '
                        f'float total; for {iterator} value in values {{ total += value; }}')
                    self.assertIn('VAR QASM2QCX_REAL_0 REAL', lines)
                    self.assertIn('LET TOTAL31 += QASM2QCX_REAL_0', lines)

    def test_nested_loops_reserve_separate_iterators_and_indices(self) -> None:
        converter = self.converter('array[float, 2] values; float total; '
            'for float value in values { for float other in values { '
            'total += value * other; } total += value; }')
        lines = list(converter)
        self.assertIn('LET QASM2QCX_REAL_0 := VALUES63:QASM2QCX_INT_0', lines)
        self.assertIn('LET QASM2QCX_REAL_1 := VALUES63:QASM2QCX_INT_1', lines)
        self.assertIn('LET TOTAL31 += QASM2QCX_REAL_0', lines)
        self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())
        self.assertEqual(converter._QASM2QCXConverter__loop_bindings, [])
        self.assertEqual(converter._QASM2QCXConverter__loop_contexts, [])
        ForLoopControlLoweringTests.assert_resolved_jumps(lines)

    def test_body_computations_do_not_overwrite_reserved_iterator(self) -> None:
        lines = convert('OPENQASM 3; array[float, 2] values = {0.25, 0.5}; int i = 0; '
            'float total; for float value in values { values[i] = -value; '
            'total += value / 2 + values[i]; i += 1; }')
        body = lines.index('@QASM2QCX_LOOP_0_BODY')
        end = lines.index('@QASM2QCX_LOOP_0_NEXT')
        targets = [line.split()[1] for line in lines[body + 2:end] if line.startswith('LET ')]
        self.assertNotIn('QASM2QCX_REAL_0', targets)
        self.assertNotIn('QASM2QCX_INT_0', targets)

    def test_break_continue_use_collection_loop_labels(self) -> None:
        lines = convert('OPENQASM 3; array[float, 2] values; float total; '
            'for float value in values { if (value < 0) { continue; } '
            'if (value > 1) { break; } total += value; }')
        self.assertIn('JUMP QASM2QCX_LOOP_0_NEXT', lines)
        self.assertIn('JUMP QASM2QCX_LOOP_0_END', lines)
        ForLoopControlLoweringTests.assert_resolved_jumps(lines)

    def test_iterator_can_shadow_source_array_and_scalar_and_restore_them(self) -> None:
        converter = self.converter('array[float, 2] values; int value = 2; float total; '
            'for float values in values { total += values; } total += values[0]; '
            'for float value in values { total += value; } value += 1;')
        lines = list(converter)
        self.assertIn('LET TOTAL31 += VALUES63:0', lines)
        self.assertIn('LET VALUE31 += 1', lines)
        self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())

    def test_temporary_declarations_are_hoisted_but_computations_remain_after_continue(self) -> None:
        lines = convert('OPENQASM 3; array[float, 2] values; float total; '
            'for float value in values { continue; total += value * 2; } total += 1;')
        body = lines.index('@QASM2QCX_LOOP_0_BODY')
        jump = lines.index('JUMP QASM2QCX_LOOP_0_NEXT')
        self.assertTrue(all(index < body for index, line in enumerate(lines)
                            if line.startswith('VAR QASM2QCX_')))
        self.assertTrue(any(index > jump and ' *= ' in line for index, line in enumerate(lines)))

    def test_failed_body_releases_iterator_index_and_loop_context(self) -> None:
        program = qasm2qcx.openqasm3.parser.parse('OPENQASM 3; array[float, 2] values; float total; '
            'for float value in values { total = missing; }')
        converter = qasm2qcx.QASM2QCXConverter(program)
        with self.assertRaises(qasm2qcx.NoVariableNameException):
            converter.visit(program)
        self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())
        self.assertEqual(converter._QASM2QCXConverter__loop_bindings, [])
        self.assertEqual(converter._QASM2QCXConverter__loop_contexts, [])

    def test_runtime_collection_size_does_not_expand_output(self) -> None:
        lengths = [len(convert(f'OPENQASM 3; array[float, {size}] values; float total; '
            'for float value in values { total += value; }'))
            for size in (1, 3, qasm2qcx.QASM2QCXConverter.QCX_INT_MAX)]
        self.assertEqual(lengths, [lengths[0]] * 3)


class FloatArrayLoopIntegrationTests(unittest.TestCase):
    def assert_clean_conversion(self, source: str) -> list[str]:
        converter = FloatArrayLoopLoweringTests.converter(source)
        self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())
        self.assertEqual(converter._QASM2QCXConverter__loop_bindings, [])
        self.assertEqual(converter._QASM2QCXConverter__loop_contexts, [])
        lines = list(converter)
        ForLoopControlLoweringTests.assert_resolved_jumps(lines)
        return lines

    def test_iterator_arithmetic_comparisons_casts_and_gate_arguments(self) -> None:
        self.assert_clean_conversion('include "stdgates.inc"; qubit q; '
            'array[float, 2] values = {0, pi}; float total = 0; complex z; '
            'int n; uint u; bool flag; for float value in values { '
            'total += -value + value / 2; z = complex(value) + 1.0im; '
            'n = int(value); u = uint[4](value); flag = bool(value); '
            'if (value > 0 && flag) { total += value; } '
            'rx(value) q; ry(-value) q; rz(value / 2) q; '
            'U(value, value / 2, -value) q; gphase(value); }')

    def test_mixed_typed_loops_and_nested_runtime_bounds(self) -> None:
        self.assert_clean_conversion('array[float, 2] values = {1, 2}; '
            'array[int, 2] counts = {1, 2}; bit[2] flags = "01"; '
            'float total = 0; int stop = 1; '
            'for int outer in [0:stop] { for float value in values { '
            'for int n in counts { total += value * n; } '
            'for bit flag in flags { if (flag) { total += value; } } '
            'for int n in [0:int(value)] { total += n; } '
            'for int n in {int(value), outer} { if (n == 0) { continue; } total += n; } '
            'while (total > 100) { break; } } }')

    def test_nested_shadowing_between_float_bit_and_integer_iterators(self) -> None:
        self.assert_clean_conversion('array[float, 2] values = {0.25, 0.5}; '
            'bit[2] flags = "01"; int value = 7; float total = 0; '
            'for float value in values { for bit value in flags { '
            'if (value) { total += int(value); } } '
            'for int value in [0:1] { total += ~value; } total += value; } value += 1;')

    def test_short_circuiting_and_skipped_float_loops_preserve_control_flow(self) -> None:
        self.assert_clean_conversion('array[float, 2] values = {0, 1}; int bad = 2; '
            'bool flag = false; float total = 0; '
            'if (false) { for float value in values { total += values[bad]; } } '
            'while (false) { for float value in values { total += value; } } '
            'for float value in values { flag = bool(value) || values[bad] > 0; '
            'if (true) { continue; } total += values[bad]; }')

    def test_float_iterator_can_supply_explicitly_cast_integer_index(self) -> None:
        self.assert_clean_conversion('array[float, 2] positions = {0, 1}; '
            'array[float, 2] values = {0.25, 0.75}; float total = 0; '
            'for float position in positions { values[int(position)] += 0.25; '
            'total += values[int(position)]; }')

    def test_read_only_iterator_and_integer_operator_rejections(self) -> None:
        for body in ('value = 0;', 'value += 1;', 'if (false) { value = 1; }',
                     'total = value % 2;', 'total = value ** 2;', 'if (value) {}',
                     'total = values[value];', 'for int n in [0:value] {}'):
            with self.subTest(body=body), self.assertRaises(qasm2qcx.QASM2QCXError):
                convert('OPENQASM 3; array[float, 2] values; float total; '
                        f'for float value in values {{ {body} }}')

    def test_failed_loop_releases_body_temporaries_but_preserves_caller_storage(self) -> None:
        for body in ('total = complex(value);', 'total = value + 1.0im;',
                     'total = missing + (value + 1);',
                     'for float other in values { total = complex(other); }'):
            with self.subTest(body=body):
                converter = FloatArrayLoopLoweringTests.converter(
                    'array[float, 2] values; float total;')
                caller = converter._QASM2QCXConverter__add_new_temporary_variable(qasm2qcx.ValueType.FLOAT)
                statement = qasm2qcx.openqasm3.parser.parse(
                    f'OPENQASM 3; for float value in values {{ {body} }}').statements[0]
                with self.assertRaises(qasm2qcx.QASM2QCXError):
                    converter.visit(statement)
                self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, {caller})
                self.assertEqual(converter._QASM2QCXConverter__loop_bindings, [])
                self.assertEqual(converter._QASM2QCXConverter__loop_contexts, [])

    def test_cli_rejection_does_not_print_partial_output(self) -> None:
        source = 'OPENQASM 3; array[float, 2] values; for float value in values { value = 0; }'
        with patch('builtins.open', mock_open(read_data=source)), patch('builtins.print') as output:
            with self.assertRaisesRegex(SystemExit, 'assignment to a for-loop iterator'):
                qasm2qcx.main(['float-loop.qasm'])
            output.assert_not_called()


class IntegerArrayLoopLoweringTests(unittest.TestCase):
    def test_loop_loads_live_element_and_generates_body_once(self) -> None:
        lines = convert('OPENQASM 3.0; array[int, 3] values = {1, 3, 5}; int total = 0; '
                        'for int value in values { total += value; values[1] = 10; }')
        body = lines.index('@QASM2QCX_LOOP_0_BODY')
        self.assertEqual(lines[body - 1], 'LET QASM2QCX_INT_1 := 0')
        self.assertEqual(lines[body + 1], 'LET QASM2QCX_INT_0 := VALUES63:QASM2QCX_INT_1')
        self.assertEqual(lines.count('LET TOTAL31 += QASM2QCX_INT_0'), 1)
        self.assertIn('JUMPIF QASM2QCX_LOOP_0_END QASM2QCX_INT_1 == 2', lines)
        self.assertIn('LET QASM2QCX_INT_1 += 1', lines)
        ForLoopControlLoweringTests.assert_resolved_jumps(lines)

    def test_static_access_supports_constants_arithmetic_negative_indices_and_compound_assignments(self) -> None:
        lines = convert('OPENQASM 3.0; const int LAST = -1; array[int, 3] a = {1, 3, 5}; '
                        'int value = a[LAST]; a[1 + 1] += 2; a[-3] %= 2;')
        self.assertIn('LET VALUE31 := A1:2', lines)
        self.assertIn('LET A1:2 += 2', lines)
        self.assertTrue(any(line.startswith('LET A1:0 := QASM2QCX_INT_') for line in lines))

    def test_singleton_retains_array_identity_and_checks_endpoint_before_advancement(self) -> None:
        lines = convert('OPENQASM 3.0; array[int, 1] a = {2}; for int i in a { continue; }')
        self.assertIn('LET QASM2QCX_INT_0 := A1:QASM2QCX_INT_1', lines)
        next_label = lines.index('@QASM2QCX_LOOP_0_NEXT')
        self.assertEqual(lines[next_label + 1], 'JUMPIF QASM2QCX_LOOP_0_END QASM2QCX_INT_1 == 0')
        self.assertIn('JUMP QASM2QCX_LOOP_0_NEXT', lines)

    def test_indices_reject_out_of_bounds_slices_and_wrong_types(self) -> None:
        for access in ('a[-4]', 'a[3]', 'a[true]', 'a[1.5]', 'a[0:1]', 'a[{0, 1}]', 'a[0, 1]'):
            for statement in (f'int value = {access};', f'{access} = 2;'):
                with self.subTest(statement=statement), self.assertRaises(qasm2qcx.QASM2QCXError):
                    convert('OPENQASM 3.0; int n = 0; array[int, 3] a = {1, 2, 3}; ' + statement)

    def test_iterators_restore_scope_and_array_source_is_resolved_before_shadowing(self) -> None:
        lines = convert('OPENQASM 3.0; array[int, 2] values = {1, 2}; int total = 0; '
                        'for int values in values { total += values; } total += values[0];')
        self.assertIn('LET TOTAL31 += QASM2QCX_INT_0', lines)
        self.assertIn('LET TOTAL31 += VALUES63:0', lines)
        for body in ('values[0] = 3;', 'for int j in values {}'):
            with self.subTest(body=body), self.assertRaises(qasm2qcx.UnsupportedOpenQASMError):
                convert('OPENQASM 3.0; array[int, 2] values = {1, 2}; '
                        'for int values in values { ' + body + ' }')
        with self.assertRaises(qasm2qcx.NoVariableNameException):
            convert('OPENQASM 3.0; array[int, 2] a; for int i in a {} int value = i;')

    def test_array_elements_can_supply_runtime_range_bounds_steps_and_set_elements(self) -> None:
        lines = convert('OPENQASM 3.0; array[int, 3] a = {0, 2, 4}; '
                        'for int i in [a[0]:a[1]:a[2]] {} for int j in {a[0], a[1]} {}')
        self.assertTrue(any(line.startswith('ASSERT ') for line in lines))
        self.assertIn('LET QASM2QCX_INT_2 := A1:1', lines)
        ForLoopControlLoweringTests.assert_resolved_jumps(lines)

    def test_nested_array_loops_transfer_to_nearest_loop(self) -> None:
        lines = convert('OPENQASM 3.0; array[int, 2] a = {1, 2}; '
                        'for int i in a { for int j in a { continue; break; } continue; break; }')
        for target in ('QASM2QCX_LOOP_0_NEXT', 'QASM2QCX_LOOP_0_END',
                       'QASM2QCX_LOOP_1_NEXT', 'QASM2QCX_LOOP_1_END'):
            self.assertIn('JUMP ' + target, lines)
        ForLoopControlLoweringTests.assert_resolved_jumps(lines)

    def test_storage_lifetime_and_runtime_iteration_budget(self) -> None:
        program = qasm2qcx.openqasm3.parser.parse('OPENQASM 3.0; array[int, 2] a = {1, 2}; '
            'int total = 0; for int i in a { total += i % 3; for int j in a { continue; } }')
        converter = qasm2qcx.QASM2QCXConverter(program)
        converter.visit(program)
        self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())
        self.assertEqual(converter._QASM2QCXConverter__loop_bindings, [])
        self.assertEqual(converter._QASM2QCXConverter__loop_contexts, [])
        with patch.object(qasm2qcx.QASM2QCXConverter, 'MAX_LOOP_ITERATIONS', 0):
            convert('OPENQASM 3.0; array[int, 1000000] a; for int i in a {}')

    def test_unsupported_array_sources_and_active_iterator_writes_fail(self) -> None:
        for source in ('for int i in missing {}', 'for int i in a {} array[int, 2] a;',
                       'array[int, 2] a; for uint i in a {}', 'bit[2] a; for int i in a {}',
                       'array[int, 2] a; for int i in a { i += 1; }',
                       'array[int, 2] a; for int i in a { int local; }'):
            with self.subTest(source=source), self.assertRaises(qasm2qcx.QASM2QCXError):
                convert('OPENQASM 3.0; ' + source)
        prefix = 'OPENQASM 3.0; array[int, 2] a = {1, 2}; '
        self.assertEqual(convert(prefix + 'for int i in [1:0] { for int j in a {} }'), convert(prefix))


class RuntimeIntegerArrayReadTests(unittest.TestCase):
    def test_runtime_index_is_captured_checked_normalized_and_loaded(self) -> None:
        lines = convert('OPENQASM 3.0; array[int, 3] a = {2, 4, 6}; '
                        'int i = -1; int value = a[i];')
        start = lines.index('LET QASM2QCX_INT_0 := I1')
        self.assertEqual(lines[start:start + 7], [
            'LET QASM2QCX_INT_0 := I1', 'ASSERT QASM2QCX_INT_0 >= -3',
            'ASSERT QASM2QCX_INT_0 < 3',
            'JUMPIF QASM2QCX_ARRAY_INDEX_0_NONNEGATIVE QASM2QCX_INT_0 >= 0',
            'LET QASM2QCX_INT_0 += 3', '@QASM2QCX_ARRAY_INDEX_0_NONNEGATIVE',
            'VAR QASM2QCX_INT_1 INT'])
        self.assertIn('LET QASM2QCX_INT_1 := A1:QASM2QCX_INT_0', lines)
        self.assertIn('LET VALUE31 := QASM2QCX_INT_1', lines)
        self.assertNotIn('LET I1 += 3', lines)

    def test_static_reads_keep_existing_output_without_assertions(self) -> None:
        self.assertEqual(convert('OPENQASM 3.0; const int I = -1; '
                                 'array[int, 2] a = {1, 2}; int value = a[I];'),
                         ['QUBITS 0', 'VAR A1 INT 2', 'LET A1:0 := 1',
                          'LET A1:1 := 2', 'VAR VALUE31 INT', 'LET VALUE31 := A1:1'])

    def test_nested_indices_and_arithmetic_use_distinct_labels(self) -> None:
        lines = convert('OPENQASM 3.0; array[int, 2] a = {0, 1}; int i = 0; '
                        'int value = a[a[i]] + a[i + 1];')
        self.assertEqual(sum(line.startswith('ASSERT ') for line in lines), 6)
        labels = [line for line in lines if line.startswith('@QASM2QCX_ARRAY_INDEX_')]
        self.assertEqual(len(labels), 3)
        self.assertEqual(len(set(labels)), 3)
        ForLoopControlLoweringTests.assert_resolved_jumps(lines)

    def test_runtime_index_arithmetic_is_evaluated_once(self) -> None:
        lines = convert('OPENQASM 3.0; array[int, 2] a = {1, 2}; int i = 1; '
                        'int value = a[(i % 2)];')
        self.assertEqual(sum(' /= 2' in line for line in lines), 1)
        self.assertEqual(sum(line.startswith('ASSERT ') for line in lines), 2)
        self.assertEqual(sum(' := A1:QASM2QCX_' in line for line in lines), 1)

    def test_runtime_index_type_requires_integer_or_explicit_cast(self) -> None:
        prefix = 'OPENQASM 3.0; array[int, 2] a = {1, 2}; '
        for declaration, expression in (('float f = 0.5;', 'f'), ('bool b = false;', 'b'),
                                        ('bit b = 0;', 'b')):
            with self.subTest(expression=expression):
                with self.assertRaises(qasm2qcx.InvalidArrayOperandException):
                    convert(prefix + declaration + f'int value = a[{expression}];')
                convert(prefix + declaration + f'int value = a[int({expression})];')
        convert(prefix + 'uint i = 0; int value = a[i];')

    def test_runtime_iterator_reads_preserve_scope_and_hoist_declarations(self) -> None:
        lines = convert('OPENQASM 3.0; array[int, 2] a = {0, 1}; int total = 0; '
                        'for int i in a { total += a[i]; }')
        body = lines.index('@QASM2QCX_LOOP_0_BODY')
        self.assertTrue(all(position < body for position, line in enumerate(lines)
                            if line.startswith('VAR QASM2QCX_')))
        ForLoopControlLoweringTests.assert_resolved_jumps(lines)

    def test_reads_release_storage_and_restore_expression_context(self) -> None:
        program = qasm2qcx.openqasm3.parser.parse('OPENQASM 3.0; array[int, 2] a = {0, 1}; '
                                               'int i = 0; int value = a[a[i]] % a[i + 1];')
        converter = qasm2qcx.QASM2QCXConverter(program)
        converter.visit(program)
        self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())
        self.assertIsNone(converter._QASM2QCXConverter__expression_kind)


class RuntimeIntegerArrayAssignmentTests(unittest.TestCase):
    def test_all_assignment_operators_capture_and_check_destination_once(self) -> None:
        for operation in ('=', '+=', '-=', '*=', '/=', '%='):
            with self.subTest(operation=operation):
                lines = convert('OPENQASM 3.0; array[int, 2] a = {1, 2}; int i = 0; '
                                f'a[i] {operation} 1;')
                self.assertEqual(lines.count('LET QASM2QCX_INT_0 := I1'), 1)
                self.assertEqual(sum(line.startswith('ASSERT ') for line in lines), 2)
                self.assertTrue(any(line.startswith('LET A1:QASM2QCX_INT_0 ') for line in lines))

    def test_destination_storage_is_reserved_while_rhs_is_evaluated(self) -> None:
        lines = convert('OPENQASM 3.0; array[int, 3] a = {0, 1, 2}; int i = 1; '
                        'a[i] += a[a[i]] % (a[i + 1] + 1);')
        normalized = lines.index('@QASM2QCX_ARRAY_INDEX_0_NONNEGATIVE')
        assignments = [line.split()[1] for line in lines[normalized + 1:] if line.startswith('LET ')]
        self.assertNotIn('QASM2QCX_INT_0', assignments)
        self.assertEqual(assignments[-1], 'A1:QASM2QCX_INT_0')
        ForLoopControlLoweringTests.assert_resolved_jumps(lines)

    def test_remainder_assignment_does_not_recompute_index_arithmetic(self) -> None:
        lines = convert('OPENQASM 3.0; array[int, 2] a = {7, 3}; int i = 1; '
                        'a[i % 2] %= a[i];')
        self.assertEqual(sum(' /= 2' in line for line in lines), 1)
        self.assertEqual(sum(line.startswith('ASSERT ') for line in lines), 4)
        self.assertEqual(sum(line.startswith('@QASM2QCX_ARRAY_INDEX_') for line in lines), 2)

    def test_assignment_releases_destination_and_expression_storage(self) -> None:
        for operation in ('=', '+=', '-=', '*=', '/=', '%='):
            program = qasm2qcx.openqasm3.parser.parse('OPENQASM 3.0; array[int, 2] a = {7, 3}; '
                f'int i = -1; a[i] {operation} a[i] + 1;')
            converter = qasm2qcx.QASM2QCXConverter(program)
            converter.visit(program)
            self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())
            self.assertIsNone(converter._QASM2QCXConverter__expression_kind)

    def test_failed_rhs_releases_destination_storage(self) -> None:
        for rhs in ('missing', 'unknown(1)'):
            program = qasm2qcx.openqasm3.parser.parse('OPENQASM 3.0; array[int, 2] a = {7, 3}; '
                                                   f'int i = 0; a[i] = {rhs};')
            converter = qasm2qcx.QASM2QCXConverter(program)
            with self.assertRaises(qasm2qcx.QASM2QCXError):
                converter.visit(program)
            self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())
            self.assertIsNone(converter._QASM2QCXConverter__expression_kind)

    def test_remainder_rejects_noninteger_rhs_and_releases_its_temporary(self) -> None:
        program = qasm2qcx.openqasm3.parser.parse('OPENQASM 3.0; array[int, 2] a = {7, 3}; '
            'int i = 0; float f = 1.5; a[i] %= f + 1.0;')
        converter = qasm2qcx.QASM2QCXConverter(program)
        with self.assertRaises(qasm2qcx.UnsupportedOpenQASMError):
            converter.visit(program)
        self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())
        self.assertIsNone(converter._QASM2QCXConverter__expression_kind)


class RuntimeIntegerArrayRegressionTests(unittest.TestCase):
    def test_normalization_guard_with_checked_native_integer_arithmetic(self) -> None:
        minimum, maximum = qasm2qcx.QASM2QCXConverter.QCX_INT_MIN, qasm2qcx.QASM2QCXConverter.QCX_INT_MAX
        for size in (1, 2, 7, maximum):
            lines = convert(f'OPENQASM 3.0; array[int, {size}] a; int i = 0; int value = a[i];')
            start = lines.index('ASSERT QASM2QCX_INT_0 >= ' + str(-size))
            guard = lines[start:start + 5]
            for original in {minimum, -size - 1, -size, -1, 0, size - 1, size, maximum}:
                with self.subTest(size=size, original=original):
                    value, valid = original, True
                    for line in guard:
                        words = line.split()
                        if words[0] == 'ASSERT':
                            bound = int(words[3])
                            valid = value >= bound if words[2] == '>=' else value < bound
                            if not valid:
                                break
                        elif words[0] == 'JUMPIF' and value >= int(words[4]):
                            break
                        elif words[0] == 'LET':
                            value += int(words[3])
                            self.assertTrue(minimum <= value <= maximum)
                    self.assertEqual(valid, -size <= original < size)
                    if valid:
                        self.assertEqual(value, original % size)

    def test_source_temporary_names_are_not_reused_for_index_storage(self) -> None:
        lines = convert('OPENQASM 3.0; int QASM2QCX_INT_ = -1; '
                        'array[int, 2] a = {1, 2}; a[QASM2QCX_INT_] += a[QASM2QCX_INT_];')
        self.assertIn('LET QASM2QCX_INT_1 := QASM2QCX_INT_0', lines)
        self.assertNotIn('LET QASM2QCX_INT_0 += 2', lines)

    def test_mixed_nested_control_flow_has_unique_resolved_labels(self) -> None:
        lines = convert('OPENQASM 3.0; array[int, 2] a = {0, 1}; int i = 0; '
                        'for int k in {0, 1} { while (i < 2) { '
                        'if (a[i] > 0) { a[i] %= a[i] + 1; } i += 1; } }')
        labels = [line for line in lines if line.startswith('@')]
        self.assertEqual(len(labels), len(set(labels)))
        ForLoopControlLoweringTests.assert_resolved_jumps(lines)

    def test_successful_mixed_index_operations_release_all_storage(self) -> None:
        program = qasm2qcx.openqasm3.parser.parse('OPENQASM 3.0; array[int, 3] a = {0, 1, 2}; '
            'int i = 1; bool ok = a[a[i]] == a[i]; '
            'a[a[i]] += int(ok && a[i] > 0); a[i] %= a[i] % 3 + 1;')
        converter = qasm2qcx.QASM2QCXConverter(program)
        converter.visit(program)
        self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())
        self.assertEqual(converter._QASM2QCXConverter__loop_bindings, [])
        self.assertEqual(converter._QASM2QCXConverter__loop_contexts, [])

    def test_skipped_accesses_still_validate_index_types_and_operators(self) -> None:
        prefix = 'OPENQASM 3.0; array[int, 2] a = {0, 1}; float f = 0.5; int i = 0; '
        for statement in ('if (false) { a[f] = 1; }', 'while (false) { i = a[f]; }',
                          'if (false) { a[i] <<= true; }', 'bool ok = false && a[f] == 0;'):
            with self.subTest(statement=statement), self.assertRaises(qasm2qcx.QASM2QCXError):
                convert(prefix + statement)

    def test_dynamic_quantum_indices_and_array_slices_remain_unsupported(self) -> None:
        prefix = 'OPENQASM 3.0; include "stdgates.inc"; qubit[2] q; bit[2] flags; '
        prefix += 'array[int, 2] a = {0, 1}; int i = 0; '
        for statement in ('x q[i];',
                          'i = a[i:1];', 'a[i:1] = 1;', 'i = a[i, 0];'):
            with self.subTest(statement=statement), self.assertRaises(qasm2qcx.QASM2QCXError):
                convert(prefix + statement)

    def test_cli_invalid_index_type_does_not_print_partial_output(self) -> None:
        source = 'OPENQASM 3.0; array[int, 2] a = {0, 1}; float f = 0.5; int value = a[f];'
        with patch('builtins.open', mock_open(read_data=source)), patch('builtins.print') as output:
            with self.assertRaises(SystemExit) as error:
                qasm2qcx.main(['runtime-array.qasm'])
        self.assertIn('index must be an integer', str(error.exception))
        output.assert_not_called()


class IntegerArrayRegressionTests(unittest.TestCase):
    def test_array_size_is_fixed_before_dimension_constant_is_shadowed(self) -> None:
        for outer in ('for int N in values', 'for int N in {0, 3}'):
            with self.subTest(outer=outer):
                lines = convert('OPENQASM 3.0; const int N = 2; '
                                'array[int, N] values = {1, 2}; int total = 0; '
                                + outer + ' { for int j in values { total += j; } }')
                self.assertIn('VAR VALUES63 INT 2', lines)
                guards = [line for line in lines if line.startswith('JUMPIF ')
                          and '_END ' in line]
                self.assertTrue(guards)
                self.assertTrue(all(line.endswith(' == 1') for line in guards))

    def test_codegen_size_and_body_count_do_not_depend_on_array_length(self) -> None:
        source = ('OPENQASM 3.0; array[int, {size}] values; int total = 0; '
                  'for int value in values {{ total += value; }}')
        reference = convert(source.format(size=1))
        for size in (3, qasm2qcx.QASM2QCXConverter.QCX_INT_MAX):
            lines = convert(source.format(size=size))
            self.assertEqual(len(lines), len(reference))
            self.assertEqual(sum(line.startswith('LET TOTAL31 +=') for line in lines), 1)
            self.assertIn(f'JUMPIF QASM2QCX_LOOP_0_END QASM2QCX_INT_1 == {size - 1}', lines)

    def test_live_iterator_and_index_storage_are_not_reused_by_nested_arithmetic(self) -> None:
        lines = convert('OPENQASM 3.0; array[int, 2] a = {1, 2}; int total = 0; '
                        'for int i in a { total += i % 3; '
                        'for int j in a { total += j % 3; continue; } }')
        body = lines.index('@QASM2QCX_LOOP_0_BODY')
        next_label = lines.index('@QASM2QCX_LOOP_0_NEXT')
        assignments = [line.split()[1] for line in lines[body + 2:next_label] if line.startswith('LET ')]
        self.assertNotIn('QASM2QCX_INT_0', assignments)
        self.assertNotIn('QASM2QCX_INT_1', assignments)
        self.assertTrue(all(index < body for index, line in enumerate(lines)
                            if line.startswith('VAR QASM2QCX_')))
        ForLoopControlLoweringTests.assert_resolved_jumps(lines)

    def test_reserved_source_array_storage_is_not_reused_by_loop_temporaries(self) -> None:
        lines = convert('OPENQASM 3.0; array[int, 2] QASM2QCX_INT_ = {1, 2}; '
                        'for int i in QASM2QCX_INT_ {}')
        self.assertIn('VAR QASM2QCX_INT_0 INT 2', lines)
        self.assertFalse(any(line.startswith('LET QASM2QCX_INT_0 ') for line in lines))
        self.assertIn('LET QASM2QCX_INT_1 := QASM2QCX_INT_0:QASM2QCX_INT_2', lines)

    def test_failure_inside_array_body_releases_storage_and_contexts(self) -> None:
        program = qasm2qcx.openqasm3.parser.parse('OPENQASM 3.0; array[int, 2] a = {1, 2}; '
                                               'for int i in a { a[3] = 1; }')
        converter = qasm2qcx.QASM2QCXConverter(program)
        with self.assertRaises(qasm2qcx.InvalidArrayOperandException):
            converter.visit(program)
        self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())
        self.assertEqual(converter._QASM2QCXConverter__loop_bindings, [])
        self.assertEqual(converter._QASM2QCXConverter__loop_contexts, [])

    def test_constant_outer_and_runtime_array_iterator_indices_are_supported(self) -> None:
        convert('OPENQASM 3.0; array[int, 2] a = {1, 2}; '
                'for int k in {0, 1} { for int value in a { a[k] += value; } }')
        lines = convert('OPENQASM 3.0; array[int, 2] a = {0, 1}; '
                        'for int k in a { a[k] = 1; }')
        self.assertEqual(sum(line.startswith('ASSERT ') for line in lines), 2)

    def test_skipped_bodies_retain_read_only_and_operator_validation(self) -> None:
        prefix = 'OPENQASM 3.0; array[int, 2] a = {1, 2}; int total = 0; qubit q; '
        for body in ('continue; i += 1;', 'break; total += 2 ** true;',
                     'if (false) { i = measure q; }', 'while (false) { int local; }'):
            with self.subTest(body=body), self.assertRaises(qasm2qcx.UnsupportedOpenQASMError):
                convert(prefix + 'for int i in a { ' + body + ' }')

    def test_array_runtime_iterations_do_not_charge_expansion_budget_but_nested_copies_do(self) -> None:
        with patch.object(qasm2qcx.QASM2QCXConverter, 'MAX_LOOP_ITERATIONS', 0):
            convert('OPENQASM 3.0; array[int, 1000000] a; for int i in a {}')
            with self.assertRaises(qasm2qcx.InvalidLoopRangeException):
                convert('OPENQASM 3.0; array[int, 2] a; '
                        'for int i in a { for int j in {i} {} }')
        with patch.object(qasm2qcx.QASM2QCXConverter, 'MAX_LOOP_OUTPUT_LINES', 10):
            with self.assertRaises(qasm2qcx.InvalidLoopRangeException):
                convert('OPENQASM 3.0; array[int, 2] a; '
                        'for int k in {0} { for int i in a {} }')

    def test_invalid_index_cli_reports_error_without_partial_output(self) -> None:
        source = 'OPENQASM 3.0; array[int, 2] a = {1, 2}; for int i in a { a[3] = i; }'
        with patch('builtins.open', mock_open(read_data=source)), patch('builtins.print') as output:
            with self.assertRaises(SystemExit) as error:
                qasm2qcx.main(['array.qasm'])
        self.assertTrue(str(error.exception).startswith('qasm2qcx.py:'))
        output.assert_not_called()


class BitIteratorInfrastructureTests(unittest.TestCase):
    @staticmethod
    def converter():
        program = qasm2qcx.openqasm3.parser.parse('OPENQASM 3.0; int value = 9; '
            'bit out; bool ready; bit[1] singleton; bit[3] flags; qubit q;')
        converter = qasm2qcx.QASM2QCXConverter(program)
        converter.visit(program)
        return converter

    @staticmethod
    def bit_binding():
        return qasm2qcx._RuntimeIteratorBinding('PRIVATE_BIT', qasm2qcx.ValueType.BIT)

    def test_bit_identifier_preserves_type_and_runtime_dependency(self) -> None:
        converter = self.converter()
        with converter._QASM2QCXConverter__iterator_binding('value', self.bit_binding()):
            self.assertEqual(RuntimeIteratorInfrastructureTests.identifier_value(converter, 'value'),
                ('PRIVATE_BIT', qasm2qcx.ValueType.BIT, qasm2qcx.ValueKind.LVALUE))
            self.assertEqual(converter._QASM2QCXConverter__loop_expression_info(qasm2qcx.ast.Identifier('value')),
                             (qasm2qcx.ValueType.BIT, True))
        self.assertEqual(RuntimeIteratorInfrastructureTests.identifier_value(converter, 'value')[1],
                         qasm2qcx.ValueType.INT)

    def test_direct_bit_conditions_and_scalar_assignments_use_private_storage(self) -> None:
        converter = self.converter()
        statements = qasm2qcx.openqasm3.parser.parse('OPENQASM 3.0; '
            'if (value) { out = value; } ready = value; out = !value;').statements
        with converter._QASM2QCXConverter__iterator_binding('value', self.bit_binding()):
            for statement in statements:
                converter.visit(statement)
        lines = list(converter)
        self.assertTrue(any(line.startswith('JUMPIF ') and ' PRIVATE_BIT \\= 0' in line for line in lines))
        self.assertIn('LET OUT7 := PRIVATE_BIT', lines)
        self.assertIn('LET READY31 := PRIVATE_BIT', lines)
        self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())

    def test_supported_bit_casts_and_logical_metadata(self) -> None:
        converter = self.converter()
        with converter._QASM2QCXConverter__iterator_binding('value', self.bit_binding()):
            for target, expected in (('int', qasm2qcx.ValueType.INT), ('uint', qasm2qcx.ValueType.UINT),
                                     ('float', qasm2qcx.ValueType.FLOAT), ('complex', qasm2qcx.ValueType.COMPLEX),
                                     ('bool', qasm2qcx.ValueType.BOOL), ('bit', qasm2qcx.ValueType.BIT)):
                expression = qasm2qcx.openqasm3.parser.parse(
                    f'OPENQASM 3.0; {target} x = {target}(value);').statements[0].init_expression
                self.assertEqual(converter._QASM2QCXConverter__loop_expression_info(expression), (expected, True))
                result, result_type, kind = converter._QASM2QCXConverter__condition_operand(expression)
                self.assertEqual(result_type, expected)
                if kind == qasm2qcx.ValueKind.TEMPORARY:
                    converter._QASM2QCXConverter__release_temporary_variable(result)
            expression = qasm2qcx.openqasm3.parser.parse(
                'OPENQASM 3.0; bool x = !value || value == true;').statements[0].init_expression
            self.assertEqual(converter._QASM2QCXConverter__loop_expression_info(expression),
                             (qasm2qcx.ValueType.BOOL, True))
        self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())

    def test_typed_placeholder_validation_preserves_shadowing(self) -> None:
        converter = self.converter()
        expression = qasm2qcx.ast.Identifier('value')
        with converter._QASM2QCXConverter__iterator_binding('value', self.bit_binding()):
            self.assertEqual(converter._QASM2QCXConverter__loop_expression_info(expression, {'value': qasm2qcx.ValueType.INT}),
                             (qasm2qcx.ValueType.INT, False))
            types = converter._QASM2QCXConverter__loop_iterator_types('inner')
            self.assertEqual(types, {'value': qasm2qcx.ValueType.BIT, 'inner': qasm2qcx.ValueType.INT})
            statement = qasm2qcx.openqasm3.parser.parse('OPENQASM 3.0; for int i in {value} {}').statements[0]
            with self.assertRaises(qasm2qcx.InvalidLoopRangeException):
                converter._QASM2QCXConverter__validate_nested_loop_bounds(statement, types)
            statement = qasm2qcx.openqasm3.parser.parse('OPENQASM 3.0; for int i in {int(value)} {}').statements[0]
            converter._QASM2QCXConverter__validate_nested_loop_bounds(statement, types)

    def test_integer_and_bit_bindings_shadow_and_restore(self) -> None:
        converter = self.converter()
        bind = converter._QASM2QCXConverter__iterator_binding
        with bind('value', self.bit_binding()):
            with bind('value', qasm2qcx._RuntimeIteratorBinding('PRIVATE_INT')):
                self.assertEqual(RuntimeIteratorInfrastructureTests.identifier_value(converter, 'value')[1],
                                 qasm2qcx.ValueType.INT)
            self.assertEqual(RuntimeIteratorInfrastructureTests.identifier_value(converter, 'value')[1],
                             qasm2qcx.ValueType.BIT)
            with bind('value', 2):
                self.assertEqual(RuntimeIteratorInfrastructureTests.identifier_value(converter, 'value'),
                                 (2, qasm2qcx.ValueType.INT, qasm2qcx.ValueKind.LITERAL))
        self.assertEqual(converter._QASM2QCXConverter__loop_bindings, [])

    def test_bit_binding_rejects_indexed_use_writes_arithmetic_and_register_copy(self) -> None:
        for source in ('value = 1;', 'value = measure q;', 'out = value[0];',
                       'out = -value;', 'value += 1;', 'singleton = value;',
                       'ready = value + 1;', 'ready = int(value[0]);'):
            converter = self.converter()
            statement = qasm2qcx.openqasm3.parser.parse('OPENQASM 3.0; ' + source).statements[0]
            with converter._QASM2QCXConverter__iterator_binding('value', self.bit_binding()), \
                    self.assertRaises(qasm2qcx.QASM2QCXError):
                converter.visit(statement)

    def test_bit_binding_is_not_a_constant(self) -> None:
        converter = self.converter()
        with converter._QASM2QCXConverter__iterator_binding('value', self.bit_binding()):
            with self.assertRaises(qasm2qcx.NoConstantExpressionException):
                converter._QASM2QCXConverter__constant_loop_integer(qasm2qcx.ast.Identifier('value'), 'start')

    def test_register_size_cache_distinguishes_scalar_and_singleton(self) -> None:
        program = qasm2qcx.openqasm3.parser.parse('OPENQASM 3.0; const int N = 3; '
            'bit scalar; bit[1] singleton; bit[N] flags;')
        converter = qasm2qcx.QASM2QCXConverter(program)
        self.assertEqual(converter._QASM2QCXConverter__bit_register_source_sizes, {'singleton': 1, 'flags': 3})
        with converter._QASM2QCXConverter__iterator_binding('N', self.bit_binding()):
            converter.visit(program)
        self.assertIn('VAR FLAGS31 INT 3', list(converter))

    def test_bit_size_evaluation_restores_state_on_success_and_failure(self) -> None:
        converter = self.converter()
        converter._QASM2QCXConverter__expression_kind = qasm2qcx.ExpressionKind.ARITHMETIC
        converter._QASM2QCXConverter__value = 17
        converter._QASM2QCXConverter__evaluate_constant = False
        fields = ('expression_kind', 'value', 'value_type', 'value_kind', 'evaluate_constant')
        before = [getattr(converter, '_QASM2QCXConverter__' + field) for field in fields]
        for size in ('2 + 1', '0', 'missing'):
            declaration = qasm2qcx.openqasm3.parser.parse(f'OPENQASM 3.0; bit[{size}] flags;').statements[0]
            if size == '2 + 1':
                self.assertEqual(converter._QASM2QCXConverter__bit_type_size(declaration.type, 'flags'), 3)
            else:
                with self.assertRaises(qasm2qcx.QASM2QCXError):
                    converter._QASM2QCXConverter__bit_type_size(declaration.type, 'flags')
            self.assertEqual([getattr(converter, '_QASM2QCXConverter__' + field) for field in fields], before)

    def test_bit_register_loop_lowering_accepts_scalar_bit_iterator(self) -> None:
        convert('OPENQASM 3.0; bit[2] flags = "01"; for bit value in flags {}')


class BitRegisterLoopLoweringTests(unittest.TestCase):
    def test_body_is_generated_once_with_live_indexed_bit_load(self) -> None:
        lines = convert('OPENQASM 3.0; bit[3] flags = "101"; int total = 0; '
                        'for bit value in flags { total += int(value); }')
        self.assertIn('LET QASM2QCX_INT_0 := FLAGS31:QASM2QCX_INT_1', lines)
        self.assertEqual(sum(line.startswith('LET TOTAL31 +=') for line in lines), 1)
        self.assertIn('JUMPIF QASM2QCX_LOOP_0_END QASM2QCX_INT_1 == 2', lines)
        ForLoopControlLoweringTests.assert_resolved_jumps(lines)

    def test_singleton_register_uses_scalar_qcx_storage(self) -> None:
        lines = convert('OPENQASM 3.0; bit[1] flags = "1"; '
                        'for bit value in flags { continue; }')
        self.assertIn('LET QASM2QCX_INT_0 := FLAGS31', lines)
        self.assertNotIn('LET QASM2QCX_INT_0 := FLAGS31:QASM2QCX_INT_1', lines)
        self.assertIn('JUMPIF QASM2QCX_LOOP_0_END QASM2QCX_INT_1 == 0', lines)

    def test_bit_iterator_conditions_assignments_and_casts(self) -> None:
        lines = convert('OPENQASM 3.0; bit[2] flags = "01"; bit out; bool ready; int total = 0; '
                        'for bit value in flags { out = value; ready = !value; '
                        'if (value && out) { total += int(value); } }')
        self.assertIn('LET OUT7 := QASM2QCX_INT_0', lines)
        self.assertTrue(any(' QASM2QCX_INT_0 \\= 0' in line for line in lines))

    def test_source_is_resolved_before_shadowing_and_outer_binding_restored(self) -> None:
        lines = convert('OPENQASM 3.0; bit[2] flags = "01"; int total = 0; '
                        'for bit flags in flags { total += int(flags); } total += int(flags[0]);')
        self.assertTrue(any(' := :INT:FLAGS31:0' in line for line in lines))
        for body in ('flags = 0;', 'flags = measure q;', 'total += int(flags[0]);',
                     'for bit value in flags {}'):
            with self.subTest(body=body), self.assertRaises(qasm2qcx.QASM2QCXError):
                convert('OPENQASM 3.0; bit[2] flags = "01"; qubit q; int total = 0; '
                        'for bit flags in flags { ' + body + ' }')

    def test_mixed_loop_types_validate_nested_bounds_and_nearest_transfers(self) -> None:
        lines = convert('OPENQASM 3.0; bit[2] flags = "01"; array[int, 2] a = {0, 1}; '
                        'for bit value in flags { for int i in {int(value)} { continue; } '
                        'for int i in a { break; } continue; }')
        ForLoopControlLoweringTests.assert_resolved_jumps(lines)
        with self.assertRaises(qasm2qcx.InvalidLoopRangeException):
            convert('OPENQASM 3.0; bit[2] flags = "01"; '
                    'for bit value in flags { for int i in {value} {} }')

    def test_invalid_sources_and_iterator_types_are_rejected_even_in_empty_outer_loop(self) -> None:
        prefix = 'OPENQASM 3.0; bit scalar; bit[1] flags; array[int, 1] a; int n = 0; '
        for loop in ('for bit value in scalar {}', 'for int value in flags {}',
                     'for bit value in a {}', 'for bit[1] value in flags {}',
                     'for bool value in flags {}', 'for bit value in n {}',
                     'for bit value in [0:1] {}', 'for bit value in {0, 1} {}'):
            for source in (loop, 'for int i in [1:0] { ' + loop + ' }'):
                with self.subTest(source=source), self.assertRaises(qasm2qcx.QASM2QCXError):
                    convert(prefix + source)
        with self.assertRaises(qasm2qcx.NoVariableNameException):
            convert('OPENQASM 3.0; for bit value in flags {} bit[1] flags;')

    def test_register_length_and_runtime_count_limits(self) -> None:
        maximum = qasm2qcx.QASM2QCXConverter.QCX_INT_MAX
        with self.assertRaises(qasm2qcx.InvalidDeclarationException):
            convert(f'OPENQASM 3.0; bit[{maximum + 1}] flags; for bit value in flags {{}}')
        with patch.object(qasm2qcx.QASM2QCXConverter, 'MAX_LOOP_ITERATIONS', 0):
            lines = convert(f'OPENQASM 3.0; bit[{maximum}] flags; for bit value in flags {{}}')
        self.assertIn(f'JUMPIF QASM2QCX_LOOP_0_END QASM2QCX_INT_1 == {maximum - 1}', lines)

    def test_nested_loop_storage_and_bindings_are_released(self) -> None:
        program = qasm2qcx.openqasm3.parser.parse('OPENQASM 3.0; bit[2] flags = "01"; int total = 0; '
            'for bit value in flags { for bit inner in flags { total += int(value && inner); } }')
        converter = qasm2qcx.QASM2QCXConverter(program)
        converter.visit(program)
        self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())
        self.assertEqual(converter._QASM2QCXConverter__loop_bindings, [])
        self.assertEqual(converter._QASM2QCXConverter__loop_contexts, [])


class BitRegisterLoopRegressionTests(unittest.TestCase):
    def test_codegen_size_is_independent_of_register_length(self) -> None:
        template = ('OPENQASM 3.0; bit[{size}] flags; int total = 0; '
                    'for bit value in flags {{ total += int(value); }}')
        lengths = [len(convert(template.format(size=size)))
                   for size in (1, 3, qasm2qcx.QASM2QCXConverter.QCX_INT_MAX)]
        self.assertEqual(lengths, [lengths[0]] * 3)

    def test_skipped_nested_loops_preserve_bit_iterator_type_validation(self) -> None:
        prefix = 'OPENQASM 3.0; bit[2] flags = "01"; '
        for bounds in ('{value}', '[0:value]', '[0:value:2]'):
            source = 'for int k in [1:0] { for bit value in flags { for int j in ' + bounds + ' {} } }'
            with self.subTest(bounds=bounds), self.assertRaises(qasm2qcx.InvalidLoopRangeException):
                convert(prefix + source)
        convert(prefix + 'for int k in [1:0] { for bit value in flags { for int j in {int(value)} {} } }')

    def test_mixed_type_shadowing_and_jumps_are_valid(self) -> None:
        lines = convert('OPENQASM 3.0; bit[2] flags = "01"; int total = 0; '
            'for bit value in flags { for int value in {2, 3} { total += value; } total += int(value); } '
            'for int value in [0:1] { for bit value in flags { total += int(value); } total += value; }')
        ForLoopControlLoweringTests.assert_resolved_jumps(lines)
        labels = [line for line in lines if line.startswith('@')]
        self.assertEqual(len(labels), len(set(labels)))

    def test_reserved_source_names_and_nested_temporaries_are_not_reused(self) -> None:
        lines = convert('OPENQASM 3.0; bit[2] QASM2QCX_INT_ = "01"; int total = 0; '
            'for bit value in QASM2QCX_INT_ { total += int(value) % 2; '
            'for bit other in QASM2QCX_INT_ { total += int(other && value); } }')
        self.assertIn('LET QASM2QCX_INT_1 := QASM2QCX_INT_0:QASM2QCX_INT_2', lines)
        body = lines.index('@QASM2QCX_LOOP_0_BODY')
        end = lines.index('@QASM2QCX_LOOP_0_NEXT')
        targets = [line.split()[1] for line in lines[body + 2:end] if line.startswith('LET ')]
        self.assertNotIn('QASM2QCX_INT_1', targets)
        self.assertNotIn('QASM2QCX_INT_2', targets)
        self.assertTrue(all(position < body for position, line in enumerate(lines)
                            if line.startswith('VAR QASM2QCX_')))

    def test_failed_body_releases_loop_storage_and_context(self) -> None:
        program = qasm2qcx.openqasm3.parser.parse('OPENQASM 3.0; bit[2] flags = "01"; int total = 0; '
            'for bit value in flags { total = missing; }')
        converter = qasm2qcx.QASM2QCXConverter(program)
        with self.assertRaises(qasm2qcx.NoVariableNameException):
            converter.visit(program)
        self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())
        self.assertEqual(converter._QASM2QCXConverter__loop_bindings, [])
        self.assertEqual(converter._QASM2QCXConverter__loop_contexts, [])

    def test_skipped_paths_still_reject_iterator_writes_and_invalid_sources(self) -> None:
        prefix = 'OPENQASM 3.0; bit[2] flags = "01"; qubit q; '
        for body in ('continue; value = 0;', 'break; value = measure q;',
                     'if (false) { value = true; }', 'while (false) { for bit v in value {} }'):
            with self.subTest(body=body), self.assertRaises(qasm2qcx.UnsupportedOpenQASMError):
                convert(prefix + 'for bit value in flags { ' + body + ' }')

    def test_runtime_count_does_not_charge_expansion_budget_but_nested_sets_do(self) -> None:
        with patch.object(qasm2qcx.QASM2QCXConverter, 'MAX_LOOP_ITERATIONS', 0):
            convert('OPENQASM 3.0; bit[2] flags; for bit value in flags {}')
            with self.assertRaises(qasm2qcx.InvalidLoopRangeException):
                convert('OPENQASM 3.0; bit[2] flags; for bit value in flags { for int i in {int(value)} {} }')
        with patch.object(qasm2qcx.QASM2QCXConverter, 'MAX_LOOP_OUTPUT_LINES', 10):
            with self.assertRaises(qasm2qcx.InvalidLoopRangeException):
                convert('OPENQASM 3.0; bit[2] flags; for int i in {0} { for bit value in flags {} }')


class RuntimeIteratorInfrastructureTests(unittest.TestCase):
    @staticmethod
    def converter():
        program = qasm2qcx.openqasm3.parser.parse('''OPENQASM 3.0;
            int i = 9; int total = 0; qubit[2] q; bit[2] flags;''')
        converter = qasm2qcx.QASM2QCXConverter(program)
        converter.visit(program)
        return converter

    @staticmethod
    def identifier_value(converter, name='i'):
        converter._QASM2QCXConverter__expression_kind = qasm2qcx.ExpressionKind.ARITHMETIC
        converter.visit(qasm2qcx.ast.Identifier(name))
        return (converter._QASM2QCXConverter__value,
                converter._QASM2QCXConverter__value_type,
                converter._QASM2QCXConverter__value_kind)

    def test_runtime_identifier_is_integer_lvalue_not_constant_or_temporary(self) -> None:
        converter = self.converter()
        binding = qasm2qcx._RuntimeIteratorBinding('PRIVATE_ITERATOR')
        with converter._QASM2QCXConverter__iterator_binding('i', binding):
            self.assertEqual(self.identifier_value(converter),
                             ('PRIVATE_ITERATOR', qasm2qcx.ValueType.INT, qasm2qcx.ValueKind.LVALUE))
        self.assertEqual(self.identifier_value(converter),
                         ('I1', qasm2qcx.ValueType.INT, qasm2qcx.ValueKind.LVALUE))

    def test_mixed_nested_bindings_shadow_and_restore_even_after_failure(self) -> None:
        converter = self.converter()
        bind = converter._QASM2QCXConverter__iterator_binding
        runtime = qasm2qcx._RuntimeIteratorBinding('PRIVATE_ITERATOR')
        with bind('i', 2):
            with bind('i', runtime):
                self.assertEqual(self.identifier_value(converter)[0], 'PRIVATE_ITERATOR')
                with self.assertRaisesRegex(RuntimeError, 'body failure'):
                    with bind('i', 3):
                        self.assertEqual(self.identifier_value(converter),
                                         (3, qasm2qcx.ValueType.INT, qasm2qcx.ValueKind.LITERAL))
                        raise RuntimeError('body failure')
                self.assertEqual(self.identifier_value(converter)[0], 'PRIVATE_ITERATOR')
            self.assertEqual(self.identifier_value(converter)[0], 2)
        self.assertEqual(converter._QASM2QCXConverter__loop_bindings, [])

    def test_runtime_iterator_cannot_be_evaluated_as_a_constant_in_either_pass(self) -> None:
        converter = self.converter()
        before = list(converter._QASM2QCXConverter__qcx_lines)
        for initialization in (True, False):
            with self.subTest(initialization=initialization):
                converter._QASM2QCXConverter__is_initialization_process = initialization
                with converter._QASM2QCXConverter__iterator_binding(
                        'i', qasm2qcx._RuntimeIteratorBinding('PRIVATE_ITERATOR')):
                    expression = qasm2qcx.ast.Identifier('i')
                    with self.assertRaises(qasm2qcx.NoConstantExpressionException):
                        converter._QASM2QCXConverter__constant_loop_integer(expression, 'start')
        self.assertEqual(converter._QASM2QCXConverter__qcx_lines, before)

    def test_static_indices_reject_runtime_iterator_but_accept_constant_expressions(self) -> None:
        converter = self.converter()
        with converter._QASM2QCXConverter__iterator_binding(
                'i', qasm2qcx._RuntimeIteratorBinding('PRIVATE_ITERATOR')):
            for body in ('reset q[i];', 'reset q[int(i)];'):
                with self.subTest(body=body):
                    statement = qasm2qcx.openqasm3.parser.parse('OPENQASM 3.0; ' + body).statements[0]
                    with self.assertRaises(qasm2qcx.QASM2QCXError):
                        converter.visit(statement)
            statement = qasm2qcx.openqasm3.parser.parse('OPENQASM 3.0; reset q[1 - 1];').statements[0]
            converter.visit(statement)
            self.assertEqual(converter._QASM2QCXConverter__qcx_lines[-1], 'RESET 0')
        with converter._QASM2QCXConverter__iterator_binding(
                'q', qasm2qcx._RuntimeIteratorBinding('PRIVATE_ITERATOR')):
            statement = qasm2qcx.openqasm3.parser.parse('OPENQASM 3.0; reset q;').statements[0]
            with self.assertRaises(qasm2qcx.InvalidQubitOperandException):
                converter.visit(statement)

    def test_runtime_iterators_retain_read_only_and_non_register_restrictions(self) -> None:
        converter = self.converter()
        with converter._QASM2QCXConverter__iterator_binding(
                'i', qasm2qcx._RuntimeIteratorBinding('PRIVATE_ITERATOR')):
            for body in ('i = 1;', 'i += 1;', 'i = measure q[0];',
                         'while (false) { i = 1; }'):
                with self.subTest(body=body):
                    statements = qasm2qcx.openqasm3.parser.parse('OPENQASM 3.0; ' + body).statements
                    with self.assertRaisesRegex(qasm2qcx.UnsupportedOpenQASMError, 'for-loop iterator'):
                        converter._QASM2QCXConverter__validate_loop_body(statements, {'i'})
            for body in ('total = i[0];', 'reset i;'):
                with self.subTest(body=body):
                    statement = qasm2qcx.openqasm3.parser.parse('OPENQASM 3.0; ' + body).statements[0]
                    with self.assertRaises(qasm2qcx.QASM2QCXError):
                        converter.visit(statement)

    def test_live_iterator_storage_is_not_released_by_expression_conversion(self) -> None:
        converter = self.converter()
        storage = converter._QASM2QCXConverter__add_new_temporary_variable(qasm2qcx.ValueType.INT)
        with converter._QASM2QCXConverter__iterator_binding('i', qasm2qcx._RuntimeIteratorBinding(storage)):
            for body in ('total = i + 1;', 'total = -i;', 'total = i % 3;', 'if (i < 2) {}'):
                statement = qasm2qcx.openqasm3.parser.parse('OPENQASM 3.0; ' + body).statements[0]
                converter.visit(statement)
                self.assertIn(storage, converter._QASM2QCXConverter__used_temporary_variables)
        self.assertFalse(any(line.startswith(f'LET {storage} ')
                             for line in converter._QASM2QCXConverter__qcx_lines))
        converter._QASM2QCXConverter__release_temporary_variable(storage)
        self.assertNotIn(storage, converter._QASM2QCXConverter__used_temporary_variables)

    def test_runtime_scope_alone_does_not_charge_constant_expansion_limits(self) -> None:
        converter = self.converter()
        statements = qasm2qcx.openqasm3.parser.parse(
            'OPENQASM 3.0; total += 1; while (false) { total += 1; }').statements
        with converter._QASM2QCXConverter__iterator_binding(
                'i', qasm2qcx._RuntimeIteratorBinding('PRIVATE_ITERATOR')), \
                patch.object(qasm2qcx.QASM2QCXConverter, 'MAX_EXPANDED_LOOP_STATEMENTS', 0), \
                patch.object(qasm2qcx.QASM2QCXConverter, 'MAX_LOOP_OUTPUT_LINES', 0):
            for statement in statements:
                converter.visit(statement)
            self.assertFalse(converter._QASM2QCXConverter__inside_expanded_loop())
        self.assertEqual(converter._QASM2QCXConverter__expanded_loop_statements, 0)

    def test_constant_scope_still_charges_limits_inside_runtime_scope(self) -> None:
        for outer_runtime in (True, False):
            with self.subTest(outer_runtime=outer_runtime):
                converter = self.converter()
                runtime = qasm2qcx._RuntimeIteratorBinding('PRIVATE_ITERATOR')
                outer, inner = (runtime, 2) if outer_runtime else (2, runtime)
                with converter._QASM2QCXConverter__iterator_binding('i', outer), \
                        converter._QASM2QCXConverter__iterator_binding('j', inner), \
                        patch.object(qasm2qcx.QASM2QCXConverter, 'MAX_EXPANDED_LOOP_STATEMENTS', 0):
                    self.assertTrue(converter._QASM2QCXConverter__inside_expanded_loop())
                    statement = qasm2qcx.openqasm3.parser.parse('OPENQASM 3.0; total += 1;').statements[0]
                    with self.assertRaises(qasm2qcx.InvalidLoopRangeException):
                        converter.visit(statement)
                self.assertEqual(converter._QASM2QCXConverter__loop_bindings, [])

    def test_runtime_range_bounds_now_use_runtime_lowering(self) -> None:
        for bounds in ('[0:limit]', '[limit:3]', '[0:-1:limit]'):
            with self.subTest(bounds=bounds):
                lines = convert(f'OPENQASM 3.0; int limit = 2; for int i in {bounds} {{}}')
                self.assertIn('@QASM2QCX_LOOP_0_NEXT', lines)
                ForLoopControlLoweringTests.assert_resolved_jumps(lines)


class RuntimeStrideInfrastructureTests(unittest.TestCase):
    @staticmethod
    def converter():
        program = qasm2qcx.openqasm3.parser.parse('''OPENQASM 3.0;
            const int stride = 2; const int backwards = 3;
            int last = 3; int dynamic = 2;''')
        converter = qasm2qcx.QASM2QCXConverter(program)
        converter.visit(program)
        return converter

    @staticmethod
    def bounds(step):
        source = '[0:last]' if step is None else f'[0:{step}:last]'
        return qasm2qcx.openqasm3.parser.parse(
            f'OPENQASM 3.0; for int i in {source} {{}}').statements[0].set_declaration

    def test_qcx_limits_match_backend_native_int_representation(self) -> None:
        bits = ctypes.sizeof(ctypes.c_int) * 8
        self.assertEqual(qasm2qcx.QASM2QCXConverter.QCX_INT_MIN, -(1 << (bits - 1)))
        self.assertEqual(qasm2qcx.QASM2QCXConverter.QCX_INT_MAX, (1 << (bits - 1)) - 1)

    def test_validation_accepts_constant_nonzero_steps_and_extreme_signed_values(self) -> None:
        converter = self.converter()
        for source, expected in ((None, 1), ('stride', 2), ('-backwards', -3),
                                  ('stride + 1', 3), ('int(2.5)', 2),
                                  (str(converter.QCX_INT_MIN), converter.QCX_INT_MIN),
                                  (str(converter.QCX_INT_MAX), converter.QCX_INT_MAX)):
            with self.subTest(step=source):
                self.assertEqual(converter._QASM2QCXConverter__validated_runtime_range_step(
                    self.bounds(source)), expected)

    def test_validation_rejects_zero_runtime_unknown_and_noninteger_steps(self) -> None:
        converter = self.converter()
        for step in ('0', 'stride - 2', 'dynamic', 'missing', '1.5', 'true', '1 / 0'):
            with self.subTest(step=step):
                with self.assertRaises(qasm2qcx.QASM2QCXError):
                    converter._QASM2QCXConverter__validated_runtime_range_step(self.bounds(step))

    def test_unrepresentable_runtime_steps_have_converter_specific_errors(self) -> None:
        converter = self.converter()
        for step in (converter.QCX_INT_MIN - 1, converter.QCX_INT_MAX + 1, 10 ** 100):
            with self.subTest(step=step):
                # Isolate range validation from checked unary negation.
                source = f'(0 - {-step})' if step < 0 else str(step)
                with self.assertRaisesRegex(qasm2qcx.InvalidLoopRangeException, 'must fit QCX INT'):
                    converter._QASM2QCXConverter__validated_runtime_range_step(self.bounds(source))
                with self.assertRaisesRegex(qasm2qcx.InvalidLoopRangeException, 'must fit QCX INT'):
                    convert('OPENQASM 3.0; int last = 3; for int i in [1:0] '
                            f'{{ for int j in [0:{source}:last] {{}} }}')

    def test_constant_range_steps_retain_unbounded_compile_time_integer_behavior(self) -> None:
        prefix = 'OPENQASM 3.0; int total = 0; '
        step = 10 ** 100
        self.assertEqual(convert(prefix + f'for int i in [0:{step}:1] {{ total += i; }}'),
                         convert(prefix + 'total += 0;'))
        self.assertEqual(convert(prefix + f'for int i in [0:0 - {step}:-1] {{ total += i; }}'),
                         convert(prefix + 'total += 0;'))

    def test_nonunit_runtime_strides_now_use_safe_advance_lowering(self) -> None:
        for step in (2, -2, qasm2qcx.QASM2QCXConverter.QCX_INT_MIN):
            with self.subTest(step=step):
                lines = convert(f'OPENQASM 3.0; int last = 3; for int i in [0:{step}:last] {{}}')
                self.assertIn('@QASM2QCX_LOOP_0_CANDIDATE', lines)
                ForLoopControlLoweringTests.assert_resolved_jumps(lines)

    def test_step_validation_preserves_expression_and_output_state(self) -> None:
        converter = self.converter()
        fields = ('expression_kind', 'value', 'value_type', 'value_kind', 'evaluate_constant')
        before = tuple(getattr(converter, '_QASM2QCXConverter__' + field) for field in fields)
        lines = converter._QASM2QCXConverter__qcx_lines.copy()
        for step in ('stride + 1', 'dynamic', '1 / 0'):
            with self.subTest(step=step):
                if step == 'stride + 1':
                    converter._QASM2QCXConverter__validated_runtime_range_step(self.bounds(step))
                else:
                    with self.assertRaises(qasm2qcx.QASM2QCXError):
                        converter._QASM2QCXConverter__validated_runtime_range_step(self.bounds(step))
                self.assertEqual(tuple(getattr(converter, '_QASM2QCXConverter__' + field)
                                       for field in fields), before)
                self.assertEqual(converter._QASM2QCXConverter__qcx_lines, lines)

    @staticmethod
    def advance_lines(step):
        converter = qasm2qcx.QASM2QCXConverter(
            qasm2qcx.openqasm3.parser.parse('OPENQASM 3.0;'))
        converter._QASM2QCXConverter__emit_runtime_range_next(
            'ITERATOR', 'STOP', step, 'BODY', 'END', 'CANDIDATE')
        if converter._QASM2QCXConverter__used_temporary_variables:
            raise AssertionError('Safe-advance helper leaked temporary storage')
        return converter._QASM2QCXConverter__qcx_lines[1:]

    def checked_advance(self, lines, iterator, stop, minimum, maximum, step=None):
        # Interpret just the emitted guard, checking every executed arithmetic
        # operation against an explicit signed integer model (not Python's
        # unlimited integers). BODY and END are external control-flow targets.
        values = {'ITERATOR': iterator, 'STOP': stop}
        if step is not None:
            values['STEP'] = step
        labels = {line[1:]: index for index, line in enumerate(lines) if line.startswith('@')}

        def operand(name):
            return values[name] if name in values else int(name)

        position = 0
        for _ in range(30):
            tokens = lines[position].split()
            target = None
            if tokens[0] == 'JUMP':
                target = tokens[1]
            elif tokens[0] == 'JUMPIF':
                lhs, rhs = operand(tokens[2]), operand(tokens[4])
                conditions = {'==': lhs == rhs, '>': lhs > rhs,
                              '<': lhs < rhs, '>=': lhs >= rhs}
                if conditions[tokens[3]]:
                    target = tokens[1]
            elif tokens[0] == 'LET':
                rhs = operand(tokens[3])
                result = (rhs if tokens[2] == ':=' else
                          values[tokens[1]] + rhs if tokens[2] == '+=' else
                          values[tokens[1]] - rhs)
                self.assertGreaterEqual(result, minimum, msg=str(lines))
                self.assertLessEqual(result, maximum, msg=str(lines))
                values[tokens[1]] = result
            if target in ('BODY', 'END'):
                return values['ITERATOR'] if target == 'BODY' else None
            position = labels[target] if target is not None else position + 1
        self.fail('Safe-advance guard did not reach BODY or END')

    def test_safe_advance_exhaustively_checks_small_signed_integer_model(self) -> None:
        minimum, maximum = -16, 15
        for step in range(minimum, maximum + 1):
            if step == 0:
                continue
            lines = self.advance_lines(step)
            for iterator in range(minimum, maximum + 1):
                for stop in range(minimum, maximum + 1):
                    if (step > 0 and iterator > stop) or (step < 0 and iterator < stop):
                        continue  # The entry check excludes these states.
                    candidate = iterator + step
                    expected = candidate if (candidate <= stop if step > 0 else candidate >= stop) else None
                    self.assertEqual(self.checked_advance(lines, iterator, stop, minimum, maximum),
                                     expected, msg=f'{iterator=}, {stop=}, {step=}')

    def test_safe_advance_checks_backend_endpoints_and_minimum_step_without_negation(self) -> None:
        minimum, maximum = qasm2qcx.QASM2QCXConverter.QCX_INT_MIN, qasm2qcx.QASM2QCXConverter.QCX_INT_MAX
        values = (minimum, minimum + 1, minimum + 2, -2, -1, 0, 1, 2,
                  maximum - 2, maximum - 1, maximum)
        for step in (minimum, minimum + 1, -maximum, -3, -1, 1, 3, maximum):
            lines = self.advance_lines(step)
            for iterator in values:
                for stop in values:
                    if (step > 0 and iterator > stop) or (step < 0 and iterator < stop):
                        continue
                    candidate = iterator + step
                    expected = candidate if (candidate <= stop if step > 0 else candidate >= stop) else None
                    self.assertEqual(self.checked_advance(lines, iterator, stop, minimum, maximum),
                                     expected, msg=f'{iterator=}, {stop=}, {step=}')

    def test_unit_step_guard_output_is_unchanged(self) -> None:
        for step, operation in ((1, '+='), (-1, '-=')):
            self.assertEqual(self.advance_lines(step), [
                'JUMPIF END ITERATOR == STOP', f'LET ITERATOR {operation} 1', 'JUMP BODY'])


class RuntimeForLoweringTests(unittest.TestCase):
    def test_capture_entry_check_and_endpoint_guard(self) -> None:
        lines = convert('OPENQASM 3.0; int limit = 3; int total = 0; for int i in [0:limit] { total += i; }')
        capture = lines.index('LET QASM2QCX_INT_1 := LIMIT31')
        self.assertEqual(lines[capture - 1:], [
            'LET QASM2QCX_INT_0 := 0', 'LET QASM2QCX_INT_1 := LIMIT31',
            'JUMPIF QASM2QCX_LOOP_0_END QASM2QCX_INT_0 > QASM2QCX_INT_1',
            '@QASM2QCX_LOOP_0_BODY', 'LET TOTAL31 += QASM2QCX_INT_0',
            '@QASM2QCX_LOOP_0_NEXT', 'JUMPIF QASM2QCX_LOOP_0_END QASM2QCX_INT_0 == QASM2QCX_INT_1',
            'LET QASM2QCX_INT_0 += 1', 'JUMP QASM2QCX_LOOP_0_BODY', '@QASM2QCX_LOOP_0_END'])
        ForLoopControlLoweringTests.assert_resolved_jumps(lines)

    def test_descending_guard_precedes_decrement(self) -> None:
        lines = convert('OPENQASM 3.0; int limit = 0; for int i in [3:-1:limit] { continue; }')
        self.assertIn('JUMPIF QASM2QCX_LOOP_0_END QASM2QCX_INT_0 < QASM2QCX_INT_1', lines)
        guard = lines.index('JUMPIF QASM2QCX_LOOP_0_END QASM2QCX_INT_0 == QASM2QCX_INT_1')
        self.assertEqual(lines[guard + 1], 'LET QASM2QCX_INT_0 -= 1')
        self.assertIn('JUMP QASM2QCX_LOOP_0_NEXT', lines)
        ForLoopControlLoweringTests.assert_resolved_jumps(lines)

    def test_bound_capture_is_before_iterator_shadowing(self) -> None:
        lines = convert('OPENQASM 3.0; int i = 1; int total = 0; for int i in [i:i + 2] { total += i; }')
        self.assertIn('LET QASM2QCX_INT_0 := I1', lines)
        body = lines.index('@QASM2QCX_LOOP_0_BODY')
        self.assertTrue(any(' := I1' in line for line in lines[:body]))
        self.assertEqual(lines[body + 1], 'LET TOTAL31 += QASM2QCX_INT_0')

    def test_explicit_integer_casts_and_int_backed_uint_bounds(self) -> None:
        for declarations, bounds in (('float first = 0.5; uint last = 3;', '[int(first):last]'),
                                      ('bit flag = 1; int last = 3;', '[int(flag):last]'),
                                      ('bit[2] flags = "01"; int last = 3;', '[int(flags[0]):last]')):
            with self.subTest(bounds=bounds):
                ForLoopControlLoweringTests.assert_resolved_jumps(
                    convert('OPENQASM 3.0; ' + declarations + f'for int i in {bounds} {{}}'))

    def test_nested_runtime_loops_hold_distinct_live_storage(self) -> None:
        lines = convert('''OPENQASM 3.0; int limit = 2; int total = 0;
            for int i in [0:limit] {
                for int j in [0:i] { total += i + j; continue; }
                break;
            }''')
        self.assertIn('@QASM2QCX_LOOP_1_NEXT', lines)
        self.assertIn('JUMP QASM2QCX_LOOP_1_NEXT', lines)
        self.assertIn('JUMP QASM2QCX_LOOP_0_END', lines)
        declarations = [line for line in lines if line.startswith('VAR QASM2QCX_INT_')]
        self.assertGreaterEqual(len(declarations), 5)
        ForLoopControlLoweringTests.assert_resolved_jumps(lines)

    def test_constant_range_output_and_nonunit_strides_remain_unrolled(self) -> None:
        prefix = 'OPENQASM 3.0; int total = 0; '
        self.assertEqual(convert(prefix + 'for int i in [0:2:4] { total += i; }'),
                         convert(prefix + 'total += 0; total += 2; total += 4;'))

    def test_unknown_names_and_invalid_range_operand_types_rejected(self) -> None:
        prefix = ('OPENQASM 3.0; int limit = 3; int step = 1; float f = 2.5; '
                  'bool ready = true; complex value = 1.0im; ')
        for bounds in ('[0:0:limit]', '[0:1.5:limit]', '[0:ready:limit]', '[0:missing]',
                       '[0:f]', '[ready:limit]', '[0:int(value)]', '[0:limit + missing]'):
            with self.subTest(bounds=bounds):
                with self.assertRaises(qasm2qcx.QASM2QCXError):
                    convert(prefix + f'for int i in {bounds} {{}}')

    def test_invalid_constant_arithmetic_is_not_reclassified_as_runtime(self) -> None:
        for bounds in ('[0:1 / 0]', '[0:1 % 0]', '[0:0:3]'):
            with self.subTest(bounds=bounds):
                with self.assertRaises(qasm2qcx.QASM2QCXError):
                    convert('OPENQASM 3.0; for int i in ' + bounds + ' {}')

    def test_empty_outer_still_checks_independent_invalid_runtime_header(self) -> None:
        for inner in ('for int j in [0:0:limit] {}', 'for int j in [0:f] {}',
                      'for int j in [0:missing] {}'):
            with self.subTest(inner=inner):
                with self.assertRaises(qasm2qcx.QASM2QCXError):
                    convert('OPENQASM 3.0; int limit = 3; float f = 2.5; for int i in [1:0] { ' + inner + ' }')

    def test_runtime_iterators_cannot_be_indices_or_assignment_targets(self) -> None:
        prefix = 'OPENQASM 3.0; int limit = 3; qubit[4] q; bit[4] flags; '
        for body in ('reset q[i];', 'i = 1;', 'i = measure q[0];',
                     'int local;'):
            with self.subTest(body=body):
                with self.assertRaises(qasm2qcx.QASM2QCXError):
                    convert(prefix + 'for int i in [0:limit] { ' + body + ' }')

    def test_zero_iteration_and_transfer_paths_hoist_storage_not_computations(self) -> None:
        lines = convert('''OPENQASM 3.0; int limit = -1; int value = 7; int total = 0;
            for int i in [0:limit] { continue; total += value % 0; }
            total += value + 1;''')
        entry = next(index for index, line in enumerate(lines) if line.startswith('JUMPIF '))
        declarations = [index for index, line in enumerate(lines) if line.startswith('VAR QASM2QCX_')]
        self.assertTrue(all(index < entry for index in declarations))
        self.assertGreater(next(index for index, line in enumerate(lines) if ' /= ' in line), entry)

    def test_runtime_iterations_do_not_consume_constant_iteration_budget(self) -> None:
        with patch.object(qasm2qcx.QASM2QCXConverter, 'MAX_LOOP_ITERATIONS', 0):
            convert('OPENQASM 3.0; int limit = 1000000; for int i in [0:limit] {}')
            with self.assertRaises(qasm2qcx.InvalidLoopRangeException):
                convert('OPENQASM 3.0; int limit = -1; for int i in [0:limit] { for int j in {1} {} }')

    def test_runtime_loop_resources_restored_after_body_failure(self) -> None:
        program = qasm2qcx.openqasm3.parser.parse('OPENQASM 3.0; int limit = 3;')
        converter = qasm2qcx.QASM2QCXConverter(program)
        converter.visit(program)
        loop = qasm2qcx.openqasm3.parser.parse(
            'OPENQASM 3.0; for int i in [0:limit] { reset missing; }').statements[0]
        with self.assertRaises(qasm2qcx.InvalidQubitOperandException):
            converter.visit(loop)
        self.assertEqual(converter._QASM2QCXConverter__loop_bindings, [])
        self.assertEqual(converter._QASM2QCXConverter__loop_contexts, [])
        self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())


class RuntimeStepLoweringTests(unittest.TestCase):
    def test_runtime_step_with_constant_bounds_dispatches_to_runtime_lowering(self) -> None:
        lines = convert('OPENQASM 3.0; int stride = 2; int total = 0; '
                        'for int i in [1:stride:6] { total += i; }')
        self.assertIn('LET QASM2QCX_INT_0 := 1', lines)
        self.assertIn('LET QASM2QCX_INT_2 := STRIDE63', lines)
        self.assertIn('LET QASM2QCX_INT_1 := 6', lines)
        self.assertIn('ASSERT QASM2QCX_INT_2 \\= 0', lines)
        self.assertEqual(sum(line.startswith('LET TOTAL31 +=') for line in lines), 1)
        ForLoopControlLoweringTests.assert_resolved_jumps(lines)

    def test_capture_precedes_binding_and_uses_source_order(self) -> None:
        lines = convert('OPENQASM 3.0; int i = 2; int last = 5; '
                        'for int i in [i:i:last] {}')
        first = lines.index('LET QASM2QCX_INT_0 := I1')
        step = lines.index('LET QASM2QCX_INT_2 := I1')
        stop = lines.index('LET QASM2QCX_INT_1 := LAST15')
        assertion = lines.index('ASSERT QASM2QCX_INT_2 \\= 0')
        self.assertLess(first, step)
        self.assertLess(step, stop)
        self.assertLess(stop, assertion)

    def test_zero_runtime_step_is_checked_before_empty_range_entry(self) -> None:
        lines = convert('OPENQASM 3.0; int stride = 0; for int i in [5:stride:1] {}')
        assertion = next(index for index, line in enumerate(lines) if line.startswith('ASSERT '))
        first_jump = next(index for index, line in enumerate(lines) if line.startswith('JUMPIF '))
        self.assertLess(assertion, first_jump)

    def test_runtime_step_next_selects_sign_without_negating_step(self) -> None:
        lines = convert('OPENQASM 3.0; int stride = -2; for int i in [4:stride:-5] { continue; }')
        self.assertIn('JUMP QASM2QCX_LOOP_0_NEXT', lines)
        next_index = lines.index('@QASM2QCX_LOOP_0_NEXT')
        self.assertEqual(lines[next_index + 1],
                         'JUMPIF QASM2QCX_LOOP_0_CANDIDATE_NEGATIVE_NEXT QASM2QCX_INT_2 < 0')
        self.assertEqual(lines.count('LET QASM2QCX_INT_3 -= QASM2QCX_INT_2'), 2)
        self.assertEqual(lines.count('LET QASM2QCX_INT_0 += QASM2QCX_INT_2'), 2)
        ForLoopControlLoweringTests.assert_resolved_jumps(lines)

    def test_runtime_steps_in_empty_constant_outer_loops_are_not_evaluated(self) -> None:
        prefix = 'OPENQASM 3.0; int stride = 0; '
        self.assertEqual(convert(prefix + 'for int outer in [1:0] '
                                 '{ for int i in [0:stride:3] {} }'), convert(prefix))
        for step in ('true', 'missing', 'float(stride)', 'stride ** true'):
            with self.subTest(step=step), self.assertRaises(qasm2qcx.QASM2QCXError):
                convert(prefix + f'for int outer in [1:0] {{ for int i in [0:{step}:3] {{}} }}')

    def test_named_constant_step_still_rejects_zero_during_conversion(self) -> None:
        with self.assertRaisesRegex(qasm2qcx.InvalidLoopRangeException, 'step cannot be zero'):
            convert('OPENQASM 3.0; const int stride = 0; int last = -1; '
                    'for int i in [0:stride:last] {}')

    def test_casts_arithmetic_and_runtime_set_iterator_steps_are_supported(self) -> None:
        prefix = 'OPENQASM 3.0; int stride = 2; float f = 2.5; '
        for step in ('stride + 1', '-stride', 'int(f)', 'uint(f)', 'stride % 3'):
            with self.subTest(step=step):
                lines = convert(prefix + f'for int i in [0:{step}:5] {{}}')
                self.assertTrue(any(line.startswith('ASSERT ') for line in lines))
                ForLoopControlLoweringTests.assert_resolved_jumps(lines)
        lines = convert(prefix + 'for int s in {stride, -stride} '
                        '{ for int i in [0:s:5] { break; } }')
        self.assertEqual(sum(line.startswith('ASSERT ') for line in lines), 3)
        ForLoopControlLoweringTests.assert_resolved_jumps(lines)

    def test_step_capture_and_iterator_temporaries_are_released(self) -> None:
        program = qasm2qcx.openqasm3.parser.parse('OPENQASM 3.0; int stride = 2; '
            'for int i in [0:stride:5] { for int j in [i:stride:-5] { continue; } break; }')
        converter = qasm2qcx.QASM2QCXConverter(program)
        converter.visit(program)
        self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())
        self.assertEqual(converter._QASM2QCXConverter__loop_bindings, [])
        self.assertEqual(converter._QASM2QCXConverter__loop_contexts, [])

    def test_dynamic_advance_exhaustively_checks_small_signed_integer_model(self) -> None:
        checker = RuntimeStrideInfrastructureTests()
        lines = checker.advance_lines('STEP')
        minimum, maximum = -8, 7
        for step in range(minimum, maximum + 1):
            if step == 0:
                continue
            for iterator in range(minimum, maximum + 1):
                for stop in range(minimum, maximum + 1):
                    if (step > 0 and iterator > stop) or (step < 0 and iterator < stop):
                        continue
                    candidate = iterator + step
                    expected = candidate if (candidate <= stop if step > 0 else candidate >= stop) else None
                    self.assertEqual(checker.checked_advance(lines, iterator, stop, minimum, maximum, step),
                                     expected, msg=f'{iterator=}, {stop=}, {step=}')


class RuntimeStepRegressionTests(unittest.TestCase):
    def test_native_endpoint_guard_arithmetic_is_checked(self) -> None:
        checker = RuntimeStrideInfrastructureTests()
        lines = checker.advance_lines('STEP')
        minimum, maximum = qasm2qcx.QASM2QCXConverter.QCX_INT_MIN, qasm2qcx.QASM2QCXConverter.QCX_INT_MAX
        endpoints = (minimum, minimum + 1, -1, 0, 1, maximum - 1, maximum)
        for step in (minimum, -maximum, -3, -1, 1, 3, maximum):
            for iterator in endpoints:
                for stop in endpoints:
                    if (step > 0 and iterator > stop) or (step < 0 and iterator < stop):
                        continue
                    candidate = iterator + step
                    expected = candidate if (candidate <= stop if step > 0 else candidate >= stop) else None
                    self.assertEqual(checker.checked_advance(lines, iterator, stop, minimum, maximum, step),
                                     expected, msg=f'{iterator=}, {stop=}, {step=}')

    def test_body_and_assertion_are_generated_once_independent_of_values(self) -> None:
        source = ('OPENQASM 3.0; int stride = {step}; int last = {stop}; int total = 0; '
                  'for int i in [0:stride:last] {{ total += i; }}')
        reference = convert(source.format(step=1, stop=2))
        for step, stop in ((0, -1), (-3, -100), (2, 1000000)):
            lines = convert(source.format(step=step, stop=stop))
            normalized = lambda output: [line for line in output
                                         if not line.startswith(('LET STRIDE63 :=', 'LET LAST15 :='))]
            self.assertEqual(normalized(lines), normalized(reference))
            self.assertEqual(sum(line.startswith('ASSERT ') for line in lines), 1)
            self.assertEqual(sum(line.startswith('LET TOTAL31 +=') for line in lines), 1)

    def test_runtime_iterations_do_not_charge_compile_time_expansion_budget(self) -> None:
        with patch.object(qasm2qcx.QASM2QCXConverter, 'MAX_LOOP_ITERATIONS', 0):
            convert('OPENQASM 3.0; int stride = 1; for int i in [0:stride:1000000] {}')
            with self.assertRaises(qasm2qcx.InvalidLoopRangeException):
                convert('OPENQASM 3.0; int stride = 1; for int i in [0:stride:3] '
                        '{ for int j in {stride} {} }')

    def test_reserved_source_storage_is_not_reused_by_captures_or_guards(self) -> None:
        lines = convert('OPENQASM 3.0; int QASM2QCX_INT_ = 2; '
                        'for int i in [0:QASM2QCX_INT_:5] {}')
        self.assertEqual([line for line in lines if line.startswith('LET QASM2QCX_INT_0 ')],
                         ['LET QASM2QCX_INT_0 := 2'])
        self.assertIn('LET QASM2QCX_INT_3 := QASM2QCX_INT_0', lines)

    def test_capture_storage_stays_reserved_across_nested_expression_lowering(self) -> None:
        lines = convert('OPENQASM 3.0; int stride = 2; int total = 0; '
                        'for int i in [0:stride:5] { total += i % stride; '
                        'for int j in [i:stride:-5] { total += j % stride; } }')
        body = lines.index('@QASM2QCX_LOOP_0_BODY')
        next_label = lines.index('@QASM2QCX_LOOP_0_NEXT')
        assignments = [line for line in lines[body + 1:next_label] if line.startswith('LET ')]
        for reserved in ('QASM2QCX_INT_0', 'QASM2QCX_INT_1', 'QASM2QCX_INT_2'):
            self.assertFalse(any(line.split()[1] == reserved for line in assignments))
        declarations = [index for index, line in enumerate(lines) if line.startswith('VAR QASM2QCX_')]
        self.assertTrue(all(index < body for index in declarations))
        ForLoopControlLoweringTests.assert_resolved_jumps(lines)

    def test_iterator_scope_restores_constant_and_rejects_escape(self) -> None:
        prefix = 'OPENQASM 3.0; int stride = 2; int total = 0; '
        with self.assertRaises(qasm2qcx.NoVariableNameException):
            convert(prefix + 'for int i in [0:stride:5] {} total += i;')
        lines = convert(prefix + 'const int i = 2; for int i in [i:stride:5] { total += i; } '
                        'for int j in [0:i:4] { total += j; }')
        self.assertEqual(lines[-3:], ['LET TOTAL31 += 0', 'LET TOTAL31 += 2', 'LET TOTAL31 += 4'])

    def test_skipped_bodies_still_validate_syntax_and_read_only_iterators(self) -> None:
        prefix = 'OPENQASM 3.0; int stride = -2; int total = 0; qubit q; '
        for body in ('continue; i += 1;', 'break; total += 2 ** true;',
                     'if (false) { int local; }', 'while (false) { i = measure q; }'):
            with self.subTest(body=body), self.assertRaises(qasm2qcx.UnsupportedOpenQASMError):
                convert(prefix + 'for int i in [0:stride:5] { ' + body + ' }')

    def test_nested_transfer_labels_are_unique_and_target_nearest_loop(self) -> None:
        lines = convert('''OPENQASM 3.0; int stride = 2; int tick = 0;
            for int i in [0:stride:5] {
                for int j in [i:-stride:-5] { continue; break; }
                while (tick < 1) { tick += 1; continue; break; }
                for int k in {stride} { continue; break; }
                continue; break;
            }''')
        for target in ('QASM2QCX_LOOP_0_NEXT', 'QASM2QCX_LOOP_0_END',
                       'QASM2QCX_LOOP_1_NEXT', 'QASM2QCX_LOOP_1_END',
                       'QASM2QCX_LOOP_2_CONDITION', 'QASM2QCX_LOOP_2_END',
                       'QASM2QCX_LOOP_3_NEXT_0', 'QASM2QCX_LOOP_3_END'):
            self.assertIn(f'JUMP {target}', lines)
        ForLoopControlLoweringTests.assert_resolved_jumps(lines)

    def test_captures_and_loop_contexts_are_released_after_emission_failure(self) -> None:
        program = qasm2qcx.openqasm3.parser.parse(
            'OPENQASM 3.0; int stride = 2; for int i in [0:stride:5] { continue; }')
        converter = qasm2qcx.QASM2QCXConverter(program)
        with patch.object(converter, '_QASM2QCXConverter__emit_runtime_range_next',
                          side_effect=RuntimeError('guard emission failed')):
            with self.assertRaisesRegex(RuntimeError, 'guard emission failed'):
                converter.visit(program)
        self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())
        self.assertEqual(converter._QASM2QCXConverter__loop_bindings, [])
        self.assertEqual(converter._QASM2QCXConverter__loop_contexts, [])

    def test_runtime_step_conversion_does_not_charge_a_value_dependent_budget(self) -> None:
        source = 'OPENQASM 3.0; int stride = 0; for int i in [0:stride:1000000] {}'
        with patch.object(qasm2qcx.QASM2QCXConverter, 'MAX_LOOP_ITERATIONS', 0), \
                patch('builtins.open', mock_open(read_data=source)), patch('builtins.print') as output:
            qasm2qcx.main(['runtime-step.qasm'])
        self.assertTrue(any('ASSERT ' in str(call) for call in output.call_args_list))


class RuntimeStrideLoweringTests(unittest.TestCase):
    def test_positive_nonunit_step_uses_correct_entry_and_safe_advance_guards(self) -> None:
        lines = convert('OPENQASM 3.0; int last = 6; for int i in [1:2:last] { continue; }')
        self.assertIn('JUMPIF QASM2QCX_LOOP_0_END QASM2QCX_INT_0 > QASM2QCX_INT_1', lines)
        next_index = lines.index('@QASM2QCX_LOOP_0_NEXT')
        self.assertEqual(lines[next_index + 1:], [
            'JUMPIF QASM2QCX_LOOP_0_CANDIDATE QASM2QCX_INT_1 < 0',
            'LET QASM2QCX_INT_2 := QASM2QCX_INT_1', 'LET QASM2QCX_INT_2 -= 2',
            'JUMPIF QASM2QCX_LOOP_0_END QASM2QCX_INT_0 > QASM2QCX_INT_2',
            'LET QASM2QCX_INT_0 += 2', 'JUMP QASM2QCX_LOOP_0_BODY',
            '@QASM2QCX_LOOP_0_CANDIDATE', 'LET QASM2QCX_INT_2 := QASM2QCX_INT_0',
            'LET QASM2QCX_INT_2 += 2',
            'JUMPIF QASM2QCX_LOOP_0_END QASM2QCX_INT_2 > QASM2QCX_INT_1',
            'LET QASM2QCX_INT_0 := QASM2QCX_INT_2', 'JUMP QASM2QCX_LOOP_0_BODY',
            '@QASM2QCX_LOOP_0_END'])
        self.assertIn('JUMP QASM2QCX_LOOP_0_NEXT', lines)
        ForLoopControlLoweringTests.assert_resolved_jumps(lines)

    def test_negative_nonunit_step_is_used_without_negation(self) -> None:
        for step in (-3, qasm2qcx.QASM2QCXConverter.QCX_INT_MIN):
            with self.subTest(step=step):
                lines = convert(f'OPENQASM 3.0; int last = -5; for int i in [4:{step}:last] {{}}')
                self.assertIn('JUMPIF QASM2QCX_LOOP_0_END QASM2QCX_INT_0 < QASM2QCX_INT_1', lines)
                self.assertIn('JUMPIF QASM2QCX_LOOP_0_CANDIDATE QASM2QCX_INT_1 >= 0', lines)
                self.assertIn(f'LET QASM2QCX_INT_2 -= {step}', lines)
                self.assertIn(f'LET QASM2QCX_INT_0 += {step}', lines)
                self.assertIn('JUMPIF QASM2QCX_LOOP_0_END QASM2QCX_INT_0 < QASM2QCX_INT_2', lines)
                ForLoopControlLoweringTests.assert_resolved_jumps(lines)

    def test_named_casted_and_outer_constant_steps_are_supported(self) -> None:
        prefix = 'OPENQASM 3.0; const int stride = 2; int last = 5; '
        for step in ('stride', '-stride', 'int(2.5)', 'uint(3.5)'):
            with self.subTest(step=step):
                ForLoopControlLoweringTests.assert_resolved_jumps(
                    convert(prefix + f'for int i in [0:{step}:last] {{}}'))
        lines = convert(prefix + 'for int stride in {2, -3} '
                        '{ for int i in [0:stride:last] {} }')
        self.assertIn('LET QASM2QCX_INT_0 += 2', lines)
        self.assertIn('LET QASM2QCX_INT_0 += -3', lines)
        ForLoopControlLoweringTests.assert_resolved_jumps(lines)

    def test_invalid_steps_and_unrepresentable_constants_remain_rejected(self) -> None:
        for step in ('missing', '1.5', '0', str(qasm2qcx.QASM2QCXConverter.QCX_INT_MAX + 1),
                     str(qasm2qcx.QASM2QCXConverter.QCX_INT_MIN - 1)):
            with self.subTest(step=step):
                with self.assertRaises(qasm2qcx.QASM2QCXError):
                    convert('OPENQASM 3.0; int dynamic = 2; int last = 5; '
                            f'for int i in [0:{step}:last] {{}}')
        lines = convert('OPENQASM 3.0; int last = 5; for int stride in {last} '
                        '{ for int i in [0:stride:last] {} }')
        self.assertTrue(any(line.startswith('ASSERT ') for line in lines))
        ForLoopControlLoweringTests.assert_resolved_jumps(lines)

    def test_empty_constant_outer_accepts_nonunit_header_without_evaluation(self) -> None:
        prefix = 'OPENQASM 3.0; int last = 5; '
        self.assertEqual(convert(prefix + 'for int i in [1:0] '
                                 '{ for int j in [i:3:last + i] {} }'), convert(prefix))
        with self.assertRaises(qasm2qcx.InvalidLoopRangeException):
            convert(prefix + 'for int i in [1:0] { for int j in [0:0:last] {} }')

    def test_nested_nonunit_ranges_and_runtime_sets_have_unique_resolved_labels(self) -> None:
        lines = convert('''OPENQASM 3.0; int last = 5; int total = 0;
            for int i in [0:2:last] {
                for int j in [i:-3:-5] { total += i + j; continue; }
                for int j in {i, i + 1} { if (j > 0) { break; } }
                break;
            }''')
        labels = [line for line in lines if line.startswith('@')]
        self.assertEqual(len(labels), len(set(labels)))
        self.assertIn('@QASM2QCX_LOOP_1_CANDIDATE', lines)
        self.assertIn('JUMP QASM2QCX_LOOP_0_END', lines)
        ForLoopControlLoweringTests.assert_resolved_jumps(lines)

    def test_guard_scratch_is_hoisted_but_advance_calculations_stay_after_body(self) -> None:
        lines = convert('''OPENQASM 3.0; int last = 5; int total = 0;
            if (false) { for int i in [0:2:last] { total += i % 3; } }
            total += last + 1;''')
        branch = lines.index('@QASM2QCX_IF_0')
        self.assertTrue(all(index < branch for index, line in enumerate(lines)
                            if line.startswith('VAR QASM2QCX_')))
        next_index = lines.index('@QASM2QCX_LOOP_0_NEXT')
        self.assertGreater(next(index for index, line in enumerate(lines) if line.endswith(' -= 2')),
                           next_index)
        ForLoopControlLoweringTests.assert_resolved_jumps(lines)


class RuntimeStrideRegressionTests(unittest.TestCase):
    def test_nonunit_range_iterations_do_not_consume_expansion_budget(self) -> None:
        with patch.object(qasm2qcx.QASM2QCXConverter, 'MAX_LOOP_ITERATIONS', 0):
            convert('OPENQASM 3.0; int last = 1000000; for int i in [0:3:last] {}')
            with self.assertRaises(qasm2qcx.InvalidLoopRangeException):
                convert('OPENQASM 3.0; int last = 5; for int i in [0:3:last] '
                        '{ for int j in {i} {} }')

    def test_guard_and_body_are_generated_once_regardless_of_runtime_bound(self) -> None:
        source = ('OPENQASM 3.0; int last = {last}; int total = 0; '
                  'for int i in [0:3:last] {{ total += i; }}')
        small = convert(source.format(last=1))
        large = convert(source.format(last=qasm2qcx.QASM2QCXConverter.QCX_INT_MAX))
        self.assertEqual([line for line in small if line != 'LET LAST15 := 1'],
                         [line for line in large if line !=
                          f'LET LAST15 := {qasm2qcx.QASM2QCXConverter.QCX_INT_MAX}'])
        self.assertEqual(large.count('LET TOTAL31 += QASM2QCX_INT_0'), 1)
        self.assertEqual(large.count('@QASM2QCX_LOOP_0_CANDIDATE'), 1)

    def test_reserved_source_storage_is_not_reused_by_stride_guard(self) -> None:
        lines = convert('''OPENQASM 3.0; int QASM2QCX_INT_ = 9; int last = 5;
            for int i in [0:2:last] {}''')
        self.assertEqual([line for line in lines if line.startswith('LET QASM2QCX_INT_0 ')],
                         ['LET QASM2QCX_INT_0 := 9'])

    def test_runtime_iterator_scope_does_not_escape_and_restores_constant_binding(self) -> None:
        prefix = 'OPENQASM 3.0; int last = 6; int total = 0; '
        with self.assertRaises(qasm2qcx.NoVariableNameException):
            convert(prefix + 'for int i in [0:2:last] {} total += i;')
        lines = convert(prefix + 'const int i = 2; for int i in [i:2:last] { total += i; } '
                        'for int j in [0:i:4] { total += j; }')
        self.assertEqual(lines[-3:], ['LET TOTAL31 += 0', 'LET TOTAL31 += 2', 'LET TOTAL31 += 4'])

    def test_empty_outer_validation_defers_iterator_dependent_step_values(self) -> None:
        prefix = 'OPENQASM 3.0; int last = 5; '
        self.assertEqual(convert(prefix + 'for int i in [1:0] '
                                 '{ for int j in [0:i:last] {} }'), convert(prefix))
        # The same source must reject a zero step once an actual value exists.
        with self.assertRaises(qasm2qcx.InvalidLoopRangeException):
            convert(prefix + 'for int i in {0} { for int j in [0:i:last] {} }')

    def test_skipped_nonunit_bodies_retain_read_only_and_syntax_checks(self) -> None:
        for body in ('continue; i += 2;', 'if (false) { int local; }',
                     'break; total += 2 ** true;', 'while (false) { i = measure q; }'):
            with self.subTest(body=body):
                with self.assertRaises(qasm2qcx.UnsupportedOpenQASMError):
                    convert('OPENQASM 3.0; int last = -1; int total = 0; qubit q; '
                            'for int i in [0:2:last] { ' + body + ' }')

    def test_cli_invalid_stride_reports_error_without_partial_output(self) -> None:
        for step in ('0', 'missing', 'true', str(qasm2qcx.QASM2QCXConverter.QCX_INT_MIN - 1)):
            with self.subTest(step=step):
                source = f'OPENQASM 3.0; int dynamic = 2; int last = 5; for int i in [0:{step}:last] {{}}'
                with patch('builtins.open', mock_open(read_data=source)), patch('builtins.print') as output:
                    with self.assertRaises(SystemExit) as error:
                        qasm2qcx.main(['stride.qasm'])
                self.assertTrue(str(error.exception).startswith('qasm2qcx.py:'))
                output.assert_not_called()

    def test_resources_are_released_after_successful_nested_stride_conversion(self) -> None:
        program = qasm2qcx.openqasm3.parser.parse('''OPENQASM 3.0; int last = 5;
            for int i in [0:2:last] { for int j in [i:-3:-5] { continue; } break; }''')
        converter = qasm2qcx.QASM2QCXConverter(program)
        converter.visit(program)
        self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())
        self.assertEqual(converter._QASM2QCXConverter__loop_bindings, [])
        self.assertEqual(converter._QASM2QCXConverter__loop_contexts, [])
        self.assertEqual(converter._QASM2QCXConverter__expanded_loop_depth, 0)


class RuntimeForRegressionTests(unittest.TestCase):
    def test_constant_complex_numeric_casts_preserve_unrolled_ranges(self) -> None:
        prefix = 'OPENQASM 3.0; const complex value = 2.0 + 3.0im; int total = 0; '
        for bounds, values in (
                ('[0:int(1.0im)]', (0,)),
                ('[int(value):int(value) + 1]', (2, 3)),
                ('[0:int(float(value))]', (0, 1, 2)),
                ('[0:uint(value)]', (0, 1, 2)),
                ('[int(-value):-1:0]', ()),
                ('[0:int(value):4]', (0, 2, 4))):
            with self.subTest(bounds=bounds):
                self.assertEqual(convert(prefix + f'for int i in {bounds} {{ total += i; }}'),
                                 convert(prefix + ''.join(f'total += {i};' for i in values)))

    def test_constant_complex_casts_work_in_nested_and_runtime_bounds(self) -> None:
        lines = convert('''OPENQASM 3.0; int total = 0;
            for int i in {1, 2} {
                for int j in [0:int(complex(i) + 1.0im)] { total += j; }
            }''')
        self.assertEqual(lines, convert('OPENQASM 3.0; int total = 0; '
                                       'total += 0; total += 1; '
                                       'total += 0; total += 1; total += 2;'))
        lines = convert('OPENQASM 3.0; int limit = 2; '
                        'for int i in [int(1.0im):limit] {}')
        self.assertIn('LET QASM2QCX_INT_0 := 0', lines)
        ForLoopControlLoweringTests.assert_resolved_jumps(lines)
        # Independent bound validation must also accept these casts when an
        # empty outer loop never supplies an iteration value.
        prefix = 'OPENQASM 3.0; int limit = 2; '
        self.assertEqual(convert(prefix + 'for int i in [1:0] '
                                 '{ for int j in [int(1.0im):limit] {} }'), convert(prefix))

    def test_runtime_complex_and_constant_complex_boolean_casts_still_rejected(self) -> None:
        for prefix, bound in (
                ('complex value = 1.0im;', 'int(value)'),
                ('complex value = 1.0im;', 'int(float(value))'),
                ('', 'int(bool(1.0im))'), ('', 'int(bit(1.0im))')):
            with self.subTest(bound=bound):
                with self.assertRaises(qasm2qcx.QASM2QCXError):
                    convert(f'OPENQASM 3.0; {prefix} for int i in [0:{bound}] {{}}')
        with self.assertRaises(qasm2qcx.QASM2QCXError):
            convert('OPENQASM 3.0; int limit = 2; '
                    'for int i in [0:limit] { for int j in [0:int(complex(i))] {} }')

    def test_range_expression_analysis_preserves_types_dependencies_and_state(self) -> None:
        program = qasm2qcx.openqasm3.parser.parse('''OPENQASM 3.0;
            const int count = 2; int limit = 3; float fraction = 1.5;
            bool ready = true; bit flag = 1; bit[2] flags = "01";''')
        converter = qasm2qcx.QASM2QCXConverter(program)
        converter.visit(program)
        lines = converter._QASM2QCXConverter__qcx_lines.copy()
        value = (converter._QASM2QCXConverter__value,
                 converter._QASM2QCXConverter__value_type,
                 converter._QASM2QCXConverter__value_kind)
        for source, value_type, runtime in (
                ('count + 1', qasm2qcx.ValueType.INT, False),
                ('-limit', qasm2qcx.ValueType.INT, True),
                ('limit + fraction', qasm2qcx.ValueType.FLOAT, True),
                ('1.0im * limit', qasm2qcx.ValueType.COMPLEX, True),
                ('int(fraction)', qasm2qcx.ValueType.INT, True),
                ('int(flags[0])', qasm2qcx.ValueType.INT, True),
                ('int(!ready || flag)', qasm2qcx.ValueType.INT, True),
                ('int(limit >= count)', qasm2qcx.ValueType.INT, True),
                ('int(pi)', qasm2qcx.ValueType.INT, False),
                ('int(1.0im)', qasm2qcx.ValueType.INT, False),
                ('int(float(2.0 + 1.0im))', qasm2qcx.ValueType.INT, False),
                ('limit % count', qasm2qcx.ValueType.INT, True)):
            with self.subTest(expression=source):
                expression = qasm2qcx.openqasm3.parser.parse(
                    f'OPENQASM 3.0; int result = {source};').statements[0].init_expression
                self.assertEqual(converter._QASM2QCXConverter__loop_expression_info(expression),
                                 (value_type, runtime))
                self.assertEqual(converter._QASM2QCXConverter__qcx_lines, lines)
                self.assertEqual((converter._QASM2QCXConverter__value,
                                  converter._QASM2QCXConverter__value_type,
                                  converter._QASM2QCXConverter__value_kind), value)
                self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())

    def test_constant_unit_step_expressions_and_outer_constant_iterators(self) -> None:
        for step in ('1 + 0', 'int(1.5)', 'stride', '-stride'):
            with self.subTest(step=step):
                lines = convert('OPENQASM 3.0; const int stride = 1; int limit = 2; '
                                f'for int i in [0:{step}:limit] {{}}')
                self.assertIn('@QASM2QCX_LOOP_0_NEXT', lines)
                ForLoopControlLoweringTests.assert_resolved_jumps(lines)
        lines = convert('OPENQASM 3.0; int limit = 2; '
                        'for int stride in {1, -1} { for int i in [0:stride:limit] {} }')
        self.assertTrue(any(line.endswith(' += 1') for line in lines))
        self.assertTrue(any(line.endswith(' -= 1') for line in lines))
        ForLoopControlLoweringTests.assert_resolved_jumps(lines)

    def test_runtime_iterator_can_supply_a_captured_runtime_step(self) -> None:
        lines = convert('OPENQASM 3.0; int limit = 2; '
                        'for int i in [1:limit] { for int j in [0:i:limit] {} }')
        self.assertTrue(any(line.startswith('ASSERT ') for line in lines))
        ForLoopControlLoweringTests.assert_resolved_jumps(lines)

    def test_runtime_bound_magnitude_does_not_expand_the_body(self) -> None:
        source = ('OPENQASM 3.0; int limit = {limit}; int total = 0; '
                  'for int i in [0:limit] {{ total += i; }}')
        small = convert(source.format(limit=1))
        large = convert(source.format(limit=1000000))
        self.assertEqual([line for line in small if line != 'LET LIMIT31 := 1'],
                         [line for line in large if line != 'LET LIMIT31 := 1000000'])
        self.assertEqual(sum(line.startswith('LET TOTAL31 +=') for line in large), 1)

    def test_runtime_iterator_does_not_escape_scope(self) -> None:
        for suffix in ('total += i;', 'for int j in [0:i] {}'):
            with self.subTest(suffix=suffix):
                with self.assertRaises(qasm2qcx.NoVariableNameException):
                    convert('OPENQASM 3.0; int limit = 2; int total = 0; '
                            'for int i in [0:limit] {} ' + suffix)

    def test_mixed_loop_and_branch_labels_are_unique_and_resolved(self) -> None:
        lines = convert('''OPENQASM 3.0; int limit = 2; bool ready = true;
            for int i in {1, 1} {
                for int j in [0:limit] {
                    while (ready) { if (j == i) { break; } else { continue; } }
                    for int k in [0:1] { if (k == 0) { continue; } break; }
                    if (j == 1) { break; } continue;
                }
            }
            for int i in [0:limit] { break; }''')
        labels = [line[1:] for line in lines if line.startswith('@')]
        self.assertEqual(len(labels), len(set(labels)))
        ForLoopControlLoweringTests.assert_resolved_jumps(lines)

    def test_runtime_only_code_avoids_expansion_budgets_but_mixed_code_does_not(self) -> None:
        source = 'OPENQASM 3.0; int limit = 2; for int i in [0:limit] { limit += 0; }'
        for budget in ('MAX_EXPANDED_LOOP_STATEMENTS', 'MAX_LOOP_OUTPUT_LINES'):
            with self.subTest(budget=budget):
                with patch.object(qasm2qcx.QASM2QCXConverter, budget, 0):
                    convert(source)
                    with self.assertRaises(qasm2qcx.InvalidLoopRangeException):
                        convert('OPENQASM 3.0; int limit = 2; for int k in {1} '
                                '{ for int i in [0:limit] { limit += 0; } }')

    def test_skipped_runtime_bodies_still_reject_unsupported_syntax(self) -> None:
        for body in ('total += 2 ** true;', 'total = sin(1.0);', 'total <<= true;',
                     'if (false) { int local; }', 'while (false) { i += 1; }'):
            with self.subTest(body=body):
                with self.assertRaises(qasm2qcx.UnsupportedOpenQASMError):
                    convert('OPENQASM 3.0; int limit = -1; int total = 0; '
                            'for int i in [0:limit] { ' + body + ' }')

    def test_invalid_range_operand_categories_are_rejected(self) -> None:
        for bound in ('flags', 'int(bits)', 'int(fraction % 2)',
                      'int(!limit)', 'int(ready && limit)', 'int(1.0im == 2.0im)',
                      'int(bit(limit))', 'int(bit[2](ready))'):
            with self.subTest(bound=bound):
                with self.assertRaises(qasm2qcx.QASM2QCXError):
                    convert('OPENQASM 3.0; int limit = 2; float fraction = 1.5; '
                            'bool ready = true; bit[2] flags = "01"; '
                            f'for int i in [0:{bound}] {{}}')


class SharedLoopInfrastructureTests(unittest.TestCase):
    @staticmethod
    def validate_while_body(body: str, iterators: set[str] | None = None,
                            condition: str = 'false'):
        program = qasm2qcx.openqasm3.parser.parse(
            'OPENQASM 3.0; const int n = 2; const bool ready = false;')
        converter = qasm2qcx.QASM2QCXConverter(program)
        converter.visit(program)
        loop = qasm2qcx.openqasm3.parser.parse(
            f'OPENQASM 3.0; while ({condition}) {{ {body} }}').statements[0]
        converter._QASM2QCXConverter__validate_loop_body(
            [loop], iterators or set())
        converter._QASM2QCXConverter__validate_loop_syntax(loop)
        return converter

    def test_shared_validation_accepts_nested_loop_structure_without_emitting(self) -> None:
        converter = self.validate_while_body('''
            if (ready) { continue; } else { break; }
            while (ready) { reset q; }
            for int j in [0:n] { sum += j; }
            for int k in [0:i] { sum += k; }
        ''', {'i'})
        self.assertEqual(converter._QASM2QCXConverter__qcx_lines, ['QUBITS 0'])
        self.assertEqual(converter._QASM2QCXConverter__loop_iterations, 0)
        self.assertEqual(converter._QASM2QCXConverter__expanded_loop_statements, 0)
        self.assertEqual(converter._QASM2QCXConverter__loop_contexts, [])
        self.assertEqual(converter._QASM2QCXConverter__loop_bindings, [])

    def test_shared_validation_rejects_unsupported_false_loop_bodies(self) -> None:
        for body in ('int local;', 'const int local = 1;', 'qubit local;',
                     'sum += 2 ** true;', 'sum = sin(1.0);',
                     'negctrl(0) @ x q;', 'while (false) { int local; }',
                     'if (false) { int local; }', 'for int j in [1:0] { int local; }'):
            with self.subTest(body=body):
                with self.assertRaises((qasm2qcx.UnsupportedOpenQASMError,
                                        qasm2qcx.openqasm3.parser.QASM3ParsingError)):
                    self.validate_while_body(body)

    def test_active_for_iterators_remain_read_only_in_nested_while_bodies(self) -> None:
        for body in ('i = 1;', 'i = measure q;', 'while (false) { i += 1; }',
                     'if (false) { i = 1; }', 'for int j in [1:0] { i = 1; }'):
            with self.subTest(body=body):
                with self.assertRaisesRegex(qasm2qcx.UnsupportedOpenQASMError,
                                            'assignment to a for-loop iterator'):
                    self.validate_while_body(body, {'i'})
        # A while loop itself introduces no read-only iterator.
        self.validate_while_body('i += 1;')

    def test_nested_for_headers_and_independent_bounds_still_validated(self) -> None:
        for body, error in (
                ('for uint j in [0:1] {}', qasm2qcx.UnsupportedOpenQASMError),
                ('for int j in [0:0:1] {}', qasm2qcx.InvalidLoopRangeException),
                ('for int j in [0:missing] {}', qasm2qcx.NoVariableNameException)):
            with self.subTest(body=body):
                with self.assertRaises(error):
                    self.validate_while_body(body)

    def test_while_condition_syntax_is_checked_without_constant_evaluation(self) -> None:
        for condition in ('sin(1.0) > 0', '(2 ** true) > 0'):
            with self.subTest(condition=condition):
                with self.assertRaises(qasm2qcx.UnsupportedOpenQASMError):
                    self.validate_while_body('', condition=condition)
        self.validate_while_body('sum += value / 0;', condition='(1 / 0) > 0')
        self.validate_while_body('', condition='~1 == 0')

    def test_context_targets_are_independent_of_iterator_bindings_and_restored(self) -> None:
        converter = qasm2qcx.QASM2QCXConverter(
            qasm2qcx.openqasm3.parser.parse('OPENQASM 3.0;'))
        converter._QASM2QCXConverter__is_initialization_process = False
        context = converter._QASM2QCXConverter__loop_context
        with context('OUTER_END', 'OUTER_NEXT') as outer:
            with self.assertRaisesRegex(RuntimeError, 'body failure'):
                with context('INNER_END', 'INNER_CONDITION') as inner:
                    self.assertEqual(converter._QASM2QCXConverter__loop_contexts, [outer, inner])
                    self.assertEqual(converter._QASM2QCXConverter__loop_bindings, [])
                    converter.visit(qasm2qcx.ast.ContinueStatement())
                    converter.visit(qasm2qcx.ast.BreakStatement())
                    raise RuntimeError('body failure')
            self.assertEqual(converter._QASM2QCXConverter__loop_contexts, [outer])
            converter.visit(qasm2qcx.ast.ContinueStatement())
        self.assertEqual(converter._QASM2QCXConverter__loop_contexts, [])
        self.assertEqual(converter._QASM2QCXConverter__qcx_lines[-3:],
                         ['JUMP INNER_CONDITION', 'JUMP INNER_END', 'JUMP OUTER_NEXT'])

    def test_while_conversion_accepts_empty_and_transfer_bodies(self) -> None:
        for source in ('while (false) {}', 'while (true) { break; }',
                       'for int i in [1:0] { while (false) {} }',
                       'for int i in [0:0] { while (false) {} }'):
            with self.subTest(source=source):
                ForLoopControlLoweringTests.assert_resolved_jumps(
                    convert('OPENQASM 3.0; ' + source))


class WhileLoopLoweringTests(unittest.TestCase):
    def test_runtime_condition_and_back_edge(self) -> None:
        lines = convert('OPENQASM 3.0; int n = 0; while (n < 3) { n += 1; }')
        self.assertEqual(lines, [
            'QUBITS 0', 'VAR N1 INT', 'LET N1 := 0',
            '@QASM2QCX_LOOP_0_CONDITION', 'JUMPIF QASM2QCX_LOOP_0_BODY N1 < 3',
            'JUMP QASM2QCX_LOOP_0_END', '@QASM2QCX_LOOP_0_BODY',
            'LET N1 += 1', 'JUMP QASM2QCX_LOOP_0_CONDITION', '@QASM2QCX_LOOP_0_END'])
        ForLoopControlLoweringTests.assert_resolved_jumps(lines)

    def test_literal_conditions_do_not_prune_body_or_unroll(self) -> None:
        for condition, destination in (('true', 'BODY'), ('false', 'END')):
            with self.subTest(condition=condition):
                lines = convert(f'OPENQASM 3.0; int n = 0; while ({condition}) {{ n += 1; break; }}')
                self.assertIn(f'JUMP QASM2QCX_LOOP_0_{destination}', lines)
                self.assertEqual(lines.count('LET N1 += 1'), 1)
                ForLoopControlLoweringTests.assert_resolved_jumps(lines)

    def test_break_and_continue_targets(self) -> None:
        lines = convert('''OPENQASM 3.0; int n = 0;
            while (n < 5) { n += 1; if (n == 2) { continue; } break; }
            n += 10;''')
        self.assertEqual(lines.count('JUMP QASM2QCX_LOOP_0_CONDITION'), 2)
        self.assertEqual(lines.count('JUMP QASM2QCX_LOOP_0_END'), 2)
        self.assertEqual(lines[-2:], ['@QASM2QCX_LOOP_0_END', 'LET N1 += 10'])
        ForLoopControlLoweringTests.assert_resolved_jumps(lines)

    def test_nested_and_mixed_loops_have_unique_nearest_targets(self) -> None:
        for source in (
                'while (true) { while (true) { continue; break; } break; }',
                'while (true) { for int i in [0:0] { continue; break; } break; }',
                'for int i in [0:0] { while (true) { continue; break; } continue; }'):
            with self.subTest(source=source):
                lines = convert('OPENQASM 3.0; ' + source)
                ForLoopControlLoweringTests.assert_resolved_jumps(lines)
                self.assertIn('JUMP QASM2QCX_LOOP_1_END', lines)
                if source.startswith('for'):
                    self.assertIn('JUMP QASM2QCX_LOOP_0_NEXT_0', lines)
                    self.assertIn('JUMP QASM2QCX_LOOP_1_CONDITION', lines)
                else:
                    self.assertIn('JUMP QASM2QCX_LOOP_0_END', lines)

    def test_inner_while_transfers_add_no_unused_for_labels(self) -> None:
        lines = convert('OPENQASM 3.0; for int i in [0:1] { while (false) { break; } }')
        self.assertFalse(any('QASM2QCX_LOOP_0_' in line for line in lines))
        self.assertIn('@QASM2QCX_LOOP_1_CONDITION', lines)
        self.assertIn('@QASM2QCX_LOOP_2_CONDITION', lines)
        ForLoopControlLoweringTests.assert_resolved_jumps(lines)

    def test_condition_calculations_stay_after_condition_label(self) -> None:
        lines = convert('''OPENQASM 3.0; int n = 5;
            while (n % 3 != 0) { n -= 1; }''')
        condition = lines.index('@QASM2QCX_LOOP_0_CONDITION')
        division = next(index for index, line in enumerate(lines) if ' /= ' in line)
        self.assertGreater(division, condition)
        self.assertTrue(all(index < condition for index, line in enumerate(lines)
                            if line.startswith('VAR QASM2QCX_')))
        ForLoopControlLoweringTests.assert_resolved_jumps(lines)

    def test_body_temporary_storage_is_safe_after_zero_iterations(self) -> None:
        lines = convert('''OPENQASM 3.0; int value = 7; int total = 0;
            while (false) { total += value % 0; }
            total += value + 1;''')
        condition = lines.index('@QASM2QCX_LOOP_0_CONDITION')
        self.assertTrue(all(index < condition for index, line in enumerate(lines)
                            if line.startswith('VAR QASM2QCX_')))
        self.assertGreater(next(index for index, line in enumerate(lines) if ' /= ' in line), condition)

    def test_unsupported_body_and_condition_rejected_even_when_false(self) -> None:
        for source in ('while (false) { int local; }',
                       'while (false) { reset missing; }',
                       'while (false) { int x = 2 ** 3; }',
                       'while (false) { delay[1ns] q; }',
                       'while (1) {}', 'while (1.0) {}',
                       'while (sin(1.0) > 0) {}'):
            with self.subTest(source=source):
                with self.assertRaises(qasm2qcx.QASM2QCXError):
                    convert('OPENQASM 3.0; ' + source)

    def test_active_for_iterators_are_read_only_inside_while(self) -> None:
        for bounds in ('[0:1]', '[1:0]'):
            with self.subTest(bounds=bounds):
                with self.assertRaisesRegex(qasm2qcx.UnsupportedOpenQASMError, 'for-loop iterator'):
                    convert(f'OPENQASM 3.0; for int i in {bounds} {{ while (false) {{ i += 1; }} }}')

    def test_context_restored_after_body_or_condition_error(self) -> None:
        converter = qasm2qcx.QASM2QCXConverter(qasm2qcx.openqasm3.parser.parse('OPENQASM 3.0;'))
        for source in ('while (true) { reset missing; }', 'while (missing) {}'):
            with self.subTest(source=source):
                loop = qasm2qcx.openqasm3.parser.parse('OPENQASM 3.0; ' + source).statements[0]
                with self.assertRaises(qasm2qcx.QASM2QCXError):
                    converter.visit(loop)
                self.assertEqual(converter._QASM2QCXConverter__loop_contexts, [])
                self.assertEqual(converter._QASM2QCXConverter__loop_bindings, [])

    def test_while_does_not_charge_runtime_iterations_to_expansion_budget(self) -> None:
        with patch.object(qasm2qcx.QASM2QCXConverter, 'MAX_LOOP_ITERATIONS', 0):
            ForLoopControlLoweringTests.assert_resolved_jumps(convert('OPENQASM 3.0; while (true) {}'))
            with self.assertRaises(qasm2qcx.InvalidLoopRangeException):
                convert('OPENQASM 3.0; while (false) { for int i in [0:0] {} }')

    def test_while_output_and_body_count_toward_enclosing_for_limits(self) -> None:
        source = 'OPENQASM 3.0; for int i in [0:0] { while (true) { break; } }'
        for attribute, limit in (('MAX_LOOP_OUTPUT_LINES', 5), ('MAX_EXPANDED_LOOP_STATEMENTS', 1)):
            with self.subTest(attribute=attribute):
                with patch.object(qasm2qcx.QASM2QCXConverter, attribute, limit):
                    with self.assertRaises(qasm2qcx.InvalidLoopRangeException):
                        convert(source)


class WhileLoopRegressionTests(unittest.TestCase):
    def test_supported_condition_forms_produce_resolved_control_flow(self) -> None:
        prefix = '''OPENQASM 3.0; bool ready = false; bit flag = 0;
            bit[2] flags = "00"; int n = 0; float f = 0.0;'''
        for condition in ('ready', 'flag', 'flags[1]', 'bool(n)', 'bool(f)',
                          'n < f', '1 < n', '!ready', 'ready || flag && !flags[0]',
                          'bool(n + 1) && n % 3 == 0'):
            with self.subTest(condition=condition):
                ForLoopControlLoweringTests.assert_resolved_jumps(
                    convert(prefix + f'while ({condition}) {{ break; }}'))

    def test_single_statement_and_braced_body_are_equivalent(self) -> None:
        prefix = 'OPENQASM 3.0; int n = 0; '
        self.assertEqual(convert(prefix + 'while (n < 2) n += 1;'),
                         convert(prefix + 'while (n < 2) { n += 1; }'))

    def test_sequential_mixed_and_expanded_instances_have_unique_labels(self) -> None:
        lines = convert('''OPENQASM 3.0; int n = 0;
            while (false) { while (true) { break; } }
            for int i in [0:1] { while (n < i) { n += 1; continue; } }
            while (true) { break; }''')
        labels = [line for line in lines if line.endswith('_CONDITION') and line.startswith('@')]
        self.assertEqual(labels, [f'@QASM2QCX_LOOP_{index}_CONDITION' for index in (0, 1, 3, 4, 5)])
        ForLoopControlLoweringTests.assert_resolved_jumps(lines)

    def test_measurement_and_following_quantum_operations_stay_in_body(self) -> None:
        for transfer in ('break', 'continue'):
            with self.subTest(transfer=transfer):
                lines = convert('''OPENQASM 3.0; include "stdgates.inc";
                    qubit q; qubit r; bit outcome = 0;
                    while (!outcome) {
                        reset q; outcome = measure q;
                        if (outcome) { ''' + transfer + '''; }
                        x r;
                    }''')
                body = lines.index('@QASM2QCX_LOOP_0_BODY')
                destination = 'END' if transfer == 'break' else 'CONDITION'
                jump = lines.index(f'JUMP QASM2QCX_LOOP_0_{destination}', body)
                self.assertLess(body, lines.index('RESET 0'))
                self.assertLess(lines.index('M 0'), jump)
                self.assertLess(lines.index('LET OUTCOME127 := :OUTCOME'), jump)
                self.assertGreater(lines.index('X 1'), jump)
                ForLoopControlLoweringTests.assert_resolved_jumps(lines)

    def test_mixed_temporary_storage_precedes_loop_but_computations_do_not(self) -> None:
        lines = convert('''OPENQASM 3.0; int n = 0; int value = 7;
            float f = 1.5; complex c = 1.0im; bool ready = false;
            while (n % 3 != 2 && !ready) {
                n += 1; if (n == 1) { continue; }
                f = f + float(n); c = c + complex(f);
                ready = value % n == 0; if (ready) { break; }
            }
            f += float(value + 1);''')
        condition = lines.index('@QASM2QCX_LOOP_0_CONDITION')
        declarations = [(index, line.split()[2]) for index, line in enumerate(lines)
                        if line.startswith('VAR QASM2QCX_')]
        self.assertEqual({kind for _, kind in declarations}, {'INT', 'REAL', 'COMPLEX'})
        self.assertTrue(all(index < condition for index, _ in declarations))
        self.assertTrue(all(index > condition for index, line in enumerate(lines)
                            if line.startswith('LET QASM2QCX_')))
        ForLoopControlLoweringTests.assert_resolved_jumps(lines)

    def test_runtime_indices_ranges_and_unsupported_types_are_rejected(self) -> None:
        prefix = 'OPENQASM 3.0; qubit[2] q; int n = 0; bit[2] flags; complex c = 1.0im; '
        for statement in ('while (false) { reset q[n]; }',
                          'while (false) { for int i in [0:float(n):n] {} }',
                          'while (false) { for int i in {0, float(n)} {} }',
                          'while (flags) {}', 'while (c) {}', 'while (bool(c)) {}',
                          'while (false) { continue; n = n ** true; }'):
            with self.subTest(statement=statement):
                with self.assertRaises(qasm2qcx.QASM2QCXError):
                    convert(prefix + statement)

    def test_break_exit_precedes_following_operations_and_final_amplitude_output(self) -> None:
        lines = convert('''OPENQASM 3.0; include "stdgates.inc";
            pragma riken_braket.amplitudes 0
            qubit q; while (true) { break; x q; } x q;''')
        self.assertEqual(lines[-3:], ['@QASM2QCX_LOOP_0_END', 'X 0', 'DO AMPLITUDES 0'])
        ForLoopControlLoweringTests.assert_resolved_jumps(lines)

    def test_cli_rejects_invalid_false_body_without_partial_output(self) -> None:
        for body in ('int local;', 'reset missing;', 'continue; int local;'):
            with self.subTest(body=body):
                source = 'OPENQASM 3.0; while (false) { ' + body + ' }'
                with patch('builtins.open', mock_open(read_data=source)), \
                        patch('builtins.print') as output:
                    with self.assertRaises(SystemExit) as error:
                        qasm2qcx.main(['while.qasm'])
                self.assertTrue(str(error.exception).startswith('qasm2qcx.py:'))
                output.assert_not_called()


class ForLoopContextTests(unittest.TestCase):
    @staticmethod
    def recording_converter(source: str):
        class RecordingConverter(qasm2qcx.QASM2QCXConverter):
            def __init__(self, program):
                self.records = []
                super().__init__(program)

            def visit_ClassicalAssignment(self, statement):
                self.records.append((
                    self._QASM2QCXConverter__is_initialization_process,
                    tuple(self._QASM2QCXConverter__loop_contexts),
                    tuple(dict(scope) for scope in self._QASM2QCXConverter__loop_bindings)))
                super().visit_ClassicalAssignment(statement)

        program = qasm2qcx.openqasm3.parser.parse('OPENQASM 3.0; ' + source)
        converter = RecordingConverter(program)
        converter.visit(program)
        return converter

    def test_one_exit_and_sequential_continue_targets_per_loop(self) -> None:
        converter = self.recording_converter('''int sum = 0;
            for int i in [5:-2:1] { sum += i; } sum += 9;''')
        records = [record for record in converter.records if not record[0]]
        self.assertEqual([record[1][0].break_label for record in records[:-1]],
                         ['QASM2QCX_LOOP_0_END'] * 3)
        self.assertEqual([record[1][0].continue_label for record in records[:-1]],
                         [f'QASM2QCX_LOOP_0_NEXT_{i}' for i in range(3)])
        self.assertEqual([record[2][0]['i'] for record in records[:-1]], [5, 3, 1])
        self.assertEqual(records[-1][1:], ((), ()))

    def test_nested_and_sequential_instances_get_unique_targets(self) -> None:
        converter = self.recording_converter('''int sum = 0;
            for int i in [0:1] {
                for int j in [0:1] { sum += j; }
                sum += i;
            }
            for int k in [0:0] { sum += k; }''')
        frames = [record[1] for record in converter.records if not record[0]]
        self.assertEqual([len(frame) for frame in frames], [2, 2, 1, 2, 2, 1, 1])
        self.assertEqual([frame[-1].break_label for frame in frames], [
            'QASM2QCX_LOOP_1_END', 'QASM2QCX_LOOP_1_END', 'QASM2QCX_LOOP_0_END',
            'QASM2QCX_LOOP_2_END', 'QASM2QCX_LOOP_2_END', 'QASM2QCX_LOOP_0_END',
            'QASM2QCX_LOOP_3_END',
        ])
        self.assertEqual(frames[0][0].continue_label, 'QASM2QCX_LOOP_0_NEXT_0')
        self.assertEqual(frames[3][0].continue_label, 'QASM2QCX_LOOP_0_NEXT_1')
        self.assertEqual(frames[0][-1].continue_label, 'QASM2QCX_LOOP_1_NEXT_0')
        self.assertEqual(frames[1][-1].continue_label, 'QASM2QCX_LOOP_1_NEXT_1')

    def test_label_numbering_restarts_after_initialization(self) -> None:
        converter = self.recording_converter('''int sum = 0;
            for int i in [0:1] { for int j in [0:i] { sum += j; } }''')
        self.assertEqual([record[1:] for record in converter.records if record[0]],
                         [record[1:] for record in converter.records if not record[0]])
        self.assertEqual(converter._QASM2QCXConverter__loop_contexts, [])
        self.assertEqual(converter._QASM2QCXConverter__loop_bindings, [])

    def test_empty_loops_leave_no_active_context(self) -> None:
        converter = self.recording_converter('''int sum = 0;
            for int i in [1:0] { sum += i; } sum += 1;
            for int j in [0:0] { sum += j; }''')
        records = [record for record in converter.records if not record[0]]
        self.assertEqual(records[0][1:], ((), ()))
        self.assertEqual(records[1][1][0].break_label, 'QASM2QCX_LOOP_1_END')
        self.assertEqual(converter._QASM2QCXConverter__loop_contexts, [])

    def test_ordinary_loops_emit_no_unused_labels(self) -> None:
        source = 'OPENQASM 3.0; int sum = 0; for int i in [0:1] { sum += i; }'
        self.assertEqual(convert(source), convert('OPENQASM 3.0; int sum = 0; sum += 0; sum += 1;'))
        self.assertFalse(any('QASM2QCX_LOOP_' in line for line in convert(source)))

    def test_nested_contexts_restored_after_body_error(self) -> None:
        converter = self.recording_converter('int sum = 0;')
        loop = qasm2qcx.openqasm3.parser.parse('''OPENQASM 3.0;
            for int i in [0:1] { for int j in [0:1] { reset missing; } }''').statements[0]
        with self.assertRaises(qasm2qcx.InvalidQubitOperandException):
            converter.visit(loop)
        self.assertEqual(converter._QASM2QCXConverter__loop_contexts, [])
        self.assertEqual(converter._QASM2QCXConverter__loop_bindings, [])

    def test_nested_contexts_restored_after_budget_error(self) -> None:
        converter = self.recording_converter('int sum = 0;')
        loop = qasm2qcx.openqasm3.parser.parse('''OPENQASM 3.0;
            for int i in [0:1] { for int j in [0:1] { sum += j; } }''').statements[0]
        with patch.object(qasm2qcx.QASM2QCXConverter, 'MAX_EXPANDED_LOOP_STATEMENTS', 1):
            with self.assertRaises(qasm2qcx.InvalidLoopRangeException):
                converter.visit(loop)
        self.assertEqual(converter._QASM2QCXConverter__loop_contexts, [])
        self.assertEqual(converter._QASM2QCXConverter__loop_bindings, [])

    def test_break_and_continue_rejected_outside_supported_loops(self) -> None:
        for statement in ('break;', 'continue;'):
            for source in (statement, 'if (true) { ' + statement + ' }'):
                with self.subTest(source=source):
                    with self.assertRaises((qasm2qcx.UnsupportedOpenQASMError,
                                            qasm2qcx.openqasm3.parser.QASM3ParsingError)):
                        convert('OPENQASM 3.0; ' + source)


class ForLoopControlLoweringTests(unittest.TestCase):
    @staticmethod
    def assert_resolved_jumps(lines: list[str]) -> None:
        labels = [line[1:] for line in lines if line.startswith('@')]
        if len(labels) != len(set(labels)):
            raise AssertionError('duplicate generated labels')
        for line in lines:
            if line.startswith(('JUMP ', 'JUMPIF ')):
                if line.split()[1] not in labels:
                    raise AssertionError(f'unresolved jump: {line}')

    def test_unconditional_break_leaves_all_iterations_converted(self) -> None:
        lines = convert('OPENQASM 3.0; int sum = 0; for int i in [0:1] { break; sum += i; }')
        self.assertEqual(lines, [
            'QUBITS 0', 'VAR SUM7 INT', 'LET SUM7 := 0',
            'JUMP QASM2QCX_LOOP_0_END', 'LET SUM7 += 0', '@QASM2QCX_LOOP_0_NEXT_0',
            'JUMP QASM2QCX_LOOP_0_END', 'LET SUM7 += 1', '@QASM2QCX_LOOP_0_NEXT_1',
            '@QASM2QCX_LOOP_0_END',
        ])
        self.assert_resolved_jumps(lines)

    def test_continue_targets_end_of_each_iteration_including_final(self) -> None:
        lines = convert('OPENQASM 3.0; int sum = 0; for int i in [0:1] { continue; sum += i; }')
        self.assertEqual(lines[3:], [
            'JUMP QASM2QCX_LOOP_0_NEXT_0', 'LET SUM7 += 0', '@QASM2QCX_LOOP_0_NEXT_0',
            'JUMP QASM2QCX_LOOP_0_NEXT_1', 'LET SUM7 += 1', '@QASM2QCX_LOOP_0_NEXT_1',
            '@QASM2QCX_LOOP_0_END',
        ])
        self.assert_resolved_jumps(lines)

    def test_conditional_transfers_use_existing_branch_lowering(self) -> None:
        lines = convert('''OPENQASM 3.0; bool skip = false; int sum = 0;
            for int i in [0:1] {
                if (skip) { continue; } else if (sum > 2) { break; }
                sum += i;
            }''')
        self.assertIn('JUMP QASM2QCX_LOOP_0_NEXT_0', lines)
        self.assertIn('JUMP QASM2QCX_LOOP_0_NEXT_1', lines)
        self.assertEqual(lines.count('JUMP QASM2QCX_LOOP_0_END'), 2)
        self.assert_resolved_jumps(lines)

    def test_nested_transfers_target_nearest_loop(self) -> None:
        lines = convert('''OPENQASM 3.0; int sum = 0;
            for int i in [0:1] {
                for int j in [0:1] { break; sum += j; }
                continue; sum += i;
            }''')
        transfers = [line for line in lines if line.startswith('JUMP QASM2QCX_LOOP_')]
        self.assertEqual(transfers, [
            'JUMP QASM2QCX_LOOP_1_END', 'JUMP QASM2QCX_LOOP_1_END',
            'JUMP QASM2QCX_LOOP_0_NEXT_0',
            'JUMP QASM2QCX_LOOP_2_END', 'JUMP QASM2QCX_LOOP_2_END',
            'JUMP QASM2QCX_LOOP_0_NEXT_1',
        ])
        self.assert_resolved_jumps(lines)

    def test_control_in_inner_loop_does_not_add_unused_outer_labels(self) -> None:
        lines = convert('''OPENQASM 3.0;
            for int i in [0:0] { for int j in [0:0] { continue; } }''')
        self.assertFalse(any('QASM2QCX_LOOP_0_' in line for line in lines))
        self.assert_resolved_jumps(lines)

    def test_empty_and_single_iteration_loops(self) -> None:
        for statement in ('break;', 'continue;'):
            with self.subTest(statement=statement):
                self.assertEqual(convert('OPENQASM 3.0; for int i in [1:0] { ' + statement + ' }'),
                                 ['QUBITS 0'])
                lines = convert('OPENQASM 3.0; for int i in [3:3] { ' + statement + ' }')
                self.assertEqual(lines[-2:], ['@QASM2QCX_LOOP_0_NEXT_0', '@QASM2QCX_LOOP_0_END'])
                self.assert_resolved_jumps(lines)

    def test_temporary_declarations_hoisted_but_calculations_not_moved(self) -> None:
        for transfer in ('break;', 'continue;'):
            with self.subTest(transfer=transfer):
                lines = convert('''OPENQASM 3.0; int value = 7; int sum = 0;
                    for int i in [0:1] { ''' + transfer + ''' sum += value % 3; }
                    sum += value + 1;''')
                jump = next(i for i, line in enumerate(lines) if line.startswith('JUMP '))
                declarations = [i for i, line in enumerate(lines) if line.startswith('VAR QASM2QCX_')]
                self.assertEqual(len(declarations), 2)
                self.assertTrue(all(i < jump for i in declarations))
                division = next(i for i, line in enumerate(lines) if ' /= ' in line)
                self.assertGreater(division, jump)
                self.assert_resolved_jumps(lines)

    def test_nested_loop_temporaries_are_available_after_skipped_first_use(self) -> None:
        lines = convert('''OPENQASM 3.0; int value = 7; int sum = 0;
            for int i in [0:1] {
                for int j in [0:1] { continue; sum += value / 3; }
                sum += value + i;
            }''')
        jump = next(i for i, line in enumerate(lines) if line.startswith('JUMP '))
        self.assertTrue(all(i < jump for i, line in enumerate(lines)
                            if line.startswith('VAR QASM2QCX_')))
        self.assert_resolved_jumps(lines)

    def test_unreachable_and_empty_bodies_still_reject_unsupported_constructs(self) -> None:
        for bounds in ('[0:1]', '[1:0]'):
            for body in ('int local;', 'while (false) { int local; }', 'sum += true ** i;', 'i = 1;'):
                with self.subTest(bounds=bounds, body=body):
                    with self.assertRaises(qasm2qcx.UnsupportedOpenQASMError):
                        convert(f'OPENQASM 3.0; int sum = 0; for int i in {bounds} {{ break; {body} }}')
        with self.assertRaises(qasm2qcx.InvalidQubitOperandException):
            convert('OPENQASM 3.0; for int i in [0:1] { continue; reset missing; }')

    def test_transfer_visitors_reject_direct_ast_use_outside_loop(self) -> None:
        converter = qasm2qcx.QASM2QCXConverter(
            qasm2qcx.openqasm3.parser.parse('OPENQASM 3.0;'))
        for statement in (qasm2qcx.ast.BreakStatement(), qasm2qcx.ast.ContinueStatement()):
            with self.subTest(statement=statement):
                with self.assertRaisesRegex(qasm2qcx.UnsupportedOpenQASMError,
                                            'outside a supported loop'):
                    converter.visit(statement)

    def test_transfers_do_not_reduce_expansion_accounting(self) -> None:
        with patch.object(qasm2qcx.QASM2QCXConverter, 'MAX_LOOP_ITERATIONS', 2):
            with self.assertRaisesRegex(qasm2qcx.InvalidLoopRangeException, 'exceeds 2 iterations'):
                convert('OPENQASM 3.0; for int i in [0:2] { break; }')
        with patch.object(qasm2qcx.QASM2QCXConverter, 'MAX_EXPANDED_LOOP_STATEMENTS', 3):
            with self.assertRaisesRegex(qasm2qcx.InvalidLoopRangeException, 'exceeds 3 statements'):
                convert('OPENQASM 3.0; int sum = 0; for int i in [0:1] { continue; sum += i; }')

    def test_generated_target_labels_count_toward_output_limit(self) -> None:
        source = 'OPENQASM 3.0; for int i in [0:0] { continue; }'
        with patch.object(qasm2qcx.QASM2QCXConverter, 'MAX_LOOP_OUTPUT_LINES', 4):
            self.assertEqual(len(convert(source)), 4)
        for limit in (2, 3):
            with self.subTest(limit=limit):
                converter = qasm2qcx.QASM2QCXConverter(
                    qasm2qcx.openqasm3.parser.parse(source))
                with patch.object(qasm2qcx.QASM2QCXConverter, 'MAX_LOOP_OUTPUT_LINES', limit):
                    with self.assertRaisesRegex(qasm2qcx.InvalidLoopRangeException,
                                                f'exceeds {limit} QCX lines'):
                        converter.visit(qasm2qcx.openqasm3.parser.parse(source))
                self.assertEqual(converter._QASM2QCXConverter__loop_contexts, [])
                self.assertEqual(converter._QASM2QCXConverter__loop_bindings, [])


class ForLoopControlRegressionTests(unittest.TestCase):
    def test_negative_step_continue_labels_use_iteration_numbers_not_values(self) -> None:
        lines = convert('''OPENQASM 3.0; int total = 0;
            for int i in [5:-2:-1] { continue; total += i; }''')
        self.assertEqual([line for line in lines if line.startswith('JUMP QASM2QCX_LOOP_')],
                         [f'JUMP QASM2QCX_LOOP_0_NEXT_{i}' for i in range(4)])
        self.assertEqual([line for line in lines if line.startswith('LET TOTAL31 += ')],
                         [f'LET TOTAL31 += {i}' for i in (5, 3, 1, -1)])
        ForLoopControlLoweringTests.assert_resolved_jumps(lines)

    def test_shadowed_iterators_do_not_change_nearest_loop_target(self) -> None:
        lines = convert('''OPENQASM 3.0; int i = 9; int total = 0;
            for int i in [1:2] {
                for int i in [0:i] { if (i == 0) { continue; } break; }
                if (i == 1) { continue; }
                total += i;
            }
            total += i;''')
        self.assertIn('JUMP QASM2QCX_LOOP_1_END', lines)
        self.assertIn('JUMP QASM2QCX_LOOP_2_END', lines)
        self.assertIn('JUMP QASM2QCX_LOOP_0_NEXT_0', lines)
        self.assertIn('JUMP QASM2QCX_LOOP_0_NEXT_1', lines)
        self.assertEqual(lines[-1], 'LET TOTAL31 += I1')
        ForLoopControlLoweringTests.assert_resolved_jumps(lines)

    def test_measurement_precedes_transfer_and_quantum_operations_remain_after_it(self) -> None:
        for transfer in ('break', 'continue'):
            with self.subTest(transfer=transfer):
                lines = convert('''OPENQASM 3.0; include "stdgates.inc";
                    qubit q; qubit r; bit outcome;
                    for int i in [0:0] {
                        outcome = measure q;
                        if (outcome) { ''' + transfer + '''; }
                        x r;
                    }''')
                jump = next(index for index, line in enumerate(lines)
                            if line.startswith('JUMP QASM2QCX_LOOP_'))
                self.assertLess(lines.index('M 0'), jump)
                self.assertLess(lines.index('LET OUTCOME127 := :OUTCOME'), jump)
                self.assertGreater(lines.index('X 1'), jump)
                ForLoopControlLoweringTests.assert_resolved_jumps(lines)

    def test_mixed_temporary_types_are_declared_before_control_transfers(self) -> None:
        lines = convert('''OPENQASM 3.0; int value = 7; float f = 1.5;
            complex c = 1.0im; bool ready = false;
            for int i in [0:3] {
                if (i == 0) { continue; }
                f = f + float(i);
                c = c + complex(f);
                ready = value % (i + 1) == 0;
                if (ready) { break; }
            }
            f += float(value + 1); ready = !ready;''')
        first_jump = next(index for index, line in enumerate(lines) if line.startswith('JUMP'))
        declarations = [(index, line.split()[2]) for index, line in enumerate(lines)
                        if line.startswith('VAR QASM2QCX_')]
        self.assertEqual({kind for _, kind in declarations}, {'INT', 'REAL', 'COMPLEX'})
        self.assertTrue(all(index < first_jump for index, _ in declarations))
        ForLoopControlLoweringTests.assert_resolved_jumps(lines)

    def test_break_exit_does_not_skip_following_operations_or_amplitude_output(self) -> None:
        lines = convert('''OPENQASM 3.0; include "stdgates.inc";
            pragma riken_braket.amplitudes 0
            qubit q;
            for int i in [0:1] { break; x q; }
            x q;''')
        self.assertEqual(lines[-2:], ['X 0', 'DO AMPLITUDES 0'])
        self.assertEqual(lines[-3], '@QASM2QCX_LOOP_0_END')
        ForLoopControlLoweringTests.assert_resolved_jumps(lines)

    def test_cli_reports_outside_loop_transfers_without_partial_output(self) -> None:
        for transfer in ('break;', 'continue;'):
            for statement in (transfer, 'if (true) { ' + transfer + ' }'):
                with self.subTest(statement=statement):
                    with patch('builtins.open', mock_open(read_data='OPENQASM 3.0; ' + statement)), \
                            patch('builtins.print') as output:
                        with self.assertRaises(SystemExit) as error:
                            qasm2qcx.main(['loop.qasm'])
                    self.assertTrue(str(error.exception).startswith('qasm2qcx.py:'))
                    self.assertIn('outside loop', str(error.exception))
                    output.assert_not_called()


class ConstantForIterationInfrastructureTests(unittest.TestCase):
    @staticmethod
    def evaluate_values(bounds: str):
        converter = qasm2qcx.QASM2QCXConverter(
            qasm2qcx.openqasm3.parser.parse('OPENQASM 3.0;'))
        loop = qasm2qcx.openqasm3.parser.parse(
            f'OPENQASM 3.0; for int i in {bounds} {{}}').statements[0]
        return converter._QASM2QCXConverter__loop_values(loop)

    def test_range_provider_retains_lazy_ranges(self) -> None:
        for bounds, expected in (('[0:3]', [0, 1, 2, 3]), ('[5:-2:0]', [5, 3, 1]),
                                 ('[-2:2:2]', [-2, 0, 2]), ('[1:0]', []), ('[2:2]', [2])):
            with self.subTest(bounds=bounds):
                values = self.evaluate_values(bounds)
                self.assertIsInstance(values, range)
                self.assertEqual(list(values), expected)

    def test_range_counts_match_python_for_both_directions_and_empty_ranges(self) -> None:
        count = qasm2qcx.QASM2QCXConverter._QASM2QCXConverter__loop_value_count
        for start in range(-3, 4):
            for stop in range(-3, 4):
                for step in (-3, -2, -1, 1, 2, 3):
                    with self.subTest(start=start, stop=stop, step=step):
                        values = range(start, stop, step)
                        self.assertEqual(count(values), len(values))

    def test_huge_range_counts_do_not_use_len_or_materialize_values(self) -> None:
        magnitude = 10 ** 100
        count = qasm2qcx.QASM2QCXConverter._QASM2QCXConverter__loop_value_count
        # Binary constant arithmetic retains its existing precision; unary
        # negation itself now requires native-range inputs and results.
        for bounds in (f'[0 - {magnitude}:{magnitude}]', f'[{magnitude}:-1:0 - {magnitude}]'):
            with self.subTest(bounds=bounds):
                values = self.evaluate_values(bounds)
                self.assertIsInstance(values, range)
                self.assertEqual(count(values), 2 * magnitude + 1)
                with self.assertRaises(qasm2qcx.InvalidLoopRangeException):
                    convert(f'OPENQASM 3.0; for int i in {bounds} {{}}')

    def test_ordered_value_counts_include_duplicates(self) -> None:
        count = qasm2qcx.QASM2QCXConverter._QASM2QCXConverter__loop_value_count
        for values in ((), (2,), (5, -1, 5, 0)):
            with self.subTest(values=values):
                self.assertEqual(count(values), len(values))

    def test_unroller_uses_provider_values_without_range_specific_attributes(self) -> None:
        # Inject ordered values into a valid range AST to exercise the provider
        # interface independently of source syntax.
        with patch.object(qasm2qcx.QASM2QCXConverter, '_QASM2QCXConverter__loop_values',
                          return_value=(5, -1, 5)):
            lines = convert('OPENQASM 3.0; int total = 0; for int i in [0:2] { total += i; }')
        self.assertEqual(lines, ['QUBITS 0', 'VAR TOTAL31 INT', 'LET TOTAL31 := 0',
                                 'LET TOTAL31 += 5', 'LET TOTAL31 += -1', 'LET TOTAL31 += 5'])

    def test_scalar_identifier_iteration_is_rejected_including_empty_outer_loops(self) -> None:
        for iteration in ('values',):
            for source in (f'for int i in {iteration} {{}}',
                           f'for int j in [1:0] {{ for int i in {iteration} {{}} }}'):
                with self.subTest(source=source):
                    with self.assertRaises(qasm2qcx.UnsupportedOpenQASMError):
                        convert('OPENQASM 3.0; int values = 1; ' + source)

    def test_provider_values_still_use_existing_iteration_budget(self) -> None:
        with patch.object(qasm2qcx.QASM2QCXConverter, '_QASM2QCXConverter__loop_values',
                          return_value=(5, 5, 5)), \
                patch.object(qasm2qcx.QASM2QCXConverter, 'MAX_LOOP_ITERATIONS', 2):
            with self.assertRaisesRegex(qasm2qcx.InvalidLoopRangeException, 'exceeds 2 iterations'):
                convert('OPENQASM 3.0; for int i in [0:0] { break; }')


class RuntimeSetInfrastructureTests(unittest.TestCase):
    @staticmethod
    def converter():
        program = qasm2qcx.openqasm3.parser.parse('''OPENQASM 3.0;
            const int count = 2; const complex value = 2.0 + 1.0im;
            int first = 1; uint last = 3; float fraction = 1.5;
            bool ready = true; bit flag = 1; bit[2] flags = "01";''')
        converter = qasm2qcx.QASM2QCXConverter(program)
        converter.visit(program)
        return converter

    @staticmethod
    def source(elements):
        return qasm2qcx.openqasm3.parser.parse(
            f'OPENQASM 3.0; for int i in {elements} {{}}').statements[0].set_declaration

    def test_constant_runtime_and_mixed_element_classification(self) -> None:
        converter = self.converter()
        for elements, runtime in (
                ('{0, -2, count, count + 1}', False),
                ('{int(value), int(float(value)), uint(1.5)}', False),
                ('{first}', True), ('{first, first, last}', True),
                ('{0, first + count, count}', True),
                ('{int(fraction), int(ready), int(flag), int(flags[0])}', True),
                ('{int(first < last), int(!ready || flag)}', True)):
            with self.subTest(elements=elements):
                self.assertEqual(converter._QASM2QCXConverter__set_has_runtime_elements(
                    self.source(elements)), runtime)

    def test_classification_does_not_evaluate_division_or_remainder(self) -> None:
        converter = self.converter()
        for elements, runtime in (('{1 / 0, 1 % 0}', False),
                                   ('{first / 0, first % 0}', True),
                                   ('{first, 1 / 0}', True)):
            with self.subTest(elements=elements):
                self.assertEqual(converter._QASM2QCXConverter__set_has_runtime_elements(
                    self.source(elements)), runtime)

    def test_all_elements_are_validated_after_a_runtime_dependency(self) -> None:
        converter = self.converter()
        for elements, error in (
                ('{first, missing}', qasm2qcx.NoVariableNameException),
                ('{first, 1.5}', qasm2qcx.InvalidLoopRangeException),
                ('{first, ready}', qasm2qcx.InvalidLoopRangeException),
                ('{first, flags}', qasm2qcx.UnsupportedOpenQASMError),
                ('{first, 2 ** true}', qasm2qcx.UnsupportedOpenQASMError)):
            with self.subTest(elements=elements):
                with self.assertRaises(error):
                    converter._QASM2QCXConverter__set_has_runtime_elements(self.source(elements))
        with self.assertRaisesRegex(qasm2qcx.InvalidLoopRangeException, 'element 2'):
            converter._QASM2QCXConverter__set_has_runtime_elements(self.source('{first, first, 1.5}'))

    def test_invalid_types_casts_and_operators_are_rejected(self) -> None:
        converter = self.converter()
        for element in ('fraction', 'ready', 'flag', 'flags[0]', '1.0im', 'pi',
                        'int(complex(first))', 'int(bool(value))', 'int(bit(first))',
                        'int(bit[2](ready))',
                        'int(!first)', 'int(fraction % 2)', 'sin(1.0)'):
            with self.subTest(element=element):
                with self.assertRaises(qasm2qcx.QASM2QCXError):
                    converter._QASM2QCXConverter__set_has_runtime_elements(
                        self.source('{' + element + '}'))

    def test_nearest_iterator_binding_determines_dependency(self) -> None:
        converter = self.converter()
        bind = converter._QASM2QCXConverter__iterator_binding
        source = self.source('{first, first + 1}')
        inspect = converter._QASM2QCXConverter__set_has_runtime_elements
        self.assertTrue(inspect(source))
        with bind('first', 7):
            self.assertFalse(inspect(source))
            with bind('first', qasm2qcx._RuntimeIteratorBinding('PRIVATE_ITERATOR')):
                self.assertTrue(inspect(source))
                with bind('first', 9):
                    self.assertFalse(inspect(source))
                self.assertTrue(inspect(source))
            self.assertFalse(inspect(source))
        self.assertTrue(inspect(source))
        self.assertEqual(converter._QASM2QCXConverter__loop_bindings, [])

    def test_structural_iterator_names_require_no_invented_values(self) -> None:
        converter = self.converter()
        inspect = converter._QASM2QCXConverter__set_has_runtime_elements
        self.assertFalse(inspect(self.source('{i, 1 / i}'), {'i'}))
        self.assertTrue(inspect(self.source('{i, first}'), {'i'}))
        with self.assertRaises(qasm2qcx.NoVariableNameException):
            inspect(self.source('{i, missing}'), {'i'})

    def test_empty_set_and_duplicate_source_elements_are_preserved(self) -> None:
        converter = self.converter()
        source = qasm2qcx.ast.DiscreteSet([])
        self.assertFalse(converter._QASM2QCXConverter__set_has_runtime_elements(source))
        source = self.source('{first, 0, first}')
        before = list(source.values)
        self.assertTrue(converter._QASM2QCXConverter__set_has_runtime_elements(source))
        self.assertEqual(source.values, before)
        self.assertEqual(len(source.values), 3)

    def test_expression_and_storage_state_unchanged_on_success_and_failure(self) -> None:
        converter = self.converter()
        fields = ('expression_kind', 'value', 'value_type', 'value_kind',
                  'evaluate_constant', 'loop_iterations', 'expanded_loop_statements')
        before = tuple(getattr(converter, '_QASM2QCXConverter__' + field) for field in fields)
        lines = converter._QASM2QCXConverter__qcx_lines.copy()
        declarations = converter._QASM2QCXConverter__declared_temporary_variables.copy()
        storage = converter._QASM2QCXConverter__used_temporary_variables.copy()
        for elements in ('{first + 1, int(fraction)}', '{first, missing}', '{first, ready}'):
            with self.subTest(elements=elements):
                if elements == '{first + 1, int(fraction)}':
                    converter._QASM2QCXConverter__set_has_runtime_elements(self.source(elements))
                else:
                    with self.assertRaises(qasm2qcx.QASM2QCXError):
                        converter._QASM2QCXConverter__set_has_runtime_elements(self.source(elements))
                self.assertEqual(tuple(getattr(converter, '_QASM2QCXConverter__' + field)
                                       for field in fields), before)
                self.assertEqual(converter._QASM2QCXConverter__qcx_lines, lines)
                self.assertEqual(converter._QASM2QCXConverter__declared_temporary_variables, declarations)
                self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, storage)

    def test_runtime_sets_now_use_runtime_lowering(self) -> None:
        for elements in ('{first}', '{0, first, first}', '{first + 1}'):
            with self.subTest(elements=elements):
                lines = convert(f'OPENQASM 3.0; int first = 1; for int i in {elements} {{}}')
                self.assertIn('@QASM2QCX_LOOP_0_NEXT_0', lines)
                ForLoopControlLoweringTests.assert_resolved_jumps(lines)

    def test_constant_set_numeric_casts_keep_existing_output(self) -> None:
        prefix = 'OPENQASM 3.0; const complex value = 2.0 + 1.0im; int total = 0; '
        self.assertEqual(convert(prefix + '''for int i in {
            int(value), int(1.0im), uint(value), int(float(value)), int(true)} {
            total += i;
        }'''), convert(prefix + 'total += 2; total += 0; total += 2; total += 2; total += 1;'))


class RuntimeSetLoweringTests(unittest.TestCase):
    def test_all_captures_precede_body_and_preserve_duplicates(self) -> None:
        lines = convert('''OPENQASM 3.0; int first = 1; int total = 0;
            for int i in {first, 0, first} { first = 9; total += i; continue; }''')
        start = lines.index('LET QASM2QCX_INT_1 := FIRST31')
        self.assertEqual(lines[start:start + 4], [
            'LET QASM2QCX_INT_1 := FIRST31', 'LET QASM2QCX_INT_2 := 0',
            'LET QASM2QCX_INT_3 := FIRST31', 'LET QASM2QCX_INT_0 := QASM2QCX_INT_1'])
        self.assertEqual(lines.count('LET TOTAL31 += QASM2QCX_INT_0'), 3)
        for index in range(3):
            self.assertIn(f'LET QASM2QCX_INT_0 := QASM2QCX_INT_{index + 1}', lines)
            self.assertIn(f'JUMP QASM2QCX_LOOP_0_NEXT_{index}', lines)
        ForLoopControlLoweringTests.assert_resolved_jumps(lines)

    def test_captures_precede_shadowing_and_unconditional_break(self) -> None:
        lines = convert('''OPENQASM 3.0; int i = 1; int total = 0;
            for int i in {i, i + 1} { total += i; break; } total += i;''')
        first_break = lines.index('JUMP QASM2QCX_LOOP_0_END')
        self.assertLess(lines.index('LET QASM2QCX_INT_1 := I1'), first_break)
        self.assertLess(lines.index('LET QASM2QCX_INT_2 := QASM2QCX_INT_3'), first_break)
        self.assertEqual(lines[-1], 'LET TOTAL31 += I1')
        self.assertEqual(lines.count('JUMP QASM2QCX_LOOP_0_END'), 2)

    def test_nested_sets_and_ranges_keep_live_captures_reserved(self) -> None:
        lines = convert('''OPENQASM 3.0; int first = 1; int total = 0;
            for int i in {first, first + 1} {
                for int j in {i, i + 1} { total += i + j; continue; }
                for int k in [0:i] { total += k; break; }
                break;
            }''')
        for storage in ('QASM2QCX_INT_1', 'QASM2QCX_INT_2'):
            assignments = [line for line in lines if line.startswith(f'LET {storage} ')]
            self.assertEqual(len(assignments), 1)
        self.assertIn('JUMP QASM2QCX_LOOP_0_END', lines)
        labels = [line for line in lines if line.startswith('@')]
        self.assertEqual(len(labels), len(set(labels)))
        ForLoopControlLoweringTests.assert_resolved_jumps(lines)

    def test_runtime_range_iterator_is_accepted_as_set_element(self) -> None:
        lines = convert('''OPENQASM 3.0; int last = 2; int total = 0;
            for int i in [0:last] { for int j in {i, i + 1} { total += j; } }''')
        self.assertIn('@QASM2QCX_LOOP_1_NEXT_0', lines)
        self.assertIn('@QASM2QCX_LOOP_1_NEXT_1', lines)
        ForLoopControlLoweringTests.assert_resolved_jumps(lines)

    def test_casts_and_int_backed_uint_elements(self) -> None:
        lines = convert('''OPENQASM 3.0; uint first = 1; float fraction = 2.5;
            bool ready = true; bit[2] flags = "01";
            for int i in {first, int(fraction), int(ready), int(flags[0]), int(1.0im)} {}''')
        for index in range(5):
            self.assertIn(f'@QASM2QCX_LOOP_0_NEXT_{index}', lines)
        ForLoopControlLoweringTests.assert_resolved_jumps(lines)

    def test_runtime_elements_inside_empty_constant_outer_are_validated_not_emitted(self) -> None:
        prefix = 'OPENQASM 3.0; int first = 1; '
        for elements in ('{first, i}', '{first + i, first}'):
            with self.subTest(elements=elements):
                self.assertEqual(convert(prefix + 'for int i in [1:0] '
                                         f'{{ for int j in {elements} {{}} }}'), convert(prefix))
        for elements in ('{first, missing}', '{first, 1.5}'):
            with self.subTest(elements=elements):
                with self.assertRaises(qasm2qcx.QASM2QCXError):
                    convert(prefix + 'for int i in [1:0] '
                            f'{{ for int j in {elements} {{}} }}')

    def test_generated_runtime_set_copies_consume_expansion_budgets(self) -> None:
        for source, budget, maximum in (
                ('for int i in {first, first, first} { break; }', 'MAX_LOOP_ITERATIONS', 2),
                ('for int i in {first, first} { continue; break; }', 'MAX_EXPANDED_LOOP_STATEMENTS', 3),
                ('for int i in {first} {}', 'MAX_LOOP_OUTPUT_LINES', 3),
                ('for int i in [0:first] { for int j in {i} {} }', 'MAX_LOOP_ITERATIONS', 0)):
            with self.subTest(budget=budget, source=source):
                with patch.object(qasm2qcx.QASM2QCXConverter, budget, maximum):
                    with self.assertRaises(qasm2qcx.InvalidLoopRangeException):
                        convert('OPENQASM 3.0; int first = 1; ' + source)

    def test_runtime_set_iterators_remain_read_only_and_not_static_indices(self) -> None:
        for body in ('i = 2;', 'i = measure q[0];', 'reset q[i];',
                     'int local;', 'for int j in [0:float(i):first] {}'):
            with self.subTest(body=body):
                with self.assertRaises(qasm2qcx.QASM2QCXError):
                    convert('OPENQASM 3.0; int first = 1; qubit[2] q; bit[2] flags; '
                            'for int i in {first} { ' + body + ' }')

    def test_resources_and_expansion_scope_restored_after_body_failure(self) -> None:
        program = qasm2qcx.openqasm3.parser.parse('OPENQASM 3.0; int first = 1;')
        converter = qasm2qcx.QASM2QCXConverter(program)
        converter.visit(program)
        loop = qasm2qcx.openqasm3.parser.parse(
            'OPENQASM 3.0; for int i in {first, first + 1} { reset missing; }').statements[0]
        with self.assertRaises(qasm2qcx.InvalidQubitOperandException):
            converter.visit(loop)
        self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())
        self.assertEqual(converter._QASM2QCXConverter__loop_bindings, [])
        self.assertEqual(converter._QASM2QCXConverter__loop_contexts, [])
        self.assertEqual(converter._QASM2QCXConverter__expanded_loop_depth, 0)
        self.assertFalse(converter._QASM2QCXConverter__inside_expanded_loop())

    def test_skipped_computations_stay_after_control_flow_but_declarations_do_not(self) -> None:
        lines = convert('''OPENQASM 3.0; int first = 1; int total = 0;
            if (false) { for int i in {first / 0, first} { total += i; } }
            for int i in {first, first} { continue; total += first % 0; }
            total += first + 1;''')
        branch = lines.index('@QASM2QCX_IF_0')
        self.assertTrue(all(index < branch for index, line in enumerate(lines)
                            if line.startswith('VAR QASM2QCX_')))
        self.assertGreater(next(index for index, line in enumerate(lines) if ' /= ' in line), branch)
        ForLoopControlLoweringTests.assert_resolved_jumps(lines)


class RuntimeSetRegressionTests(unittest.TestCase):
    def test_all_mixed_loop_and_branch_labels_are_unique_and_resolved(self) -> None:
        lines = convert('''OPENQASM 3.0; int a = 1; bool ready = true;
            for int i in {a, a} {
                for int j in [0:i] {
                    while (ready) {
                        for int k in {j, j + 1} {
                            if (k == 0) { continue; } else { break; }
                        }
                        break;
                    }
                    continue;
                }
                for int k in {0, 1} { if (ready) { continue; } break; }
                break;
            }
            for int i in {a} { continue; }''')
        labels = [line[1:] for line in lines if line.startswith('@')]
        self.assertEqual(len(labels), len(set(labels)))
        ForLoopControlLoweringTests.assert_resolved_jumps(lines)

    def test_iterator_does_not_escape_or_retain_runtime_status_after_scope(self) -> None:
        prefix = 'OPENQASM 3.0; int a = 1; int total = 0; '
        for suffix in ('total += i;', 'for int j in {i} {}'):
            with self.subTest(suffix=suffix):
                with self.assertRaises(qasm2qcx.NoVariableNameException):
                    convert(prefix + 'for int i in {a} {} ' + suffix)
        lines = convert(prefix + '''const int i = 2;
            for int i in {a} { total += i; }
            for int j in {i, i} { total += j; }''')
        self.assertEqual(lines[-2:], ['LET TOTAL31 += 2', 'LET TOTAL31 += 2'])

    def test_iterator_shadowing_rejects_register_operands_inside_body(self) -> None:
        for body in ('reset q;', 'if (flags[0]) {}', 'total += int(flags[0]);'):
            name = 'q' if body == 'reset q;' else 'flags'
            with self.subTest(body=body):
                with self.assertRaises(qasm2qcx.QASM2QCXError):
                    convert('OPENQASM 3.0; int a = 1; int total = 0; qubit q; bit[2] flags; '
                            f'for int {name} in {{a}} {{ {body} }}')
        lines = convert('OPENQASM 3.0; int a = 1; qubit q; '
                        'for int q in {a} {} reset q;')
        self.assertEqual(lines[-1], 'RESET 0')

    def test_exact_iteration_budget_is_independent_between_passes(self) -> None:
        for source, count in (
                ('for int i in {a, a} { for int j in {0, 1} {} }', 6),
                ('for int i in {a, a} { for int i in {a} {} }', 4),
                ('for int i in [0:a] { for int j in {i, i} {} }', 2)):
            with self.subTest(source=source):
                with patch.object(qasm2qcx.QASM2QCXConverter, 'MAX_LOOP_ITERATIONS', count):
                    convert('OPENQASM 3.0; int a = 1; ' + source)
                with patch.object(qasm2qcx.QASM2QCXConverter, 'MAX_LOOP_ITERATIONS', count - 1):
                    with self.assertRaises(qasm2qcx.InvalidLoopRangeException):
                        convert('OPENQASM 3.0; int a = 1; ' + source)

    def test_shadowed_runtime_scopes_still_charge_body_statement_expansion(self) -> None:
        source = ('OPENQASM 3.0; int a = 1; int total = 0; '
                  'for int i in {a, a} { for int i in {a} { total += i; } }')
        with patch.object(qasm2qcx.QASM2QCXConverter, 'MAX_EXPANDED_LOOP_STATEMENTS', 4):
            convert(source)
        with patch.object(qasm2qcx.QASM2QCXConverter, 'MAX_EXPANDED_LOOP_STATEMENTS', 3):
            with self.assertRaises(qasm2qcx.InvalidLoopRangeException):
                convert(source)

    def test_capture_storage_released_after_partial_capture_failure(self) -> None:
        program = qasm2qcx.openqasm3.parser.parse('OPENQASM 3.0; int a = 1; bit[1] flags;')
        converter = qasm2qcx.QASM2QCXConverter(program)
        converter.visit(program)
        loop = qasm2qcx.openqasm3.parser.parse(
            'OPENQASM 3.0; for int i in {a, int(flags[2])} {}').statements[0]
        with self.assertRaises(qasm2qcx.QASM2QCXError):
            converter.visit(loop)
        self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())
        self.assertEqual(converter._QASM2QCXConverter__loop_bindings, [])
        self.assertEqual(converter._QASM2QCXConverter__loop_contexts, [])
        self.assertEqual(converter._QASM2QCXConverter__expanded_loop_depth, 0)

    def test_source_names_are_reserved_against_iterator_and_capture_storage(self) -> None:
        lines = convert('''OPENQASM 3.0; int QASM2QCX_INT_ = 9; int a = 1; int total = 0;
            for int i in {a, a} { total += i; }
            total += QASM2QCX_INT_;''')
        self.assertEqual([line for line in lines if line.startswith('LET QASM2QCX_INT_0 ')],
                         ['LET QASM2QCX_INT_0 := 9'])
        self.assertEqual(lines[-1], 'LET TOTAL31 += QASM2QCX_INT_0')

    def test_unreachable_bodies_and_late_elements_remain_validated(self) -> None:
        for body in ('total = sin(1.0);', 'total += 2 ** true;', 'i += 1;',
                     'if (false) { int local; }'):
            with self.subTest(body=body):
                with self.assertRaises(qasm2qcx.UnsupportedOpenQASMError):
                    convert('OPENQASM 3.0; int a = 1; int total = 0; '
                            'for int i in {a} { break; ' + body + ' }')
        for element in ('missing', 'true', 'sin(1.0)'):
            with self.subTest(element=element):
                with self.assertRaises(qasm2qcx.QASM2QCXError):
                    convert('OPENQASM 3.0; int a = 1; if (false) '
                            f'{{ for int i in {{a, {element}}} {{ break; }} }}')

    def test_amplitudes_follow_runtime_set_and_subsequent_operations(self) -> None:
        lines = convert('''OPENQASM 3.0; include "stdgates.inc";
            pragma riken_braket.amplitudes 0
            int a = 1; qubit q; for int i in {a, a} { break; x q; } reset q;''')
        self.assertEqual(lines[-3:], ['@QASM2QCX_LOOP_0_END', 'RESET 0', 'DO AMPLITUDES 0'])


class ConstantForSetLoweringTests(unittest.TestCase):
    def test_order_negative_values_and_duplicates_preserved(self) -> None:
        lines = convert('OPENQASM 3.0; int total = 0; for int i in {5, -1, 5, 0} { total += i; }')
        self.assertEqual(lines, ['QUBITS 0', 'VAR TOTAL31 INT', 'LET TOTAL31 := 0',
                                 'LET TOTAL31 += 5', 'LET TOTAL31 += -1',
                                 'LET TOTAL31 += 5', 'LET TOTAL31 += 0'])

    def test_constants_expressions_and_explicit_casts(self) -> None:
        prefix = 'OPENQASM 3.0; const int n = 3; const uint u = 2; int total = 0; '
        self.assertEqual(convert(prefix + '''for int[32] i in {n - 4, u + 1, int(2.5), uint(2.5)} {
            total += i;
        }'''), convert(prefix + 'total += -1; total += 3; total += 2; total += 2;'))

    def test_single_statement_and_static_quantum_indexing(self) -> None:
        prefix = 'OPENQASM 3.0; include "stdgates.inc"; qubit[6] q; '
        self.assertEqual(convert(prefix + 'for int i in {0, 2, 5, 2} x q[i];'),
                         convert(prefix + 'x q[0]; x q[2]; x q[5]; x q[2];'))

    def test_nested_set_uses_outer_iterator_before_shadowing(self) -> None:
        prefix = 'OPENQASM 3.0; int i = 9; int total = 0; '
        lines = convert(prefix + '''for int i in {2, 3} {
            for int i in {i, i + 1} { total += i; }
            total += i;
        } total += i;''')
        self.assertEqual(lines, convert(prefix + '''total += 2; total += 3; total += 2;
            total += 3; total += 4; total += 3; total += i;'''))

    def test_empty_outer_validates_independent_elements_without_fabricating_iterator(self) -> None:
        for elements in ('{1.5, i}', '{missing, i}', '{1 / 0, i}'):
            with self.subTest(elements=elements):
                with self.assertRaises(qasm2qcx.QASM2QCXError):
                    convert(f'OPENQASM 3.0; int n = 1; for int i in [1:0] {{ for int j in {elements} {{}} }}')
        self.assertEqual(convert('''OPENQASM 3.0; qubit q;
            for int i in [1:0] { for int j in {1 / i, i + 99} { reset q[j]; } }'''), ['QUBITS 1'])
        with self.assertRaises(qasm2qcx.ZeroDivisorException):
            convert('OPENQASM 3.0; for int i in {0} { for int j in {1 / i} {} }')

    def test_unknown_and_non_integer_elements_rejected(self) -> None:
        for elements in ('{missing}', '{1.5}', '{true}', '{1.0im}', '{pi}'):
            with self.subTest(elements=elements):
                with self.assertRaises(qasm2qcx.QASM2QCXError):
                    convert(f'OPENQASM 3.0; int n = 1; for int i in {elements} {{}}')
        for elements in ('{1 / 0}', '{1 % 0}'):
            with self.subTest(elements=elements):
                with self.assertRaises(qasm2qcx.ZeroDivisorException):
                    convert(f'OPENQASM 3.0; for int i in {elements} {{}}')

    def test_other_iterator_types_and_iterator_writes_rejected(self) -> None:
        for iteration_type in ('uint', 'float', 'bool', 'bit'):
            with self.subTest(iteration_type=iteration_type):
                with self.assertRaises(qasm2qcx.UnsupportedOpenQASMError):
                    convert(f'OPENQASM 3.0; for {iteration_type} i in {{1}} {{}}')
        for body in ('i += 1;', 'i = measure q;', 'while (false) { i = 2; }', 'int local;'):
            with self.subTest(body=body):
                with self.assertRaises(qasm2qcx.UnsupportedOpenQASMError):
                    convert('OPENQASM 3.0; qubit q; for int i in {1} { ' + body + ' }')

    def test_duplicate_iterations_have_distinct_continue_targets(self) -> None:
        lines = convert('''OPENQASM 3.0; bool skip = false;
            for int i in {2, 2, -1} { if (skip) { continue; } break; }''')
        for iteration in range(3):
            self.assertIn(f'JUMP QASM2QCX_LOOP_0_NEXT_{iteration}', lines)
        self.assertEqual(lines.count('JUMP QASM2QCX_LOOP_0_END'), 3)
        ForLoopControlLoweringTests.assert_resolved_jumps(lines)

    def test_mixed_nested_loop_transfers_are_resolved(self) -> None:
        lines = convert('''OPENQASM 3.0; int total = 0;
            for int i in {1, 2} {
                for int j in [0:i] { if (j == 0) { continue; } break; }
                while (true) { for int k in {i, i + 1} { break; } break; }
                if (i == 1) { continue; } total += i;
            }''')
        ForLoopControlLoweringTests.assert_resolved_jumps(lines)

    def test_skipped_temporaries_hoisted_without_moving_computations(self) -> None:
        lines = convert('''OPENQASM 3.0; int n = 7; int total = 0;
            for int i in {2, 2} { continue; total += n % 0; }
            total += n + 1;''')
        jump = lines.index('JUMP QASM2QCX_LOOP_0_NEXT_0')
        self.assertTrue(all(index < jump for index, line in enumerate(lines)
                            if line.startswith('VAR QASM2QCX_')))
        self.assertGreater(next(index for index, line in enumerate(lines) if ' /= ' in line), jump)

    def test_budget_counts_duplicates_nested_iterations_and_unreachable_statements(self) -> None:
        for source, attribute, limit in (
                ('for int i in {1, 1, 1} { break; }', 'MAX_LOOP_ITERATIONS', 2),
                ('for int i in {1, 1} { for int j in {i, i} {} }', 'MAX_LOOP_ITERATIONS', 5),
                ('for int i in {1, 1} { continue; break; }', 'MAX_EXPANDED_LOOP_STATEMENTS', 3),
                ('for int i in {1} { continue; }', 'MAX_LOOP_OUTPUT_LINES', 3)):
            with self.subTest(source=source):
                with patch.object(qasm2qcx.QASM2QCXConverter, attribute, limit):
                    with self.assertRaises(qasm2qcx.InvalidLoopRangeException):
                        convert('OPENQASM 3.0; ' + source)

    def test_empty_set_ast_emits_no_body_but_still_validates(self) -> None:
        # The installed parser does not accept the source spelling `{}`.
        for body, expected_error in (('', None), ('break;', None),
                                     ('while (false) { int local; }', qasm2qcx.UnsupportedOpenQASMError)):
            program = qasm2qcx.openqasm3.parser.parse(
                'OPENQASM 3.0; for int i in [0:0] { ' + body + ' }')
            program.statements[0].set_declaration = qasm2qcx.ast.DiscreteSet([])
            if expected_error:
                with self.assertRaises(expected_error):
                    qasm2qcx.QASM2QCXConverter(program)
            else:
                converter = qasm2qcx.QASM2QCXConverter(program)
                converter.visit(program)
                self.assertEqual(list(converter), ['QUBITS 0'])


class ConstantForSetRegressionTests(unittest.TestCase):
    def test_static_range_set_selections_and_bit_indices_use_iterator_values(self) -> None:
        prefix = 'OPENQASM 3.0; include "stdgates.inc"; qubit[4] q; bit[4] flags; '
        self.assertEqual(convert(prefix + '''for int i in {1, 0, 1} {
            x q[i:i + 1]; h q[{i, i + 2}]; flags[i] = 1;
        }'''), convert(prefix + '''
            x q[1:2]; h q[{1, 3}]; flags[1] = 1;
            x q[0:1]; h q[{0, 2}]; flags[0] = 1;
            x q[1:2]; h q[{1, 3}]; flags[1] = 1;'''))

    def test_all_mixed_nested_loop_jumps_and_duplicate_instances_are_unique(self) -> None:
        lines = convert('''OPENQASM 3.0; bool ready = false;
            for int i in {2, 2} {
                while (ready) {
                    for int j in {i, i + 1} {
                        for int k in [0:j] { if (ready) { continue; } break; }
                        continue;
                    }
                    break;
                }
                continue;
            }''')
        self.assertIn('JUMP QASM2QCX_LOOP_0_NEXT_0', lines)
        self.assertIn('JUMP QASM2QCX_LOOP_0_NEXT_1', lines)
        ForLoopControlLoweringTests.assert_resolved_jumps(lines)

    def test_set_evaluation_restores_expression_state_on_success_and_failure(self) -> None:
        converter = qasm2qcx.QASM2QCXConverter(qasm2qcx.openqasm3.parser.parse('OPENQASM 3.0;'))
        converter._QASM2QCXConverter__expression_kind = qasm2qcx.ExpressionKind.ARITHMETIC
        converter._QASM2QCXConverter__value = 99
        converter._QASM2QCXConverter__value_type = qasm2qcx.ValueType.INT
        converter._QASM2QCXConverter__value_kind = qasm2qcx.ValueKind.LITERAL
        converter._QASM2QCXConverter__evaluate_constant = False
        fields = ('expression_kind', 'value', 'value_type', 'value_kind', 'evaluate_constant')
        before = tuple(getattr(converter, '_QASM2QCXConverter__' + field) for field in fields)
        for elements, error in (('{1 + 2, int(2.5), 3}', None),
                                ('{1 + 2, missing}', qasm2qcx.NoVariableNameException),
                                ('{1, 2 % 0}', qasm2qcx.ZeroDivisorException)):
            with self.subTest(elements=elements):
                loop = qasm2qcx.openqasm3.parser.parse(
                    'OPENQASM 3.0; for int i in ' + elements + ' {}').statements[0]
                if error:
                    with self.assertRaises(error):
                        converter._QASM2QCXConverter__loop_values(loop)
                else:
                    self.assertEqual(converter._QASM2QCXConverter__loop_values(loop), (3, 2, 3))
                self.assertEqual(tuple(getattr(converter, '_QASM2QCXConverter__' + field)
                                       for field in fields), before)

    def test_unknown_names_and_unsupported_syntax_rejected_in_empty_outer_loop(self) -> None:
        for elements in ('{i + missing}', '{i, sin(1.0)}', '{i, true ** i}'):
            with self.subTest(elements=elements):
                with self.assertRaises(qasm2qcx.QASM2QCXError):
                    convert(f'''OPENQASM 3.0; int n = 2;
                        for int i in [1:0] {{ for int j in {elements} {{}} }}''')

    def test_skipped_body_still_rejects_invalid_elements_and_iterator_writes(self) -> None:
        for inner in ('for int j in {i, 1.5} {}', 'for int j in {i, missing} {}',
                      'for int j in {i, 1 / 0} {}', 'for int j in {i} { i = 1; }'):
            with self.subTest(inner=inner):
                with self.assertRaises(qasm2qcx.QASM2QCXError):
                    convert('OPENQASM 3.0; for int i in {1} { break; ' + inner + ' }')

    def test_nested_set_contexts_restored_after_body_error(self) -> None:
        converter = qasm2qcx.QASM2QCXConverter(qasm2qcx.openqasm3.parser.parse('OPENQASM 3.0;'))
        loop = qasm2qcx.openqasm3.parser.parse('''OPENQASM 3.0;
            for int i in {1, 1} { for int j in {i, i} { reset missing; } }''').statements[0]
        with self.assertRaises(qasm2qcx.InvalidQubitOperandException):
            converter.visit(loop)
        self.assertEqual(converter._QASM2QCXConverter__loop_contexts, [])
        self.assertEqual(converter._QASM2QCXConverter__loop_bindings, [])

    def test_amplitude_output_is_after_set_loop_exit_and_following_operations(self) -> None:
        lines = convert('''OPENQASM 3.0; include "stdgates.inc";
            pragma riken_braket.amplitudes 0
            qubit q; for int i in {0, 0} { break; x q; } x q;''')
        self.assertEqual(lines[-3:], ['@QASM2QCX_LOOP_0_END', 'X 0', 'DO AMPLITUDES 0'])
        ForLoopControlLoweringTests.assert_resolved_jumps(lines)

    def test_cli_invalid_set_element_reports_error_without_partial_output(self) -> None:
        for elements in ('{0, bool(n)}', '{0, 1.5}', '{0, 1 / 0}'):
            with self.subTest(elements=elements):
                source = f'OPENQASM 3.0; int n = 1; for int i in {elements} {{}}'
                with patch('builtins.open', mock_open(read_data=source)), \
                        patch('builtins.print') as output:
                    with self.assertRaises(SystemExit) as error:
                        qasm2qcx.main(['set.qasm'])
                self.assertTrue(str(error.exception).startswith('qasm2qcx.py:'))
                output.assert_not_called()


class ConstantForLoopRangeTests(unittest.TestCase):
    @staticmethod
    def evaluate_range(bounds: str, prefix: str = '', iteration_type: str = 'int') -> range:
        # Stage 1 tests range evaluation independently of the pending unroller.
        program = qasm2qcx.openqasm3.parser.parse('OPENQASM 3.0; ' + prefix)
        converter = qasm2qcx.QASM2QCXConverter(program)
        converter.visit(program)
        statement = qasm2qcx.openqasm3.parser.parse(
            f'OPENQASM 3.0; for {iteration_type} i in {bounds} {{}}').statements[0]
        return converter._QASM2QCXConverter__loop_range(statement)

    def test_inclusive_ranges_and_default_step(self) -> None:
        for bounds, expected in (
                ('[0:3]', [0, 1, 2, 3]), ('[-2:1]', [-2, -1, 0, 1]),
                ('[2:2]', [2]), ('[0:2:5]', [0, 2, 4]),
                ('[0:2:4]', [0, 2, 4]), ('[3:-1:0]', [3, 2, 1, 0]),
                ('[5:-2:0]', [5, 3, 1]), ('[2:-3:2]', [2])):
            with self.subTest(bounds=bounds):
                self.assertEqual(list(self.evaluate_range(bounds)), expected)

    def test_direction_mismatch_produces_empty_range(self) -> None:
        for bounds in ('[3:0]', '[0:-1:3]', '[3:2:0]'):
            with self.subTest(bounds=bounds):
                self.assertEqual(list(self.evaluate_range(bounds)), [])

    def test_constant_expressions_named_constants_and_integer_casts(self) -> None:
        prefix = 'const int lower = -2; const uint upper = 5; const int step = 2;'
        self.assertEqual(list(self.evaluate_range('[lower + 1:step:upper % 4 + 2]', prefix)),
                         [-1, 1, 3])
        self.assertEqual(list(self.evaluate_range('[int(0.5):uint(2.5)]')), [0, 1, 2])
        self.assertEqual(list(self.evaluate_range('[0:2]', iteration_type='int[32]')),
                         [0, 1, 2])

    def test_large_range_is_lazy_and_preserves_integer_precision(self) -> None:
        magnitude = 2**100
        values = self.evaluate_range(f'[0 - {magnitude}:{magnitude}]')
        self.assertIsInstance(values, range)
        self.assertEqual(values.start, -magnitude)
        self.assertEqual(values.stop, magnitude + 1)
        self.assertEqual(values[0], -magnitude)

    def test_rejects_zero_steps_even_for_empty_ranges(self) -> None:
        for bounds in ('[0:0:3]', '[3:0:0]', '[0:2 - 2:3]'):
            with self.subTest(bounds=bounds):
                with self.assertRaisesRegex(qasm2qcx.InvalidLoopRangeException,
                                            'step cannot be zero'):
                    self.evaluate_range(bounds)

    def test_rejects_non_integer_bounds_and_steps(self) -> None:
        for bounds in ('[0.0:3]', '[0:3.0]', '[0:1.0:3]', '[false:3]',
                       '[0:true:3]', '[0:3.0im]', '[0:bit(true):3]'):
            with self.subTest(bounds=bounds):
                with self.assertRaisesRegex(qasm2qcx.InvalidLoopRangeException,
                                            'must be a constant integer'):
                    self.evaluate_range(bounds)

    def test_rejects_missing_bounds(self) -> None:
        program = qasm2qcx.openqasm3.parser.parse('OPENQASM 3.0;')
        converter = qasm2qcx.QASM2QCXConverter(program)
        for start, end in ((None, qasm2qcx.ast.IntegerLiteral(3)),
                           (qasm2qcx.ast.IntegerLiteral(0), None), (None, None)):
            statement = qasm2qcx.ast.ForInLoop(
                qasm2qcx.ast.IntType(), qasm2qcx.ast.Identifier('i'),
                qasm2qcx.ast.RangeDefinition(start, end, None), [])
            with self.subTest(start=start, end=end):
                with self.assertRaisesRegex(qasm2qcx.InvalidLoopRangeException,
                                            'requires both bounds'):
                    converter._QASM2QCXConverter__loop_range(statement)

    def test_rejects_runtime_dependent_and_unknown_bounds(self) -> None:
        for bounds in ('[0:n]', '[n:3]', '[0:n:3]', '[0:n + 1]', '[0:missing]'):
            with self.subTest(bounds=bounds):
                with self.assertRaises(qasm2qcx.QASM2QCXError):
                    self.evaluate_range(bounds, 'int n = 3;')

    def test_rejects_other_iteration_types_and_sources(self) -> None:
        for iteration_type in ('uint', 'float', 'bool', 'bit'):
            with self.subTest(iteration_type=iteration_type):
                with self.assertRaisesRegex(qasm2qcx.UnsupportedOpenQASMError,
                                            'iterator requires int'):
                    self.evaluate_range('[0:3]', iteration_type=iteration_type)
        for bounds in ('{0, 1, 2}', 'items'):
            with self.subTest(bounds=bounds):
                with self.assertRaisesRegex(qasm2qcx.UnsupportedOpenQASMError,
                                            'iteration other than a constant range'):
                    self.evaluate_range(bounds)

    def test_zero_divisors_retain_constant_expression_diagnostics(self) -> None:
        for bounds in ('[0:3 / 0]', '[0:3 % 0]', '[0:1 / 0:3]'):
            with self.subTest(bounds=bounds):
                with self.assertRaises(qasm2qcx.ZeroDivisorException):
                    self.evaluate_range(bounds)

    def test_validation_runs_during_conversion(self) -> None:
        with self.assertRaisesRegex(qasm2qcx.InvalidLoopRangeException, 'step cannot be zero'):
            convert('OPENQASM 3.0; for int i in [0:0:3] {}')
        self.assertEqual(convert('OPENQASM 3.0; const int n = 3; for int i in [0:n] {}'),
                         ['QUBITS 0'])

    def test_range_evaluation_restores_enclosing_expression_state_on_success_and_error(self) -> None:
        converter = qasm2qcx.QASM2QCXConverter(
            qasm2qcx.openqasm3.parser.parse('OPENQASM 3.0;'))
        fields = ('expression_kind', 'value', 'value_type', 'value_kind', 'evaluate_constant')
        original = (qasm2qcx.ExpressionKind.ARITHMETIC, 'SENTINEL',
                    qasm2qcx.ValueType.FLOAT, qasm2qcx.ValueKind.LVALUE, False)
        for field, value in zip(fields, original):
            setattr(converter, '_QASM2QCXConverter__' + field, value)
        for bounds in ('[1:3]', '[0:3 / 0]'):
            statement = qasm2qcx.openqasm3.parser.parse(
                f'OPENQASM 3.0; for int i in {bounds} {{}}').statements[0]
            try:
                converter._QASM2QCXConverter__loop_range(statement)
            except qasm2qcx.ZeroDivisorException:
                pass
            self.assertEqual(tuple(getattr(converter, '_QASM2QCXConverter__' + field)
                                   for field in fields), original)
        self.assertEqual(list(converter), ['QUBITS 0'])


class ConstantForLoopUnrollingTests(unittest.TestCase):
    def test_unrolls_gate_indices_in_order(self) -> None:
        self.assertEqual(convert('''OPENQASM 3.0; include "stdgates.inc";
            qubit[4] q; for int i in [0:3] { h q[i]; }'''),
                         ['QUBITS 4', 'H 0', 'H 1', 'H 2', 'H 3'])
        self.assertEqual(convert('''OPENQASM 3.0; include "stdgates.inc";
            qubit[4] q; for int i in [3:-2:0] x q[i];'''),
                         ['QUBITS 4', 'X 3', 'X 1'])

    def test_iterator_reads_in_arithmetic_and_gate_parameters(self) -> None:
        loop = '''OPENQASM 3.0; include "stdgates.inc"; qubit[4] q; int sum = 0;
            for int i in [0:3] { sum += i * 2; rx(i * pi / 4) q[i]; }'''
        explicit = '''OPENQASM 3.0; include "stdgates.inc"; qubit[4] q; int sum = 0;
            sum += 0 * 2; rx(0 * pi / 4) q[0];
            sum += 1 * 2; rx(1 * pi / 4) q[1];
            sum += 2 * 2; rx(2 * pi / 4) q[2];
            sum += 3 * 2; rx(3 * pi / 4) q[3];'''
        self.assertEqual(convert(loop), convert(explicit))

    def test_constant_index_arithmetic_and_register_selections(self) -> None:
        self.assertEqual(convert('''OPENQASM 3.0; include "stdgates.inc"; qubit[4] q;
            for int i in [0:1] { h q[i * 2:i * 2 + 1]; }'''),
                         ['QUBITS 4', 'H 0', 'H 1', 'H 2', 'H 3'])

    def test_shadowing_restores_outer_variables_and_constants(self) -> None:
        for declaration in ('int i = 9;', 'const int i = 9;', 'float i = 9.0;'):
            with self.subTest(declaration=declaration):
                prefix = 'OPENQASM 3.0; ' + declaration + ' int sum = 0; '
                self.assertEqual(convert(prefix + 'for int i in [0:2] { sum += i; } sum += int(i);'),
                                 convert(prefix + 'sum += 0; sum += 1; sum += 2; sum += int(i);'))

    def test_iterator_is_not_visible_after_loop_or_to_its_own_bounds(self) -> None:
        for source in ('for int i in [0:2] {} int sum = i;',
                       'for int i in [0:i] {}'):
            with self.subTest(source=source):
                with self.assertRaises(qasm2qcx.NoVariableNameException):
                    convert('OPENQASM 3.0; ' + source)
        self.assertEqual(convert('OPENQASM 3.0; for int i in [0:2] {} int i = 7;'),
                         ['QUBITS 0', 'VAR I1 INT', 'LET I1 := 7'])

    def test_named_bounds_and_empty_range(self) -> None:
        prefix = 'OPENQASM 3.0; const int n = 3; int sum = 0; '
        self.assertEqual(convert(prefix + 'for int i in [n:-1:1] { sum += i; }'),
                         convert(prefix + 'sum += 3; sum += 2; sum += 1;'))
        self.assertEqual(convert(prefix + 'for int i in [3:1] { sum += i; }'),
                         convert(prefix))

    def test_iterator_reads_in_conditional_and_bit_indices(self) -> None:
        prefix = 'OPENQASM 3.0; bit[2] flags = "10"; int sum = 0; '
        self.assertEqual(convert(prefix + '''for int i in [0:1] {
            if (flags[i] && i == 0) { sum += i + 1; } else { sum += 2; }
        }'''), convert(prefix + '''
            if (flags[0] && 0 == 0) { sum += 0 + 1; } else { sum += 2; }
            if (flags[1] && 1 == 0) { sum += 1 + 1; } else { sum += 2; }'''))

    def test_measurement_reset_barrier_and_bit_assignments(self) -> None:
        prefix = 'OPENQASM 3.0; qubit[2] q; bit[2] flags; '
        self.assertEqual(convert(prefix + '''for int i in [0:1] {
            flags[i] = 1; flags[i] = measure q[i]; barrier q[i]; reset q[i];
        }'''), convert(prefix + '''
            flags[0] = 1; flags[0] = measure q[0]; barrier q[0]; reset q[0];
            flags[1] = 1; flags[1] = measure q[1]; barrier q[1]; reset q[1];'''))

    def test_iterator_writes_are_rejected_including_empty_ranges(self) -> None:
        for bounds in ('[0:1]', '[1:0]'):
            for body in ('i = 1;', 'i += 1;', 'i[0] = 1;', 'i = measure q;',
                         'if (false) { i = 1; }'):
                with self.subTest(bounds=bounds, body=body):
                    with self.assertRaisesRegex(qasm2qcx.UnsupportedOpenQASMError,
                                                'assignment to a for-loop iterator'):
                        convert(f'OPENQASM 3.0; qubit q; for int i in {bounds} {{ {body} }}')

    def test_unsupported_bodies_rejected_even_when_empty(self) -> None:
        for body in ('int local = 0;', 'const int local = 0;', 'qubit local;',
                     'while (false) { int local; }',
                     'pragma riken_braket.amplitudes\n'):
            with self.subTest(body=body):
                with self.assertRaises((qasm2qcx.UnsupportedOpenQASMError,
                                        qasm2qcx.openqasm3.parser.QASM3ParsingError)):
                    convert('OPENQASM 3.0; for int i in [1:0] { ' + body + ' }')

    def test_iterator_cannot_refer_to_shadowed_qubit_or_bit_register(self) -> None:
        for declaration, body in (('qubit i;', 'reset i;'),
                                  ('qubit[2] i;', 'reset i[0];'),
                                  ('bit[2] i;', 'if (i[0]) {}'),
                                  ('bit i; bit other;', 'other = i;')):
            with self.subTest(declaration=declaration, body=body):
                with self.assertRaises(qasm2qcx.QASM2QCXError):
                    convert(f'OPENQASM 3.0; {declaration} for int i in [0:1] {{ {body} }}')

    def test_basic_expansion_safeguard_handles_huge_ranges(self) -> None:
        for bounds in ('[0:10000]', '[10000:-1:0]', f'[0:{2**100}]'):
            with self.subTest(bounds=bounds):
                with self.assertRaisesRegex(qasm2qcx.InvalidLoopRangeException,
                                            'exceeds 10000 iterations'):
                    convert(f'OPENQASM 3.0; for int i in {bounds} {{}}')

    def test_loop_inside_runtime_branch_keeps_operations_in_branch(self) -> None:
        prefix = 'OPENQASM 3.0; bool ready = false; int value = 7; int sum = 0; '
        loop = 'if (ready) { for int i in [0:2] { sum += value % (i + 1); } } sum += value;'
        expanded = '''if (ready) {
            sum += value % (0 + 1); sum += value % (1 + 1); sum += value % (2 + 1);
        } sum += value;'''
        self.assertEqual(convert(prefix + loop), convert(prefix + expanded))

    def test_iteration_scope_is_restored_after_body_failure(self) -> None:
        program = qasm2qcx.openqasm3.parser.parse('''OPENQASM 3.0;
            qubit q; int i = 9;''')
        converter = qasm2qcx.QASM2QCXConverter(program)
        converter.visit(program)
        loop = qasm2qcx.openqasm3.parser.parse('''OPENQASM 3.0;
            for int i in [0:1] { reset missing; }''').statements[0]
        with self.assertRaises(qasm2qcx.InvalidQubitOperandException):
            converter.visit(loop)
        self.assertEqual(converter._QASM2QCXConverter__loop_bindings, [])
        converter.visit(qasm2qcx.openqasm3.parser.parse('OPENQASM 3.0; i += 1;'))
        self.assertEqual(list(converter)[-1], 'LET I1 += 1')

    def test_forward_named_bounds_are_rejected_in_initialization_pass(self) -> None:
        with self.assertRaises(qasm2qcx.NoVariableNameException):
            convert('OPENQASM 3.0; for int i in [0:n] {} const int n = 2;')


class ConstantForLoopNestingTests(unittest.TestCase):
    def test_nested_loops_emit_in_lexical_iteration_order(self) -> None:
        prefix = 'OPENQASM 3.0; include "stdgates.inc"; qubit[4] q; '
        self.assertEqual(convert(prefix + '''for int i in [0:1] {
            for int j in [0:1] { x q[2 * i + j]; }
        }'''), ['QUBITS 4', 'X 0', 'X 1', 'X 2', 'X 3'])

    def test_inner_bounds_and_steps_depend_on_outer_iterator(self) -> None:
        prefix = 'OPENQASM 3.0; int sum = 0; '
        source = '''for int i in [1:3] {
            for int j in [0:i:i] { sum += 10 * i + j; }
        }'''
        expanded = 'sum += 10; sum += 11; sum += 20; sum += 22; sum += 30; sum += 33;'
        self.assertEqual(convert(prefix + source), convert(prefix + expanded))

    def test_inner_shadowing_uses_outer_value_in_bounds_and_restores_it(self) -> None:
        prefix = 'OPENQASM 3.0; const int i = 9; int sum = 0; '
        source = '''for int i in [1:2] {
            sum += i;
            for int i in [0:i] { sum += i; }
            sum += i;
        } sum += i;'''
        expanded = '''sum += 1; sum += 0; sum += 1; sum += 1;
            sum += 2; sum += 0; sum += 1; sum += 2; sum += 2; sum += 9;'''
        self.assertEqual(convert(prefix + source), convert(prefix + expanded))

    def test_negative_steps_and_empty_inner_ranges(self) -> None:
        prefix = 'OPENQASM 3.0; int sum = 0; '
        source = '''for int i in [2:-1:0] {
            for int j in [i:-1:1] { sum += i * 10 + j; }
        }'''
        self.assertEqual(convert(prefix + source),
                         convert(prefix + 'sum += 22; sum += 21; sum += 11;'))

    def test_empty_outer_loop_does_not_invent_iterator_values(self) -> None:
        self.assertEqual(convert('''OPENQASM 3.0; include "stdgates.inc"; qubit q;
            for int i in [1:0] {
                for int j in [0:1 / i] { x q[j]; }
            }'''), ['QUBITS 1'])

    def test_unknown_inner_bounds_are_rejected_even_when_outer_empty(self) -> None:
        for bound in ('missing', 'i + missing'):
            with self.subTest(bound=bound):
                with self.assertRaises(qasm2qcx.NoVariableNameException):
                    convert(f'''OPENQASM 3.0; int n = 3;
                        for int i in [1:0] {{ for int j in [0:{bound}] {{}} }}''')

    def test_runtime_inner_bounds_are_validated_without_expanding_empty_outer(self) -> None:
        for bound in ('n', 'n + i'):
            with self.subTest(bound=bound):
                self.assertEqual(convert(f'''OPENQASM 3.0; int n = 3;
                    for int i in [1:0] {{ for int j in [0:{bound}] {{}} }}'''),
                    convert('OPENQASM 3.0; int n = 3;'))

    def test_unsupported_nested_ranges_are_rejected_even_when_outer_empty(self) -> None:
        for inner in ('for uint j in [0:1] {}', 'for int j in {0, missing} {}',
                      'for int j in [0:0:1] {}', 'for int j in [0:1.5] {}'):
            with self.subTest(inner=inner):
                with self.assertRaises(qasm2qcx.QASM2QCXError):
                    convert('OPENQASM 3.0; for int i in [1:0] { ' + inner + ' }')

    def test_writes_to_any_active_iterator_are_rejected(self) -> None:
        for body in ('i = 3;', 'j += 1;', 'if (false) { i = measure q; }'):
            with self.subTest(body=body):
                with self.assertRaisesRegex(qasm2qcx.UnsupportedOpenQASMError,
                                            'assignment to a for-loop iterator'):
                    convert('''OPENQASM 3.0; qubit q;
                        for int i in [1:0] { for int j in [1:0] { ''' + body + ' } }')

    def test_unsupported_expressions_and_gates_rejected_even_when_empty(self) -> None:
        for body in ('sum += true ** i;', 'sum <<= true;',
                     'sum = sin(i);', 'unknown q;', 'pow(1 / 0) @ x q;',
                     'x(1) q;', 'gphase(0) q;'):
            with self.subTest(body=body):
                with self.assertRaises(qasm2qcx.QASM2QCXError):
                    convert('''OPENQASM 3.0; include "stdgates.inc"; qubit q; int sum = 0;
                        for int i in [1:0] { for int j in [1:0] { ''' + body + ' } }')
        with self.assertRaisesRegex(qasm2qcx.UnsupportedOpenQASMError, 'gate x'):
            convert('OPENQASM 3.0; qubit q; for int i in [1:0] { x q; }')

    def test_nested_runtime_conditionals_keep_unique_labels_and_temp_declarations(self) -> None:
        prefix = 'OPENQASM 3.0; bool ready = false; int value = 7; int sum = 0; '
        source = '''if (ready) { for int i in [0:1] {
            for int j in [1:2] { if (value % j == i) { sum += value / j; } }
        } } sum += value;'''
        expanded = '''if (ready) {
            if (value % 1 == 0) { sum += value / 1; }
            if (value % 2 == 0) { sum += value / 2; }
            if (value % 1 == 1) { sum += value / 1; }
            if (value % 2 == 1) { sum += value / 2; }
        } sum += value;'''
        self.assertEqual(convert(prefix + source), convert(prefix + expanded))

    def test_all_bindings_restored_after_nested_body_failure(self) -> None:
        program = qasm2qcx.openqasm3.parser.parse('OPENQASM 3.0; int i = 9;')
        converter = qasm2qcx.QASM2QCXConverter(program)
        converter.visit(program)
        loop = qasm2qcx.openqasm3.parser.parse('''OPENQASM 3.0;
            for int i in [1:2] { for int j in [0:i] { reset missing; } }''').statements[0]
        with self.assertRaises(qasm2qcx.InvalidQubitOperandException):
            converter.visit(loop)
        self.assertEqual(converter._QASM2QCXConverter__loop_bindings, [])


class ConstantForLoopExpansionTests(unittest.TestCase):
    def test_iteration_budget_is_shared_by_sequential_loops(self) -> None:
        with patch.object(qasm2qcx.QASM2QCXConverter, 'MAX_LOOP_ITERATIONS', 4):
            self.assertEqual(convert('OPENQASM 3.0; for int i in [0:1] {} for int j in [0:1] {}'),
                             ['QUBITS 0'])
            with self.assertRaisesRegex(qasm2qcx.InvalidLoopRangeException, 'exceeds 4 iterations'):
                convert('OPENQASM 3.0; for int i in [0:1] {} for int j in [0:2] {}')

    def test_iteration_budget_counts_outer_and_inner_iterations(self) -> None:
        source = 'OPENQASM 3.0; for int i in [0:1] { for int j in [0:1] {} }'
        with patch.object(qasm2qcx.QASM2QCXConverter, 'MAX_LOOP_ITERATIONS', 6):
            self.assertEqual(convert(source), ['QUBITS 0'])
        with patch.object(qasm2qcx.QASM2QCXConverter, 'MAX_LOOP_ITERATIONS', 5):
            with self.assertRaisesRegex(qasm2qcx.InvalidLoopRangeException, 'exceeds 5 iterations'):
                convert(source)

    def test_statement_budget_limits_large_bodies_with_few_iterations(self) -> None:
        source = 'OPENQASM 3.0; int sum = 0; for int i in [0:1] { sum += 1; sum += 2; }'
        with patch.object(qasm2qcx.QASM2QCXConverter, 'MAX_EXPANDED_LOOP_STATEMENTS', 4):
            self.assertEqual(convert(source)[-4:], ['LET SUM7 += 1', 'LET SUM7 += 2'] * 2)
        with patch.object(qasm2qcx.QASM2QCXConverter, 'MAX_EXPANDED_LOOP_STATEMENTS', 3):
            with self.assertRaisesRegex(qasm2qcx.InvalidLoopRangeException, 'exceeds 3 statements'):
                convert(source)

    def test_output_budget_counts_broadcast_instructions(self) -> None:
        source = 'OPENQASM 3.0; include "stdgates.inc"; qubit[3] q; for int i in [0:1] { x q; }'
        with patch.object(qasm2qcx.QASM2QCXConverter, 'MAX_LOOP_OUTPUT_LINES', 7):
            self.assertEqual(convert(source), ['QUBITS 3', 'X 0', 'X 1', 'X 2'] + ['X 0', 'X 1', 'X 2'])
        with patch.object(qasm2qcx.QASM2QCXConverter, 'MAX_LOOP_OUTPUT_LINES', 6):
            with self.assertRaisesRegex(qasm2qcx.InvalidLoopRangeException, 'exceeds 6 QCX lines'):
                convert(source)

    def test_budget_counts_nested_loops_in_runtime_skipped_branches(self) -> None:
        source = '''OPENQASM 3.0; if (false) {
            for int i in [0:1] { for int j in [0:1] {} }
        }'''
        with patch.object(qasm2qcx.QASM2QCXConverter, 'MAX_LOOP_ITERATIONS', 5):
            with self.assertRaisesRegex(qasm2qcx.InvalidLoopRangeException, 'exceeds 5 iterations'):
                convert(source)

    def test_budget_failure_restores_all_bindings(self) -> None:
        program = qasm2qcx.openqasm3.parser.parse('OPENQASM 3.0;')
        converter = qasm2qcx.QASM2QCXConverter(program)
        loop = qasm2qcx.openqasm3.parser.parse('''OPENQASM 3.0;
            for int i in [0:1] { for int j in [0:1] {} }''').statements[0]
        with patch.object(qasm2qcx.QASM2QCXConverter, 'MAX_LOOP_ITERATIONS', 5):
            with self.assertRaises(qasm2qcx.InvalidLoopRangeException):
                converter.visit(loop)
        self.assertEqual(converter._QASM2QCXConverter__loop_bindings, [])

    def test_cli_reports_range_and_expansion_errors_without_partial_output(self) -> None:
        for source, message in (
                ('OPENQASM 3.0; for int i in [0:0:1] {}',
                 'For-loop range step cannot be zero'),
                ('OPENQASM 3.0; for int i in [0:10000] {}',
                 'For-loop expansion exceeds 10000 iterations'),
                ('OPENQASM 3.0; for int i in [0:1] { i = 0; }',
                 'Unsupported OpenQASM construct: assignment to a for-loop iterator')):
            with self.subTest(source=source):
                with patch('builtins.open', mock_open(read_data=source)), patch('builtins.print') as output:
                    with self.assertRaises(SystemExit) as error:
                        qasm2qcx.main(['loop.qasm'])
                self.assertEqual(str(error.exception), f'qasm2qcx.py: {message}')
                output.assert_not_called()


class IntegerRemainderConstantTests(unittest.TestCase):
    def test_signed_remainder_uses_dividend_sign(self) -> None:
        for lhs in (-7, -6, -2, -1, 0, 1, 2, 6, 7):
            for rhs in (-3, -1, 1, 3):
                with self.subTest(lhs=lhs, rhs=rhs):
                    expected = abs(lhs) % abs(rhs)
                    if lhs < 0:
                        expected = -expected
                    self.assertEqual(convert(f'''OPENQASM 3.0;
                        const int value = {lhs} % {rhs}; int result = value;'''), [
                            'QUBITS 0', 'VAR RESULT63 INT',
                            f'LET RESULT63 := {expected}',
                        ])

    def test_preserves_precision_for_large_constant_integers(self) -> None:
        magnitude = 2**100 + 17
        for lhs in (magnitude, -magnitude):
            for rhs in (7, -7):
                with self.subTest(lhs=lhs, rhs=rhs):
                    expected = abs(lhs) % abs(rhs) * (-1 if lhs < 0 else 1)
                    # Test remainder precision independently of native unary
                    # negation's operand-range checks.
                    source = f'(0 - {-lhs})' if lhs < 0 else str(lhs)
                    self.assertEqual(convert(f'''OPENQASM 3.0;
                        const int value = {source} % {rhs}; int result = value;''')[-1],
                                     f'LET RESULT63 := {expected}')

    def test_nested_expressions_and_precedence(self) -> None:
        for expression, expected in (
                ('2 + 7 % 3 * 4', 6), ('(2 + 7) % 4', 1),
                ('20 % 6 % 3', 2), ('(-7 % 3) * 4', -4),
                ('7 % (5 % 3)', 1), ('(7 / 2) % 2', 1)):
            with self.subTest(expression=expression):
                self.assertEqual(convert(f'''OPENQASM 3.0;
                    const int value = {expression}; int result = value;''')[-1],
                                 f'LET RESULT63 := {expected}')

    def test_uint_and_mixed_integer_constants(self) -> None:
        self.assertEqual(convert('''OPENQASM 3.0;
            const uint a = 7; const int b = 3;
            const uint value = a % b; uint result = value;'''), [
                'QUBITS 0', 'VAR RESULT63 UINT', 'LET RESULT63 := 1',
            ])

    def test_folds_literal_remainder_and_explicit_integer_casts(self) -> None:
        self.assertEqual(convert('''OPENQASM 3.0;
            int a = -7 % 3; int b = int(7.5) % uint(3.5); a = 7 % -3;'''), [
                'QUBITS 0', 'VAR A1 INT', 'LET A1 := -1',
                'VAR B1 INT', 'LET B1 := 1', 'LET A1 := 1',
            ])

    def test_remainder_in_constant_sizes_and_boolean_constants(self) -> None:
        self.assertEqual(convert('''OPENQASM 3.0;
            const uint size = 7 % 4; qubit[size] q;
            const bool yes = -7 % 3 == -1; bool result = yes;
            bit[7 % 4] flags;'''), [
                'QUBITS 3', 'VAR RESULT63 INT', 'LET RESULT63 := 1',
                'VAR FLAGS31 INT 3',
            ])

    def test_evaluated_zero_divisor_has_a_conversion_error(self) -> None:
        for source in (
                'const int value = 7 % 0;', 'const int value = 0 % 0;',
                'const int zero = 0; const int value = -7 % zero;',
                'const int value = 7 % (3 - 3);',
                'const bool value = true && 7 % 0 == 0;'):
            with self.subTest(source=source):
                with self.assertRaisesRegex(qasm2qcx.ZeroDivisorException,
                                            'divisor must not be zero'):
                    convert('OPENQASM 3.0; ' + source)

    def test_skipped_constant_remainder_checks_types_without_evaluation(self) -> None:
        for expression, expected in (
                ('false && 7 % 0 == 0', 0), ('true || 7 % 0 == 0', 1),
                ('!(false && 7 % 0 == 0)', 1)):
            with self.subTest(expression=expression):
                self.assertEqual(convert(f'''OPENQASM 3.0;
                    const bool value = {expression}; bool result = value;'''), [
                        'QUBITS 0', 'VAR RESULT63 INT', f'LET RESULT63 := {expected}',
                    ])
        for expression in ('true || 7 % missing == 0', 'false && 7.0 % 0 == 0'):
            with self.subTest(expression=expression):
                with self.assertRaises(qasm2qcx.QASM2QCXError):
                    convert(f'OPENQASM 3.0; const bool value = {expression};')

    def test_rejects_non_integer_operands(self) -> None:
        for expression in ('7.0 % 3', '7 % 3.0', '7.0im % 3',
                           'true % 3', '7 % false', 'bit(true) % 3'):
            with self.subTest(expression=expression):
                with self.assertRaisesRegex(qasm2qcx.UnsupportedOpenQASMError,
                                            'requires int or uint operands'):
                    convert(f'OPENQASM 3.0; const int value = {expression};')
        for declaration in ('float a = 7.0;', 'complex a = 7.0im;',
                            'bool a = true;', 'bit a = 1;'):
            with self.subTest(declaration=declaration):
                with self.assertRaisesRegex(qasm2qcx.UnsupportedOpenQASMError,
                                            'requires int or uint operands'):
                    convert(f'OPENQASM 3.0; {declaration} int value = a % 3;')

class IntegerRemainderRuntimeTests(unittest.TestCase):
    def test_emits_remainder_using_existing_qcx_operations(self) -> None:
        self.assertEqual(convert('OPENQASM 3.0; int a = 7; int b = 3; int result = a % b;'), [
            'QUBITS 0', 'VAR A1 INT', 'LET A1 := 7', 'VAR B1 INT', 'LET B1 := 3',
            'VAR RESULT63 INT', 'VAR QASM2QCX_INT_0 INT', 'VAR QASM2QCX_INT_1 INT',
            'LET QASM2QCX_INT_0 := A1', 'LET QASM2QCX_INT_1 := A1',
            'LET QASM2QCX_INT_1 /= B1', 'LET QASM2QCX_INT_1 *= B1',
            'LET QASM2QCX_INT_0 -= QASM2QCX_INT_1', 'LET RESULT63 := QASM2QCX_INT_0',
        ])

    def test_does_not_modify_source_operands(self) -> None:
        lines = convert('OPENQASM 3.0; int a = 7; int b = 3; int result = a % b;')
        self.assertEqual([line for line in lines if line.startswith('LET A1 ')], ['LET A1 := 7'])
        self.assertEqual([line for line in lines if line.startswith('LET B1 ')], ['LET B1 := 3'])

    def test_accepts_mixed_integer_operands_and_casts(self) -> None:
        for expression in ('a % 3', '7 % b', 'a % b', 'a % a',
                           '(a + 1) % (b - 1)', '(a % b) % b',
                           'a % (b % a)', 'int(f) % uint(b)'):
            with self.subTest(expression=expression):
                lines = convert('OPENQASM 3.0; int a = 7; uint b = 3; float f = 7.5; '
                                f'int result = {expression};')
                prefix = ('LET RESULT63 := QASM2QCX_INT_' if expression in ('a % 3', 'a % a')
                          else 'LET RESULT63 := :INT:QASM2QCX_UINT_')
                self.assertTrue(lines[-1].startswith(prefix))
                self.assertFalse(any('%=' in line for line in lines))

    def test_self_assignment_is_emitted_after_complete_remainder(self) -> None:
        lines = convert('OPENQASM 3.0; int a = 7; a = a % a;')
        self.assertEqual(lines[-2:], [
            'LET QASM2QCX_INT_0 -= QASM2QCX_INT_1', 'LET A1 := QASM2QCX_INT_0',
        ])

    def test_reuses_remainder_temporaries(self) -> None:
        lines = convert('OPENQASM 3.0; int a = 7; int b = 3; '
                        'int result = a % b; result = b % a; result = a + b;')
        self.assertEqual([line for line in lines if line.startswith('VAR QASM2QCX_')], [
            'VAR QASM2QCX_INT_0 INT', 'VAR QASM2QCX_INT_1 INT',
        ])

    def test_avoids_reserved_source_names(self) -> None:
        lines = convert('OPENQASM 3.0; int a = 7; int result = a % 3; '
                        'int QASM2QCX_INT_ = 9;')
        self.assertIn('LET RESULT63 := QASM2QCX_INT_1', lines)
        self.assertIn('LET QASM2QCX_INT_1 -= QASM2QCX_INT_2', lines)
        self.assertEqual([line for line in lines if line.startswith('LET QASM2QCX_INT_0 ')],
                         ['LET QASM2QCX_INT_0 := 9'])

    def test_zero_divisor_runtime_operations_remain_inside_short_circuit_rhs(self) -> None:
        for expression in ('a && 7 % 0 == 0', '!a || bool(7 % 0)',
                           'a && 7 % divisor == 0', 'a && divisor % 0 == 0'):
            with self.subTest(expression=expression):
                lines = convert('OPENQASM 3.0; bool a = false; int divisor = 0; '
                                f'bool result = {expression}; divisor = divisor + 1;')
                rhs_start = lines.index('@QASM2QCX_CONDITION_0')
                division = next(i for i, line in enumerate(lines) if ' /= ' in line)
                self.assertLess(rhs_start, division)
                first_jump = next(i for i, line in enumerate(lines) if line.startswith('JUMP'))
                self.assertTrue(all(i < first_jump for i, line in enumerate(lines)
                                    if line.startswith('VAR QASM2QCX_')))

    def test_skipped_runtime_remainder_operands_are_validated(self) -> None:
        for expression in ('a && 7 % missing == 0', 'a && 7.0 % 0 == 0',
                           'a && b % 3 == 0'):
            with self.subTest(expression=expression):
                with self.assertRaises(qasm2qcx.QASM2QCXError):
                    convert('OPENQASM 3.0; bool a = false; bit b = 1; '
                            f'bool result = {expression};')


class IntegerRemainderAssignmentTests(unittest.TestCase):
    def test_compound_assignment_matches_explicit_expression_assignment(self) -> None:
        for rhs in ('3', 'b', 'a', 'a - b', '(a % b) + 1', 'int(f)'):
            with self.subTest(rhs=rhs):
                prefix = 'OPENQASM 3.0; int a = 7; uint b = 3; float f = 3.5; '
                self.assertEqual(convert(prefix + f'a %= {rhs};'),
                                 convert(prefix + f'a = a % ({rhs});'))

    def test_self_referencing_assignment_updates_target_after_join(self) -> None:
        lines = convert('OPENQASM 3.0; int a = 7; a %= a;')
        self.assertEqual(lines[-2:], [
            'LET QASM2QCX_INT_0 -= QASM2QCX_INT_1', 'LET A1 := QASM2QCX_INT_0',
        ])
        self.assertEqual([line for line in lines if line.startswith('LET A1 ')], [
            'LET A1 := 7', 'LET A1 := QASM2QCX_INT_0',
        ])

    def test_uint_target_and_mixed_integer_rhs(self) -> None:
        lines = convert('OPENQASM 3.0; uint a = 7; int b = 3; a %= b;')
        self.assertEqual(lines[-1], 'LET A1 := QASM2QCX_UINT_0')
        self.assertIn('LET QASM2QCX_UINT_1 /= :UINT:B1', lines)

    def test_compound_assignment_reuses_temporaries(self) -> None:
        lines = convert('OPENQASM 3.0; int a = 7; int b = 3; '
                        'a %= b; b %= a; a += b;')
        self.assertEqual([line for line in lines if line.startswith('VAR QASM2QCX_')], [
            'VAR QASM2QCX_INT_0 INT', 'VAR QASM2QCX_INT_1 INT',
        ])
        self.assertFalse(any('%=' in line for line in lines))

    def test_zero_divisor_stays_in_skipped_branch(self) -> None:
        lines = convert('OPENQASM 3.0; int a = 7; if (false) { a %= 0; } a %= 3;')
        self.assertLess(lines.index('@QASM2QCX_IF_0'),
                        lines.index('LET QASM2QCX_INT_1 /= 0'))
        self.assertEqual(sum(line.startswith('VAR QASM2QCX_') for line in lines), 2)

    def test_rejects_non_integer_targets_and_rhs(self) -> None:
        for declaration, statement in (
                ('float a = 7.0;', 'a %= 3;'),
                ('complex a = 7.0im;', 'a %= 3;'),
                ('bool a = true;', 'a %= 3;'),
                ('bit a = 1;', 'a %= 3;'),
                ('bit[1] a = "1";', 'a[0] %= 3;'),
                ('int a = 7;', 'a %= 3.0;'),
                ('int a = 7;', 'a %= 3.0im;'),
                ('int a = 7;', 'a %= true;'),
                ('int a = 7; bit b = 1;', 'a %= b;'),
                ('int a = 7;', 'a[0] %= 3;')):
            with self.subTest(declaration=declaration, statement=statement):
                with self.assertRaises(qasm2qcx.UnsupportedOpenQASMError):
                    convert('OPENQASM 3.0; ' + declaration + statement)

    def test_rejects_constant_and_undeclared_targets(self) -> None:
        for source in ('const int a = 7; a %= 3;', 'a %= 3;'):
            with self.subTest(source=source):
                with self.assertRaises(qasm2qcx.NoVariableNameException):
                    convert('OPENQASM 3.0; ' + source)


class ConstantZeroDivisorTests(unittest.TestCase):
    def test_constant_division_reports_converter_error_for_all_numeric_types(self) -> None:
        for source in (
                'const int value = 7 / 0;', 'const uint value = 7 / 0;',
                'const float value = 7.0 / 0.0;',
                'const complex value = 7.0im / 0.0im;',
                'const int zero = 0; const int value = -7 / zero;',
                'const int value = 7 / (3 - 3);',
                'const bool value = true && 7 / 0 > 0;'):
            with self.subTest(source=source):
                with self.assertRaisesRegex(qasm2qcx.ZeroDivisorException,
                                            'Constant / expression divisor must not be zero'):
                    convert('OPENQASM 3.0; ' + source)

    def test_skipped_constant_division_still_checks_names_and_types(self) -> None:
        for expression, expected in (
                ('false && 7 / 0 > 0', 0), ('true || bool(7.0 / 0.0)', 1),
                ('!(false && bool(7 / 0))', 1)):
            with self.subTest(expression=expression):
                self.assertEqual(convert(f'''OPENQASM 3.0;
                    const bool value = {expression}; bool result = value;'''), [
                        'QUBITS 0', 'VAR RESULT63 INT', f'LET RESULT63 := {expected}',
                    ])
        for expression in ('true || bool(7 / missing)', 'false && bool(true / 0)'):
            with self.subTest(expression=expression):
                with self.assertRaises(qasm2qcx.QASM2QCXError):
                    convert(f'OPENQASM 3.0; const bool value = {expression};')

    def test_cli_reports_both_zero_divisor_errors_without_traceback(self) -> None:
        for operator in ('/', '%'):
            with self.subTest(operator=operator):
                source = f'OPENQASM 3.0; const int value = 7 {operator} 0;'
                with patch('builtins.open', mock_open(read_data=source)):
                    with self.assertRaises(SystemExit) as caught:
                        qasm2qcx.main(['zero.qasm'])
                self.assertEqual(str(caught.exception),
                                 f'qasm2qcx.py: Constant {operator} expression divisor must not be zero')


class WorkingBaselineTests(unittest.TestCase):
    def test_flattens_registers_and_broadcasts_gates(self) -> None:
        source = """
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[2] a;
            qubit[2] b;
            h a;
            cx a, b;
        """

        self.assertEqual(
            convert(source),
            ["QUBITS 4", "H 0", "H 1", "CX 0 2", "CX 1 3"],
        )

    def test_converts_literal_rotation_parameters(self) -> None:
        source = """
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit q;
            rx(0.5) q;
            ry(-1.0) q;
            rz(2) q;
        """

        self.assertEqual(
            convert(source),
            ["QUBITS 1", "EX 0 -0.25", "EY 0 0.5", "EZ 0 -1.0"],
        )

    def test_uses_scalar_constant_as_gate_parameter(self) -> None:
        source = """
            OPENQASM 3.0;
            include "stdgates.inc";
            const float theta = 1.5;
            qubit q;
            rz(theta) q;
        """

        self.assertEqual(convert(source), ["QUBITS 1", "EZ 0 -0.75"])

    def test_rejects_mismatched_register_broadcasting(self) -> None:
        source = """
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[2] a;
            qubit[3] b;
            cx a, b;
        """

        with self.assertRaises(qasm2qcx.WrongBroadcastingException):
            convert(source)

    def test_rejects_size_one_register_broadcasting(self) -> None:
        source = """
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[1] a;
            qubit[2] b;
            cx a, b;
        """

        with self.assertRaises(qasm2qcx.WrongBroadcastingException):
            convert(source)

    def test_broadcasts_indexed_size_one_register_as_scalar(self) -> None:
        source = """
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[1] a;
            qubit[2] b;
            cx a[0], b;
        """

        self.assertEqual(
            convert(source),
            ["QUBITS 3", "CX 0 1", "CX 0 2"],
        )


class BooleanStorageTests(unittest.TestCase):
    def test_declares_initializes_and_copies_booleans(self) -> None:
        self.assertEqual(convert('''OPENQASM 3.0;
            bool a = true; bool b = false; bool c;
            c = a; a = b; b = true;'''), [
                "QUBITS 0", "VAR A1 INT", "LET A1 := 1",
                "VAR B1 INT", "LET B1 := 0", "VAR C1 INT",
                "LET C1 := A1", "LET A1 := B1", "LET B1 := 1",
            ])

    def test_boolean_constants_are_inlined(self) -> None:
        self.assertEqual(convert('''OPENQASM 3.0;
            const bool yes = true; const bool no = false;
            const bool alias = yes; bool a = alias; a = no;'''), [
                "QUBITS 0", "VAR A1 INT", "LET A1 := 1", "LET A1 := 0",
            ])

    def test_direct_boolean_condition(self) -> None:
        self.assertEqual(convert('''OPENQASM 3.0;
            bool a = true; if (a) { a = false; }'''), [
                "QUBITS 0", "VAR A1 INT", "LET A1 := 1",
                "JUMPIF QASM2QCX_IF_0 A1 \\= 0",
                "JUMP QASM2QCX_END_IF_0", "@QASM2QCX_IF_0",
                "LET A1 := 0", "@QASM2QCX_END_IF_0",
            ])

    def test_literal_and_constant_conditions_jump_without_literal_jumpif(self) -> None:
        lines = convert('''OPENQASM 3.0; const bool yes = true;
            if (true) {} if (false) {} if (!yes) {} if (!!yes) {}''')
        self.assertEqual([line for line in lines if line.startswith("JUMP ")], [
            "JUMP QASM2QCX_IF_0", "JUMP QASM2QCX_END_IF_1",
            "JUMP QASM2QCX_END_IF_2", "JUMP QASM2QCX_IF_3",
        ])
        self.assertFalse(any(line.startswith("JUMPIF ") for line in lines))

    def test_boolean_and_bit_logical_conditions(self) -> None:
        lines = convert('''OPENQASM 3.0; bool a = true; bit b = 0;
            if (a && !b || false) { a = false; }''')
        self.assertIn("JUMPIF QASM2QCX_CONDITION_1 A1 \\= 0", lines)
        self.assertIn("JUMPIF QASM2QCX_CONDITION_0 B1 \\= 0", lines)
        self.assertIn("JUMP QASM2QCX_END_IF_0", lines)

    def test_boolean_equality_conditions(self) -> None:
        lines = convert('''OPENQASM 3.0; bool a = true;
            if (a == false) {} if (true != a) {}''')
        self.assertIn("JUMPIF QASM2QCX_IF_0 A1 == 0", lines)
        self.assertIn("LET QASM2QCX_INT_0 := 1", lines)
        self.assertIn("JUMPIF QASM2QCX_IF_1 QASM2QCX_INT_0 \\= A1", lines)

    def test_rejects_boolean_constants_used_before_declaration(self) -> None:
        for source in (
                'bool a = yes; const bool yes = true;',
                'if (yes) {} const bool yes = true;',
                'const bool yes = yes;',
                'const bool yes = later; const bool later = true;',
                'bool a = true; const bool yes = a;'):
            with self.subTest(source=source):
                with self.assertRaises(qasm2qcx.NoVariableNameException):
                    convert('OPENQASM 3.0; ' + source)

    def test_boolean_constants_cannot_be_assigned(self) -> None:
        with self.assertRaises(qasm2qcx.NoVariableNameException):
            convert('OPENQASM 3.0; const bool yes = true; yes = false;')

    def test_rejects_numeric_to_boolean_without_explicit_cast(self) -> None:
        for source in (
                'bool a = 2;', 'bool a = 1.0;', 'bool a = 1.0im;',
                'bool a = true; int b = 1; a = b;',
                'const bool a = 2;', 'const bool a = 1.0;'):
            with self.subTest(source=source):
                with self.assertRaises(qasm2qcx.NoImplicitCastException):
                    convert('OPENQASM 3.0; ' + source)

    def test_rejects_unsupported_boolean_value_operations(self) -> None:
        for statement in (
                'a += true;', 'a -= false;', 'a *= true;', 'a /= true;',
                'a = -a;', 'a[0] = false;', 'if (a > false) {}',
                'if (a == 1) {}'):
            with self.subTest(statement=statement):
                with self.assertRaises(qasm2qcx.UnsupportedOpenQASMError):
                    convert('OPENQASM 3.0; bool a = true; bit b = 0; '
                            + statement)
        with self.assertRaises(qasm2qcx.NoImplicitCastException):
            convert('OPENQASM 3.0; bool a = true; int b = a + 1;')

    def test_booleans_are_not_measurement_targets(self) -> None:
        for source in (
                'qubit q; bool a = measure q;',
                'qubit q; bool a; a = measure q;'):
            with self.subTest(source=source):
                with self.assertRaises(qasm2qcx.InvalidBitOperandException):
                    convert('OPENQASM 3.0; ' + source)

    def test_rejects_boolean_gate_parameters_and_global_phase(self) -> None:
        for value in ('true', 'a', 'yes'):
            for statement in (f'rx({value}) q;', f'gphase({value});'):
                with self.subTest(statement=statement):
                    with self.assertRaises(qasm2qcx.WrongParameterTypeException):
                        convert('OPENQASM 3.0; include "stdgates.inc"; '
                                'qubit q; bool a = true; const bool yes = true; '
                                + statement)

    def test_rejects_boolean_local_declarations_and_duplicates(self) -> None:
        for source in (
                'if (true) { bool a = false; }',):
            with self.subTest(source=source):
                with self.assertRaises(qasm2qcx.UnsupportedOpenQASMError):
                    convert('OPENQASM 3.0; ' + source)
        for source in (
                'bool a; bool a;', 'bool a; bit a;',
                'const bool a = true; bool a;'):
            with self.subTest(source=source):
                with self.assertRaises(qasm2qcx.DuplicateIdentifierException):
                    convert('OPENQASM 3.0; ' + source)


class BooleanExpressionTests(unittest.TestCase):
    def test_runtime_short_circuit_defers_literal_division_by_zero(self) -> None:
        for expression in ('false && 1 / 0 > 0', 'true || bool(1 / 0)',
                           'a && 1 / 0 > 0', '!a || bool(1.0 / 0.0)'):
            with self.subTest(expression=expression):
                lines = convert('OPENQASM 3.0; bool a = false; '
                                f'bool result = {expression};')
                divisions = [i for i, line in enumerate(lines) if ' /= ' in line]
                self.assertEqual(len(divisions), 1)
                self.assertLess(lines.index('@QASM2QCX_CONDITION_0'), divisions[0])

    def test_skipped_runtime_boolean_operands_are_still_validated(self) -> None:
        for expression in ('false && missing', 'true || missing',
                           'false && 1', 'true || bool(1.0im)',
                           'false && flags', 'true || flags[0:0]'):
            with self.subTest(expression=expression):
                with self.assertRaises(qasm2qcx.QASM2QCXError):
                    convert('OPENQASM 3.0; bit[1] flags = "1"; '
                            f'bool result = {expression};')

    def test_materializes_comparison_as_zero_or_one(self) -> None:
        self.assertEqual(convert('''OPENQASM 3.0;
            int n = 1; bool a = n > 0;'''), [
                'QUBITS 0', 'VAR QASM2QCX_INT_0 INT',
                'VAR N1 INT', 'LET N1 := 1', 'VAR A1 INT',
                'JUMPIF QASM2QCX_BOOL_TRUE_0 N1 > 0',
                'JUMP QASM2QCX_BOOL_FALSE_0', '@QASM2QCX_BOOL_TRUE_0',
                'LET QASM2QCX_INT_0 := 1', 'JUMP QASM2QCX_BOOL_END_0',
                '@QASM2QCX_BOOL_FALSE_0', 'LET QASM2QCX_INT_0 := 0',
                '@QASM2QCX_BOOL_END_0', 'LET A1 := QASM2QCX_INT_0',
            ])

    def test_materializes_all_comparisons_and_logical_operators(self) -> None:
        for expression in (
                'n == 1', 'n != 1', 'n < 1', 'n <= 1', 'n > 1', 'n >= 1',
                '!a', '!!a', 'a && b', 'a || b',
                '(n > 0 && !a) || b', '(a || b) == !a',
                'flags[0] != flags[1]', '!flags[0]',
                'flags[0] && n > 0'):
            with self.subTest(expression=expression):
                lines = convert('''OPENQASM 3.0; int n = 1;
                    bool a = true; bool b = false; bit[2] flags = "01";
                    bool result; result = ''' + expression + ';')
                self.assertTrue(lines[-1].startswith('LET RESULT63 := QASM2QCX_INT_'))
                labels = [line[1:] for line in lines if line.startswith('@')]
                targets = [line.split()[1] for line in lines
                           if line.startswith(('JUMP ', 'JUMPIF '))]
                self.assertEqual(len(labels), len(set(labels)))
                self.assertTrue(set(targets) <= set(labels))

    def test_self_referencing_assignment_writes_destination_after_join(self) -> None:
        lines = convert('OPENQASM 3.0; bool a = true; a = !a;')
        self.assertIn('JUMPIF QASM2QCX_BOOL_FALSE_0 A1 \\= 0', lines)
        self.assertEqual(lines[-2:], [
            '@QASM2QCX_BOOL_END_0', 'LET A1 := QASM2QCX_INT_0',
        ])
        self.assertEqual([line for line in lines if line.startswith('LET A1 ')], [
            'LET A1 := 1', 'LET A1 := QASM2QCX_INT_0',
        ])

    def test_short_circuit_destinations_in_value_expressions(self) -> None:
        for operator, true_target, false_target in (
                ('&&', 'QASM2QCX_CONDITION_0', 'QASM2QCX_BOOL_FALSE_0'),
                ('||', 'QASM2QCX_BOOL_TRUE_0', 'QASM2QCX_CONDITION_0')):
            with self.subTest(operator=operator):
                lines = convert(f'''OPENQASM 3.0; bool a = true;
                    int divisor = 0; bool result = a {operator} 1 / divisor > 0;''')
                index = lines.index(f'JUMPIF {true_target} A1 \\= 0')
                self.assertEqual(lines[index + 1], f'JUMP {false_target}')
                self.assertLess(lines.index('@QASM2QCX_CONDITION_0'),
                                next(i for i, line in enumerate(lines)
                                     if line.startswith('LET ') and ' /= ' in line))

    def test_nested_values_keep_live_temporaries_distinct_and_declared_once(self) -> None:
        lines = convert('''OPENQASM 3.0; int n = 1; bool a = true;
            bool result = (n > 0) == (a || n < 0);
            result = !result; n = n + 1;''')
        declarations = [line for line in lines if line.startswith('VAR QASM2QCX_')]
        self.assertEqual(len(declarations), len(set(declarations)))
        self.assertGreaterEqual(len(declarations), 3)
        first_jump = next(i for i, line in enumerate(lines) if line.startswith('JUMP'))
        self.assertTrue(all(lines.index(line) < first_jump for line in declarations))

    def test_respects_reserved_temporary_names(self) -> None:
        lines = convert('''OPENQASM 3.0; bool a = true; bool b = !a;
            int QASM2QCX_INT_ = 7;''')
        self.assertIn('VAR QASM2QCX_INT_1 INT', lines)
        self.assertIn('LET B1 := QASM2QCX_INT_1', lines)

    def test_folds_boolean_constant_expressions_without_runtime_code(self) -> None:
        self.assertEqual(convert('''OPENQASM 3.0;
            const int n = 2; const bool positive = n > 0;
            const bool yes = positive && !(n < 1);
            const bool no = !yes || n / 2 != 1;
            bool a = yes; a = no;'''), [
                'QUBITS 0', 'VAR A1 INT', 'LET A1 := 1', 'LET A1 := 0',
            ])

    def test_constant_boolean_short_circuit_skips_arithmetic_not_validation(self) -> None:
        for expression, expected in (
                ('false && 1 / 0 > 0', 0), ('true || 1 / 0 > 0', 1),
                ('!(false && 1 / 0 > 0)', 1),
                ('true || (false && 1 / 0 > 0)', 1)):
            with self.subTest(expression=expression):
                self.assertEqual(convert(f'''OPENQASM 3.0;
                    const bool value = {expression}; bool a = value;'''), [
                        'QUBITS 0', 'VAR A1 INT', f'LET A1 := {expected}',
                    ])
        for expression in ('true || missing', 'false && missing',
                           'true || 1', 'false && 1', 'true || (1.0im == 0)'):
            with self.subTest(expression=expression):
                with self.assertRaises(qasm2qcx.QASM2QCXError):
                    convert(f'OPENQASM 3.0; const bool value = {expression};')

    def test_invalid_value_operands_and_targets_are_rejected(self) -> None:
        for statement in (
                'bool c = a && n;', 'bool c = !n;',
                'bool c = flags && a;', 'bool c = flags[0:0] && a;',
                'bool c = a & a;', 'bool c = ~a;',
                'bool c = a == n;', 'bool c = z == z;',
                'a += n > 0;'):
            with self.subTest(statement=statement):
                with self.assertRaises(qasm2qcx.QASM2QCXError):
                    convert('''OPENQASM 3.0; bool a = true; int n = 1;
                        bit[1] flags = "1"; complex z = 0.0im;''' + statement)

    def test_boolean_value_gate_parameters_are_rejected(self) -> None:
        for statement in ('rx(!a) q;', 'rx(n > 0) q;', 'gphase(a && a);'):
            with self.subTest(statement=statement):
                with self.assertRaises(qasm2qcx.WrongParameterTypeException):
                    convert('''OPENQASM 3.0; include "stdgates.inc";
                        qubit q; bool a = true; int n = 1;''' + statement)


class BooleanConversionTests(unittest.TestCase):
    def test_scalar_boolean_bit_interchange(self) -> None:
        self.assertEqual(convert('''OPENQASM 3.0;
            bool a = true; bit b = a; bool c = b;
            a = b; b = c; b = false;'''), [
                'QUBITS 0', 'VAR A1 INT', 'LET A1 := 1',
                'VAR B1 INT', 'LET B1 := A1', 'VAR C1 INT', 'LET C1 := B1',
                'LET A1 := B1', 'LET B1 := C1', 'LET B1 := 0',
            ])

    def test_indexed_bit_boolean_interchange(self) -> None:
        lines = convert('''OPENQASM 3.0; bit[2] flags = "01";
            bool a = flags[0]; a = flags[1]; flags[0] = a;
            flags[1] = true; flags[0] = !flags[1];''')
        self.assertIn('LET A1 := FLAGS31:0', lines)
        self.assertIn('LET A1 := FLAGS31:1', lines)
        self.assertIn('LET FLAGS31:0 := A1', lines)
        self.assertIn('LET FLAGS31:1 := 1', lines)
        self.assertEqual(lines[-1], 'LET FLAGS31:0 := QASM2QCX_INT_0')

    def test_accepts_zero_one_as_boolean_initializers_and_constants(self) -> None:
        self.assertEqual(convert('''OPENQASM 3.0;
            const bool yes = 1; const bool no = 0;
            bool a = 1; a = 0; bit b = yes; b = no;'''), [
                'QUBITS 0', 'VAR A1 INT', 'LET A1 := 1', 'LET A1 := 0',
                'VAR B1 INT', 'LET B1 := 1', 'LET B1 := 0',
            ])

    def test_numeric_boolean_cast_constants(self) -> None:
        for expression, expected in (
                ('bool(0)', 0), ('bool(7)', 1), ('bool(-7)', 1),
                ('bool(0.0)', 0), ('bool(-0.0)', 0), ('bool(0.25)', 1),
                ('bool(-0.25)', 1), ('bool(true)', 1), ('bool(false)', 0),
                ('bool(bit(true))', 1), ('bool(int(false))', 0),
                ('!bit(true)', 0), ('bit(true) && true', 1),
                ('bit(false) || true', 1)):
            with self.subTest(expression=expression):
                self.assertEqual(convert(f'''OPENQASM 3.0;
                    const bool value = {expression}; bool a = value;'''), [
                        'QUBITS 0', 'VAR A1 INT', f'LET A1 := {expected}',
                    ])

    def test_boolean_cast_in_condition_and_numeric_assignment(self) -> None:
        lines = convert('''OPENQASM 3.0; int n = -2; float f = -0.25;
            bool a = bool(n); a = bool(f);
            if (bool(n) && !bool(f)) { a = false; }''')
        self.assertIn('JUMPIF QASM2QCX_BOOL_TRUE_0 N1 \\= 0', lines)
        self.assertIn('JUMPIF QASM2QCX_BOOL_TRUE_1 F1 \\= 0', lines)
        self.assertIn('JUMPIF QASM2QCX_CONDITION_0 N1 \\= 0', lines)
        self.assertIn('JUMPIF QASM2QCX_END_IF_0 F1 \\= 0', lines)

    def test_boolean_cast_materializes_native_constants(self) -> None:
        lines = convert('OPENQASM 3.0; bool a = bool(pi); if (bool(tau)) {}')
        self.assertIn('LET QASM2QCX_REAL_0 := :PI', lines)
        self.assertIn('JUMPIF QASM2QCX_BOOL_TRUE_0 QASM2QCX_REAL_0 \\= 0', lines)
        self.assertIn('LET QASM2QCX_REAL_0 := :TWO_PI', lines)
        self.assertIn('JUMPIF QASM2QCX_IF_0 QASM2QCX_REAL_0 \\= 0', lines)

    def test_numeric_casts_and_promotions_from_boolean(self) -> None:
        lines = convert('''OPENQASM 3.0; bool a = true;
            int n = int(a); uint u = uint(a); float f = float(a);
            complex z = complex(a); n = a; f = a; z = a;''')
        self.assertIn('LET N1 := A1', lines)
        self.assertIn('LET QASM2QCX_UINT_0 := :UINT:A1', lines)
        self.assertIn('LET U1 := QASM2QCX_UINT_0', lines)
        self.assertIn('LET QASM2QCX_REAL_0 := :REAL:A1', lines)
        self.assertIn('LET QASM2QCX_COMPLEX_0 := :COMPLEX:A1', lines)
        self.assertIn('LET F1 := :REAL:A1', lines)
        self.assertIn('LET Z1 := :COMPLEX:A1', lines)

    def test_numeric_constant_promotions_from_boolean(self) -> None:
        self.assertEqual(convert('''OPENQASM 3.0;
            const bool yes = true; const int n = yes;
            const uint u = false; const float f = yes; const complex z = yes;
            int a = n; uint b = u; float c = f; complex d = z;'''), [
                'QUBITS 0', 'VAR A1 INT', 'LET A1 := 1',
                'VAR B1 UINT', 'LET B1 := 0', 'VAR C1 REAL', 'LET C1 := 1.0',
                'VAR D1 COMPLEX', 'LET D1 := :COMPLEX:1.0',
            ])

    def test_boolean_bit_cast_and_equality(self) -> None:
        lines = convert('''OPENQASM 3.0; bool a = true; bit b = bit(a);
            bool c = bool(b); if (a == b) { b = !b; }
            c = b != a;''')
        self.assertIn('LET B1 := A1', lines)
        self.assertIn('JUMPIF QASM2QCX_IF_0 A1 == B1', lines)
        self.assertIn('JUMPIF QASM2QCX_BOOL_TRUE_2 B1 \\= A1', lines)

    def test_scalar_bit_accepts_boolean_expression_values(self) -> None:
        for expression in ('!a', 'a && a', 'n > 0', 'bool(n)', 'bit(a)'):
            with self.subTest(expression=expression):
                lines = convert('OPENQASM 3.0; bool a = true; int n = 1; '
                                f'bit b = {expression};')
                self.assertTrue(lines[-1].startswith('LET B1 := '))

    def test_constant_casts_preserve_short_circuit_validation(self) -> None:
        self.assertEqual(convert('''OPENQASM 3.0;
            const bool yes = true || bool(1 / 0);
            const bool no = false && bool(1 / 0);
            bool a = yes; a = no;'''), [
                'QUBITS 0', 'VAR A1 INT', 'LET A1 := 1', 'LET A1 := 0',
            ])
        for expression in ('true || bool(missing)', 'false && bool(1.0im)'):
            with self.subTest(expression=expression):
                with self.assertRaises(qasm2qcx.QASM2QCXError):
                    convert(f'OPENQASM 3.0; const bool a = {expression};')

    def test_cast_temporaries_are_released_after_conditions(self) -> None:
        for condition in ('bool(!a)', 'bit(!a)', 'bool(int(!a))'):
            with self.subTest(condition=condition):
                lines = convert('OPENQASM 3.0; bool a = true; '
                                f'if ({condition}) {{}} if ({condition}) {{}}')
                self.assertEqual([line for line in lines
                                  if line.startswith('VAR QASM2QCX_')], [
                    'VAR QASM2QCX_INT_0 INT',
                ])

    def test_rejects_unsupported_casts_and_whole_register_conversions(self) -> None:
        for statement in (
                'bool a = bool(1.0im);', 'bool a = bool(z);',
                'bool a = bool(flags);', 'bool a = flags;',
                'bool a = bool(flags[0:0]);', 'bool a = bool(flags[{0}]);',
                'bit[1] copy = true;',
                'bit[1] copy = bit[1](true);', 'bit a = bit(2);',
                'bit a = bit(0.25);', 'bool a = bool(n[0]);'):
            with self.subTest(statement=statement):
                with self.assertRaises(qasm2qcx.QASM2QCXError):
                    convert('''OPENQASM 3.0; bit[1] flags = "1";
                        complex z = 1.0im; int index = 0; int n = 0;''' + statement)


class StaticIndexingTests(unittest.TestCase):
    def test_converts_inclusive_and_stepped_qubit_ranges(self) -> None:
        source = """
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[5] q;
            x q[1:3];
            y q[0:2:4];
            z q[4:-2:0];
        """

        self.assertEqual(
            convert(source),
            [
                "QUBITS 5",
                "X 1", "X 2", "X 3",
                "Y 0", "Y 2", "Y 4",
                "Z 4", "Z 2", "Z 0",
            ],
        )

    def test_converts_negative_qubit_indices(self) -> None:
        source = """
            OPENQASM 3.0;
            qubit[5] q;
            reset q[-1];
            reset q[-3:-1];
            reset q[{-1, -3}];
        """

        self.assertEqual(
            convert(source),
            [
                "QUBITS 5",
                "RESET 4",
                "RESET 2", "RESET 3", "RESET 4",
                "RESET 4", "RESET 2",
            ],
        )

    def test_preserves_discrete_set_order_and_repeated_indices(self) -> None:
        source = """
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[4] q;
            x q[{3, 1, 3}];
        """

        self.assertEqual(
            convert(source), ["QUBITS 4", "X 3", "X 1", "X 3"])

    def test_broadcasts_compatible_selected_registers(self) -> None:
        source = """
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[3] a;
            qubit[3] b;
            cx a[{0, 2}], b[1:2];
            cx a[1], b[{2, 0}];
        """

        self.assertEqual(
            convert(source),
            ["QUBITS 6", "CX 0 4", "CX 2 5", "CX 1 5", "CX 1 3"],
        )

    def test_rejects_one_element_register_selection_broadcasting(self) -> None:
        source = """
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[2] a;
            qubit[2] b;
            cx a[0:0], b;
        """

        with self.assertRaises(qasm2qcx.WrongBroadcastingException):
            convert(source)

    def test_converts_selected_measurement_and_bit_copy(self) -> None:
        source = """
            OPENQASM 3.0;
            qubit[4] q;
            bit[4] source = "0101";
            bit[4] target;
            target[{3, 1}] = source[0:2:2];
            target[{2, 0}] = measure q[{3, 1}];
        """

        self.assertEqual(
            convert(source),
            [
                "QUBITS 4",
                "VAR SOURCE63 INT 4",
                "LET SOURCE63:0 := 1",
                "LET SOURCE63:1 := 0",
                "LET SOURCE63:2 := 1",
                "LET SOURCE63:3 := 0",
                "VAR TARGET63 INT 4",
                "LET TARGET63:3 := SOURCE63:0",
                "LET TARGET63:1 := SOURCE63:2",
                "M 3",
                "LET TARGET63:2 := :OUTCOME",
                "M 1",
                "LET TARGET63:0 := :OUTCOME",
            ],
        )

    def test_rejects_invalid_qubit_ranges(self) -> None:
        cases = [
            ("q[0:0:3]", qasm2qcx.InvalidQubitOperandException, "step cannot be zero"),
            ("q[3:1]", qasm2qcx.InvalidQubitOperandException, "range is empty"),
            ("q[0:4]", qasm2qcx.InvalidQubitOperandException, "outside register"),
            ("q[-5]", qasm2qcx.InvalidQubitOperandException, "outside register"),
            ("q[:3]", qasm2qcx.UnsupportedOpenQASMError, "omitted bound"),
            ("q[1:]", qasm2qcx.UnsupportedOpenQASMError, "omitted bound"),
        ]

        for operand, exception, message in cases:
            source = f"OPENQASM 3.0; qubit[4] q; reset {operand};"
            with self.subTest(operand=operand):
                with self.assertRaisesRegex(exception, message):
                    convert(source)

    def test_rejects_dynamic_and_multidimensional_indices(self) -> None:
        sources = [
            "OPENQASM 3.0; const int i = 1; qubit[3] q; reset q[i:2];",
            "OPENQASM 3.0; const int i = 1; qubit[3] q; reset q[{0, i}];",
            "OPENQASM 3.0; qubit[3] q; reset q[0, 1];",
        ]

        for source in sources:
            with self.subTest(source=source):
                with self.assertRaisesRegex(
                        qasm2qcx.UnsupportedOpenQASMError,
                        "non-literal qubit index|multidimensional qubit indexing"):
                    convert(source)


class GateConversionTests(unittest.TestCase):
    def test_converts_phase_gate_parameter(self) -> None:
        source = """
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit q;
            p(0.25) q;
        """

        self.assertEqual(convert(source), ["QUBITS 1", "U1 0 0.25"])

    def test_resolves_pi_in_constant_expression(self) -> None:
        source = """
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit q;
            rx(pi / 2) q;
        """

        self.assertEqual(
            convert(source),
            [
                "QUBITS 1",
                "VAR QASM2QCX_REAL_0 REAL",
                "LET QASM2QCX_REAL_0 := :PI",
                "LET QASM2QCX_REAL_0 /= 2.0",
                "LET QASM2QCX_REAL_0 *= -0.5",
                "EX 0 QASM2QCX_REAL_0",
            ],
        )

    def test_passes_pi_directly_to_nonrotation_gate(self) -> None:
        source = """
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit q;
            p(pi) q;
        """

        self.assertEqual(convert(source), ["QUBITS 1", "U1 0 :PI"])

    def test_evaluates_pi_numerically_for_constant_declaration(self) -> None:
        source = """
            OPENQASM 3.0;
            include "stdgates.inc";
            const float theta = pi / 2;
            qubit q;
            p(theta) q;
        """

        self.assertEqual(
            convert(source),
            ["QUBITS 1", f"U1 0 {3.141592653589793 / 2}"],
        )

    def test_converts_identity_gate(self) -> None:
        source = """
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit q;
            id q;
        """

        self.assertEqual(convert(source), ["QUBITS 1", "I 0"])

    def test_converts_global_phase(self) -> None:
        source = """
            OPENQASM 3.0;
            qubit q;
            gphase(pi);
        """

        self.assertEqual(
            convert(source),
            ["QUBITS 1", "PHASE :PI"],
        )

    def test_converts_runtime_global_phase(self) -> None:
        source = """
            OPENQASM 3.0;
            float theta = 1.0;
            qubit q;
            gphase(theta + 0.5);
        """

        lines = convert(source)
        self.assertIn("LET QASM2QCX_REAL_0 := THETA31", lines)
        self.assertIn("LET QASM2QCX_REAL_0 += 0.5", lines)
        self.assertEqual(lines[-1], "PHASE QASM2QCX_REAL_0")

    def test_promotes_runtime_integer_global_phase(self) -> None:
        source = """
            OPENQASM 3.0;
            int theta = 1;
            qubit q;
            gphase(theta);
        """

        self.assertEqual(convert(source)[-1], "PHASE :REAL:THETA31")

    def test_promotes_runtime_integer_gate_parameters(self) -> None:
        source = """
            OPENQASM 3.0;
            include "stdgates.inc";
            int theta = 1;
            int phi = 2;
            int lambda = 3;
            int gamma = 4;
            qubit[2] q;
            p(theta) q[0];
            u3(theta, phi, lambda) q[0];
            cp(theta) q[0], q[1];
            cu(theta, phi, lambda, gamma) q[0], q[1];
        """

        lines = convert(source)
        self.assertIn("U1 0 :REAL:THETA31", lines)
        self.assertIn(
            "U3 0 :REAL:THETA31 :REAL:PHI7 :REAL:LAMBDA63", lines)
        self.assertIn("CU1 0 1 :REAL:THETA31", lines)
        self.assertIn("LET QASM2QCX_REAL_0 := :REAL:THETA31", lines)
        self.assertNotIn("LET QASM2QCX_REAL_0 += :REAL:PHI7", lines)
        self.assertNotIn("LET QASM2QCX_REAL_0 += :REAL:LAMBDA63", lines)
        self.assertIn("LET QASM2QCX_REAL_0 += :REAL:GAMMA31", lines)
        self.assertEqual(
            lines[-1],
            "CU3 0 1 :REAL:THETA31 :REAL:PHI7 :REAL:LAMBDA63",
        )

    def test_converts_runtime_integer_rotation_through_real_temporary(self) -> None:
        source = """
            OPENQASM 3.0;
            include "stdgates.inc";
            int theta = 1;
            qubit q;
            rx(theta) q;
        """

        self.assertEqual(
            convert(source)[-4:],
            [
                "VAR QASM2QCX_REAL_0 REAL",
                "LET QASM2QCX_REAL_0 := :REAL:THETA31",
                "LET QASM2QCX_REAL_0 *= -0.5",
                "EX 0 QASM2QCX_REAL_0",
            ],
        )

    def test_promotes_runtime_integer_expression_gate_parameter(self) -> None:
        source = """
            OPENQASM 3.0;
            include "stdgates.inc";
            int theta = 1;
            qubit q;
            p(theta + 1) q;
        """

        lines = convert(source)
        self.assertEqual(lines[-1], "U1 0 :REAL:QASM2QCX_INT_0")

    def test_converts_cu_with_openqasm_phase_convention(self) -> None:
        source = """
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[2] q;
            cu(1, 2, 3, 4) q[0], q[1];
        """

        self.assertEqual(
            convert(source),
            ["QUBITS 2", "U1 0 4.5", "CU3 0 1 1 2 3"],
        )

    def test_reuses_runtime_cu_phase_when_broadcasting(self) -> None:
        source = """
            OPENQASM 3.0;
            include "stdgates.inc";
            float theta = 1.0;
            float phi = 2.0;
            float lambda = 3.0;
            float gamma = 4.0;
            qubit[2] control;
            qubit[2] target;
            cu(theta, phi, lambda, gamma) control, target;
        """

        lines = convert(source)
        self.assertIn("LET QASM2QCX_REAL_0 := THETA31", lines)
        self.assertNotIn("LET QASM2QCX_REAL_0 += PHI7", lines)
        self.assertNotIn("LET QASM2QCX_REAL_0 += LAMBDA63", lines)
        self.assertIn("LET QASM2QCX_REAL_0 /= 2.0", lines)
        self.assertIn("LET QASM2QCX_REAL_0 += GAMMA31", lines)
        self.assertEqual(
            lines[-4:],
            [
                "U1 0 QASM2QCX_REAL_0",
                "CU3 0 2 THETA31 PHI7 LAMBDA63",
                "U1 1 QASM2QCX_REAL_0",
                "CU3 1 3 THETA31 PHI7 LAMBDA63",
            ],
        )

    def test_converts_all_u_gate_parameters(self) -> None:
        source = """
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit q;
            p(0.1) q;
            u2(0.2, 0.3) q;
            u3(0.4, 0.5, 0.6) q;
            U(0.7, 0.8, 0.9) q;
        """

        self.assertEqual(
            convert(source),
            [
                "QUBITS 1",
                "U1 0 0.1",
                "U2 0 0.2 0.3",
                "U3 0 0.4 0.5 0.6",
                "PHASE 0.35",
                "U3 0 0.7 0.8 0.9",
            ],
        )

    def test_converts_runtime_builtin_u_global_phase(self) -> None:
        source = """
            OPENQASM 3.0;
            float theta = 0.1;
            int phi = 1;
            float lambda = 0.3;
            qubit q;
            U(theta, phi, lambda) q;
        """

        lines = convert(source)
        self.assertIn("LET QASM2QCX_REAL_0 := THETA31", lines)
        self.assertNotIn("LET QASM2QCX_REAL_0 += :REAL:PHI7", lines)
        self.assertNotIn("LET QASM2QCX_REAL_0 += LAMBDA63", lines)
        self.assertIn("LET QASM2QCX_REAL_0 /= 2.0", lines)
        self.assertEqual(
            lines[-2:],
            [
                "PHASE QASM2QCX_REAL_0",
                "U3 0 THETA31 :REAL:PHI7 LAMBDA63",
            ],
        )

    def test_repeats_builtin_u_global_phase_when_broadcasting(self) -> None:
        source = """
            OPENQASM 3.0;
            qubit[2] q;
            U(1, 2, 3) q;
        """

        self.assertEqual(
            convert(source),
            [
                "QUBITS 2",
                "PHASE 0.5",
                "U3 0 1 2 3",
                "PHASE 0.5",
                "U3 1 1 2 3",
            ],
        )

    def test_reuses_expression_parameter_when_broadcasting(self) -> None:
        source = """
            OPENQASM 3.0;
            include "stdgates.inc";
            float theta = 0.5;
            qubit[2] q;
            rx(theta + 0.25) q;
        """

        lines = convert(source)
        self.assertEqual(
            lines[-2:],
            ["EX 0 QASM2QCX_REAL_0", "EX 1 QASM2QCX_REAL_0"],
        )
        self.assertEqual(lines.count("LET QASM2QCX_REAL_0 *= -0.5"), 1)
        self.assertTrue(
            all(not line.startswith("VAR __") for line in lines),
            "QCX variable names must begin with a letter",
        )

    def test_temporary_does_not_collide_with_user_variable(self) -> None:
        source = """
            OPENQASM 3.0;
            include "stdgates.inc";
            float QASM2QCX_REAL_ = 0.5;
            qubit q;
            rx(QASM2QCX_REAL_ + 0.25) q;
        """

        lines = convert(source)
        self.assertIn("VAR QASM2QCX_REAL_0 REAL", lines)
        self.assertIn("VAR QASM2QCX_REAL_1 REAL", lines)
        self.assertEqual(lines[-1], "EX 0 QASM2QCX_REAL_1")

    def test_rejects_wrong_gate_arity(self) -> None:
        source = """
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[2] q;
            x q[0], q[1];
        """

        with self.assertRaises(qasm2qcx.WrongGateArityException):
            convert(source)

    def test_rejects_out_of_range_qubit_index(self) -> None:
        source = """
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[2] q;
            x q[2];
        """

        with self.assertRaisesRegex(
                qasm2qcx.InvalidQubitOperandException, "outside register"):
            convert(source)

    def test_rejects_nonliteral_qubit_index(self) -> None:
        source = """
            OPENQASM 3.0;
            include "stdgates.inc";
            const int index = 0;
            qubit q;
            x q[index];
        """

        with self.assertRaisesRegex(
                qasm2qcx.UnsupportedOpenQASMError, "non-literal qubit index"):
            convert(source)

    def test_preserves_noncommutative_expression_order(self) -> None:
        source = """
            OPENQASM 3.0;
            include "stdgates.inc";
            float theta = 0.5;
            qubit q;
            rx(1.0 - theta) q;
        """

        lines = convert(source)
        self.assertIn("LET QASM2QCX_REAL_0 := 1.0", lines)
        self.assertIn("LET QASM2QCX_REAL_0 -= THETA31", lines)

    def test_requires_stdgates_include(self) -> None:
        source = """
            OPENQASM 3.0;
            qubit q;
            x q;
        """

        with self.assertRaisesRegex(
                qasm2qcx.UnsupportedOpenQASMError, "gate x"):
            convert(source)

    def test_requires_stdgates_include_before_gate_use(self) -> None:
        source = """
            OPENQASM 3.0;
            qubit q;
            x q;
            include "stdgates.inc";
        """

        with self.assertRaisesRegex(
                qasm2qcx.UnsupportedOpenQASMError, "gate x"):
            convert(source)

    def test_enables_stdgates_at_include_position(self) -> None:
        source = """
            OPENQASM 3.0;
            qubit q;
            U(0, 0, 0) q;
            include "stdgates.inc";
            x q;
        """

        self.assertEqual(
            convert(source),
            ["QUBITS 1", "PHASE 0.0", "U3 0 0 0 0", "X 0"],
        )

    def test_requires_qubit_declaration_before_use(self) -> None:
        source = """
            OPENQASM 3.0;
            include "stdgates.inc";
            x q;
            qubit q;
        """

        with self.assertRaisesRegex(
                qasm2qcx.InvalidQubitOperandException, "not declared"):
            convert(source)

    def test_requires_qubit_declaration_before_measurement(self) -> None:
        source = """
            OPENQASM 3.0;
            measure q;
            qubit q;
        """

        with self.assertRaisesRegex(
                qasm2qcx.InvalidQubitOperandException, "not declared"):
            convert(source)

    def test_rejects_unsupported_quantum_statement(self) -> None:
        source = """
            OPENQASM 3.0;
            qubit q;
            delay[1ns] q;
        """

        with self.assertRaisesRegex(qasm2qcx.UnsupportedOpenQASMError, "delay"):
            convert(source)

    def test_converts_qubit_index_range(self) -> None:
        source = """
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[2] q;
            x q[0:1];
        """

        self.assertEqual(convert(source), ["QUBITS 2", "X 0", "X 1"])

    def test_converts_negative_controlled_global_phase(self) -> None:
        source = """
            OPENQASM 3.0;
            qubit q;
            negctrl @ gphase(pi) q;
        """

        self.assertEqual(convert(source), ['QUBITS 1', 'U1 0#0 :PI'])

    def test_rejects_bit_global_phase_parameter(self) -> None:
        source = """
            OPENQASM 3.0;
            bit phase;
            gphase(phase);
        """

        with self.assertRaises(qasm2qcx.WrongParameterTypeException):
            convert(source)

    def test_rejects_nonpositive_qubit_register_size(self) -> None:
        source = """
            OPENQASM 3.0;
            qubit[0] q;
        """

        with self.assertRaisesRegex(
                qasm2qcx.InvalidDeclarationException, "positive size"):
            convert(source)


class ClassicalScalarTests(unittest.TestCase):
    def test_maps_narrow_uint_to_qcx_uint_with_masks(self) -> None:
        source = """
            OPENQASM 3.0;
            uint[8] value = 3;
            value += 2;
            float promoted = value;
            qubit q;
        """

        self.assertEqual(
            convert(source),
            [
                "QUBITS 1",
                "VAR VALUE31 UINT",
                "LET VALUE31 := 3",
                "LET VALUE31 &= 255",
                "VAR QASM2QCX_INT_0 INT",
                "LET QASM2QCX_INT_0 := :INT:VALUE31",
                "LET QASM2QCX_INT_0 += 2",
                "LET VALUE31 := :UINT:QASM2QCX_INT_0",
                "LET VALUE31 &= 255",
                "VAR PROMOTED255 REAL",
                "LET PROMOTED255 := :REAL:VALUE31",
            ],
        )

    def test_emits_valid_complex_literal_operations(self) -> None:
        source = """
            OPENQASM 3.0;
            complex[float[64]] value = 1.0 + 2.0im;
            value *= 2;
            qubit q;
        """

        lines = convert(source)
        self.assertIn("VAR VALUE31 COMPLEX", lines)
        self.assertIn("LET QASM2QCX_COMPLEX_0 := :COMPLEX:1.0", lines)
        self.assertIn("LET QASM2QCX_COMPLEX_1 := :I", lines)
        self.assertIn("LET QASM2QCX_COMPLEX_1 *= :COMPLEX:2.0", lines)
        self.assertIn("LET VALUE31 := QASM2QCX_COMPLEX_0", lines)
        self.assertEqual(lines[-1], "LET VALUE31 *= :COMPLEX:2.0")
        self.assertTrue(all("j" not in line for line in lines))

    def test_promotes_int_and_float_to_complex(self) -> None:
        source = """
            OPENQASM 3.0;
            int integer = 2;
            float real = integer;
            complex number = real;
            qubit q;
        """

        lines = convert(source)
        self.assertIn("LET REAL15 := :REAL:INTEGER127", lines)
        self.assertIn("LET NUMBER63 := :COMPLEX:REAL15", lines)

    def test_rejects_narrowing_implicit_cast(self) -> None:
        source = """
            OPENQASM 3.0;
            float real = 2.5;
            int integer = real;
            qubit q;
        """

        with self.assertRaises(qasm2qcx.NoImplicitCastException):
            convert(source)

    def test_supports_explicit_numeric_casts(self) -> None:
        source = """
            OPENQASM 3.0;
            float real = 2.5;
            int integer = int(real);
            uint unsigned = uint(real);
            complex number = complex(real);
            qubit q;
        """

        lines = convert(source)
        self.assertIn("LET QASM2QCX_INT_0 := :INT:REAL15", lines)
        self.assertIn("LET INTEGER127 := QASM2QCX_INT_0", lines)
        self.assertIn("LET QASM2QCX_UINT_0 := :UINT:REAL15", lines)
        self.assertIn("LET UNSIGNED255 := QASM2QCX_UINT_0", lines)
        self.assertIn("LET QASM2QCX_COMPLEX_0 := :COMPLEX:REAL15", lines)
        self.assertIn("LET NUMBER63 := QASM2QCX_COMPLEX_0", lines)

    def test_uses_typed_constant_values(self) -> None:
        source = """
            OPENQASM 3.0;
            include "stdgates.inc";
            const uint count = 3;
            const float theta = count / 2.0;
            qubit q;
            p(theta) q;
        """

        self.assertEqual(convert(source), ["QUBITS 1", "U1 0 1.5"])

    def test_rejects_constant_use_before_declaration(self) -> None:
        sources = [
            """
                OPENQASM 3.0;
                qubit q;
                U(theta, 0, 0) q;
                const float theta = 1.0;
            """,
            """
                OPENQASM 3.0;
                float value = theta;
                const float theta = 1.0;
            """,
            """
                OPENQASM 3.0;
                bit[width] value;
                const uint width = 2;
            """,
            """
                OPENQASM 3.0;
                gphase(theta);
                const float theta = 1.0;
            """,
        ]

        for source in sources:
            with self.subTest(source=source):
                with self.assertRaises(qasm2qcx.NoVariableNameException):
                    convert(source)

    def test_integer_constant_division_truncates_toward_zero(self) -> None:
        source = """
            OPENQASM 3.0;
            const int positive = 3 / 2;
            const int negative = -3 / 2;
            int positive_result = positive;
            int negative_result = negative;
            qubit q;
        """

        lines = convert(source)
        self.assertIn("LET POSITIVE_RESULT32703 := 1", lines)
        self.assertIn("LET NEGATIVE_RESULT32703 := -1", lines)

    def test_rejects_duplicate_identifier(self) -> None:
        source = """
            OPENQASM 3.0;
            int value;
            float value;
            qubit q;
        """

        with self.assertRaisesRegex(
                qasm2qcx.DuplicateIdentifierException, "value"):
            convert(source)

    def test_rejects_undeclared_variable(self) -> None:
        source = """
            OPENQASM 3.0;
            missing = 1;
            qubit q;
        """

        with self.assertRaisesRegex(qasm2qcx.NoVariableNameException, "MISSING"):
            convert(source)

    def test_rejects_unsupported_classical_array_type(self) -> None:
        source = """
            OPENQASM 3.0;
            array[bit, 2] values;
        """

        with self.assertRaisesRegex(
                qasm2qcx.UnsupportedOpenQASMError, "classical array"):
            convert(source)

    def test_rejects_unary_arithmetic_on_bit(self) -> None:
        source = """
            OPENQASM 3.0;
            bit value;
            int result = -value;
        """

        with self.assertRaisesRegex(
                qasm2qcx.UnsupportedOpenQASMError,
                "unary arithmetic on bit values"):
            convert(source)


class BranchingTests(unittest.TestCase):
    def test_lowers_comparison_condition_and_if_else(self) -> None:
        source = """
            OPENQASM 3.0;
            int value = 2;
            int result = 0;
            if (value != 0) {
                result = 1;
            } else {
                result = 2;
            }
        """

        self.assertEqual(
            convert(source),
            [
                "QUBITS 0",
                "VAR VALUE31 INT",
                "LET VALUE31 := 2",
                "VAR RESULT63 INT",
                "LET RESULT63 := 0",
                "JUMPIF QASM2QCX_IF_0 VALUE31 \\= 0",
                "JUMP QASM2QCX_ELSE_0",
                "@QASM2QCX_IF_0",
                "LET RESULT63 := 1",
                "JUMP QASM2QCX_END_IF_0",
                "@QASM2QCX_ELSE_0",
                "LET RESULT63 := 2",
                "@QASM2QCX_END_IF_0",
            ],
        )

    def test_lowers_all_comparison_operators(self) -> None:
        source = """
            OPENQASM 3.0;
            int value = 2;
            if (value == 2) {}
            if (value != 2) {}
            if (value > 2) {}
            if (value < 2) {}
            if (value >= 2) {}
            if (value <= 2) {}
        """

        jumpif_lines = [
            line for line in convert(source) if line.startswith("JUMPIF")
        ]
        self.assertEqual(
            jumpif_lines,
            [
                "JUMPIF QASM2QCX_IF_0 VALUE31 == 2",
                "JUMPIF QASM2QCX_IF_1 VALUE31 \\= 2",
                "JUMPIF QASM2QCX_IF_2 VALUE31 > 2",
                "JUMPIF QASM2QCX_IF_3 VALUE31 < 2",
                "JUMPIF QASM2QCX_IF_4 VALUE31 >= 2",
                "JUMPIF QASM2QCX_IF_5 VALUE31 <= 2",
            ],
        )

    def test_lowers_indexed_bit_condition(self) -> None:
        source = """
            OPENQASM 3.0;
            bit[2] flags = "10";
            int result = 0;
            if (flags[1] == 1) {
                result = 1;
            }
        """

        lines = convert(source)
        self.assertIn(
            "JUMPIF QASM2QCX_IF_0 FLAGS31:1 == 1", lines)

    def test_promotes_int_operand_for_float_comparison(self) -> None:
        source = """
            OPENQASM 3.0;
            int value = 2;
            if (value < 2.5) {}
        """

        lines = convert(source)
        self.assertIn("VAR QASM2QCX_REAL_0 REAL", lines)
        self.assertIn(
            "LET QASM2QCX_REAL_0 := :REAL:VALUE31", lines)
        self.assertIn(
            "JUMPIF QASM2QCX_IF_0 QASM2QCX_REAL_0 < 2.5", lines)

    def test_materializes_native_constant_on_left_side(self) -> None:
        source = """
            OPENQASM 3.0;
            if (pi > 3.0) {}
            if (tau > 6.0) {}
        """

        lines = convert(source)
        self.assertIn("VAR QASM2QCX_REAL_0 REAL", lines)
        self.assertIn("LET QASM2QCX_REAL_0 := :PI", lines)
        self.assertIn(
            "JUMPIF QASM2QCX_IF_0 QASM2QCX_REAL_0 > 3.0", lines)
        self.assertIn("LET QASM2QCX_REAL_0 := :TWO_PI", lines)
        self.assertIn(
            "JUMPIF QASM2QCX_IF_1 QASM2QCX_REAL_0 > 6.0", lines)

    def test_lowers_direct_scalar_bit_condition(self) -> None:
        source = """
            OPENQASM 3.0;
            bit condition = 1;
            if (condition) {}
        """

        self.assertEqual(convert(source), [
            "QUBITS 0", "VAR CONDITION511 INT", "LET CONDITION511 := 1",
            "JUMPIF QASM2QCX_IF_0 CONDITION511 \\= 0",
            "JUMP QASM2QCX_END_IF_0", "@QASM2QCX_IF_0",
            "@QASM2QCX_END_IF_0",
        ])

    def test_lowers_indexed_bit_and_negation(self) -> None:
        lines = convert('''OPENQASM 3.0; bit[1] flags = "1";
            if (!flags[0]) {} else { flags[0] = 0; }
            if (!!flags[0]) {}
            if (!(flags[0] == 1)) {}''')
        self.assertIn("JUMPIF QASM2QCX_ELSE_0 FLAGS31 \\= 0", lines)
        self.assertIn("JUMP QASM2QCX_IF_0", lines)
        self.assertIn("JUMPIF QASM2QCX_IF_1 FLAGS31 \\= 0", lines)
        self.assertIn("JUMPIF QASM2QCX_END_IF_2 FLAGS31 == 1", lines)

    def test_rejects_direct_non_bit_conditions(self) -> None:
        for declaration in (
                "int value = 1;", "float value = 1.0;",
                "complex value = 1.0;", 'bit[1] value = "1";',
                'bit[2] value = "01";'):
            for condition in ("value", "!value"):
                with self.subTest(declaration=declaration, condition=condition):
                    with self.assertRaises(qasm2qcx.UnsupportedOpenQASMError):
                        convert(f'OPENQASM 3.0; {declaration} '
                                f'if ({condition}) {{}}')

    def test_lowers_short_circuit_destinations(self) -> None:
        for operator, lhs_true, lhs_false in (
                ("&&", "QASM2QCX_CONDITION_0", "QASM2QCX_END_IF_0"),
                ("||", "QASM2QCX_IF_0", "QASM2QCX_CONDITION_0")):
            with self.subTest(operator=operator):
                lines = convert(f'''OPENQASM 3.0; bit a; bit b;
                    if (a {operator} b) {{}}''')
                self.assertEqual(lines[3:8], [
                    f"JUMPIF {lhs_true} A1 \\= 0", f"JUMP {lhs_false}",
                    "@QASM2QCX_CONDITION_0",
                    "JUMPIF QASM2QCX_IF_0 B1 \\= 0",
                    "JUMP QASM2QCX_END_IF_0",
                ])

    def test_nested_logical_conditions_have_unique_resolved_labels(self) -> None:
        lines = convert('''OPENQASM 3.0; bit a; bit b; int n = 1;
            if (!(a || b) && n > 0) {
                if (a || (b && n == 1)) { n = 2; }
            } else if (a && b) { n = 3; }''')
        labels = [line[1:] for line in lines if line.startswith("@")]
        targets = [line.split()[1] for line in lines
                   if line.startswith(("JUMP ", "JUMPIF "))]
        self.assertEqual(len(labels), len(set(labels)))
        self.assertEqual(sum(label.startswith("QASM2QCX_CONDITION_")
                             for label in labels), 5)
        self.assertTrue(set(targets) <= set(labels))

    def test_rejects_unsupported_logical_operands_and_value_expressions(self) -> None:
        for statement in (
                "if (a && n) {}", "if (n || a) {}", "if (a & n) {}",
                "if (~n) {}"):
            with self.subTest(statement=statement):
                with self.assertRaises(qasm2qcx.UnsupportedOpenQASMError):
                    convert('OPENQASM 3.0; bit a = 1; int n = 1; '
                            + statement)

    def test_rejects_unsupported_direct_index_selections(self) -> None:
        for condition in ("flags[0:0]", "flags[{0}]"):
            with self.subTest(condition=condition):
                with self.assertRaises(qasm2qcx.UnsupportedOpenQASMError):
                    convert('OPENQASM 3.0; bit[2] flags; int index = 0; '
                            f'if ({condition}) {{}}')

    def test_lowers_nested_if_and_else_if_with_unique_labels(self) -> None:
        source = """
            OPENQASM 3.0;
            int value = 1;
            int result = 0;
            if (value == 1) {
                if (result == 0) {
                    result = 1;
                }
            } else if (value == 2) {
                result = 2;
            } else {
                result = 3;
            }
        """

        lines = convert(source)
        jumpif_targets = [
            line.split()[1]
            for line in lines
            if line.startswith("JUMPIF ")
        ]
        jump_targets = [
            line.split()[1]
            for line in lines
            if line.startswith("JUMP ")
        ]
        labels = {
            line[1:]
            for line in lines
            if line.startswith("@")
        }

        self.assertEqual(len(jumpif_targets), 3)
        self.assertEqual(len(set(jumpif_targets)), 3)
        self.assertTrue(set(jumpif_targets + jump_targets) <= labels)

    def test_lowers_measurement_controlled_gate(self) -> None:
        source = """
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[2] q;
            bit outcome = measure q[0];
            if (outcome == 0) {
                x q[1];
            }
        """

        self.assertEqual(
            convert(source),
            [
                "QUBITS 2",
                "VAR OUTCOME127 INT",
                "M 0",
                "LET OUTCOME127 := :OUTCOME",
                "JUMPIF QASM2QCX_IF_0 OUTCOME127 == 0",
                "JUMP QASM2QCX_END_IF_0",
                "@QASM2QCX_IF_0",
                "X 1",
                "@QASM2QCX_END_IF_0",
            ],
        )

    def test_rejects_block_local_declarations(self) -> None:
        declarations = (
            "int local = 1;",
            "const int local = 1;",
        )

        for declaration in declarations:
            source = f"""
                OPENQASM 3.0;
                int selector = 0;
                if (selector == 0) {{
                    {declaration}
                }}
            """
            with self.subTest(declaration=declaration):
                with self.assertRaisesRegex(
                        qasm2qcx.UnsupportedOpenQASMError,
                        "block-local declaration"):
                    convert(source)


class AmplitudePragmaTests(unittest.TestCase):
    def test_outputs_all_final_amplitudes(self) -> None:
        source = """
            OPENQASM 3.0;
            pragma riken_braket.amplitudes
            include "stdgates.inc";
            qubit q;
            h q;
        """

        self.assertEqual(
            convert(source),
            ["QUBITS 1", "H 0", "DO AMPLITUDES"],
        )

    def test_outputs_selected_final_amplitudes(self) -> None:
        source = """
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[2] q;
            h q[0];
            pragma riken_braket.amplitudes 0 3
            x q[1];
        """

        self.assertEqual(
            convert(source),
            ["QUBITS 2", "H 0", "X 1", "DO AMPLITUDES 0 3"],
        )

    def test_rejects_invalid_amplitude_indices(self) -> None:
        commands = [
            "riken_braket.amplitudes -1",
            "riken_braket.amplitudes 1.0",
            "riken_braket.amplitudes index",
        ]

        for command in commands:
            with self.subTest(command=command):
                source = f"OPENQASM 3.0;\npragma {command}\nqubit q;"
                with self.assertRaisesRegex(
                        qasm2qcx.InvalidPragmaException,
                        "Invalid amplitude index"):
                    convert(source)

    def test_rejects_duplicate_amplitude_indices(self) -> None:
        source = """
            OPENQASM 3.0;
            pragma riken_braket.amplitudes 0 0
            qubit q;
        """

        with self.assertRaisesRegex(
                qasm2qcx.InvalidPragmaException,
                "Duplicate amplitude index"):
            convert(source)

    def test_rejects_out_of_range_amplitude_index(self) -> None:
        source = """
            OPENQASM 3.0;
            pragma riken_braket.amplitudes 4
            qubit[2] q;
        """

        with self.assertRaisesRegex(
                qasm2qcx.InvalidPragmaException, "outside the state vector"):
            convert(source)

    def test_rejects_multiple_amplitudes_pragmas(self) -> None:
        source = """
            OPENQASM 3.0;
            pragma riken_braket.amplitudes
            qubit q;
            pragma riken_braket.amplitudes 0
        """

        with self.assertRaisesRegex(
                qasm2qcx.InvalidPragmaException, "may appear only once"):
            convert(source)

    def test_rejects_unsupported_pragma(self) -> None:
        source = """
            OPENQASM 3.0;
            pragma another_implementation.option
            qubit q;
        """

        with self.assertRaisesRegex(
                qasm2qcx.UnsupportedOpenQASMError,
                "another_implementation.option"):
            convert(source)


class BarrierTests(unittest.TestCase):
    def test_accepts_barrier_without_operands(self) -> None:
        source = """
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit q;
            x q;
            barrier;
            h q;
        """

        self.assertEqual(convert(source), ["QUBITS 1", "X 0", "H 0"])

    def test_accepts_supported_barrier_operands(self) -> None:
        source = """
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit scalar;
            qubit[2] register;
            x scalar;
            barrier scalar, register, register[1];
            h register[0];
        """

        self.assertEqual(convert(source), ["QUBITS 3", "X 0", "H 1"])

    def test_requires_barrier_qubits_to_be_declared(self) -> None:
        source = """
            OPENQASM 3.0;
            barrier q;
            qubit q;
        """

        with self.assertRaisesRegex(
                qasm2qcx.InvalidQubitOperandException, "not declared"):
            convert(source)

    def test_rejects_out_of_range_barrier_index(self) -> None:
        source = """
            OPENQASM 3.0;
            qubit[2] q;
            barrier q[2];
        """

        with self.assertRaisesRegex(
                qasm2qcx.InvalidQubitOperandException, "outside register"):
            convert(source)

    def test_accepts_static_barrier_selections(self) -> None:
        source = """
            OPENQASM 3.0;
            qubit[3] q;
            barrier q[0:2], q[{2, 0}];
        """

        self.assertEqual(convert(source), ["QUBITS 3"])

    def test_rejects_dynamic_barrier_indices(self) -> None:
        sources = [
            "OPENQASM 3.0; const int index = 0; qubit q; barrier q[index];",
            "OPENQASM 3.0; const int index = 0; qubit[2] q; barrier q[0:index];",
        ]

        for source in sources:
            with self.subTest(source=source):
                with self.assertRaisesRegex(
                        qasm2qcx.UnsupportedOpenQASMError,
                        "non-literal qubit index"):
                    convert(source)


class ResetTests(unittest.TestCase):
    def test_converts_single_qubit_reset(self) -> None:
        source = """
            OPENQASM 3.0;
            qubit q;
            reset q;
        """

        self.assertEqual(convert(source), ["QUBITS 1", "RESET 0"])

    def test_converts_register_reset(self) -> None:
        source = """
            OPENQASM 3.0;
            qubit scalar;
            qubit[3] register;
            reset register;
        """

        self.assertEqual(
            convert(source),
            ["QUBITS 4", "RESET 1", "RESET 2", "RESET 3"],
        )

    def test_converts_indexed_reset(self) -> None:
        source = """
            OPENQASM 3.0;
            qubit[3] q;
            reset q[1];
        """

        self.assertEqual(convert(source), ["QUBITS 3", "RESET 1"])

    def test_requires_reset_qubits_to_be_declared(self) -> None:
        source = """
            OPENQASM 3.0;
            reset q;
            qubit q;
        """

        with self.assertRaisesRegex(
                qasm2qcx.InvalidQubitOperandException, "not declared"):
            convert(source)

    def test_rejects_out_of_range_reset_index(self) -> None:
        source = """
            OPENQASM 3.0;
            qubit[2] q;
            reset q[2];
        """

        with self.assertRaisesRegex(
                qasm2qcx.InvalidQubitOperandException, "outside register"):
            convert(source)

    def test_converts_static_reset_selections(self) -> None:
        source = """
            OPENQASM 3.0;
            qubit[3] q;
            reset q[0:2];
            reset q[{2, 0}];
        """

        self.assertEqual(
            convert(source),
            ["QUBITS 3", "RESET 0", "RESET 1", "RESET 2", "RESET 2", "RESET 0"],
        )

    def test_rejects_dynamic_reset_indices(self) -> None:
        sources = [
            "OPENQASM 3.0; const int index = 0; qubit q; reset q[index];",
            "OPENQASM 3.0; const int index = 0; qubit[2] q; reset q[{0, index}];",
        ]

        for source in sources:
            with self.subTest(source=source):
                with self.assertRaisesRegex(
                        qasm2qcx.UnsupportedOpenQASMError,
                        "non-literal qubit index"):
                    convert(source)


class MeasurementAndBitTests(unittest.TestCase):
    def test_converts_single_qubit_measurement(self) -> None:
        source = """
            OPENQASM 3.0;
            qubit q;
            bit c = measure q;
        """

        self.assertEqual(
            convert(source),
            ["QUBITS 1", "VAR C1 INT", "M 0", "LET C1 := :OUTCOME"],
        )

    def test_converts_register_measurement(self) -> None:
        source = """
            OPENQASM 3.0;
            qubit[2] q;
            bit[2] c = measure q;
        """

        self.assertEqual(
            convert(source),
            [
                "QUBITS 2",
                "VAR C1 INT 2",
                "M 0",
                "LET C1:0 := :OUTCOME",
                "M 1",
                "LET C1:1 := :OUTCOME",
            ],
        )

    def test_converts_indexed_measurement_assignment(self) -> None:
        source = """
            OPENQASM 3.0;
            qubit[2] q;
            bit[2] c;
            c[1] = measure q[0];
        """

        self.assertEqual(
            convert(source),
            [
                "QUBITS 2",
                "VAR C1 INT 2",
                "M 0",
                "LET C1:1 := :OUTCOME",
            ],
        )

    def test_converts_measurement_without_target(self) -> None:
        source = """
            OPENQASM 3.0;
            qubit[2] q;
            measure q;
        """

        self.assertEqual(convert(source), ["QUBITS 2", "M 0", "M 1"])

    def test_initializes_and_copies_bit_values(self) -> None:
        source = """
            OPENQASM 3.0;
            bit scalar = 1;
            bit[3] source = "101";
            bit[3] destination = source;
            destination[1] = source[0];
        """

        self.assertEqual(
            convert(source),
            [
                "QUBITS 0",
                "VAR SCALAR63 INT",
                "LET SCALAR63 := 1",
                "VAR SOURCE63 INT 3",
                "LET SOURCE63:0 := 1",
                "LET SOURCE63:1 := 0",
                "LET SOURCE63:2 := 1",
                "VAR DESTINATION2047 INT 3",
                "LET DESTINATION2047:0 := SOURCE63:0",
                "LET DESTINATION2047:1 := SOURCE63:1",
                "LET DESTINATION2047:2 := SOURCE63:2",
                "LET DESTINATION2047:1 := SOURCE63:0",
            ],
        )

    def test_rejects_scalar_and_register_bit_copies(self) -> None:
        sources = [
            """
                OPENQASM 3.0;
                bit[1] source = "1";
                bit target = source;
            """,
            """
                OPENQASM 3.0;
                bit source = 1;
                bit[1] target = source;
            """,
            """
                OPENQASM 3.0;
                bit[1] source = "1";
                bit[1] target;
                target = source[0];
            """,
        ]

        for source in sources:
            with self.subTest(source=source):
                with self.assertRaisesRegex(
                        qasm2qcx.InvalidBitOperandException,
                        "both be scalars or both be registers|scalar bit"):
                    convert(source)

    def test_rejects_bit_literals_for_wrong_target_kind(self) -> None:
        sources = [
            'OPENQASM 3.0; bit target = "1";',
            'OPENQASM 3.0; bit[1] target = 1;',
        ]

        for source in sources:
            with self.subTest(source=source):
                with self.assertRaises(qasm2qcx.InvalidBitOperandException):
                    convert(source)

    def test_accepts_scalar_and_size_one_register_bit_forms(self) -> None:
        source = """
            OPENQASM 3.0;
            bit scalar = 1;
            bit[1] register = "1";
            bit scalar_copy = register[0];
            bit[1] register_copy = register;
        """

        lines = convert(source)
        self.assertIn("LET SCALAR_COPY2031 := REGISTER255", lines)
        self.assertIn("LET REGISTER_COPY8175 := REGISTER255", lines)

    def test_accepts_constant_bit_register_size(self) -> None:
        source = """
            OPENQASM 3.0;
            const uint width = 2;
            bit[width] c;
        """

        self.assertEqual(convert(source), ["QUBITS 0", "VAR C1 INT 2"])

    def test_rejects_nonpositive_bit_register_size(self) -> None:
        source = """
            OPENQASM 3.0;
            bit[0] c;
        """

        with self.assertRaisesRegex(
                qasm2qcx.InvalidDeclarationException, "positive size"):
            convert(source)

    def test_rejects_measurement_size_mismatch(self) -> None:
        source = """
            OPENQASM 3.0;
            qubit[2] q;
            bit[1] c = measure q;
        """

        with self.assertRaisesRegex(
                qasm2qcx.MeasurementSizeMismatchException,
                "2 qubit.*1 bit"):
            convert(source)

    def test_rejects_mixed_scalar_and_register_measurement(self) -> None:
        source = """
            OPENQASM 3.0;
            qubit[1] q;
            bit c = measure q;
        """

        with self.assertRaisesRegex(
                qasm2qcx.InvalidBitOperandException,
                "both be scalars or both be registers"):
            convert(source)

    def test_rejects_non_bit_measurement_target(self) -> None:
        source = """
            OPENQASM 3.0;
            qubit q;
            int result = measure q;
        """

        with self.assertRaisesRegex(
                qasm2qcx.InvalidBitOperandException, "not a bit variable"):
            convert(source)

    def test_rejects_out_of_range_bit_index(self) -> None:
        source = """
            OPENQASM 3.0;
            qubit q;
            bit[2] c;
            c[2] = measure q;
        """

        with self.assertRaisesRegex(
                qasm2qcx.InvalidBitOperandException, "outside variable"):
            convert(source)


if __name__ == "__main__":
    unittest.main()
