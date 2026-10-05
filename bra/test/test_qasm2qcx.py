#!/usr/bin/env python3

import importlib.util
from pathlib import Path
import unittest


CONVERTER_PATH = Path(__file__).parents[1] / "qcx" / "qasm2qcx.py"
SPEC = importlib.util.spec_from_file_location("qasm2qcx", CONVERTER_PATH)
qasm2qcx = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(qasm2qcx)


def convert(source: str) -> list[str]:
    return qasm2qcx.convert(source)


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

    def test_rejects_boolean_arrays_local_declarations_and_duplicates(self) -> None:
        for source in (
                'array[bool, 2] a;',
                'if (true) { bool a = false; }'):
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
        self.assertIn('LET U1 := A1', lines)
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
                'VAR B1 INT', 'LET B1 := 0', 'VAR C1 REAL', 'LET C1 := 1.0',
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
                'bool a = bool(flags[index]);', 'bit[1] copy = true;',
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
        self.assertIn("LET QASM2QCX_REAL_0 += :REAL:PHI7", lines)
        self.assertIn("LET QASM2QCX_REAL_0 += :REAL:LAMBDA63", lines)
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
            ["QUBITS 2", "U1 0 7.0", "CU3 0 1 1 2 3"],
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
        self.assertIn("LET QASM2QCX_REAL_0 += PHI7", lines)
        self.assertIn("LET QASM2QCX_REAL_0 += LAMBDA63", lines)
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
                "PHASE 1.2",
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
        self.assertIn("LET QASM2QCX_REAL_0 += :REAL:PHI7", lines)
        self.assertIn("LET QASM2QCX_REAL_0 += LAMBDA63", lines)
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
                "PHASE 3.0",
                "U3 0 1 2 3",
                "PHASE 3.0",
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

    def test_rejects_global_phase_modifier(self) -> None:
        source = """
            OPENQASM 3.0;
            qubit q;
            ctrl @ gphase(pi) q;
        """

        with self.assertRaisesRegex(
                qasm2qcx.UnsupportedOpenQASMError, "modifiers on gphase"):
            convert(source)

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
    def test_maps_uint_to_qcx_int(self) -> None:
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
                "VAR VALUE31 INT",
                "LET VALUE31 := 3",
                "LET VALUE31 += 2",
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
        self.assertIn("LET UNSIGNED255 := QASM2QCX_INT_0", lines)
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

    def test_rejects_classical_array(self) -> None:
        source = """
            OPENQASM 3.0;
            array[int[32], 2] values;
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
                "if (a && n) {}", "if (n || a) {}", "if (a & a) {}",
                "if (~a) {}"):
            with self.subTest(statement=statement):
                with self.assertRaises(qasm2qcx.UnsupportedOpenQASMError):
                    convert('OPENQASM 3.0; bit a = 1; int n = 1; '
                            + statement)

    def test_rejects_unsupported_direct_index_selections(self) -> None:
        for condition in ("flags[0:0]", "flags[{0}]", "flags[index]"):
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
