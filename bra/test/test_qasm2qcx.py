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

    def test_rejects_unsupported_quantum_statement(self) -> None:
        source = """
            OPENQASM 3.0;
            qubit q;
            reset q;
        """

        with self.assertRaisesRegex(qasm2qcx.UnsupportedOpenQASMError, "reset"):
            convert(source)

    def test_rejects_qubit_index_range(self) -> None:
        source = """
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[2] q;
            x q[0:1];
        """

        with self.assertRaisesRegex(
                qasm2qcx.UnsupportedOpenQASMError, "qubit index ranges"):
            convert(source)

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
                "both be scalars or both be complete registers"):
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
