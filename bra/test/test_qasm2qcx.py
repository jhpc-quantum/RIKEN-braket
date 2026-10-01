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

    def test_does_not_silently_drop_unsupported_gate(self) -> None:
        source = """
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit q;
            id q;
        """

        with self.assertRaisesRegex(
                qasm2qcx.UnsupportedOpenQASMError, "gate id"):
            convert(source)

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
                "U3 0 0.7 0.8 0.9",
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

    def test_rejects_unsupported_quantum_statement(self) -> None:
        source = """
            OPENQASM 3.0;
            qubit q;
            reset q;
        """

        with self.assertRaisesRegex(qasm2qcx.UnsupportedOpenQASMError, "reset"):
            convert(source)


class KnownDefectTests(unittest.TestCase):

    @unittest.expectedFailure
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


if __name__ == "__main__":
    unittest.main()
