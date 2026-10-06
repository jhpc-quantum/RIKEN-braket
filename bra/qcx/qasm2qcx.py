import sys
import itertools
import math
import enum
import dataclasses

import openqasm3.parser
import openqasm3.ast as ast
import openqasm3.visitor as visitor

ValueType = enum.Enum(
    'ValueType', [('INT', 1), ('FLOAT', 2), ('BIT', 3), ('COMPLEX', 4), ('BOOL', 5)])
ValueKind = enum.Enum('ValueKind', [('LITERAL', 1), ('TEMPORARY', 2), ('LVALUE', 3)])
# CONST_ARITHMETIC => ARITHMETIC
ExpressionKind = enum.Enum('ExpressionKind', [('ARITHMETIC', 1), ('CONST_ARITHMETIC', 2), ('CONDITIONAL', 3)])


class QASM2QCXError(Exception):
    """Base class for conversion errors."""


class UnsupportedOpenQASMError(QASM2QCXError):
    def __init__(self, construct: str) -> None:
        self.construct = construct

    def __str__(self) -> str:
        return f'Unsupported OpenQASM construct: {self.construct}'


class WrongGateArityException(QASM2QCXError):
    def __init__(
            self, gate_name: str, expected_parameters: int,
            expected_qubits: int, actual_parameters: int,
            actual_qubits: int) -> None:
        self.gate_name = gate_name
        self.expected_parameters = expected_parameters
        self.expected_qubits = expected_qubits
        self.actual_parameters = actual_parameters
        self.actual_qubits = actual_qubits

    def __str__(self) -> str:
        return (
            f'Gate {self.gate_name} expects {self.expected_parameters} '
            f'parameter(s) and {self.expected_qubits} qubit operand(s), but got '
            f'{self.actual_parameters} and {self.actual_qubits}'
        )


class InvalidQubitOperandException(QASM2QCXError):
    def __init__(self, message: str) -> None:
        self.message = message

    def __str__(self) -> str:
        return self.message


class InvalidBitOperandException(QASM2QCXError):
    def __init__(self, message: str) -> None:
        self.message = message

    def __str__(self) -> str:
        return self.message


class InvalidDeclarationException(QASM2QCXError):
    def __init__(self, message: str) -> None:
        self.message = message

    def __str__(self) -> str:
        return self.message


class InvalidPragmaException(QASM2QCXError):
    def __init__(self, message: str) -> None:
        self.message = message

    def __str__(self) -> str:
        return self.message


class InvalidLoopRangeException(QASM2QCXError):
    def __init__(self, message: str) -> None:
        self.message = message

    def __str__(self) -> str:
        return self.message


class MeasurementSizeMismatchException(QASM2QCXError):
    def __init__(self, num_qubits: int, num_bits: int) -> None:
        self.num_qubits = num_qubits
        self.num_bits = num_bits

    def __str__(self) -> str:
        return (
            f'Measurement has {self.num_qubits} qubit(s), but its target has '
            f'{self.num_bits} bit(s)')


class DuplicateIdentifierException(QASM2QCXError):
    def __init__(self, identifier: str) -> None:
        self.identifier = identifier

    def __str__(self) -> str:
        return f'Identifier {self.identifier} is declared more than once'


class WrongBroadcastingException(QASM2QCXError):
    def __str__(self):
        return 'Wrong broadcasting; see https://openqasm.com/language/gates.html#broadcasting'

class WrongClassicalDeclarationException(QASM2QCXError):
    def __init__(self, variable_name: str):
        self.__variable_name = variable_name

    def __str__(self):
        return f'Wrong declaration of classical variable: {self.__variable_name}'

class UninitializedValueException(QASM2QCXError):
    def __str__(self):
        return 'Wrong value type'

class NoImplicitCastException(QASM2QCXError):
    def __str__(self):
        return 'No implicit cast'

class NoVariableNameException(QASM2QCXError):
    def __init__(self, variable_name: str):
        self.__variable_name = variable_name

    def __str__(self):
        return f'No variable {self.__variable_name} is found'

class WrongParameterTypeException(QASM2QCXError):
    def __init__(self, value: str):
        self.__value = value

    def __str__(self):
        return f'The type of variable {self.__value} is wrong'

class NoConstantExpressionException(QASM2QCXError):
    def __str__(self):
        return 'No constant expression'

class ZeroDivisorException(QASM2QCXError):
    def __init__(self, operator: str) -> None:
        self.operator = operator

    def __str__(self):
        return f'Constant {self.operator} expression divisor must not be zero'

class WrongConstantVariableException(QASM2QCXError):
    def __str__(self):
        return 'Wrong constant variable'


@dataclasses.dataclass(frozen=True)
class _LoopContext:
    break_label: str
    continue_label: str


class QASM2QCXConverter(visitor.QASMVisitor):
    MAX_LOOP_ITERATIONS = 10000
    MAX_EXPANDED_LOOP_STATEMENTS = 100000
    MAX_LOOP_OUTPUT_LINES = 1000000
    default_gates_qcx_map: dict[str, str] = {'U': 'U3'}

    stdgates_qcx_map: dict[str, str] = {
            'p': 'U1', 'phase': 'U1', 'u1': 'U1', 'u2': 'U2', 'u3': 'U3',
            'x': 'X', 'y': 'Y', 'z': 'Z', 'h': 'H', 'id': 'I',
            's': 'S', 'sdg': 'S+', 't': 'T', 'tdg': 'T+',
            'sx': 'SX', 'rx': 'EX', 'ry': 'EY', 'rz': 'EZ',
            'cx': 'CX', 'CX': 'CX', 'cy': 'CY', 'cz': 'CZ', 'cp': 'CU1', 'cphase': 'CU1',
            'crx': 'CEX', 'cry': 'CEY', 'crz': 'CEZ', 'ch': 'CH',
            'cu': 'CU3', 'swap': 'SWAP', 'ccx': 'CCX', 'cswap': 'CSWAP'}

    gate_signatures: dict[str, tuple[int, int]] = {
            'U': (3, 1),
            'p': (1, 1), 'phase': (1, 1), 'u1': (1, 1), 'u2': (2, 1), 'u3': (3, 1),
            'x': (0, 1), 'y': (0, 1), 'z': (0, 1), 'h': (0, 1), 'id': (0, 1),
            's': (0, 1),
            'sdg': (0, 1), 't': (0, 1), 'tdg': (0, 1), 'sx': (0, 1),
            'rx': (1, 1), 'ry': (1, 1), 'rz': (1, 1),
            'cx': (0, 2), 'CX': (0, 2), 'cy': (0, 2), 'cz': (0, 2),
            'cp': (1, 2), 'cphase': (1, 2), 'crx': (1, 2), 'cry': (1, 2),
            'crz': (1, 2), 'ch': (0, 2), 'cu': (4, 2), 'swap': (0, 2),
            'ccx': (0, 3), 'cswap': (0, 3)}

    def __init__(self, qasm_ast_root: ast.Program) -> None:
        self.__value: str | int | float | complex | None = None
        self.__value_type: ValueType | None = None
        self.__value_kind: ValueKind | None = None
        self.__expression_kind: ExpressionKind | None = None
        self.__declared_temporary_variables: set[str] = set()
        self.__used_temporary_variables: set[str] = set()
        self.__reserved_variable_names: set[str] = set()
        self.__source_identifiers: set[str] = set()

        self.__int_variable_name_size_map: dict[str, int] = {}
        self.__float_variable_name_size_map: dict[str, int] = {}
        self.__bit_variable_name_size_map: dict[str, int] = {}
        self.__bool_variable_names: set[str] = set()
        self.__sized_bit_variables: set[str] = set()
        self.__complex_variable_name_size_map: dict[str, int] = {}

        self.__const_int_variable_name_values_map: dict[str, list[int]] = {}
        self.__const_float_variable_name_values_map: dict[str, list[float]] = {}
        self.__const_complex_variable_name_values_map: dict[str, list[complex]] = {}
        self.__const_bool_variable_values_map: dict[str, int] = {}
        self.__declared_constant_variables: set[str] = set()

        self.__is_stdgates_included: bool = False
        self.__quantum_registers: dict[str, int] = {}
        self.__sized_quantum_registers: set[str] = set()
        self.__declared_quantum_registers: set[str] = set()
        self.__amplitude_indices: list[int] | None = None
        self.__branch_index: int = 0
        self.__condition_index: int = 0
        self.__branch_depth: int = 0
        self.__boolean_expression_index: int = 0
        self.__evaluate_constant: bool = True
        self.__loop_bindings: list[dict[str, int]] = []
        self.__loop_contexts: list[_LoopContext] = []
        self.__loop_index = 0
        self.__loop_iterations = 0
        self.__expanded_loop_statements = 0

        self.__is_initialization_process = True
        self.visit(qasm_ast_root)
        self.__is_initialization_process = False
        # Both passes traverse loops, but each checks the same independent
        # conversion-wide budget rather than charging the source twice.
        self.__loop_iterations = 0
        self.__expanded_loop_statements = 0
        self.__loop_index = 0

        self.__quantum_register_names: list[str] = list(self.__quantum_registers.keys())
        self.__first_qubit_indices: list[int] = list(itertools.accumulate(self.__quantum_registers.values(), initial=0))

        self.__qcx_lines: list[str] = [f'QUBITS {sum(self.__quantum_registers.values())}']
        self.__current: int = 0

    def __iter__(self):
        return self

    def __next__(self) -> str:
        if self.__current >= len(self.__qcx_lines):
            raise StopIteration

        self.__current += 1
        return self.__qcx_lines[self.__current - 1]

    def visit(self, node: ast.QASMNode, context=None):
        inside_loop = bool(self.__loop_bindings)
        if inside_loop and isinstance(node, ast.Statement):
            self.__expanded_loop_statements += 1
            if self.__expanded_loop_statements > self.MAX_EXPANDED_LOOP_STATEMENTS:
                raise InvalidLoopRangeException(
                    f'For-loop expansion exceeds {self.MAX_EXPANDED_LOOP_STATEMENTS} statements')
        result = super().visit(node, context)
        if (inside_loop and not self.__is_initialization_process
                and len(self.__qcx_lines) > self.MAX_LOOP_OUTPUT_LINES):
            raise InvalidLoopRangeException(
                f'For-loop output exceeds {self.MAX_LOOP_OUTPUT_LINES} QCX lines')
        return result

    def visit_Program(self, program: ast.Program) -> None:
        for statement in program.statements:
            self.visit(statement)

        if self.__is_initialization_process:
            if self.__amplitude_indices is None:
                return

            num_qubits = sum(self.__quantum_registers.values())
            for index in self.__amplitude_indices:
                if index.bit_length() > num_qubits:
                    raise InvalidPragmaException(
                        f'Amplitude index {index} is outside the state vector '
                        f'for {num_qubits} qubit(s)')
            return

        if self.__amplitude_indices is not None:
            suffix = ''.join(
                f' {index}' for index in self.__amplitude_indices)
            self.__qcx_lines.append(f'DO AMPLITUDES{suffix}')

    def __type_of(self, identifier_name: str) -> ValueType:
        if any(identifier_name == self.__capitalize_variable_name(name)
               for scope in self.__loop_bindings for name in scope):
            raise UnsupportedOpenQASMError('indexed or register use of a for-loop iterator')
        if identifier_name in self.__bool_variable_names:
            return ValueType.BOOL
        elif identifier_name in self.__bit_variable_name_size_map:
            return ValueType.BIT
        elif identifier_name in self.__int_variable_name_size_map:
            return ValueType.INT
        elif identifier_name in self.__float_variable_name_size_map:
            return ValueType.FLOAT
        elif identifier_name in self.__complex_variable_name_size_map:
            return ValueType.COMPLEX

        raise NoVariableNameException(identifier_name)

    def __capitalize_variable_name(self, variable_name: str) -> str:
        capitalized = variable_name.upper()

        lastnum = 0
        for c1, c2 in zip(variable_name, capitalized):
            lastnum <<= 1
            lastnum += 0 if c1 == c2 else 1

        return f'{capitalized}{lastnum}'

    def __register_source_identifier(self, identifier: str) -> None:
        if identifier in self.__source_identifiers:
            raise DuplicateIdentifierException(identifier)
        self.__source_identifiers.add(identifier)

    def __add_new_temporary_variable(self, value_type: ValueType) -> str:
        match value_type:
            case ValueType.INT | ValueType.BOOL:
                type_str = 'INT'
            case ValueType.FLOAT:
                type_str = 'REAL'
            case ValueType.COMPLEX:
                type_str = 'COMPLEX'

        temporary_variable_index: int = 0
        temporary_variable: str = f'QASM2QCX_{type_str}_{temporary_variable_index}'
        while (temporary_variable in self.__used_temporary_variables
               or temporary_variable in self.__reserved_variable_names):
            temporary_variable_index += 1
            temporary_variable = f'QASM2QCX_{type_str}_{temporary_variable_index}'

        self.__used_temporary_variables.add(temporary_variable)
        if temporary_variable not in self.__declared_temporary_variables:
            self.__declared_temporary_variables.add(temporary_variable)
            self.__qcx_lines.append(f'VAR {temporary_variable} {type_str}')

        return temporary_variable

    def __release_temporary_variable(self, temporary_variable: str) -> None:
        self.__used_temporary_variables.remove(temporary_variable)

    def visit_Identifier(self, expression: ast.Identifier) -> None:
        if self.__expression_kind is None:
            return

        for scope in reversed(self.__loop_bindings):
            if expression.name in scope:
                self.__value = scope[expression.name]
                self.__value_type = ValueType.INT
                self.__value_kind = ValueKind.LITERAL
                return

        is_user_constant = any(
            expression.name in constant_values
            for constant_values in (
                self.__const_int_variable_name_values_map,
                self.__const_float_variable_name_values_map,
                self.__const_complex_variable_name_values_map,
                self.__const_bool_variable_values_map,
            )
        )
        if (is_user_constant and not self.__is_initialization_process
                and expression.name not in self.__declared_constant_variables):
            raise NoVariableNameException(expression.name)

        builtin_constants = {
            'pi': (math.pi, ':PI'),
            'tau': (math.tau, ':TWO_PI'),
            'euler': (math.e, None),
        }
        if expression.name in builtin_constants:
            literal_value, qcx_symbol = builtin_constants[expression.name]
            if (self.__expression_kind == ExpressionKind.CONST_ARITHMETIC
                    or qcx_symbol is None):
                self.__value = literal_value
                self.__value_kind = ValueKind.LITERAL
            else:
                self.__value = qcx_symbol
                self.__value_kind = ValueKind.LVALUE
            self.__value_type = ValueType.FLOAT
        elif expression.name in self.__const_bool_variable_values_map:
            self.__value = self.__const_bool_variable_values_map[expression.name]
            self.__value_type = ValueType.BOOL
            self.__value_kind = ValueKind.LITERAL
        elif expression.name in self.__const_int_variable_name_values_map:
            self.__value = self.__const_int_variable_name_values_map[expression.name][0]
            self.__value_type = ValueType.INT
            self.__value_kind = ValueKind.LITERAL
        elif expression.name in self.__const_float_variable_name_values_map:
            self.__value = self.__const_float_variable_name_values_map[expression.name][0]
            self.__value_type = ValueType.FLOAT
            self.__value_kind = ValueKind.LITERAL
        elif expression.name in self.__const_complex_variable_name_values_map:
            self.__value = self.__const_complex_variable_name_values_map[expression.name][0]
            self.__value_type = ValueType.COMPLEX
            self.__value_kind = ValueKind.LITERAL
        else:
            if self.__expression_kind == ExpressionKind.CONST_ARITHMETIC:
                raise NoVariableNameException(expression.name)

            self.__value = self.__capitalize_variable_name(expression.name)
            self.__value_type = self.__type_of(self.__value)
            self.__value_kind = ValueKind.LVALUE
            if (self.__value_type == ValueType.BIT
                    and self.__value in self.__sized_bit_variables):
                raise UnsupportedOpenQASMError('whole bit register in scalar expression')

    def visit_UnaryExpression(self, expression: ast.UnaryExpression) -> None:
        if self.__expression_kind is None:
            return

        if expression.op == ast.UnaryOperator['!']:
            self.__visit_boolean_expression(expression)
            return
        self.visit(expression.expression)
        if self.__value is None or self.__value_type is None or self.__value_kind is None:
            raise UninitializedValueException
        if self.__value_type == ValueType.BIT:
            raise UnsupportedOpenQASMError('unary arithmetic on bit values')
        if self.__value_type == ValueType.BOOL:
            raise UnsupportedOpenQASMError('unary value expression on Boolean values')

        if expression.op != ast.UnaryOperator['-']:
            raise UnsupportedOpenQASMError(f'unary operator {expression.op.name}')

        if self.__value_kind == ValueKind.LITERAL:
            self.__value = -self.__value
            return
        if self.__expression_kind == ExpressionKind.CONST_ARITHMETIC:
            raise NoConstantExpressionException

        operand = self.__value
        operand_kind = self.__value_kind
        temporary_variable = self.__add_new_temporary_variable(self.__value_type)
        self.__qcx_lines.append(f'LET {temporary_variable} := {operand}')
        multiplier = ':COMPLEX:-1.0' if self.__value_type == ValueType.COMPLEX else '-1'
        self.__qcx_lines.append(f'LET {temporary_variable} *= {multiplier}')
        if operand_kind == ValueKind.TEMPORARY:
            self.__release_temporary_variable(operand)
        self.__value = temporary_variable
        self.__value_kind = ValueKind.TEMPORARY

    @staticmethod
    def __promoted_type(lhs_type: ValueType, rhs_type: ValueType) -> ValueType:
        arithmetic_types = (ValueType.INT, ValueType.FLOAT, ValueType.COMPLEX)
        if lhs_type not in arithmetic_types or rhs_type not in arithmetic_types:
            raise NoImplicitCastException
        if ValueType.COMPLEX in (lhs_type, rhs_type):
            return ValueType.COMPLEX
        if ValueType.FLOAT in (lhs_type, rhs_type):
            return ValueType.FLOAT
        return ValueType.INT

    def __converted_operand(
            self, value: str | int | float | complex, value_type: ValueType,
            value_kind: ValueKind, result_type: ValueType) -> tuple[str, set[str]]:
        if result_type == ValueType.BOOL:
            if value_type in (ValueType.BOOL, ValueType.BIT):
                return str(value), set()
            if (value_type == ValueType.INT and value_kind == ValueKind.LITERAL
                    and value in (0, 1)):
                return str(int(value)), set()
            raise NoImplicitCastException
        if value_type == ValueType.BOOL:
            if result_type in (ValueType.INT, ValueType.BIT):
                return str(value), set()
            # Boolean-to-numeric promotion preserves its canonical 0/1 value.
            value_type = ValueType.INT
        if value_kind == ValueKind.LITERAL:
            if result_type == ValueType.BIT:
                if value_type not in (ValueType.INT, ValueType.BIT) or value not in (0, 1):
                    raise NoImplicitCastException
                return str(int(value)), set()
            if result_type == ValueType.INT:
                if value_type != ValueType.INT:
                    raise NoImplicitCastException
                return str(int(value)), set()
            if result_type == ValueType.FLOAT:
                if value_type == ValueType.COMPLEX:
                    raise NoImplicitCastException
                return str(float(value)), set()

            complex_value = complex(value)
            if complex_value.imag == 0.0:
                return f':COMPLEX:{complex_value.real}', set()
            if complex_value.real == 0.0 and complex_value.imag == 1.0:
                return ':I', set()
            if complex_value.real == 0.0 and complex_value.imag == -1.0:
                return ':MINUS_I', set()

            temporary = self.__add_new_temporary_variable(ValueType.COMPLEX)
            self.__qcx_lines.append(
                f'LET {temporary} := :COMPLEX:{complex_value.real}')
            imaginary = self.__add_new_temporary_variable(ValueType.COMPLEX)
            imaginary_unit = ':I' if complex_value.imag > 0.0 else ':MINUS_I'
            self.__qcx_lines.append(f'LET {imaginary} := {imaginary_unit}')
            if abs(complex_value.imag) != 1.0:
                self.__qcx_lines.append(
                    f'LET {imaginary} *= :COMPLEX:{abs(complex_value.imag)}')
            self.__qcx_lines.append(f'LET {temporary} += {imaginary}')
            self.__release_temporary_variable(imaginary)
            return temporary, {temporary}

        if value_type == result_type:
            return str(value), set()
        if result_type == ValueType.FLOAT and value_type == ValueType.INT:
            return f':REAL:{value}', set()
        if result_type == ValueType.COMPLEX:
            return f':COMPLEX:{value}', set()
        raise NoImplicitCastException

    def __emit_assignment(
            self, variable_name: str, operator: str, variable_type: ValueType,
            value: str | int | float | complex, value_type: ValueType,
            value_kind: ValueKind) -> None:
        rhs, materialized_temporaries = self.__converted_operand(
            value, value_type, value_kind, variable_type)
        self.__qcx_lines.append(f'LET {variable_name} {operator} {rhs}')

        temporaries_to_release = set(materialized_temporaries)
        if value_kind == ValueKind.TEMPORARY:
            temporaries_to_release.add(str(value))
        for temporary in temporaries_to_release:
            self.__release_temporary_variable(temporary)

    def visit_BinaryExpression(self, expression: ast.BinaryExpression) -> None:
        if self.__expression_kind is None:
            return

        if expression.op in (
                ast.BinaryOperator['&&'], ast.BinaryOperator['||'],
                ast.BinaryOperator['=='], ast.BinaryOperator['!='],
                ast.BinaryOperator['<'], ast.BinaryOperator['<='],
                ast.BinaryOperator['>'], ast.BinaryOperator['>=']):
            self.__visit_boolean_expression(expression)
            return
        self.visit(expression.rhs)
        if self.__value is None or self.__value_type is None or self.__value_kind is None:
            raise UninitializedValueException
        rhs_value_type = self.__value_type
        rhs_value_kind = self.__value_kind
        rhs_value = self.__value

        self.visit(expression.lhs)
        if self.__value is None or self.__value_type is None or self.__value_kind is None:
            raise UninitializedValueException
        lhs_value_type = self.__value_type
        lhs_value_kind = self.__value_kind
        lhs_value = self.__value

        is_remainder = expression.op == ast.BinaryOperator['%']
        if is_remainder:
            if lhs_value_type != ValueType.INT or rhs_value_type != ValueType.INT:
                raise UnsupportedOpenQASMError('integer remainder requires int or uint operands')

        operators = {
            ast.BinaryOperator['+']: '+=',
            ast.BinaryOperator['-']: '-=',
            ast.BinaryOperator['*']: '*=',
            ast.BinaryOperator['/']: '/=',
        }
        if expression.op not in operators and not is_remainder:
            raise UnsupportedOpenQASMError(f'binary operator {expression.op.name}')

        result_type = self.__promoted_type(lhs_value_type, rhs_value_type)
        # A runtime expression may occur in a short-circuited operand. Do not
        # raise during folding for a division/remainder QCX might never execute.
        defer_division = (self.__expression_kind == ExpressionKind.ARITHMETIC
                          and expression.op in (ast.BinaryOperator['/'], ast.BinaryOperator['%'])
                          and rhs_value == 0)
        if (lhs_value_kind == ValueKind.LITERAL and rhs_value_kind == ValueKind.LITERAL
                and not defer_division):
            def divide(lhs, rhs):
                if rhs == 0:
                    raise ZeroDivisorException('/')
                if result_type != ValueType.INT:
                    return lhs / rhs

                return self.__integer_quotient(lhs, rhs)

            def remainder(lhs, rhs):
                if rhs == 0:
                    raise ZeroDivisorException('%')
                return lhs - self.__integer_quotient(lhs, rhs) * rhs

            operations = {
                ast.BinaryOperator['+']: lambda lhs, rhs: lhs + rhs,
                ast.BinaryOperator['-']: lambda lhs, rhs: lhs - rhs,
                ast.BinaryOperator['*']: lambda lhs, rhs: lhs * rhs,
                ast.BinaryOperator['/']: divide,
                ast.BinaryOperator['%']: remainder,
            }
            self.__value = (operations[expression.op](lhs_value, rhs_value)
                            if self.__evaluate_constant else 0)
            self.__value_type = result_type
            self.__value_kind = ValueKind.LITERAL
            return

        if self.__expression_kind == ExpressionKind.CONST_ARITHMETIC:
            raise NoConstantExpressionException
        if self.__expression_kind != ExpressionKind.ARITHMETIC:
            raise UnsupportedOpenQASMError('conditional expression')

        lhs, lhs_materialized_temporaries = self.__converted_operand(
            lhs_value, lhs_value_type, lhs_value_kind, result_type)
        rhs, rhs_materialized_temporaries = self.__converted_operand(
            rhs_value, rhs_value_type, rhs_value_kind, result_type)
        if is_remainder:
            result = self.__emit_integer_remainder(lhs, rhs)
        else:
            result = self.__add_new_temporary_variable(result_type)
            self.__qcx_lines.append(f'LET {result} := {lhs}')
            self.__qcx_lines.append(f'LET {result} {operators[expression.op]} {rhs}')

        temporaries_to_release = (
            lhs_materialized_temporaries | rhs_materialized_temporaries)
        for value, value_kind in (
                (lhs_value, lhs_value_kind), (rhs_value, rhs_value_kind)):
            if value_kind == ValueKind.TEMPORARY:
                temporaries_to_release.add(str(value))
        for temporary in temporaries_to_release:
            self.__release_temporary_variable(temporary)

        self.__value = result
        self.__value_type = result_type
        self.__value_kind = ValueKind.TEMPORARY

    @staticmethod
    def __integer_quotient(lhs: int, rhs: int) -> int:
        # Match truncation toward zero without converting to float or using
        # Python's floor division on signed operands. Remainder uses the same
        # quotient so that lhs == quotient * rhs + remainder.
        quotient = abs(lhs) // abs(rhs)
        return -quotient if (lhs < 0) != (rhs < 0) else quotient

    def __emit_integer_remainder(self, lhs: str, rhs: str) -> str:
        # Keep both operands live throughout the calculation. In particular,
        # neither the quotient nor the result may alias an operand temporary.
        result = self.__add_new_temporary_variable(ValueType.INT)
        quotient = self.__add_new_temporary_variable(ValueType.INT)
        self.__qcx_lines.extend([
            f'LET {result} := {lhs}', f'LET {quotient} := {lhs}',
            f'LET {quotient} /= {rhs}', f'LET {quotient} *= {rhs}',
            f'LET {result} -= {quotient}',
        ])
        self.__release_temporary_variable(quotient)
        return result

    def visit_IntegerLiteral(self, expression: ast.IntegerLiteral) -> None:
        if self.__expression_kind is None:
            return

        self.__value = expression.value
        self.__value_type = ValueType.INT
        self.__value_kind = ValueKind.LITERAL

    def visit_FloatLiteral(self, expression: ast.FloatLiteral) -> None:
        if self.__expression_kind is None:
            return

        self.__value = expression.value
        self.__value_type = ValueType.FLOAT
        self.__value_kind = ValueKind.LITERAL

    def visit_ImaginaryLiteral(self, expression: ast.ImaginaryLiteral) -> None:
        if self.__expression_kind is None:
            return

        self.__value = complex(imag=expression.value)
        self.__value_type = ValueType.COMPLEX
        self.__value_kind = ValueKind.LITERAL

    def visit_Include(self, statement: ast.Include) -> None:
        if statement.filename != 'stdgates.inc':
            raise UnsupportedOpenQASMError(f'include "{statement.filename}"')
        if not self.__is_initialization_process:
            self.__is_stdgates_included = True

    def visit_QubitDeclaration(self, statement: ast.QubitDeclaration) -> None:
        if not self.__is_initialization_process:
            self.__declared_quantum_registers.add(statement.qubit.name)
            return

        register_name = statement.qubit.name
        self.__register_source_identifier(register_name)

        if statement.size is None:
            self.__quantum_registers[register_name] = 1
            return

        self.__expression_kind = ExpressionKind.CONST_ARITHMETIC
        self.visit(statement.size)
        self.__expression_kind = None
        if self.__value_kind != ValueKind.LITERAL or self.__value_type != ValueType.INT:
            raise InvalidDeclarationException(
                f'Qubit register {register_name} must have a constant integer size')
        if self.__value <= 0:
            raise InvalidDeclarationException(
                f'Qubit register {register_name} must have a positive size')
        self.__quantum_registers[register_name] = int(self.__value)
        self.__sized_quantum_registers.add(register_name)

    def __to_qubit_index(self, quantum_register_name: str, index: int) -> int:
        quantum_register_index = self.__quantum_register_names.index(quantum_register_name)
        return self.__first_qubit_indices[quantum_register_index] + index

    def __literal_index(self, expression: ast.Expression, kind: str) -> int:
        if isinstance(expression, ast.IntegerLiteral):
            return expression.value
        if (isinstance(expression, ast.UnaryExpression)
                and expression.op == ast.UnaryOperator['-']
                and isinstance(expression.expression, ast.IntegerLiteral)):
            return -expression.expression.value
        if self.__loop_bindings:
            return self.__constant_loop_integer(expression, f'{kind} index')
        raise UnsupportedOpenQASMError(f'non-literal {kind} index')

    def __index_selection(
            self, index, size: int, kind: str, container: str,
            invalid_operand_exception: type[QASM2QCXError]
            ) -> tuple[list[int], bool]:
        if isinstance(index, ast.DiscreteSet):
            raw_indices = [
                self.__literal_index(value, kind) for value in index.values
            ]
            is_register = True
        elif isinstance(index, list):
            if len(index) != 1:
                raise UnsupportedOpenQASMError(
                    f'multidimensional {kind} indexing')

            selector = index[0]
            if isinstance(selector, ast.RangeDefinition):
                if selector.start is None or selector.end is None:
                    raise UnsupportedOpenQASMError(
                        f'{kind} range with omitted bound')
                start = self.__literal_index(selector.start, kind)
                end = self.__literal_index(selector.end, kind)
                step = (
                    1 if selector.step is None
                    else self.__literal_index(selector.step, kind))
                if step == 0:
                    raise invalid_operand_exception(
                        f'{kind.capitalize()} range step cannot be zero')
                stop = end + (1 if step > 0 else -1)
                raw_indices = list(range(start, stop, step))
                if not raw_indices:
                    raise invalid_operand_exception(
                        f'{kind.capitalize()} range is empty')
                is_register = True
            else:
                raw_indices = [self.__literal_index(selector, kind)]
                is_register = False
        else:
            raise UnsupportedOpenQASMError(
                f'multidimensional {kind} indexing')

        if not raw_indices:
            raise invalid_operand_exception(
                f'{kind.capitalize()} index set is empty')

        result = []
        for index_value in raw_indices:
            normalized_index = (
                index_value if index_value >= 0 else size + index_value)
            if normalized_index < 0 or normalized_index >= size:
                raise invalid_operand_exception(
                    f'{kind.capitalize()} index {index_value} is outside '
                    f'{container}')
            result.append(normalized_index)
        return result, is_register

    def __qubit_operand_indices(
            self, qubit: ast.Identifier | ast.IndexedIdentifier
            ) -> tuple[list[int], bool]:
        if any(self.__operand_name(qubit) in scope for scope in self.__loop_bindings):
            raise InvalidQubitOperandException('For-loop iterator is not a qubit operand')
        if isinstance(qubit, ast.Identifier):
            register_name = qubit.name
            if register_name not in self.__declared_quantum_registers:
                raise InvalidQubitOperandException(
                    f'Qubit register {register_name} is not declared')
            return (
                list(range(self.__quantum_registers[register_name])),
                register_name in self.__sized_quantum_registers,
            )

        register_name = qubit.name.name
        if register_name not in self.__declared_quantum_registers:
            raise InvalidQubitOperandException(
                f'Qubit register {register_name} is not declared')
        if len(qubit.indices) != 1:
            raise UnsupportedOpenQASMError('multidimensional qubit indexing')

        register_size = self.__quantum_registers[register_name]
        return self.__index_selection(
            qubit.indices[0], register_size, 'qubit',
            f'register {register_name}[{register_size}]',
            InvalidQubitOperandException)

    @staticmethod
    def __operand_name(operand: ast.Identifier | ast.IndexedIdentifier) -> str:
        return operand.name if isinstance(operand, ast.Identifier) else operand.name.name

    def __flattened_qubit_operand(
            self, qubit: ast.Identifier | ast.IndexedIdentifier
            ) -> tuple[list[int], bool]:
        register_name = self.__operand_name(qubit)
        indices, is_register = self.__qubit_operand_indices(qubit)
        return (
            [
                self.__to_qubit_index(register_name, index)
                for index in indices
            ],
            is_register,
        )

    def __bit_operand(
            self, bit: ast.Identifier | ast.IndexedIdentifier, *,
            role: str = 'Bit operand'
            ) -> tuple[str, list[int], bool]:
        source_name = self.__operand_name(bit)
        variable_name = self.__capitalize_variable_name(source_name)
        if self.__type_of(variable_name) != ValueType.BIT:
            raise InvalidBitOperandException(
                f'{role} {source_name} is not a bit variable')

        size = self.__bit_variable_name_size_map[variable_name]
        if isinstance(bit, ast.Identifier):
            return (
                variable_name, list(range(size)),
                variable_name in self.__sized_bit_variables,
            )
        if len(bit.indices) != 1:
            raise UnsupportedOpenQASMError('multidimensional bit indexing')

        indices, is_register = self.__index_selection(
            bit.indices[0], size, 'bit',
            f'variable {source_name}[{size}]', InvalidBitOperandException)
        return variable_name, indices, is_register

    @staticmethod
    def __qcx_bit_name(variable_name: str, size: int, index: int) -> str:
        return variable_name if size == 1 else f'{variable_name}:{index}'

    def __emit_measurement(
            self, measurement: ast.QuantumMeasurement,
            target: ast.Identifier | ast.IndexedIdentifier | None) -> None:
        qubit_indices, qubit_is_register = self.__flattened_qubit_operand(
            measurement.qubit)

        target_names: list[str] | None = None
        if target is not None:
            variable_name, bit_indices, target_is_register = self.__bit_operand(
                target, role='Measurement target')
            if qubit_is_register != target_is_register:
                raise InvalidBitOperandException(
                    'Measurement operands must both be scalars or both be registers')
            if len(qubit_indices) != len(bit_indices):
                raise MeasurementSizeMismatchException(
                    len(qubit_indices), len(bit_indices))
            size = self.__bit_variable_name_size_map[variable_name]
            target_names = [
                self.__qcx_bit_name(variable_name, size, index)
                for index in bit_indices
            ]

        for index, qubit_index in enumerate(qubit_indices):
            self.__qcx_lines.append(f'M {qubit_index}')
            if target_names is not None:
                self.__qcx_lines.append(
                    f'LET {target_names[index]} := :OUTCOME')

    @staticmethod
    def __angle_operand(
            parameter: tuple[
                str | int | float | complex, ValueType, ValueKind]) -> str:
        value, value_type, value_kind = parameter
        if value_type == ValueType.INT and value_kind != ValueKind.LITERAL:
            return f':REAL:{value}'
        return str(value)

    def __convert_parameter(
            self, parameter: tuple[str | int | float | complex, ValueType, ValueKind],
            qcx_name: str) -> tuple[str, set[str]]:
        operand = self.__angle_operand(parameter)
        temporaries = (
            {str(parameter[0])}
            if parameter[2] == ValueKind.TEMPORARY else set())

        if qcx_name not in ['EX', 'EY', 'EZ', 'CEX', 'CEY', 'CEZ']:
            return operand, temporaries

        if parameter[2] == ValueKind.LITERAL:
            return str(-0.5 * float(parameter[0])), set()

        if parameter[1] == ValueType.FLOAT:
            if parameter[2] == ValueKind.TEMPORARY:
                self.__qcx_lines.append(f'LET {parameter[0]} *= -0.5')
                return str(parameter[0]), temporaries

        temporary_variable = self.__add_new_temporary_variable(ValueType.FLOAT)
        self.__qcx_lines.append(f'LET {temporary_variable} := {operand}')
        self.__qcx_lines.append(f'LET {temporary_variable} *= -0.5')
        temporaries.add(temporary_variable)

        return temporary_variable, temporaries

    def __cu_control_phase(
            self, parameters: list[tuple[str | int | float, ValueType, ValueKind]],
            converted_parameters: list[str]) -> tuple[str, ValueKind]:
        # QCX CU3 uses the OpenQASM 2 u3 convention.  Since
        # u3(theta, phi, lambda) = exp(-i (theta + phi + lambda) / 2)
        # U(theta, phi, lambda), cu needs this additional control phase.
        theta, phi, lambda_, gamma = parameters
        if all(parameter[2] == ValueKind.LITERAL for parameter in parameters):
            phase = (
                float(gamma[0])
                + (float(theta[0]) + float(phi[0]) + float(lambda_[0])) / 2.0)
            return str(phase), ValueKind.LITERAL

        temporary = self.__add_new_temporary_variable(ValueType.FLOAT)
        self.__qcx_lines.append(
            f'LET {temporary} := {converted_parameters[0]}')
        self.__qcx_lines.append(
            f'LET {temporary} += {converted_parameters[1]}')
        self.__qcx_lines.append(
            f'LET {temporary} += {converted_parameters[2]}')
        self.__qcx_lines.append(f'LET {temporary} /= 2.0')
        self.__qcx_lines.append(
            f'LET {temporary} += {converted_parameters[3]}')
        return temporary, ValueKind.TEMPORARY

    def __u_global_phase(
            self, parameters: list[
                tuple[str | int | float, ValueType, ValueKind]],
            converted_parameters: list[str]) -> tuple[str, ValueKind]:
        # QCX U3 uses the OpenQASM 2 u3 convention.  Since
        # u3(theta, phi, lambda) = exp(-i (theta + phi + lambda) / 2)
        # U(theta, phi, lambda), U needs this additional global phase.
        if all(parameter[2] == ValueKind.LITERAL for parameter in parameters):
            phase = sum(float(parameter[0]) for parameter in parameters) / 2.0
            return str(phase), ValueKind.LITERAL

        temporary = self.__add_new_temporary_variable(ValueType.FLOAT)
        self.__qcx_lines.append(
            f'LET {temporary} := {converted_parameters[0]}')
        self.__qcx_lines.append(
            f'LET {temporary} += {converted_parameters[1]}')
        self.__qcx_lines.append(
            f'LET {temporary} += {converted_parameters[2]}')
        self.__qcx_lines.append(f'LET {temporary} /= 2.0')
        return temporary, ValueKind.TEMPORARY

    def visit_QuantumGate(self, statement: ast.QuantumGate) -> None:
        if self.__is_initialization_process:
            return

        qasm_gate_name: str = statement.name.name
        if statement.modifiers:
            raise UnsupportedOpenQASMError(f'modifiers on gate {qasm_gate_name}')
        if statement.duration is not None:
            raise UnsupportedOpenQASMError(f'duration on gate {qasm_gate_name}')

        if qasm_gate_name in QASM2QCXConverter.default_gates_qcx_map:
            qcx_gate_name = QASM2QCXConverter.default_gates_qcx_map[qasm_gate_name]
        elif self.__is_stdgates_included and qasm_gate_name in QASM2QCXConverter.stdgates_qcx_map:
            qcx_gate_name = QASM2QCXConverter.stdgates_qcx_map[qasm_gate_name]
        else:
            raise UnsupportedOpenQASMError(f'gate {qasm_gate_name}')

        expected_parameters, expected_qubits = QASM2QCXConverter.gate_signatures[qasm_gate_name]
        if (len(statement.arguments) != expected_parameters
                or len(statement.qubits) != expected_qubits):
            raise WrongGateArityException(
                qasm_gate_name, expected_parameters, expected_qubits,
                len(statement.arguments), len(statement.qubits))

        operands = [
            self.__qubit_operand_indices(qubit) for qubit in statement.qubits
        ]
        register_sizes = [
            len(indices)
            for indices, is_register in operands if is_register
        ]
        if len(set(register_sizes)) > 1:
            raise WrongBroadcastingException
        loop_size = register_sizes[0] if register_sizes else 1

        parameters: list[
            tuple[str | int | float | complex, ValueType, ValueKind]
        ] = []
        for argument in statement.arguments:
            self.__expression_kind = ExpressionKind.ARITHMETIC
            self.visit(argument)
            self.__expression_kind = None

            if self.__value_type in (ValueType.BIT, ValueType.BOOL, ValueType.COMPLEX):
                raise WrongParameterTypeException(self.__value)

            parameters.append((self.__value, self.__value_type, self.__value_kind))

        converted_parameters: list[str] = []
        temporary_variables: set[str] = set()
        for parameter in parameters:
            converted_parameter, parameter_temporaries = self.__convert_parameter(
                parameter, qcx_gate_name)
            converted_parameters.append(converted_parameter)
            temporary_variables.update(parameter_temporaries)

        cu_control_phase: str | None = None
        if qasm_gate_name == 'cu':
            cu_control_phase, value_kind = self.__cu_control_phase(
                parameters, converted_parameters)
            if value_kind == ValueKind.TEMPORARY:
                temporary_variables.add(cu_control_phase)

        u_global_phase: str | None = None
        if qasm_gate_name == 'U':
            u_global_phase, value_kind = self.__u_global_phase(
                parameters, converted_parameters)
            if value_kind == ValueKind.TEMPORARY:
                temporary_variables.add(u_global_phase)

        for index in range(loop_size):
            qubit_indices = []
            for qubit, (indices, is_register) in zip(
                    statement.qubits, operands):
                register_name = qubit.name if isinstance(qubit, ast.Identifier) else qubit.name.name
                register_index = indices[index] if is_register else indices[0]
                qubit_indices.append(str(self.__to_qubit_index(register_name, register_index)))

            if qasm_gate_name == 'cu':
                self.__qcx_lines.append(
                    f'U1 {qubit_indices[0]} {cu_control_phase}')
                self.__qcx_lines.append(
                    f'CU3 {" ".join(qubit_indices)} '
                    f'{" ".join(converted_parameters[:3])}')
            else:
                if u_global_phase is not None:
                    self.__qcx_lines.append(f'PHASE {u_global_phase}')
                qcx_line = f'{qcx_gate_name} {" ".join(qubit_indices)}'
                if converted_parameters:
                    qcx_line += f' {" ".join(converted_parameters)}'
                self.__qcx_lines.append(qcx_line)

        for temporary_variable in temporary_variables:
            self.__release_temporary_variable(temporary_variable)

    def visit_QuantumPhase(self, statement: ast.QuantumPhase) -> None:
        if self.__is_initialization_process:
            return
        if statement.modifiers:
            raise UnsupportedOpenQASMError('modifiers on gphase')
        if statement.qubits:
            raise UnsupportedOpenQASMError('qubit operands on gphase')

        self.__expression_kind = ExpressionKind.ARITHMETIC
        self.visit(statement.argument)
        self.__expression_kind = None

        if self.__value_type in (ValueType.BIT, ValueType.BOOL, ValueType.COMPLEX):
            raise WrongParameterTypeException(self.__value)

        phase = self.__angle_operand(
            (self.__value, self.__value_type, self.__value_kind))
        self.__qcx_lines.append(f'PHASE {phase}')
        if self.__value_kind == ValueKind.TEMPORARY:
            self.__release_temporary_variable(str(self.__value))

    def visit_QuantumGateDefinition(self, statement: ast.QuantumGateDefinition) -> None:
        raise UnsupportedOpenQASMError('gate definition')

    def visit_QuantumMeasurementStatement(
            self, statement: ast.QuantumMeasurementStatement) -> None:
        if self.__is_initialization_process:
            return
        self.__emit_measurement(statement.measure, statement.target)

    def visit_QuantumReset(self, statement: ast.QuantumReset) -> None:
        if self.__is_initialization_process:
            return

        qubit_indices, _ = self.__flattened_qubit_operand(statement.qubits)
        for qubit_index in qubit_indices:
            self.__qcx_lines.append(f'RESET {qubit_index}')

    def visit_QuantumBarrier(self, statement: ast.QuantumBarrier) -> None:
        if self.__is_initialization_process:
            return

        # QCX execution preserves source order, so the OpenQASM ordering
        # constraint requires no emitted instruction.  Still validate every
        # explicit operand just as gates and measurements do.
        for qubit in statement.qubits:
            self.__qubit_operand_indices(qubit)

    def visit_DelayInstruction(self, statement: ast.DelayInstruction) -> None:
        raise UnsupportedOpenQASMError('delay')

    def visit_Box(self, statement: ast.Box) -> None:
        raise UnsupportedOpenQASMError('box')

    def visit_AliasStatement(self, statement: ast.AliasStatement) -> None:
        raise UnsupportedOpenQASMError('alias')

    def __condition_operand(
            self, expression: ast.Expression
            ) -> tuple[str | int | float | complex, ValueType, ValueKind]:
        if isinstance(expression, ast.IndexExpression):
            if not isinstance(expression.collection, ast.Identifier):
                raise UnsupportedOpenQASMError(
                    'multidimensional conditional indexing')

            source_name = expression.collection.name
            variable_name = self.__capitalize_variable_name(source_name)
            if self.__type_of(variable_name) != ValueType.BIT:
                raise UnsupportedOpenQASMError(
                    'indexed non-bit conditional operand')

            size = self.__bit_variable_name_size_map[variable_name]
            previous_expression_kind = self.__expression_kind
            try:
                indices, is_register = self.__index_selection(
                    expression.index, size, 'bit',
                    f'variable {source_name}[{size}]',
                    InvalidBitOperandException)
            finally:
                self.__expression_kind = previous_expression_kind
            if is_register or len(indices) != 1:
                raise UnsupportedOpenQASMError(
                    'bit-register conditional operand')
            return (
                self.__qcx_bit_name(variable_name, size, indices[0]),
                ValueType.BIT,
                ValueKind.LVALUE,
            )

        previous_expression_kind = self.__expression_kind
        self.__expression_kind = ExpressionKind.ARITHMETIC
        try:
            self.visit(expression)
        finally:
            self.__expression_kind = previous_expression_kind
        if (self.__value is None or self.__value_type is None
                or self.__value_kind is None):
            raise UninitializedValueException

        if (isinstance(expression, ast.Identifier)
                and self.__value_type == ValueType.BIT
                and self.__bit_variable_name_size_map[str(self.__value)] > 1):
            raise UnsupportedOpenQASMError(
                'bit-register conditional operand')
        return self.__value, self.__value_type, self.__value_kind

    @staticmethod
    def __comparison_types(
            lhs_type: ValueType, rhs_type: ValueType,
            operator: ast.BinaryOperator
            ) -> tuple[ValueType, ValueType, ValueType]:
        if ValueType.COMPLEX in (lhs_type, rhs_type):
            raise UnsupportedOpenQASMError('comparison of complex values')
        if ValueType.BOOL in (lhs_type, rhs_type):
            if (lhs_type not in (ValueType.BOOL, ValueType.BIT)
                    or rhs_type not in (ValueType.BOOL, ValueType.BIT)
                    or operator not in (
                    ast.BinaryOperator['=='], ast.BinaryOperator['!='])):
                raise UnsupportedOpenQASMError(
                    'Boolean comparison requires Boolean/scalar bit operands and == or !=')
        numeric_lhs_type = (
            ValueType.INT if lhs_type in (ValueType.BIT, ValueType.BOOL) else lhs_type)
        numeric_rhs_type = (
            ValueType.INT if rhs_type in (ValueType.BIT, ValueType.BOOL) else rhs_type)
        return (numeric_lhs_type, numeric_rhs_type,
                QASM2QCXConverter.__promoted_type(numeric_lhs_type, numeric_rhs_type))

    def __emit_comparison_condition(
            self, condition: ast.Expression, true_label: str,
            false_label: str) -> None:
        if not isinstance(condition, ast.BinaryExpression):
            raise UnsupportedOpenQASMError(
                'non-comparison branching condition')

        operators = {
            ast.BinaryOperator['==']: '==',
            ast.BinaryOperator['!=']: '\\=',
            ast.BinaryOperator['>']: '>',
            ast.BinaryOperator['<']: '<',
            ast.BinaryOperator['>=']: '>=',
            ast.BinaryOperator['<=']: '<=',
        }
        if condition.op not in operators:
            raise UnsupportedOpenQASMError(
                f'branching condition operator {condition.op.name}')

        rhs_value, rhs_type, rhs_kind = self.__condition_operand(
            condition.rhs)
        lhs_value, lhs_type, lhs_kind = self.__condition_operand(
            condition.lhs)

        numeric_lhs_type, numeric_rhs_type, result_type = self.__comparison_types(
            lhs_type, rhs_type, condition.op)

        lhs, lhs_materialized_temporaries = self.__converted_operand(
            lhs_value, numeric_lhs_type, lhs_kind, result_type)
        rhs, rhs_materialized_temporaries = self.__converted_operand(
            rhs_value, numeric_rhs_type, rhs_kind, result_type)

        temporaries_to_release = (
            lhs_materialized_temporaries | rhs_materialized_temporaries)
        for value, value_kind in (
                (lhs_value, lhs_kind), (rhs_value, rhs_kind)):
            if value_kind == ValueKind.TEMPORARY:
                temporaries_to_release.add(str(value))

        # QCX JUMPIF requires its left operand to name a variable.  Native
        # symbols such as :PI are valid values, but must first be assigned to
        # a temporary when they occur on the left side of a comparison.
        lhs_is_variable = isinstance(lhs, str) and lhs[0].isalpha()
        if (not lhs_is_variable or lhs_kind == ValueKind.LITERAL
                or numeric_lhs_type != result_type):
            comparison_lhs = self.__add_new_temporary_variable(result_type)
            self.__qcx_lines.append(f'LET {comparison_lhs} := {lhs}')
            temporaries_to_release.add(comparison_lhs)
        else:
            comparison_lhs = lhs

        self.__qcx_lines.append(
            f'JUMPIF {true_label} {comparison_lhs} '
            f'{operators[condition.op]} {rhs}')
        self.__qcx_lines.append(f'JUMP {false_label}')
        for temporary in temporaries_to_release:
            self.__release_temporary_variable(temporary)

    def __emit_condition(
            self, condition: ast.Expression, true_label: str,
            false_label: str) -> None:
        if isinstance(condition, ast.Cast) and isinstance(condition.type, ast.BoolType):
            value, value_type, value_kind = self.__condition_operand(condition.argument)
            if value_type not in (ValueType.BOOL, ValueType.BIT, ValueType.INT, ValueType.FLOAT):
                raise UnsupportedOpenQASMError('cast to bool from complex or unsupported type')
            if value_kind == ValueKind.LITERAL:
                self.__qcx_lines.append(f'JUMP {true_label if value != 0 else false_label}')
                return
            temporaries_to_release = set()
            if value_kind == ValueKind.TEMPORARY:
                temporaries_to_release.add(str(value))
            if not isinstance(value, str) or not value[0].isalpha():
                temporary = self.__add_new_temporary_variable(value_type)
                self.__qcx_lines.append(f'LET {temporary} := {value}')
                temporaries_to_release.add(temporary)
                value = temporary
            self.__qcx_lines.append(f'JUMPIF {true_label} {value} \\= 0')
            self.__qcx_lines.append(f'JUMP {false_label}')
            for temporary in temporaries_to_release:
                self.__release_temporary_variable(temporary)
            return
        if isinstance(condition, ast.UnaryExpression):
            if condition.op != ast.UnaryOperator['!']:
                raise UnsupportedOpenQASMError(
                    f'branching condition unary operator {condition.op.name}')
            self.__emit_condition(
                condition.expression, false_label, true_label)
            return

        if isinstance(condition, ast.BinaryExpression):
            if condition.op in (
                    ast.BinaryOperator['&&'], ast.BinaryOperator['||']):
                rhs_label = f'QASM2QCX_CONDITION_{self.__condition_index}'
                self.__condition_index += 1
                if condition.op == ast.BinaryOperator['&&']:
                    self.__emit_condition(
                        condition.lhs, rhs_label, false_label)
                else:
                    self.__emit_condition(
                        condition.lhs, true_label, rhs_label)
                self.__qcx_lines.append(f'@{rhs_label}')
                self.__emit_condition(condition.rhs, true_label, false_label)
                return
            self.__emit_comparison_condition(
                condition, true_label, false_label)
            return

        if not isinstance(condition, (
                ast.Identifier, ast.IndexExpression, ast.BooleanLiteral, ast.Cast)):
            raise UnsupportedOpenQASMError(
                'non-comparison branching condition')

        value, value_type, value_kind = self.__condition_operand(condition)
        if value_type not in (ValueType.BIT, ValueType.BOOL):
            raise UnsupportedOpenQASMError(
                'direct branching condition must be a scalar bit or Boolean')
        if (isinstance(condition, ast.Identifier)
                and value in self.__sized_bit_variables):
            raise UnsupportedOpenQASMError(
                'bit-register conditional operand')

        if value_kind == ValueKind.LITERAL:
            self.__qcx_lines.append(f'JUMP {true_label if value else false_label}')
            return
        self.__qcx_lines.append(f'JUMPIF {true_label} {value} \\= 0')
        self.__qcx_lines.append(f'JUMP {false_label}')
        if value_kind == ValueKind.TEMPORARY:
            self.__release_temporary_variable(str(value))

    def __visit_boolean_expression(
            self, expression: ast.Expression) -> None:
        if self.__expression_kind == ExpressionKind.CONST_ARITHMETIC:
            self.__evaluate_boolean_constant(expression)
            return

        previous_temporaries = self.__declared_temporary_variables.copy()
        index = self.__boolean_expression_index
        self.__boolean_expression_index += 1
        true_label = f'QASM2QCX_BOOL_TRUE_{index}'
        false_label = f'QASM2QCX_BOOL_FALSE_{index}'
        end_label = f'QASM2QCX_BOOL_END_{index}'
        # Reserve storage before evaluating the condition. Nested expression
        # temporaries must not alias the result, nor overwrite an assignment's
        # destination before the complete RHS has been evaluated.
        result = self.__add_new_temporary_variable(ValueType.BOOL)
        self.__emit_condition(expression, true_label, false_label)
        self.__qcx_lines.extend([
            f'@{true_label}', f'LET {result} := 1', f'JUMP {end_label}',
            f'@{false_label}', f'LET {result} := 0', f'@{end_label}',
        ])
        self.__hoist_temporary_declarations(previous_temporaries)
        self.__value = result
        self.__value_type = ValueType.BOOL
        self.__value_kind = ValueKind.TEMPORARY

    def __evaluate_boolean_constant(
            self, expression: ast.UnaryExpression | ast.BinaryExpression) -> None:
        if isinstance(expression, ast.UnaryExpression):
            self.visit(expression.expression)
            if self.__value_type not in (ValueType.BOOL, ValueType.BIT):
                raise NoImplicitCastException
            self.__value = int(not self.__value) if self.__evaluate_constant else 0
        else:
            self.visit(expression.lhs)
            lhs_value, lhs_type = self.__value, self.__value_type
            logical = expression.op in (
                ast.BinaryOperator['&&'], ast.BinaryOperator['||'])
            evaluate = self.__evaluate_constant
            skip_rhs = logical and (
                (expression.op == ast.BinaryOperator['&&'] and not lhs_value)
                or (expression.op == ast.BinaryOperator['||'] and lhs_value))
            # Still visit a skipped operand to validate its names and types,
            # but suppress arithmetic evaluation (e.g. division by zero).
            self.__evaluate_constant = evaluate and not skip_rhs
            try:
                self.visit(expression.rhs)
            finally:
                self.__evaluate_constant = evaluate
            rhs_value, rhs_type = self.__value, self.__value_type
            if logical:
                if (lhs_type not in (ValueType.BOOL, ValueType.BIT)
                        or rhs_type not in (ValueType.BOOL, ValueType.BIT)):
                    raise NoImplicitCastException
                value = (lhs_value and rhs_value if expression.op == ast.BinaryOperator['&&']
                         else lhs_value or rhs_value)
            else:
                _, _, result_type = self.__comparison_types(
                    lhs_type, rhs_type, expression.op)
                if result_type == ValueType.FLOAT:
                    lhs_value, rhs_value = float(lhs_value), float(rhs_value)
                comparisons = {
                    ast.BinaryOperator['==']: lambda lhs, rhs: lhs == rhs,
                    ast.BinaryOperator['!=']: lambda lhs, rhs: lhs != rhs,
                    ast.BinaryOperator['<']: lambda lhs, rhs: lhs < rhs,
                    ast.BinaryOperator['<=']: lambda lhs, rhs: lhs <= rhs,
                    ast.BinaryOperator['>']: lambda lhs, rhs: lhs > rhs,
                    ast.BinaryOperator['>=']: lambda lhs, rhs: lhs >= rhs,
                }
                value = comparisons[expression.op](lhs_value, rhs_value) if evaluate else False
            self.__value = int(bool(value)) if evaluate else 0
        self.__value_type = ValueType.BOOL
        self.__value_kind = ValueKind.LITERAL

    def __hoist_temporary_declarations(self, previous_temporaries: set[str]) -> None:
        # Storage must exist even when its first expression is skipped. Only
        # declarations move; computations stay in their original branches.
        new_temporaries = self.__declared_temporary_variables - previous_temporaries
        declarations = [
            line for line in self.__qcx_lines
            if line.startswith('VAR ') and line.split()[1] in new_temporaries
        ]
        declaration_set = set(declarations)
        self.__qcx_lines = [line for line in self.__qcx_lines
                            if line not in declaration_set]
        self.__qcx_lines[1:1] = declarations

    def visit_BranchingStatement(self, statement: ast.BranchingStatement) -> None:
        if self.__is_initialization_process:
            for block in (statement.if_block, statement.else_block):
                for child_statement in block:
                    if isinstance(
                            child_statement,
                            (ast.ClassicalDeclaration,
                             ast.ConstantDeclaration,
                             ast.QubitDeclaration)):
                        raise UnsupportedOpenQASMError(
                            'block-local declaration')
                    self.visit(child_statement)
            return

        branch_index = self.__branch_index
        previous_temporaries = self.__declared_temporary_variables.copy()
        self.__branch_depth += 1
        self.__branch_index += 1
        if_label = f'QASM2QCX_IF_{branch_index}'
        else_label = f'QASM2QCX_ELSE_{branch_index}'
        end_label = f'QASM2QCX_END_IF_{branch_index}'

        false_label = else_label if statement.else_block else end_label
        self.__emit_condition(
            statement.condition, if_label, false_label)
        self.__qcx_lines.append(f'@{if_label}')
        for child_statement in statement.if_block:
            self.visit(child_statement)
        if statement.else_block:
            self.__qcx_lines.append(f'JUMP {end_label}')
            self.__qcx_lines.append(f'@{else_label}')
            for child_statement in statement.else_block:
                self.visit(child_statement)
        self.__qcx_lines.append(f'@{end_label}')
        self.__branch_depth -= 1
        if self.__branch_depth == 0:
            self.__hoist_temporary_declarations(previous_temporaries)

    def __constant_loop_integer(self, expression: ast.Expression, part: str) -> int:
        # Range evaluation must not inherit skipped-expression state or disturb
        # the expression being converted by the enclosing visitor.
        previous = (self.__expression_kind, self.__value, self.__value_type,
                    self.__value_kind, self.__evaluate_constant)
        try:
            self.__expression_kind = ExpressionKind.CONST_ARITHMETIC
            self.__value = self.__value_type = self.__value_kind = None
            self.__evaluate_constant = True
            self.visit(expression)
            if self.__value_type != ValueType.INT or self.__value_kind != ValueKind.LITERAL:
                raise InvalidLoopRangeException(
                    f'For-loop range {part} must be a constant integer')
            return int(self.__value)
        finally:
            (self.__expression_kind, self.__value, self.__value_type,
             self.__value_kind, self.__evaluate_constant) = previous

    def __loop_range(self, statement: ast.ForInLoop) -> range:
        self.__validate_loop_header(statement)
        bounds = statement.set_declaration
        start = self.__constant_loop_integer(bounds.start, 'start')
        end = self.__constant_loop_integer(bounds.end, 'end')
        step = (1 if bounds.step is None
                else self.__constant_loop_integer(bounds.step, 'step'))
        if step == 0:
            raise InvalidLoopRangeException('For-loop range step cannot be zero')
        # OpenQASM includes the end when reachable; Python excludes the stop.
        # Keep this lazy so a huge range cannot allocate a huge intermediate list.
        return range(start, end + (1 if step > 0 else -1), step)

    @staticmethod
    def __validate_loop_header(statement: ast.ForInLoop) -> None:
        if not isinstance(statement.type, ast.IntType):
            raise UnsupportedOpenQASMError('for-loop iteration type other than int')
        bounds = statement.set_declaration
        if not isinstance(bounds, ast.RangeDefinition):
            raise UnsupportedOpenQASMError('for-loop iteration other than a constant range')
        if bounds.start is None or bounds.end is None:
            raise InvalidLoopRangeException('For-loop range requires both bounds')

    def __validate_loop_syntax(self, node: ast.QASMNode) -> None:
        # Structural checks never evaluate an expression. In particular, an
        # empty loop must not silently accept an unsupported operator/function,
        # nor execute arithmetic or inspect an out-of-range iteration value.
        if isinstance(node, ast.Expression) and not isinstance(node, (
                ast.Identifier, ast.IntegerLiteral, ast.FloatLiteral,
                ast.ImaginaryLiteral, ast.BooleanLiteral, ast.BitstringLiteral,
                ast.UnaryExpression, ast.BinaryExpression, ast.Cast, ast.IndexExpression)):
            raise UnsupportedOpenQASMError(f'for-loop expression {type(node).__name__}')
        if isinstance(node, ast.BinaryExpression) and node.op.name not in (
                '+', '-', '*', '/', '%', '==', '!=', '<', '<=', '>', '>=', '&&', '||'):
            raise UnsupportedOpenQASMError(f'binary operator {node.op.name}')
        if isinstance(node, ast.UnaryExpression) and node.op.name not in ('-', '!'):
            raise UnsupportedOpenQASMError(f'unary operator {node.op.name}')
        if isinstance(node, ast.Cast) and not isinstance(node.type, (
                ast.IntType, ast.UintType, ast.FloatType, ast.ComplexType, ast.BoolType, ast.BitType)):
            raise UnsupportedOpenQASMError(f'for-loop cast {type(node.type).__name__}')
        for field in dataclasses.fields(node):
            value = getattr(node, field.name)
            if isinstance(value, ast.QASMNode):
                self.__validate_loop_syntax(value)
            elif isinstance(value, list):
                self.__validate_loop_syntax_list(value)

    def __validate_loop_syntax_list(self, values: list) -> None:
        for value in values:
            if isinstance(value, ast.QASMNode):
                self.__validate_loop_syntax(value)
            elif isinstance(value, list):
                self.__validate_loop_syntax_list(value)

    def __loop_bound_names(self, node: ast.QASMNode) -> set[str]:
        if isinstance(node, ast.Identifier):
            return {node.name}
        names: set[str] = set()
        for field in dataclasses.fields(node):
            value = getattr(node, field.name)
            if isinstance(value, ast.QASMNode):
                names.update(self.__loop_bound_names(value))
            elif isinstance(value, list):
                for element in value:
                    if isinstance(element, ast.QASMNode):
                        names.update(self.__loop_bound_names(element))
        return names

    def __validate_nested_loop_bounds(self, statement: ast.ForInLoop, iterators: set[str]) -> None:
        # Outer iterator values may not exist for an empty outer loop. Validate
        # names and independent bounds now; defer value-dependent checks until
        # an actual iteration, rather than fabricating an outer value.
        bounds = statement.set_declaration
        constants = (set(self.__const_int_variable_name_values_map)
                     | set(self.__const_float_variable_name_values_map)
                     | set(self.__const_complex_variable_name_values_map)
                     | set(self.__const_bool_variable_values_map))
        for expression, part in ((bounds.start, 'start'), (bounds.end, 'end'),
                                 (bounds.step, 'step')):
            if expression is None:
                continue
            names = self.__loop_bound_names(expression)
            for name in names - iterators - constants - {'pi', 'tau', 'euler'}:
                raise NoVariableNameException(name)
            if not names & iterators:
                value = self.__constant_loop_integer(expression, part)
                if part == 'step' and value == 0:
                    raise InvalidLoopRangeException('For-loop range step cannot be zero')

    def __validate_loop_body(self, statements: list[ast.Statement], iterators: set[str]) -> None:
        # Validate even an empty iteration range, without executing its body.
        supported = (ast.QuantumGate, ast.QuantumPhase,
                     ast.QuantumMeasurementStatement, ast.QuantumReset,
                     ast.QuantumBarrier, ast.ClassicalAssignment,
                     ast.BranchingStatement, ast.ForInLoop)
        for child in statements:
            if isinstance(child, (ast.ClassicalDeclaration, ast.ConstantDeclaration,
                                  ast.QubitDeclaration)):
                raise UnsupportedOpenQASMError('block-local declaration')
            if not isinstance(child, supported):
                raise UnsupportedOpenQASMError(f'for-loop body {type(child).__name__}')
            target = (child.lvalue if isinstance(child, ast.ClassicalAssignment)
                      else child.target if isinstance(child, ast.QuantumMeasurementStatement)
                      else None)
            if target is not None and self.__operand_name(target) in iterators:
                raise UnsupportedOpenQASMError('assignment to a for-loop iterator')
            if isinstance(child, ast.ClassicalAssignment) and child.op.name not in (
                    '=', '+=', '-=', '*=', '/=', '%='):
                raise UnsupportedOpenQASMError(f'assignment operator {child.op.name}')
            if isinstance(child, ast.QuantumGate):
                if child.modifiers or child.duration is not None:
                    raise UnsupportedOpenQASMError('gate modifiers or duration in a for loop')
                if child.name.name not in self.gate_signatures:
                    raise UnsupportedOpenQASMError(f'gate {child.name.name}')
                if (not self.__is_initialization_process
                        and child.name.name not in self.default_gates_qcx_map
                        and not self.__is_stdgates_included):
                    raise UnsupportedOpenQASMError(f'gate {child.name.name}')
                parameters, qubits = self.gate_signatures[child.name.name]
                if len(child.arguments) != parameters or len(child.qubits) != qubits:
                    raise WrongGateArityException(child.name.name, parameters, qubits,
                                                  len(child.arguments), len(child.qubits))
            if isinstance(child, ast.QuantumPhase) and (child.modifiers or child.qubits):
                raise UnsupportedOpenQASMError('modifiers or qubit operands on gphase')
            if isinstance(child, ast.BranchingStatement):
                self.__validate_loop_body(child.if_block, iterators)
                self.__validate_loop_body(child.else_block, iterators)
            if isinstance(child, ast.ForInLoop):
                self.__validate_loop_header(child)
                self.__validate_nested_loop_bounds(child, iterators)
                self.__validate_loop_body(child.block, iterators | {child.identifier.name})

    def visit_ForInLoop(self, statement: ast.ForInLoop) -> None:
        values = self.__loop_range(statement)
        self.__validate_loop_body(statement.block,
                                  {name for scope in self.__loop_bindings for name in scope}
                                  | {statement.identifier.name})
        self.__validate_loop_syntax(statement)
        distance = (values.stop - values.start) * (1 if values.step > 0 else -1)
        stride = abs(values.step)
        count = max(0, (distance + stride - 1) // stride)
        if count > self.MAX_LOOP_ITERATIONS - self.__loop_iterations:
            raise InvalidLoopRangeException(
                f'For-loop expansion exceeds {self.MAX_LOOP_ITERATIONS} iterations')
        self.__loop_iterations += count
        # Allocate one exit target per expanded loop instance and a distinct
        # continuation target per iteration. Ordinary loops do not emit these
        # unused labels; break/continue lowering will use the innermost frame.
        loop_index = self.__loop_index
        self.__loop_index += 1
        break_label = f'QASM2QCX_LOOP_{loop_index}_END'
        for iteration, value in enumerate(values):
            self.__loop_bindings.append({statement.identifier.name: value})
            self.__loop_contexts.append(_LoopContext(
                break_label, f'QASM2QCX_LOOP_{loop_index}_NEXT_{iteration}'))
            try:
                for child in statement.block:
                    self.visit(child)
            finally:
                self.__loop_contexts.pop()
                self.__loop_bindings.pop()

    def visit_WhileLoop(self, statement: ast.WhileLoop) -> None:
        raise UnsupportedOpenQASMError('while loop')

    def visit_SwitchStatement(self, statement: ast.SwitchStatement) -> None:
        raise UnsupportedOpenQASMError('switch statement')

    def visit_ExpressionStatement(self, statement: ast.ExpressionStatement) -> None:
        raise UnsupportedOpenQASMError('expression statement')

    def visit_CalibrationGrammarDeclaration(
            self, statement: ast.CalibrationGrammarDeclaration) -> None:
        raise UnsupportedOpenQASMError('calibration grammar declaration')

    def visit_CalibrationDefinition(self, statement: ast.CalibrationDefinition) -> None:
        raise UnsupportedOpenQASMError('calibration definition')

    def visit_CalibrationStatement(self, statement: ast.CalibrationStatement) -> None:
        raise UnsupportedOpenQASMError('calibration statement')

    def visit_ExternDeclaration(self, statement: ast.ExternDeclaration) -> None:
        raise UnsupportedOpenQASMError('extern declaration')

    def visit_IODeclaration(self, statement: ast.IODeclaration) -> None:
        raise UnsupportedOpenQASMError('input/output declaration')

    def visit_Pragma(self, statement: ast.Pragma) -> None:
        if not self.__is_initialization_process:
            return

        arguments = statement.command.split()
        if not arguments or arguments[0] != 'riken_braket.amplitudes':
            raise UnsupportedOpenQASMError(f'pragma {statement.command}')
        if self.__amplitude_indices is not None:
            raise InvalidPragmaException(
                'The riken_braket.amplitudes pragma may appear only once')

        indices: list[int] = []
        for argument in arguments[1:]:
            if not argument.isascii() or not argument.isdecimal():
                raise InvalidPragmaException(
                    f'Invalid amplitude index: {argument}')
            index = int(argument)
            if index in indices:
                raise InvalidPragmaException(
                    f'Duplicate amplitude index: {index}')
            indices.append(index)
        self.__amplitude_indices = indices

    def visit_SubroutineDefinition(self, statement: ast.SubroutineDefinition) -> None:
        raise UnsupportedOpenQASMError('subroutine definition')

    def visit_ReturnStatement(self, statement: ast.ReturnStatement) -> None:
        raise UnsupportedOpenQASMError('return statement')

    def visit_BreakStatement(self, statement: ast.BreakStatement) -> None:
        raise UnsupportedOpenQASMError('break statement')

    def visit_ContinueStatement(self, statement: ast.ContinueStatement) -> None:
        raise UnsupportedOpenQASMError('continue statement')

    def visit_EndStatement(self, statement: ast.EndStatement) -> None:
        raise UnsupportedOpenQASMError('end statement')

    def visit_ArrayLiteral(self, expression: ast.ArrayLiteral) -> None:
        raise UnsupportedOpenQASMError('array literal')

    def visit_BitstringLiteral(self, expression: ast.BitstringLiteral) -> None:
        raise UnsupportedOpenQASMError('bit-string literal')

    def visit_BooleanLiteral(self, expression: ast.BooleanLiteral) -> None:
        if self.__expression_kind is None:
            return
        self.__value = int(expression.value)
        self.__value_type = ValueType.BOOL
        self.__value_kind = ValueKind.LITERAL

    def visit_Cast(self, expression: ast.Cast) -> None:
        if self.__expression_kind is None:
            return

        if isinstance(expression.type, ast.BoolType):
            if self.__expression_kind == ExpressionKind.CONST_ARITHMETIC:
                self.visit(expression.argument)
                if self.__value_type not in (ValueType.BOOL, ValueType.BIT, ValueType.INT, ValueType.FLOAT):
                    raise UnsupportedOpenQASMError('cast to bool from complex or unsupported type')
                self.__value = int(self.__value != 0) if self.__evaluate_constant else 0
                self.__value_type = ValueType.BOOL
                self.__value_kind = ValueKind.LITERAL
            else:
                self.__visit_boolean_expression(expression)
            return
        if isinstance(expression.type, ast.BitType):
            if expression.type.size is not None:
                raise UnsupportedOpenQASMError('cast to bit register')
            self.visit(expression.argument)
            if self.__value_type not in (ValueType.BOOL, ValueType.BIT):
                raise UnsupportedOpenQASMError('scalar bit cast requires a Boolean or scalar bit')
            self.__value_type = ValueType.BIT
            return
        if isinstance(expression.type, (ast.IntType, ast.UintType)):
            target_type = ValueType.INT
            cast_name = 'INT'
        elif isinstance(expression.type, ast.FloatType):
            target_type = ValueType.FLOAT
            cast_name = 'REAL'
        elif isinstance(expression.type, ast.ComplexType):
            target_type = ValueType.COMPLEX
            cast_name = 'COMPLEX'
        else:
            raise UnsupportedOpenQASMError(
                f'cast to {type(expression.type).__name__}')

        self.visit(expression.argument)
        if self.__value is None or self.__value_type is None or self.__value_kind is None:
            raise UninitializedValueException

        if self.__value_type == ValueType.BOOL:
            self.__value_type = ValueType.INT
        if self.__value_kind == ValueKind.LITERAL:
            if target_type == ValueType.INT:
                self.__value = int(complex(self.__value).real)
            elif target_type == ValueType.FLOAT:
                self.__value = float(complex(self.__value).real)
            else:
                self.__value = complex(self.__value)
            self.__value_type = target_type
            return

        if self.__value_type == target_type:
            return

        original_value = self.__value
        original_kind = self.__value_kind
        temporary = self.__add_new_temporary_variable(target_type)
        self.__qcx_lines.append(
            f'LET {temporary} := :{cast_name}:{original_value}')
        if original_kind == ValueKind.TEMPORARY:
            self.__release_temporary_variable(str(original_value))
        self.__value = temporary
        self.__value_type = target_type
        self.__value_kind = ValueKind.TEMPORARY

    def visit_Concatenation(self, expression: ast.Concatenation) -> None:
        raise UnsupportedOpenQASMError('concatenation')

    def visit_DurationLiteral(self, expression: ast.DurationLiteral) -> None:
        raise UnsupportedOpenQASMError('duration literal')

    def visit_DurationOf(self, expression: ast.DurationOf) -> None:
        raise UnsupportedOpenQASMError('durationof expression')

    def visit_FunctionCall(self, expression: ast.FunctionCall) -> None:
        raise UnsupportedOpenQASMError('function call')

    def visit_IndexExpression(self, expression: ast.IndexExpression) -> None:
        if self.__expression_kind is None:
            return
        if self.__expression_kind == ExpressionKind.CONST_ARITHMETIC:
            raise NoConstantExpressionException
        self.__value, self.__value_type, self.__value_kind = self.__condition_operand(expression)

    def visit_SizeOf(self, expression: ast.SizeOf) -> None:
        raise UnsupportedOpenQASMError('sizeof expression')

    def __make_constant_variable(self, variable_name: str, variable_type, num_elements: int = 1) -> None:
        if num_elements <= 0:
            raise InvalidDeclarationException(
                f'Constant {variable_name} must have a positive size')

        match variable_type:
            case ast.BoolType():
                # Publish the constant only after evaluating its initializer,
                # so a self-reference cannot read a fabricated default value.
                pass

            case ast.IntType() | ast.UintType():
                if variable_name in self.__const_int_variable_name_values_map:
                    raise WrongConstantVariableException(variable_name)

                self.__const_int_variable_name_values_map[variable_name] = [0] * num_elements

            case ast.FloatType():
                if variable_name in self.__const_float_variable_name_values_map:
                    raise WrongConstantVariableException(variable_name)

                self.__const_float_variable_name_values_map[variable_name] = [0.0] * num_elements

            #case ast.AngleType():
            #    pass

            #case ast.BitType():
            #    pass

            case ast.ComplexType():
                if variable_name in self.__const_complex_variable_name_values_map:
                    raise WrongConstantVariableException(variable_name)

                self.__const_complex_variable_name_values_map[variable_name] = [complex()] * num_elements

    def __set_constant_variable(self, variable_name: str, variable_type) -> None:
        if isinstance(variable_type, ast.BoolType):
            if (self.__value_type not in (ValueType.BOOL, ValueType.BIT)
                    and not (self.__value_type == ValueType.INT and self.__value in (0, 1))):
                raise NoImplicitCastException
            self.__const_bool_variable_values_map[variable_name] = int(self.__value)
            return
        if self.__value_type == ValueType.BOOL:
            self.__value_type = ValueType.INT
        match variable_type:
            case ast.IntType() | ast.UintType():
                if variable_name not in self.__const_int_variable_name_values_map:
                    raise WrongConstantVariableException(variable_name)

                if self.__value_type == ValueType.FLOAT or self.__value_type == ValueType.COMPLEX:
                    raise NoImplicitCastException

                self.__const_int_variable_name_values_map[variable_name][0] = int(self.__value)

            case ast.FloatType():
                if variable_name not in self.__const_float_variable_name_values_map:
                    raise WrongConstantVariableException(variable_name)

                if self.__value_type == ValueType.COMPLEX:
                    raise NoImplicitCastException

                self.__const_float_variable_name_values_map[variable_name][0] = float(self.__value)

            #case ast.AngleType():
            #    pass

            #case ast.BitType():
            #    pass

            case ast.ComplexType():
                if variable_name not in self.__const_complex_variable_name_values_map:
                    raise WrongConstantVariableException(variable_name)

                self.__const_complex_variable_name_values_map[variable_name][0] = complex(self.__value)

    def visit_ConstantDeclaration(self, statement: ast.ConstantDeclaration) -> None:
        if not self.__is_initialization_process:
            self.__declared_constant_variables.add(statement.identifier.name)
            return

        variable_type = statement.type
        variable_name = statement.identifier.name
        self.__register_source_identifier(variable_name)

        if isinstance(variable_type, ast.ArrayType):
            raise UnsupportedOpenQASMError('constant array')
        if not isinstance(variable_type, (
                ast.IntType, ast.UintType, ast.FloatType, ast.ComplexType,
                ast.BoolType)):
            raise UnsupportedOpenQASMError(
                f'constant type {type(variable_type).__name__}')
        self.__make_constant_variable(variable_name, variable_type)

        if statement.init_expression is None:
            raise NoConstantExpressionException

        match statement.init_expression:
            case ast.Expression():
                self.__expression_kind = ExpressionKind.CONST_ARITHMETIC
                self.visit(statement.init_expression)
                self.__expression_kind = None

                self.__set_constant_variable(variable_name, variable_type)

            case _:
                raise NoConstantExpressionException

    def __declare_classical_variable(self, variable_type, variable_name, num_elements = 1) -> None:
        if num_elements <= 0:
            raise InvalidDeclarationException(
                f'Classical variable {variable_name} must have a positive size')

        match variable_type:
            case ast.BoolType():
                self.__bool_variable_names.add(variable_name)
                self.__qcx_lines.append(f'VAR {variable_name} INT')

            case ast.IntType() | ast.UintType():
                if variable_name in self.__int_variable_name_size_map:
                    raise WrongClassicalDeclarationException(variable_name)

                self.__int_variable_name_size_map[variable_name] = num_elements
                self.__qcx_lines.append(f'VAR {variable_name} INT' + (f' {num_elements}' if num_elements > 1 else ''))

            case ast.FloatType():
                if variable_name in self.__float_variable_name_size_map:
                    raise WrongClassicalDeclarationException(variable_name)

                self.__float_variable_name_size_map[variable_name] = num_elements
                self.__qcx_lines.append(f'VAR {variable_name} REAL' + (f' {num_elements}' if num_elements > 1 else ''))

            case ast.BitType():
                if variable_name in self.__bit_variable_name_size_map:
                    raise WrongClassicalDeclarationException(variable_name)

                self.__bit_variable_name_size_map[variable_name] = num_elements
                self.__qcx_lines.append(
                    f'VAR {variable_name} INT'
                    + (f' {num_elements}' if num_elements > 1 else ''))

            case ast.ComplexType():
                if variable_name in self.__complex_variable_name_size_map:
                    raise WrongClassicalDeclarationException(variable_name)

                self.__complex_variable_name_size_map[variable_name] = num_elements
                self.__qcx_lines.append(f'VAR {variable_name} COMPLEX' + (f' {num_elements}' if num_elements > 1 else ''))

    def __bit_type_size(self, variable_type: ast.BitType, variable_name: str) -> int:
        if variable_type.size is None:
            return 1

        self.__expression_kind = ExpressionKind.CONST_ARITHMETIC
        self.visit(variable_type.size)
        self.__expression_kind = None
        if (self.__value_kind != ValueKind.LITERAL
                or self.__value_type != ValueType.INT):
            raise InvalidDeclarationException(
                f'Bit variable {variable_name} must have a constant integer size')
        if self.__value <= 0:
            raise InvalidDeclarationException(
                f'Bit variable {variable_name} must have a positive size')
        return int(self.__value)

    def __emit_bit_expression_assignment(
            self, target: ast.Identifier | ast.IndexedIdentifier,
            expression: ast.Expression) -> None:
        variable_name, target_indices, target_is_register = self.__bit_operand(
            target)
        target_size = self.__bit_variable_name_size_map[variable_name]
        target_names = [
            self.__qcx_bit_name(variable_name, target_size, index)
            for index in target_indices
        ]

        boolean_identifier = isinstance(expression, ast.Identifier) and (
            self.__capitalize_variable_name(expression.name) in self.__bool_variable_names
            or expression.name in self.__const_bool_variable_values_map)
        if not target_is_register and (boolean_identifier or not isinstance(
                expression, (ast.Identifier, ast.IndexExpression,
                             ast.IntegerLiteral, ast.BitstringLiteral))):
            previous_expression_kind = self.__expression_kind
            self.__expression_kind = ExpressionKind.ARITHMETIC
            try:
                self.visit(expression)
            finally:
                self.__expression_kind = previous_expression_kind
            if self.__value_type not in (ValueType.BOOL, ValueType.BIT):
                raise InvalidBitOperandException('Scalar bit assignment requires a bit or Boolean value')
            self.__emit_assignment(target_names[0], ':=', ValueType.BIT,
                                   self.__value, self.__value_type, self.__value_kind)
            return
        if isinstance(expression, ast.BitstringLiteral):
            if not target_is_register:
                raise InvalidBitOperandException(
                    'A bit-string value requires a bit-register target')
            if expression.width != len(target_names):
                raise InvalidBitOperandException(
                    f'Bit-string width {expression.width} does not match target '
                    f'size {len(target_names)}')
            values = [
                (expression.value >> index) & 1
                for index in range(expression.width)
            ]
        elif isinstance(expression, ast.IntegerLiteral):
            if target_is_register or expression.value not in (0, 1):
                raise InvalidBitOperandException(
                    'An integer bit initializer must be 0 or 1 and target a '
                    'scalar bit')
            values = [expression.value]
        elif isinstance(expression, ast.Identifier):
            source_name, source_indices, source_is_register = self.__bit_operand(
                expression, role='Bit source')
            if source_is_register != target_is_register:
                raise InvalidBitOperandException(
                    'Bit assignment operands must both be scalars or both be registers')
            if len(source_indices) != len(target_names):
                raise InvalidBitOperandException(
                    f'Bit source has size {len(source_indices)}, but target has '
                    f'size {len(target_names)}')
            source_size = self.__bit_variable_name_size_map[source_name]
            values = [
                self.__qcx_bit_name(source_name, source_size, index)
                for index in source_indices
            ]
        elif isinstance(expression, ast.IndexExpression):
            if not isinstance(expression.collection, ast.Identifier):
                raise UnsupportedOpenQASMError('multidimensional bit indexing')

            source_identifier = expression.collection.name
            source_name = self.__capitalize_variable_name(source_identifier)
            if self.__type_of(source_name) != ValueType.BIT:
                raise InvalidBitOperandException(
                    f'Bit source {source_identifier} is not a bit variable')
            source_size = self.__bit_variable_name_size_map[source_name]
            source_indices, source_is_register = self.__index_selection(
                expression.index, source_size, 'bit',
                f'variable {source_identifier}[{source_size}]',
                InvalidBitOperandException)
            if source_is_register != target_is_register:
                raise InvalidBitOperandException(
                    'Bit assignment operands must both be scalars or both be registers')
            if len(source_indices) != len(target_names):
                raise InvalidBitOperandException(
                    f'Bit source has size {len(source_indices)}, but target has '
                    f'size {len(target_names)}')
            values = [
                self.__qcx_bit_name(source_name, source_size, source_index)
                for source_index in source_indices
            ]
        else:
            raise UnsupportedOpenQASMError(
                f'bit assignment from {type(expression).__name__}')

        for target_name, value in zip(target_names, values):
            self.__qcx_lines.append(f'LET {target_name} := {value}')

    def visit_ClassicalDeclaration(self, statement: ast.ClassicalDeclaration) -> None:
        if self.__is_initialization_process:
            self.__register_source_identifier(statement.identifier.name)
            self.__reserved_variable_names.add(
                self.__capitalize_variable_name(statement.identifier.name))
            return

        variable_type = statement.type
        variable_name = self.__capitalize_variable_name(statement.identifier.name)

        if isinstance(variable_type, ast.ArrayType):
            raise UnsupportedOpenQASMError('classical array')
        if not isinstance(
                variable_type,
                (ast.IntType, ast.UintType, ast.FloatType, ast.BitType,
                 ast.ComplexType, ast.BoolType)):
            raise UnsupportedOpenQASMError(
                f'classical type {type(variable_type).__name__}')
        num_elements = (
            self.__bit_type_size(variable_type, statement.identifier.name)
            if isinstance(variable_type, ast.BitType) else 1)
        self.__declare_classical_variable(
            variable_type, variable_name, num_elements)
        if (isinstance(variable_type, ast.BitType)
                and variable_type.size is not None):
            self.__sized_bit_variables.add(variable_name)

        if statement.init_expression is None:
            return

        match statement.init_expression:
            case ast.Expression():
                if isinstance(variable_type, ast.BitType):
                    self.__emit_bit_expression_assignment(
                        statement.identifier, statement.init_expression)
                    return

                self.__expression_kind = ExpressionKind.ARITHMETIC
                self.visit(statement.init_expression)
                self.__expression_kind = None

                if (self.__value is None or self.__value_type is None
                        or self.__value_kind is None):
                    raise UninitializedValueException
                self.__emit_assignment(
                    variable_name, ':=', self.__type_of(variable_name),
                    self.__value, self.__value_type, self.__value_kind)

            case ast.QuantumMeasurement():
                if not isinstance(variable_type, ast.BitType):
                    raise InvalidBitOperandException(
                        f'Measurement target {statement.identifier.name} is not '
                        'a bit variable')
                self.__emit_measurement(
                    statement.init_expression, statement.identifier)

            case _:
                raise UnsupportedOpenQASMError(
                    f'classical initializer {type(statement.init_expression).__name__}')

    def visit_ClassicalAssignment(self, statement: ast.ClassicalAssignment) -> None:
        if self.__is_initialization_process:
            return

        source_name = self.__operand_name(statement.lvalue)
        variable_name: str = self.__capitalize_variable_name(source_name)
        variable_type: ValueType = self.__type_of(variable_name)

        if variable_type == ValueType.BIT:
            if statement.op != ast.AssignmentOperator['=']:
                raise UnsupportedOpenQASMError(
                    f'bit assignment operator {statement.op.name}')
            self.__emit_bit_expression_assignment(
                statement.lvalue, statement.rvalue)
            return

        if isinstance(statement.lvalue, ast.IndexedIdentifier):
            raise UnsupportedOpenQASMError('indexed classical assignment')

        if (variable_type == ValueType.BOOL
                and statement.op != ast.AssignmentOperator['=']):
            raise UnsupportedOpenQASMError(
                f'Boolean assignment operator {statement.op.name}')
        expression = statement.rvalue
        if statement.op == ast.AssignmentOperator['=']:
            operator = ':='
        elif statement.op == ast.AssignmentOperator['+=']:
            operator = '+='
        elif statement.op == ast.AssignmentOperator['-=']:
            operator = '-='
        elif statement.op == ast.AssignmentOperator['*=']:
            operator = '*='
        elif statement.op == ast.AssignmentOperator['/=']:
            operator = '/='
        elif statement.op == ast.AssignmentOperator['%=']:
            if variable_type != ValueType.INT:
                raise UnsupportedOpenQASMError('integer remainder assignment requires an int or uint target')
            # Reuse expression lowering and assign only its completed result;
            # the destination may also occur anywhere in the RHS expression.
            operator = ':='
            expression = ast.BinaryExpression(
                ast.BinaryOperator['%'], statement.lvalue, statement.rvalue)
        else:
            raise UnsupportedOpenQASMError(
                f'assignment operator {statement.op.name}')

        self.__expression_kind = ExpressionKind.ARITHMETIC
        self.visit(expression)
        self.__expression_kind = None
        if (self.__value is None or self.__value_type is None
                or self.__value_kind is None):
            raise UninitializedValueException
        self.__emit_assignment(
            variable_name, operator, variable_type, self.__value,
            self.__value_type, self.__value_kind)

def convert(source: str) -> list[str]:
    """Convert an OpenQASM 3 source string into QCX input lines."""
    qasm_ast_root = openqasm3.parse(source)
    converter = QASM2QCXConverter(qasm_ast_root)
    converter.visit(qasm_ast_root)
    return list(converter)


def main(arguments: list[str] | None = None) -> None:
    if arguments is None:
        arguments = sys.argv[1:]
    if len(arguments) != 1:
        raise SystemExit('usage: qasm2qcx.py <OpenQASM file name>')

    try:
        with open(arguments[0], encoding='utf-8') as qasm_file:
            qcx_lines = convert(qasm_file.read())
    except (OSError, QASM2QCXError, openqasm3.parser.QASM3ParsingError) as error:
        raise SystemExit(f'qasm2qcx.py: {error}') from error

    for qcx_line in qcx_lines:
        print(qcx_line)


if __name__ == '__main__':
    main()
