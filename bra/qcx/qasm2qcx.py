import sys
import itertools
import math
import enum

import openqasm3.parser
import openqasm3.ast as ast
import openqasm3.visitor as visitor

ValueType = enum.Enum(
    'ValueType', [('INT', 1), ('FLOAT', 2), ('BIT', 3), ('COMPLEX', 4)])
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

class WrongConstantVariableException(QASM2QCXError):
    def __str__(self):
        return 'Wrong constant variable'

class QASM2QCXConverter(visitor.QASMVisitor):
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
        self.__sized_bit_variables: set[str] = set()
        self.__complex_variable_name_size_map: dict[str, int] = {}

        self.__const_int_variable_name_values_map: dict[str, list[int]] = {}
        self.__const_float_variable_name_values_map: dict[str, list[float]] = {}
        self.__const_complex_variable_name_values_map: dict[str, list[complex]] = {}

        self.__is_stdgates_included: bool = False
        self.__quantum_registers: dict[str, int] = {}
        self.__sized_quantum_registers: set[str] = set()

        self.__is_initialization_process = True
        self.visit(qasm_ast_root)
        self.__is_initialization_process = False

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

    def __type_of(self, identifier_name: str) -> ValueType:
        if identifier_name in self.__bit_variable_name_size_map:
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
            case ValueType.INT:
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

    def visit_UnaryExpression(self, expression: ast.UnaryExpression) -> None:
        if self.__expression_kind is None:
            return

        self.visit(expression.expression)
        if self.__value is None or self.__value_type is None or self.__value_kind is None:
            raise UninitializedValueException
        if self.__value_type == ValueType.BIT:
            raise UnsupportedOpenQASMError('unary arithmetic on bit values')

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

        operators = {
            ast.BinaryOperator['+']: '+=',
            ast.BinaryOperator['-']: '-=',
            ast.BinaryOperator['*']: '*=',
            ast.BinaryOperator['/']: '/=',
        }
        if expression.op not in operators:
            raise UnsupportedOpenQASMError(f'binary operator {expression.op.name}')

        result_type = self.__promoted_type(lhs_value_type, rhs_value_type)
        if lhs_value_kind == ValueKind.LITERAL and rhs_value_kind == ValueKind.LITERAL:
            def divide(lhs, rhs):
                if result_type != ValueType.INT:
                    return lhs / rhs

                # OpenQASM integer division truncates toward zero, as does the
                # QCX integer operation.  Python's // rounds toward negative
                # infinity, so calculate the sign separately.
                quotient = abs(lhs) // abs(rhs)
                return -quotient if (lhs < 0) != (rhs < 0) else quotient

            operations = {
                ast.BinaryOperator['+']: lambda lhs, rhs: lhs + rhs,
                ast.BinaryOperator['-']: lambda lhs, rhs: lhs - rhs,
                ast.BinaryOperator['*']: lambda lhs, rhs: lhs * rhs,
                ast.BinaryOperator['/']: divide,
            }
            self.__value = operations[expression.op](lhs_value, rhs_value)
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
        if not self.__is_initialization_process:
            return

        if statement.filename != 'stdgates.inc':
            raise UnsupportedOpenQASMError(f'include "{statement.filename}"')
        self.__is_stdgates_included = True

    def visit_QubitDeclaration(self, statement: ast.QubitDeclaration) -> None:
        if not self.__is_initialization_process:
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

    def __qubit_operand_indices(
            self, qubit: ast.Identifier | ast.IndexedIdentifier) -> list[int]:
        if isinstance(qubit, ast.Identifier):
            register_name = qubit.name
            if register_name not in self.__quantum_registers:
                raise InvalidQubitOperandException(
                    f'Qubit register {register_name} is not declared')
            return list(range(self.__quantum_registers[register_name]))

        register_name = qubit.name.name
        if register_name not in self.__quantum_registers:
            raise InvalidQubitOperandException(
                f'Qubit register {register_name} is not declared')
        if len(qubit.indices) != 1:
            raise UnsupportedOpenQASMError('multidimensional qubit indexing')

        indices = qubit.indices[0]
        if (not isinstance(indices, list) or len(indices) != 1
                or isinstance(
                    indices[0], (ast.RangeDefinition, ast.DiscreteSet))):
            raise UnsupportedOpenQASMError('qubit index ranges or discrete index sets')
        if not isinstance(indices[0], ast.IntegerLiteral):
            raise UnsupportedOpenQASMError('non-literal qubit index')

        index = indices[0].value
        register_size = self.__quantum_registers[register_name]
        if index < 0 or index >= register_size:
            raise InvalidQubitOperandException(
                f'Qubit index {index} is outside register {register_name}[{register_size}]')
        return [index]

    @staticmethod
    def __operand_name(operand: ast.Identifier | ast.IndexedIdentifier) -> str:
        return operand.name if isinstance(operand, ast.Identifier) else operand.name.name

    def __flattened_qubit_operand(
            self, qubit: ast.Identifier | ast.IndexedIdentifier) -> list[int]:
        register_name = self.__operand_name(qubit)
        return [
            self.__to_qubit_index(register_name, index)
            for index in self.__qubit_operand_indices(qubit)
        ]

    def __bit_operand(
            self, bit: ast.Identifier | ast.IndexedIdentifier, *,
            role: str = 'Bit operand'
            ) -> tuple[str, list[int]]:
        source_name = self.__operand_name(bit)
        variable_name = self.__capitalize_variable_name(source_name)
        if self.__type_of(variable_name) != ValueType.BIT:
            raise InvalidBitOperandException(
                f'{role} {source_name} is not a bit variable')

        size = self.__bit_variable_name_size_map[variable_name]
        if isinstance(bit, ast.Identifier):
            return variable_name, list(range(size))
        if len(bit.indices) != 1:
            raise UnsupportedOpenQASMError('multidimensional bit indexing')

        indices = bit.indices[0]
        if (not isinstance(indices, list) or len(indices) != 1
                or isinstance(
                    indices[0], (ast.RangeDefinition, ast.DiscreteSet))):
            raise UnsupportedOpenQASMError('bit index ranges or discrete index sets')
        if not isinstance(indices[0], ast.IntegerLiteral):
            raise UnsupportedOpenQASMError('non-literal bit index')

        index = indices[0].value
        if index < 0 or index >= size:
            raise InvalidBitOperandException(
                f'Bit index {index} is outside variable {source_name}[{size}]')
        return variable_name, [index]

    @staticmethod
    def __qcx_bit_name(variable_name: str, size: int, index: int) -> str:
        return variable_name if size == 1 else f'{variable_name}:{index}'

    def __emit_measurement(
            self, measurement: ast.QuantumMeasurement,
            target: ast.Identifier | ast.IndexedIdentifier | None) -> None:
        qubit_indices = self.__flattened_qubit_operand(measurement.qubit)

        target_names: list[str] | None = None
        if target is not None:
            variable_name, bit_indices = self.__bit_operand(
                target, role='Measurement target')
            qubit_is_register = (
                isinstance(measurement.qubit, ast.Identifier)
                and measurement.qubit.name in self.__sized_quantum_registers)
            target_is_register = (
                isinstance(target, ast.Identifier)
                and variable_name in self.__sized_bit_variables)
            if qubit_is_register != target_is_register:
                raise InvalidBitOperandException(
                    'Measurement operands must both be scalars or both be '
                    'complete registers')
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

        operand_indices = [
            self.__qubit_operand_indices(qubit) for qubit in statement.qubits
        ]
        loop_size = max(map(len, operand_indices), default=1)
        for indices in operand_indices:
            if len(indices) not in (1, loop_size):
                raise WrongBroadcastingException

        parameters: list[
            tuple[str | int | float | complex, ValueType, ValueKind]
        ] = []
        for argument in statement.arguments:
            self.__expression_kind = ExpressionKind.ARITHMETIC
            self.visit(argument)
            self.__expression_kind = None

            if self.__value_type in (ValueType.BIT, ValueType.COMPLEX):
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

        for index in range(loop_size):
            qubit_indices = []
            for qubit, indices in zip(statement.qubits, operand_indices):
                register_name = qubit.name if isinstance(qubit, ast.Identifier) else qubit.name.name
                register_index = indices[0] if len(indices) == 1 else indices[index]
                qubit_indices.append(str(self.__to_qubit_index(register_name, register_index)))

            if qasm_gate_name == 'cu':
                self.__qcx_lines.append(
                    f'U1 {qubit_indices[0]} {cu_control_phase}')
                self.__qcx_lines.append(
                    f'CU3 {" ".join(qubit_indices)} '
                    f'{" ".join(converted_parameters[:3])}')
            else:
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

        if self.__value_type in (ValueType.BIT, ValueType.COMPLEX):
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
        raise UnsupportedOpenQASMError('reset')

    def visit_QuantumBarrier(self, statement: ast.QuantumBarrier) -> None:
        raise UnsupportedOpenQASMError('barrier')

    def visit_DelayInstruction(self, statement: ast.DelayInstruction) -> None:
        raise UnsupportedOpenQASMError('delay')

    def visit_Box(self, statement: ast.Box) -> None:
        raise UnsupportedOpenQASMError('box')

    def visit_AliasStatement(self, statement: ast.AliasStatement) -> None:
        raise UnsupportedOpenQASMError('alias')

    def visit_BranchingStatement(self, statement: ast.BranchingStatement) -> None:
        raise UnsupportedOpenQASMError('branching statement')

    def visit_ForInLoop(self, statement: ast.ForInLoop) -> None:
        raise UnsupportedOpenQASMError('for loop')

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
        raise UnsupportedOpenQASMError('pragma')

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
        raise UnsupportedOpenQASMError('Boolean literal')

    def visit_Cast(self, expression: ast.Cast) -> None:
        if self.__expression_kind is None:
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
        raise UnsupportedOpenQASMError('index expression')

    def visit_SizeOf(self, expression: ast.SizeOf) -> None:
        raise UnsupportedOpenQASMError('sizeof expression')

    def __make_constant_variable(self, variable_name: str, variable_type, num_elements: int = 1) -> None:
        if num_elements <= 0:
            raise InvalidDeclarationException(
                f'Constant {variable_name} must have a positive size')

        match variable_type:
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

            #case ast.BoolType():
            #    pass

            case ast.ComplexType():
                if variable_name in self.__const_complex_variable_name_values_map:
                    raise WrongConstantVariableException(variable_name)

                self.__const_complex_variable_name_values_map[variable_name] = [complex()] * num_elements

    def __set_constant_variable(self, variable_name: str, variable_type) -> None:
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

            #case ast.BoolType():
            #    pass

            case ast.ComplexType():
                if variable_name not in self.__const_complex_variable_name_values_map:
                    raise WrongConstantVariableException(variable_name)

                self.__const_complex_variable_name_values_map[variable_name][0] = complex(self.__value)

    def visit_ConstantDeclaration(self, statement: ast.ConstantDeclaration) -> None:
        if not self.__is_initialization_process:
            return

        variable_type = statement.type
        variable_name = statement.identifier.name
        self.__register_source_identifier(variable_name)

        if isinstance(variable_type, ast.ArrayType):
            raise UnsupportedOpenQASMError('constant array')
        if not isinstance(variable_type, (ast.IntType, ast.UintType, ast.FloatType, ast.ComplexType)):
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
        variable_name, target_indices = self.__bit_operand(target)
        target_size = self.__bit_variable_name_size_map[variable_name]
        target_names = [
            self.__qcx_bit_name(variable_name, target_size, index)
            for index in target_indices
        ]

        if isinstance(expression, ast.BitstringLiteral):
            if expression.width != len(target_names):
                raise InvalidBitOperandException(
                    f'Bit-string width {expression.width} does not match target '
                    f'size {len(target_names)}')
            values = [
                (expression.value >> index) & 1
                for index in range(expression.width)
            ]
        elif isinstance(expression, ast.IntegerLiteral):
            if len(target_names) != 1 or expression.value not in (0, 1):
                raise InvalidBitOperandException(
                    'An integer bit initializer must be 0 or 1 and target one bit')
            values = [expression.value]
        elif isinstance(expression, ast.Identifier):
            source_name, source_indices = self.__bit_operand(
                expression, role='Bit source')
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
            if (not isinstance(expression.collection, ast.Identifier)
                    or not isinstance(expression.index, list)
                    or len(expression.index) != 1
                    or not isinstance(expression.index[0], ast.IntegerLiteral)):
                raise UnsupportedOpenQASMError(
                    'bit index ranges or non-literal bit indices')
            if len(target_names) != 1:
                raise InvalidBitOperandException(
                    'An indexed bit source must target one bit')

            source_name = self.__capitalize_variable_name(
                expression.collection.name)
            if self.__type_of(source_name) != ValueType.BIT:
                raise InvalidBitOperandException(
                    f'Bit source {expression.collection.name} is not a bit '
                    'variable')
            source_size = self.__bit_variable_name_size_map[source_name]
            source_index = expression.index[0].value
            if source_index < 0 or source_index >= source_size:
                raise InvalidBitOperandException(
                    f'Bit index {source_index} is outside variable '
                    f'{expression.collection.name}[{source_size}]')
            values = [
                self.__qcx_bit_name(
                    source_name, source_size, source_index)
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
                 ast.ComplexType)):
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
        else:
            raise UnsupportedOpenQASMError(
                f'assignment operator {statement.op.name}')

        self.__expression_kind = ExpressionKind.ARITHMETIC
        self.visit(statement.rvalue)
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
