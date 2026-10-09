import sys
import itertools
import math
import enum
import dataclasses
import contextlib
import ctypes
from collections.abc import Iterator

import openqasm3.parser
import openqasm3.ast as ast
import openqasm3.visitor as visitor

ValueType = enum.Enum(
    'ValueType', [('INT', 1), ('FLOAT', 2), ('BIT', 3), ('COMPLEX', 4), ('BOOL', 5), ('UINT', 6)])
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


class InvalidArrayOperandException(QASM2QCXError):
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

class InvalidShiftException(QASM2QCXError):
    """Invalid constant shift count or overflowing signed left shift."""

class WrongConstantVariableException(QASM2QCXError):
    def __str__(self):
        return 'Wrong constant variable'


@dataclasses.dataclass(frozen=True)
class _UnsignedIntegerType:
    # Keep specified and target-defined widths distinct even when both use
    # the same number of bits. All runtime storage will use native QCX UINT.
    width: int
    is_sized: bool

    def __post_init__(self) -> None:
        if self.width <= 0:
            raise ValueError('unsigned integer width must be positive')

    @property
    def mask(self) -> int:
        return (1 << self.width) - 1

    def normalize(self, value: int) -> int:
        return value & self.mask


@dataclasses.dataclass(frozen=True)
class _RuntimeIteratorBinding:
    # Unlike an unrolled iterator's integer value, this names live QCX storage.
    storage: str
    value_type: ValueType = ValueType.INT


@dataclasses.dataclass(frozen=True)
class _LoopContext:
    # Control-flow targets are independent of iterator bindings.
    # A for-loop continues at the iteration end; a while-loop at its condition.
    break_label: str
    continue_label: str


class QASM2QCXConverter(visitor.QASMVisitor):
    # bra::int_type is C++ int. These bounds describe a backend built for the
    # converter's host; OpenQASM declared widths do not change QCX storage.
    QCX_INT_MIN = -(1 << (ctypes.sizeof(ctypes.c_int) * 8 - 1))
    QCX_INT_MAX = (1 << (ctypes.sizeof(ctypes.c_int) * 8 - 1)) - 1
    QCX_UINT_WIDTH = ctypes.sizeof(ctypes.c_uint) * 8
    QCX_UINT_MAX = (1 << QCX_UINT_WIDTH) - 1

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
        self.__classical_source_types: dict[str, ast.QASMNode] = {}
        self.__constant_source_types: dict[str, ast.QASMNode] = {}
        self.__integer_array_source_sizes: dict[str, int] = {}
        self.__bit_register_source_sizes: dict[str, int] = {}

        self.__int_variable_name_size_map: dict[str, int] = {}
        self.__uint_variable_name_size_map: dict[str, int] = {}
        self.__uint_variable_types: dict[str, _UnsignedIntegerType] = {}
        self.__temporary_uint_types: dict[str, _UnsignedIntegerType] = {}
        # Retain the node as well as its id: synthetic compound-assignment
        # expressions must not leave entries whose ids can later be reused.
        self.__uint_expression_types: dict[int, tuple[ast.Expression, _UnsignedIntegerType]] = {}
        self.__integer_array_names: set[str] = set()
        self.__float_variable_name_size_map: dict[str, int] = {}
        self.__bit_variable_name_size_map: dict[str, int] = {}
        self.__bool_variable_names: set[str] = set()
        self.__sized_bit_variables: set[str] = set()
        self.__complex_variable_name_size_map: dict[str, int] = {}

        self.__const_int_variable_name_values_map: dict[str, list[int]] = {}
        self.__const_uint_variable_name_values_map: dict[str, list[int]] = {}
        self.__const_uint_types: dict[str, _UnsignedIntegerType] = {}
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
        self.__array_index: int = 0
        self.__condition_index: int = 0
        self.__branch_depth: int = 0
        self.__boolean_expression_index: int = 0
        self.__evaluate_constant: bool = True
        self.__loop_bindings: list[dict[str, int | _RuntimeIteratorBinding]] = []
        self.__loop_contexts: list[_LoopContext] = []
        self.__loop_index = 0
        self.__expanded_loop_depth = 0
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
        inside_loop = self.__inside_expanded_loop()
        if inside_loop and isinstance(node, ast.Statement):
            self.__expanded_loop_statements += 1
            if self.__expanded_loop_statements > self.MAX_EXPANDED_LOOP_STATEMENTS:
                raise InvalidLoopRangeException(
                    f'For-loop expansion exceeds {self.MAX_EXPANDED_LOOP_STATEMENTS} statements')
        result = super().visit(node, context)
        if isinstance(node, ast.Expression) and self.__value_type == ValueType.UINT:
            if isinstance(node, ast.Identifier):
                info = (self.__const_uint_types.get(node.name)
                        or self.__uint_variable_types.get(self.__capitalize_variable_name(node.name)))
                if info is not None:
                    self.__uint_expression_types[id(node)] = (node, info)
            elif isinstance(node, ast.UnaryExpression):
                info = self.__uint_expression_type(node.expression)
                if info is not None:
                    self.__uint_expression_types[id(node)] = (node, info)
        if inside_loop:
            self.__check_loop_output_limit()
        return result

    def __inside_expanded_loop(self) -> bool:
        # Runtime ranges emit one body, but runtime-valued sets expand their
        # fixed element count. An enclosing expansion still charges nested code.
        return self.__expanded_loop_depth > 0 or any(isinstance(value, int)
                   for scope in self.__loop_bindings for value in scope.values())

    def __check_loop_output_limit(self) -> None:
        if (not self.__is_initialization_process
                and len(self.__qcx_lines) > self.MAX_LOOP_OUTPUT_LINES):
            raise InvalidLoopRangeException(
                f'For-loop output exceeds {self.MAX_LOOP_OUTPUT_LINES} QCX lines')

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
        elif identifier_name in self.__uint_variable_name_size_map:
            return ValueType.UINT
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

    def __add_new_temporary_variable(
            self, value_type: ValueType,
            uint_type: _UnsignedIntegerType | None = None) -> str:
        if uint_type is not None and value_type != ValueType.UINT:
            raise ValueError('unsigned width supplied for a non-UINT temporary')
        match value_type:
            case ValueType.INT | ValueType.BOOL | ValueType.BIT:
                type_str = 'INT'
            case ValueType.UINT:
                type_str = 'UINT'
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
        if value_type == ValueType.UINT:
            self.__temporary_uint_types[temporary_variable] = (
                uint_type if uint_type is not None
                else _UnsignedIntegerType(self.QCX_UINT_WIDTH, False))
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
                binding = scope[expression.name]
                if isinstance(binding, _RuntimeIteratorBinding):
                    if self.__expression_kind == ExpressionKind.CONST_ARITHMETIC:
                        raise NoConstantExpressionException
                    self.__value = binding.storage
                    self.__value_kind = ValueKind.LVALUE
                    self.__value_type = binding.value_type
                else:
                    self.__value = binding
                    self.__value_kind = ValueKind.LITERAL
                    self.__value_type = ValueType.INT
                return

        is_user_constant = any(
            expression.name in constant_values
            for constant_values in (
                self.__const_int_variable_name_values_map,
                self.__const_uint_variable_name_values_map,
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
        elif expression.name in self.__const_uint_variable_name_values_map:
            self.__value = self.__const_uint_variable_name_values_map[expression.name][0]
            self.__value_type = ValueType.UINT
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
            if self.__value in self.__integer_array_names:
                raise UnsupportedOpenQASMError('whole integer array in scalar expression')
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
        if expression.op == ast.UnaryOperator['~']:
            self.__visit_bitwise_complement(expression.expression)
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
            uint_type = self.__uint_expression_type(expression.expression)
            if uint_type is not None:
                self.__value = uint_type.normalize(self.__value)
            return
        if self.__expression_kind == ExpressionKind.CONST_ARITHMETIC:
            raise NoConstantExpressionException

        operand = self.__value
        operand_kind = self.__value_kind
        uint_type = self.__uint_expression_type(expression.expression)
        if self.__value_type == ValueType.UINT:
            self.__uint_expression_types[id(expression)] = (expression, uint_type)
        temporary_variable = self.__add_new_temporary_variable(self.__value_type, uint_type)
        self.__qcx_lines.append(f'LET {temporary_variable} := {operand}')
        multiplier = (':COMPLEX:-1.0' if self.__value_type == ValueType.COMPLEX
                      else ':UINT:-1' if self.__value_type == ValueType.UINT else '-1')
        self.__qcx_lines.append(f'LET {temporary_variable} *= {multiplier}')
        if uint_type is not None:
            self.__normalize_uint_storage(temporary_variable, uint_type)
        if operand_kind == ValueKind.TEMPORARY:
            self.__release_temporary_variable(operand)
        self.__value = temporary_variable
        self.__value_kind = ValueKind.TEMPORARY

    @staticmethod
    def __bitwise_type(lhs_type: ValueType, rhs_type: ValueType | None = None) -> ValueType:
        if lhs_type in (ValueType.INT, ValueType.UINT) and rhs_type in (None, ValueType.INT, ValueType.UINT):
            return ValueType.UINT if ValueType.UINT in (lhs_type, rhs_type) else ValueType.INT
        if lhs_type != ValueType.BIT or rhs_type not in (None, ValueType.BIT):
            raise UnsupportedOpenQASMError(
                'bitwise operators require matching integer or scalar bit operands')
        return lhs_type

    def __bitwise_operand(
            self, expression: ast.Expression
            ) -> tuple[str | int | float | complex, ValueType, ValueKind]:
        self.visit(expression)
        if self.__value is None or self.__value_type is None or self.__value_kind is None:
            raise UninitializedValueException
        if (self.__value_kind == ValueKind.LITERAL and self.__value_type == ValueType.INT
                and not self.QCX_INT_MIN <= self.__value <= self.QCX_UINT_MAX):
            raise UnsupportedOpenQASMError('bitwise integer operand outside QCX INT range')
        return self.__value, self.__value_type, self.__value_kind

    def __visit_bitwise_complement(self, expression: ast.Expression) -> None:
        value, value_type, value_kind = self.__bitwise_operand(expression)
        try:
            result_type = self.__bitwise_type(value_type)
            uint_type = self.__uint_expression_type(expression)
            mask = 1 if result_type == ValueType.BIT else uint_type.mask if uint_type else -1
            if result_type == ValueType.INT and value_kind == ValueKind.LITERAL and int(value) > self.QCX_INT_MAX:
                raise UnsupportedOpenQASMError('bitwise integer operand outside QCX INT range')
            if value_kind == ValueKind.LITERAL:
                self.__value = (int(value) ^ mask) if self.__evaluate_constant else 0
                self.__value_type = result_type
                self.__value_kind = ValueKind.LITERAL
                return
            if self.__expression_kind == ExpressionKind.CONST_ARITHMETIC:
                raise NoConstantExpressionException
            result = self.__add_new_temporary_variable(result_type, uint_type)
            self.__qcx_lines.extend([f'LET {result} := {value}', f'LET {result} ^= {mask}'])
            self.__value, self.__value_type, self.__value_kind = result, result_type, ValueKind.TEMPORARY
        finally:
            if value_kind == ValueKind.TEMPORARY:
                self.__release_temporary_variable(str(value))

    def __visit_bitwise_binary(self, expression: ast.BinaryExpression) -> None:
        operands = []
        try:
            # Evaluate in source order and keep both operand temporaries live.
            operands.append(self.__bitwise_operand(expression.lhs))
            operands.append(self.__bitwise_operand(expression.rhs))
            (lhs, lhs_type, lhs_kind), (rhs, rhs_type, rhs_kind) = operands
            result_type = self.__bitwise_type(lhs_type, rhs_type)
            uint_type = None
            if result_type != ValueType.BIT:
                result_type, uint_type = self.__integer_expression_type(
                    lhs_type, rhs_type, expression.lhs, expression.rhs)
            if result_type == ValueType.INT and any(
                    kind == ValueKind.LITERAL and not self.QCX_INT_MIN <= int(value) <= self.QCX_INT_MAX
                    for value, _, kind in operands):
                raise UnsupportedOpenQASMError('bitwise integer operand outside QCX INT range')
            if uint_type is not None:
                self.__uint_expression_types[id(expression)] = (expression, uint_type)
            if lhs_kind == rhs_kind == ValueKind.LITERAL:
                operations = {
                    '&': lambda lhs, rhs: lhs & rhs,
                    '|': lambda lhs, rhs: lhs | rhs,
                    '^': lambda lhs, rhs: lhs ^ rhs,
                }
                self.__value = (operations[expression.op.name](int(lhs), int(rhs))
                                if self.__evaluate_constant else 0)
                if uint_type is not None:
                    self.__value = uint_type.normalize(self.__value)
                elif result_type == ValueType.INT and any(
                        not self.QCX_INT_MIN <= int(value) <= self.QCX_INT_MAX for value in (lhs, rhs)):
                    raise UnsupportedOpenQASMError('bitwise integer operand outside QCX INT range')
                self.__value_type, self.__value_kind = result_type, ValueKind.LITERAL
                return
            if self.__expression_kind == ExpressionKind.CONST_ARITHMETIC:
                raise NoConstantExpressionException
            left, left_temporaries = self.__converted_operand(lhs, lhs_type, lhs_kind, result_type)
            right, right_temporaries = self.__converted_operand(rhs, rhs_type, rhs_kind, result_type)
            result = self.__add_new_temporary_variable(result_type, uint_type)
            self.__qcx_lines.extend([f'LET {result} := {left}',
                                    f'LET {result} {expression.op.name}= {right}'])
            if uint_type is not None:
                self.__normalize_uint_storage(result, uint_type)
            for temporary in left_temporaries | right_temporaries:
                self.__release_temporary_variable(temporary)
            self.__value, self.__value_type, self.__value_kind = result, result_type, ValueKind.TEMPORARY
        finally:
            for value, _, kind in operands:
                if kind == ValueKind.TEMPORARY:
                    self.__release_temporary_variable(str(value))

    @staticmethod
    def __promoted_type(lhs_type: ValueType, rhs_type: ValueType) -> ValueType:
        arithmetic_types = (ValueType.INT, ValueType.UINT, ValueType.FLOAT, ValueType.COMPLEX)
        if lhs_type not in arithmetic_types or rhs_type not in arithmetic_types:
            raise NoImplicitCastException
        if ValueType.COMPLEX in (lhs_type, rhs_type):
            return ValueType.COMPLEX
        if ValueType.FLOAT in (lhs_type, rhs_type):
            return ValueType.FLOAT
        if ValueType.UINT in (lhs_type, rhs_type):
            return ValueType.UINT
        return ValueType.INT

    def __uint_expression_type(self, expression: ast.Expression) -> _UnsignedIntegerType | None:
        entry = self.__uint_expression_types.get(id(expression))
        if entry is not None:
            return entry[1]
        return None

    def __integer_expression_type(
            self, lhs_type: ValueType, rhs_type: ValueType,
            lhs: ast.Expression, rhs: ast.Expression) -> tuple[ValueType, _UnsignedIntegerType | None]:
        result_type = self.__promoted_type(lhs_type, rhs_type)
        if result_type != ValueType.UINT:
            return result_type, None
        infos = [self.__uint_expression_type(node)
                 for node, value_type in ((lhs, lhs_type), (rhs, rhs_type)) if value_type == ValueType.UINT]
        if any(info is None for info in infos):
            raise UninitializedValueException
        width = max(info.width for info in infos)
        if ValueType.INT in (lhs_type, rhs_type):
            for node, value_type in ((lhs, lhs_type), (rhs, rhs_type)):
                if value_type == ValueType.INT and not self.__native_signed_operand(node):
                    raise UnsupportedOpenQASMError('mixed signed/unsigned expression with a non-native signed width')
            if width < self.QCX_UINT_WIDTH:
                return ValueType.INT, None
            return ValueType.UINT, _UnsignedIntegerType(self.QCX_UINT_WIDTH, False)
        return ValueType.UINT, _UnsignedIntegerType(width, any(info.is_sized for info in infos))

    def __native_signed_operand(
            self, expression: ast.Expression,
            iterators: set[str] | dict[str, ValueType] | None = None) -> bool:
        # Signed width emulation is not part of UINT support. Do not silently
        # assign native signed rank to a specified non-native signed type.
        source_type = None
        if isinstance(expression, (ast.Identifier, ast.IndexedIdentifier)):
            name = self.__operand_name(expression)
            if (iterators is not None and name in iterators) or any(name in scope for scope in self.__loop_bindings):
                return True
            source_type = self.__classical_source_types.get(name) or self.__constant_source_types.get(name)
        elif isinstance(expression, ast.IndexExpression) and isinstance(expression.collection, ast.Identifier):
            source_type = self.__classical_source_types.get(expression.collection.name)
        elif isinstance(expression, ast.Cast):
            source_type = expression.type
        elif isinstance(expression, ast.UnaryExpression):
            return self.__native_signed_operand(expression.expression, iterators)
        elif isinstance(expression, ast.BinaryExpression):
            return (self.__native_signed_operand(expression.lhs, iterators)
                    and self.__native_signed_operand(expression.rhs, iterators))
        if isinstance(source_type, ast.ArrayType):
            source_type = source_type.base_type
        if not isinstance(source_type, ast.IntType) or source_type.size is None:
            return True
        return self.__constant_loop_integer(source_type.size, 'signed operand width') == self.QCX_UINT_WIDTH

    def __converted_operand(
            self, value: str | int | float | complex, value_type: ValueType,
            value_kind: ValueKind, result_type: ValueType) -> tuple[str, set[str]]:
        if result_type == ValueType.BOOL:
            if value_type in (ValueType.BOOL, ValueType.BIT):
                return str(value), set()
            if (value_type in (ValueType.INT, ValueType.UINT) and value_kind == ValueKind.LITERAL
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
                if value_type not in (ValueType.INT, ValueType.UINT, ValueType.BIT) or value not in (0, 1):
                    raise NoImplicitCastException
                return str(int(value)), set()
            if result_type == ValueType.INT:
                if value_type not in (ValueType.INT, ValueType.UINT):
                    raise NoImplicitCastException
                if value_type == ValueType.UINT and not 0 <= int(value) <= self.QCX_INT_MAX:
                    return f':INT::UINT:{value}', set()
                return str(int(value)), set()
            if result_type == ValueType.UINT:
                if value_type not in (ValueType.INT, ValueType.UINT):
                    raise NoImplicitCastException
                return str(int(value) & self.QCX_UINT_MAX), set()
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
        if result_type == ValueType.UINT and value_type == ValueType.INT:
            return f':UINT:{value}', set()
        if result_type == ValueType.INT and value_type == ValueType.UINT:
            return f':INT:{value}', set()
        if result_type == ValueType.FLOAT and value_type in (ValueType.INT, ValueType.UINT):
            return f':REAL:{value}', set()
        if result_type == ValueType.COMPLEX:
            return f':COMPLEX:{value}', set()
        raise NoImplicitCastException

    def __emit_assignment(
            self, variable_name: str, operator: str, variable_type: ValueType,
            value: str | int | float | complex, value_type: ValueType,
            value_kind: ValueKind,
            value_uint_type: _UnsignedIntegerType | None = None) -> None:
        if (variable_type == ValueType.INT and value_type == ValueType.UINT
                and operator != ':='):
            uint_type = (value_uint_type or self.__temporary_uint_types.get(str(value))
                         or self.__uint_variable_types.get(str(value).split(':')[0])
                         or _UnsignedIntegerType(self.QCX_UINT_WIDTH, False))
            if uint_type.width == self.QCX_UINT_WIDTH:
                temporary = self.__add_new_temporary_variable(ValueType.UINT)
                self.__qcx_lines.append(f'LET {temporary} := :UINT:{variable_name}')
                self.__qcx_lines.append(f'LET {temporary} {operator} {value}')
                self.__qcx_lines.append(f'LET {variable_name} := :INT:{temporary}')
                self.__release_temporary_variable(temporary)
                if value_kind == ValueKind.TEMPORARY:
                    self.__release_temporary_variable(str(value))
                return
        rhs, materialized_temporaries = self.__converted_operand(
            value, value_type, value_kind, variable_type)
        self.__qcx_lines.append(f'LET {variable_name} {operator} {rhs}')
        if variable_type == ValueType.UINT:
            uint_type = (self.__uint_variable_types.get(variable_name.split(':')[0])
                         or self.__temporary_uint_types.get(variable_name))
            if uint_type is not None:
                self.__normalize_uint_storage(variable_name, uint_type)

        temporaries_to_release = set(materialized_temporaries)
        if value_kind == ValueKind.TEMPORARY:
            temporaries_to_release.add(str(value))
        for temporary in temporaries_to_release:
            self.__release_temporary_variable(temporary)

    def visit_BinaryExpression(self, expression: ast.BinaryExpression) -> None:
        if self.__expression_kind is None:
            return

        if expression.op.name in ('<<', '>>'):
            self.__visit_shift_binary(expression)
            return
        if expression.op.name in ('&', '|', '^'):
            self.__visit_bitwise_binary(expression)
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
            if lhs_value_type not in (ValueType.INT, ValueType.UINT) or rhs_value_type not in (ValueType.INT, ValueType.UINT):
                raise UnsupportedOpenQASMError('integer remainder requires int or uint operands')

        operators = {
            ast.BinaryOperator['+']: '+=',
            ast.BinaryOperator['-']: '-=',
            ast.BinaryOperator['*']: '*=',
            ast.BinaryOperator['/']: '/=',
        }
        if expression.op not in operators and not is_remainder:
            raise UnsupportedOpenQASMError(f'binary operator {expression.op.name}')

        result_type, uint_type = self.__integer_expression_type(
            lhs_value_type, rhs_value_type, expression.lhs, expression.rhs)
        if uint_type is not None:
            self.__uint_expression_types[id(expression)] = (expression, uint_type)
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
                if result_type not in (ValueType.INT, ValueType.UINT):
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
            if uint_type is not None:
                lhs_value = uint_type.normalize(int(lhs_value))
                rhs_value = uint_type.normalize(int(rhs_value))
            self.__value = (operations[expression.op](lhs_value, rhs_value)
                            if self.__evaluate_constant else 0)
            if uint_type is not None:
                self.__value = uint_type.normalize(self.__value)
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
            result = self.__emit_integer_remainder(lhs, rhs, result_type, uint_type)
        else:
            result = self.__add_new_temporary_variable(result_type, uint_type)
            self.__qcx_lines.append(f'LET {result} := {lhs}')
            self.__qcx_lines.append(f'LET {result} {operators[expression.op]} {rhs}')
            if uint_type is not None:
                self.__normalize_uint_storage(result, uint_type)

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

    def __validate_shift_types(
            self, expression: ast.BinaryExpression, lhs_type: ValueType, rhs_type: ValueType,
            iterators: set[str] | dict[str, ValueType] | None = None) -> None:
        if lhs_type not in (ValueType.INT, ValueType.UINT) or rhs_type not in (ValueType.INT, ValueType.UINT):
            raise UnsupportedOpenQASMError('shift operators require scalar integer operands')
        if lhs_type == ValueType.INT and not self.__native_signed_operand(expression.lhs, iterators):
            raise UnsupportedOpenQASMError('shift of an integer with a non-native signed width')

    def __fold_shift(
            self, lhs: int, rhs: int, operator: str,
            uint_type: _UnsignedIntegerType | None) -> int:
        width = uint_type.width if uint_type is not None else self.QCX_UINT_WIDTH
        if not 0 <= rhs < width:
            raise InvalidShiftException(f'Shift count {rhs} must be in [0, {width - 1}]')
        result = lhs << rhs if operator == '<<' else lhs >> rhs
        if uint_type is not None:
            return uint_type.normalize(result)
        if not self.QCX_INT_MIN <= result <= self.QCX_INT_MAX:
            raise InvalidShiftException('Signed left shift result outside QCX INT range')
        return result

    def __visit_shift_binary(self, expression: ast.BinaryExpression) -> None:
        operands = []
        try:
            # A count is independent of the shifted value's type. Do not use
            # arithmetic promotion or narrow a UINT count through signed INT.
            for operand in (expression.lhs, expression.rhs):
                self.visit(operand)
                if self.__value is None or self.__value_type is None or self.__value_kind is None:
                    raise UninitializedValueException
                operands.append((self.__value, self.__value_type, self.__value_kind))
            (lhs, lhs_type, lhs_kind), (rhs, rhs_type, rhs_kind) = operands
            self.__validate_shift_types(expression, lhs_type, rhs_type)
            uint_type = self.__uint_expression_type(expression.lhs) if lhs_type == ValueType.UINT else None
            if lhs_type == ValueType.UINT:
                if uint_type is None:
                    raise UninitializedValueException
                self.__uint_expression_types[id(expression)] = (expression, uint_type)
            if (lhs_type == ValueType.INT and lhs_kind == ValueKind.LITERAL
                    and not self.QCX_INT_MIN <= int(lhs) <= self.QCX_INT_MAX):
                raise UnsupportedOpenQASMError('shift operand outside QCX INT range')
            if lhs_kind == rhs_kind == ValueKind.LITERAL:
                if not self.__evaluate_constant:
                    self.__value, self.__value_type, self.__value_kind = 0, lhs_type, ValueKind.LITERAL
                    return
                try:
                    folded = self.__fold_shift(int(lhs), int(rhs), expression.op.name, uint_type)
                except InvalidShiftException:
                    if self.__expression_kind == ExpressionKind.CONST_ARITHMETIC:
                        raise
                    # Emit the check at runtime even for literal operands:
                    # enclosing branches or logical operators may skip it.
                else:
                    self.__value, self.__value_type, self.__value_kind = folded, lhs_type, ValueKind.LITERAL
                    return
            if self.__expression_kind == ExpressionKind.CONST_ARITHMETIC:
                raise NoConstantExpressionException
            self.__emit_shift_expression(expression.op.name, operands, uint_type)
        finally:
            for value, _, kind in operands:
                if kind == ValueKind.TEMPORARY:
                    self.__release_temporary_variable(str(value))

    def __emit_shift_expression(
            self, operator: str,
            operands: list[tuple[str | int | float | complex, ValueType, ValueKind]],
            uint_type: _UnsignedIntegerType | None) -> None:
        (lhs, lhs_type, _), (rhs, rhs_type, rhs_kind) = operands
        count_temporary = None
        try:
            if uint_type is not None and uint_type.width < self.QCX_UINT_WIDTH:
                # bra checks the native width. Narrow source widths need their
                # own guards; ASSERT requires a variable on the left side.
                if rhs_kind == ValueKind.LITERAL:
                    count_temporary = self.__add_new_temporary_variable(rhs_type)
                    self.__qcx_lines.append(f'LET {count_temporary} := {rhs}')
                    rhs = count_temporary
                self.__qcx_lines.extend([f'ASSERT {rhs} >= 0', f'ASSERT {rhs} < {uint_type.width}'])
            result = self.__add_new_temporary_variable(lhs_type, uint_type)
            self.__qcx_lines.extend([f'LET {result} := {lhs}', f'LET {result} {operator}= {rhs}'])
            if uint_type is not None:
                self.__normalize_uint_storage(result, uint_type)
            self.__value, self.__value_type, self.__value_kind = result, lhs_type, ValueKind.TEMPORARY
        finally:
            if count_temporary is not None:
                self.__release_temporary_variable(count_temporary)

    @staticmethod
    def __integer_quotient(lhs: int, rhs: int) -> int:
        # Match truncation toward zero without converting to float or using
        # Python's floor division on signed operands. Remainder uses the same
        # quotient so that lhs == quotient * rhs + remainder.
        quotient = abs(lhs) // abs(rhs)
        return -quotient if (lhs < 0) != (rhs < 0) else quotient

    def __emit_integer_remainder(
            self, lhs: str, rhs: str, value_type: ValueType = ValueType.INT,
            uint_type: _UnsignedIntegerType | None = None) -> str:
        # Keep both operands live throughout the calculation. In particular,
        # neither the quotient nor the result may alias an operand temporary.
        result = self.__add_new_temporary_variable(value_type, uint_type)
        quotient = self.__add_new_temporary_variable(value_type, uint_type)
        self.__qcx_lines.extend([
            f'LET {result} := {lhs}', f'LET {quotient} := {lhs}',
            f'LET {quotient} /= {rhs}', f'LET {quotient} *= {rhs}',
            f'LET {result} -= {quotient}',
        ])
        self.__release_temporary_variable(quotient)
        if uint_type is not None:
            self.__normalize_uint_storage(result, uint_type)
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
        if self.__value_kind != ValueKind.LITERAL or self.__value_type not in (ValueType.INT, ValueType.UINT):
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
        if self.__loop_bindings or kind == 'bit':
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
        if role == 'Bit source' and isinstance(bit, ast.Identifier):
            for scope in reversed(self.__loop_bindings):
                if source_name in scope:
                    binding = scope[source_name]
                    if isinstance(binding, _RuntimeIteratorBinding) and binding.value_type == ValueType.BIT:
                        return binding.storage, [0], False
                    raise InvalidBitOperandException(f'{role} {source_name} is not a bit variable')
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

    @contextlib.contextmanager
    def __bit_target(
            self, target: ast.Identifier | ast.IndexedIdentifier, *,
            role: str = 'Bit operand') -> Iterator[tuple[list[str], bool]]:
        if isinstance(target, ast.IndexedIdentifier) and len(target.indices) == 1:
            source_name = target.name.name
            variable_name = self.__capitalize_variable_name(source_name)
            if self.__type_of(variable_name) != ValueType.BIT:
                raise InvalidBitOperandException(f'{role} {source_name} is not a bit variable')
            size = self.__bit_variable_name_size_map[variable_name]
            access = self.__runtime_bit_access(target.indices[0], variable_name, size)
            if access is not None:
                operand, index = access
                try:
                    # Reserve the destination across RHS evaluation or measurement.
                    yield [operand], False
                finally:
                    self.__release_temporary_variable(index)
                return
        variable_name, indices, is_register = self.__bit_operand(target, role=role)
        size = self.__bit_variable_name_size_map[variable_name]
        yield [self.__qcx_bit_name(variable_name, size, index) for index in indices], is_register

    def __emit_measurement(
            self, measurement: ast.QuantumMeasurement,
            target: ast.Identifier | ast.IndexedIdentifier | None) -> None:
        qubit_indices, qubit_is_register = self.__flattened_qubit_operand(
            measurement.qubit)

        target_context = (self.__bit_target(target, role='Measurement target')
                          if target is not None else contextlib.nullcontext((None, False)))
        with target_context as (target_names, target_is_register):
            if target_names is not None and qubit_is_register != target_is_register:
                raise InvalidBitOperandException(
                    'Measurement operands must both be scalars or both be registers')
            if target_names is not None and len(qubit_indices) != len(target_names):
                raise MeasurementSizeMismatchException(
                    len(qubit_indices), len(target_names))

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
        if value_type in (ValueType.INT, ValueType.UINT) and value_kind != ValueKind.LITERAL:
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
            if variable_name in self.__integer_array_names:
                operand, index = self.__integer_array_access(expression)
                if index is None:
                    return operand, ValueType.INT, ValueKind.LVALUE
                try:
                    result = self.__add_new_temporary_variable(ValueType.INT)
                    self.__qcx_lines.append(f'LET {result} := {operand}')
                    return result, ValueType.INT, ValueKind.TEMPORARY
                finally:
                    self.__release_temporary_variable(index)
            if self.__type_of(variable_name) != ValueType.BIT:
                raise UnsupportedOpenQASMError(
                    'indexed non-bit conditional operand')

            size = self.__bit_variable_name_size_map[variable_name]
            runtime_read = self.__runtime_bit_read(expression, variable_name, size)
            if runtime_read is not None:
                return runtime_read
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
                and str(self.__value) in self.__sized_bit_variables):
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
        if ValueType.UINT in (numeric_lhs_type, numeric_rhs_type):
            result_type, _ = self.__integer_expression_type(
                numeric_lhs_type, numeric_rhs_type, condition.lhs, condition.rhs)

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
            if value_type not in (ValueType.BOOL, ValueType.BIT, ValueType.INT, ValueType.UINT, ValueType.FLOAT):
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
            if condition.op == ast.UnaryOperator['!']:
                self.__emit_condition(
                    condition.expression, false_label, true_label)
                return
            if condition.op != ast.UnaryOperator['~']:
                raise UnsupportedOpenQASMError(
                    f'branching condition unary operator {condition.op.name}')

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
            if condition.op.name not in ('&', '|', '^'):
                self.__emit_comparison_condition(
                    condition, true_label, false_label)
                return

        if not isinstance(condition, (
                ast.Identifier, ast.IndexExpression, ast.BooleanLiteral, ast.Cast,
                ast.UnaryExpression, ast.BinaryExpression)):
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
                if ValueType.UINT in (lhs_type, rhs_type):
                    result_type, uint_type = self.__integer_expression_type(
                        lhs_type, rhs_type, expression.lhs, expression.rhs)
                    if uint_type is not None:
                        lhs_value = uint_type.normalize(int(lhs_value))
                        rhs_value = uint_type.normalize(int(rhs_value))
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

    def __constant_loop_integer(
            self, expression: ast.Expression, part: str, *, kind: str = 'range',
            signed_capture: bool = False) -> int:
        # Iteration-value evaluation must not inherit skipped-expression state or disturb
        # the expression being converted by the enclosing visitor.
        previous = (self.__expression_kind, self.__value, self.__value_type,
                    self.__value_kind, self.__evaluate_constant)
        try:
            self.__expression_kind = ExpressionKind.CONST_ARITHMETIC
            self.__value = self.__value_type = self.__value_kind = None
            self.__evaluate_constant = True
            self.visit(expression)
            if self.__value_type not in (ValueType.INT, ValueType.UINT) or self.__value_kind != ValueKind.LITERAL:
                raise InvalidLoopRangeException(
                    f'For-loop {kind} {part} must be a constant integer')
            if (signed_capture and self.__value_type == ValueType.UINT
                    and self.__value > self.QCX_INT_MAX):
                # Runtime loop captures use checked UINT-to-INT conversion.
                # Expansion must not silently relabel an unrepresentable UINT
                # as an INT iterator just because its value is constant.
                raise InvalidLoopRangeException(
                    f'For-loop {kind} {part} UINT value must fit QCX INT')
            return int(self.__value)
        finally:
            (self.__expression_kind, self.__value, self.__value_type,
             self.__value_kind, self.__evaluate_constant) = previous

    def __loop_range(self, statement: ast.ForInLoop) -> range:
        self.__validate_loop_header(statement)
        bounds = statement.set_declaration
        if not isinstance(bounds, ast.RangeDefinition):
            raise UnsupportedOpenQASMError('for-loop iteration other than a constant range')
        start = self.__constant_loop_integer(bounds.start, 'start', signed_capture=True)
        end = self.__constant_loop_integer(bounds.end, 'end', signed_capture=True)
        step = (1 if bounds.step is None
                else self.__constant_loop_integer(bounds.step, 'step', signed_capture=True))
        if step == 0:
            raise InvalidLoopRangeException('For-loop range step cannot be zero')
        # OpenQASM includes the end when reachable; Python excludes the stop.
        # Keep this lazy so a huge range cannot allocate a huge intermediate list.
        return range(start, end + (1 if step > 0 else -1), step)

    def __loop_values(self, statement: ast.ForInLoop) -> range | tuple[int, ...]:
        self.__validate_loop_header(statement)
        source = statement.set_declaration
        if isinstance(source, ast.DiscreteSet):
            if self.__set_has_runtime_elements(source):
                # This helper evaluates only the constant path; runtime-valued
                # sets are dispatched separately by visit_ForInLoop.
                raise NoConstantExpressionException
            # This is an ordered sequence, not a Python set: repeated values
            # represent distinct iterations and retain their original positions.
            return tuple(self.__constant_loop_integer(value, f'element {index}', kind='set', signed_capture=True)
                         for index, value in enumerate(source.values))
        return self.__loop_range(statement)

    @staticmethod
    def __source_value_type(variable_type: ast.QASMNode) -> ValueType:
        for types, value_type in (
                ((ast.IntType,), ValueType.INT), ((ast.UintType,), ValueType.UINT),
                ((ast.FloatType,), ValueType.FLOAT), ((ast.ComplexType,), ValueType.COMPLEX),
                ((ast.BoolType,), ValueType.BOOL), ((ast.BitType,), ValueType.BIT)):
            if isinstance(variable_type, types):
                return value_type
        raise UnsupportedOpenQASMError('non-scalar loop operand')

    def __loop_expression_info(
            self, expression: ast.Expression, iterators: set[str] | dict[str, ValueType] | None = None
            ) -> tuple[ValueType, bool]:
        # Inspect types and dependency without evaluating values or emitting
        # instructions. Unknown names are errors, not a runtime fallback.
        if isinstance(expression, ast.Identifier):
            return self.__loop_identifier_info(expression, iterators)
        for literal, value_type in ((ast.IntegerLiteral, ValueType.INT),
                                    (ast.FloatLiteral, ValueType.FLOAT),
                                    (ast.ImaginaryLiteral, ValueType.COMPLEX),
                                    (ast.BooleanLiteral, ValueType.BOOL)):
            if isinstance(expression, literal):
                return value_type, False
        if isinstance(expression, ast.Cast):
            return self.__loop_cast_info(expression, iterators)
        if isinstance(expression, ast.UnaryExpression):
            return self.__loop_unary_info(expression, iterators)
        if isinstance(expression, ast.BinaryExpression):
            return self.__loop_binary_info(expression, iterators)
        if isinstance(expression, ast.IndexExpression) and isinstance(expression.collection, ast.Identifier):
            return self.__loop_index_info(expression, iterators)
        raise UnsupportedOpenQASMError(f'loop expression {type(expression).__name__}')

    def __loop_identifier_info(
            self, expression: ast.Identifier, iterators: set[str] | dict[str, ValueType] | None
            ) -> tuple[ValueType, bool]:
        name = expression.name
        if iterators is not None and name in iterators:
            value_type = iterators[name] if isinstance(iterators, dict) else ValueType.INT
            return value_type, value_type == ValueType.BIT
        for scope in reversed(self.__loop_bindings):
            if name in scope:
                binding = scope[name]
                return ((binding.value_type, True) if isinstance(binding, _RuntimeIteratorBinding)
                        else (ValueType.INT, False))
        for constants, value_type in (
                (self.__const_int_variable_name_values_map, ValueType.INT),
                (self.__const_uint_variable_name_values_map, ValueType.UINT),
                (self.__const_float_variable_name_values_map, ValueType.FLOAT),
                (self.__const_complex_variable_name_values_map, ValueType.COMPLEX),
                (self.__const_bool_variable_values_map, ValueType.BOOL)):
            if name in constants:
                return value_type, False
        if name in ('pi', 'tau', 'euler'):
            return ValueType.FLOAT, False
        if name not in self.__classical_source_types:
            raise NoVariableNameException(name)
        variable_type = self.__classical_source_types[name]
        if isinstance(variable_type, ast.BitType) and variable_type.size is not None:
            raise UnsupportedOpenQASMError('whole bit register in loop expression')
        return self.__source_value_type(variable_type), True

    def __loop_cast_info(
            self, expression: ast.Cast, iterators: set[str] | dict[str, ValueType] | None
            ) -> tuple[ValueType, bool]:
        if isinstance(expression.type, ast.UintType):
            self.__unsigned_integer_type(expression.type, iterators)
        operand_type, runtime = self.__loop_expression_info(expression.argument, iterators)
        target_type = self.__source_value_type(expression.type)
        if operand_type == ValueType.COMPLEX and target_type != ValueType.COMPLEX:
            # Constant numeric casts already extract the real component in
            # visit_Cast. Native UINT also supports checked conversion of a
            # runtime complex operand's real component. Other runtime complex
            # conversions and complex-to-Boolean/bit casts remain unsupported here.
            if ((runtime and target_type != ValueType.UINT)
                    or target_type not in (ValueType.INT, ValueType.UINT, ValueType.FLOAT)):
                raise NoImplicitCastException
        if target_type == ValueType.BIT and (
                operand_type not in (ValueType.BIT, ValueType.BOOL)
                or expression.type.size is not None):
            raise UnsupportedOpenQASMError('unsupported bit cast in loop expression')
        return target_type, runtime

    def __loop_unary_info(
            self, expression: ast.UnaryExpression, iterators: set[str] | dict[str, ValueType] | None
            ) -> tuple[ValueType, bool]:
        value_type, runtime = self.__loop_expression_info(expression.expression, iterators)
        if expression.op.name == '!':
            if value_type not in (ValueType.BIT, ValueType.BOOL):
                raise UnsupportedOpenQASMError('non-Boolean logical loop operand')
            return ValueType.BOOL, runtime
        if expression.op.name == '~':
            return self.__bitwise_type(value_type), runtime
        if expression.op.name != '-' or value_type not in (
                ValueType.INT, ValueType.UINT, ValueType.FLOAT, ValueType.COMPLEX):
            raise UnsupportedOpenQASMError('unsupported unary loop expression')
        return value_type, runtime

    def __loop_binary_info(
            self, expression: ast.BinaryExpression, iterators: set[str] | dict[str, ValueType] | None
            ) -> tuple[ValueType, bool]:
        lhs_type, lhs_runtime = self.__loop_expression_info(expression.lhs, iterators)
        rhs_type, rhs_runtime = self.__loop_expression_info(expression.rhs, iterators)
        runtime = lhs_runtime or rhs_runtime
        if expression.op.name in ('&&', '||'):
            if (lhs_type not in (ValueType.BIT, ValueType.BOOL)
                    or rhs_type not in (ValueType.BIT, ValueType.BOOL)):
                raise UnsupportedOpenQASMError('non-Boolean logical loop operand')
            return ValueType.BOOL, runtime
        if expression.op.name in ('==', '!=', '<', '<=', '>', '>='):
            self.__comparison_types(lhs_type, rhs_type, expression.op)
            return ValueType.BOOL, runtime
        if expression.op.name in ('&', '|', '^'):
            return self.__bitwise_type(lhs_type, rhs_type), runtime
        if expression.op.name in ('<<', '>>'):
            self.__validate_shift_types(expression, lhs_type, rhs_type, iterators)
            return lhs_type, runtime
        if expression.op.name not in ('+', '-', '*', '/', '%'):
            raise UnsupportedOpenQASMError(f'binary operator {expression.op.name}')
        result_type = self.__promoted_type(lhs_type, rhs_type)
        if expression.op.name == '%' and result_type not in (ValueType.INT, ValueType.UINT):
            raise UnsupportedOpenQASMError('non-integer remainder in loop expression')
        return result_type, runtime

    def __loop_index_info(
            self, expression: ast.IndexExpression, iterators: set[str] | dict[str, ValueType] | None
            ) -> tuple[ValueType, bool]:
        name = expression.collection.name
        if ((iterators is not None and name in iterators)
                or any(name in scope for scope in self.__loop_bindings)):
            raise UnsupportedOpenQASMError('indexed use of a for-loop iterator')
        variable_type = self.__classical_source_types.get(name)
        if isinstance(variable_type, ast.ArrayType) and isinstance(variable_type.base_type, ast.IntType):
            return ValueType.INT, True
        if not isinstance(variable_type, ast.BitType):
            raise UnsupportedOpenQASMError('indexed non-bit loop operand')
        return ValueType.BIT, True

    def __set_has_runtime_elements(
            self, source: ast.DiscreteSet, iterators: set[str] | dict[str, ValueType] | None = None
            ) -> bool:
        # Inspect every element, including those after the first runtime
        # dependency. This never evaluates arithmetic or emits instructions.
        runtime = False
        for index, expression in enumerate(source.values):
            value_type, dependency = self.__loop_expression_info(expression, iterators)
            if value_type not in (ValueType.INT, ValueType.UINT):
                raise InvalidLoopRangeException(
                    f'For-loop set element {index} must be an integer')
            runtime |= dependency
        return runtime

    def __runtime_range_step(self, statement: ast.ForInLoop) -> int | ast.Expression | None:
        bounds = statement.set_declaration
        if not isinstance(bounds, ast.RangeDefinition):
            return None
        runtime = False
        for expression in (bounds.start, bounds.end):
            value_type, dependency = self.__loop_expression_info(expression)
            if value_type not in (ValueType.INT, ValueType.UINT):
                raise InvalidLoopRangeException('For-loop range bounds must be integers')
            runtime |= dependency
        if bounds.step is not None:
            value_type, step_runtime = self.__loop_expression_info(bounds.step)
            if value_type not in (ValueType.INT, ValueType.UINT):
                raise InvalidLoopRangeException('For-loop range step must be an integer')
            if step_runtime:
                return bounds.step
        if not runtime:
            return None
        return self.__validated_runtime_range_step(bounds)

    def __validated_runtime_range_step(self, bounds: ast.RangeDefinition) -> int:
        step = 1 if bounds.step is None else self.__constant_loop_integer(bounds.step, 'step')
        if step == 0:
            raise InvalidLoopRangeException('For-loop range step cannot be zero')
        if not self.QCX_INT_MIN <= step <= self.QCX_INT_MAX:
            raise InvalidLoopRangeException(
                f'Runtime range step must fit QCX INT '
                f'[{self.QCX_INT_MIN}, {self.QCX_INT_MAX}]')
        return step

    @staticmethod
    def __loop_value_count(values: range | tuple[int, ...]) -> int:
        if isinstance(values, range):
            # len(range) can overflow for very large bounds. Arithmetic counting
            # remains exact without materializing values, so budget checks can
            # report a converter error before expansion begins.
            distance = (values.stop - values.start) * (1 if values.step > 0 else -1)
            stride = abs(values.step)
            return max(0, (distance + stride - 1) // stride)
        return len(values)

    @staticmethod
    def __validate_loop_header(statement: ast.ForInLoop) -> None:
        bit_register_loop = (isinstance(statement.type, ast.BitType)
                             and statement.type.size is None
                             and isinstance(statement.set_declaration, ast.Identifier))
        if not isinstance(statement.type, ast.IntType) and not bit_register_loop:
            raise UnsupportedOpenQASMError('for-loop iteration type other than int')
        bounds = statement.set_declaration
        if not isinstance(bounds, (ast.RangeDefinition, ast.DiscreteSet, ast.Identifier)):
            raise UnsupportedOpenQASMError('for-loop iteration other than a constant range or integer set')
        if isinstance(bounds, ast.RangeDefinition) and (bounds.start is None or bounds.end is None):
            raise InvalidLoopRangeException('For-loop range requires both bounds')

    def __validate_loop_syntax(self, node: ast.QASMNode) -> None:
        # Structural checks never evaluate an expression. In particular, an
        # empty loop must not silently accept an unsupported operator/function,
        # nor execute arithmetic or inspect an out-of-range iteration value.
        if isinstance(node, ast.Expression) and not isinstance(node, (
                ast.Identifier, ast.IntegerLiteral, ast.FloatLiteral,
                ast.ImaginaryLiteral, ast.BooleanLiteral, ast.BitstringLiteral,
                ast.UnaryExpression, ast.BinaryExpression, ast.Cast, ast.IndexExpression)):
            raise UnsupportedOpenQASMError(f'loop expression {type(node).__name__}')
        if isinstance(node, ast.BinaryExpression) and node.op.name not in (
                '+', '-', '*', '/', '%', '&', '|', '^', '<<', '>>',
                '==', '!=', '<', '<=', '>', '>=', '&&', '||'):
            raise UnsupportedOpenQASMError(f'binary operator {node.op.name}')
        if isinstance(node, ast.UnaryExpression) and node.op.name not in ('-', '!', '~'):
            raise UnsupportedOpenQASMError(f'unary operator {node.op.name}')
        if isinstance(node, ast.Cast) and not isinstance(node.type, (
                ast.IntType, ast.UintType, ast.FloatType, ast.ComplexType, ast.BoolType, ast.BitType)):
            raise UnsupportedOpenQASMError(f'loop cast {type(node.type).__name__}')
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

    def __validate_nested_loop_bounds(
            self, statement: ast.ForInLoop, iterators: set[str] | dict[str, ValueType]) -> None:
        # Outer iterator values may not exist for an empty outer loop. Validate
        # names and independent iteration values now; defer value-dependent checks until
        # an actual iteration, rather than fabricating an outer value.
        bounds = statement.set_declaration
        iterator_names = set(iterators)
        if isinstance(bounds, ast.Identifier):
            self.__collection_loop_info(bounds, statement.type, iterators)
            return
        is_set = isinstance(bounds, ast.DiscreteSet)
        expressions = ([(value, f'element {index}') for index, value in enumerate(bounds.values)]
                       if is_set else [(bounds.start, 'start'), (bounds.end, 'end'),
                                       (bounds.step, 'step')])
        constants = (set(self.__const_int_variable_name_values_map)
                     | set(self.__const_uint_variable_name_values_map)
                     | set(self.__const_float_variable_name_values_map)
                     | set(self.__const_complex_variable_name_values_map)
                     | set(self.__const_bool_variable_values_map))
        runtime_range = False
        runtime_step = False
        for expression, part in expressions:
            if expression is None:
                continue
            names = self.__loop_bound_names(expression)
            allowed_runtime = set(self.__classical_source_types)
            for name in names - iterator_names - constants - {'pi', 'tau', 'euler'} - allowed_runtime:
                raise NoVariableNameException(name)
            value_type, runtime = self.__loop_expression_info(expression, iterators)
            if value_type not in (ValueType.INT, ValueType.UINT):
                raise InvalidLoopRangeException(
                    f'For-loop set {part} must be an integer' if is_set
                    else 'For-loop range step must be an integer' if part == 'step'
                    else 'For-loop range bounds must be integers')
            if runtime:
                runtime_range |= not is_set
                runtime_step |= part == 'step'
                continue
            if not names & iterator_names:
                value = self.__constant_loop_integer(
                    expression, part, kind='set' if is_set else 'range', signed_capture=True)
                if part == 'step' and value == 0:
                    raise InvalidLoopRangeException('For-loop range step cannot be zero')
        if (runtime_range and not runtime_step
                and (bounds.step is None or not self.__loop_bound_names(bounds.step) & iterator_names)):
            self.__validated_runtime_range_step(bounds)

    def __loop_iterator_types(
            self, name: str | None = None, value_type: ValueType = ValueType.INT) -> dict[str, ValueType]:
        types = {source: binding.value_type if isinstance(binding, _RuntimeIteratorBinding) else ValueType.INT
                 for scope in self.__loop_bindings for source, binding in scope.items()}
        if name is not None:
            types[name] = value_type
        return types

    def __validate_loop_body(
            self, statements: list[ast.Statement], iterators: set[str] | dict[str, ValueType]) -> None:
        # Validate without executing the body, including zero-iteration loops.
        if isinstance(iterators, set):
            active_types = self.__loop_iterator_types()
            iterators = {name: active_types.get(name, ValueType.INT) for name in iterators}
        supported = (ast.QuantumGate, ast.QuantumPhase,
                     ast.QuantumMeasurementStatement, ast.QuantumReset,
                     ast.QuantumBarrier, ast.ClassicalAssignment,
                     ast.BranchingStatement, ast.ForInLoop, ast.WhileLoop,
                     ast.BreakStatement, ast.ContinueStatement)
        for child in statements:
            if isinstance(child, (ast.ClassicalDeclaration, ast.ConstantDeclaration,
                                  ast.QubitDeclaration)):
                raise UnsupportedOpenQASMError('block-local declaration')
            if not isinstance(child, supported):
                raise UnsupportedOpenQASMError(f'loop body {type(child).__name__}')
            self.__validate_bitwise_types(child, iterators)
            target = (child.lvalue if isinstance(child, ast.ClassicalAssignment)
                      else child.target if isinstance(child, ast.QuantumMeasurementStatement)
                      else None)
            if target is not None and self.__operand_name(target) in iterators:
                raise UnsupportedOpenQASMError('assignment to a for-loop iterator')
            if isinstance(child, ast.ClassicalAssignment) and child.op.name not in (
                    '=', '+=', '-=', '*=', '/=', '%=', '&=', '|=', '^='):
                raise UnsupportedOpenQASMError(f'assignment operator {child.op.name}')
            if isinstance(child, ast.ClassicalAssignment) and child.op.name in ('&=', '|=', '^='):
                self.__validate_bitwise_assignment_types(child, iterators)
            if isinstance(child, ast.QuantumGate):
                if child.modifiers or child.duration is not None:
                    raise UnsupportedOpenQASMError('gate modifiers or duration in a loop')
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
                self.__validate_loop_body(child.block, iterators | {
                    child.identifier.name: self.__source_value_type(child.type)})
            if isinstance(child, ast.WhileLoop):
                self.__validate_loop_syntax(child.while_condition)
                self.__validate_loop_body(child.block, iterators)

    def __validate_bitwise_assignment_types(
            self, statement: ast.ClassicalAssignment, iterators: dict[str, ValueType]) -> None:
        target = statement.lvalue
        if isinstance(target, ast.IndexedIdentifier):
            if (len(target.indices) != 1 or not isinstance(target.indices[0], list)
                    or len(target.indices[0]) != 1
                    or not isinstance(target.indices[0][0], ast.Expression)
                    or isinstance(target.indices[0][0], (ast.RangeDefinition, ast.DiscreteSet))):
                raise UnsupportedOpenQASMError('bitwise assignment requires a scalar target')
            target = ast.IndexExpression(target.name, target.indices[0])
        target_type, _ = self.__loop_expression_info(target, iterators)
        rhs_type, _ = self.__loop_expression_info(statement.rvalue, iterators)
        self.__bitwise_type(target_type, rhs_type)
        self.__validate_bitwise_indices(target, iterators)
        self.__validate_bitwise_indices(statement.rvalue, iterators)

    def __validate_bitwise_indices(
            self, expression: ast.Expression, iterators: dict[str, ValueType]) -> None:
        if isinstance(expression, ast.IndexExpression):
            selectors = expression.index
            if (not isinstance(selectors, list) or len(selectors) != 1
                    or not isinstance(selectors[0], ast.Expression)
                    or isinstance(selectors[0], (ast.RangeDefinition, ast.DiscreteSet))):
                raise UnsupportedOpenQASMError('bitwise operand requires a scalar element')
            index_type, _ = self.__loop_expression_info(selectors[0], iterators)
            if index_type not in (ValueType.INT, ValueType.UINT):
                raise UnsupportedOpenQASMError('bitwise element index must be an integer')
            self.__validate_bitwise_indices(selectors[0], iterators)
            return
        for field in dataclasses.fields(expression):
            value = getattr(expression, field.name)
            if isinstance(value, ast.Expression):
                self.__validate_bitwise_indices(value, iterators)

    def __validate_bitwise_types(
            self, node: ast.QASMNode | list, iterators: dict[str, ValueType]) -> None:
        if isinstance(node, list):
            for element in node:
                if isinstance(element, (ast.QASMNode, list)):
                    self.__validate_bitwise_types(element, iterators)
            return
        # Nested for bodies need their own typed iterator scope and are checked
        # by __validate_loop_body. Never invent values for an empty loop.
        if isinstance(node, ast.ForInLoop):
            self.__validate_bitwise_types(node.set_declaration, iterators)
            return
        if ((isinstance(node, ast.UnaryExpression) and node.op.name == '~')
                or (isinstance(node, ast.BinaryExpression) and node.op.name in ('&', '|', '^', '<<', '>>'))):
            self.__loop_expression_info(node, iterators)
            self.__validate_bitwise_indices(node, iterators)
        for field in dataclasses.fields(node):
            value = getattr(node, field.name)
            if isinstance(value, (ast.QASMNode, list)):
                self.__validate_bitwise_types(value, iterators)

    @contextlib.contextmanager
    def __expanded_loop_scope(self) -> Iterator[None]:
        # Runtime-valued sets still emit one body copy per source element.
        # Account for those copies without inventing a constant iterator value.
        self.__expanded_loop_depth += 1
        try:
            yield
        finally:
            self.__expanded_loop_depth -= 1

    @contextlib.contextmanager
    def __iterator_binding(
            self, name: str, value: int | _RuntimeIteratorBinding) -> Iterator[None]:
        self.__loop_bindings.append({name: value})
        try:
            yield
        finally:
            self.__loop_bindings.pop()

    @contextlib.contextmanager
    def __loop_context(self, break_label: str, continue_label: str) -> Iterator[_LoopContext]:
        frame = _LoopContext(break_label, continue_label)
        self.__loop_contexts.append(frame)
        try:
            yield frame
        finally:
            self.__loop_contexts.pop()

    def __loop_has_control(self, statements: list[ast.Statement]) -> bool:
        # Transfers inside nested loops belong to those loops, not this one.
        for child in statements:
            if isinstance(child, (ast.BreakStatement, ast.ContinueStatement)):
                return True
            if isinstance(child, ast.BranchingStatement) and (
                    self.__loop_has_control(child.if_block)
                    or self.__loop_has_control(child.else_block)):
                return True
        return False

    def visit_ForInLoop(self, statement: ast.ForInLoop) -> None:
        self.__validate_loop_header(statement)
        source = statement.set_declaration
        if isinstance(source, ast.Identifier):
            self.__visit_collection_for(statement)
            return
        if isinstance(source, ast.DiscreteSet) and self.__set_has_runtime_elements(source):
            self.__visit_runtime_set(statement)
            return
        step = self.__runtime_range_step(statement)
        if step is not None:
            self.__visit_runtime_for(statement, step)
            return
        values = self.__loop_values(statement)
        self.__validate_loop_body(statement.block, self.__loop_iterator_types(statement.identifier.name))
        self.__validate_loop_syntax(statement)
        count = self.__loop_value_count(values)
        if count > self.MAX_LOOP_ITERATIONS - self.__loop_iterations:
            raise InvalidLoopRangeException(
                f'For-loop expansion exceeds {self.MAX_LOOP_ITERATIONS} iterations')
        self.__loop_iterations += count
        # Allocate one exit target per expanded loop instance and a distinct
        # continuation target per iteration. Ordinary loops do not emit these
        # unused labels; break/continue lowering uses the innermost frame.
        loop_index = self.__loop_index
        self.__loop_index += 1
        break_label = f'QASM2QCX_LOOP_{loop_index}_END'
        emit_control_labels = count > 0 and self.__loop_has_control(statement.block)
        previous_temporaries = self.__declared_temporary_variables.copy()
        for iteration, value in enumerate(values):
            with self.__iterator_binding(statement.identifier.name, value):
                with self.__loop_context(
                        break_label, f'QASM2QCX_LOOP_{loop_index}_NEXT_{iteration}') as frame:
                    for child in statement.block:
                        self.visit(child)
                    if emit_control_labels and not self.__is_initialization_process:
                        self.__qcx_lines.append(f'@{frame.continue_label}')
                        self.__check_loop_output_limit()
        if emit_control_labels and not self.__is_initialization_process:
            self.__qcx_lines.append(f'@{break_label}')
            # A transfer may skip a temporary's first declaration but a later
            # iteration or statement can still reuse that storage. Move only
            # declarations; calculations must stay behind their original jumps.
            self.__hoist_temporary_declarations(previous_temporaries)
            self.__check_loop_output_limit()

    def __collection_loop_info(
            self, source: ast.Identifier, iteration_type: ast.QASMNode,
            iterators: set[str] | dict[str, ValueType] | None = None) -> tuple[str, int, ValueType]:
        name = source.name
        if ((iterators is not None and name in iterators)
                or any(name in scope for scope in self.__loop_bindings)):
            raise UnsupportedOpenQASMError('iteration over a scalar for-loop iterator')
        variable_type = self.__classical_source_types.get(name)
        if variable_type is None:
            raise NoVariableNameException(name)
        variable_name = self.__capitalize_variable_name(name)
        if isinstance(variable_type, ast.BitType) and variable_type.size is not None:
            if not isinstance(iteration_type, ast.BitType):
                raise UnsupportedOpenQASMError('bit-register iteration requires a bit iterator')
            if not self.__is_initialization_process and variable_name not in self.__sized_bit_variables:
                raise NoVariableNameException(name)
            size = self.__bit_register_source_sizes[name]
            if not 1 <= size <= self.QCX_INT_MAX:
                raise InvalidDeclarationException(f'Bit register {name} loop size must be in [1, {self.QCX_INT_MAX}]')
            return variable_name, size, ValueType.BIT
        if isinstance(variable_type, ast.ArrayType):
            if not isinstance(iteration_type, ast.IntType):
                raise UnsupportedOpenQASMError('integer-array iteration requires an int iterator')
            if not self.__is_initialization_process and variable_name not in self.__integer_array_names:
                raise NoVariableNameException(name)
            return variable_name, self.__integer_array_source_sizes[name], ValueType.INT
        raise UnsupportedOpenQASMError('for-loop identifier other than an integer array or bit register')

    def __visit_collection_for(self, statement: ast.ForInLoop) -> None:
        variable_name, size, value_type = self.__collection_loop_info(statement.set_declaration, statement.type)
        self.__validate_loop_body(statement.block, self.__loop_iterator_types(statement.identifier.name, value_type))
        self.__validate_loop_syntax(statement)
        loop_index = self.__loop_index
        self.__loop_index += 1
        body_label = f'QASM2QCX_LOOP_{loop_index}_BODY'
        next_label = f'QASM2QCX_LOOP_{loop_index}_NEXT'
        end_label = f'QASM2QCX_LOOP_{loop_index}_END'
        if self.__is_initialization_process:
            binding = _RuntimeIteratorBinding(f'QASM2QCX_LOOP_{loop_index}_ITERATOR', value_type)
            with self.__iterator_binding(statement.identifier.name, binding), self.__loop_context(end_label, next_label):
                for child in statement.block:
                    self.visit(child)
            return

        previous_temporaries = self.__declared_temporary_variables.copy()
        iterator = self.__add_new_temporary_variable(ValueType.INT)
        index = self.__add_new_temporary_variable(ValueType.INT)
        try:
            # QCX stores a singleton bit register as a scalar INT, unlike
            # integer arrays, whose scalar accesses retain explicit indexing.
            operand = variable_name if value_type == ValueType.BIT and size == 1 else f'{variable_name}:{index}'
            self.__qcx_lines.extend([f'LET {index} := 0', f'@{body_label}',
                                    f'LET {iterator} := {operand}'])
            # Read the current element on each iteration, not a loop-entry
            # snapshot. Keep its copied value and index reserved across the body.
            with self.__iterator_binding(statement.identifier.name, _RuntimeIteratorBinding(iterator, value_type)), \
                    self.__loop_context(end_label, next_label):
                for child in statement.block:
                    self.visit(child)
            self.__qcx_lines.extend([f'@{next_label}', f'JUMPIF {end_label} {index} == {size - 1}',
                                    f'LET {index} += 1', f'JUMP {body_label}', f'@{end_label}'])
            self.__hoist_temporary_declarations(previous_temporaries)
            if self.__inside_expanded_loop():
                self.__check_loop_output_limit()
        finally:
            self.__release_temporary_variable(index)
            self.__release_temporary_variable(iterator)

    def __visit_runtime_set(self, statement: ast.ForInLoop) -> None:
        self.__validate_loop_body(statement.block, self.__loop_iterator_types(statement.identifier.name))
        self.__validate_loop_syntax(statement)
        count = len(statement.set_declaration.values)
        if count > self.MAX_LOOP_ITERATIONS - self.__loop_iterations:
            raise InvalidLoopRangeException(
                f'For-loop expansion exceeds {self.MAX_LOOP_ITERATIONS} iterations')
        self.__loop_iterations += count
        loop_index = self.__loop_index
        self.__loop_index += 1
        end_label = f'QASM2QCX_LOOP_{loop_index}_END'
        if self.__is_initialization_process:
            binding = _RuntimeIteratorBinding(f'QASM2QCX_LOOP_{loop_index}_ITERATOR')
            with self.__iterator_binding(statement.identifier.name, binding), self.__expanded_loop_scope():
                for index in range(count):
                    with self.__loop_context(end_label, f'QASM2QCX_LOOP_{loop_index}_NEXT_{index}'):
                        for child in statement.block:
                            self.visit(child)
            return

        previous_temporaries = self.__declared_temporary_variables.copy()
        iterator = self.__add_new_temporary_variable(ValueType.INT)
        captures: list[str] = []
        try:
            # Capture every element in source order before the iterator exists
            # or the first body can mutate an element's source variable.
            for index, expression in enumerate(statement.set_declaration.values):
                capture = self.__add_new_temporary_variable(ValueType.INT)
                captures.append(capture)
                value, value_type, value_kind = self.__condition_operand(expression)
                if value_type not in (ValueType.INT, ValueType.UINT):
                    raise InvalidLoopRangeException(
                        f'For-loop set element {index} must be an integer')
                self.__emit_assignment(capture, ':=', ValueType.INT, value, value_type, value_kind)
            with self.__iterator_binding(statement.identifier.name, _RuntimeIteratorBinding(iterator)), \
                    self.__expanded_loop_scope():
                for index, capture in enumerate(captures):
                    self.__qcx_lines.append(f'LET {iterator} := {capture}')
                    with self.__loop_context(end_label, f'QASM2QCX_LOOP_{loop_index}_NEXT_{index}') as frame:
                        for child in statement.block:
                            self.visit(child)
                    self.__qcx_lines.append(f'@{frame.continue_label}')
                    self.__check_loop_output_limit()
            self.__qcx_lines.append(f'@{end_label}')
            self.__hoist_temporary_declarations(previous_temporaries)
            self.__check_loop_output_limit()
        finally:
            for capture in reversed(captures):
                self.__release_temporary_variable(capture)
            self.__release_temporary_variable(iterator)

    def __visit_runtime_for(self, statement: ast.ForInLoop, step: int | ast.Expression) -> None:
        self.__validate_loop_body(statement.block, self.__loop_iterator_types(statement.identifier.name))
        self.__validate_loop_syntax(statement)
        loop_index = self.__loop_index
        self.__loop_index += 1
        body_label = f'QASM2QCX_LOOP_{loop_index}_BODY'
        next_label = f'QASM2QCX_LOOP_{loop_index}_NEXT'
        end_label = f'QASM2QCX_LOOP_{loop_index}_END'
        if self.__is_initialization_process:
            with self.__iterator_binding(statement.identifier.name,
                                         _RuntimeIteratorBinding(f'QASM2QCX_LOOP_{loop_index}_ITERATOR')), \
                    self.__loop_context(end_label, next_label):
                for child in statement.block:
                    self.visit(child)
            return

        previous_temporaries = self.__declared_temporary_variables.copy()
        iterator = self.__add_new_temporary_variable(ValueType.INT)
        stop = self.__add_new_temporary_variable(ValueType.INT)
        captured_step = None if isinstance(step, int) else self.__add_new_temporary_variable(ValueType.INT)
        try:
            # Capture the range in source order before installing the iterator
            # binding. Keep all captures reserved across body conversion.
            captures = [(statement.set_declaration.start, iterator)]
            if captured_step is not None:
                captures.append((step, captured_step))
            captures.append((statement.set_declaration.end, stop))
            for expression, destination in captures:
                value, value_type, value_kind = self.__condition_operand(expression)
                if value_type not in (ValueType.INT, ValueType.UINT):
                    raise InvalidLoopRangeException('For-loop range operands must be integers')
                self.__emit_assignment(destination, ':=', ValueType.INT, value, value_type, value_kind)
            emitted_step = step if captured_step is None else captured_step
            self.__emit_runtime_range_entry(iterator, stop, emitted_step, body_label, end_label,
                                            f'QASM2QCX_LOOP_{loop_index}_NEGATIVE_ENTRY')
            self.__qcx_lines.append(f'@{body_label}')
            with self.__iterator_binding(statement.identifier.name, _RuntimeIteratorBinding(iterator)), \
                    self.__loop_context(end_label, next_label):
                for child in statement.block:
                    self.visit(child)
            self.__qcx_lines.append(f'@{next_label}')
            self.__emit_runtime_range_next(
                iterator, stop, emitted_step, body_label, end_label,
                f'QASM2QCX_LOOP_{loop_index}_CANDIDATE')
            self.__qcx_lines.append(f'@{end_label}')
            self.__hoist_temporary_declarations(previous_temporaries)
            if self.__inside_expanded_loop():
                self.__check_loop_output_limit()
        finally:
            if captured_step is not None:
                self.__release_temporary_variable(captured_step)
            self.__release_temporary_variable(stop)
            self.__release_temporary_variable(iterator)

    def __emit_runtime_range_entry(
            self, iterator: str, stop: str, step: int | str,
            body_label: str, end_label: str, negative_label: str) -> None:
        if isinstance(step, int):
            comparison = '>' if step > 0 else '<'
            self.__qcx_lines.append(f'JUMPIF {end_label} {iterator} {comparison} {stop}')
            return
        # Check before testing for an empty range, but only when execution
        # reaches this loop. The captured step remains immutable in the body.
        self.__qcx_lines.append(f'ASSERT {step} \\= 0')
        self.__qcx_lines.append(f'JUMPIF {negative_label} {step} < 0')
        self.__qcx_lines.append(f'JUMPIF {end_label} {iterator} > {stop}')
        self.__qcx_lines.append(f'JUMP {body_label}')
        self.__qcx_lines.append(f'@{negative_label}')
        self.__qcx_lines.append(f'JUMPIF {end_label} {iterator} < {stop}')

    def __emit_runtime_range_next(
            self, iterator: str, stop: str, step: int | str,
            body_label: str, end_label: str, candidate_label: str) -> None:
        if isinstance(step, str):
            negative_label = candidate_label + '_NEGATIVE_NEXT'
            self.__qcx_lines.append(f'JUMPIF {negative_label} {step} < 0')
            self.__emit_strided_range_next(iterator, stop, step, body_label, end_label,
                                           candidate_label + '_POSITIVE', positive=True)
            self.__qcx_lines.append(f'@{negative_label}')
            self.__emit_strided_range_next(iterator, stop, step, body_label, end_label,
                                           candidate_label + '_NEGATIVE', positive=False)
            return
        if step in (1, -1):
            # Preserve unit-step output and stop before advancing at an endpoint.
            self.__qcx_lines.append(f'JUMPIF {end_label} {iterator} == {stop}')
            self.__qcx_lines.append(f'LET {iterator} {"+=" if step == 1 else "-="} 1')
            self.__qcx_lines.append(f'JUMP {body_label}')
            return

        self.__emit_strided_range_next(iterator, stop, step, body_label, end_label,
                                       candidate_label, positive=step > 0)

    def __emit_strided_range_next(
            self, iterator: str, stop: str, step: int | str,
            body_label: str, end_label: str, candidate_label: str, *, positive: bool) -> None:

        scratch = self.__add_new_temporary_variable(ValueType.INT)
        try:
            # The entry check and read-only iterator guarantee iterator <= stop
            # for positive steps, or iterator >= stop for negative steps.
            # stop - step is safe for stop >= 0 with a positive step, and for
            # stop < 0 with a negative step (including QCX_INT_MIN).
            sign_comparison = '<' if positive else '>='
            exit_comparison = '>' if positive else '<'
            self.__qcx_lines.append(f'JUMPIF {candidate_label} {stop} {sign_comparison} 0')
            self.__qcx_lines.append(f'LET {scratch} := {stop}')
            self.__qcx_lines.append(f'LET {scratch} -= {step}')
            self.__qcx_lines.append(f'JUMPIF {end_label} {iterator} {exit_comparison} {scratch}')
            self.__qcx_lines.append(f'LET {iterator} += {step}')
            self.__qcx_lines.append(f'JUMP {body_label}')
            self.__qcx_lines.append(f'@{candidate_label}')
            # Otherwise iterator is negative for a positive step, or nonnegative
            # for a negative step, so iterator + step is representable. Compare
            # that candidate with stop before assigning it to the iterator.
            self.__qcx_lines.append(f'LET {scratch} := {iterator}')
            self.__qcx_lines.append(f'LET {scratch} += {step}')
            self.__qcx_lines.append(f'JUMPIF {end_label} {scratch} {exit_comparison} {stop}')
            self.__qcx_lines.append(f'LET {iterator} := {scratch}')
            self.__qcx_lines.append(f'JUMP {body_label}')
        finally:
            self.__release_temporary_variable(scratch)

    def visit_WhileLoop(self, statement: ast.WhileLoop) -> None:
        self.__validate_loop_body(statement.block,
                                  {name for scope in self.__loop_bindings for name in scope})
        self.__validate_loop_syntax(statement)
        loop_index = self.__loop_index
        self.__loop_index += 1
        condition_label = f'QASM2QCX_LOOP_{loop_index}_CONDITION'
        body_label = f'QASM2QCX_LOOP_{loop_index}_BODY'
        end_label = f'QASM2QCX_LOOP_{loop_index}_END'
        with self.__loop_context(end_label, condition_label):
            if self.__is_initialization_process:
                # Discover and validate the body once, without executing or
                # unrolling this runtime loop (even for a literal condition).
                for child in statement.block:
                    self.visit(child)
                return

            previous_temporaries = self.__declared_temporary_variables.copy()
            self.__qcx_lines.append(f'@{condition_label}')
            self.__emit_condition(statement.while_condition, body_label, end_label)
            self.__qcx_lines.append(f'@{body_label}')
            for child in statement.block:
                self.visit(child)
            self.__qcx_lines.append(f'JUMP {condition_label}')
            self.__qcx_lines.append(f'@{end_label}')
            # The body can execute zero times, or a transfer can skip a
            # temporary's first use. Only its storage declaration moves.
            self.__hoist_temporary_declarations(previous_temporaries)
            if self.__inside_expanded_loop():
                self.__check_loop_output_limit()

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
        if not self.__loop_contexts:
            raise UnsupportedOpenQASMError('break outside a supported loop')
        if not self.__is_initialization_process:
            self.__qcx_lines.append(f'JUMP {self.__loop_contexts[-1].break_label}')

    def visit_ContinueStatement(self, statement: ast.ContinueStatement) -> None:
        if not self.__loop_contexts:
            raise UnsupportedOpenQASMError('continue outside a supported loop')
        if not self.__is_initialization_process:
            self.__qcx_lines.append(f'JUMP {self.__loop_contexts[-1].continue_label}')

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
                if self.__value_type not in (ValueType.BOOL, ValueType.BIT, ValueType.INT, ValueType.UINT, ValueType.FLOAT):
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
        uint_type = None
        if isinstance(expression.type, ast.UintType):
            target_type = ValueType.UINT
            cast_name = 'UINT'
            uint_type = self.__unsigned_integer_type(expression.type)
            self.__uint_expression_types[id(expression)] = (expression, uint_type)
        elif isinstance(expression.type, ast.IntType):
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
        if (target_type == ValueType.INT and self.__value_type == ValueType.UINT
                and expression.type.size is not None):
            raise UnsupportedOpenQASMError('fixed-width signed cast from uint')
        if self.__value_kind == ValueKind.LITERAL:
            if not self.__evaluate_constant:
                self.__value, self.__value_type = 0, target_type
                return
            if target_type == ValueType.INT:
                if self.__value_type == ValueType.UINT and int(self.__value) > self.QCX_INT_MAX:
                    if self.__expression_kind == ExpressionKind.CONST_ARITHMETIC:
                        raise UnsupportedOpenQASMError('UINT to INT conversion outside QCX INT range')
                    temporary = self.__add_new_temporary_variable(ValueType.INT)
                    self.__qcx_lines.append(f'LET {temporary} := :INT::UINT:{self.__value}')
                    self.__value, self.__value_type, self.__value_kind = temporary, ValueType.INT, ValueKind.TEMPORARY
                    return
                self.__value = (int(self.__value) if self.__value_type in (ValueType.INT, ValueType.UINT, ValueType.BIT)
                                else int(complex(self.__value).real))
            elif target_type == ValueType.UINT:
                if self.__value_type in (ValueType.INT, ValueType.UINT, ValueType.BIT):
                    self.__value = uint_type.normalize(int(self.__value))
                else:
                    real = complex(self.__value).real
                    if not math.isfinite(real) or not 0 <= math.trunc(real) <= self.QCX_UINT_MAX:
                        if self.__expression_kind == ExpressionKind.CONST_ARITHMETIC:
                            raise UnsupportedOpenQASMError('floating-point to UINT conversion outside QCX UINT range')
                        temporary = self.__add_new_temporary_variable(ValueType.UINT, uint_type)
                        self.__qcx_lines.append(f'LET {temporary} := :UINT::REAL:{real}')
                        self.__normalize_uint_storage(temporary, uint_type)
                        self.__value, self.__value_type, self.__value_kind = temporary, ValueType.UINT, ValueKind.TEMPORARY
                        return
                    self.__value = uint_type.normalize(math.trunc(real))
            elif target_type == ValueType.FLOAT:
                self.__value = float(complex(self.__value).real)
            else:
                self.__value = complex(self.__value)
            self.__value_type = target_type
            return

        if self.__value_type == target_type and target_type != ValueType.UINT:
            return

        original_value = self.__value
        original_kind = self.__value_kind
        temporary = self.__add_new_temporary_variable(target_type, uint_type)
        self.__qcx_lines.append(
            f'LET {temporary} := :{cast_name}:{original_value}')
        if uint_type is not None:
            self.__normalize_uint_storage(temporary, uint_type)
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
                values_map = (self.__const_uint_variable_name_values_map
                              if isinstance(variable_type, ast.UintType)
                              else self.__const_int_variable_name_values_map)
                if variable_name in values_map:
                    raise WrongConstantVariableException(variable_name)

                values_map[variable_name] = [0] * num_elements

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
                    and not (self.__value_type in (ValueType.INT, ValueType.UINT) and self.__value in (0, 1))):
                raise NoImplicitCastException
            self.__const_bool_variable_values_map[variable_name] = int(self.__value)
            return
        if self.__value_type == ValueType.BOOL:
            self.__value_type = ValueType.INT
        match variable_type:
            case ast.IntType() | ast.UintType():
                values_map = (self.__const_uint_variable_name_values_map
                              if isinstance(variable_type, ast.UintType)
                              else self.__const_int_variable_name_values_map)
                if variable_name not in values_map:
                    raise WrongConstantVariableException(variable_name)

                if self.__value_type == ValueType.FLOAT or self.__value_type == ValueType.COMPLEX:
                    raise NoImplicitCastException

                value = int(self.__value)
                if isinstance(variable_type, ast.UintType):
                    value = self.__const_uint_types[variable_name].normalize(value)
                elif self.__value_type == ValueType.UINT and value > self.QCX_INT_MAX:
                    raise UnsupportedOpenQASMError('UINT to INT conversion outside QCX INT range')
                values_map[variable_name][0] = value

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
        self.__constant_source_types[variable_name] = variable_type

        if isinstance(variable_type, ast.ArrayType):
            raise UnsupportedOpenQASMError('constant array')
        if not isinstance(variable_type, (
                ast.IntType, ast.UintType, ast.FloatType, ast.ComplexType,
                ast.BoolType)):
            raise UnsupportedOpenQASMError(
                f'constant type {type(variable_type).__name__}')
        if isinstance(variable_type, ast.UintType):
            self.__const_uint_types[variable_name] = self.__unsigned_integer_type(variable_type)
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
                sizes_map = (self.__uint_variable_name_size_map
                             if isinstance(variable_type, ast.UintType)
                             else self.__int_variable_name_size_map)
                if variable_name in sizes_map:
                    raise WrongClassicalDeclarationException(variable_name)

                sizes_map[variable_name] = num_elements
                storage_type = 'UINT' if isinstance(variable_type, ast.UintType) else 'INT'
                self.__qcx_lines.append(f'VAR {variable_name} {storage_type}' + (f' {num_elements}' if num_elements > 1 else ''))

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

    def __unsigned_integer_type(
            self, variable_type: ast.UintType,
            iterators: set[str] | dict[str, ValueType] | None = None) -> _UnsignedIntegerType:
        if variable_type.size is None:
            return _UnsignedIntegerType(self.QCX_UINT_WIDTH, False)
        names = self.__loop_bound_names(variable_type.size)
        iterator_names = set(iterators or ()) | {name for scope in self.__loop_bindings for name in scope}
        if names & iterator_names:
            raise InvalidDeclarationException('UINT width must be a constant integer, not a loop iterator')
        value_type, runtime = self.__loop_expression_info(variable_type.size)
        if value_type not in (ValueType.INT, ValueType.UINT) or runtime:
            raise InvalidDeclarationException('UINT width must be a constant integer')
        previous = (self.__expression_kind, self.__value, self.__value_type,
                    self.__value_kind, self.__evaluate_constant)
        try:
            self.__expression_kind = ExpressionKind.CONST_ARITHMETIC
            self.__value = self.__value_type = self.__value_kind = None
            self.__evaluate_constant = True
            self.visit(variable_type.size)
            if self.__value_kind != ValueKind.LITERAL or self.__value_type not in (ValueType.INT, ValueType.UINT):
                raise InvalidDeclarationException('UINT width must be a constant integer')
            width = int(self.__value)
            if width <= 0:
                raise InvalidDeclarationException('UINT width must be positive')
            if width > self.QCX_UINT_WIDTH:
                raise UnsupportedOpenQASMError(
                    f'UINT width {width} exceeds native QCX UINT width {self.QCX_UINT_WIDTH}')
            return _UnsignedIntegerType(width, True)
        finally:
            (self.__expression_kind, self.__value, self.__value_type,
             self.__value_kind, self.__evaluate_constant) = previous

    def __normalize_uint_storage(self, name: str, uint_type: _UnsignedIntegerType) -> None:
        # Native unsigned arithmetic already wraps at QCX_UINT_WIDTH. Narrow
        # values require a mask after each operation, not only at assignment.
        if uint_type.width < self.QCX_UINT_WIDTH:
            self.__qcx_lines.append(f'LET {name} &= {uint_type.mask}')

    def __bit_type_size(self, variable_type: ast.BitType, variable_name: str) -> int:
        if variable_type.size is None:
            return 1
        previous = (self.__expression_kind, self.__value, self.__value_type,
                    self.__value_kind, self.__evaluate_constant)
        try:
            self.__expression_kind = ExpressionKind.CONST_ARITHMETIC
            self.__evaluate_constant = True
            self.visit(variable_type.size)
            if (self.__value_kind != ValueKind.LITERAL
                    or self.__value_type not in (ValueType.INT, ValueType.UINT)):
                raise InvalidDeclarationException(
                    f'Bit variable {variable_name} must have a constant integer size')
            if self.__value <= 0:
                raise InvalidDeclarationException(
                    f'Bit variable {variable_name} must have a positive size')
            return int(self.__value)
        finally:
            (self.__expression_kind, self.__value, self.__value_type,
             self.__value_kind, self.__evaluate_constant) = previous

    def __emit_bit_expression_assignment(
            self, target: ast.Identifier | ast.IndexedIdentifier,
            expression: ast.Expression) -> None:
        with self.__bit_target(target) as (target_names, target_is_register):
            self.__emit_bit_value_assignment(target_names, target_is_register, expression)

    def __emit_bit_value_assignment(
            self, target_names: list[str], target_is_register: bool,
            expression: ast.Expression) -> None:
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
            source_size = self.__bit_variable_name_size_map.get(source_name, 1)
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
            runtime_read = self.__runtime_bit_read(expression, source_name, source_size)
            if runtime_read is not None:
                value, value_type, value_kind = runtime_read
                if target_is_register:
                    self.__release_temporary_variable(str(value))
                    raise InvalidBitOperandException(
                        'Bit assignment operands must both be scalars or both be registers')
                self.__emit_assignment(target_names[0], ':=', ValueType.BIT,
                                       value, value_type, value_kind)
                return
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

    def __runtime_bit_read(
            self, expression: ast.IndexExpression, variable_name: str, size: int
            ) -> tuple[str, ValueType, ValueKind] | None:
        access = self.__runtime_bit_access(expression.index, variable_name, size)
        if access is None:
            return None
        operand, index = access
        try:
            result = self.__add_new_temporary_variable(ValueType.INT)
            self.__qcx_lines.append(f'LET {result} := {operand}')
            return result, ValueType.BIT, ValueKind.TEMPORARY
        finally:
            self.__release_temporary_variable(index)

    def __runtime_bit_access(
            self, selectors: list[ast.Expression] | ast.DiscreteSet,
            variable_name: str, size: int
            ) -> tuple[str, str] | None:
        # Keep static selections (including slices) on their existing path.
        if (not isinstance(selectors, list) or len(selectors) != 1
                or not isinstance(selectors[0], ast.Expression)
                or isinstance(selectors[0], (ast.RangeDefinition, ast.DiscreteSet))):
            return None
        value_type, runtime = self.__loop_expression_info(selectors[0])
        if not runtime:
            return None
        if value_type not in (ValueType.INT, ValueType.UINT):
            raise InvalidBitOperandException('Bit register index must be an integer')
        if variable_name not in self.__sized_bit_variables:
            raise InvalidBitOperandException('Runtime indexing requires a bit register')
        if not 1 <= size <= self.QCX_INT_MAX:
            raise InvalidBitOperandException(
                f'Runtime-indexed bit register size must be in [1, {self.QCX_INT_MAX}]')
        index = self.__capture_array_index(selectors[0], size)
        # QCX stores bit[1] as a scalar, but its index still needs checking.
        return (variable_name if size == 1 else f'{variable_name}:{index}'), index

    def __integer_array_access(
            self, expression: ast.IndexExpression | ast.IndexedIdentifier) -> tuple[str, str | None]:
        if isinstance(expression, ast.IndexExpression):
            if not isinstance(expression.collection, ast.Identifier):
                raise UnsupportedOpenQASMError('multidimensional integer array indexing')
            name, selectors = expression.collection.name, expression.index
        else:
            if len(expression.indices) != 1:
                raise UnsupportedOpenQASMError('multidimensional integer array indexing')
            name, selectors = expression.name.name, expression.indices[0]
        variable_name = self.__capitalize_variable_name(name)
        self.__type_of(variable_name)  # Reject an active iterator shadowing the array.
        if variable_name not in self.__integer_array_names:
            raise UnsupportedOpenQASMError('indexed non-array integer operand')
        if not isinstance(selectors, list) or len(selectors) != 1 or not isinstance(selectors[0], ast.Expression):
            raise UnsupportedOpenQASMError('integer array slice or multidimensional selection')
        if isinstance(selectors[0], (ast.RangeDefinition, ast.DiscreteSet)):
            raise UnsupportedOpenQASMError('integer array slice or multidimensional selection')
        value_type, runtime = self.__loop_expression_info(selectors[0])
        if value_type not in (ValueType.INT, ValueType.UINT):
            raise InvalidArrayOperandException(f'Integer array {name} index must be an integer')
        size = self.__int_variable_name_size_map[variable_name]
        if runtime:
            index = self.__capture_array_index(selectors[0], size)
            return f'{variable_name}:{index}', index
        index = self.__constant_loop_integer(selectors[0], 'array index')
        if not -size <= index < size:
            raise InvalidArrayOperandException(f'Integer array {name} index {index} is outside [-{size}, {size - 1}]')
        return f'{variable_name}:{index if index >= 0 else index + size}', None

    def __capture_array_index(self, expression: ast.Expression, size: int) -> str:
        value, value_type, value_kind = self.__condition_operand(expression)
        index = self.__add_new_temporary_variable(ValueType.INT)
        try:
            self.__emit_assignment(index, ':=', ValueType.INT, value, value_type, value_kind)
            label = f'QASM2QCX_ARRAY_INDEX_{self.__array_index}_NONNEGATIVE'
            self.__array_index += 1
            # Check the original signed index before adding the size. This
            # keeps normalization representable, even at native INT endpoints.
            self.__qcx_lines.extend([
                f'ASSERT {index} >= {-size}', f'ASSERT {index} < {size}',
                f'JUMPIF {label} {index} >= 0', f'LET {index} += {size}', f'@{label}',
            ])
            return index
        except Exception:
            self.__release_temporary_variable(index)
            raise

    def __integer_array_size(self, variable_type: ast.ArrayType, variable_name: str) -> int:
        if len(variable_type.dimensions) != 1:
            raise UnsupportedOpenQASMError('multidimensional classical array')
        if not isinstance(variable_type.base_type, ast.IntType):
            raise UnsupportedOpenQASMError('non-int classical array')
        previous = (self.__expression_kind, self.__value, self.__value_type,
                    self.__value_kind, self.__evaluate_constant)
        try:
            value_type, runtime = self.__loop_expression_info(variable_type.dimensions[0])
            if value_type not in (ValueType.INT, ValueType.UINT) or runtime:
                raise InvalidDeclarationException(
                    f'Integer array {variable_name} must have a constant integer size')
            self.__expression_kind = ExpressionKind.CONST_ARITHMETIC
            self.__value = self.__value_type = self.__value_kind = None
            self.__evaluate_constant = True
            self.visit(variable_type.dimensions[0])
            if self.__value_kind != ValueKind.LITERAL or self.__value_type not in (ValueType.INT, ValueType.UINT):
                raise InvalidDeclarationException(
                    f'Integer array {variable_name} must have a constant integer size')
            size = int(self.__value)
            if not 1 <= size <= self.QCX_INT_MAX:
                raise InvalidDeclarationException(
                    f'Integer array {variable_name} size must be in [1, {self.QCX_INT_MAX}]')
            return size
        finally:
            (self.__expression_kind, self.__value, self.__value_type,
             self.__value_kind, self.__evaluate_constant) = previous

    def __declare_integer_array(self, statement: ast.ClassicalDeclaration, variable_name: str) -> None:
        size = self.__integer_array_source_sizes[statement.identifier.name]
        initializer = statement.init_expression
        if initializer is not None:
            if not isinstance(initializer, ast.ArrayLiteral):
                raise UnsupportedOpenQASMError('integer array initializer other than an array literal')
            if len(initializer.values) != size:
                raise InvalidDeclarationException(
                    f'Integer array {statement.identifier.name} initializer has '
                    f'{len(initializer.values)} elements; expected {size}')
            if any(isinstance(value, ast.ArrayLiteral) for value in initializer.values):
                raise UnsupportedOpenQASMError('nested integer array initializer')
        self.__declare_classical_variable(statement.type.base_type, variable_name, size)
        self.__integer_array_names.add(variable_name)
        if initializer is None:
            return
        for index, expression in enumerate(initializer.values):
            value, value_type, value_kind = self.__condition_operand(expression)
            if value_type not in (ValueType.INT, ValueType.UINT):
                if value_kind == ValueKind.TEMPORARY:
                    self.__release_temporary_variable(str(value))
                raise InvalidDeclarationException(
                    f'Integer array {statement.identifier.name} initializer element {index} must be an integer')
            self.__emit_assignment(f'{variable_name}:{index}', ':=', ValueType.INT,
                                   value, value_type, value_kind)

    def visit_ClassicalDeclaration(self, statement: ast.ClassicalDeclaration) -> None:
        if self.__is_initialization_process:
            self.__register_source_identifier(statement.identifier.name)
            if isinstance(statement.type, ast.UintType):
                self.__uint_variable_types[self.__capitalize_variable_name(statement.identifier.name)] = (
                    self.__unsigned_integer_type(statement.type))
            self.__classical_source_types[statement.identifier.name] = statement.type
            if isinstance(statement.type, ast.ArrayType):
                # Resolve dimensions in declaration scope, before loop iterators
                # can shadow constants referenced by the type expression.
                self.__integer_array_source_sizes[statement.identifier.name] = self.__integer_array_size(
                    statement.type, statement.identifier.name)
            if isinstance(statement.type, ast.BitType) and statement.type.size is not None:
                # Keep register identity and size in declaration scope, just
                # as for integer arrays; loop iterators may shadow size constants.
                self.__bit_register_source_sizes[statement.identifier.name] = self.__bit_type_size(
                    statement.type, statement.identifier.name)
            self.__reserved_variable_names.add(
                self.__capitalize_variable_name(statement.identifier.name))
            return

        variable_type = statement.type
        variable_name = self.__capitalize_variable_name(statement.identifier.name)

        if isinstance(variable_type, ast.ArrayType):
            self.__declare_integer_array(statement, variable_name)
            return
        if not isinstance(
                variable_type,
                (ast.IntType, ast.UintType, ast.FloatType, ast.BitType,
                 ast.ComplexType, ast.BoolType)):
            raise UnsupportedOpenQASMError(
                f'classical type {type(variable_type).__name__}')
        num_elements = (
            self.__bit_register_source_sizes.get(statement.identifier.name, 1)
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
        if variable_name in self.__integer_array_names:
            if not isinstance(statement.lvalue, ast.IndexedIdentifier):
                raise UnsupportedOpenQASMError('whole integer array assignment')
            operand, index = self.__integer_array_access(statement.lvalue)
            try:
                self.__emit_classical_value_assignment(
                    statement, operand, variable_type, runtime_array=index is not None)
            finally:
                if index is not None:
                    self.__release_temporary_variable(index)
            return

        if variable_type == ValueType.BIT:
            if statement.op.name in ('&=', '|=', '^='):
                with self.__bit_target(statement.lvalue) as (targets, is_register):
                    if is_register:
                        raise UnsupportedOpenQASMError('bitwise assignment requires a scalar target')
                    self.__emit_bitwise_assignment(targets[0], statement.op.name,
                                                   ValueType.BIT, statement.rvalue)
                return
            if statement.op != ast.AssignmentOperator['=']:
                raise UnsupportedOpenQASMError(
                    f'bit assignment operator {statement.op.name}')
            self.__emit_bit_expression_assignment(
                statement.lvalue, statement.rvalue)
            return

        if isinstance(statement.lvalue, ast.IndexedIdentifier):
            raise UnsupportedOpenQASMError('indexed classical assignment')

        self.__emit_classical_value_assignment(statement, variable_name, variable_type)

    def __emit_classical_value_assignment(
            self, statement: ast.ClassicalAssignment, variable_name: str,
            variable_type: ValueType, *, runtime_array: bool = False) -> None:
        if (variable_type == ValueType.INT and statement.op.name != '='
                and not self.__native_signed_operand(statement.lvalue)):
            rhs_type, _ = self.__loop_expression_info(statement.rvalue)
            if rhs_type == ValueType.UINT:
                raise UnsupportedOpenQASMError('mixed signed/unsigned assignment with a non-native signed width')
        if variable_type == ValueType.UINT and statement.op.name != '=':
            binary_name = statement.op.name[:-1]
            if binary_name not in ('+', '-', '*', '/', '%', '&', '|', '^'):
                raise UnsupportedOpenQASMError(f'assignment operator {statement.op.name}')
            expression = ast.BinaryExpression(ast.BinaryOperator[binary_name], statement.lvalue, statement.rvalue)
            value, value_type, value_kind = self.__condition_operand(expression)
            self.__emit_assignment(variable_name, ':=', variable_type, value, value_type, value_kind)
            return
        if statement.op.name in ('&=', '|=', '^='):
            self.__emit_bitwise_assignment(variable_name, statement.op.name,
                                           variable_type, statement.rvalue)
            return
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
            if runtime_array:
                self.__emit_runtime_array_remainder_assignment(variable_name, statement.rvalue)
                return
            # Reuse expression lowering and assign only its completed result;
            # the destination may also occur anywhere in the RHS expression.
            operator = ':='
            lhs = (ast.IndexExpression(statement.lvalue.name, statement.lvalue.indices[0])
                   if isinstance(statement.lvalue, ast.IndexedIdentifier) else statement.lvalue)
            expression = ast.BinaryExpression(ast.BinaryOperator['%'], lhs, statement.rvalue)
        else:
            raise UnsupportedOpenQASMError(
                f'assignment operator {statement.op.name}')

        previous_expression_kind = self.__expression_kind
        try:
            self.__expression_kind = ExpressionKind.ARITHMETIC
            self.visit(expression)
        finally:
            self.__expression_kind = previous_expression_kind
        if (self.__value is None or self.__value_type is None
                or self.__value_kind is None):
            raise UninitializedValueException
        self.__emit_assignment(
            variable_name, operator, variable_type, self.__value,
            self.__value_type, self.__value_kind, self.__uint_expression_type(expression))

    def __emit_bitwise_assignment(
            self, target: str, operator: str, target_type: ValueType,
            expression: ast.Expression) -> None:
        self.__bitwise_type(target_type)
        previous_expression_kind = self.__expression_kind
        operand = None
        try:
            self.__expression_kind = ExpressionKind.ARITHMETIC
            operand = self.__bitwise_operand(expression)
            value, value_type, value_kind = operand
            self.__bitwise_type(target_type, value_type)
            if target_type == value_type == ValueType.INT and value_kind == ValueKind.LITERAL:
                if not self.QCX_INT_MIN <= int(value) <= self.QCX_INT_MAX:
                    raise UnsupportedOpenQASMError('bitwise integer operand outside QCX INT range')
            if target_type == ValueType.INT and value_type == ValueType.UINT:
                uint_type = self.__uint_expression_type(expression)
                if uint_type is None:
                    raise UninitializedValueException
                if uint_type.width == self.QCX_UINT_WIDTH:
                    temporary = self.__add_new_temporary_variable(ValueType.UINT)
                    self.__qcx_lines.extend([f'LET {temporary} := :UINT:{target}',
                                            f'LET {temporary} {operator} {value}'])
                    self.__emit_assignment(target, ':=', ValueType.INT,
                                           temporary, ValueType.UINT, ValueKind.TEMPORARY)
                    return
                value, _ = self.__converted_operand(value, value_type, value_kind, ValueType.INT)
            # Evaluate the complete RHS before writing. The caller keeps any
            # captured destination index reserved throughout this operation.
            self.__qcx_lines.append(f'LET {target} {operator} {value}')
        finally:
            self.__expression_kind = previous_expression_kind
            if operand is not None and operand[2] == ValueKind.TEMPORARY:
                self.__release_temporary_variable(str(operand[0]))

    def __emit_runtime_array_remainder_assignment(
            self, target: str, expression: ast.Expression) -> None:
        # The destination index is already captured and reserved. Rebuilding
        # its AST would evaluate it twice, so read from that exact operand.
        value, value_type, value_kind = self.__condition_operand(expression)
        try:
            if value_type not in (ValueType.INT, ValueType.UINT):
                raise UnsupportedOpenQASMError('integer remainder requires int or uint operands')
            uint_type = self.__uint_expression_type(expression)
            result_type = ValueType.INT
            lhs = target
            if uint_type is not None and uint_type.width == self.QCX_UINT_WIDTH:
                result_type = ValueType.UINT
                lhs = f':UINT:{target}'
                rhs = str(value)
            else:
                rhs, _ = self.__converted_operand(value, value_type, value_kind, ValueType.INT)
                uint_type = None
            result = self.__emit_integer_remainder(lhs, rhs, result_type, uint_type)
            self.__emit_assignment(target, ':=', ValueType.INT,
                                   result, result_type, ValueKind.TEMPORARY)
        finally:
            if value_kind == ValueKind.TEMPORARY:
                self.__release_temporary_variable(str(value))

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
