import sys
import itertools
import math
import cmath
import enum
import dataclasses
import contextlib
import ctypes
from fractions import Fraction
from collections.abc import Iterator

import openqasm3.parser
import openqasm3.ast as ast
import openqasm3.visitor as visitor

ValueType = enum.Enum(
    'ValueType', [('INT', 1), ('FLOAT', 2), ('BIT', 3), ('COMPLEX', 4), ('BOOL', 5), ('UINT', 6),
                  ('BIT_REGISTER', 7), ('ANGLE', 8)])  # BIT_REGISTER is a signature marker.
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

class InvalidPowerException(QASM2QCXError):
    """Negative integer exponent or overflowing signed integer power."""

class InvalidNegationException(QASM2QCXError):
    """Native INT negation has an unrepresentable operand or result."""

class InvalidFunctionCallException(QASM2QCXError):
    """A known built-in function has invalid arity or operand types."""

class InvalidConstantFunctionException(QASM2QCXError):
    """A constant built-in call has an invalid value or overflowing result."""

class WrongConstantVariableException(QASM2QCXError):
    def __str__(self):
        return 'Wrong constant variable'


@dataclasses.dataclass(frozen=True)
class _BuiltinFunctionSignature:
    operand_types: tuple[ValueType, ...]
    result_type: ValueType
    qcx_function: str | None = None
    component: str | None = None
    integer_remainder: bool = False
    population_count: bool = False
    rotation: bool = False


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
class _AngleType:
    width: int
    is_sized: bool

    def __post_init__(self) -> None:
        if self.width <= 0:
            raise ValueError('angle width must be positive')

    @property
    def mask(self) -> int:
        return (1 << self.width) - 1


@dataclasses.dataclass(frozen=True)
class _ConstantAngle:
    angle_type: _AngleType
    value: int

    def __post_init__(self) -> None:
        if not 0 <= self.value <= self.angle_type.mask:
            raise ValueError('constant angle bit pattern must fit its width')


@dataclasses.dataclass(frozen=True)
class _AngleMultiplicativeInfo:
    lhs: _AngleType | _UnsignedIntegerType
    rhs: _AngleType | _UnsignedIntegerType
    result_type: ValueType
    result: _AngleType | _UnsignedIntegerType
    runtime: bool


@dataclasses.dataclass(frozen=True)
class _ConstantBitRegister:
    width: int
    value: int

    def __post_init__(self) -> None:
        if self.width <= 0:
            raise ValueError('constant bit-register width must be positive')
        if self.value < 0 or self.value.bit_length() > self.width:
            raise ValueError('constant bit-register value must fit its width')


@dataclasses.dataclass(frozen=True)
class _StaticBitSelection:
    source_name: str
    indices: tuple[int, ...]
    is_register: bool

    @property
    def width(self) -> int:
        return len(self.indices)


@dataclasses.dataclass(frozen=True)
class _RuntimeIteratorBinding:
    # Unlike an unrolled iterator's integer value, this names live QCX storage.
    storage: str
    value_type: ValueType = ValueType.INT
    uint_type: _UnsignedIntegerType | None = None

    def __post_init__(self) -> None:
        if (self.value_type == ValueType.UINT) != (self.uint_type is not None):
            raise ValueError('UINT iterator bindings require width metadata; other types must not have it')


@dataclasses.dataclass(frozen=True)
class _RuntimeGateExponent:
    expression: ast.Expression
    value_type: ValueType


@dataclasses.dataclass(frozen=True)
class _CapturedGateExponent:
    storage: str
    value_type: ValueType


@dataclasses.dataclass(frozen=True)
class _RuntimeGatePower:
    magnitudes: tuple[str, ...]
    # A sign carrier (0 or -1), not a product of signed exponents.
    signed_exponent: str | None
    constant_factors: tuple[int, ...] = ()


@dataclasses.dataclass(frozen=True)
class _GateModifierInfo:
    # Integer powers and inversion commute with adding controls. Keep control
    # groups in source order, since each group prepends its own operands.
    power: int
    controls: tuple[tuple[int, int], ...]
    # Keep runtime factors separate: their product need not fit a native type.
    runtime_exponents: tuple[_RuntimeGateExponent, ...] = ()
    constant_factors: tuple[int, ...] = ()


@dataclasses.dataclass(frozen=True)
class _OrderedGateModifier:
    kind: ast.GateModifierName
    argument: int | float | _RuntimeGateExponent | None = None


@dataclasses.dataclass(frozen=True)
class _OrderedGateModifierInfo:
    # Source order, not execution order: the last modifier acts first.
    modifiers: tuple[_OrderedGateModifier, ...]
    controls: tuple[tuple[int, int], ...]


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
    MAX_GATE_POWER_OUTPUT_LINES = 100000
    # Keep literal instruction metadata within bra's ANGLEBITS2PHASE limits.
    MAX_ANGLE_PHASE_MODIFIERS = 1024
    MAX_ANGLE_PHASE_LITERAL_CHARS = 4096
    default_gates_qcx_map: dict[str, str] = {'U': 'U3'}

    # OpenQASM overload order is significant: inputs, never the destination,
    # select the first signature permitting all implicit promotions. Native
    # APPLY itself does not promote, so lowering must make those casts explicit.
    builtin_function_signatures: dict[str, tuple[_BuiltinFunctionSignature, ...]] = {
        'exp': (_BuiltinFunctionSignature((ValueType.FLOAT,), ValueType.FLOAT, qcx_function='EXP'),
                _BuiltinFunctionSignature((ValueType.COMPLEX,), ValueType.COMPLEX, qcx_function='EXP')),
        'sqrt': (_BuiltinFunctionSignature((ValueType.FLOAT,), ValueType.FLOAT, qcx_function='SQRT'),
                 _BuiltinFunctionSignature((ValueType.COMPLEX,), ValueType.COMPLEX, qcx_function='SQRT')),
        'log': (_BuiltinFunctionSignature((ValueType.FLOAT,), ValueType.FLOAT, qcx_function='LOG'),),
        'sin': (_BuiltinFunctionSignature((ValueType.FLOAT,), ValueType.FLOAT, qcx_function='SIN'),),
        'cos': (_BuiltinFunctionSignature((ValueType.FLOAT,), ValueType.FLOAT, qcx_function='COS'),),
        'tan': (_BuiltinFunctionSignature((ValueType.FLOAT,), ValueType.FLOAT, qcx_function='TAN'),),
        'arcsin': (_BuiltinFunctionSignature((ValueType.FLOAT,), ValueType.FLOAT, qcx_function='ARCSIN'),),
        'arccos': (_BuiltinFunctionSignature((ValueType.FLOAT,), ValueType.FLOAT, qcx_function='ARCCOS'),),
        'arctan': (_BuiltinFunctionSignature((ValueType.FLOAT,), ValueType.FLOAT, qcx_function='ARCTAN'),),
        'ceiling': (_BuiltinFunctionSignature((ValueType.FLOAT,), ValueType.FLOAT, qcx_function='CEILING'),),
        'floor': (_BuiltinFunctionSignature((ValueType.FLOAT,), ValueType.FLOAT, qcx_function='FLOOR'),),
        'mod': (_BuiltinFunctionSignature((ValueType.INT, ValueType.INT), ValueType.INT, integer_remainder=True),
                _BuiltinFunctionSignature((ValueType.FLOAT, ValueType.FLOAT), ValueType.FLOAT, qcx_function='MOD')),
        'popcount': (_BuiltinFunctionSignature((ValueType.BIT_REGISTER,), ValueType.UINT, population_count=True),
                     _BuiltinFunctionSignature((ValueType.UINT,), ValueType.UINT, population_count=True)),
        'rotl': (_BuiltinFunctionSignature((ValueType.BIT_REGISTER, ValueType.INT), ValueType.BIT_REGISTER, rotation=True),
                 _BuiltinFunctionSignature((ValueType.UINT, ValueType.INT), ValueType.UINT, rotation=True)),
        'rotr': (_BuiltinFunctionSignature((ValueType.BIT_REGISTER, ValueType.INT), ValueType.BIT_REGISTER, rotation=True),
                 _BuiltinFunctionSignature((ValueType.UINT, ValueType.INT), ValueType.UINT, rotation=True)),
        'real': (_BuiltinFunctionSignature((ValueType.COMPLEX,), ValueType.FLOAT, component='REAL'),),
        'imag': (_BuiltinFunctionSignature((ValueType.COMPLEX,), ValueType.FLOAT, component='IMAG'),),
    }

    stdgates_qcx_map: dict[str, str] = {
            'p': 'U1', 'phase': 'U1', 'u1': 'U1', 'u2': 'U2', 'u3': 'U3',
            'x': 'X', 'y': 'Y', 'z': 'Z', 'h': 'H', 'id': 'I',
            's': 'S', 'sdg': 'S+', 't': 'T', 'tdg': 'T+',
            'sx': 'SX', 'rx': 'EX', 'ry': 'EY', 'rz': 'EZ',
            'cx': 'CX', 'CX': 'CX', 'cy': 'CY', 'cz': 'CZ', 'cp': 'CU1', 'cphase': 'CU1',
            'crx': 'CEX', 'cry': 'CEY', 'crz': 'CEZ', 'ch': 'CH',
            'cu': 'CU3', 'swap': 'SWAP', 'ccx': 'CCX', 'cswap': 'CSWAP'}

    # Only mnemonics accepted by QCX are listed. Self-adjoint gates keep
    # their original spelling; in particular, QCX has no H+ or CX+ alias.
    qcx_adjoint_gate_map: dict[str, str] = {
            'I': 'I', 'H': 'H', 'X': 'X', 'Y': 'Y', 'Z': 'Z',
            'CX': 'CX', 'CY': 'CY', 'CZ': 'CZ', 'CH': 'CH',
            'SWAP': 'SWAP', 'CCX': 'CCX', 'CSWAP': 'CSWAP',
            'S': 'S+', 'S+': 'S', 'T': 'T+', 'T+': 'T',
            'SX': 'SX+', 'SX+': 'SX',
            'U1': 'U1+', 'U1+': 'U1', 'U2': 'U2+', 'U2+': 'U2',
            'U3': 'U3+', 'U3+': 'U3',
            'EX': 'EX+', 'EX+': 'EX', 'EY': 'EY+', 'EY+': 'EY',
            'EZ': 'EZ+', 'EZ+': 'EZ',
            'CU1': 'CU1+', 'CU1+': 'CU1', 'CU3': 'CU3+', 'CU3+': 'CU3',
            'CEX': 'CEX+', 'CEX+': 'CEX', 'CEY': 'CEY+', 'CEY+': 'CEY',
            'CEZ': 'CEZ+', 'CEZ+': 'CEZ'}

    # Root QCX mnemonic and controls already present in a lowered gate.
    # Use explicit I1 to distinguish the identity target from control-only IC.
    qcx_controlled_gate_families: dict[str, tuple[str, int]] = {
            'I': ('I1', 0), 'H': ('H', 0),
            'X': ('X', 0), 'Y': ('Y', 0), 'Z': ('Z', 0),
            'CX': ('X', 1), 'CY': ('Y', 1), 'CZ': ('Z', 1), 'CH': ('H', 1),
            'CCX': ('X', 2), 'SWAP': ('SWAP', 0), 'CSWAP': ('SWAP', 1),
            'S': ('S', 0), 'S+': ('S+', 0), 'T': ('T', 0), 'T+': ('T+', 0),
            'SX': ('SX', 0), 'SX+': ('SX+', 0),
            'U1': ('U1', 0), 'U1+': ('U1+', 0),
            'U2': ('U2', 0), 'U2+': ('U2+', 0),
            'U3': ('U3', 0), 'U3+': ('U3+', 0),
            'EX': ('EX', 0), 'EX+': ('EX+', 0),
            'EY': ('EY', 0), 'EY+': ('EY+', 0),
            'EZ': ('EZ', 0), 'EZ+': ('EZ+', 0),
            'CU1': ('U1', 1), 'CU1+': ('U1+', 1),
            'CU3': ('U3', 1), 'CU3+': ('U3+', 1),
            'CEX': ('EX', 1), 'CEX+': ('EX+', 1),
            'CEY': ('EY', 1), 'CEY+': ('EY+', 1),
            'CEZ': ('EZ', 1), 'CEZ+': ('EZ+', 1)}

    # Exact unitary identities, not equivalences up to global phase. These
    # remain valid after adding positive, negative, or intrinsic controls.
    qcx_gate_power_cycles: dict[str, tuple[tuple[str, ...], ...]] = {
            'I1': ((),),
            **{name: ((), (name,)) for name in ('H', 'X', 'Y', 'Z', 'SWAP')},
            'S': ((), ('S',), ('Z',), ('S+',)),
            'T': ((), ('T',), ('S',), ('S', 'T'), ('Z',), ('Z', 'T'), ('S+',), ('T+',)),
            'SX': ((), ('SX',), ('X',), ('SX+',))}

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

    angle_phase_gates = frozenset(('p', 'phase', 'u1', 'cp', 'cphase'))

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
        self.__array_source_sizes: dict[str, int] = {}
        self.__bit_register_source_sizes: dict[str, int] = {}

        self.__int_variable_name_size_map: dict[str, int] = {}
        self.__uint_variable_name_size_map: dict[str, int] = {}
        self.__uint_variable_types: dict[str, _UnsignedIntegerType] = {}
        self.__temporary_uint_types: dict[str, _UnsignedIntegerType] = {}
        # Retain the node as well as its id: synthetic compound-assignment
        # expressions must not leave entries whose ids can later be reused.
        self.__uint_expression_types: dict[int, tuple[ast.Expression, _UnsignedIntegerType]] = {}
        self.__angle_variable_types: dict[str, _AngleType] = {}
        self.__angle_variable_names: set[str] = set()
        self.__angle_expression_types: dict[int, tuple[ast.Expression, _AngleType]] = {}
        self.__array_element_types: dict[str, ValueType] = {}
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
        self.__const_bit_variable_values_map: dict[str, int] = {}
        self.__const_angle_values: dict[str, _ConstantAngle] = {}
        self.__const_bit_register_values: dict[str, _ConstantBitRegister] = {}
        self.__declared_constant_variables: set[str] = set()

        self.__is_stdgates_included: bool = False
        self.__quantum_registers: dict[str, int] = {}
        self.__sized_quantum_registers: set[str] = set()
        self.__declared_quantum_registers: set[str] = set()
        self.__amplitude_indices: list[int] | None = None
        self.__branch_index: int = 0
        self.__array_index: int = 0
        self.__register_shift_index: int = 0
        self.__rotation_index: int = 0
        self.__population_count_index: int = 0
        self.__power_index: int = 0
        self.__gate_power_index: int = 0
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
        if isinstance(node, ast.Identifier) and self.__value_type == ValueType.ANGLE:
            info = self.__identifier_angle_type(node.name)
            if info is not None:
                self.__angle_expression_types[id(node)] = (node, info)
        if isinstance(node, ast.Expression) and self.__value_type == ValueType.UINT:
            if isinstance(node, ast.Identifier):
                info = self.__identifier_uint_type(node.name)
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
        elif identifier_name in self.__angle_variable_names:
            return ValueType.ANGLE
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
        if value_type == ValueType.ANGLE:
            raise UnsupportedOpenQASMError('angle temporaries require UINT storage and angle width metadata')
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
                self.__const_bit_variable_values_map,
                self.__const_angle_values,
                self.__const_bit_register_values,
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
        elif expression.name in self.__const_bit_register_values:
            raise UnsupportedOpenQASMError('constant bit register requires a register-expression context')
        elif expression.name in self.__const_angle_values:
            self.__value = self.__const_angle_values[expression.name].value
            self.__value_type, self.__value_kind = ValueType.ANGLE, ValueKind.LITERAL
        elif expression.name in self.__const_bit_variable_values_map:
            self.__value = self.__const_bit_variable_values_map[expression.name]
            self.__value_type = ValueType.BIT
            self.__value_kind = ValueKind.LITERAL
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
            if self.__value in self.__array_element_types:
                raise UnsupportedOpenQASMError('whole classical array in scalar expression')
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
        if expression.op.name == '-' and self.__expression_mentions_angle(expression):
            value_type, _, _ = self.__angle_operand_info(expression)
            if value_type == ValueType.ANGLE:
                self.__visit_angle_arithmetic(expression)
                return
        previous_temporaries = self.__used_temporary_variables.copy()
        try:
            self.visit(expression.expression)
        except Exception:
            # A failed nested operand may not return its partial results for
            # normal cleanup. Keep caller-owned storage reserved.
            for temporary in self.__used_temporary_variables - previous_temporaries:
                self.__release_temporary_variable(temporary)
            raise
        if self.__value is None or self.__value_type is None or self.__value_kind is None:
            raise UninitializedValueException
        if self.__value_type == ValueType.ANGLE:
            if self.__value_kind == ValueKind.TEMPORARY:
                self.__release_temporary_variable(str(self.__value))
            raise UnsupportedOpenQASMError('angle unary arithmetic')
        if self.__value_type in (ValueType.BIT, ValueType.BOOL):
            if self.__value_kind == ValueKind.TEMPORARY:
                self.__release_temporary_variable(str(self.__value))
            feature = ('unary arithmetic on bit values' if self.__value_type == ValueType.BIT
                       else 'unary value expression on Boolean values')
            raise UnsupportedOpenQASMError(feature)

        if expression.op != ast.UnaryOperator['-']:
            raise UnsupportedOpenQASMError(f'unary operator {expression.op.name}')

        if self.__value_type == ValueType.INT:
            try:
                if not self.__native_signed_operand(expression.expression):
                    raise UnsupportedOpenQASMError('unary negation of an integer with a non-native signed width')
            except QASM2QCXError:
                if self.__value_kind == ValueKind.TEMPORARY:
                    self.__release_temporary_variable(str(self.__value))
                raise
            self.__visit_native_int_negation(expression)
            return
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
        if self.__value_type == ValueType.COMPLEX:
            # (-0,-0) - z flips both component signs, unlike complex
            # multiplication by -1, whose cross terms can erase signed zeros.
            # Only the finite zero seed goes through APPLY, so LET retains its
            # existing nonfinite propagation for the actual operand.
            self.__qcx_lines.extend([f'APPLY {temporary_variable} <- COMPLEX -0.0 -0.0',
                                     f'LET {temporary_variable} -= {operand}'])
        else:
            self.__qcx_lines.append(f'LET {temporary_variable} := {operand}')
            multiplier = ':UINT:-1' if self.__value_type == ValueType.UINT else '-1'
            self.__qcx_lines.append(f'LET {temporary_variable} *= {multiplier}')
        if uint_type is not None:
            self.__normalize_uint_storage(temporary_variable, uint_type)
        if operand_kind == ValueKind.TEMPORARY:
            self.__release_temporary_variable(operand)
        self.__value = temporary_variable
        self.__value_kind = ValueKind.TEMPORARY

    def __visit_native_int_negation(self, expression: ast.UnaryExpression) -> None:
        operand, kind = self.__value, self.__value_kind
        result = None
        keep = False
        try:
            if kind == ValueKind.LITERAL:
                if not self.__evaluate_constant:
                    self.__value = 0
                    return
                try:
                    self.__validate_native_negation_value(expression, int(operand))
                except InvalidNegationException:
                    if (self.__expression_kind == ExpressionKind.CONST_ARITHMETIC
                            or not self.QCX_INT_MIN <= int(operand) <= self.QCX_INT_MAX):
                        raise
                    # Preserve runtime reachability for a known overflowing
                    # result. Invalid inputs that cannot be represented in an
                    # INT LET remain conversion errors on all paths.
                else:
                    self.__value = -int(operand)
                    return
            elif self.__expression_kind == ExpressionKind.CONST_ARITHMETIC:
                raise NoConstantExpressionException
            result = self.__add_new_temporary_variable(ValueType.INT)
            self.__qcx_lines.extend([
                f'LET {result} := {operand}',
                f'ASSERT {result} \\= {self.QCX_INT_MIN}',
                f'LET {result} *= -1',
            ])
            self.__value, self.__value_type, self.__value_kind = result, ValueType.INT, ValueKind.TEMPORARY
            keep = True
        finally:
            if kind == ValueKind.TEMPORARY:
                self.__release_temporary_variable(str(operand))
            if result is not None and not keep:
                self.__release_temporary_variable(result)

    def __negation_expression_info(
            self, expression: ast.UnaryExpression,
            iterators: set[str] | dict[str, ValueType] | None = None
            ) -> tuple[ValueType, bool]:
        # Inspect types/dependencies only, including in empty loop bodies.
        if self.__expression_mentions_angle(expression.expression):
            angle_value_type, _, angle_runtime = self.__angle_operand_info(expression.expression, iterators)
            if angle_value_type == ValueType.ANGLE:
                return angle_value_type, angle_runtime
        value_type, _, runtime = self.__integer_operand_info(expression.expression, iterators)
        if value_type == ValueType.BIT:
            raise UnsupportedOpenQASMError('unary arithmetic on bit values')
        if value_type == ValueType.BOOL:
            raise UnsupportedOpenQASMError('unary value expression on Boolean values')
        if value_type not in (ValueType.INT, ValueType.UINT, ValueType.FLOAT, ValueType.COMPLEX):
            raise UnsupportedOpenQASMError('unary negation requires a numeric scalar operand')
        if value_type == ValueType.INT and not self.__native_signed_operand(expression.expression, iterators):
            raise UnsupportedOpenQASMError('unary negation of an integer with a non-native signed width')
        self.__validate_bitwise_indices(expression.expression, iterators)
        return value_type, runtime

    def __is_minimum_int_literal(self, expression: ast.UnaryExpression) -> bool:
        # The sign is an AST operator, not part of an IntegerLiteral. Admit
        # this spelling of INT_MIN, but not negation of an existing INT_MIN.
        return (expression.op.name == '-'
                and isinstance(expression.expression, ast.IntegerLiteral)
                and expression.expression.value == -self.QCX_INT_MIN)

    def __validate_native_negation_value(self, expression: ast.UnaryExpression, value: int) -> None:
        # Pure boundary validation, without emitting instructions or arithmetic.
        if self.__is_minimum_int_literal(expression) and value == -self.QCX_INT_MIN:
            return
        if not self.QCX_INT_MIN <= value <= self.QCX_INT_MAX:
            raise InvalidNegationException('Integer negation operand outside QCX INT range')
        if value == self.QCX_INT_MIN:
            raise InvalidNegationException('Integer negation result outside QCX INT range')

    @staticmethod
    def __bitwise_type(lhs_type: ValueType, rhs_type: ValueType | None = None) -> ValueType:
        if lhs_type in (ValueType.INT, ValueType.UINT) and rhs_type in (None, ValueType.INT, ValueType.UINT):
            return ValueType.UINT if ValueType.UINT in (lhs_type, rhs_type) else ValueType.INT
        if lhs_type != ValueType.BIT or rhs_type not in (None, ValueType.BIT):
            raise UnsupportedOpenQASMError(
                'bitwise operators require matching integer or scalar bit operands')
        return lhs_type

    def __bit_register_expression_width(
            self, expression: ast.Expression,
            iterators: set[str] | dict[str, ValueType] | None = None) -> int | None:
        # None denotes a scalar, not a one-bit register. Inspect declaration
        # metadata and constant selection bounds only: never emit QCX or
        # evaluate runtime count/index values in an empty loop body.
        if isinstance(expression, ast.Identifier):
            name = expression.name
            shadowed = ((iterators is not None and name in iterators)
                        or any(name in scope for scope in self.__loop_bindings))
            if not shadowed and name in self.__const_bit_register_values:
                return self.__constant_bit_register(name).width
            if not shadowed and name in self.__bit_register_source_sizes:
                return self.__bit_register_source_sizes[name]
        if isinstance(expression, ast.IndexExpression) and self.__is_register_bit_selection(expression):
            return self.__static_bit_selection(expression, iterators).width
        if isinstance(expression, ast.BitstringLiteral):
            if expression.width <= 0:
                raise InvalidBitOperandException('A bit-string operand must have a positive width')
            return expression.width
        if isinstance(expression, ast.FunctionCall) and expression.name.name in ('rotl', 'rotr'):
            signature, _ = self.__function_call_info(expression, iterators)
            if signature.result_type == ValueType.BIT_REGISTER:
                return self.__bit_register_expression_width(expression.arguments[0], iterators)
            return None
        if isinstance(expression, ast.UnaryExpression) and expression.op.name == '~':
            width = self.__bit_register_expression_width(expression.expression, iterators)
            if width is not None:
                return width
        if isinstance(expression, ast.BinaryExpression) and expression.op.name in ('<<', '>>'):
            width = self.__bit_register_expression_width(expression.lhs, iterators)
            if width is not None:
                count_type, _ = self.__loop_expression_info(expression.rhs, iterators)
                if count_type not in (ValueType.INT, ValueType.UINT):
                    raise InvalidBitOperandException('A register shift count must be a scalar integer')
                self.__validate_bitwise_indices(expression.rhs, iterators)
                return width
        if isinstance(expression, ast.BinaryExpression) and expression.op.name in ('&', '|', '^'):
            lhs = self.__bit_register_expression_width(expression.lhs, iterators)
            rhs = self.__bit_register_expression_width(expression.rhs, iterators)
            if lhs is not None or rhs is not None:
                if lhs is None or rhs is None:
                    raise InvalidBitOperandException(
                        'Bitwise operands must both be scalars or both be registers')
                if lhs != rhs:
                    raise InvalidBitOperandException(
                        f'Bitwise register widths {lhs} and {rhs} do not match')
                return lhs
        # Preserve existing scalar operator/type rules and check indexed
        # operand syntax without reading an element or computing its index.
        self.__loop_expression_info(expression, iterators)
        self.__validate_bitwise_indices(expression, iterators)
        return None

    @staticmethod
    def __fold_rotation(value: int, width: int, distance: int, name: str) -> int:
        # Modulo handles oversized and negative distances without abs/negation
        # of a possibly minimum native signed integer.
        count = distance % width
        if name == 'rotr' and count:
            count = width - count
        mask = (1 << width) - 1
        value &= mask
        if count == 0:
            return value
        return ((value << count) | (value >> (width - count))) & mask

    @staticmethod
    def __fold_bit_register_shift(value: int, width: int, count: int, operator: str) -> int:
        if width <= 0:
            raise ValueError('Register shift width must be positive')
        if operator not in ('<<', '>>'):
            raise ValueError('Invalid register shift operator')
        if count < 0:
            raise InvalidShiftException('Register shift count must not be negative')
        # Register shifts are not native integer shifts. Discarding all bits
        # yields zero, without constructing an enormous Python left shift.
        if count >= width:
            return 0
        mask = (1 << width) - 1
        value &= mask
        return ((value << count) if operator == '<<' else (value >> count)) & mask

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

    def __identifier_uint_type(self, name: str) -> _UnsignedIntegerType | None:
        # An active iterator shadows constants, variables, and source arrays,
        # including UINT sources with a different width or sizedness.
        for scope in reversed(self.__loop_bindings):
            if name in scope:
                binding = scope[name]
                return binding.uint_type if isinstance(binding, _RuntimeIteratorBinding) else None
        return (self.__const_uint_types.get(name)
                or self.__uint_variable_types.get(self.__capitalize_variable_name(name)))

    def __integer_expression_type(
            self, lhs_type: ValueType, rhs_type: ValueType,
            lhs: ast.Expression, rhs: ast.Expression) -> tuple[ValueType, _UnsignedIntegerType | None]:
        infos = [self.__uint_expression_type(node)
                 for node, value_type in ((lhs, lhs_type), (rhs, rhs_type)) if value_type == ValueType.UINT]
        return self.__integer_result_type(lhs_type, rhs_type, lhs, rhs, infos)

    def __integer_result_type(
            self, lhs_type: ValueType, rhs_type: ValueType,
            lhs: ast.Expression, rhs: ast.Expression,
            infos: list[_UnsignedIntegerType | None],
            iterators: set[str] | dict[str, ValueType] | None = None
            ) -> tuple[ValueType, _UnsignedIntegerType | None]:
        # Shared by value lowering and side-effect-free power metadata.
        result_type = self.__promoted_type(lhs_type, rhs_type)
        if result_type != ValueType.UINT:
            return result_type, None
        if any(info is None for info in infos):
            raise UninitializedValueException
        width = max(info.width for info in infos)
        if ValueType.INT in (lhs_type, rhs_type):
            for node, value_type in ((lhs, lhs_type), (rhs, rhs_type)):
                if value_type == ValueType.INT and not self.__native_signed_operand(node, iterators):
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
            if expression.op.name in ('<<', '>>'):
                # Shift counts do not contribute to the result's signed rank.
                return self.__native_signed_operand(expression.lhs, iterators)
            return (self.__native_signed_operand(expression.lhs, iterators)
                    and self.__native_signed_operand(expression.rhs, iterators))
        if isinstance(source_type, ast.ArrayType):
            source_type = source_type.base_type
        if not isinstance(source_type, ast.IntType) or source_type.size is None:
            return True
        if isinstance(expression, ast.Cast):
            names = self.__loop_bound_names(source_type.size)
            active = set(iterators or ()) | {name for scope in self.__loop_bindings for name in scope}
            if names & active:
                raise InvalidDeclarationException('INT cast width must be a constant integer, not a loop iterator')
        else:
            # Source declarations are global; their widths must not be
            # reinterpreted through the current loop's iterator bindings.
            previous_bindings = self.__loop_bindings
            self.__loop_bindings = []
            try:
                return self.__constant_loop_integer(source_type.size, 'signed operand width') == self.QCX_UINT_WIDTH
            finally:
                self.__loop_bindings = previous_bindings
        return self.__constant_loop_integer(source_type.size, 'signed operand width') == self.QCX_UINT_WIDTH

    def __converted_operand(
            self, value: str | int | float | complex, value_type: ValueType,
            value_kind: ValueKind, result_type: ValueType) -> tuple[str, set[str]]:
        if ValueType.ANGLE in (value_type, result_type):
            if value_type == result_type:
                return str(value), set()
            raise NoImplicitCastException
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
            if complex_value.imag == 0.0 and math.copysign(1.0, complex_value.imag) > 0:
                return f':COMPLEX:{complex_value.real}', set()
            positive_real_zero = complex_value.real == 0.0 and math.copysign(1.0, complex_value.real) > 0
            if positive_real_zero and complex_value.imag == 1.0:
                return ':I', set()
            if positive_real_zero and complex_value.imag == -1.0:
                return ':MINUS_I', set()

            # Shared literal conversion covers function arguments, variables,
            # constants copied into runtime storage, and array initializers.
            # Arithmetic assembly loses signed zeros and tiny components.
            temporary = self.__add_new_temporary_variable(ValueType.COMPLEX)
            self.__qcx_lines.append(
                f'APPLY {temporary} <- COMPLEX :REAL:{complex_value.real} :REAL:{complex_value.imag}')
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

        if expression.op.name in ('*', '/') and self.__expression_mentions_angle(expression):
            info = self.__angle_multiplicative_info(expression)
            if info is not None:
                self.__visit_angle_multiplicative(expression, info)
                return
        if expression.op.name == '**':
            self.__visit_power_binary(expression)
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
        if expression.op.name in ('+', '-') and self.__expression_mentions_angle(expression):
            value_type, _, _ = self.__angle_operand_info(expression)
            if value_type == ValueType.ANGLE:
                self.__visit_angle_arithmetic(expression)
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

    def __validate_power_types(
            self, expression: ast.BinaryExpression, lhs_type: ValueType, rhs_type: ValueType,
            iterators: set[str] | dict[str, ValueType] | None = None) -> None:
        if lhs_type not in (ValueType.INT, ValueType.UINT) or rhs_type not in (ValueType.INT, ValueType.UINT):
            raise UnsupportedOpenQASMError('integer exponentiation requires scalar int or uint operands')
        for operand, value_type in ((expression.lhs, lhs_type), (expression.rhs, rhs_type)):
            if value_type == ValueType.INT and not self.__native_signed_operand(operand, iterators):
                raise UnsupportedOpenQASMError('integer exponentiation with a non-native signed width')

    def __power_expression_info(
            self, expression: ast.BinaryExpression,
            iterators: set[str] | dict[str, ValueType] | None = None
            ) -> tuple[ValueType, _UnsignedIntegerType | None, bool]:
        lhs_type, lhs_uint, lhs_runtime = self.__integer_operand_info(expression.lhs, iterators)
        rhs_type, rhs_uint, rhs_runtime = self.__integer_operand_info(expression.rhs, iterators)
        self.__validate_power_types(expression, lhs_type, rhs_type, iterators)
        self.__validate_bitwise_indices(expression.lhs, iterators)
        self.__validate_bitwise_indices(expression.rhs, iterators)
        infos = [info for value_type, info in ((lhs_type, lhs_uint), (rhs_type, rhs_uint))
                 if value_type == ValueType.UINT]
        result_type, uint_type = self.__integer_result_type(
            lhs_type, rhs_type, expression.lhs, expression.rhs, infos, iterators)
        return result_type, uint_type, lhs_runtime or rhs_runtime

    def __integer_operand_info(
            self, expression: ast.Expression,
            iterators: set[str] | dict[str, ValueType] | None
            ) -> tuple[ValueType, _UnsignedIntegerType | None, bool]:
        # Resolve widths from declarations/AST, never from mutable visit state.
        # Inspect both operands even if one already has a runtime dependency.
        if (isinstance(expression, ast.BinaryExpression) and expression.op.name in ('*', '/')
                and self.__expression_mentions_angle(expression)):
            info = self.__angle_multiplicative_info(expression, iterators)
            if info is not None:
                return info.result_type, info.result if info.result_type == ValueType.UINT else None, info.runtime
        if isinstance(expression, ast.BinaryExpression) and expression.op.name == '**':
            return self.__power_expression_info(expression, iterators)
        value_type, runtime = self.__loop_expression_info(expression, iterators)
        if isinstance(expression, ast.BinaryExpression) and expression.op.name in ('+', '-', '*', '/', '%', '&', '|', '^', '<<', '>>'):
            lhs_type, lhs_uint, _ = self.__integer_operand_info(expression.lhs, iterators)
            rhs_type, rhs_uint, _ = self.__integer_operand_info(expression.rhs, iterators)
            if expression.op.name in ('<<', '>>'):
                return lhs_type, lhs_uint, runtime
            infos = [info for operand_type, info in ((lhs_type, lhs_uint), (rhs_type, rhs_uint))
                     if operand_type == ValueType.UINT]
            result_type, uint_type = self.__integer_result_type(
                lhs_type, rhs_type, expression.lhs, expression.rhs, infos, iterators)
            return result_type, uint_type, runtime
        if isinstance(expression, ast.UnaryExpression) and expression.op.name in ('-', '~'):
            return self.__integer_operand_info(expression.expression, iterators)
        if value_type != ValueType.UINT:
            return value_type, None, runtime
        if isinstance(expression, ast.Identifier):
            uint_type = self.__identifier_uint_type(expression.name)
            if uint_type is None:
                raise UninitializedValueException
            # A declaration's width is already resolved; a loop iterator may
            # now shadow a constant used in that width without changing it.
            return value_type, uint_type, runtime
        elif isinstance(expression, ast.IndexExpression) and isinstance(expression.collection, ast.Identifier):
            uint_type = self.__uint_variable_types.get(
                self.__capitalize_variable_name(expression.collection.name))
            if uint_type is None:
                raise UninitializedValueException
            return value_type, uint_type, runtime
        elif isinstance(expression, ast.Cast):
            source_type = expression.type
        elif isinstance(expression, ast.FunctionCall):
            if expression.name.name in ('rotl', 'rotr'):
                _, uint_type, _ = self.__integer_operand_info(expression.arguments[0], iterators)
                return value_type, uint_type, runtime
            # popcount returns native UINT, unlike width-preserving rotations.
            return value_type, _UnsignedIntegerType(self.QCX_UINT_WIDTH, False), runtime
        else:
            raise UnsupportedOpenQASMError('unsupported unsigned exponentiation operand')
        return value_type, self.__unsigned_integer_type(source_type, iterators), runtime

    def __fold_power(self, base: int, exponent: int,
                     uint_type: _UnsignedIntegerType | None) -> int:
        if exponent < 0:
            raise InvalidPowerException('Integer exponent must not be negative')
        if uint_type is not None:
            return pow(base, exponent, 1 << uint_type.width)
        result = 1
        while exponent:
            if exponent & 1:
                result *= base
                if not self.QCX_INT_MIN <= result <= self.QCX_INT_MAX:
                    raise InvalidPowerException('Integer power result outside QCX INT range')
            exponent >>= 1
            if exponent:
                base *= base
                if not self.QCX_INT_MIN <= base <= self.QCX_INT_MAX:
                    raise InvalidPowerException('Integer power result outside QCX INT range')
        return result

    def __visit_power_binary(self, expression: ast.BinaryExpression) -> None:
        self.__power_expression_info(expression)
        operands = []
        try:
            for operand in (expression.lhs, expression.rhs):
                self.visit(operand)
                if self.__value is None or self.__value_type is None or self.__value_kind is None:
                    raise UninitializedValueException
                operands.append((self.__value, self.__value_type, self.__value_kind))
            (lhs, lhs_type, lhs_kind), (rhs, rhs_type, rhs_kind) = operands
            result_type, uint_type = self.__integer_expression_type(
                lhs_type, rhs_type, expression.lhs, expression.rhs)
            if uint_type is not None:
                self.__uint_expression_types[id(expression)] = (expression, uint_type)
            self.__validate_power_literal_operands(operands)
            if lhs_kind == rhs_kind == ValueKind.LITERAL:
                if not self.__evaluate_constant:
                    self.__value, self.__value_type, self.__value_kind = 0, result_type, ValueKind.LITERAL
                    return
                try:
                    # Check the signed exponent before unsigned promotion.
                    if int(rhs) < 0:
                        raise InvalidPowerException('Integer exponent must not be negative')
                    base = uint_type.normalize(int(lhs)) if uint_type is not None else int(lhs)
                    folded = self.__fold_power(base, int(rhs), uint_type)
                except InvalidPowerException:
                    if self.__expression_kind == ExpressionKind.CONST_ARITHMETIC:
                        raise
                    # Literal failures on runtime paths must also be skippable.
                else:
                    self.__value, self.__value_type, self.__value_kind = folded, result_type, ValueKind.LITERAL
                    return
            if self.__expression_kind == ExpressionKind.CONST_ARITHMETIC:
                raise NoConstantExpressionException
            self.__emit_power_expression(operands, result_type, uint_type)
        finally:
            for value, _, kind in operands:
                if kind == ValueKind.TEMPORARY:
                    self.__release_temporary_variable(str(value))

    def __validate_power_literal_operands(
            self, operands: list[tuple[str | int | float | complex, ValueType, ValueKind]]) -> None:
        for value, value_type, kind in operands:
            if (kind == ValueKind.LITERAL and value_type == ValueType.INT
                    and self.__evaluate_constant
                    and not self.QCX_INT_MIN <= int(value) <= self.QCX_INT_MAX):
                raise UnsupportedOpenQASMError('integer exponentiation operand outside QCX INT range')

    def __emit_power_expression(
            self, operands: list[tuple[str | int | float | complex, ValueType, ValueKind]],
            result_type: ValueType, uint_type: _UnsignedIntegerType | None) -> None:
        self.__validate_power_literal_operands(operands)
        (lhs, lhs_type, lhs_kind), (rhs, rhs_type, rhs_kind) = operands
        owned: set[str] = set()
        keep = None

        def allocate(value_type=ValueType.UINT, info=None):
            name = self.__add_new_temporary_variable(value_type, info)
            owned.add(name)
            return name

        prefix = f'QASM2QCX_POWER_{self.__power_index}'
        self.__power_index += 1
        try:
            count, parity = allocate(), allocate()
            base, accumulator = allocate(info=uint_type), allocate(info=uint_type)
            result = accumulator if uint_type is not None else allocate(ValueType.INT)
            limit = quotient = negative = None
            if uint_type is None:
                limit, quotient, negative = allocate(), allocate(), allocate(ValueType.INT)
            if rhs_type == ValueType.INT:
                if rhs_kind == ValueKind.LITERAL:
                    captured = allocate(ValueType.INT)
                    self.__qcx_lines.append(f'LET {captured} := {rhs}')
                else:
                    captured = str(rhs)
                self.__qcx_lines.append(f'ASSERT {captured} >= 0')
            # After the domain check, no exponent is narrowed through INT.
            exponent, _ = self.__converted_operand(rhs, rhs_type, rhs_kind, ValueType.UINT)
            factor, _ = self.__converted_operand(lhs, lhs_type, lhs_kind, ValueType.UINT)
            self.__qcx_lines.extend([f'LET {count} := {exponent}',
                                     f'LET {base} := {factor}', f'LET {accumulator} := 1'])
            if uint_type is None:
                signed_base, _ = self.__converted_operand(lhs, lhs_type, lhs_kind, ValueType.INT)
                # Compare only variables in QCX. Use result as the base snapshot
                # until the final signed result is reconstructed.
                self.__qcx_lines.extend([
                    f'LET {result} := {signed_base}', f'LET {negative} := 0',
                    f'LET {limit} := {self.QCX_INT_MAX}',
                    f'JUMPIF {prefix}_READY {result} >= 0',
                    f'LET {base} := 0', f'LET {base} -= {factor}',
                    f'LET {parity} := {count}', f'LET {parity} &= 1',
                    f'JUMPIF {prefix}_READY {parity} == 0',
                    f'LET {negative} := 1', f'LET {limit} := {-self.QCX_INT_MIN}',
                    f'@{prefix}_READY',
                ])
            self.__qcx_lines.extend([
                f'@{prefix}_LOOP', f'JUMPIF {prefix}_DONE {count} == 0',
                f'LET {parity} := {count}', f'LET {parity} &= 1',
                f'JUMPIF {prefix}_NEXT {parity} == 0',
            ])
            self.__emit_power_multiply(accumulator, base, uint_type, limit, quotient, prefix + '_PRODUCT')
            self.__qcx_lines.extend([
                f'@{prefix}_NEXT', f'LET {count} >>= 1',
                f'JUMPIF {prefix}_DONE {count} == 0',
            ])
            # Do not square once the exponent is exhausted: x ** 1 must not
            # fail just because the unused x ** 2 would overflow.
            self.__emit_power_multiply(base, base, uint_type, limit, quotient, prefix + '_SQUARE')
            self.__qcx_lines.extend([f'JUMP {prefix}_LOOP', f'@{prefix}_DONE'])
            if uint_type is None:
                self.__qcx_lines.extend([
                    f'JUMPIF {prefix}_POSITIVE {negative} == 0',
                    f'JUMPIF {prefix}_MINIMUM {accumulator} == {-self.QCX_INT_MIN}',
                    f'LET {result} := :INT:{accumulator}', f'LET {result} *= -1',
                    f'JUMP {prefix}_END', f'@{prefix}_MINIMUM',
                    f'LET {result} := {self.QCX_INT_MIN}', f'JUMP {prefix}_END',
                    f'@{prefix}_POSITIVE', f'LET {result} := :INT:{accumulator}', f'@{prefix}_END',
                ])
            self.__value, self.__value_type, self.__value_kind = result, result_type, ValueKind.TEMPORARY
            keep = result
        finally:
            for name in owned - {keep}:
                self.__release_temporary_variable(name)

    def __emit_power_multiply(
            self, target: str, factor: str, uint_type: _UnsignedIntegerType | None,
            limit: str | None, quotient: str | None, label: str) -> None:
        if uint_type is None:
            # Magnitudes use UINT, including abs(INT_MIN). Division guards the
            # multiplication without evaluating any potentially overflowing INT
            # operation, and a zero factor never becomes a divisor.
            self.__qcx_lines.extend([
                f'JUMPIF {label} {factor} == 0', f'LET {quotient} := {limit}',
                f'LET {quotient} /= {factor}', f'ASSERT {target} <= {quotient}', f'@{label}',
            ])
        self.__qcx_lines.append(f'LET {target} *= {factor}')
        if uint_type is not None:
            self.__normalize_uint_storage(target, uint_type)

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
        # Keep both operands live until %= has read them. The result must not
        # alias either operand, including a captured array destination index.
        result = self.__add_new_temporary_variable(value_type, uint_type)
        self.__qcx_lines.extend([
            f'LET {result} := {lhs}', f'LET {result} %= {rhs}',
        ])
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

    @classmethod
    def __is_literal_qubit_selector(cls, expression: ast.QASMNode) -> bool:
        if isinstance(expression, ast.IntegerLiteral):
            return True
        if isinstance(expression, ast.UnaryExpression):
            return expression.op.name == '-' and isinstance(expression.expression, ast.IntegerLiteral)
        if isinstance(expression, ast.RangeDefinition):
            return all(part is None or cls.__is_literal_qubit_selector(part)
                       for part in (expression.start, expression.step, expression.end))
        if isinstance(expression, ast.DiscreteSet):
            return all(cls.__is_literal_qubit_selector(value) for value in expression.values)
        return False

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
                # Validate incrementally instead of materializing a potentially
                # enormous invalid range before its first bounds failure.
                raw_indices = range(start, stop, step)
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

    @staticmethod
    def __is_register_bit_selection(operand: ast.QASMNode) -> bool:
        if isinstance(operand, ast.IndexExpression):
            selector = operand.index
        elif isinstance(operand, ast.IndexedIdentifier) and len(operand.indices) == 1:
            selector = operand.indices[0]
        else:
            return False
        return (isinstance(selector, ast.DiscreteSet)
                or (isinstance(selector, list) and any(isinstance(value, ast.RangeDefinition) for value in selector)))

    def __static_bit_selection(
            self, operand: ast.Identifier | ast.IndexExpression | ast.IndexedIdentifier,
            iterators: set[str] | dict[str, ValueType] | None = None, *,
            destination: bool = False) -> _StaticBitSelection:
        if isinstance(operand, ast.IndexExpression):
            if not isinstance(operand.collection, ast.Identifier):
                raise UnsupportedOpenQASMError('indexing a computed bit-register expression')
            name, selector = operand.collection.name, operand.index
        elif isinstance(operand, ast.IndexedIdentifier):
            if len(operand.indices) != 1:
                raise UnsupportedOpenQASMError('multidimensional bit indexing')
            name, selector = operand.name.name, operand.indices[0]
        else:
            name, selector = operand.name, None
        active = ((iterators is not None and name in iterators)
                  or any(name in scope for scope in self.__loop_bindings))
        if active:
            raise UnsupportedOpenQASMError('static bit selection of a for-loop iterator')
        constant = self.__constant_bit_register(name) if name in self.__const_bit_register_values else None
        if destination and constant is not None:
            raise InvalidBitOperandException('A constant bit register is read-only')
        source_type = (self.__constant_source_types[name] if constant is not None
                       else self.__classical_source_types.get(name))
        if source_type is None:
            raise NoVariableNameException(name)
        if not isinstance(source_type, ast.BitType):
            raise InvalidBitOperandException(f'Bit selection source {name} is not a bit variable')
        size = constant.width if constant is not None else self.__bit_register_source_sizes.get(name, 1)
        if selector is None:
            return _StaticBitSelection(name, tuple(range(size)), source_type.size is not None)
        # Even an unrolled iterator is not a declaration-scope constant. Do
        # not give empty-loop validation a fabricated value for selection bounds.
        bound_names = self.__loop_bound_names(selector) if isinstance(selector, ast.QASMNode) else set().union(
            *(self.__loop_bound_names(value) for value in selector if isinstance(value, ast.QASMNode)))
        iterator_names = set(iterators or ()) | {name for scope in self.__loop_bindings for name in scope}
        if bound_names & iterator_names:
            raise UnsupportedOpenQASMError('iterator-dependent bit-register selection')
        for bound in bound_names:
            if bound in self.__classical_source_types:
                raise UnsupportedOpenQASMError('runtime-dependent bit-register selection')
        previous = (self.__expression_kind, self.__value, self.__value_type,
                    self.__value_kind, self.__evaluate_constant)
        try:
            indices, is_register = self.__index_selection(
                selector, size, 'bit', f'variable {name}[{size}]', InvalidBitOperandException)
        finally:
            (self.__expression_kind, self.__value, self.__value_type,
             self.__value_kind, self.__evaluate_constant) = previous
        if destination and len(set(indices)) != len(indices):
            raise InvalidBitOperandException('A bit selection destination must not repeat an index')
        return _StaticBitSelection(name, tuple(indices), is_register)

    def __bit_operand(
            self, bit: ast.Identifier | ast.IndexedIdentifier, *,
            role: str = 'Bit operand'
            ) -> tuple[str, list[int], bool]:
        source_name = self.__operand_name(bit)
        if role == 'Bit source' and isinstance(bit, ast.Identifier):
            for scope in reversed(self.__loop_bindings):
                if source_name in scope:
                    binding = scope[source_name]
                    if (isinstance(binding, _RuntimeIteratorBinding)
                            and binding.value_type in (ValueType.BIT, ValueType.BOOL)):
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
        self.__reject_constant_bit_target(target)
        self.__reject_constant_angle_target(target)
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
        if value_type == ValueType.ANGLE:
            raise UnsupportedOpenQASMError('angle-typed gate parameters')
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

    def __u_global_phase(
            self, parameters: list[
                tuple[str | int | float, ValueType, ValueKind]],
            converted_parameters: list[str]) -> tuple[str, ValueKind]:
        # The published built-in U matrix is exp(i*theta/2) times the
        # conventional OpenQASM 2 matrix implemented by QCX U3.
        if all(parameter[2] == ValueKind.LITERAL for parameter in parameters):
            phase = float(parameters[0][0]) / 2.0
            return str(phase), ValueKind.LITERAL

        temporary = self.__add_new_temporary_variable(ValueType.FLOAT)
        self.__qcx_lines.append(
            f'LET {temporary} := {converted_parameters[0]}')
        self.__qcx_lines.append(f'LET {temporary} /= 2.0')
        return temporary, ValueKind.TEMPORARY

    @staticmethod
    def __inverse_modifier_parity(modifiers: list[ast.QuantumGateModifier]) -> bool:
        # Validate the entire list before reducing it: cancelling inversions
        # must not hide an unsupported modifier or an invalid inv argument.
        for modifier in modifiers:
            if modifier.modifier != ast.GateModifierName.inv:
                raise UnsupportedOpenQASMError(f'gate modifier {modifier.modifier.name}')
            if modifier.argument is not None:
                raise UnsupportedOpenQASMError('inv modifier with an argument')
        return len(modifiers) % 2 == 1

    @classmethod
    def __adjoint_gate_name(cls, mnemonic: str) -> str:
        try:
            return cls.qcx_adjoint_gate_map[mnemonic]
        except KeyError:
            raise UnsupportedOpenQASMError(f'adjoint of QCX gate {mnemonic}') from None

    def __control_modifiers(
            self, modifiers: list[ast.QuantumGateModifier],
            iterators: set[str] | dict[str, ValueType] | None = None) -> list[tuple[int, int]]:
        for modifier in modifiers:
            if modifier.modifier not in (ast.GateModifierName.ctrl, ast.GateModifierName.negctrl):
                raise UnsupportedOpenQASMError(f'gate modifier {modifier.modifier.name} mixed with controls')
        total = 0
        groups = []
        iterator_names = set(iterators or ()) | {name for scope in self.__loop_bindings for name in scope}
        for modifier in modifiers:
            name = modifier.modifier.name
            count = 1
            if modifier.argument is not None:
                expression = modifier.argument
                if self.__loop_bound_names(expression) & iterator_names:
                    raise UnsupportedOpenQASMError(f'{name} count must not depend on a loop iterator')
                value_type, runtime = self.__loop_expression_info(expression, iterators)
                if value_type not in (ValueType.INT, ValueType.UINT) or runtime:
                    raise UnsupportedOpenQASMError(f'{name} count must be a constant integer')
                count = self.__constant_loop_integer(expression, 'control count', kind='modifier')
            if not 1 <= count <= self.QCX_INT_MAX:
                raise UnsupportedOpenQASMError(f'{name} count must be in [1, {self.QCX_INT_MAX}]')
            total += count
            if total > self.QCX_INT_MAX:
                raise UnsupportedOpenQASMError(f'total control count exceeds {self.QCX_INT_MAX}')
            # Keep counts compact until arity has been checked against actual
            # operands: ctrl(INT_MAX) must not allocate INT_MAX descriptors.
            groups.append((int(modifier.modifier == ast.GateModifierName.ctrl), count))
        return groups

    @classmethod
    def __controlled_gate_name(cls, mnemonic: str, added_controls: int) -> str:
        if not 1 <= added_controls <= cls.QCX_INT_MAX:
            raise UnsupportedOpenQASMError(f'added ctrl count must be in [1, {cls.QCX_INT_MAX}]')
        try:
            root, existing_controls = cls.qcx_controlled_gate_families[mnemonic]
        except KeyError:
            raise UnsupportedOpenQASMError(f'controlled form of QCX gate {mnemonic}') from None
        count = added_controls + existing_controls
        if count > cls.QCX_INT_MAX:
            raise UnsupportedOpenQASMError(f'total QCX ctrl count exceeds {cls.QCX_INT_MAX}')
        return ('C' if count == 1 else f'C{count}') + root

    @classmethod
    def __controlled_phase_gate_name(cls, num_controls: int) -> str:
        if not 1 <= num_controls <= cls.QCX_INT_MAX:
            raise UnsupportedOpenQASMError(f'phase ctrl count must be in [1, {cls.QCX_INT_MAX}]')
        # A conditioned global phase is a U1 on one condition qubit,
        # controlled by the remaining condition qubits (not by an extra target).
        return 'U1' if num_controls == 1 else cls.__controlled_gate_name('U1', num_controls - 1)

    def __power_modifier_exponent(
            self, modifier: ast.QuantumGateModifier,
            iterators: set[str] | dict[str, ValueType] | None = None, *,
            allow_runtime: bool = False, allow_float: bool = False) -> int | float | _RuntimeGateExponent:
        expression = modifier.argument
        if expression is None:
            raise UnsupportedOpenQASMError('pow modifier requires an exponent')
        iterator_names = set(iterators or ()) | {name for scope in self.__loop_bindings for name in scope}
        depends_on_iterator = bool(self.__loop_bound_names(expression) & iterator_names)
        if depends_on_iterator and not allow_runtime:
            raise UnsupportedOpenQASMError('pow exponent must not depend on a loop iterator')
        value_type, runtime = self.__loop_expression_info(expression, iterators)
        if allow_float and value_type == ValueType.FLOAT:
            if runtime or depends_on_iterator:
                raise UnsupportedOpenQASMError('floating-point pow exponent must be a compile-time constant')
            return self.__constant_float_gate_exponent(expression)
        if value_type not in (ValueType.INT, ValueType.UINT):
            raise UnsupportedOpenQASMError('pow exponent must be an integer')
        if runtime or depends_on_iterator:
            if not allow_runtime:
                raise UnsupportedOpenQASMError('pow exponent must be a constant integer')
            return _RuntimeGateExponent(expression, value_type)
        return self.__constant_loop_integer(expression, 'pow exponent', kind='modifier')

    def __constant_float_gate_exponent(self, expression: ast.Expression) -> float:
        # Analyze only when explicitly requested by the floating-power path.
        # In particular, never multiply this value into integer modifier info.
        previous = (self.__expression_kind, self.__value, self.__value_type,
                    self.__value_kind, self.__evaluate_constant)
        try:
            self.__expression_kind = ExpressionKind.CONST_ARITHMETIC
            self.__value = self.__value_type = self.__value_kind = None
            self.__evaluate_constant = True
            try:
                self.visit(expression)
                if self.__value_type != ValueType.FLOAT or self.__value_kind != ValueKind.LITERAL:
                    raise UnsupportedOpenQASMError('floating-point pow exponent must be a compile-time constant')
                value = float(self.__value)
            except (OverflowError, ValueError, ZeroDivisionError) as error:
                raise UnsupportedOpenQASMError('invalid constant floating-point pow exponent') from error
            if not math.isfinite(value):
                raise UnsupportedOpenQASMError('floating-point pow exponent must be finite')
            return value
        finally:
            (self.__expression_kind, self.__value, self.__value_type,
             self.__value_kind, self.__evaluate_constant) = previous

    def __analyze_gate_modifiers(
            self, modifiers: list[ast.QuantumGateModifier],
            iterators: set[str] | dict[str, ValueType] | None = None, *,
            allow_runtime: bool = False, allow_float: bool = False
            ) -> _GateModifierInfo | _OrderedGateModifierInfo:
        for modifier in modifiers:
            if modifier.modifier not in (
                    ast.GateModifierName.inv, ast.GateModifierName.ctrl,
                    ast.GateModifierName.negctrl, ast.GateModifierName.pow):
                raise UnsupportedOpenQASMError(f'gate modifier {modifier.modifier.name}')
        inverse = self.__inverse_modifier_parity([
            modifier for modifier in modifiers if modifier.modifier == ast.GateModifierName.inv])
        controls = self.__control_modifiers([
            modifier for modifier in modifiers if modifier.modifier in (
                ast.GateModifierName.ctrl, ast.GateModifierName.negctrl)], iterators)
        power = -1 if inverse else 1
        runtime_exponents = []
        constant_factors = []
        ordered = []
        control_groups = iter(controls)
        has_float = False
        for modifier in modifiers:
            if modifier.modifier == ast.GateModifierName.pow:
                # Do not short-circuit at zero: every exponent must be valid.
                exponent = self.__power_modifier_exponent(
                    modifier, iterators, allow_runtime=allow_runtime, allow_float=allow_float)
                ordered.append(_OrderedGateModifier(modifier.modifier, exponent))
                if isinstance(exponent, _RuntimeGateExponent):
                    runtime_exponents.append(exponent)
                elif isinstance(exponent, float):
                    has_float = True
                else:
                    constant_factors.append(exponent)
            elif modifier.modifier == ast.GateModifierName.inv:
                ordered.append(_OrderedGateModifier(modifier.modifier))
            else:
                _, count = next(control_groups)
                ordered.append(_OrderedGateModifier(modifier.modifier, count))
        if has_float:
            # Do not use the integer product or inverse parity for this path.
            # An intermediate operation may cross the principal branch cut.
            return _OrderedGateModifierInfo(tuple(ordered), tuple(controls))
        for exponent in constant_factors:
            power *= exponent
        return _GateModifierInfo(power, tuple(controls), tuple(runtime_exponents),
                                 tuple(constant_factors) if runtime_exponents else ())

    @staticmethod
    def __principal_gate_phase(phase: Fraction) -> Fraction:
        # Eigenphases are rational multiples of pi for the initial fixed gate
        # subset. Exact arithmetic fixes -1's phase at +pi, without a tolerance
        # that could accidentally select the wrong side of the branch cut.
        result = phase % 2
        return result - 2 if result > 1 else result

    @classmethod
    def __ordered_gate_eigenphases(
            cls, phases: tuple[Fraction, ...], info: _OrderedGateModifierInfo) -> tuple[Fraction, ...]:
        for modifier in reversed(info.modifiers):
            if modifier.kind == ast.GateModifierName.inv:
                phases = tuple(-phase for phase in phases)
            elif modifier.kind == ast.GateModifierName.pow:
                if isinstance(modifier.argument, _RuntimeGateExponent):
                    raise UnsupportedOpenQASMError('runtime integer factors in floating-point pow chains')
                exponent = Fraction(modifier.argument)
                phases = tuple(cls.__principal_gate_phase(phase) * exponent for phase in phases)
            # Controls add inactive eigenvalues of 1, which every real power
            # and inverse leaves at 1. Their polarity/order is kept in info.
        return tuple(cls.__principal_gate_phase(phase) for phase in phases)

    @contextlib.contextmanager
    def __captured_gate_exponents(
            self, exponents: tuple[_RuntimeGateExponent, ...]) -> Iterator[tuple[_CapturedGateExponent, ...]]:
        # Snapshots stay reserved through the caller's complete gate emission,
        # not merely until the next exponent has been evaluated.
        previous_used = self.__used_temporary_variables.copy()
        previous = (self.__expression_kind, self.__value, self.__value_type,
                    self.__value_kind, self.__evaluate_constant)
        captured = []
        try:
            try:
                for exponent in exponents:
                    before_expression = self.__used_temporary_variables.copy()
                    self.__expression_kind = ExpressionKind.ARITHMETIC
                    self.__value = self.__value_type = self.__value_kind = None
                    self.__evaluate_constant = True
                    self.visit(exponent.expression)
                    if (self.__value_type != exponent.value_type
                            or self.__value_type not in (ValueType.INT, ValueType.UINT)
                            or self.__value_kind is None or self.__value is None):
                        raise UnsupportedOpenQASMError('pow exponent must be an integer')
                    if self.__value_kind == ValueKind.LITERAL:
                        minimum, maximum = ((self.QCX_INT_MIN, self.QCX_INT_MAX)
                                            if exponent.value_type == ValueType.INT else (0, self.QCX_UINT_MAX))
                        if not minimum <= int(self.__value) <= maximum:
                            raise UnsupportedOpenQASMError('captured pow exponent outside its native integer range')
                    storage = self.__add_new_temporary_variable(exponent.value_type)
                    self.__qcx_lines.append(f'LET {storage} := {self.__value}')
                    captured.append(_CapturedGateExponent(storage, exponent.value_type))
                    for temporary in self.__used_temporary_variables - before_expression - {storage}:
                        self.__release_temporary_variable(temporary)
            finally:
                (self.__expression_kind, self.__value, self.__value_type,
                 self.__value_kind, self.__evaluate_constant) = previous
            yield tuple(captured)
        finally:
            for temporary in self.__used_temporary_variables - previous_used:
                self.__release_temporary_variable(temporary)

    @staticmethod
    def __condition_qubits(qubits: list[str], controls: tuple[tuple[int, int], ...]) -> list[str]:
        # Each modifier prepends its controls. Thus source modifier order is
        # also operand order, even with interleaved inverse modifiers.
        result = []
        offset = 0
        for value, count in controls:
            result.extend(qubit if value else f'{qubit}#0' for qubit in qubits[offset:offset + count])
            offset += count
        return result

    def __gate_power_label(self) -> str:
        index = self.__gate_power_index
        self.__gate_power_index += 1
        return f'QASM2QCX_GATE_POWER_{index}'

    @contextlib.contextmanager
    def __runtime_gate_power(self, info: _GateModifierInfo) -> Iterator[_RuntimeGatePower | None]:
        with self.__captured_gate_exponents(info.runtime_exponents) as captured:
            if not captured:
                yield None
                return
            sign = None
            if any(exponent.value_type == ValueType.INT for exponent in captured):
                sign = self.__add_new_temporary_variable(ValueType.INT)
                self.__qcx_lines.append(f'LET {sign} := 0')
            magnitudes = []
            for exponent in captured:
                if exponent.value_type == ValueType.UINT:
                    magnitudes.append(exponent.storage)
                    continue
                magnitude = self.__add_new_temporary_variable(ValueType.UINT)
                ready = self.__gate_power_label() + '_READY'
                self.__qcx_lines.extend([
                    f'LET {magnitude} := :UINT:{exponent.storage}',
                    f'JUMPIF {ready} {exponent.storage} >= 0',
                    f'LET {magnitude} := 0', f'LET {magnitude} -= :UINT:{exponent.storage}',
                    f'LET {sign} ^= -1', f'@{ready}'])
                magnitudes.append(magnitude)
            yield _RuntimeGatePower(tuple(magnitudes), sign, info.constant_factors)

    def __qubit_operand_tuples(self, qubits, *, distinct: bool = False) -> Iterator[list[str]]:
        operands = [self.__qubit_operand_indices(qubit) for qubit in qubits]
        register_sizes = [len(indices) for indices, is_register in operands if is_register]
        if len(set(register_sizes)) > 1:
            raise WrongBroadcastingException
        for index in range(register_sizes[0] if register_sizes else 1):
            physical = []
            for qubit, (indices, is_register) in zip(qubits, operands):
                name = qubit.name if isinstance(qubit, ast.Identifier) else qubit.name.name
                physical.append(str(self.__to_qubit_index(name, indices[index] if is_register else indices[0])))
            if distinct and len(set(physical)) != len(physical):
                raise UnsupportedOpenQASMError('controlled operation requires distinct qubit operands')
            yield physical

    def visit_QuantumGate(self, statement: ast.QuantumGate) -> None:
        if self.__is_initialization_process:
            return
        previous_used = self.__used_temporary_variables.copy()
        previous_kind = self.__expression_kind
        try:
            self.__emit_quantum_gate(statement)
        finally:
            self.__expression_kind = previous_kind
            for temporary in self.__used_temporary_variables - previous_used:
                self.__release_temporary_variable(temporary)

    def __emit_quantum_gate(self, statement: ast.QuantumGate) -> None:
        info = self.__analyze_gate_modifiers(statement.modifiers, allow_runtime=True, allow_float=True)
        if isinstance(info, _OrderedGateModifierInfo):
            self.__emit_quantum_gate_operation(statement, info, None)
            return
        with self.__runtime_gate_power(info) as runtime:
            self.__emit_quantum_gate_operation(statement, info, runtime)

    def __emit_quantum_gate_operation(
            self, statement: ast.QuantumGate, info: _GateModifierInfo | _OrderedGateModifierInfo,
            runtime: _RuntimeGatePower | None) -> None:
        qasm_gate_name: str = statement.name.name
        control_groups = info.controls
        controls = sum(count for _, count in control_groups)
        if statement.duration is not None:
            raise UnsupportedOpenQASMError(f'duration on gate {qasm_gate_name}')

        if qasm_gate_name in QASM2QCXConverter.default_gates_qcx_map:
            qcx_gate_name = QASM2QCXConverter.default_gates_qcx_map[qasm_gate_name]
        elif self.__is_stdgates_included and qasm_gate_name in QASM2QCXConverter.stdgates_qcx_map:
            qcx_gate_name = QASM2QCXConverter.stdgates_qcx_map[qasm_gate_name]
        else:
            raise UnsupportedOpenQASMError(f'gate {qasm_gate_name}')

        expected_parameters, expected_qubits = QASM2QCXConverter.gate_signatures[qasm_gate_name]
        expected_qubits += controls
        if (len(statement.arguments) != expected_parameters
                or len(statement.qubits) != expected_qubits):
            raise WrongGateArityException(
                qasm_gate_name, expected_parameters, expected_qubits,
                len(statement.arguments), len(statement.qubits))

        angle_parameter = self.__angle_phase_parameter(statement)
        if angle_parameter is not None:
            if isinstance(info, _OrderedGateModifierInfo):
                self.__emit_parameterized_phase_gate(statement, info, qcx_gate_name)
                return
            output_start = (len(self.__qcx_lines) if any(
                modifier.modifier == ast.GateModifierName.pow for modifier in statement.modifiers) else None)
            constant = self.__constant_angle_gate_value(angle_parameter)
            if constant is not None and runtime is None:
                powered = _ConstantAngle(constant.angle_type,
                                         (constant.value * abs(info.power)) & constant.angle_type.mask)
                self.__emit_constant_angle_phase_gate(statement, powered, info, output_start)
            else:
                self.__emit_runtime_angle_phase_gate(statement, angle_parameter, info, runtime, output_start)
            return

        if isinstance(info, _OrderedGateModifierInfo):
            family = self.qcx_controlled_gate_families[qcx_gate_name][0]
            if family == 'U1':
                self.__emit_parameterized_phase_gate(statement, info, qcx_gate_name)
            elif family in ('EX', 'EY', 'EZ'):
                self.__emit_floating_rotation_gate(statement, info, qcx_gate_name)
            elif qasm_gate_name in ('u2', 'u3', 'U', 'cu'):
                self.__emit_floating_u_gate(statement, info)
            else:
                self.__emit_floating_gate_power(statement, info, qcx_gate_name)
            return
        inverse = info.power < 0

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

        u_global_phase: str | None = None
        inverse_phase: str | None = None
        if qasm_gate_name in ('U', 'cu'):
            u_global_phase, value_kind = self.__u_global_phase(
                parameters, converted_parameters)
            if value_kind == ValueKind.TEMPORARY:
                temporary_variables.add(u_global_phase)
            if qasm_gate_name == 'U' and (
                    inverse or (runtime is not None and runtime.signed_exponent is not None)):
                inverse_phase = self.__negated_phase_operand(u_global_phase, value_kind)

        powered = any(modifier.modifier == ast.GateModifierName.pow for modifier in statement.modifiers)
        mnemonics, repetitions = [qcx_gate_name], abs(info.power)
        if powered:
            mnemonics, repetitions = self.__gate_power_plan(qcx_gate_name, repetitions)
            root, _ = self.qcx_controlled_gate_families[qcx_gate_name]
            if root in ('U1', 'EX', 'EY', 'EZ') and repetitions > 1:
                scaled = self.__scaled_phase_operand(
                    converted_parameters[0], parameters[0][2], repetitions)
                if scaled is not None:
                    converted_parameters[0] = scaled
                    repetitions = 1

        output_start = len(self.__qcx_lines) if powered else None
        for all_qubits in self.__qubit_operand_tuples(statement.qubits, distinct=powered or controls > 0):
            condition_qubits = self.__condition_qubits(all_qubits[:controls], control_groups)
            qubit_indices = all_qubits[controls:]
            if not mnemonics:
                sequence = []
            elif qasm_gate_name == 'cu':
                # The active block is exp(i*gamma)*U, with
                # U = exp(i*theta/2)*conventional-u3. Keep both phases
                # separate: gamma+theta/2 can lose theta's correction or
                # overflow even when each original parameter is finite.
                sequence = [('U1', [qubit_indices[0], converted_parameters[3]]),
                            ('U1', [qubit_indices[0], u_global_phase]),
                            ('CU3', qubit_indices + converted_parameters[:3])]
            else:
                sequence = []
                if u_global_phase is not None:
                    sequence.append(('PHASE', [u_global_phase]))
                sequence.extend((mnemonic, qubit_indices + converted_parameters) for mnemonic in mnemonics)
            lowered = self.__lower_gate_sequence(sequence, inverse, condition_qubits, inverse_phase)
            if runtime is None:
                self.__emit_gate_power(lowered, repetitions, output_start)
            else:
                adjoint = (self.__lower_gate_sequence(sequence, not inverse, condition_qubits, inverse_phase)
                           if runtime.signed_exponent is not None else lowered)
                root, _ = self.qcx_controlled_gate_families[qcx_gate_name]
                cycle = self.qcx_gate_power_cycles.get(root.rstrip('+'))
                period = len(cycle) // math.gcd(len(cycle), info.power) if cycle else None
                self.__emit_runtime_gate_power(lowered, adjoint, repetitions, runtime, output_start, period)

        for temporary_variable in temporary_variables:
            self.__release_temporary_variable(temporary_variable)

    def __angle_phase_parameter(
            self, statement: ast.QuantumGate | ast.QuantumPhase,
            iterators: set[str] | dict[str, ValueType] | None = None) -> ast.Expression | None:
        # Shared by emission and non-evaluating loop validation. Do not let
        # unsupported angle uses disappear in empty loops or zero powers.
        arguments = statement.arguments if isinstance(statement, ast.QuantumGate) else [statement.argument]
        angle_arguments = [argument for argument in arguments
                           if self.__expression_mentions_angle(argument)
                           and self.__loop_expression_info(argument, iterators)[0] == ValueType.ANGLE]
        if not angle_arguments:
            return None
        if isinstance(statement, ast.QuantumGate) and statement.name.name not in self.angle_phase_gates:
            raise UnsupportedOpenQASMError('angle-valued parameter on a non-phase gate')
        if len(arguments) != 1:
            raise UnsupportedOpenQASMError('angle-valued phase gate requires one parameter')
        # Fractional powers operate on the whole exact angle, never on the
        # individual R contributions used by the integer-only lowering.
        info = self.__analyze_gate_modifiers(statement.modifiers, iterators, allow_runtime=True, allow_float=True)
        if isinstance(info, _OrderedGateModifierInfo):
            self.__angle_phase_descriptors(info)
        return angle_arguments[0]

    @classmethod
    def __angle_phase_descriptors(cls, info: _OrderedGateModifierInfo) -> tuple[str, ...]:
        descriptors = []
        count = 0
        # Source modifiers are outermost first; the native descriptor chain
        # is in evaluation order. Controls stay in the quantum operands.
        for modifier in reversed(info.modifiers):
            if modifier.kind not in (ast.GateModifierName.inv, ast.GateModifierName.pow):
                continue
            count += 1
            if count > cls.MAX_ANGLE_PHASE_MODIFIERS:
                raise UnsupportedOpenQASMError(
                    f'ANGLEBITS2PHASE chain exceeds {cls.MAX_ANGLE_PHASE_MODIFIERS} modifiers')
            if modifier.kind == ast.GateModifierName.inv:
                descriptors.append('INV')
                continue
            if isinstance(modifier.argument, _RuntimeGateExponent):
                raise UnsupportedOpenQASMError('runtime integer factors in floating-point pow chains')
            try:
                # repr retains the binary64 value and floating spelling;
                # str preserves integer factors beyond binary64/native INT.
                literal = repr(modifier.argument) if isinstance(modifier.argument, float) else str(modifier.argument)
            except ValueError as error:
                raise UnsupportedOpenQASMError('ANGLEBITS2PHASE exponent literal exceeds host serialization limits') from error
            if len(literal) > cls.MAX_ANGLE_PHASE_LITERAL_CHARS:
                raise UnsupportedOpenQASMError(
                    f'ANGLEBITS2PHASE exponent exceeds {cls.MAX_ANGLE_PHASE_LITERAL_CHARS} characters')
            descriptors.extend(('POW', literal))
        return tuple(descriptors)

    @contextlib.contextmanager
    def __powered_angle_phase(
            self, expression: ast.Expression, info: _OrderedGateModifierInfo) -> Iterator[str]:
        descriptors = self.__angle_phase_descriptors(info)
        with self.__captured_angle_bits(expression) as (snapshot, angle_type):
            phase = self.__add_new_temporary_variable(ValueType.FLOAT)
            self.__qcx_lines.append(
                f'APPLY {phase} <- ANGLEBITS2PHASE {snapshot} {angle_type.width} {" ".join(descriptors)}')
            yield phase

    def __constant_angle_gate_value(
            self, expression: ast.Expression,
            iterators: set[str] | dict[str, ValueType] | None = None) -> _ConstantAngle | None:
        if isinstance(expression, ast.Identifier):
            value_type, runtime = self.__loop_identifier_info(expression, iterators)
            return self.__const_angle_values.get(expression.name) if value_type == ValueType.ANGLE and not runtime else None
        if isinstance(expression, ast.Cast) and isinstance(expression.type, ast.AngleType):
            target = self.__angle_type(expression.type, iterators)
            source = self.__constant_angle_gate_value(expression.argument, iterators)
            if source is not None:
                return _ConstantAngle(target, self.__resize_angle_bits(source.value, source.angle_type, target))
        # Even literal float-to-angle casts are executable native conversions;
        # do not replace their backend precision with constant host folding.
        return None

    def __angle_phase_operands(
            self, statement: ast.QuantumGate | ast.QuantumPhase,
            info: _GateModifierInfo) -> Iterator[tuple[str, list[str]]]:
        added_controls = sum(count for _, count in info.controls)
        count = len(statement.qubits) - 1
        mnemonic = ('' if count == 0 else 'C' if count == 1 else f'C{count}') + 'R'
        if info.power < 0:
            mnemonic += '+'
        for qubits in self.__qubit_operand_tuples(statement.qubits, distinct=count > 0):
            conditions = self.__condition_qubits(qubits[:added_controls], info.controls)
            yield mnemonic, conditions + qubits[added_controls:]

    def __emit_constant_angle_phase_gate(
            self, statement: ast.QuantumGate | ast.QuantumPhase, constant: _ConstantAngle,
            info: _GateModifierInfo, output_start: int | None) -> None:
        exponents = [constant.angle_type.width - bit for bit in range(constant.angle_type.width)
                     if constant.value & (1 << bit)]
        # Expand operands even for zero: identity must not hide invalid qubits,
        # repeated controlled operands, or incompatible broadcast sizes.
        for mnemonic, qubits in self.__angle_phase_operands(statement, info):
            self.__check_gate_power_output(len(exponents), output_start)
            for exponent in exponents:
                self.__qcx_lines.append(f'{mnemonic} {" ".join(qubits)} {exponent}')

    def __emit_constant_angle_global_phase(
            self, statement: ast.QuantumPhase, constant: _ConstantAngle,
            info: _GateModifierInfo, output_start: int | None) -> None:
        if statement.qubits:
            # A controlled global phase has no extra target. One condition
            # uses R; two use CR; all controls retain their own polarity.
            self.__emit_constant_angle_phase_gate(statement, constant, info, output_start)
            return
        exponents = sorted(constant.angle_type.width - bit for bit in range(constant.angle_type.width)
                           if constant.value & (1 << bit))
        if not exponents:
            return
        phase = self.__add_new_temporary_variable(ValueType.FLOAT)
        try:
            self.__qcx_lines.append(f'LET {phase} := :TWO_PI')
            if info.power < 0:
                self.__qcx_lines.append(f'LET {phase} *= -1.0')
            previous = 0
            for exponent in exponents:
                # A power-of-two REAL literal is exact at every backend
                # precision. Do not fold radian coefficients on the host.
                divisor = f'{1 << (exponent - previous)}.0'
                self.__emit_gate_power([f'LET {phase} /= {divisor}', f'PHASE {phase}'], 1, output_start)
                previous = exponent
        finally:
            self.__release_temporary_variable(phase)

    def __emit_runtime_angle_global_phase(
            self, statement: ast.QuantumPhase, expression: ast.Expression, info: _GateModifierInfo,
            runtime: _RuntimeGatePower | None, output_start: int | None) -> None:
        if statement.qubits:
            self.__emit_runtime_angle_phase_gate(statement, expression, info, runtime, output_start)
            return
        with self.__captured_angle_bits(expression) as (snapshot, angle_type):
            self.__power_angle_bits(snapshot, angle_type, info, runtime)
            scratch = self.__add_new_temporary_variable(ValueType.UINT)
            phase = self.__add_new_temporary_variable(ValueType.FLOAT)
            prefix = self.__gate_power_label() + '_GLOBAL_ANGLE'
            self.__qcx_lines.append(f'LET {phase} := :TWO_PI')
            if info.power < 0:
                self.__qcx_lines.append(f'LET {phase} *= -1.0')
            for exponent in range(1, angle_type.width + 1):
                skip = f'{prefix}_{exponent}_SKIP'
                # Scale on every bit, including zero bits, so later nonzero
                # contributions retain their correct backend-precision angle.
                self.__emit_gate_power([
                    f'LET {phase} /= 2.0', f'LET {scratch} := {snapshot}',
                    f'LET {scratch} &= {1 << (angle_type.width - exponent)}',
                    f'JUMPIF {skip} {scratch} == 0', f'PHASE {phase}', f'@{skip}'], 1, output_start)

    def __validate_angle_phase_qubits(
            self, statement: ast.QuantumGate | ast.QuantumPhase, iterators: dict[str, ValueType]) -> None:
        if self.__is_initialization_process:
            return
        for qubit in statement.qubits:
            name = self.__operand_name(qubit)
            if name in iterators or name not in self.__declared_quantum_registers:
                raise InvalidQubitOperandException(f'{name} is not a declared qubit operand')
        indices = [index for qubit in statement.qubits if isinstance(qubit, ast.IndexedIdentifier)
                   for group in qubit.indices
                   for index in (group if isinstance(group, list) else [group])]
        if all(self.__is_literal_qubit_selector(index) for index in indices):
            for _ in self.__qubit_operand_tuples(statement.qubits, distinct=len(statement.qubits) > 1):
                pass

    @contextlib.contextmanager
    def __captured_angle_bits(self, expression: ast.Expression) -> Iterator[tuple[str, _AngleType]]:
        previous_used = self.__used_temporary_variables.copy()
        try:
            value, value_type, _ = self.__condition_operand(expression)
            angle_type = self.__angle_expression_type(expression)
            if value_type != ValueType.ANGLE or angle_type is None:
                raise UnsupportedOpenQASMError('angle gate parameter without angle width metadata')
            snapshot = self.__add_new_temporary_variable(
                ValueType.UINT, _UnsignedIntegerType(angle_type.width, True))
            # Copy raw internal bits, not an OpenQASM angle-to-integer cast.
            # Keep this private snapshot live across the complete broadcast.
            self.__qcx_lines.append(f'LET {snapshot} := {value}')
            yield snapshot, angle_type
        finally:
            for temporary in self.__used_temporary_variables - previous_used:
                self.__release_temporary_variable(temporary)

    def __power_angle_bits(
            self, snapshot: str, angle_type: _AngleType,
            info: _GateModifierInfo, runtime: _RuntimeGatePower | None) -> None:
        # Native unsigned wrap is modulo 2^W. Because n <= W, masking that
        # product gives its exact residue modulo 2^n, with no signed overflow.
        factor = abs(info.power) & angle_type.mask
        if factor != 1:
            self.__qcx_lines.extend([f'LET {snapshot} *= {factor}', f'LET {snapshot} &= {angle_type.mask}'])
        if runtime is None:
            return
        for magnitude in runtime.magnitudes:
            self.__qcx_lines.extend([f'LET {snapshot} *= {magnitude}', f'LET {snapshot} &= {angle_type.mask}'])
        if runtime.signed_exponent is not None:
            ready = self.__gate_power_label() + '_ANGLE_SIGN_READY'
            # -1 modulo 2^n is the mask. INT_MIN was already captured as an
            # unsigned magnitude by the shared exponent machinery.
            self.__qcx_lines.extend([
                f'JUMPIF {ready} {runtime.signed_exponent} >= 0',
                f'LET {snapshot} *= {angle_type.mask}', f'LET {snapshot} &= {angle_type.mask}', f'@{ready}'])

    def __emit_runtime_angle_phase_gate(
            self, statement: ast.QuantumGate | ast.QuantumPhase, expression: ast.Expression, info: _GateModifierInfo,
            runtime: _RuntimeGatePower | None, output_start: int | None) -> None:
        with self.__captured_angle_bits(expression) as (snapshot, angle_type):
            self.__power_angle_bits(snapshot, angle_type, info, runtime)
            # Allocate scratch before any bit-test jumps. Never mutate the
            # snapshot or source while testing individual bit contributions.
            scratch = self.__add_new_temporary_variable(ValueType.UINT)
            prefix = self.__gate_power_label() + '_ANGLE'
            for index, (mnemonic, qubits) in enumerate(self.__angle_phase_operands(statement, info)):
                for bit in range(angle_type.width):
                    skip = f'{prefix}_{index}_{bit}_SKIP'
                    self.__check_gate_power_output(5, output_start)
                    self.__qcx_lines.extend([
                        f'LET {scratch} := {snapshot}', f'LET {scratch} &= {1 << bit}',
                        f'JUMPIF {skip} {scratch} == 0',
                        f'{mnemonic} {" ".join(qubits)} {angle_type.width - bit}', f'@{skip}'])

    @contextlib.contextmanager
    def __captured_phase_angle(
            self, parameter: tuple[str | int | float | complex, ValueType, ValueKind]
            ) -> Iterator[tuple[str, str]]:
        # The caller evaluates the source expression once. Reserve its snapshot
        # and coefficient through the entire modifier chain and broadcasting.
        # EXP validates even a zero power or an inactive quantum control. These
        # instructions must remain inside the reached classical path.
        if parameter[1] not in (ValueType.INT, ValueType.UINT, ValueType.FLOAT):
            raise WrongParameterTypeException(parameter[0])
        previous_used = self.__used_temporary_variables.copy()
        try:
            angle = self.__add_new_temporary_variable(ValueType.FLOAT)
            coefficient = self.__add_new_temporary_variable(ValueType.COMPLEX)
            if (parameter[2] == ValueKind.LITERAL and parameter[1] == ValueType.FLOAT
                    and not math.isfinite(parameter[0])):
                # QCX has no portable inf/nan literal. Materialize a folded
                # nonfinite angle at runtime so a skipped classical path does
                # not fail while a reached zero power still invokes EXP's check.
                numerator = 0.0 if math.isnan(parameter[0]) else math.copysign(1.0, parameter[0])
                self.__qcx_lines.extend([f'LET {angle} := {numerator!r}', f'LET {angle} /= 0.0'])
            else:
                try:
                    operand, _ = self.__converted_operand(*parameter, ValueType.FLOAT)
                except OverflowError as error:
                    raise UnsupportedOpenQASMError('phase angle outside the supported REAL literal range') from error
                self.__qcx_lines.append(f'LET {angle} := {operand}')
            self.__emit_phase_coefficient(angle, coefficient)
            yield angle, coefficient
        finally:
            for temporary in self.__used_temporary_variables - previous_used:
                self.__release_temporary_variable(temporary)

    def __emit_phase_coefficient(self, angle: str, coefficient: str) -> None:
        # Do not use arithmetic modulo a machine approximation of 2*pi.
        # Recompute after each modifier so an intermediate nonfinite product
        # fails before a later zero power could hide it.
        self.__qcx_lines.extend([
            f'LET {coefficient} := :I',
            f'LET {coefficient} *= :COMPLEX:{angle}',
            f'APPLY {coefficient} <- EXP {coefficient}'])

    def __emit_principal_phase(self, angle: str, coefficient: str) -> None:
        # ARG preserves the side of the cut, including a nearby eigenvalue
        # whose argument rounds to -PI. Only an exactly negative-real
        # coefficient selects the gate-power convention +PI for either zero
        # sign. Inspect components, not angle == -PI or an epsilon window.
        previous_used = self.__used_temporary_variables.copy()
        try:
            imaginary = self.__add_new_temporary_variable(ValueType.FLOAT)
            real = self.__add_new_temporary_variable(ValueType.FLOAT)
            ready = self.__gate_power_label() + '_PRINCIPAL'
            self.__qcx_lines.extend([
                f'APPLY {angle} <- ARG {coefficient}',
                f'LET {imaginary} := :IMAG:{coefficient}',
                f'JUMPIF {ready} {imaginary} \\= 0.0',
                f'LET {real} := :REAL:{coefficient}',
                f'JUMPIF {ready} {real} >= 0.0',
                f'LET {angle} := :PI', f'@{ready}'])
        finally:
            for temporary in self.__used_temporary_variables - previous_used:
                self.__release_temporary_variable(temporary)

    def __floating_gate_power_phase(self, mnemonic: str, info: _OrderedGateModifierInfo) -> Fraction:
        root, _ = self.qcx_controlled_gate_families[mnemonic]
        phases = {'I1': Fraction(0), 'X': Fraction(1), 'Y': Fraction(1),
                  'Z': Fraction(1), 'H': Fraction(1), 'SWAP': Fraction(1),
                  'S': Fraction(1, 2), 'S+': Fraction(-1, 2),
                  'T': Fraction(1, 4), 'T+': Fraction(-1, 4), 'SX': Fraction(1, 2)}
        if root not in phases:
            raise UnsupportedOpenQASMError(f'floating-point pow on QCX gate family {root}')
        # Every supported family has one nontrivial eigenphase. All remaining
        # phases are zero and remain zero under powers, inversion, and controls.
        return self.__ordered_gate_eigenphases((phases[root],), info)[0]

    @staticmethod
    def __phase_power_factors(info: _OrderedGateModifierInfo) -> tuple[float | None, ...]:
        # Validate the whole chain before evaluating an angle. Integer factors
        # in this floating path are promoted individually, never multiplied or
        # cancelled together; the integer-only lowering is unchanged.
        factors = []
        for modifier in info.modifiers:
            factor = None
            if modifier.kind == ast.GateModifierName.pow:
                if isinstance(modifier.argument, _RuntimeGateExponent):
                    raise UnsupportedOpenQASMError('runtime integer factors in floating-point pow chains')
                try:
                    factor = float(modifier.argument)
                except (OverflowError, ValueError) as error:
                    raise UnsupportedOpenQASMError('phase-gate pow factor must fit a finite REAL literal') from error
                if not math.isfinite(factor):
                    raise UnsupportedOpenQASMError('phase-gate pow factor must fit a finite REAL literal')
            factors.append(factor)
        return tuple(factors)

    @contextlib.contextmanager
    def __evaluated_phase_angle(self, expression: ast.Expression) -> Iterator[tuple[str, str]]:
        previous_used = self.__used_temporary_variables.copy()
        previous = (self.__expression_kind, self.__value, self.__value_type,
                    self.__value_kind, self.__evaluate_constant)
        try:
            self.__expression_kind = ExpressionKind.ARITHMETIC
            self.__evaluate_constant = True
            self.__value = self.__value_type = self.__value_kind = None
            self.visit(expression)
            parameter = (self.__value, self.__value_type, self.__value_kind)
            with self.__captured_phase_angle(parameter) as captured:
                yield captured
        finally:
            (self.__expression_kind, self.__value, self.__value_type,
             self.__value_kind, self.__evaluate_constant) = previous
            for temporary in self.__used_temporary_variables - previous_used:
                self.__release_temporary_variable(temporary)

    def __emit_phase_modifiers(
            self, phases: tuple[tuple[str, str], ...], info: _OrderedGateModifierInfo,
            factors: tuple[float | None, ...], *, preserve_inverse_coefficients: bool = False) -> None:
        # Finish and validate each modifier for every eigenvalue before moving
        # outward. Neither a later zero nor the other eigenvalue may hide an
        # intermediate nonfinite product. Select each principal phase separately:
        # even conjugate coefficients can both select +PI at the exact cut.
        for modifier, factor in reversed(tuple(zip(info.modifiers, factors))):
            if modifier.kind not in (ast.GateModifierName.pow, ast.GateModifierName.inv):
                continue
            for angle, coefficient in phases:
                if modifier.kind == ast.GateModifierName.pow:
                    self.__emit_principal_phase(angle, coefficient)
                    self.__qcx_lines.append(f'LET {angle} *= {factor!r}')
                else:
                    self.__qcx_lines.append(f'LET {angle} *= -1.0')
                    if preserve_inverse_coefficients:
                        # General eigenvalues are constructed algebraically,
                        # not from this angle. Conjugate them directly: EXP of
                        # a rounded argument could move an exact cut eigenvalue
                        # to one side, changing the next principal power.
                        self.__emit_conjugate_coefficient(coefficient)
                        continue
                self.__emit_phase_coefficient(angle, coefficient)

    def __emit_complex_components(self, destination: str, real: str, imaginary: str) -> None:
        # The scalar components must survive overwriting destination.
        self.__qcx_lines.extend([
            f'LET {destination} := :I', f'LET {destination} *= :COMPLEX:{imaginary}',
            f'LET {destination} += :COMPLEX:{real}'])

    def __emit_conjugate_coefficient(self, coefficient: str) -> None:
        real = self.__add_new_temporary_variable(ValueType.FLOAT)
        imaginary = self.__add_new_temporary_variable(ValueType.FLOAT)
        try:
            self.__qcx_lines.extend([
                f'LET {real} := :REAL:{coefficient}', f'LET {imaginary} := :IMAG:{coefficient}',
                f'LET {imaginary} *= -1.0'])
            self.__emit_complex_components(coefficient, real, imaginary)
        finally:
            self.__release_temporary_variable(real)
            self.__release_temporary_variable(imaginary)

    def __emit_u_gate_quaternion(
            self, name: str, captured: list[tuple[str, str]]) -> tuple[str, tuple[str, str, str], str]:
        # All original angles have already been captured and checked. Form
        # half-angle unit factors separately, never sums/differences of large
        # source angles. W = overall * (scalar I - i vector . Pauli).
        for angle, coefficient in captured:
            self.__qcx_lines.append(f'LET {angle} *= 0.5')
            self.__emit_phase_coefficient(angle, coefficient)
        phi, lam = (coefficient for _, coefficient in captured[-2:])
        summed = self.__add_new_temporary_variable(ValueType.COMPLEX)
        difference = self.__add_new_temporary_variable(ValueType.COMPLEX)
        self.__qcx_lines.extend([f'LET {summed} := {phi}', f'LET {summed} *= {lam}',
                                 f'LET {difference} := {lam}'])
        self.__emit_conjugate_coefficient(difference)
        self.__qcx_lines.append(f'LET {difference} *= {phi}')

        cosine = self.__add_new_temporary_variable(ValueType.FLOAT)
        sine = self.__add_new_temporary_variable(ValueType.FLOAT)
        if name == 'u2':
            # Native U2 has identical sine/cosine = 1/sqrt(2). ABS supplies
            # sqrt(2) in native precision without needing a new SQRT function.
            diagonal = self.__add_new_temporary_variable(ValueType.COMPLEX)
            self.__qcx_lines.extend([
                f'LET {diagonal} := :I', f'LET {diagonal} += :COMPLEX:1.0',
                f'APPLY {sine} <- ABS {diagonal}', f'LET {cosine} := 1.0',
                f'LET {cosine} /= {sine}', f'LET {sine} := {cosine}'])
        else:
            theta = captured[0][1]
            self.__qcx_lines.extend([
                f'LET {cosine} := :REAL:{theta}', f'LET {sine} := :IMAG:{theta}'])

        scalar, x, y, z = (self.__add_new_temporary_variable(ValueType.FLOAT) for _ in range(4))
        self.__qcx_lines.extend([
            f'LET {scalar} := :REAL:{summed}', f'LET {scalar} *= {cosine}',
            f'LET {x} := :IMAG:{difference}', f'LET {x} *= {sine}', f'LET {x} *= -1.0',
            f'LET {y} := :REAL:{difference}', f'LET {y} *= {sine}',
            f'LET {z} := :IMAG:{summed}', f'LET {z} *= {cosine}'])
        if name == 'U':
            self.__qcx_lines.append(f'LET {summed} *= {captured[0][1]}')
        return scalar, (x, y, z), summed

    def __emit_u_gate_eigenbasis(
            self, scalar: str, vector: tuple[str, str, str], overall: str
            ) -> tuple[str, str, tuple[str, str]]:
        x, y, z = vector
        xy = self.__add_new_temporary_variable(ValueType.COMPLEX)
        zr = self.__add_new_temporary_variable(ValueType.COMPLEX)
        radial = self.__add_new_temporary_variable(ValueType.FLOAT)
        length = self.__add_new_temporary_variable(ValueType.FLOAT)
        beta = self.__add_new_temporary_variable(ValueType.FLOAT)
        eta = self.__add_new_temporary_variable(ValueType.FLOAT)
        self.__emit_complex_components(xy, x, y)
        self.__qcx_lines.append(f'APPLY {radial} <- ABS {xy}')
        self.__emit_complex_components(zr, z, radial)
        self.__qcx_lines.append(f'APPLY {length} <- ABS {zr}')
        ready = self.__gate_power_label() + '_U_BASIS'
        polar = self.__gate_power_label() + '_U_POLAR'
        self.__qcx_lines.extend([
            f'LET {beta} := 0.0', f'LET {eta} := 0.0',
            f'JUMPIF {ready} {length} == 0.0', f'JUMPIF {polar} {radial} == 0.0',
            f'APPLY {eta} <- ARG {xy}', f'@{polar}', f'APPLY {beta} <- ARG {zr}', f'@{ready}'])
        # Exactly scalar => identity basis and equal full eigenvalues. Polar
        # axes skip only the undefined azimuth. No epsilon-based merging and
        # no division by the eigenvalue gap or tiny vector components.
        plus = self.__add_new_temporary_variable(ValueType.COMPLEX)
        minus = self.__add_new_temporary_variable(ValueType.COMPLEX)
        self.__emit_complex_components(minus, scalar, length)
        self.__qcx_lines.append(f'LET {length} *= -1.0')
        self.__emit_complex_components(plus, scalar, length)
        self.__qcx_lines.extend([f'LET {plus} *= {overall}', f'LET {minus} *= {overall}'])
        return beta, eta, (plus, minus)

    @contextlib.contextmanager
    def __powered_u_gate_spectrum(
            self, statement: ast.QuantumGate, info: _OrderedGateModifierInfo
            ) -> Iterator[tuple[str, str, str, str]]:
        factors = self.__phase_power_factors(info)
        name = statement.name.name
        if name not in ('u2', 'u3', 'U', 'cu'):
            raise UnsupportedOpenQASMError(f'U-family spectrum for {name}')
        expected = 2 if name == 'u2' else 4 if name == 'cu' else 3
        if len(statement.arguments) != expected:
            qubits = (2 if name == 'cu' else 1) + sum(count for _, count in info.controls)
            raise WrongGateArityException(name, expected, qubits, len(statement.arguments), len(statement.qubits))
        with contextlib.ExitStack() as stack:
            # Nested capture scopes retain every source snapshot and indexed
            # expression temporary until the full modifier/broadcast lifetime,
            # restoring the caller's expression state even on exceptions.
            captured = [stack.enter_context(self.__evaluated_phase_angle(expression))
                        for expression in statement.arguments]
            core_name, core_parameters = ('U', captured[:3]) if name == 'cu' else (name, captured)
            scalar, vector, overall = self.__emit_u_gate_quaternion(core_name, core_parameters)
            beta, eta, coefficients = self.__emit_u_gate_eigenbasis(scalar, vector, overall)
            if name == 'cu':
                # Gamma's ORIGINAL full-angle factor belongs to both active
                # eigenvalues before any principal argument or modifier. Do
                # not halve it, mix it into phi/lambda, or add gamma+theta/2.
                gamma = captured[3][1]
                self.__qcx_lines.extend(f'LET {coefficient} *= {gamma}' for coefficient in coefficients)
            phases = tuple((self.__add_new_temporary_variable(ValueType.FLOAT), coefficient)
                           for coefficient in coefficients)
            for angle, coefficient in phases:
                self.__emit_principal_phase(angle, coefficient)
            self.__emit_phase_modifiers(phases, info, factors, preserve_inverse_coefficients=True)
            yield beta, eta, phases[0][0], phases[1][0]

    @contextlib.contextmanager
    def __powered_phase_angle(
            self, expression: ast.Expression, info: _OrderedGateModifierInfo) -> Iterator[str]:
        if ((isinstance(expression, ast.Cast) and isinstance(expression.type, ast.AngleType))
                or self.__angle_expression_type(expression) is not None):
            with self.__powered_angle_phase(expression, info) as angle:
                yield angle
            return
        factors = self.__phase_power_factors(info)
        with self.__evaluated_phase_angle(expression) as (angle, coefficient):
            self.__emit_phase_modifiers(((angle, coefficient),), info, factors)
            yield angle

    @contextlib.contextmanager
    def __powered_rotation_angles(
            self, expression: ast.Expression, info: _OrderedGateModifierInfo) -> Iterator[tuple[str, str]]:
        factors = self.__phase_power_factors(info)
        # The shared context snapshots and checks theta before it is halved,
        # preserving the source and the caller's expression/temporary state.
        with self.__evaluated_phase_angle(expression) as (plus_angle, plus_coefficient):
            minus_angle = self.__add_new_temporary_variable(ValueType.FLOAT)
            minus_coefficient = self.__add_new_temporary_variable(ValueType.COMPLEX)
            # R_P(theta) has phases -theta/2 on P=+1 and +theta/2 on P=-1.
            # Form the second initial phase before modifying the first, and keep
            # both independently thereafter. No phase difference is necessary.
            self.__qcx_lines.extend([
                f'LET {plus_angle} *= -0.5',
                f'LET {minus_angle} := {plus_angle}', f'LET {minus_angle} *= -1.0'])
            phases = ((plus_angle, plus_coefficient), (minus_angle, minus_coefficient))
            for angle, coefficient in phases:
                self.__emit_phase_coefficient(angle, coefficient)
            self.__emit_phase_modifiers(phases, info, factors)
            yield plus_angle, minus_angle

    def __emit_parameterized_phase_gate(
            self, statement: ast.QuantumGate, info: _OrderedGateModifierInfo, mnemonic: str) -> None:
        added_controls = sum(count for _, count in info.controls)
        output_start = len(self.__qcx_lines)
        with self.__powered_phase_angle(statement.arguments[0], info) as angle:
            for qubits in self.__qubit_operand_tuples(statement.qubits, distinct=True):
                conditions = self.__condition_qubits(qubits[:added_controls], info.controls)
                gate = self.__controlled_gate_name(mnemonic, added_controls) if added_controls else mnemonic
                operands = conditions + qubits[added_controls:]
                self.__emit_gate_power([f'{gate} {" ".join(operands)} {angle}'], 1, output_start)

    def __emit_floating_rotation_gate(
            self, statement: ast.QuantumGate, info: _OrderedGateModifierInfo, mnemonic: str) -> None:
        root, intrinsic_controls = self.qcx_controlled_gate_families[mnemonic]
        added_controls = sum(count for _, count in info.controls)
        output_start = len(self.__qcx_lines)
        with self.__powered_rotation_angles(statement.arguments[0], info) as (plus_angle, minus_angle):
            for qubits in self.__qubit_operand_tuples(statement.qubits, distinct=True):
                conditions = self.__condition_qubits(qubits[:added_controls], info.controls)
                conditions += qubits[added_controls:added_controls + intrinsic_controls]
                target = qubits[-1]
                before, after = [], []
                if root == 'EX':
                    before, after = [f'H {target}'], [f'H {target}']
                elif root == 'EY':
                    before, after = [f'S+ {target}', f'H {target}'], [f'H {target}', f'S {target}']
                # Phase each projector independently, rather than subtracting
                # the eigenangles (which can overflow) or discarding a common
                # phase. Controls condition both phases; inactive basis changes
                # cancel. The final condition is the target's basis value.
                diagonal = self.__controlled_phase_gate_name(len(conditions) + 1)
                zero = ' '.join(conditions + [f'{target}#0'])
                one = ' '.join(conditions + [target])
                sequence = before + [f'{diagonal} {zero} {plus_angle}',
                                     f'{diagonal} {one} {minus_angle}'] + after
                self.__emit_gate_power(sequence, 1, output_start)

    def __emit_floating_u_gate(self, statement: ast.QuantumGate, info: _OrderedGateModifierInfo) -> None:
        added_controls = sum(count for _, count in info.controls)
        intrinsic_controls = 1 if statement.name.name == 'cu' else 0
        output_start = len(self.__qcx_lines)
        with self.__powered_u_gate_spectrum(statement, info) as (beta, eta, plus_angle, minus_angle):
            for qubits in self.__qubit_operand_tuples(statement.qubits, distinct=True):
                conditions = self.__condition_qubits(qubits[:added_controls], info.controls)
                conditions += qubits[added_controls:added_controls + intrinsic_controls]
                target = qubits[-1]
                diagonal = self.__controlled_phase_gate_name(len(conditions) + 1)
                zero = ' '.join(conditions + [f'{target}#0'])
                one = ' '.join(conditions + [target])
                # V diag(exp(i alpha+), exp(i alpha-)) V-dagger. Controls
                # condition both full eigenphases, not the basis changes.
                # QCX U3 is conventional W even when the source gate is U;
                # U's extra global factor (and cu's full gamma factor) is
                # already in both eigenvalues. cu's intrinsic control stays
                # positive independently of added ctrl/negctrl modifiers.
                sequence = [f'U3+ {target} {beta} {eta} 0.0',
                            f'{diagonal} {zero} {plus_angle}', f'{diagonal} {one} {minus_angle}',
                            f'U3 {target} {beta} {eta} 0.0']
                self.__emit_gate_power(sequence, 1, output_start)

    def __emit_floating_quantum_phase(
            self, statement: ast.QuantumPhase, info: _OrderedGateModifierInfo) -> None:
        controls = sum(count for _, count in info.controls)
        if len(statement.qubits) != controls:
            raise WrongGateArityException('gphase', 1, controls, 1, len(statement.qubits))
        output_start = len(self.__qcx_lines)
        with self.__powered_phase_angle(statement.argument, info) as angle:
            if not controls:
                self.__emit_gate_power([f'PHASE {angle}'], 1, output_start)
            else:
                mnemonic = self.__controlled_phase_gate_name(controls)
                for qubits in self.__qubit_operand_tuples(statement.qubits, distinct=True):
                    conditions = self.__condition_qubits(qubits, info.controls)
                    self.__emit_gate_power([f'{mnemonic} {" ".join(conditions)} {angle}'], 1, output_start)

    def __emit_floating_gate_power(
            self, statement: ast.QuantumGate, info: _OrderedGateModifierInfo, mnemonic: str) -> None:
        phase = self.__floating_gate_power_phase(mnemonic, info)
        root, intrinsic_controls = self.qcx_controlled_gate_families[mnemonic]
        added_controls = sum(count for _, count in info.controls)
        output_start = len(self.__qcx_lines)
        # Multiply exactly by the host pi approximation before the one float
        # rounding. This also avoids prematurely underflowing tiny pi fractions.
        angle = repr(float(phase * Fraction(math.pi)))
        for qubits in self.__qubit_operand_tuples(statement.qubits, distinct=True):
            conditions = self.__condition_qubits(qubits[:added_controls], info.controls)
            conditions += qubits[added_controls:added_controls + intrinsic_controls]
            targets = qubits[added_controls + intrinsic_controls:]
            sequence = []
            if phase:
                target = targets[0]
                before, after = [], []
                if root in ('X', 'SX'):
                    before, after = [f'H {target}'], [f'H {target}']
                elif root == 'Y':
                    before, after = [f'S+ {target}', f'H {target}'], [f'H {target}', f'S {target}']
                elif root == 'H':
                    before = [f'EY {target} {math.pi / 8}']
                    after = [f'EY {target} {-math.pi / 8}']
                elif root == 'SWAP':
                    other = targets[1]
                    before = [f'CX {target} {other}', f'H {target}']
                    after = [f'H {target}', f'CX {target} {other}']
                    conditions += [target]
                    target = other
                diagonal = self.__controlled_gate_name('U1', len(conditions)) if conditions else 'U1'
                # Only the diagonal phase is controlled. When inactive, the
                # unconditional basis changes cancel, including for SWAP.
                sequence = before + [f'{diagonal} {" ".join(conditions + [target])} {angle}'] + after
            # Even identity powers above must validate operands/broadcasting.
            self.__emit_gate_power(sequence, 1, output_start)

    def __lower_gate_sequence(
            self, sequence: list[tuple[str, list[str]]], inverse: bool,
            conditions: list[str], inverse_phase: str | None) -> list[str]:
        lowered = []
        for mnemonic, arguments in reversed(sequence) if inverse else sequence:
            if inverse:
                if mnemonic == 'PHASE':
                    if inverse_phase is None:
                        raise UninitializedValueException
                    arguments = [inverse_phase]
                else:
                    mnemonic = self.__adjoint_gate_name(mnemonic)
            if conditions:
                mnemonic = (self.__controlled_phase_gate_name(len(conditions)) if mnemonic == 'PHASE'
                            else self.__controlled_gate_name(mnemonic, len(conditions)))
                arguments = conditions + arguments
            lowered.append(f'{mnemonic} {" ".join(arguments)}')
        return lowered

    def __prepare_runtime_power_counts(
            self, runtime: _RuntimeGatePower, repetitions: int, period: int | None) -> tuple[list[str], int]:
        magnitudes = list(runtime.magnitudes)
        if period is not None:
            reduced = self.__add_new_temporary_variable(ValueType.UINT)
            factor = self.__add_new_temporary_variable(ValueType.UINT)
            self.__qcx_lines.append(f'LET {reduced} := 1')
            for magnitude in magnitudes:
                # Known periods are 1, 2, 4, or 8, so a mask gives the residue.
                # Reduce both factors first: their product is at most 7*7.
                self.__qcx_lines.extend([
                    f'LET {factor} := {magnitude}', f'LET {factor} &= {period - 1}',
                    f'LET {reduced} *= {factor}', f'LET {reduced} &= {period - 1}'])
            magnitudes = [reduced]
        elif repetitions > 1:
            # Keep source constant factors separate too: their Python
            # product may exceed UINT even when each factor fits.
            for value in runtime.constant_factors:
                value = abs(value)
                if value <= 1:
                    continue
                if value > self.QCX_UINT_MAX:
                    raise UnsupportedOpenQASMError('runtime pow constant factor exceeds native UINT range')
                magnitude = self.__add_new_temporary_variable(ValueType.UINT)
                self.__qcx_lines.append(f'LET {magnitude} := {value}')
                magnitudes.append(magnitude)
            repetitions = 1
        return magnitudes, repetitions

    def __emit_runtime_gate_power(
            self, forward: list[str], adjoint: list[str], repetitions: int,
            runtime: _RuntimeGatePower, output_start: int | None, period: int | None = None) -> None:
        if repetitions == 0 or not forward:
            return
        previous_used = self.__used_temporary_variables.copy()
        prefix = self.__gate_power_label()
        try:
            magnitudes, repetitions = self.__prepare_runtime_power_counts(runtime, repetitions, period)
            # Declare every counter before the first runtime jump. Inner
            # counters reset inside outer loops, but declarations must not.
            counters = [self.__add_new_temporary_variable(ValueType.UINT) for _ in magnitudes]
            self.__qcx_lines.extend(f'JUMPIF {prefix}_DONE {value} == 0' for value in magnitudes)
            for depth, (counter, magnitude) in enumerate(zip(counters, magnitudes)):
                self.__qcx_lines.extend([
                    f'LET {counter} := {magnitude}', f'@{prefix}_{depth}_LOOP',
                    f'JUMPIF {prefix}_{depth}_END {counter} == 0'])
            if runtime.signed_exponent is not None:
                self.__qcx_lines.append(f'JUMPIF {prefix}_INVERSE {runtime.signed_exponent} < 0')
            self.__emit_gate_power(forward, repetitions, output_start)
            if runtime.signed_exponent is not None:
                self.__qcx_lines.extend([f'JUMP {prefix}_NEXT', f'@{prefix}_INVERSE'])
                self.__emit_gate_power(adjoint, repetitions, output_start)
                self.__qcx_lines.append(f'@{prefix}_NEXT')
            tail = []
            for depth in reversed(range(len(counters))):
                tail.extend([f'LET {counters[depth]} -= 1', f'JUMP {prefix}_{depth}_LOOP', f'@{prefix}_{depth}_END'])
            tail.append(f'@{prefix}_DONE')
            self.__check_gate_power_output(len(tail), output_start)
            self.__qcx_lines.extend(tail)
        finally:
            for temporary in self.__used_temporary_variables - previous_used:
                self.__release_temporary_variable(temporary)

    @classmethod
    def __gate_power_plan(cls, mnemonic: str, repetitions: int) -> tuple[list[str], int]:
        if repetitions == 0:
            return [], 0
        root, controls = cls.qcx_controlled_gate_families[mnemonic]
        inverse_root = root.endswith('+')
        cycle = cls.qcx_gate_power_cycles.get(root[:-1] if inverse_root else root)
        if cycle is None:
            return [mnemonic], repetitions
        reduced = cycle[(-repetitions if inverse_root else repetitions) % len(cycle)]
        if controls:
            reduced = tuple(mnemonic if name == root else cls.__controlled_gate_name(name, controls)
                            for name in reduced)
        return list(reduced), 1

    @staticmethod
    def __scaled_phase_operand(operand: str, value_kind: ValueKind, multiplier: int) -> str | None:
        # Scaling is only an optimization. Runtime angles stay on the bounded
        # repetition path: scaling an unknown value could introduce overflow
        # even though every original angle is finite.
        if value_kind != ValueKind.LITERAL:
            return None
        try:
            coefficient = float(multiplier)
            if not math.isfinite(coefficient) or coefficient != multiplier:
                return None
            angle = float(operand)
            # Do not narrow an integer literal that a higher-precision bra
            # backend could otherwise read directly.
            try:
                integer = int(operand)
            except ValueError:
                integer = None
            if integer is not None and angle != integer:
                return None
            result = angle * coefficient
            if not math.isfinite(result):
                return None
            numerator, denominator = angle.as_integer_ratio()
            product_numerator, product_denominator = result.as_integer_ratio()
            # A rounded product of a large angle can change its phase by an
            # arbitrary amount modulo 2*pi. Require an exact binary product.
            if product_numerator * denominator != numerator * multiplier * product_denominator:
                return None
            return str(result)
        except (OverflowError, ValueError):
            return None

    def __check_gate_power_output(self, count: int, output_start: int | None) -> None:
        if (output_start is not None
                and count + len(self.__qcx_lines) - output_start > self.MAX_GATE_POWER_OUTPUT_LINES):
            raise UnsupportedOpenQASMError(
                f'gate power expansion exceeds {self.MAX_GATE_POWER_OUTPUT_LINES} QCX lines')
        if self.__inside_expanded_loop() and count + len(self.__qcx_lines) > self.MAX_LOOP_OUTPUT_LINES:
            raise InvalidLoopRangeException(
                f'For-loop output exceeds {self.MAX_LOOP_OUTPUT_LINES} QCX lines')

    def __emit_gate_power(self, sequence: list[str], repetitions: int, output_start: int | None) -> None:
        # Check before allocating a repeated list or entering a repetition
        # loop. Include all broadcast elements in the same expansion budget.
        count = len(sequence) * repetitions
        self.__check_gate_power_output(count, output_start)
        if count:
            self.__qcx_lines.extend(sequence * repetitions)

    def __negated_phase_operand(self, operand: str, value_kind: ValueKind) -> str:
        if value_kind == ValueKind.LITERAL:
            return str(-float(operand))
        # Never mutate the source expression or negate a signed INT directly.
        temporary = self.__add_new_temporary_variable(ValueType.FLOAT)
        self.__qcx_lines.extend([f'LET {temporary} := {operand}', f'LET {temporary} *= -1.0'])
        return temporary

    def visit_QuantumPhase(self, statement: ast.QuantumPhase) -> None:
        if self.__is_initialization_process:
            return
        previous_used = self.__used_temporary_variables.copy()
        previous_kind = self.__expression_kind
        try:
            self.__emit_quantum_phase(statement)
        finally:
            self.__expression_kind = previous_kind
            for temporary in self.__used_temporary_variables - previous_used:
                self.__release_temporary_variable(temporary)

    def __emit_quantum_phase(self, statement: ast.QuantumPhase) -> None:
        self.__angle_phase_parameter(statement)
        info = self.__analyze_gate_modifiers(statement.modifiers, allow_runtime=True, allow_float=True)
        if isinstance(info, _OrderedGateModifierInfo):
            self.__emit_floating_quantum_phase(statement, info)
            return
        with self.__runtime_gate_power(info) as runtime:
            self.__emit_quantum_phase_operation(statement, info, runtime)

    def __emit_quantum_phase_operation(
            self, statement: ast.QuantumPhase, info: _GateModifierInfo, runtime: _RuntimeGatePower | None) -> None:
        inverse, control_groups = info.power < 0, info.controls
        controls = sum(count for _, count in control_groups)
        if not controls and statement.qubits:
            raise UnsupportedOpenQASMError('qubit operands on gphase')
        if controls and len(statement.qubits) != controls:
            raise WrongGateArityException('gphase', 1, controls, 1, len(statement.qubits))

        angle_parameter = self.__angle_phase_parameter(statement)
        if angle_parameter is not None:
            output_start = (len(self.__qcx_lines) if any(
                modifier.modifier == ast.GateModifierName.pow for modifier in statement.modifiers) else None)
            constant = self.__constant_angle_gate_value(angle_parameter)
            if constant is not None and runtime is None:
                powered = _ConstantAngle(constant.angle_type,
                                         (constant.value * abs(info.power)) & constant.angle_type.mask)
                self.__emit_constant_angle_global_phase(statement, powered, info, output_start)
            else:
                self.__emit_runtime_angle_global_phase(statement, angle_parameter, info, runtime, output_start)
            return

        self.__expression_kind = ExpressionKind.ARITHMETIC
        self.visit(statement.argument)
        self.__expression_kind = None

        if self.__value_type in (ValueType.BIT, ValueType.BOOL, ValueType.COMPLEX):
            raise WrongParameterTypeException(self.__value)

        phase = self.__angle_operand(
            (self.__value, self.__value_type, self.__value_kind))
        if inverse:
            phase = self.__negated_phase_operand(phase, self.__value_kind)
        powered = any(modifier.modifier == ast.GateModifierName.pow for modifier in statement.modifiers)
        repetitions = abs(info.power)
        if powered and repetitions > 1:
            scaled = self.__scaled_phase_operand(phase, self.__value_kind, repetitions)
            if scaled is not None:
                phase, repetitions = scaled, 1
        opposite_phase = (self.__negated_phase_operand(phase, self.__value_kind)
                          if runtime is not None and runtime.signed_exponent is not None else phase)
        output_start = len(self.__qcx_lines) if powered else None
        if controls:
            mnemonic = self.__controlled_phase_gate_name(controls)
            for qubits in self.__qubit_operand_tuples(statement.qubits, distinct=True):
                conditions = self.__condition_qubits(qubits, control_groups)
                forward = [f'{mnemonic} {" ".join(conditions)} {phase}']
                if runtime is None:
                    self.__emit_gate_power(forward, repetitions, output_start)
                else:
                    self.__emit_runtime_gate_power(forward, [f'{mnemonic} {" ".join(conditions)} {opposite_phase}'],
                                                   repetitions, runtime, output_start)
        else:
            if runtime is None:
                self.__emit_gate_power([f'PHASE {phase}'], repetitions, output_start)
            else:
                self.__emit_runtime_gate_power([f'PHASE {phase}'], [f'PHASE {opposite_phase}'],
                                               repetitions, runtime, output_start)
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
            if source_name in self.__const_bit_register_values:
                return self.__constant_bit_element(expression), ValueType.BIT, ValueKind.LITERAL
            variable_name = self.__capitalize_variable_name(source_name)
            if variable_name in self.__array_element_types:
                operand, index = self.__array_access(expression)
                element_type = self.__array_element_types[variable_name]
                uint_type = self.__uint_variable_types.get(variable_name) if element_type == ValueType.UINT else None
                if uint_type is not None:
                    self.__uint_expression_types[id(expression)] = (expression, uint_type)
                if index is None:
                    return operand, element_type, ValueKind.LVALUE
                try:
                    result = self.__add_new_temporary_variable(element_type, uint_type)
                    self.__qcx_lines.append(f'LET {result} := {operand}')
                    return result, element_type, ValueKind.TEMPORARY
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

    def __register_comparison_width(
            self, expression: ast.BinaryExpression,
            iterators: set[str] | dict[str, ValueType] | None = None) -> int | None:
        # Scalar comparisons continue through their existing promotion rules.
        # None is a scalar operand, never a synonym for a one-bit register.
        lhs = self.__bit_register_expression_width(expression.lhs, iterators)
        rhs = self.__bit_register_expression_width(expression.rhs, iterators)
        if lhs is None and rhs is None:
            return None
        if lhs is None or rhs is None:
            raise InvalidBitOperandException('Comparison operands must both be scalars or both be registers')
        if expression.op.name not in ('==', '!='):
            raise UnsupportedOpenQASMError('ordered comparisons of bit-register values')
        if lhs != rhs:
            raise InvalidBitOperandException(f'Comparison register widths {lhs} and {rhs} do not match')
        return lhs

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

        if self.__register_comparison_width(condition) is not None:
            self.__emit_register_equality_condition(condition, true_label, false_label)
            return
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

    def __emit_register_equality_condition(
            self, expression: ast.BinaryExpression, true_label: str, false_label: str) -> None:
        with contextlib.ExitStack() as operands:
            lhs = operands.enter_context(self.__bit_register_expression_value(expression.lhs))
            rhs = operands.enter_context(self.__bit_register_expression_value(expression.rhs))
            # Both complete operands must be evaluated before any early exit
            # on a mismatching element. Surrounding logical operators, not
            # equality itself, may skip register-expression evaluation.
            mismatch = false_label if expression.op.name == '==' else true_label
            match = true_label if expression.op.name == '==' else false_label
            for left, right in zip(lhs, rhs):
                if isinstance(left, int) and isinstance(right, int):
                    if left != right:
                        self.__qcx_lines.append(f'JUMP {mismatch}')
                        return
                    continue
                if isinstance(left, int):
                    left, right = right, left  # JUMPIF requires a variable LHS.
                self.__qcx_lines.append(f'JUMPIF {mismatch} {left} \\= {right}')
            self.__qcx_lines.append(f'JUMP {match}')

    def __constant_bit_register_value(self, expression: ast.Expression, width: int) -> int:
        if isinstance(expression, ast.BitstringLiteral):
            return expression.value if self.__evaluate_constant else 0
        if isinstance(expression, ast.Identifier) and expression.name in self.__const_bit_register_values:
            constant = self.__constant_bit_register(expression.name)
            return constant.value if self.__evaluate_constant else 0
        if (isinstance(expression, ast.IndexExpression)
                and isinstance(expression.collection, ast.Identifier)
                and expression.collection.name in self.__const_bit_register_values):
            selection = self.__static_bit_selection(expression)
            if not selection.is_register:
                raise InvalidBitOperandException('A register expression requires a register selection')
            constant = self.__constant_bit_register(selection.source_name)
            return (sum(((constant.value >> source) & 1) << target
                        for target, source in enumerate(selection.indices))
                    if self.__evaluate_constant else 0)
        if isinstance(expression, ast.FunctionCall) and expression.name.name in ('rotl', 'rotr'):
            self.__function_call_info(expression)
            value = self.__constant_bit_register_value(expression.arguments[0], width)
            distance = self.__constant_rotation_distance(expression.arguments[1])
            return (self.__fold_rotation(value, width, distance, expression.name.name)
                    if self.__evaluate_constant else 0)
        if isinstance(expression, ast.UnaryExpression) and expression.op.name == '~':
            value = self.__constant_bit_register_value(expression.expression, width)
            return value ^ ((1 << width) - 1) if self.__evaluate_constant else 0
        if isinstance(expression, ast.BinaryExpression):
            lhs = self.__constant_bit_register_value(expression.lhs, width)
            if expression.op.name in ('<<', '>>'):
                self.visit(expression.rhs)
                if (self.__value_kind != ValueKind.LITERAL
                        or self.__value_type not in (ValueType.INT, ValueType.UINT)):
                    raise NoConstantExpressionException
                return (self.__fold_bit_register_shift(lhs, width, int(self.__value), expression.op.name)
                        if self.__evaluate_constant else 0)
            rhs = self.__constant_bit_register_value(expression.rhs, width)
            if expression.op.name == '&':
                return lhs & rhs
            if expression.op.name == '|':
                return lhs | rhs
            if expression.op.name == '^':
                return lhs ^ rhs
        # Mutable registers and their selections cannot supply constant values.
        raise NoConstantExpressionException

    def __emit_condition(
            self, condition: ast.Expression, true_label: str,
            false_label: str) -> None:
        previous_used = self.__used_temporary_variables.copy()
        try:
            self.__emit_condition_body(condition, true_label, false_label)
        except Exception:
            # A completed operand may still be reserved when another fails.
            # This covers condition values and direct if/while conditions.
            for temporary in self.__used_temporary_variables - previous_used:
                self.__release_temporary_variable(temporary)
            raise

    def __emit_condition_body(
            self, condition: ast.Expression, true_label: str,
            false_label: str) -> None:
        if isinstance(condition, ast.Cast) and isinstance(condition.type, ast.BoolType):
            value, value_type, value_kind = self.__condition_operand(condition.argument)
            if value_type not in (ValueType.BOOL, ValueType.BIT, ValueType.INT, ValueType.UINT, ValueType.FLOAT,
                                   ValueType.ANGLE):
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
        try:
            self.__emit_condition(expression, true_label, false_label)
        except Exception:
            self.__release_temporary_variable(result)
            raise
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
            if expression.op.name in ('==', '!=', '<', '<=', '>', '>='):
                width = self.__register_comparison_width(expression)
                if width is not None:
                    lhs = self.__constant_bit_register_value(expression.lhs, width)
                    rhs = self.__constant_bit_register_value(expression.rhs, width)
                    equal = lhs == rhs
                    self.__value = int(equal if expression.op.name == '==' else not equal) if self.__evaluate_constant else 0
                    self.__value_type, self.__value_kind = ValueType.BOOL, ValueKind.LITERAL
                    return
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
                ((ast.BoolType,), ValueType.BOOL), ((ast.BitType,), ValueType.BIT),
                ((ast.AngleType,), ValueType.ANGLE)):
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
        if isinstance(expression, ast.FunctionCall):
            signature, runtime = self.__function_call_info(expression, iterators)
            return signature.result_type, runtime
        if isinstance(expression, ast.UnaryExpression):
            return self.__loop_unary_info(expression, iterators)
        if isinstance(expression, ast.BinaryExpression):
            return self.__loop_binary_info(expression, iterators)
        if isinstance(expression, ast.IndexExpression) and isinstance(expression.collection, ast.Identifier):
            return self.__loop_index_info(expression, iterators)
        raise UnsupportedOpenQASMError(f'loop expression {type(expression).__name__}')

    @classmethod
    def __builtin_function_overloads(cls, name: str, arity: int) -> tuple[_BuiltinFunctionSignature, ...]:
        overloads = cls.builtin_function_signatures.get(name)
        if overloads is None:
            raise UnsupportedOpenQASMError(f'function {name}')
        matching = tuple(signature for signature in overloads if len(signature.operand_types) == arity)
        if not matching:
            expected = ', '.join(str(count) for count in sorted({len(signature.operand_types) for signature in overloads}))
            raise InvalidFunctionCallException(f'Function {name} expects {expected} argument(s), but got {arity}')
        return matching

    @staticmethod
    def __function_operand_promotes(source: ValueType, target: ValueType) -> bool:
        # Standard classical types use C99-style promotion. A scalar bit is
        # a special OpenQASM type, not an implicit integer/Boolean here.
        if source == target:
            return True
        if target == ValueType.INT:
            # Integer inputs select the signed overload before a real one.
            # UINT conversion is checked during evaluation, not overload lookup.
            return source in (ValueType.BOOL, ValueType.UINT)
        if source in (ValueType.BOOL, ValueType.INT, ValueType.UINT):
            return target in (ValueType.FLOAT, ValueType.COMPLEX)
        return source == ValueType.FLOAT and target == ValueType.COMPLEX

    @classmethod
    def __resolve_builtin_function(cls, name: str, operands: tuple[ValueType, ...]) -> _BuiltinFunctionSignature:
        for signature in cls.__builtin_function_overloads(name, len(operands)):
            if all(cls.__function_operand_promotes(source, target)
                   for source, target in zip(operands, signature.operand_types)):
                return signature
        types = ', '.join(value_type.name.lower() for value_type in operands)
        raise InvalidFunctionCallException(f'Function {name} has no matching overload for ({types})')

    def __function_call_info(
            self, expression: ast.FunctionCall, iterators: set[str] | dict[str, ValueType] | None = None
            ) -> tuple[_BuiltinFunctionSignature, bool]:
        # Diagnose the function/arity before inspecting its arguments. This
        # path must not visit/evaluate operands or allocate scratch storage.
        name = expression.name.name
        overloads = self.__builtin_function_overloads(name, len(expression.arguments))
        if overloads[0].rotation:
            argument, distance = expression.arguments
            width = self.__bit_register_expression_width(argument, iterators)
            if width is not None:
                source_type = ValueType.BIT_REGISTER
                source_runtime = self.__register_expression_runtime(argument, iterators)
            else:
                source_type, uint_type, source_runtime = self.__integer_operand_info(argument, iterators)
                if source_type != ValueType.UINT or uint_type is None or not uint_type.is_sized:
                    raise InvalidFunctionCallException(f'Function {name} requires bit[n] or sized uint[n]')
            distance_type, distance_runtime = self.__loop_expression_info(distance, iterators)
            signature = self.__resolve_builtin_function(name, (source_type, distance_type))
            self.__validate_bitwise_indices(distance, iterators)
            return signature, source_runtime or distance_runtime
        if overloads[0].population_count:
            argument = expression.arguments[0]
            width = self.__bit_register_expression_width(argument, iterators)
            if width is None:
                source_type, uint_type, runtime = self.__integer_operand_info(argument, iterators)
                if source_type != ValueType.UINT or uint_type is None or not uint_type.is_sized:
                    raise InvalidFunctionCallException(
                        'Function popcount requires a bit-register operand or sized uint[n]')
                return self.__resolve_builtin_function(name, (ValueType.UINT,)), runtime
            if width > self.QCX_UINT_MAX:
                raise InvalidFunctionCallException('Function popcount operand width exceeds QCX UINT range')
            return overloads[0], self.__register_expression_runtime(argument, iterators)
        operands = [self.__loop_expression_info(argument, iterators) for argument in expression.arguments]
        signature = self.__resolve_builtin_function(name, tuple(value_type for value_type, _ in operands))
        return signature, any(runtime for _, runtime in operands)

    def __loop_identifier_info(
            self, expression: ast.Identifier, iterators: set[str] | dict[str, ValueType] | None
            ) -> tuple[ValueType, bool]:
        name = expression.name
        if iterators is not None and name in iterators:
            value_type = iterators[name] if isinstance(iterators, dict) else ValueType.INT
            # Non-INT collection iterators always refer to runtime element
            # values, unlike possibly expanded signed iterators.
            return value_type, value_type != ValueType.INT
        for scope in reversed(self.__loop_bindings):
            if name in scope:
                binding = scope[name]
                return ((binding.value_type, True) if isinstance(binding, _RuntimeIteratorBinding)
                        else (ValueType.INT, False))
        if ((name in self.__const_bit_variable_values_map or name in self.__const_angle_values)
                and not self.__is_initialization_process
                and name not in self.__declared_constant_variables):
            raise NoVariableNameException(name)
        for constants, value_type in (
                (self.__const_int_variable_name_values_map, ValueType.INT),
                (self.__const_uint_variable_name_values_map, ValueType.UINT),
                (self.__const_float_variable_name_values_map, ValueType.FLOAT),
                (self.__const_complex_variable_name_values_map, ValueType.COMPLEX),
                (self.__const_bool_variable_values_map, ValueType.BOOL),
                (self.__const_bit_variable_values_map, ValueType.BIT),
                (self.__const_angle_values, ValueType.ANGLE)):
            if name in constants:
                return value_type, False
        if name in ('pi', 'tau', 'euler'):
            return ValueType.FLOAT, False
        if name in self.__const_bit_register_values:
            self.__constant_bit_register(name)
            raise UnsupportedOpenQASMError('whole constant bit register in scalar expression')
        if name not in self.__classical_source_types:
            raise NoVariableNameException(name)
        variable_type = self.__classical_source_types[name]
        if (isinstance(variable_type, ast.AngleType) and not self.__is_initialization_process
                and self.__capitalize_variable_name(name) not in self.__angle_variable_names):
            raise NoVariableNameException(name)
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
        if target_type == ValueType.ANGLE:
            self.__angle_type(expression.type, iterators)
            if operand_type not in (ValueType.FLOAT, ValueType.ANGLE):
                raise UnsupportedOpenQASMError('angle conversion requires float or angle input')
            return ValueType.ANGLE, runtime
        if operand_type == ValueType.ANGLE:
            if target_type != ValueType.BOOL:
                raise UnsupportedOpenQASMError('angle cast other than angle or bool')
            return ValueType.BOOL, runtime
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
        if expression.op.name == '-':
            return self.__negation_expression_info(expression, iterators)
        value_type, runtime = self.__loop_expression_info(expression.expression, iterators)
        if expression.op.name == '!':
            if value_type not in (ValueType.BIT, ValueType.BOOL):
                raise UnsupportedOpenQASMError('non-Boolean logical loop operand')
            return ValueType.BOOL, runtime
        if expression.op.name == '~':
            return self.__bitwise_type(value_type), runtime
        raise UnsupportedOpenQASMError('unsupported unary loop expression')

    def __loop_binary_info(
            self, expression: ast.BinaryExpression, iterators: set[str] | dict[str, ValueType] | None
            ) -> tuple[ValueType, bool]:
        if expression.op.name in ('*', '/') and self.__expression_mentions_angle(expression):
            info = self.__angle_multiplicative_info(expression, iterators)
            if info is not None:
                return info.result_type, info.runtime
        if expression.op.name in ('+', '-') and self.__expression_mentions_angle(expression):
            value_type, _, runtime = self.__angle_operand_info(expression, iterators)
            return value_type, runtime
        if expression.op.name == '**':
            value_type, _, runtime = self.__power_expression_info(expression, iterators)
            return value_type, runtime
        if (expression.op.name in ('==', '!=', '<', '<=', '>', '>=')
                and self.__register_comparison_width(expression, iterators) is not None):
            lhs_runtime = self.__register_expression_runtime(expression.lhs, iterators)
            rhs_runtime = self.__register_expression_runtime(expression.rhs, iterators)
            return ValueType.BOOL, lhs_runtime or rhs_runtime
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

    def __register_expression_runtime(
            self, expression: ast.Expression,
            iterators: set[str] | dict[str, ValueType] | None) -> bool:
        # Width/type validation has already established a register expression.
        # Inspect both sides without evaluating shift counts or reading bits.
        if isinstance(expression, ast.BitstringLiteral):
            return False
        if isinstance(expression, ast.Identifier):
            if expression.name in self.__const_bit_register_values:
                self.__constant_bit_register(expression.name)
                return False
            return True
        if isinstance(expression, ast.IndexExpression):
            if (isinstance(expression.collection, ast.Identifier)
                    and expression.collection.name in self.__const_bit_register_values):
                self.__static_bit_selection(expression, iterators)
                return False
            return True
        if isinstance(expression, ast.UnaryExpression):
            return self.__register_expression_runtime(expression.expression, iterators)
        if isinstance(expression, ast.FunctionCall):
            return self.__function_call_info(expression, iterators)[1]
        lhs = self.__register_expression_runtime(expression.lhs, iterators)
        rhs = (self.__loop_expression_info(expression.rhs, iterators)[1]
               if expression.op.name in ('<<', '>>')
               else self.__register_expression_runtime(expression.rhs, iterators))
        return lhs or rhs

    def __loop_index_info(
            self, expression: ast.IndexExpression, iterators: set[str] | dict[str, ValueType] | None
            ) -> tuple[ValueType, bool]:
        name = expression.collection.name
        if ((iterators is not None and name in iterators)
                or any(name in scope for scope in self.__loop_bindings)):
            raise UnsupportedOpenQASMError('indexed use of a for-loop iterator')
        if name in self.__const_bit_register_values:
            selection = self.__static_bit_selection(expression, iterators)
            if selection.is_register:
                raise UnsupportedOpenQASMError('register selection in scalar expression')
            return ValueType.BIT, False
        variable_type = self.__classical_source_types.get(name)
        if isinstance(variable_type, ast.ArrayType):
            return self.__source_value_type(variable_type.base_type), True
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
        typed_collection_loop = (
            isinstance(statement.set_declaration, ast.Identifier)
            and (isinstance(statement.type, (ast.FloatType, ast.UintType, ast.BoolType, ast.ComplexType))
                 or (isinstance(statement.type, ast.BitType) and statement.type.size is None)))
        if not isinstance(statement.type, ast.IntType) and not typed_collection_loop:
            raise UnsupportedOpenQASMError('for-loop iterator requires int or a supported collection type')
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
                ast.UnaryExpression, ast.BinaryExpression, ast.Cast, ast.IndexExpression, ast.FunctionCall)):
            raise UnsupportedOpenQASMError(f'loop expression {type(node).__name__}')
        if isinstance(node, ast.FunctionCall):
            self.__builtin_function_overloads(node.name.name, len(node.arguments))
        if isinstance(node, ast.BinaryExpression) and node.op.name not in (
                '+', '-', '*', '/', '%', '**', '&', '|', '^', '<<', '>>',
                '==', '!=', '<', '<=', '>', '>=', '&&', '||'):
            raise UnsupportedOpenQASMError(f'binary operator {node.op.name}')
        if isinstance(node, ast.UnaryExpression) and node.op.name not in ('-', '!', '~'):
            raise UnsupportedOpenQASMError(f'unary operator {node.op.name}')
        if isinstance(node, ast.Cast) and not isinstance(node.type, (
                ast.IntType, ast.UintType, ast.FloatType, ast.ComplexType, ast.BoolType, ast.BitType,
                ast.AngleType)):
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
        if isinstance(node, ast.FunctionCall):
            # A callee identifier is not a variable or iterator dependency.
            return set().union(*(self.__loop_bound_names(argument) for argument in node.arguments))
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
            self, statement: ast.ForInLoop, iterators: set[str] | dict[str, ValueType]
            ) -> _UnsignedIntegerType | None:
        # Outer iterator values may not exist for an empty outer loop. Validate
        # names and independent iteration values now; defer value-dependent checks until
        # an actual iteration, rather than fabricating an outer value.
        bounds = statement.set_declaration
        iterator_names = set(iterators)
        if isinstance(bounds, ast.Identifier):
            return self.__collection_loop_info(bounds, statement.type, iterators)[3]
        is_set = isinstance(bounds, ast.DiscreteSet)
        expressions = ([(value, f'element {index}') for index, value in enumerate(bounds.values)]
                       if is_set else [(bounds.start, 'start'), (bounds.end, 'end'),
                                       (bounds.step, 'step')])
        constants = (set(self.__const_int_variable_name_values_map)
                     | set(self.__const_uint_variable_name_values_map)
                     | set(self.__const_float_variable_name_values_map)
                     | set(self.__const_complex_variable_name_values_map)
                     | set(self.__const_bool_variable_values_map)
                     | set(self.__const_bit_variable_values_map)
                     | set(self.__const_angle_values))
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
            if isinstance(child, ast.ClassicalAssignment):
                source_type = self.__classical_source_types.get(self.__operand_name(child.lvalue))
                if isinstance(source_type, ast.ArrayType):
                    self.__validate_array_assignment_operator(
                        self.__source_value_type(source_type.base_type), child.op.name)
            if isinstance(child, ast.ClassicalAssignment) and child.op.name not in (
                    '=', '+=', '-=', '*=', '/=', '%=', '**=', '&=', '|=', '^=', '<<=', '>>='):
                raise UnsupportedOpenQASMError(f'assignment operator {child.op.name}')
            if isinstance(child, ast.ClassicalAssignment) and child.op.name in ('&=', '|=', '^=', '<<=', '>>=', '**='):
                self.__validate_integer_operator_assignment_types(child, iterators)
            angle_parameter = (self.__angle_phase_parameter(child, iterators)
                               if isinstance(child, (ast.QuantumGate, ast.QuantumPhase)) else None)
            if (isinstance(child, (ast.QuantumGate, ast.QuantumPhase))
                    and any(modifier.modifier == ast.GateModifierName.pow for modifier in child.modifiers)):
                # A skipped loop must still reject an invalid powered gate's
                # parameter types/names, without evaluating its arithmetic.
                arguments = child.arguments if isinstance(child, ast.QuantumGate) else [child.argument]
                for argument in arguments:
                    value_type, _ = self.__loop_expression_info(argument, iterators)
                    if value_type in (ValueType.BIT, ValueType.BOOL, ValueType.COMPLEX):
                        raise WrongParameterTypeException(value_type)
                if not self.__is_initialization_process:
                    for qubit in child.qubits:
                        name = self.__operand_name(qubit)
                        if name in iterators or name not in self.__declared_quantum_registers:
                            raise InvalidQubitOperandException(f'{name} is not a declared qubit operand')
            if isinstance(child, ast.QuantumGate):
                modifier_info = self.__analyze_gate_modifiers(
                    child.modifiers, iterators, allow_runtime=True, allow_float=True)
                control_groups = modifier_info.controls
                controls = sum(count for _, count in control_groups)
                if child.duration is not None:
                    raise UnsupportedOpenQASMError('gate duration in a loop')
                if child.name.name not in self.gate_signatures:
                    raise UnsupportedOpenQASMError(f'gate {child.name.name}')
                if (not self.__is_initialization_process
                        and child.name.name not in self.default_gates_qcx_map
                        and not self.__is_stdgates_included):
                    raise UnsupportedOpenQASMError(f'gate {child.name.name}')
                parameters, qubits = self.gate_signatures[child.name.name]
                qubits += controls
                if len(child.arguments) != parameters or len(child.qubits) != qubits:
                    raise WrongGateArityException(child.name.name, parameters, qubits,
                                                  len(child.arguments), len(child.qubits))
                if angle_parameter is not None:
                    self.__validate_angle_phase_qubits(child, iterators)
                if isinstance(modifier_info, _OrderedGateModifierInfo):
                    mnemonic = (self.default_gates_qcx_map | self.stdgates_qcx_map)[child.name.name]
                    if angle_parameter is not None:
                        self.__angle_phase_descriptors(modifier_info)
                    elif (self.qcx_controlled_gate_families[mnemonic][0] in ('U1', 'EX', 'EY', 'EZ')
                            or child.name.name in ('u2', 'u3', 'U', 'cu')):
                        self.__phase_power_factors(modifier_info)
                    else:
                        self.__floating_gate_power_phase(mnemonic, modifier_info)
                    indices = [index for qubit in child.qubits if isinstance(qubit, ast.IndexedIdentifier)
                               for group in qubit.indices
                               for index in (group if isinstance(group, list) else [group])]
                    # Named/compound selections use the actual expansion's
                    # constant-index context. No iterator value exists here.
                    literal_indices = all(self.__is_literal_qubit_selector(index) for index in indices)
                    if not self.__is_initialization_process and literal_indices:
                        for _ in self.__qubit_operand_tuples(child.qubits, distinct=True):
                            pass
            if isinstance(child, ast.QuantumPhase):
                modifier_info = self.__analyze_gate_modifiers(
                    child.modifiers, iterators, allow_runtime=True, allow_float=True)
                if isinstance(modifier_info, _OrderedGateModifierInfo):
                    if angle_parameter is not None:
                        self.__angle_phase_descriptors(modifier_info)
                    else:
                        self.__phase_power_factors(modifier_info)
                    if not self.__is_initialization_process:
                        indices = [index for qubit in child.qubits if isinstance(qubit, ast.IndexedIdentifier)
                                   for group in qubit.indices
                                   for index in (group if isinstance(group, list) else [group])]
                        if all(self.__is_literal_qubit_selector(index) for index in indices):
                            for _ in self.__qubit_operand_tuples(child.qubits, distinct=True):
                                pass
                control_groups = modifier_info.controls
                controls = sum(count for _, count in control_groups)
                if not controls and child.qubits:
                    raise UnsupportedOpenQASMError('qubit operands on gphase')
                if controls and len(child.qubits) != controls:
                    raise WrongGateArityException('gphase', 1, controls, 1, len(child.qubits))
                if angle_parameter is not None:
                    self.__validate_angle_phase_qubits(child, iterators)
            if isinstance(child, (ast.BranchingStatement, ast.WhileLoop)):
                condition = (child.condition if isinstance(child, ast.BranchingStatement)
                             else child.while_condition)
                condition_type, _ = self.__loop_expression_info(condition, iterators)
                if condition_type not in (ValueType.BOOL, ValueType.BIT):
                    raise UnsupportedOpenQASMError('non-Boolean loop condition')
            if isinstance(child, ast.BranchingStatement):
                self.__validate_loop_body(child.if_block, iterators)
                self.__validate_loop_body(child.else_block, iterators)
            if isinstance(child, ast.ForInLoop):
                self.__validate_loop_header(child)
                uint_type = self.__validate_nested_loop_bounds(child, iterators)
                value_type = self.__source_value_type(child.type)
                # Width metadata must follow the same nested shadowing as
                # value types, even when no enclosing iteration executes.
                binding = _RuntimeIteratorBinding('QASM2QCX_VALIDATION_ITERATOR', value_type, uint_type)
                with self.__iterator_binding(child.identifier.name, binding):
                    self.__validate_loop_body(child.block, iterators | {child.identifier.name: value_type})
            if isinstance(child, ast.WhileLoop):
                self.__validate_loop_syntax(child.while_condition)
                self.__validate_loop_body(child.block, iterators)

    def __validate_integer_operator_assignment_types(
            self, statement: ast.ClassicalAssignment, iterators: dict[str, ValueType]) -> None:
        if self.__validate_register_operator_assignment(statement, iterators):
            return
        target = statement.lvalue
        if isinstance(target, ast.IndexedIdentifier):
            if (len(target.indices) != 1 or not isinstance(target.indices[0], list)
                    or len(target.indices[0]) != 1
                    or not isinstance(target.indices[0][0], ast.Expression)
                    or isinstance(target.indices[0][0], (ast.RangeDefinition, ast.DiscreteSet))):
                raise UnsupportedOpenQASMError('integer operator assignment requires a scalar target')
            target = ast.IndexExpression(target.name, target.indices[0])
        if statement.op.name == '**=':
            expression = ast.BinaryExpression(ast.BinaryOperator['**'], target, statement.rvalue)
            self.__power_expression_info(expression, iterators)
            return
        target_type, _ = self.__loop_expression_info(target, iterators)
        rhs_type, _ = self.__loop_expression_info(statement.rvalue, iterators)
        if statement.op.name in ('<<=', '>>='):
            expression = ast.BinaryExpression(ast.BinaryOperator[statement.op.name[:-1]], target, statement.rvalue)
            self.__validate_shift_types(expression, target_type, rhs_type, iterators)
        else:
            self.__bitwise_type(target_type, rhs_type)
        self.__validate_bitwise_indices(target, iterators)
        self.__validate_bitwise_indices(statement.rvalue, iterators)

    def __validate_bitwise_indices(
            self, expression: ast.Expression, iterators: dict[str, ValueType]) -> None:
        if (isinstance(expression, ast.BinaryExpression)
                and expression.op.name in ('==', '!=', '<', '<=', '>', '>=')
                and self.__register_comparison_width(expression, iterators) is not None):
            return  # Register selections/counts were checked by width validation.
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

    def __validate_register_operator_assignment(
            self, statement: ast.ClassicalAssignment,
            iterators: dict[str, ValueType] | None = None) -> bool:
        if statement.op.name not in ('=', '&=', '|=', '^=', '<<=', '>>='):
            return False
        selected_target = self.__is_register_bit_selection(statement.lvalue)
        if selected_target:
            selection = self.__static_bit_selection(statement.lvalue, iterators, destination=True)
            width = selection.width
            lhs = ast.IndexExpression(statement.lvalue.name, statement.lvalue.indices[0])
        else:
            if not isinstance(statement.lvalue, ast.Identifier) or statement.lvalue.name not in self.__bit_register_source_sizes:
                return False
            lhs = statement.lvalue
            width = self.__bit_register_expression_width(lhs, iterators)
        if width is None:
            return False  # An active scalar iterator shadows the register.
        if (statement.op.name == '=' and not selected_target
                and not isinstance(statement.rvalue, (ast.UnaryExpression, ast.BinaryExpression, ast.FunctionCall))
                and not self.__is_constant_bit_source(statement.rvalue)
                and not self.__is_register_bit_selection(statement.rvalue)):
            return False  # Preserve the existing literal/copy assignment path.
        expression = (ast.BinaryExpression(ast.BinaryOperator[statement.op.name[:-1]],
                                          lhs, statement.rvalue)
                      if statement.op.name in ('<<=', '>>=') else statement.rvalue)
        rhs_width = self.__bit_register_expression_width(expression, iterators)
        if rhs_width is None:
            raise InvalidBitOperandException('A bit-register assignment requires a register value')
        if width != rhs_width:
            raise InvalidBitOperandException(
                f'Bit expression width {rhs_width} does not match target size {width}')
        return True

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
        if (isinstance(node, ast.Identifier)
                and (node.name in self.__const_bit_variable_values_map
                     or node.name in self.__const_angle_values
                     or isinstance(self.__classical_source_types.get(node.name), ast.AngleType))):
            # A skipped plain copy or numeric cast might never otherwise
            # inspect its operand. Still enforce source-order visibility,
            # while letting iterator bindings shadow the constant.
            self.__loop_identifier_info(node, iterators)
            return
        target = (node.lvalue if isinstance(node, ast.ClassicalAssignment)
                  else node.target if isinstance(node, ast.QuantumMeasurementStatement) else None)
        if target is not None:
            self.__reject_constant_bit_target(target, iterators)
            self.__reject_constant_angle_target(target, iterators)
            if (isinstance(node, ast.QuantumMeasurementStatement)
                    and self.__is_angle_identifier(self.__operand_name(target), iterators)):
                raise InvalidBitOperandException('A measurement target must be a bit variable')
        if isinstance(node, ast.ClassicalAssignment):
            self.__validate_angle_assignment(node, iterators)
        # Register values need register-aware contexts, not accidental use as
        # scalar loop bounds, casts, or indices.
        if isinstance(node, ast.FunctionCall) and node.name.name in ('popcount', 'rotl', 'rotr'):
            # Its register argument is checked as a unit, not as a scalar
            # expression while recursively validating an empty loop body.
            self.__function_call_info(node, iterators)
            return
        if isinstance(node, ast.ClassicalAssignment) and self.__validate_register_operator_assignment(node, iterators):
            return
        if (isinstance(node, ast.ClassicalAssignment) and node.op.name == '='
                and self.__is_constant_bit_source(node.rvalue)):
            if self.__bit_register_expression_width(node.rvalue, iterators) is not None:
                raise InvalidBitOperandException('A constant register source requires a register target')
            # Scalar selections were checked without reading mutable storage;
            # still walk the assignment's destination and nested index syntax.
        if (isinstance(node, ast.BinaryExpression)
                and node.op.name in ('==', '!=', '<', '<=', '>', '>=')
                and self.__register_comparison_width(node, iterators) is not None):
            return
        # Logical values and casts also need type checking when a body is
        # empty; this inspects types without executing skipped arithmetic.
        if (isinstance(node, ast.FunctionCall)
                or (isinstance(node, ast.UnaryExpression) and node.op.name in ('~', '-', '!'))
                or (isinstance(node, ast.Cast)
                    and (isinstance(node.type, (ast.BoolType, ast.BitType, ast.AngleType))
                         or self.__expression_mentions_angle(node)))
                or (isinstance(node, ast.BinaryExpression) and node.op.name in (
                    '&', '|', '^', '<<', '>>', '**', '%', '&&', '||', '==', '!=', '<', '<=', '>', '>='))
                or (isinstance(node, ast.BinaryExpression) and self.__expression_mentions_angle(node))):
            self.__loop_expression_info(node, iterators)
            self.__validate_bitwise_indices(node, iterators)
        for field in dataclasses.fields(node):
            value = getattr(node, field.name)
            if isinstance(value, (ast.QASMNode, list)):
                self.__validate_bitwise_types(value, iterators)

    @contextlib.contextmanager
    def __expanded_loop_scope(self) -> Iterator[None]:
        # Runtime-valued sets and constant bit registers emit one body copy
        # per element without making their iterators compile-time constants.
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
            iterators: set[str] | dict[str, ValueType] | None = None
            ) -> tuple[str | _ConstantBitRegister, int, ValueType, _UnsignedIntegerType | None]:
        name = source.name
        if ((iterators is not None and name in iterators)
                or any(name in scope for scope in self.__loop_bindings)):
            raise UnsupportedOpenQASMError('iteration over a scalar for-loop iterator')
        if name in self.__const_bit_register_values:
            constant = self.__constant_bit_register(name)
            if not isinstance(iteration_type, ast.BitType) or iteration_type.size is not None:
                raise UnsupportedOpenQASMError('constant bit-register iteration requires a scalar bit iterator')
            # Return the immutable value itself, never a fabricated QCX
            # storage name or an array packed into a native integer.
            return constant, constant.width, ValueType.BIT, None
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
            return variable_name, size, ValueType.BIT, None
        if isinstance(variable_type, ast.ArrayType):
            if not isinstance(variable_type.base_type, (
                    ast.IntType, ast.UintType, ast.FloatType, ast.BoolType, ast.ComplexType)):
                raise UnsupportedOpenQASMError('iteration over classical array other than int, uint, float, bool, or complex')
            value_type = self.__source_value_type(variable_type.base_type)
            if self.__source_value_type(iteration_type) != value_type:
                raise UnsupportedOpenQASMError('array iteration requires matching element and iterator types')
            uint_type = None
            if value_type == ValueType.UINT:
                uint_type = self.__unsigned_integer_type(iteration_type, iterators)
                if uint_type.width != self.__uint_variable_types[variable_name].width:
                    raise UnsupportedOpenQASMError('UINT array iteration requires matching effective widths')
            if (not self.__is_initialization_process
                    and variable_name not in self.__array_element_types):
                raise NoVariableNameException(name)
            return variable_name, self.__array_source_sizes[name], value_type, uint_type
        raise UnsupportedOpenQASMError('for-loop identifier other than an int/uint/float/bool/complex array or bit register')

    def __visit_collection_for(self, statement: ast.ForInLoop) -> None:
        source, size, value_type, uint_type = self.__collection_loop_info(
            statement.set_declaration, statement.type)
        binding = _RuntimeIteratorBinding(f'QASM2QCX_LOOP_{self.__loop_index}_ITERATOR', value_type, uint_type)
        with self.__iterator_binding(statement.identifier.name, binding):
            self.__validate_loop_body(statement.block, self.__loop_iterator_types())
        self.__validate_loop_syntax(statement)
        loop_index = self.__loop_index
        self.__loop_index += 1
        body_label = f'QASM2QCX_LOOP_{loop_index}_BODY'
        next_label = f'QASM2QCX_LOOP_{loop_index}_NEXT'
        end_label = f'QASM2QCX_LOOP_{loop_index}_END'
        if isinstance(source, _ConstantBitRegister):
            self.__visit_constant_bit_register_for(statement, source, loop_index)
            return
        if self.__is_initialization_process:
            with self.__iterator_binding(statement.identifier.name, binding), self.__loop_context(end_label, next_label):
                for child in statement.block:
                    self.visit(child)
            return

        previous_temporaries = self.__declared_temporary_variables.copy()
        previous_used = self.__used_temporary_variables.copy()
        iterator = self.__add_new_temporary_variable(value_type, uint_type)
        index = self.__add_new_temporary_variable(ValueType.INT)
        try:
            # QCX stores a singleton bit register as a scalar INT, unlike
            # classical arrays, whose scalar accesses retain explicit indexing.
            operand = source if value_type == ValueType.BIT and size == 1 else f'{source}:{index}'
            self.__qcx_lines.extend([f'LET {index} := 0', f'@{body_label}',
                                    f'LET {iterator} := {operand}'])
            # Read the current element on each iteration, not a loop-entry
            # snapshot. Keep its copied value and index reserved across the body.
            with self.__iterator_binding(statement.identifier.name, _RuntimeIteratorBinding(iterator, value_type, uint_type)), \
                    self.__loop_context(end_label, next_label):
                for child in statement.block:
                    self.visit(child)
            self.__qcx_lines.extend([f'@{next_label}', f'JUMPIF {end_label} {index} == {size - 1}',
                                    f'LET {index} += 1', f'JUMP {body_label}', f'@{end_label}'])
            self.__hoist_temporary_declarations(previous_temporaries)
            if self.__inside_expanded_loop():
                self.__check_loop_output_limit()
        finally:
            # No expression result escapes a collection loop. Also release
            # body temporaries left behind by failed expression lowering,
            # without disturbing storage reserved by enclosing callers.
            for temporary in self.__used_temporary_variables - previous_used:
                self.__release_temporary_variable(temporary)

    def __visit_constant_bit_register_for(
            self, statement: ast.ForInLoop, source: _ConstantBitRegister, loop_index: int) -> None:
        count = source.width
        if count > self.MAX_LOOP_ITERATIONS - self.__loop_iterations:
            raise InvalidLoopRangeException(
                f'For-loop expansion exceeds {self.MAX_LOOP_ITERATIONS} iterations')
        self.__loop_iterations += count
        end_label = f'QASM2QCX_LOOP_{loop_index}_END'
        previous_temporaries = self.__declared_temporary_variables.copy()
        previous_used = self.__used_temporary_variables.copy()
        iterator = (f'QASM2QCX_LOOP_{loop_index}_ITERATOR' if self.__is_initialization_process
                    else self.__add_new_temporary_variable(ValueType.BIT))
        try:
            # Keep BIT/runtime dependency semantics even though each source
            # bit is known. A plain integer binding would change typing and
            # incorrectly permit compile-time uses of the loop variable.
            with self.__iterator_binding(statement.identifier.name, _RuntimeIteratorBinding(iterator, ValueType.BIT)), \
                    self.__expanded_loop_scope():
                for index in range(count):
                    if not self.__is_initialization_process:
                        self.__qcx_lines.append(f'LET {iterator} := {(source.value >> index) & 1}')
                    with self.__loop_context(end_label, f'QASM2QCX_LOOP_{loop_index}_NEXT_{index}') as frame:
                        for child in statement.block:
                            self.visit(child)
                    if not self.__is_initialization_process:
                        self.__qcx_lines.append(f'@{frame.continue_label}')
                        self.__check_loop_output_limit()
            if not self.__is_initialization_process:
                self.__qcx_lines.append(f'@{end_label}')
                self.__hoist_temporary_declarations(previous_temporaries)
                self.__check_loop_output_limit()
        finally:
            for temporary in self.__used_temporary_variables - previous_used:
                self.__release_temporary_variable(temporary)

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

        if isinstance(expression.type, ast.AngleType):
            angle_type = self.__angle_type(expression.type)
            self.__visit_angle_conversion(expression.argument, angle_type)
            self.__angle_expression_types[id(expression)] = (expression, angle_type)
            return
        if isinstance(expression.type, ast.BoolType):
            if self.__expression_kind == ExpressionKind.CONST_ARITHMETIC:
                self.visit(expression.argument)
                if self.__value_type not in (ValueType.BOOL, ValueType.BIT, ValueType.INT, ValueType.UINT, ValueType.FLOAT,
                                            ValueType.ANGLE):
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

        if self.__value_type == ValueType.ANGLE:
            if self.__value_kind == ValueKind.TEMPORARY:
                self.__release_temporary_variable(str(self.__value))
            raise UnsupportedOpenQASMError('angle cast other than angle or bool')
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
        signature, runtime = self.__function_call_info(expression)
        if signature.rotation:
            self.__visit_rotation(expression, signature, runtime)
            return
        if signature.population_count:
            self.__visit_population_count(expression, signature, runtime)
            return
        if self.__expression_kind != ExpressionKind.CONST_ARITHMETIC:
            if self.__expression_kind not in (ExpressionKind.ARITHMETIC, ExpressionKind.CONDITIONAL):
                raise UnsupportedOpenQASMError('function call outside a value expression')
            self.__visit_runtime_function(expression, signature)
            return
        if runtime:
            raise NoConstantExpressionException
        previous = (self.__expression_kind, self.__value, self.__value_type,
                    self.__value_kind, self.__evaluate_constant)
        previous_used = self.__used_temporary_variables.copy()
        try:
            operands = []
            for argument, target_type in zip(expression.arguments, signature.operand_types):
                self.visit(argument)
                if self.__value_kind != ValueKind.LITERAL:
                    raise NoConstantExpressionException
                # A skipped constant Boolean operand is still validated, but
                # does not perform casts or numerical function evaluation.
                if not self.__evaluate_constant:
                    continue
                try:
                    if target_type == ValueType.INT:
                        value = int(self.__value)
                        if self.__value_type == ValueType.UINT and value > self.QCX_INT_MAX:
                            raise InvalidConstantFunctionException(
                                f'Function {expression.name.name}: UINT to INT conversion outside QCX INT range')
                    else:
                        value = complex(self.__value) if target_type == ValueType.COMPLEX else float(self.__value)
                    operands.append(value)
                except (OverflowError, ValueError) as error:
                    raise InvalidConstantFunctionException(
                        f'Function {expression.name.name}: operand outside supported constant precision') from error
            result = (complex() if signature.result_type == ValueType.COMPLEX
                      else 0 if signature.result_type == ValueType.INT else 0.0)
            if self.__evaluate_constant:
                result = self.__fold_builtin_function(expression.name.name, signature, tuple(operands))
            self.__value, self.__value_type, self.__value_kind = result, signature.result_type, ValueKind.LITERAL
        except Exception:
            (self.__expression_kind, self.__value, self.__value_type,
             self.__value_kind, self.__evaluate_constant) = previous
            raise
        finally:
            for temporary in self.__used_temporary_variables - previous_used:
                self.__release_temporary_variable(temporary)

    def __constant_rotation_distance(self, expression: ast.Expression) -> int:
        self.visit(expression)
        if self.__value_kind != ValueKind.LITERAL:
            raise NoConstantExpressionException
        if not self.__evaluate_constant:
            return 0
        distance = int(self.__value)
        if not self.QCX_INT_MIN <= distance <= self.QCX_INT_MAX:
            raise InvalidConstantFunctionException('Rotation distance outside QCX INT range')
        return distance

    def __visit_rotation(
            self, expression: ast.FunctionCall, signature: _BuiltinFunctionSignature, runtime: bool) -> None:
        if signature.result_type == ValueType.BIT_REGISTER:
            raise InvalidBitOperandException('A rotation register result requires a register context')
        if self.__expression_kind not in (
                ExpressionKind.CONST_ARITHMETIC, ExpressionKind.ARITHMETIC, ExpressionKind.CONDITIONAL):
            raise UnsupportedOpenQASMError('function call outside a value expression')
        if self.__expression_kind != ExpressionKind.CONST_ARITHMETIC:
            self.__visit_runtime_uint_rotation(expression)
            return
        if runtime:
            raise NoConstantExpressionException
        previous = (self.__expression_kind, self.__value, self.__value_type, self.__value_kind)
        try:
            self.__expression_kind = ExpressionKind.CONST_ARITHMETIC
            argument, distance_expression = expression.arguments
            _, uint_type, _ = self.__integer_operand_info(argument, None)
            self.visit(argument)
            if self.__value_kind != ValueKind.LITERAL:
                raise NoConstantExpressionException
            value = int(self.__value)
            distance = self.__constant_rotation_distance(distance_expression)
            self.__value = (self.__fold_rotation(value, uint_type.width, distance, expression.name.name)
                            if self.__evaluate_constant else 0)
            self.__value_type, self.__value_kind = ValueType.UINT, ValueKind.LITERAL
            self.__uint_expression_types[id(expression)] = (expression, uint_type)
            self.__expression_kind = previous[0]
        except Exception:
            self.__expression_kind, self.__value, self.__value_type, self.__value_kind = previous
            raise

    def __visit_runtime_uint_rotation(self, expression: ast.FunctionCall) -> None:
        previous = (self.__expression_kind, self.__value, self.__value_type, self.__value_kind)
        previous_used = self.__used_temporary_variables.copy()
        keep = None
        prefix = f'QASM2QCX_ROTATION_{self.__rotation_index}'
        self.__rotation_index += 1
        try:
            self.__expression_kind = ExpressionKind.ARITHMETIC
            argument, distance_expression = expression.arguments
            _, uint_type, _ = self.__integer_operand_info(argument, None)
            value, value_type, value_kind = self.__condition_operand(argument)
            # Snapshot the value before evaluating the distance, and keep all
            # nested operand/index storage reserved until the result is ready.
            result = self.__add_new_temporary_variable(ValueType.UINT, uint_type)
            operand, _ = self.__converted_operand(value, value_type, value_kind, ValueType.UINT)
            self.__qcx_lines.append(f'LET {result} := {operand}')
            value, value_type, value_kind = self.__condition_operand(distance_expression)
            operand, _ = self.__converted_operand(value, value_type, value_kind, ValueType.INT)
            count = self.__add_new_temporary_variable(ValueType.INT)
            complement = self.__add_new_temporary_variable(ValueType.INT)
            tail = self.__add_new_temporary_variable(ValueType.UINT, uint_type)
            self.__qcx_lines.extend([
                # UINT-to-INT conversion is checked before normalization. Do
                # not negate a signed distance: INT_MIN must remain valid.
                f'LET {count} := {operand}', f'LET {count} %= {uint_type.width}',
                f'JUMPIF {prefix}_NORMALIZED {count} >= 0',
                f'LET {count} += {uint_type.width}', f'@{prefix}_NORMALIZED',
                # Both native-width and width-one inputs must avoid a shift
                # by their width when the normalized distance is zero.
                f'JUMPIF {prefix}_DONE {count} == 0',
                f'LET {complement} := {uint_type.width}', f'LET {complement} -= {count}',
                f'LET {tail} := {result}',
            ])
            operator, opposite = ('<<=', '>>=') if expression.name.name == 'rotl' else ('>>=', '<<=')
            self.__qcx_lines.extend([
                f'LET {result} {operator} {count}', f'LET {tail} {opposite} {complement}',
                f'LET {result} |= {tail}',
            ])
            self.__normalize_uint_storage(result, uint_type)
            self.__qcx_lines.append(f'@{prefix}_DONE')
            self.__uint_expression_types[id(expression)] = (expression, uint_type)
            self.__expression_kind = previous[0]
            self.__value, self.__value_type, self.__value_kind = result, ValueType.UINT, ValueKind.TEMPORARY
            keep = result
        except Exception:
            self.__expression_kind, self.__value, self.__value_type, self.__value_kind = previous
            raise
        finally:
            for temporary in self.__used_temporary_variables - previous_used:
                if temporary != keep:
                    self.__release_temporary_variable(temporary)

    def __visit_population_count(
            self, expression: ast.FunctionCall, signature: _BuiltinFunctionSignature, runtime: bool) -> None:
        if self.__expression_kind not in (
                ExpressionKind.CONST_ARITHMETIC, ExpressionKind.ARITHMETIC, ExpressionKind.CONDITIONAL):
            raise UnsupportedOpenQASMError('function call outside a value expression')
        if self.__expression_kind == ExpressionKind.CONST_ARITHMETIC and runtime:
            raise NoConstantExpressionException
        previous = (self.__expression_kind, self.__value, self.__value_type,
                    self.__value_kind, self.__evaluate_constant)
        previous_used = self.__used_temporary_variables.copy()
        result = None
        kind = ValueKind.LITERAL
        keep = False
        try:
            argument = expression.arguments[0]
            unsigned_operand = signature.operand_types == (ValueType.UINT,)
            if self.__expression_kind == ExpressionKind.CONST_ARITHMETIC:
                if unsigned_operand:
                    _, uint_type, _ = self.__integer_operand_info(argument, None)
                    self.visit(argument)
                    if self.__value_kind != ValueKind.LITERAL:
                        raise NoConstantExpressionException
                    # Input expressions already apply their unsigned width
                    # rules. Mask explicitly at the counting boundary as well.
                    packed = int(self.__value) & uint_type.mask if self.__evaluate_constant else 0
                else:
                    width = self.__bit_register_expression_width(argument)
                    packed = self.__constant_bit_register_value(argument, width)
                result = packed.bit_count() if self.__evaluate_constant else 0
            elif unsigned_operand:
                self.__expression_kind = ExpressionKind.ARITHMETIC
                result = self.__emit_uint_population_count(argument)
                kind = ValueKind.TEMPORARY
            else:
                self.__expression_kind = ExpressionKind.ARITHMETIC
                with self.__bit_register_expression_value(argument) as values:
                    # A register can be wider than a packed UINT. Count its
                    # individual bits without shifting or packing the register.
                    result = self.__add_new_temporary_variable(ValueType.UINT)
                    literal_count = sum(value for value in values if isinstance(value, int))
                    self.__qcx_lines.append(f'LET {result} := {literal_count}')
                    for value in values:
                        if isinstance(value, str):
                            self.__qcx_lines.append(f'LET {result} += :UINT:{value}')
                kind = ValueKind.TEMPORARY
            self.__expression_kind = previous[0]
            self.__value, self.__value_type, self.__value_kind = result, ValueType.UINT, kind
            self.__uint_expression_types[id(expression)] = (
                expression, _UnsignedIntegerType(self.QCX_UINT_WIDTH, False))
            keep = True
        except Exception:
            (self.__expression_kind, self.__value, self.__value_type,
             self.__value_kind, self.__evaluate_constant) = previous
            raise
        finally:
            for temporary in self.__used_temporary_variables - previous_used:
                if not keep or kind != ValueKind.TEMPORARY or temporary != result:
                    self.__release_temporary_variable(temporary)

    def __emit_uint_population_count(self, argument: ast.Expression) -> str:
        # The enclosing visitor owns all scratch storage and preserves only
        # the returned native UINT accumulator, including on failure.
        _, uint_type, _ = self.__integer_operand_info(argument, None)
        value, value_type, value_kind = self.__condition_operand(argument)
        operand, _ = self.__converted_operand(value, value_type, value_kind, ValueType.UINT)
        remaining = self.__add_new_temporary_variable(ValueType.UINT, uint_type)
        self.__qcx_lines.append(f'LET {remaining} := {operand}')
        self.__normalize_uint_storage(remaining, uint_type)
        result = self.__add_new_temporary_variable(ValueType.UINT)
        bit = self.__add_new_temporary_variable(ValueType.UINT)
        prefix = f'QASM2QCX_POPCOUNT_{self.__population_count_index}'
        self.__population_count_index += 1
        # Consume a private copy using only unsigned operations. A one-bit
        # native shift is valid even when the source has declared width one;
        # no narrowing through INT or shift by the input width is required.
        self.__qcx_lines.extend([
            f'LET {result} := 0', f'@{prefix}_LOOP',
            f'JUMPIF {prefix}_DONE {remaining} == 0',
            f'LET {bit} := {remaining}', f'LET {bit} &= 1',
            f'LET {result} += {bit}', f'LET {remaining} >>= 1',
            f'JUMP {prefix}_LOOP', f'@{prefix}_DONE',
        ])
        return result

    def __emit_function_real_literal(self, destination: str, value: int | float) -> None:
        # QCX has no portable inf/nan literal. Materialize nonfinite values
        # inside the reached path, leaving native math to perform its checks.
        # Do not narrow integer literals through Python float first.
        if isinstance(value, float) and not math.isfinite(value):
            numerator = 0.0 if math.isnan(value) else math.copysign(1.0, value)
            self.__qcx_lines.extend([f'LET {destination} := {numerator!r}', f'LET {destination} /= 0.0'])
        else:
            self.__qcx_lines.append(f'LET {destination} := :REAL:{value}')

    def __runtime_function_operand(
            self, value: str | int | float | complex, value_type: ValueType,
            value_kind: ValueKind, target_type: ValueType) -> str:
        if target_type == ValueType.INT:
            # Do not narrow integers through REAL. UINT-to-INT casts stay in
            # the reached QCX path, where they are checked before use.
            operand, _ = self.__converted_operand(value, value_type, value_kind, target_type)
            return operand
        if value_kind == ValueKind.LITERAL:
            if value_type != ValueType.COMPLEX:
                scalar = self.__add_new_temporary_variable(ValueType.FLOAT)
                self.__emit_function_real_literal(scalar, value)
                return scalar if target_type == ValueType.FLOAT else f':COMPLEX:{scalar}'
            if not cmath.isfinite(value):
                # Literal component extraction is handled directly by the
                # caller. Native math rejects either nonfinite component; using
                # a typed nonfinite operand avoids unparseable QCX literals.
                scalar = self.__add_new_temporary_variable(ValueType.FLOAT)
                component = value.real if not math.isfinite(value.real) else value.imag
                self.__emit_function_real_literal(scalar, component)
                return f':COMPLEX:{scalar}'
        operand, _ = self.__converted_operand(value, value_type, value_kind, target_type)
        return operand

    def __visit_runtime_function(self, expression: ast.FunctionCall, signature: _BuiltinFunctionSignature) -> None:
        # Keep all argument/conversion storage reserved until the result is
        # allocated. Later arguments, nested calls, and indexed destinations
        # must not alias or overwrite earlier operands.
        previous = (self.__expression_kind, self.__value, self.__value_type,
                    self.__value_kind, self.__evaluate_constant)
        previous_used = self.__used_temporary_variables.copy()
        result = None
        keep = False
        try:
            self.__expression_kind = ExpressionKind.ARITHMETIC
            operands = []
            for argument, target_type in zip(expression.arguments, signature.operand_types):
                self.visit(argument)
                value, value_type, value_kind = self.__value, self.__value_type, self.__value_kind
                if value_kind is None or value_type is None:
                    raise UninitializedValueException
                if signature.component is not None and value_kind == ValueKind.LITERAL:
                    # Literal component selection is exact, not arithmetic that
                    # could contaminate finite components or discard zero signs.
                    if value_type == ValueType.COMPLEX:
                        component = value.real if signature.component == 'REAL' else value.imag
                    else:
                        component = value if signature.component == 'REAL' else 0.0
                    result = self.__add_new_temporary_variable(signature.result_type)
                    self.__emit_function_real_literal(result, component)
                else:
                    operands.append(self.__runtime_function_operand(value, value_type, value_kind, target_type))
            if signature.integer_remainder:
                result = self.__emit_integer_remainder(*operands)
            elif result is None:
                result = self.__add_new_temporary_variable(signature.result_type)
                if signature.component is not None:
                    self.__qcx_lines.append(f'LET {result} := :{signature.component}:{operands[0]}')
                else:
                    self.__qcx_lines.append(f'APPLY {result} <- {signature.qcx_function} {" ".join(operands)}')
            self.__expression_kind = previous[0]
            self.__value, self.__value_type, self.__value_kind = result, signature.result_type, ValueKind.TEMPORARY
            keep = True
        except Exception:
            (self.__expression_kind, self.__value, self.__value_type,
             self.__value_kind, self.__evaluate_constant) = previous
            raise
        finally:
            for temporary in self.__used_temporary_variables - previous_used:
                if not keep or temporary != result:
                    self.__release_temporary_variable(temporary)

    @staticmethod
    def __fold_builtin_function(
            name: str, signature: _BuiltinFunctionSignature, operands: tuple[int | float | complex, ...]
            ) -> int | float | complex:
        # Keep integer folding exact; real/complex folding uses Python's host
        # precision, as existing constant arithmetic does. Runtime uses bra.
        value = operands[0]
        if signature.integer_remainder:
            dividend, divisor = operands
            if divisor == 0:
                raise InvalidConstantFunctionException(f'Function {name}: zero divisor')
            return dividend - QASM2QCXConverter.__integer_quotient(dividend, divisor) * divisor
        if signature.component is not None:
            # Component extraction is LET, not APPLY: preserve its IEEE values
            # (including signed zeros/nonfinite components), without EXP checks.
            return value.real if signature.component == 'REAL' else value.imag
        functions = {
            'EXP': {ValueType.FLOAT: math.exp, ValueType.COMPLEX: cmath.exp},
            'SQRT': {ValueType.FLOAT: math.sqrt, ValueType.COMPLEX: cmath.sqrt},
            'LOG': {ValueType.FLOAT: math.log},
            'SIN': {ValueType.FLOAT: math.sin},
            'COS': {ValueType.FLOAT: math.cos},
            'TAN': {ValueType.FLOAT: math.tan},
            'ARCSIN': {ValueType.FLOAT: math.asin},
            'ARCCOS': {ValueType.FLOAT: math.acos},
            'ARCTAN': {ValueType.FLOAT: math.atan},
            'CEILING': {ValueType.FLOAT: math.ceil},
            'FLOOR': {ValueType.FLOAT: math.floor},
            'MOD': {ValueType.FLOAT: math.fmod},
        }
        operand_type = signature.operand_types[0]
        function = functions.get(signature.qcx_function, {}).get(operand_type)
        if function is None:
            raise UnsupportedOpenQASMError(f'constant evaluation of function {name}')
        if any(not cmath.isfinite(operand) for operand in operands):
            raise InvalidConstantFunctionException(f'Function {name}: nonfinite input')
        if signature.qcx_function == 'MOD' and operands[1] == 0:
            raise InvalidConstantFunctionException(f'Function {name}: zero divisor')
        if signature.qcx_function == 'SQRT' and operand_type == ValueType.FLOAT and value < 0:
            raise InvalidConstantFunctionException(f'Function {name}: negative real input')
        if signature.qcx_function == 'LOG' and value <= 0:
            raise InvalidConstantFunctionException(f'Function {name}: nonpositive real input')
        if signature.qcx_function in ('ARCSIN', 'ARCCOS') and not -1 <= value <= 1:
            raise InvalidConstantFunctionException(f'Function {name}: real input outside [-1, 1]')
        try:
            result = function(*operands)
            if signature.qcx_function in ('CEILING', 'FLOOR'):
                # Python returns arbitrary-precision integers, but OpenQASM
                # rounding calls return float. Recover the input's zero sign,
                # which is lost by the intermediate Python integer result.
                result = float(result)
                if result == 0:
                    result = math.copysign(result, value)
        except (OverflowError, ValueError) as error:
            raise InvalidConstantFunctionException(f'Function {name}: overflow/nonfinite result') from error
        if not cmath.isfinite(result):
            raise InvalidConstantFunctionException(f'Function {name}: overflow/nonfinite result')
        return result

    def visit_IndexExpression(self, expression: ast.IndexExpression) -> None:
        if self.__expression_kind is None:
            return
        if self.__expression_kind == ExpressionKind.CONST_ARITHMETIC:
            if (isinstance(expression.collection, ast.Identifier)
                    and expression.collection.name in self.__const_bit_register_values):
                self.__value = self.__constant_bit_element(expression)
                self.__value_type, self.__value_kind = ValueType.BIT, ValueKind.LITERAL
                return
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
        if self.__value_type == ValueType.ANGLE:
            raise NoImplicitCastException
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

    def __constant_bit_register(self, name: str) -> _ConstantBitRegister:
        constant = self.__const_bit_register_values.get(name)
        if (constant is None or any(name in scope for scope in self.__loop_bindings)
                or (not self.__is_initialization_process
                    and name not in self.__declared_constant_variables)):
            raise NoVariableNameException(name)
        return constant

    def __constant_bit_element(self, expression: ast.IndexExpression) -> int:
        selection = self.__static_bit_selection(expression)
        if selection.is_register:
            raise UnsupportedOpenQASMError('constant bit selection requires a register-expression context')
        constant = self.__constant_bit_register(selection.source_name)
        return (constant.value >> selection.indices[0]) & 1 if self.__evaluate_constant else 0

    def __is_constant_bit_source(self, expression: ast.Expression) -> bool:
        return ((isinstance(expression, ast.Identifier)
                 and (expression.name in self.__const_bit_register_values
                      or expression.name in self.__const_bit_variable_values_map))
                or (isinstance(expression, ast.IndexExpression)
                    and isinstance(expression.collection, ast.Identifier)
                    and expression.collection.name in self.__const_bit_register_values))

    def __reject_constant_bit_target(
            self, target: ast.Identifier | ast.IndexedIdentifier,
            iterators: set[str] | dict[str, ValueType] | None = None) -> None:
        name = self.__operand_name(target)
        shadowed = ((iterators is not None and name in iterators)
                    or any(name in scope for scope in self.__loop_bindings))
        if not shadowed and name in self.__const_bit_variable_values_map:
            self.__loop_identifier_info(ast.Identifier(name), None)
            raise InvalidBitOperandException('A constant scalar bit is read-only')
        if not shadowed and name in self.__const_bit_register_values:
            self.__constant_bit_register(name)
            raise InvalidBitOperandException('A constant bit register is read-only')

    def __expression_mentions_angle(self, expression: ast.Expression) -> bool:
        if isinstance(expression, ast.Cast) and isinstance(expression.type, ast.AngleType):
            return True
        if isinstance(expression, ast.Identifier):
            return (expression.name in self.__const_angle_values
                    or isinstance(self.__classical_source_types.get(expression.name), ast.AngleType))
        for field in dataclasses.fields(expression):
            value = getattr(expression, field.name)
            children = value if isinstance(value, list) else [value]
            if any(isinstance(child, ast.Expression) and self.__expression_mentions_angle(child)
                   for child in children):
                return True
        return False

    def __is_angle_identifier(self, name: str, iterators: dict[str, ValueType]) -> bool:
        # Do not ask scalar expression lookup about unrelated register/array
        # names. Active iterators can shadow either angle kind.
        return ((name in self.__const_angle_values
                 or isinstance(self.__classical_source_types.get(name), ast.AngleType))
                and self.__loop_identifier_info(ast.Identifier(name), iterators)[0] == ValueType.ANGLE)

    def __validate_angle_assignment(
            self, statement: ast.ClassicalAssignment, iterators: dict[str, ValueType]) -> None:
        if self.__is_angle_identifier(self.__operand_name(statement.lvalue), iterators):
            expression = self.__angle_assignment_expression(statement)
            source_type, _ = self.__loop_expression_info(expression, iterators)
            if source_type not in (ValueType.FLOAT, ValueType.ANGLE):
                raise UnsupportedOpenQASMError('angle conversion requires float or angle input')
        elif self.__expression_mentions_angle(statement.rvalue):
            if self.__loop_expression_info(statement.rvalue, iterators)[0] == ValueType.ANGLE:
                raise NoImplicitCastException

    @staticmethod
    def __angle_assignment_expression(statement: ast.ClassicalAssignment) -> ast.Expression:
        if isinstance(statement.lvalue, ast.IndexedIdentifier):
            raise UnsupportedOpenQASMError('angle indexing')
        if statement.op.name == '=':
            return statement.rvalue
        if statement.op.name not in ('+=', '-='):
            raise UnsupportedOpenQASMError(f'angle assignment operator {statement.op.name}')
        # Preserve the binary operation's promoted width before converting
        # back to the destination. Pre-narrowing the RHS could lose a borrow.
        return ast.BinaryExpression(ast.BinaryOperator[statement.op.name[:-1]], statement.lvalue, statement.rvalue)

    def __reject_constant_angle_target(
            self, target: ast.Identifier | ast.IndexedIdentifier,
            iterators: set[str] | dict[str, ValueType] | None = None) -> None:
        name = self.__operand_name(target)
        if ((iterators is not None and name in iterators)
                or any(name in scope for scope in self.__loop_bindings)):
            return
        if name in self.__const_angle_values:
            self.__loop_identifier_info(ast.Identifier(name), iterators)
            raise InvalidDeclarationException('A constant angle is read-only')

    def __declare_constant_angle(self, statement: ast.ConstantDeclaration) -> None:
        name = statement.identifier.name
        if statement.init_expression is None:
            raise NoConstantExpressionException
        previous = (self.__expression_kind, self.__value, self.__value_type,
                    self.__value_kind, self.__evaluate_constant)
        try:
            angle_type = self.__angle_type(statement.type)
            _, runtime = self.__loop_expression_info(statement.init_expression)
            if runtime:
                raise NoConstantExpressionException
            self.__expression_kind = ExpressionKind.CONST_ARITHMETIC
            self.__evaluate_constant = True
            self.__visit_angle_conversion(statement.init_expression, angle_type)
            if self.__value_type != ValueType.ANGLE or self.__value_kind != ValueKind.LITERAL:
                raise NoConstantExpressionException
            constant = _ConstantAngle(angle_type, int(self.__value))
        finally:
            (self.__expression_kind, self.__value, self.__value_type,
             self.__value_kind, self.__evaluate_constant) = previous
        self.__const_angle_values[name] = constant
        self.__constant_source_types[name] = statement.type

    def __declare_constant_scalar_bit(self, statement: ast.ConstantDeclaration) -> None:
        name = statement.identifier.name
        if statement.init_expression is None:
            raise NoConstantExpressionException
        previous = (self.__expression_kind, self.__value, self.__value_type,
                    self.__value_kind, self.__evaluate_constant)
        try:
            # Check dependencies before evaluating: short-circuiting cannot
            # turn an expression involving mutable storage into a constant.
            _, runtime = self.__loop_expression_info(statement.init_expression)
            if runtime:
                raise NoConstantExpressionException
            self.__expression_kind = ExpressionKind.CONST_ARITHMETIC
            self.__evaluate_constant = True
            self.visit(statement.init_expression)
            if self.__value_kind != ValueKind.LITERAL:
                raise NoConstantExpressionException
            if (self.__value_type not in (ValueType.BIT, ValueType.BOOL)
                    and not (self.__value_type in (ValueType.INT, ValueType.UINT)
                             and self.__value in (0, 1))):
                raise NoImplicitCastException
            value = int(self.__value)
        finally:
            (self.__expression_kind, self.__value, self.__value_type,
             self.__value_kind, self.__evaluate_constant) = previous
        # Publish only a successfully initialized value; self/forward
        # references must never observe a placeholder zero.
        self.__const_bit_variable_values_map[name] = value
        self.__constant_source_types[name] = statement.type

    def __declare_constant_bit_register(self, statement: ast.ConstantDeclaration) -> None:
        name = statement.identifier.name
        if statement.init_expression is None:
            raise NoConstantExpressionException
        previous = (self.__expression_kind, self.__value, self.__value_type,
                    self.__value_kind, self.__evaluate_constant)
        try:
            width = self.__bit_type_size(statement.type, name)
            self.__expression_kind = ExpressionKind.CONST_ARITHMETIC
            self.__evaluate_constant = True
            source_width = self.__bit_register_expression_width(statement.init_expression)
            if source_width is None or source_width != width:
                raise InvalidDeclarationException(
                    f'Constant bit register {name}[{width}] requires a matching-width register initializer')
            if self.__register_expression_runtime(statement.init_expression, None):
                raise NoConstantExpressionException
            value = self.__constant_bit_register_value(statement.init_expression, width)
            constant = _ConstantBitRegister(width, value)
        finally:
            (self.__expression_kind, self.__value, self.__value_type,
             self.__value_kind, self.__evaluate_constant) = previous
        # Never expose a fabricated zero while evaluating an initializer or
        # its width. Reserve the source name separately for duplicate checks.
        self.__const_bit_register_values[name] = constant
        self.__constant_source_types[name] = statement.type

    def visit_ConstantDeclaration(self, statement: ast.ConstantDeclaration) -> None:
        if not self.__is_initialization_process:
            self.__declared_constant_variables.add(statement.identifier.name)
            return

        variable_type = statement.type
        variable_name = statement.identifier.name
        self.__register_source_identifier(variable_name)
        if isinstance(variable_type, ast.AngleType):
            self.__declare_constant_angle(statement)
            return
        if isinstance(variable_type, ast.BitType):
            if variable_type.size is None:
                self.__declare_constant_scalar_bit(statement)
            else:
                self.__declare_constant_bit_register(statement)
            return
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
            case ast.AngleType():
                if num_elements != 1 or variable_name in self.__angle_variable_names:
                    raise InvalidDeclarationException('only scalar angle storage is supported')
                self.__angle_variable_names.add(variable_name)
                self.__qcx_lines.append(f'VAR {variable_name} UINT')

            case ast.BoolType():
                self.__bool_variable_names.add(variable_name)
                self.__qcx_lines.append(f'VAR {variable_name} INT' + (f' {num_elements}' if num_elements > 1 else ''))

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

    def __angle_type(
            self, variable_type: ast.AngleType,
            iterators: set[str] | dict[str, ValueType] | None = None) -> _AngleType:
        if variable_type.size is None:
            return _AngleType(self.QCX_UINT_WIDTH, False)
        names = self.__loop_bound_names(variable_type.size)
        active = set(iterators or ()) | {name for scope in self.__loop_bindings for name in scope}
        if names & active:
            raise InvalidDeclarationException('Angle width must be a constant integer, not a loop iterator')
        value_type, runtime = self.__loop_expression_info(variable_type.size)
        if value_type not in (ValueType.INT, ValueType.UINT) or runtime:
            raise InvalidDeclarationException('Angle width must be a constant integer')
        previous = (self.__expression_kind, self.__value, self.__value_type,
                    self.__value_kind, self.__evaluate_constant)
        try:
            self.__expression_kind = ExpressionKind.CONST_ARITHMETIC
            self.__evaluate_constant = True
            self.visit(variable_type.size)
            if self.__value_kind != ValueKind.LITERAL or self.__value_type not in (ValueType.INT, ValueType.UINT):
                raise InvalidDeclarationException('Angle width must be a constant integer')
            width = int(self.__value)
            if not 1 <= width <= self.QCX_UINT_WIDTH:
                raise InvalidDeclarationException(f'Angle width must be in [1, {self.QCX_UINT_WIDTH}]')
            return _AngleType(width, True)
        finally:
            (self.__expression_kind, self.__value, self.__value_type,
             self.__value_kind, self.__evaluate_constant) = previous

    def __identifier_angle_type(self, name: str) -> _AngleType | None:
        if any(name in scope for scope in self.__loop_bindings):
            return None
        constant = self.__const_angle_values.get(name)
        return (constant.angle_type if constant is not None
                else self.__angle_variable_types.get(self.__capitalize_variable_name(name)))

    def __angle_operand_info(
            self, expression: ast.Expression,
            iterators: set[str] | dict[str, ValueType] | None = None
            ) -> tuple[ValueType, _AngleType | None, bool]:
        # Shared, non-evaluating inference for arithmetic and skipped bodies.
        # Recurse through additive nodes once, rather than re-analyzing both
        # subtrees separately for value types, widths, and runtime dependence.
        if (isinstance(expression, ast.BinaryExpression) and expression.op.name in ('*', '/')
                and self.__expression_mentions_angle(expression)):
            info = self.__angle_multiplicative_info(expression, iterators)
            if info is not None:
                return info.result_type, info.result if info.result_type == ValueType.ANGLE else None, info.runtime
        if isinstance(expression, ast.BinaryExpression) and expression.op.name in ('+', '-'):
            lhs_type, lhs_angle, lhs_runtime = self.__angle_operand_info(expression.lhs, iterators)
            rhs_type, rhs_angle, rhs_runtime = self.__angle_operand_info(expression.rhs, iterators)
            runtime = lhs_runtime or rhs_runtime
            if ValueType.ANGLE not in (lhs_type, rhs_type):
                return self.__promoted_type(lhs_type, rhs_type), None, runtime
            if lhs_angle is None or rhs_angle is None:
                raise UnsupportedOpenQASMError('angle addition and subtraction require two angle operands')
            # Match UINT's sizedness convention, but preserve angle values by
            # appending low zeros when widening, not by integer zero-extension.
            angle_type = _AngleType(max(lhs_angle.width, rhs_angle.width),
                                    lhs_angle.is_sized or rhs_angle.is_sized)
            return ValueType.ANGLE, angle_type, runtime
        if isinstance(expression, ast.UnaryExpression) and expression.op.name == '-':
            value_type, angle_type, runtime = self.__angle_operand_info(expression.expression, iterators)
            if value_type == ValueType.ANGLE:
                return value_type, angle_type, runtime
            value_type, runtime = self.__negation_expression_info(expression, iterators)
            return value_type, None, runtime
        value_type, runtime = self.__loop_expression_info(expression, iterators)
        if value_type != ValueType.ANGLE:
            return value_type, None, runtime
        if isinstance(expression, ast.Identifier):
            angle_type = self.__identifier_angle_type(expression.name)
        elif isinstance(expression, ast.Cast) and isinstance(expression.type, ast.AngleType):
            angle_type = self.__angle_type(expression.type, iterators)
        else:
            angle_type = None
        if angle_type is None:
            raise UnsupportedOpenQASMError('angle expression without width metadata')
        return value_type, angle_type, runtime

    def __angle_multiplicative_info(
            self, expression: ast.BinaryExpression,
            iterators: set[str] | dict[str, ValueType] | None = None) -> _AngleMultiplicativeInfo | None:
        lhs_type, lhs_angle, lhs_runtime = self.__angle_operand_info(expression.lhs, iterators)
        rhs_type, rhs_angle, rhs_runtime = self.__angle_operand_info(expression.rhs, iterators)
        if ValueType.ANGLE not in (lhs_type, rhs_type):
            # Integer expressions may contain angle quotients without having
            # angle operands themselves; leave them on the normal UINT path.
            return None
        if lhs_type == rhs_type == ValueType.ANGLE:
            if expression.op.name != '/':
                raise UnsupportedOpenQASMError('multiplication of two angles')
            result = _UnsignedIntegerType(max(lhs_angle.width, rhs_angle.width),
                                          lhs_angle.is_sized or rhs_angle.is_sized)
            return _AngleMultiplicativeInfo(lhs_angle, rhs_angle, ValueType.UINT, result, lhs_runtime or rhs_runtime)
        if ((lhs_type, rhs_type) not in ((ValueType.ANGLE, ValueType.UINT), (ValueType.UINT, ValueType.ANGLE))
                or (lhs_type == ValueType.UINT and expression.op.name == '/')):
            raise UnsupportedOpenQASMError('angle scaling requires angle * uint, uint * angle, or angle / uint')
        factor = expression.rhs if rhs_type == ValueType.UINT else expression.lhs
        factor_type, uint_type, factor_runtime = self.__integer_operand_info(factor, iterators)
        if factor_type != ValueType.UINT or uint_type is None:
            # A narrow UINT mixed with INT can promote to INT. Do not let
            # the preliminary value-type lookup admit signed scaling.
            raise UnsupportedOpenQASMError('angle scaling requires an unsigned integer factor')
        angle_type = lhs_angle if lhs_type == ValueType.ANGLE else rhs_angle
        if angle_type.width != uint_type.width:
            raise UnsupportedOpenQASMError('angle and uint scaling operands must have matching effective widths')
        return _AngleMultiplicativeInfo(
            lhs_angle if lhs_type == ValueType.ANGLE else uint_type,
            rhs_angle if rhs_type == ValueType.ANGLE else uint_type,
            ValueType.ANGLE, angle_type, lhs_runtime or rhs_runtime or factor_runtime)

    def __visit_angle_multiplicative(self, expression: ast.BinaryExpression, info: _AngleMultiplicativeInfo) -> None:
        if info.runtime:
            if self.__expression_kind == ExpressionKind.CONST_ARITHMETIC:
                raise NoConstantExpressionException
            raise UnsupportedOpenQASMError('runtime angle multiplication/division')
        previous_used = self.__used_temporary_variables.copy()
        try:
            values = []
            for operand, metadata in ((expression.lhs, info.lhs), (expression.rhs, info.rhs)):
                self.visit(operand)
                if self.__value_kind != ValueKind.LITERAL:
                    # Literal float-to-angle casts are executable outside a
                    # constant context. Runtime emission is a separate stage.
                    raise UnsupportedOpenQASMError('runtime angle multiplication/division')
                value = int(self.__value)
                if isinstance(metadata, _AngleType):
                    value <<= info.result.width - metadata.width
                else:
                    value = metadata.normalize(value)
                values.append(value)
            if not self.__evaluate_constant:
                value = 0
            elif expression.op.name == '*':
                value = values[0] * values[1]
            elif values[1] != 0:
                value = values[0] // values[1]
            elif self.__expression_kind == ExpressionKind.CONST_ARITHMETIC:
                raise ZeroDivisorException('/')
            else:
                # Do not turn a possibly skipped ordinary expression into a
                # constant-domain error. Native division will handle this in
                # the runtime-emission stage, even for literal zero divisors.
                raise UnsupportedOpenQASMError('runtime angle division with zero divisor')
            self.__value = value & info.result.mask
            self.__value_type, self.__value_kind = info.result_type, ValueKind.LITERAL
            metadata_map = (self.__angle_expression_types if info.result_type == ValueType.ANGLE
                            else self.__uint_expression_types)
            metadata_map[id(expression)] = (expression, info.result)
        finally:
            for temporary in self.__used_temporary_variables - previous_used:
                self.__release_temporary_variable(temporary)

    def __visit_angle_arithmetic(self, expression: ast.BinaryExpression | ast.UnaryExpression) -> None:
        _, angle_type, runtime = self.__angle_operand_info(expression)
        if runtime and self.__expression_kind == ExpressionKind.CONST_ARITHMETIC:
            raise NoConstantExpressionException
        operands = ([expression.lhs, expression.rhs] if isinstance(expression, ast.BinaryExpression)
                    else [expression.expression])
        previous_used = self.__used_temporary_variables.copy()
        result = None
        keep = False
        try:
            # Capture each operand before evaluating the next, retaining both
            # snapshots until the arithmetic is complete. Sources stay intact.
            values = [self.__angle_arithmetic_operand(operand, angle_type) for operand in operands]
            if all(isinstance(value, int) for value in values):
                bits = (-values[0] if isinstance(expression, ast.UnaryExpression)
                        else values[0] + values[1] if expression.op.name == '+' else values[0] - values[1])
                self.__value = (bits & angle_type.mask) if self.__evaluate_constant else 0
                self.__value_type, self.__value_kind = ValueType.ANGLE, ValueKind.LITERAL
            else:
                if self.__expression_kind == ExpressionKind.CONST_ARITHMETIC:
                    raise NoConstantExpressionException
                result = self.__add_new_temporary_variable(ValueType.UINT, _UnsignedIntegerType(angle_type.width, True))
                unary = isinstance(expression, ast.UnaryExpression)
                initial = 0 if unary else values[0]
                rhs = values[0] if unary else values[1]
                operator = '-=' if unary or expression.op.name == '-' else '+='
                self.__qcx_lines.extend([f'LET {result} := {initial}', f'LET {result} {operator} {rhs}'])
                self.__normalize_uint_storage(result, _UnsignedIntegerType(angle_type.width, True))
                self.__value, self.__value_type, self.__value_kind = result, ValueType.ANGLE, ValueKind.TEMPORARY
            self.__angle_expression_types[id(expression)] = (expression, angle_type)
            keep = result is not None
        finally:
            for temporary in self.__used_temporary_variables - previous_used:
                if not keep or temporary != result:
                    self.__release_temporary_variable(temporary)

    def __angle_arithmetic_operand(self, expression: ast.Expression, target: _AngleType) -> int | str:
        # Called inside the arithmetic visitor's temporary-storage scope.
        self.visit(expression)
        source = self.__angle_expression_type(expression)
        if self.__value_type != ValueType.ANGLE or source is None:
            raise UnsupportedOpenQASMError('angle arithmetic requires angle operands with width metadata')
        if self.__value_kind == ValueKind.LITERAL:
            return self.__resize_angle_bits(int(self.__value), source, target)
        if self.__value_kind not in (ValueKind.LVALUE, ValueKind.TEMPORARY):
            raise UninitializedValueException
        snapshot = self.__add_new_temporary_variable(ValueType.UINT, _UnsignedIntegerType(target.width, True))
        self.__qcx_lines.append(f'LET {snapshot} := {self.__value}')
        if target.width > source.width:
            self.__qcx_lines.append(f'LET {snapshot} <<= {target.width - source.width}')
        return snapshot

    def __angle_expression_type(self, expression: ast.Expression) -> _AngleType | None:
        if isinstance(expression, ast.Identifier):
            return self.__identifier_angle_type(expression.name)
        if isinstance(expression, (ast.BinaryExpression, ast.UnaryExpression)):
            # A loop iterator can shadow an angle name between uses of the
            # same AST. Never reuse compound metadata from a previous binding.
            return (self.__angle_operand_info(expression)[1]
                    if self.__expression_mentions_angle(expression) else None)
        entry = self.__angle_expression_types.get(id(expression))
        if entry is None and isinstance(expression, ast.Cast) and isinstance(expression.type, ast.AngleType):
            return self.__angle_operand_info(expression)[1]
        return entry[1] if entry is not None else None

    @staticmethod
    def __constant_angle_bits(value: float, angle_type: _AngleType) -> int:
        if not math.isfinite(value):
            raise InvalidConstantFunctionException('angle conversion: nonfinite input')
        # Quantize the exact ratio of stored binary64 values, not a rounded
        # floating-point quotient. Python round(Fraction) uses ties to even;
        # masking also handles negative inputs and rounding across a full turn.
        return round(Fraction(value) / Fraction(math.tau) * (1 << angle_type.width)) & angle_type.mask

    @staticmethod
    def __resize_angle_bits(value: int, source: _AngleType, target: _AngleType) -> int:
        difference = target.width - source.width
        return value << difference if difference >= 0 else value >> -difference

    def __visit_angle_conversion(self, expression: ast.Expression, angle_type: _AngleType) -> None:
        source_type, runtime = self.__loop_expression_info(expression)
        if source_type not in (ValueType.FLOAT, ValueType.ANGLE):
            raise UnsupportedOpenQASMError('angle conversion requires float or angle input')
        if self.__expression_kind == ExpressionKind.CONST_ARITHMETIC and runtime:
            raise NoConstantExpressionException
        previous_used = self.__used_temporary_variables.copy()
        result = None
        keep = False
        try:
            self.visit(expression)
            value, value_kind = self.__value, self.__value_kind
            if source_type == ValueType.FLOAT and self.__expression_kind == ExpressionKind.CONST_ARITHMETIC:
                self.__value = self.__constant_angle_bits(float(value), angle_type) if self.__evaluate_constant else 0
                self.__value_type, self.__value_kind = ValueType.ANGLE, ValueKind.LITERAL
                return
            source_angle_type = self.__angle_expression_type(expression) if source_type == ValueType.ANGLE else None
            if source_type == ValueType.ANGLE:
                if source_angle_type is None:
                    raise UnsupportedOpenQASMError('angle expression without width metadata')
                if value_kind == ValueKind.LITERAL:
                    self.__value = self.__resize_angle_bits(int(value), source_angle_type, angle_type)
                    self.__value_type, self.__value_kind = ValueType.ANGLE, ValueKind.LITERAL
                    return
                if source_angle_type.width == angle_type.width:
                    # A completed operand may own a temporary. Transfer it
                    # to the caller rather than releasing it in this scope.
                    result, keep = str(value), value_kind == ValueKind.TEMPORARY
                    return
            result = self.__add_new_temporary_variable(
                ValueType.UINT, _UnsignedIntegerType(angle_type.width, True))
            if source_type == ValueType.FLOAT:
                operand = str(value)
                if value_kind == ValueKind.LITERAL and not math.isfinite(float(value)):
                    operand = '+nan' if math.isnan(float(value)) else '-inf' if float(value) < 0 else '+inf'
                self.__qcx_lines.append(f'APPLY {result} <- ANGLEBITS :REAL:{operand} {angle_type.width}')
            else:
                self.__qcx_lines.append(f'LET {result} := {value}')
                difference = angle_type.width - source_angle_type.width
                operator = '<<=' if difference > 0 else '>>='
                self.__qcx_lines.append(f'LET {result} {operator} {abs(difference)}')
            self.__value, self.__value_type, self.__value_kind = result, ValueType.ANGLE, ValueKind.TEMPORARY
            keep = True
        finally:
            for temporary in self.__used_temporary_variables - previous_used:
                if not keep or temporary != result:
                    self.__release_temporary_variable(temporary)

    def __emit_angle_copy(self, destination: str, angle_type: _AngleType, expression: ast.Expression) -> None:
        previous = self.__expression_kind
        previous_used = self.__used_temporary_variables.copy()
        try:
            self.__expression_kind = ExpressionKind.ARITHMETIC
            self.__visit_angle_conversion(expression, angle_type)
            self.__emit_assignment(destination, ':=', ValueType.ANGLE,
                                   self.__value, self.__value_type, self.__value_kind)
        finally:
            self.__expression_kind = previous
            for temporary in self.__used_temporary_variables - previous_used:
                self.__release_temporary_variable(temporary)

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
        with self.__bit_assignment_target(target) as (target_names, target_is_register):
            self.__emit_bit_value_assignment(target_names, target_is_register, expression)

    @contextlib.contextmanager
    def __bit_assignment_target(
            self, target: ast.Identifier | ast.IndexedIdentifier) -> Iterator[tuple[list[str], bool]]:
        if self.__is_register_bit_selection(target):
            selection = self.__static_bit_selection(target, destination=True)
            name = self.__capitalize_variable_name(selection.source_name)
            if self.__type_of(name) != ValueType.BIT:
                raise InvalidBitOperandException('A bit selection destination must have bit type')
            size = self.__bit_variable_name_size_map[name]
            yield [self.__qcx_bit_name(name, size, index) for index in selection.indices], True
            return
        with self.__bit_target(target) as resolved:
            yield resolved

    @contextlib.contextmanager
    def __shift_bit_register_value(
            self, expression: ast.BinaryExpression,
            values: tuple[str | int, ...]) -> Iterator[tuple[str | int, ...]]:
        operand = None
        temporaries = []
        try:
            operand = self.__condition_operand(expression.rhs)
            count, count_type, count_kind = operand
            if count_type not in (ValueType.INT, ValueType.UINT):
                raise InvalidBitOperandException('A register shift count must be a scalar integer')
            width = len(values)
            if count_kind == ValueKind.LITERAL and int(count) >= 0:
                distance = int(count)
                if all(isinstance(value, int) for value in values):
                    packed = sum(int(value) << index for index, value in enumerate(values))
                    folded = self.__fold_bit_register_shift(packed, width, distance, expression.op.name)
                    yield tuple((folded >> index) & 1 for index in range(width))
                    return
                result = []
                for index in range(width):
                    source = index - distance if expression.op.name == '<<' else index + distance
                    value = values[source] if 0 <= source < width else 0
                    if isinstance(value, str):
                        storage = self.__add_new_temporary_variable(ValueType.BIT)
                        temporaries.append(storage)
                        self.__qcx_lines.append(f'LET {storage} := {value}')
                        value = storage
                    result.append(value)
                yield tuple(result)
                return
            if width > self.QCX_INT_MAX:
                raise InvalidBitOperandException('Runtime register shifts require a width within QCX INT range')
            captured = self.__add_new_temporary_variable(count_type)
            temporaries.append(captured)
            self.__qcx_lines.append(f'LET {captured} := {count}')
            if count_type == ValueType.INT:
                self.__qcx_lines.append(f'ASSERT {captured} >= 0')
            source_buffer = self.__add_temporary_bit_register(width)
            result_buffer = self.__add_temporary_bit_register(width)
            temporaries.extend((source_buffer, result_buffer))
            result = tuple(self.__qcx_bit_name(result_buffer, width, index) for index in range(width))
            for index, value in enumerate(values):
                self.__qcx_lines.extend([
                    f'LET {self.__qcx_bit_name(source_buffer, width, index)} := {value}',
                    f'LET {result[index]} := 0'])
            source_index = self.__add_new_temporary_variable(ValueType.INT)
            target_index = self.__add_new_temporary_variable(ValueType.INT)
            temporaries.extend((source_index, target_index))
            label = f'QASM2QCX_REGISTER_SHIFT_{self.__register_shift_index}'
            self.__register_shift_index += 1
            # Oversized UINT counts must reach the zero result before a checked
            # INT cast or index arithmetic. Valid indices increment at most to
            # width, which is representable even at the native INT endpoint.
            count_as_int = f':INT:{captured}' if count_type == ValueType.UINT else captured
            self.__qcx_lines.extend([
                f'JUMPIF {label}_END {captured} >= {width}',
                f'LET {source_index} := {0 if expression.op.name == "<<" else count_as_int}',
                f'LET {target_index} := {count_as_int if expression.op.name == "<<" else 0}',
                f'@{label}_COPY',
                f'JUMPIF {label}_END {source_index} >= {width}',
                f'JUMPIF {label}_END {target_index} >= {width}',
                f'LET {result_buffer}:{target_index} := {source_buffer}:{source_index}',
                f'LET {source_index} += 1', f'LET {target_index} += 1',
                f'JUMP {label}_COPY', f'@{label}_END'])
            yield result
        finally:
            for temporary in temporaries:
                self.__release_temporary_variable(temporary)
            if operand is not None and operand[2] == ValueKind.TEMPORARY:
                self.__release_temporary_variable(str(operand[0]))

    def __add_temporary_bit_register(self, width: int) -> str:
        index = 0
        name = f'QASM2QCX_BITS_{width}_{index}'
        while name in self.__used_temporary_variables or name in self.__reserved_variable_names:
            index += 1
            name = f'QASM2QCX_BITS_{width}_{index}'
        self.__used_temporary_variables.add(name)
        if name not in self.__declared_temporary_variables:
            self.__declared_temporary_variables.add(name)
            self.__qcx_lines.append(f'VAR {name} INT {width}')
        return name

    @contextlib.contextmanager
    def __rotate_bit_register_value(
            self, expression: ast.FunctionCall,
            values: tuple[str | int, ...]) -> Iterator[tuple[str | int, ...]]:
        operand = None
        temporaries = []
        try:
            operand = self.__condition_operand(expression.arguments[1])
            distance, distance_type, distance_kind = operand
            width = len(values)
            if distance_kind == ValueKind.LITERAL and self.QCX_INT_MIN <= int(distance) <= self.QCX_INT_MAX:
                # Snapshot borrowed source bits even for a zero rotation: a
                # selected destination can overlap the source in another order.
                count = int(distance) % width
                if expression.name.name == 'rotr' and count:
                    count = width - count
                result = []
                for index in range(width):
                    value = values[(index - count) % width]
                    if isinstance(value, str):
                        temporary = self.__add_new_temporary_variable(ValueType.BIT)
                        temporaries.append(temporary)
                        self.__qcx_lines.append(f'LET {temporary} := {value}')
                        value = temporary
                    result.append(value)
                yield tuple(result)
                return
            if width > self.QCX_INT_MAX:
                raise InvalidBitOperandException('Runtime register rotations require a width within QCX INT range')
            prefix = f'QASM2QCX_ROTATION_{self.__rotation_index}'
            self.__rotation_index += 1
            count = self.__add_new_temporary_variable(ValueType.INT)
            source_index = self.__add_new_temporary_variable(ValueType.INT)
            target_index = self.__add_new_temporary_variable(ValueType.INT)
            source_buffer = self.__add_temporary_bit_register(width)
            result_buffer = self.__add_temporary_bit_register(width)
            temporaries.extend((count, source_index, target_index, source_buffer, result_buffer))
            converted, _ = self.__converted_operand(distance, distance_type, distance_kind, ValueType.INT)
            self.__qcx_lines.extend([
                f'LET {count} := {converted}', f'LET {count} %= {width}',
                f'JUMPIF {prefix}_NORMALIZED {count} >= 0',
                f'LET {count} += {width}', f'@{prefix}_NORMALIZED',
                f'LET {target_index} := {count}',
            ])
            if expression.name.name == 'rotr':
                self.__qcx_lines.extend([
                    f'JUMPIF {prefix}_DIRECTION {count} == 0',
                    f'LET {target_index} := {width}', f'LET {target_index} -= {count}',
                    f'@{prefix}_DIRECTION',
                ])
            for index, value in enumerate(values):
                self.__qcx_lines.append(
                    f'LET {self.__qcx_bit_name(source_buffer, width, index)} := {value}')
            # Copy by indices, never by packing the register into native UINT.
            # Incrementing indices at most to width avoids signed overflow,
            # unlike computing (index + distance) before taking a remainder.
            self.__qcx_lines.extend([
                f'LET {source_index} := 0', f'@{prefix}_COPY',
                f'JUMPIF {prefix}_DONE {source_index} >= {width}',
                f'JUMPIF {prefix}_READY {target_index} < {width}',
                f'LET {target_index} := 0', f'@{prefix}_READY',
                f'LET {result_buffer}:{target_index} := {source_buffer}:{source_index}',
                f'LET {source_index} += 1', f'LET {target_index} += 1',
                f'JUMP {prefix}_COPY', f'@{prefix}_DONE',
            ])
            yield tuple(self.__qcx_bit_name(result_buffer, width, index) for index in range(width))
        finally:
            for temporary in temporaries:
                self.__release_temporary_variable(temporary)
            if operand is not None and operand[2] == ValueKind.TEMPORARY:
                self.__release_temporary_variable(str(operand[0]))

    @contextlib.contextmanager
    def __bit_register_expression_value(
            self, expression: ast.Expression) -> Iterator[tuple[str | int, ...]]:
        # The caller validates widths first. Each result owns its temporary
        # bits until the assignment finishes; no destination is written here.
        if isinstance(expression, ast.FunctionCall) and expression.name.name in ('rotl', 'rotr'):
            with self.__bit_register_expression_value(expression.arguments[0]) as values:
                with self.__rotate_bit_register_value(expression, values) as result:
                    yield result
            return
        if isinstance(expression, ast.BitstringLiteral):
            yield tuple((expression.value >> index) & 1 for index in range(expression.width))
            return
        if isinstance(expression, ast.Identifier):
            if (expression.name in self.__const_bit_register_values
                    and not any(expression.name in scope for scope in self.__loop_bindings)):
                constant = self.__constant_bit_register(expression.name)
                yield tuple((constant.value >> index) & 1 for index in range(constant.width))
                return
            name, indices, is_register = self.__bit_operand(expression, role='Bit source')
            if not is_register:
                raise InvalidBitOperandException('A bit-register expression requires register operands')
            size = self.__bit_variable_name_size_map[name]
            yield tuple(self.__qcx_bit_name(name, size, index) for index in indices)
            return
        if isinstance(expression, ast.IndexExpression):
            selection = self.__static_bit_selection(expression)
            if not selection.is_register:
                raise InvalidBitOperandException('A bit-register expression requires register operands')
            if selection.source_name in self.__const_bit_register_values:
                constant = self.__constant_bit_register(selection.source_name)
                yield tuple((constant.value >> index) & 1 for index in selection.indices)
                return
            name = self.__capitalize_variable_name(selection.source_name)
            # Declaration metadata alone is not permission to read storage
            # before its actual declaration in the emitted program.
            if self.__type_of(name) != ValueType.BIT:
                raise InvalidBitOperandException('A bit selection source must have bit type')
            size = self.__bit_variable_name_size_map[name]
            yield tuple(self.__qcx_bit_name(name, size, index) for index in selection.indices)
            return
        if isinstance(expression, ast.BinaryExpression) and expression.op.name in ('<<', '>>'):
            with self.__bit_register_expression_value(expression.lhs) as values:
                with self.__shift_bit_register_value(expression, values) as result:
                    yield result
            return
        temporaries = []
        try:
            with contextlib.ExitStack() as operands:
                if isinstance(expression, ast.UnaryExpression) and expression.op.name == '~':
                    left = operands.enter_context(self.__bit_register_expression_value(expression.expression))
                    right = (1,) * len(left)
                    operator = '^'
                elif isinstance(expression, ast.BinaryExpression) and expression.op.name in ('&', '|', '^'):
                    left = operands.enter_context(self.__bit_register_expression_value(expression.lhs))
                    right = operands.enter_context(self.__bit_register_expression_value(expression.rhs))
                    operator = expression.op.name
                else:
                    raise UnsupportedOpenQASMError(f'bit-register expression {type(expression).__name__}')
                values = []
                for lhs, rhs in zip(left, right):
                    if isinstance(lhs, int) and isinstance(rhs, int):
                        values.append(lhs & rhs if operator == '&' else lhs | rhs if operator == '|' else lhs ^ rhs)
                    else:
                        temporary = self.__add_new_temporary_variable(ValueType.BIT)
                        temporaries.append(temporary)
                        self.__qcx_lines.extend([f'LET {temporary} := {lhs}', f'LET {temporary} {operator}= {rhs}'])
                        values.append(temporary)
            # Child temporaries can be reused now that this complete result
            # has been computed. Its own storage stays reserved across yield.
            yield tuple(values)
        finally:
            for temporary in temporaries:
                self.__release_temporary_variable(temporary)

    def __emit_bit_value_assignment(
            self, target_names: list[str], target_is_register: bool,
            expression: ast.Expression) -> None:
        constant_source = self.__is_constant_bit_source(expression)
        if target_is_register and (constant_source or isinstance(
                expression, (ast.UnaryExpression, ast.BinaryExpression, ast.FunctionCall))):
            width = self.__bit_register_expression_width(expression)
            if width is None:
                raise InvalidBitOperandException('A bit-register assignment requires a register value')
            if width != len(target_names):
                raise InvalidBitOperandException(
                    f'Bit expression width {width} does not match target size {len(target_names)}')
            with self.__bit_register_expression_value(expression) as values:
                for target, value in zip(target_names, values):
                    self.__qcx_lines.append(f'LET {target} := {value}')
            return
        if constant_source:
            value, value_type, value_kind = self.__condition_operand(expression)
            if value_type not in (ValueType.BIT, ValueType.BOOL):
                raise InvalidBitOperandException('Scalar bit assignment requires a bit or Boolean value')
            self.__emit_assignment(target_names[0], ':=', ValueType.BIT, value, value_type, value_kind)
            return
        boolean_identifier = isinstance(expression, ast.Identifier) and (
            self.__capitalize_variable_name(expression.name) in self.__bool_variable_names
            or expression.name in self.__const_bool_variable_values_map)
        boolean_element = (isinstance(expression, ast.IndexExpression)
                           and isinstance(expression.collection, ast.Identifier)
                           and self.__array_element_types.get(
                               self.__capitalize_variable_name(expression.collection.name)) == ValueType.BOOL)
        if not target_is_register and (boolean_identifier or boolean_element or not isinstance(
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

        temporaries = []
        try:
            destinations = set(target_names)
            if target_is_register and any(isinstance(value, str) and value in destinations for value in values):
                # Direct register copies also have value semantics. Snapshot
                # all source bits before any overlapping destination write.
                snapshot = []
                for value in values:
                    temporary = self.__add_new_temporary_variable(ValueType.BIT)
                    temporaries.append(temporary)
                    self.__qcx_lines.append(f'LET {temporary} := {value}')
                    snapshot.append(temporary)
                values = snapshot
            for target_name, value in zip(target_names, values):
                self.__qcx_lines.append(f'LET {target_name} := {value}')
        finally:
            for temporary in temporaries:
                self.__release_temporary_variable(temporary)

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

    def __array_access(
            self, expression: ast.IndexExpression | ast.IndexedIdentifier) -> tuple[str, str | None]:
        if isinstance(expression, ast.IndexExpression):
            if not isinstance(expression.collection, ast.Identifier):
                raise UnsupportedOpenQASMError('multidimensional classical array indexing')
            name, selectors = expression.collection.name, expression.index
        else:
            if len(expression.indices) != 1:
                raise UnsupportedOpenQASMError('multidimensional classical array indexing')
            name, selectors = expression.name.name, expression.indices[0]
        variable_name = self.__capitalize_variable_name(name)
        self.__type_of(variable_name)  # Reject an active iterator shadowing the array.
        if variable_name not in self.__array_element_types:
            raise UnsupportedOpenQASMError('indexed non-array classical operand')
        if not isinstance(selectors, list) or len(selectors) != 1 or not isinstance(selectors[0], ast.Expression):
            raise UnsupportedOpenQASMError('classical array slice or multidimensional selection')
        if isinstance(selectors[0], (ast.RangeDefinition, ast.DiscreteSet)):
            raise UnsupportedOpenQASMError('classical array slice or multidimensional selection')
        value_type, runtime = self.__loop_expression_info(selectors[0])
        if value_type not in (ValueType.INT, ValueType.UINT):
            raise InvalidArrayOperandException(f'Classical array {name} index must be an integer')
        size = self.__array_source_sizes[name]
        if runtime:
            index = self.__capture_array_index(selectors[0], size)
            return f'{variable_name}:{index}', index
        index = self.__constant_loop_integer(selectors[0], 'array index')
        if not -size <= index < size:
            raise InvalidArrayOperandException(f'Classical array {name} index {index} is outside [-{size}, {size - 1}]')
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

    def __array_size(self, variable_type: ast.ArrayType, variable_name: str) -> int:
        if len(variable_type.dimensions) != 1:
            raise UnsupportedOpenQASMError('multidimensional classical array')
        if not isinstance(variable_type.base_type, (
                ast.IntType, ast.UintType, ast.FloatType, ast.BoolType, ast.ComplexType)):
            raise UnsupportedOpenQASMError('classical array base type other than int, uint, float, bool, or complex')
        previous = (self.__expression_kind, self.__value, self.__value_type,
                    self.__value_kind, self.__evaluate_constant)
        try:
            value_type, runtime = self.__loop_expression_info(variable_type.dimensions[0])
            if value_type not in (ValueType.INT, ValueType.UINT) or runtime:
                raise InvalidDeclarationException(
                    f'Classical array {variable_name} must have a constant integer size')
            self.__expression_kind = ExpressionKind.CONST_ARITHMETIC
            self.__value = self.__value_type = self.__value_kind = None
            self.__evaluate_constant = True
            self.visit(variable_type.dimensions[0])
            if self.__value_kind != ValueKind.LITERAL or self.__value_type not in (ValueType.INT, ValueType.UINT):
                raise InvalidDeclarationException(
                    f'Classical array {variable_name} must have a constant integer size')
            size = int(self.__value)
            if not 1 <= size <= self.QCX_INT_MAX:
                raise InvalidDeclarationException(
                    f'Classical array {variable_name} size must be in [1, {self.QCX_INT_MAX}]')
            return size
        finally:
            (self.__expression_kind, self.__value, self.__value_type,
             self.__value_kind, self.__evaluate_constant) = previous

    def __declare_array(self, statement: ast.ClassicalDeclaration, variable_name: str) -> None:
        size = self.__array_source_sizes[statement.identifier.name]
        element_type = self.__source_value_type(statement.type.base_type)
        initializer = statement.init_expression
        if initializer is not None:
            if not isinstance(initializer, ast.ArrayLiteral):
                raise UnsupportedOpenQASMError('classical array initializer other than an array literal')
            if len(initializer.values) != size:
                raise InvalidDeclarationException(
                    f'Classical array {statement.identifier.name} initializer has '
                    f'{len(initializer.values)} elements; expected {size}')
            if any(isinstance(value, ast.ArrayLiteral) for value in initializer.values):
                raise UnsupportedOpenQASMError('nested classical array initializer')
        self.__declare_classical_variable(statement.type.base_type, variable_name, size)
        self.__array_element_types[variable_name] = element_type
        if initializer is None:
            return
        for index, expression in enumerate(initializer.values):
            value, value_type, value_kind = self.__condition_operand(expression)
            if element_type == ValueType.INT and value_type not in (ValueType.INT, ValueType.UINT):
                if value_kind == ValueKind.TEMPORARY:
                    self.__release_temporary_variable(str(value))
                raise InvalidDeclarationException(
                    f'Integer array {statement.identifier.name} initializer element {index} must be an integer')
            try:
                self.__emit_assignment(f'{variable_name}:{index}', ':=', element_type,
                                       value, value_type, value_kind)
            except Exception:
                if (value_kind == ValueKind.TEMPORARY
                        and str(value) in self.__used_temporary_variables):
                    self.__release_temporary_variable(str(value))
                raise

    def visit_ClassicalDeclaration(self, statement: ast.ClassicalDeclaration) -> None:
        if self.__is_initialization_process:
            self.__register_source_identifier(statement.identifier.name)
            base_type = (statement.type.base_type if isinstance(statement.type, ast.ArrayType)
                         else statement.type)
            if isinstance(base_type, ast.UintType):
                # Scalar and array element widths are resolved once, before
                # a later loop iterator can shadow their dimension constants.
                self.__uint_variable_types[self.__capitalize_variable_name(statement.identifier.name)] = (
                    self.__unsigned_integer_type(base_type))
            if isinstance(statement.type, ast.AngleType):
                self.__angle_variable_types[self.__capitalize_variable_name(statement.identifier.name)] = (
                    self.__angle_type(statement.type))
            self.__classical_source_types[statement.identifier.name] = statement.type
            if isinstance(statement.type, ast.ArrayType):
                # Resolve dimensions in declaration scope, before loop iterators
                # can shadow constants referenced by the type expression.
                self.__array_source_sizes[statement.identifier.name] = self.__array_size(
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
            self.__declare_array(statement, variable_name)
            return
        if not isinstance(
                variable_type,
                (ast.IntType, ast.UintType, ast.FloatType, ast.BitType,
                 ast.ComplexType, ast.BoolType, ast.AngleType)):
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

        if isinstance(variable_type, ast.AngleType):
            if not isinstance(statement.init_expression, ast.Expression):
                raise UnsupportedOpenQASMError('angle initializer other than an expression')
            self.__emit_angle_copy(variable_name, self.__angle_variable_types[variable_name], statement.init_expression)
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
        self.__reject_constant_bit_target(statement.lvalue)
        self.__reject_constant_angle_target(statement.lvalue)
        variable_name: str = self.__capitalize_variable_name(source_name)
        variable_type: ValueType = self.__type_of(variable_name)
        if variable_type == ValueType.ANGLE:
            expression = self.__angle_assignment_expression(statement)
            self.__emit_angle_copy(variable_name, self.__angle_variable_types[variable_name], expression)
            return
        if variable_name in self.__array_element_types:
            if not isinstance(statement.lvalue, ast.IndexedIdentifier):
                raise UnsupportedOpenQASMError('whole classical array assignment')
            self.__validate_array_assignment_operator(variable_type, statement.op.name)
            operand, index = self.__array_access(statement.lvalue)
            previous_temporaries = self.__used_temporary_variables.copy()
            try:
                self.__emit_classical_value_assignment(
                    statement, operand, variable_type, runtime_array=index is not None)
            except Exception:
                # A rejected RHS may have allocated expression temporaries.
                # Preserve the captured destination index until the finally
                # block, and any temporaries owned by enclosing expressions.
                for temporary in self.__used_temporary_variables - previous_temporaries:
                    self.__release_temporary_variable(temporary)
                raise
            finally:
                if index is not None:
                    self.__release_temporary_variable(index)
            return

        if variable_type == ValueType.BIT:
            if statement.op.name in ('&=', '|=', '^=', '<<=', '>>='):
                with self.__bit_assignment_target(statement.lvalue) as (targets, is_register):
                    if is_register:
                        lhs = (ast.IndexExpression(statement.lvalue.name, statement.lvalue.indices[0])
                               if isinstance(statement.lvalue, ast.IndexedIdentifier) else statement.lvalue)
                        expression = ast.BinaryExpression(
                            ast.BinaryOperator[statement.op.name[:-1]], lhs, statement.rvalue)
                        self.__emit_bit_value_assignment(targets, True, expression)
                        return
                    if statement.op.name in ('<<=', '>>='):
                        raise UnsupportedOpenQASMError('shift assignment requires a whole bit-register target')
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

    @staticmethod
    def __validate_array_assignment_operator(element_type: ValueType, operator: str) -> None:
        if element_type == ValueType.BOOL and operator != '=':
            raise UnsupportedOpenQASMError(f'Boolean array assignment operator {operator}')
        if element_type == ValueType.FLOAT and operator not in ('=', '+=', '-=', '*=', '/='):
            raise UnsupportedOpenQASMError(f'floating-point array assignment operator {operator}')
        if element_type == ValueType.COMPLEX and operator not in ('=', '+=', '-=', '*=', '/='):
            raise UnsupportedOpenQASMError(f'complex array assignment operator {operator}')

    def __emit_classical_value_assignment(
            self, statement: ast.ClassicalAssignment, variable_name: str,
            variable_type: ValueType, *, runtime_array: bool = False) -> None:
        if statement.op.name == '**=':
            self.__emit_power_assignment(statement, variable_name, variable_type)
            return
        if statement.op.name in ('<<=', '>>='):
            self.__emit_shift_assignment(statement, variable_name, variable_type)
            return
        if (variable_type == ValueType.INT and statement.op.name != '='
                and not self.__native_signed_operand(statement.lvalue)):
            rhs_type, _ = self.__loop_expression_info(statement.rvalue)
            if rhs_type == ValueType.UINT:
                raise UnsupportedOpenQASMError('mixed signed/unsigned assignment with a non-native signed width')
        if variable_type == ValueType.UINT and statement.op.name != '=':
            binary_name = statement.op.name[:-1]
            if binary_name not in ('+', '-', '*', '/', '%', '&', '|', '^'):
                raise UnsupportedOpenQASMError(f'assignment operator {statement.op.name}')
            if isinstance(statement.lvalue, ast.IndexedIdentifier):
                self.__emit_uint_array_compound_assignment(statement, variable_name)
                return
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

    def __emit_uint_array_compound_assignment(
            self, statement: ast.ClassicalAssignment, target: str) -> None:
        # The caller has captured and checked the destination index. Use a
        # value snapshot, not a second visit to the indexed source expression.
        source_uint = self.__uint_variable_types[target.split(':')[0]]
        lhs_node = ast.IndexExpression(statement.lvalue.name, statement.lvalue.indices[0])
        self.__uint_expression_types[id(lhs_node)] = (lhs_node, source_uint)
        previous_used = self.__used_temporary_variables.copy()
        previous_kind = self.__expression_kind
        try:
            snapshot = self.__add_new_temporary_variable(ValueType.UINT, source_uint)
            self.__qcx_lines.append(f'LET {snapshot} := {target}')
            self.__expression_kind = ExpressionKind.ARITHMETIC
            bitwise = statement.op.name in ('&=', '|=', '^=')
            rhs, rhs_type, rhs_kind = (self.__bitwise_operand(statement.rvalue) if bitwise
                                       else self.__condition_operand(statement.rvalue))
            if bitwise:
                self.__bitwise_type(ValueType.UINT, rhs_type)
            if statement.op.name == '%=' and rhs_type not in (ValueType.INT, ValueType.UINT):
                raise UnsupportedOpenQASMError('integer remainder requires int or uint operands')
            result_type, uint_type = self.__integer_expression_type(
                ValueType.UINT, rhs_type, lhs_node, statement.rvalue)
            if (bitwise and result_type == ValueType.INT and rhs_kind == ValueKind.LITERAL
                    and not self.QCX_INT_MIN <= int(rhs) <= self.QCX_INT_MAX):
                raise UnsupportedOpenQASMError('bitwise integer operand outside QCX INT range')
            left, _ = self.__converted_operand(snapshot, ValueType.UINT, ValueKind.LVALUE, result_type)
            right, _ = self.__converted_operand(rhs, rhs_type, rhs_kind, result_type)
            if statement.op.name == '%=':
                result = self.__emit_integer_remainder(left, right, result_type, uint_type)
            else:
                result = self.__add_new_temporary_variable(result_type, uint_type)
                self.__qcx_lines.extend([f'LET {result} := {left}',
                                         f'LET {result} {statement.op.name} {right}'])
                if uint_type is not None:
                    self.__normalize_uint_storage(result, uint_type)
            self.__emit_assignment(target, ':=', ValueType.UINT, result, result_type, ValueKind.TEMPORARY)
        finally:
            self.__expression_kind = previous_kind
            for temporary in self.__used_temporary_variables - previous_used:
                self.__release_temporary_variable(temporary)

    def __emit_power_assignment(
            self, statement: ast.ClassicalAssignment, target: str,
            target_type: ValueType) -> None:
        lhs = (ast.IndexExpression(statement.lvalue.name, statement.lvalue.indices[0])
               if isinstance(statement.lvalue, ast.IndexedIdentifier) else statement.lvalue)
        expression = ast.BinaryExpression(ast.BinaryOperator['**'], lhs, statement.rvalue)
        result_type, uint_type, _ = self.__power_expression_info(expression)
        source_uint = self.__uint_variable_types.get(target.split(':')[0]) if target_type == ValueType.UINT else None
        snapshot = self.__add_new_temporary_variable(target_type, source_uint)
        operand = None
        try:
            # The caller has already captured an indexed destination. Snapshot
            # its value before the RHS, without evaluating its index again.
            self.__qcx_lines.append(f'LET {snapshot} := {target}')
            operand = self.__condition_operand(statement.rvalue)
            self.__emit_power_expression(
                [(snapshot, target_type, ValueKind.TEMPORARY), operand], result_type, uint_type)
            # Apply assignment conversion/masking only to the completed result.
            self.__emit_assignment(target, ':=', target_type, self.__value,
                                   self.__value_type, self.__value_kind, uint_type)
        finally:
            self.__release_temporary_variable(snapshot)
            if operand is not None and operand[2] == ValueKind.TEMPORARY:
                self.__release_temporary_variable(str(operand[0]))

    def __emit_shift_assignment(
            self, statement: ast.ClassicalAssignment, target: str,
            target_type: ValueType) -> None:
        operand = None
        try:
            operand = self.__condition_operand(statement.rvalue)
            expression = ast.BinaryExpression(
                ast.BinaryOperator[statement.op.name[:-1]], statement.lvalue, statement.rvalue)
            self.__validate_shift_types(expression, target_type, operand[1])
            uint_type = self.__uint_variable_types.get(target.split(':')[0]) if target_type == ValueType.UINT else None
            # Use the already captured destination operand, not its AST. In
            # particular, a runtime array index must only be evaluated once.
            self.__emit_shift_expression(statement.op.name[:-1],
                                         [(target, target_type, ValueKind.LVALUE), operand], uint_type)
            # Only write the destination after all checks and the shift have
            # succeeded; the count may refer to the destination itself.
            self.__emit_assignment(target, ':=', target_type, self.__value,
                                   self.__value_type, self.__value_kind, uint_type)
        finally:
            if operand is not None and operand[2] == ValueKind.TEMPORARY:
                self.__release_temporary_variable(str(operand[0]))

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
