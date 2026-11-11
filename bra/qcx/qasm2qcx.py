import sys
import itertools
import math
import cmath
import enum
import dataclasses
import contextlib
import ctypes
from fractions import Fraction
from collections.abc import Iterable, Iterator

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
class _RuntimeSingletonBitSelection:
    # Only loop-source metadata: a runtime index does not make ordinary
    # register expressions or assignments support dynamic set selections.
    source_name: str
    source_width: int
    index: ast.Expression
    index_type: ValueType

    @property
    def width(self) -> int:
        return 1


@dataclasses.dataclass(frozen=True)
class _RuntimeBitSetSelection:
    source_name: str
    source_width: int
    # Retain all entries, including constants, until the access is captured.
    indices: tuple[ast.Expression, ...]
    index_types: tuple[ValueType, ...]

    @property
    def width(self) -> int:
        return len(self.indices)


@dataclasses.dataclass(frozen=True)
class _RuntimeBitRangeSelection:
    # Runtime ranges have no fixed selection width. Retain the raw operands
    # until loop-entry capture, including constants in mixed runtime ranges.
    source_name: str
    source_width: int
    bounds: ast.RangeDefinition
    start_type: ValueType
    step_type: ValueType
    stop_type: ValueType


@dataclasses.dataclass(frozen=True)
class _RuntimeBitRangePopcountShift:
    expression: ast.BinaryExpression
    count_type: ValueType


@dataclasses.dataclass(frozen=True)
class _RuntimeBitRangePopcountOperation:
    expression: ast.UnaryExpression | ast.BinaryExpression | ast.FunctionCall
    count_type: ValueType | None = None


@dataclasses.dataclass(frozen=True)
class _RuntimeBitRangePopcountRotation:
    expression: ast.FunctionCall
    distance_type: ValueType


@dataclasses.dataclass(frozen=True)
class _RuntimeBitRangePopcountOperand:
    expression: ast.Expression
    selection: _RuntimeBitRangeSelection
    # Every original wrapper, innermost to outermost, including complements.
    operations: tuple[_RuntimeBitRangePopcountOperation, ...]


@dataclasses.dataclass(frozen=True)
class _RuntimeBitRangePopcountBitwise:
    expression: ast.BinaryExpression
    fixed_operand: ast.Expression
    width: int
    range_on_left: bool


@dataclasses.dataclass(frozen=True)
class _RuntimeBitRangePopcountPair:
    expression: ast.BinaryExpression
    # The left selection is stored in the enclosing popcount info.
    right_selection: _RuntimeBitRangeSelection


@dataclasses.dataclass(frozen=True)
class _RuntimeBitRangePopcountComplements:
    expression: ast.Expression
    left: bool
    right: bool
    outer: bool


@dataclasses.dataclass(frozen=True)
class _RuntimeAngleBitRangeSelection:
    # A range has no compile-time selection width. Keep it out of the fixed-
    # width register-expression APIs and retain all expressions unevaluated.
    source: ast.Identifier | ast.IndexExpression
    angle_type: _AngleType
    bounds: ast.RangeDefinition
    start_type: ValueType
    step_type: ValueType
    stop_type: ValueType


@dataclasses.dataclass(frozen=True)
class _RuntimeBitRegisterExpression:
    expression: ast.Expression
    capacity: int
    # Fixed leaves have known widths. Runtime lengths are never inferred from
    # source widths, buffer capacities, or neighbouring fixed leaves.
    width: int | None = None
    children: tuple['_RuntimeBitRegisterExpression', ...] = ()
    selection: _RuntimeBitRangeSelection | _RuntimeAngleBitRangeSelection | None = None
    count_type: ValueType | None = None

    @property
    def first_selection(self) -> _RuntimeBitRangeSelection | _RuntimeAngleBitRangeSelection | None:
        if self.selection is not None:
            return self.selection
        for child in self.children:
            selection = child.first_selection
            if selection is not None:
                return selection
        return None


@dataclasses.dataclass(frozen=True)
class _RuntimeBitRegisterComparison:
    expression: ast.BinaryExpression
    left: _RuntimeBitRegisterExpression
    right: _RuntimeBitRegisterExpression


@dataclasses.dataclass(frozen=True)
class _RuntimeBitRegisterAssignment:
    target: ast.Identifier | ast.IndexedIdentifier
    width: int
    value: _RuntimeBitRegisterExpression
    # None denotes a whole register. Static positions are normalized; runtime
    # index sets retain original expressions until destination capture.
    # Neither selection width is an inferred length for a dynamic RHS leaf.
    selection: _StaticBitSelection | _RuntimeBitSetSelection | None = None
    operator: str = '='

    @property
    def binary_operator(self) -> str | None:
        return None if self.operator == '=' else self.operator[:-1]


@dataclasses.dataclass(frozen=True)
class _RuntimeBitRangeExpressionAssignment:
    target: ast.IndexedIdentifier
    selection: _RuntimeBitRangeSelection
    # Distinct destinations cannot outnumber the containing register's bits.
    # This bounds destination storage, not either operand's selected length.
    capacity: int
    value: _RuntimeBitRegisterExpression
    operator: str = '='

    @property
    def binary_operator(self) -> str | None:
        return None if self.operator == '=' else self.operator[:-1]


@dataclasses.dataclass(frozen=True)
class _RuntimeBitRangePopcountInfo:
    selection: _RuntimeBitRangeSelection
    # Parity on the source range; outer parity applies after a shift/rotation.
    complemented: bool
    shift: ast.BinaryExpression | None = None
    count_type: ValueType | None = None
    outer_complemented: bool = False
    rotation: ast.FunctionCall | None = None
    # Innermost to outermost. The single-shift fields retain the existing path.
    shifts: tuple[_RuntimeBitRangePopcountShift, ...] = ()
    # Complete interleaved chain, innermost to outermost. Empty for fast paths.
    operations: tuple[_RuntimeBitRangePopcountOperation, ...] = ()
    # Innermost to outermost; parity includes complements between calls.
    rotations: tuple[_RuntimeBitRangePopcountRotation, ...] = ()
    rotation_complemented: bool = False
    # Mixed chains retain every original node, innermost to outermost.
    mixed_operations: tuple[_RuntimeBitRangePopcountOperation, ...] = ()
    bitwise: _RuntimeBitRangePopcountBitwise | None = None
    pair: _RuntimeBitRangePopcountPair | None = None
    binary_complements: _RuntimeBitRangePopcountComplements | None = None
    # Shift/rotation/complement wrappers around a binary result, innermost to outermost.
    result_operations: tuple[_RuntimeBitRangePopcountOperation, ...] = ()
    # Written operand order. None entries use the existing fixed-register reader.
    operand_chains: tuple[_RuntimeBitRangePopcountOperand | None, _RuntimeBitRangePopcountOperand | None] | None = None
    tree: _RuntimeBitRegisterExpression | None = None


@dataclasses.dataclass(frozen=True)
class _BitRangeCastOperation:
    expression: ast.UnaryExpression | ast.BinaryExpression
    count_type: ValueType | None = None


@dataclasses.dataclass(frozen=True)
class _RuntimeAngleRangeRegisterAssignment:
    target: ast.Identifier
    width: int
    selection: _RuntimeAngleBitRangeSelection
    # Raw positions lie in [-source_width, source_width). This bounds the
    # number of entries, including repeated positions after normalization.
    capacity: int


@dataclasses.dataclass(frozen=True)
class _BitRangeCastExpression:
    expression: ast.Expression
    # Required width, not an inferred count for a runtime range.
    width: int
    children: tuple['_BitRangeCastExpression', ...] = ()
    selection: _RuntimeBitRangeSelection | _RuntimeAngleBitRangeSelection | None = None
    count_type: ValueType | None = None


@dataclasses.dataclass(frozen=True)
class _UIntBitCastInfo:
    result_type: ValueType
    uint_type: _UnsignedIntegerType
    runtime: bool
    range_selection: _RuntimeBitRangeSelection | None = None
    range_operations: tuple[_BitRangeCastOperation, ...] = ()
    range_tree: _BitRangeCastExpression | None = None

    @property
    def width(self) -> int:
        return self.uint_type.width


@dataclasses.dataclass(frozen=True)
class _RuntimeRangeAngleCastInfo:
    expression: ast.Cast
    angle_type: _AngleType
    # Reuse the UINT bit-pattern plan, retaining ANGLE as the public result.
    packing: _UIntBitCastInfo

    @property
    def result_type(self) -> ValueType:
        return ValueType.ANGLE

    @property
    def width(self) -> int:
        return self.angle_type.width

    @property
    def runtime(self) -> bool:
        return self.packing.runtime


_BitSelectionLoop = (
    _StaticBitSelection | _RuntimeSingletonBitSelection | _RuntimeBitSetSelection | _RuntimeBitRangeSelection
)


@dataclasses.dataclass(frozen=True)
class _StaticAngleBitSelection:
    # Keep the angle source separate from register storage. An array element
    # retains its unevaluated index until runtime extraction or assignment.
    source: ast.Identifier | ast.IndexExpression
    angle_type: _AngleType
    indices: tuple[int, ...]
    runtime: bool

    @property
    def width(self) -> int:
        return len(self.indices)


@dataclasses.dataclass(frozen=True)
class _RuntimeAngleBitSelection:
    # Retain both indices unevaluated; array-element selection precedes bit
    # selection when this descriptor is eventually lowered.
    source: ast.Identifier | ast.IndexExpression
    angle_type: _AngleType
    index: ast.Expression
    index_type: ValueType

    @property
    def width(self) -> int:
        return 1

    @property
    def runtime(self) -> bool:
        return True


@dataclasses.dataclass(frozen=True)
class _RuntimeAngleBitSetSelection:
    source: ast.Identifier | ast.IndexExpression
    angle_type: _AngleType
    indices: tuple[ast.Expression, ...]
    index_types: tuple[ValueType, ...]

    @property
    def width(self) -> int:
        return len(self.indices)

    @property
    def runtime(self) -> bool:
        return True


_AngleBitSelection = _StaticAngleBitSelection | _RuntimeAngleBitSelection | _RuntimeAngleBitSetSelection


_AngleBitLoopSelection = _AngleBitSelection | _RuntimeAngleBitRangeSelection


@dataclasses.dataclass(frozen=True)
class _AngleBitAssignmentInfo:
    selection: _AngleBitSelection
    operator: str
    shift_count_type: ValueType | None = None

    @property
    def binary_operator(self) -> str | None:
        return None if self.operator == '=' else self.operator[:-1]


@dataclasses.dataclass(frozen=True)
class _RuntimeBitSetAssignmentInfo:
    selection: _RuntimeBitSetSelection
    operator: str
    shift_count_type: ValueType | None = None

    @property
    def binary_operator(self) -> str | None:
        return None if self.operator == '=' else self.operator[:-1]


@dataclasses.dataclass(frozen=True)
class _RuntimeBitRangeCopyInfo:
    selection: _RuntimeBitRangeSelection
    destination_name: str
    destination_width: int
    # A whole register has neither static positions nor a runtime set.
    destination_indices: tuple[int, ...] | None = None
    runtime_destination: _RuntimeBitSetSelection | None = None
    operator: str = '='

    @property
    def binary_operator(self) -> str | None:
        return None if self.operator == '=' else self.operator[:-1]

    @property
    def width(self) -> int:
        if self.runtime_destination is not None:
            return self.runtime_destination.width
        return self.destination_width if self.destination_indices is None else len(self.destination_indices)


@dataclasses.dataclass(frozen=True)
class _RuntimeBitRangeToRangeCopyInfo:
    destination: _RuntimeBitRangeSelection
    source: _RuntimeBitRangeSelection
    capacity: int
    operator: str = '='

    @property
    def binary_operator(self) -> str | None:
        return None if self.operator == '=' else self.operator[:-1]


@dataclasses.dataclass(frozen=True)
class _RuntimeBitRangeAssignmentInfo:
    selection: _RuntimeBitRangeSelection
    rhs_width: int
    operator: str

    @property
    def binary_operator(self) -> str | None:
        return None if self.operator == '=' else self.operator[:-1]


@dataclasses.dataclass(frozen=True)
class _RuntimeBitRangeShiftAssignmentInfo:
    selection: _RuntimeBitRangeSelection
    operator: str
    shift_count_type: ValueType
    capacity: int

    @property
    def binary_operator(self) -> str:
        return self.operator[:-1]


@dataclasses.dataclass(frozen=True)
class _RuntimeBitSetMeasurementInfo:
    destination: _RuntimeBitSetSelection
    qubit_source_name: str
    qubit_indices: tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class _RuntimeBitRangeMeasurementInfo:
    destination: _RuntimeBitRangeSelection
    qubit_source_name: str
    qubit_indices: tuple[int, ...]

    @property
    def width(self) -> int:
        return len(self.qubit_indices)


@dataclasses.dataclass(frozen=True)
class _RuntimeIteratorBinding:
    # Unlike an unrolled iterator's integer value, this names live QCX storage.
    storage: str
    value_type: ValueType = ValueType.INT
    uint_type: _UnsignedIntegerType | None = None
    angle_type: _AngleType | None = None

    def __post_init__(self) -> None:
        if (self.value_type == ValueType.UINT) != (self.uint_type is not None):
            raise ValueError('UINT iterator bindings require width metadata; other types must not have it')
        if (self.value_type == ValueType.ANGLE) != (self.angle_type is not None):
            raise ValueError('ANGLE iterator bindings require width metadata; other types must not have it')


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
    MAX_RUNTIME_BIT_RANGE_POSITIONS = 1000000
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
        'sin': (_BuiltinFunctionSignature((ValueType.FLOAT,), ValueType.FLOAT, qcx_function='SIN'),
                _BuiltinFunctionSignature((ValueType.ANGLE,), ValueType.FLOAT, qcx_function='SIN')),
        'cos': (_BuiltinFunctionSignature((ValueType.FLOAT,), ValueType.FLOAT, qcx_function='COS'),
                _BuiltinFunctionSignature((ValueType.ANGLE,), ValueType.FLOAT, qcx_function='COS')),
        'tan': (_BuiltinFunctionSignature((ValueType.FLOAT,), ValueType.FLOAT, qcx_function='TAN'),
                _BuiltinFunctionSignature((ValueType.ANGLE,), ValueType.FLOAT, qcx_function='TAN')),
        'arcsin': (_BuiltinFunctionSignature((ValueType.FLOAT,), ValueType.FLOAT, qcx_function='ARCSIN'),),
        'arccos': (_BuiltinFunctionSignature((ValueType.FLOAT,), ValueType.FLOAT, qcx_function='ARCCOS'),),
        'arctan': (_BuiltinFunctionSignature((ValueType.FLOAT,), ValueType.FLOAT, qcx_function='ARCTAN'),),
        'ceiling': (_BuiltinFunctionSignature((ValueType.FLOAT,), ValueType.FLOAT, qcx_function='CEILING'),),
        'floor': (_BuiltinFunctionSignature((ValueType.FLOAT,), ValueType.FLOAT, qcx_function='FLOOR'),),
        'mod': (_BuiltinFunctionSignature((ValueType.INT, ValueType.INT), ValueType.INT, integer_remainder=True),
                _BuiltinFunctionSignature((ValueType.FLOAT, ValueType.FLOAT), ValueType.FLOAT, qcx_function='MOD')),
        'popcount': (_BuiltinFunctionSignature((ValueType.BIT_REGISTER,), ValueType.UINT, population_count=True),
                     _BuiltinFunctionSignature((ValueType.UINT,), ValueType.UINT, population_count=True),
                     _BuiltinFunctionSignature((ValueType.ANGLE,), ValueType.UINT, population_count=True)),
        'rotl': (_BuiltinFunctionSignature((ValueType.BIT_REGISTER, ValueType.INT), ValueType.BIT_REGISTER, rotation=True),
                 _BuiltinFunctionSignature((ValueType.UINT, ValueType.INT), ValueType.UINT, rotation=True),
                 _BuiltinFunctionSignature((ValueType.ANGLE, ValueType.INT), ValueType.ANGLE, rotation=True)),
        'rotr': (_BuiltinFunctionSignature((ValueType.BIT_REGISTER, ValueType.INT), ValueType.BIT_REGISTER, rotation=True),
                 _BuiltinFunctionSignature((ValueType.UINT, ValueType.INT), ValueType.UINT, rotation=True),
                 _BuiltinFunctionSignature((ValueType.ANGLE, ValueType.INT), ValueType.ANGLE, rotation=True)),
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
        self.__angle_array_types: dict[str, _AngleType] = {}
        self.__captured_angle_array_values: dict[int, tuple[ast.IndexExpression, str]] = {}
        self.__captured_bit_register_values: dict[int, tuple[ast.IndexExpression, tuple[str | int, ...]]] = {}
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
        self.__register_equality_index: int = 0
        self.__register_assignment_index: int = 0
        self.__angle_shift_index: int = 0
        self.__rotation_index: int = 0
        self.__angle_trigonometric_index: int = 0
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
            if self.__expression_mentions_angle(expression):
                value_type, _, _ = self.__angle_operand_info(expression)
                if value_type == ValueType.ANGLE:
                    self.__visit_angle_bitwise(expression)
                    return
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
        selection = self.__angle_bit_selection(expression, iterators)
        if selection is not None:
            return selection.width
        if isinstance(expression, ast.Cast) and isinstance(expression.type, ast.BitType) and expression.type.size is not None:
            uint_info = self.__uint_bit_cast_info(expression, iterators)
            if uint_info is not None:
                return uint_info.width
            info = self.__angle_bit_cast_info(expression, iterators)
            if info is not None:
                return info[1]
        if isinstance(expression, ast.Identifier):
            name = expression.name
            shadowed = ((iterators is not None and name in iterators)
                        or any(name in scope for scope in self.__loop_bindings))
            if not shadowed and name in self.__const_bit_register_values:
                return self.__constant_bit_register(name).width
            if not shadowed and name in self.__bit_register_source_sizes:
                return self.__bit_register_source_sizes[name]
        if isinstance(expression, ast.IndexExpression) and self.__is_register_bit_selection(expression):
            runtime_selection = self.__runtime_bit_set_selection(expression, iterators)
            if runtime_selection is not None:
                return runtime_selection.width
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
            if self.__expression_mentions_angle(expression):
                info = self.__angle_shift_info(expression)
                if info is not None:
                    self.__visit_angle_shift(expression, info)
                    return
            self.__visit_shift_binary(expression)
            return
        if expression.op.name in ('&', '|', '^'):
            if self.__expression_mentions_angle(expression):
                value_type, _, _ = self.__angle_operand_info(expression)
                if value_type == ValueType.ANGLE:
                    self.__visit_angle_bitwise(expression)
                    return
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
        if value_type == ValueType.ANGLE:
            return value_type, None, runtime
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
            if (iterators is not None and expression.name in iterators
                    and not any(expression.name in scope for scope in self.__loop_bindings)):
                # A type-only iterator entry cannot reveal its effective
                # width; never borrow a shadowed declaration's UINT width.
                raise UnsupportedOpenQASMError('UINT iterator requires width metadata')
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
            invalid_operand_exception: type[QASM2QCXError], *, constant_indices: bool = False,
            max_indices: int | None = None
            ) -> tuple[list[int], bool]:
        def resolve_index(value):
            return (self.__constant_loop_integer(value, f'{kind} index') if constant_indices
                    else self.__literal_index(value, kind))

        if isinstance(index, ast.DiscreteSet):
            raw_indices = [
                resolve_index(value) for value in index.values
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
                start = resolve_index(selector.start)
                end = resolve_index(selector.end)
                step = (
                    1 if selector.step is None
                    else resolve_index(selector.step))
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
                raw_indices = [resolve_index(selector)]
                is_register = False
        else:
            raise UnsupportedOpenQASMError(
                f'multidimensional {kind} indexing')

        if not raw_indices:
            raise invalid_operand_exception(
                f'{kind.capitalize()} index set is empty')
        if max_indices is not None:
            count = ((abs(raw_indices.stop - raw_indices.start) + abs(raw_indices.step) - 1)
                     // abs(raw_indices.step) if isinstance(raw_indices, range) else len(raw_indices))
            if count > max_indices:
                raise invalid_operand_exception(f'{kind.capitalize()} selection exceeds supported size limit')

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
            destination: bool = False, max_indices: int | None = None) -> _StaticBitSelection:
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
        indices, is_register = self.__static_bit_indices(
            selector, size, name, iterators, destination=destination, max_indices=max_indices)
        return _StaticBitSelection(name, tuple(indices), is_register)

    def __static_bit_indices(
            self, selector, size: int, name: str,
            iterators: set[str] | dict[str, ValueType] | None = None, *,
            destination: bool = False, kind: str = 'bit',
            max_indices: int | None = None) -> tuple[list[int], bool]:
        # Shared normalization for static register/angle selectors, including
        # qubit sources of measurements with runtime classical destinations.
        # Even an unrolled iterator is not a declaration-scope constant. Do
        # not give empty-loop validation a fabricated value for selection bounds.
        bound_names = self.__loop_bound_names(selector) if isinstance(selector, ast.QASMNode) else set().union(
            *(self.__loop_bound_names(value) for value in selector if isinstance(value, ast.QASMNode)))
        iterator_names = set(iterators or ()) | {name for scope in self.__loop_bindings for name in scope}
        if bound_names & iterator_names:
            raise UnsupportedOpenQASMError(f'iterator-dependent {kind}-register selection')
        for bound in bound_names:
            if bound in self.__classical_source_types:
                raise UnsupportedOpenQASMError(f'runtime-dependent {kind}-register selection')
        previous = (self.__expression_kind, self.__value, self.__value_type,
                    self.__value_kind, self.__evaluate_constant)
        try:
            self.__evaluate_constant = True
            indices, is_register = self.__index_selection(
                selector, size, kind, f'variable {name}[{size}]',
                InvalidQubitOperandException if kind == 'qubit' else InvalidBitOperandException,
                constant_indices=kind == 'qubit', max_indices=max_indices)
        finally:
            (self.__expression_kind, self.__value, self.__value_type,
             self.__value_kind, self.__evaluate_constant) = previous
        if destination and len(set(indices)) != len(indices):
            raise InvalidBitOperandException('A bit selection destination must not repeat an index')
        return indices, is_register

    def __angle_bit_index_info(
            self, index: ast.Expression, iterators: set[str] | dict[str, ValueType] | None
            ) -> tuple[ValueType, bool]:
        return self.__bit_index_info(index, iterators, role='angle bit index')

    def __bit_index_info(
            self, index: ast.Expression, iterators: set[str] | dict[str, ValueType] | None, *,
            role: str = 'bit index') -> tuple[ValueType, bool]:
        value_type, runtime = self.__loop_expression_info(index, iterators)
        if value_type not in (ValueType.INT, ValueType.UINT):
            raise UnsupportedOpenQASMError(f'{role} must be an integer')
        names = self.__loop_bound_names(index)
        active_names = set(iterators or ()) | {name for scope in self.__loop_bindings for name in scope}
        self.__validate_bitwise_indices(index, iterators)
        if not self.__is_initialization_process:
            for name in names - active_names:
                if name in self.__constant_source_types:
                    if name not in self.__declared_constant_variables:
                        raise NoVariableNameException(name)
                elif name in self.__classical_source_types:
                    self.__type_of(self.__capitalize_variable_name(name))
        return value_type, runtime or bool(names & active_names)

    def __runtime_bit_set_selection(
            self, operand: ast.QASMNode,
            iterators: set[str] | dict[str, ValueType] | None = None, *,
            destination: bool = False) -> _RuntimeBitSetSelection | None:
        # A fixed entry count gives a register width without evaluating indices.
        if (destination and isinstance(operand, ast.IndexedIdentifier) and len(operand.indices) != 1
                and isinstance(self.__constant_source_types.get(operand.name.name)
                               or self.__classical_source_types.get(operand.name.name), ast.BitType)):
            raise UnsupportedOpenQASMError('further indexing of a bit-register assignment destination')
        if (isinstance(operand, ast.IndexExpression)
                and isinstance(operand.collection, ast.Identifier)
                and isinstance(operand.index, ast.DiscreteSet)):
            name, selector = operand.collection.name, operand.index
        elif (destination and isinstance(operand, ast.IndexedIdentifier)
              and len(operand.indices) == 1 and isinstance(operand.indices[0], ast.DiscreteSet)):
            name, selector = operand.name.name, operand.indices[0]
        else:
            return None
        source_type = self.__constant_source_types.get(name) or self.__classical_source_types.get(name)
        if not isinstance(source_type, ast.BitType):
            return None
        if ((iterators is not None and name in iterators)
                or any(name in scope for scope in self.__loop_bindings)):
            raise UnsupportedOpenQASMError('bit selection of a for-loop iterator')
        if destination:
            self.__reject_constant_bit_target(ast.Identifier(name), iterators)
        info = tuple(self.__bit_index_info(index, iterators) for index in selector.values)
        if not any(runtime for _, runtime in info):
            return None
        if source_type.size is None:
            raise UnsupportedOpenQASMError('runtime set selection requires a bit-register source')
        constant = self.__constant_bit_register(name) if name in self.__const_bit_register_values else None
        if (not self.__is_initialization_process and constant is None
                and self.__capitalize_variable_name(name) not in self.__sized_bit_variables):
            raise NoVariableNameException(name)
        size = constant.width if constant is not None else self.__bit_register_source_sizes[name]
        if not 1 <= size <= self.QCX_INT_MAX:
            raise InvalidBitOperandException('Runtime selection source width must fit QCX INT indexing')
        if len(info) > self.QCX_INT_MAX:
            raise InvalidBitOperandException('Runtime selection width exceeds QCX INT range')
        return _RuntimeBitSetSelection(name, size, tuple(selector.values),
                                       tuple(value_type for value_type, _ in info))

    def __runtime_angle_bit_selector(
            self, source: ast.Identifier | ast.IndexExpression, angle_type: _AngleType,
            selector, iterators: set[str] | dict[str, ValueType] | None
            ) -> _RuntimeAngleBitSelection | _RuntimeAngleBitSetSelection | None:
        if isinstance(selector, ast.DiscreteSet):
            # A mixed set retains every expression, including constants. Its
            # bounds and domains are checked when the entries are captured;
            # metadata must not evaluate values in an empty enclosing loop.
            info = tuple(self.__angle_bit_index_info(index, iterators) for index in selector.values)
            if any(runtime for _, runtime in info):
                return _RuntimeAngleBitSetSelection(
                    source, angle_type, tuple(selector.values), tuple(value_type for value_type, _ in info))
        elif (isinstance(selector, list) and len(selector) == 1
                and isinstance(selector[0], ast.Expression)
                and not isinstance(selector[0], (ast.RangeDefinition, ast.DiscreteSet))):
            index = selector[0]
            value_type, runtime = self.__angle_bit_index_info(index, iterators)
            if runtime:
                return _RuntimeAngleBitSelection(source, angle_type, index, value_type)
        return None

    def __angle_bit_selection(
            self, operand: ast.QASMNode,
            iterators: set[str] | dict[str, ValueType] | None = None, *,
            destination: bool = False,
            allow_runtime_range: bool = False) -> _AngleBitLoopSelection | None:
        if isinstance(operand, ast.IndexExpression):
            source, selector = operand.collection, operand.index
        elif isinstance(operand, ast.IndexedIdentifier):
            if not operand.indices:
                return None
            source = operand.name
            for index in operand.indices[:-1]:
                source = ast.IndexExpression(source, index)
            selector = operand.indices[-1]
        else:
            return None
        if isinstance(source, ast.IndexExpression) and self.__angle_bit_selection(source, iterators) is not None:
            raise UnsupportedOpenQASMError('further indexing of an angle bit selection')
        if isinstance(source, ast.Identifier):
            name = source.name
            active = ((iterators is not None and name in iterators)
                      or any(name in scope for scope in self.__loop_bindings))
            if active:
                if self.__loop_identifier_info(source, iterators)[0] != ValueType.ANGLE:
                    return None
            elif (name not in self.__const_angle_values
                  and not isinstance(self.__classical_source_types.get(name), ast.AngleType)):
                return None  # Includes an angle-array element without a bit subscript.
        elif isinstance(source, ast.IndexExpression) and isinstance(source.collection, ast.Identifier):
            name = source.collection.name
            active = ((iterators is not None and name in iterators)
                      or any(name in scope for scope in self.__loop_bindings))
            source_type = self.__classical_source_types.get(name)
            if (not isinstance(source_type, ast.ArrayType)
                    or not isinstance(source_type.base_type, ast.AngleType)):
                return None
            if active:
                raise UnsupportedOpenQASMError('indexed use of a for-loop iterator')
        else:
            return None
        _, angle_type, runtime = self.__angle_operand_info(source, iterators)
        if not angle_type.is_sized:
            raise UnsupportedOpenQASMError('bit selection requires an explicitly sized angle')
        if destination:
            if active:
                raise UnsupportedOpenQASMError('assignment to a for-loop iterator')
            if name in self.__const_angle_values:
                self.__reject_constant_angle_target(ast.Identifier(name), iterators)
        if allow_runtime_range:
            range_selection = self.__runtime_angle_bit_range(source, angle_type, selector, iterators)
            if range_selection is not None:
                return range_selection
        runtime_selection = self.__runtime_angle_bit_selector(source, angle_type, selector, iterators)
        if runtime_selection is not None:
            return runtime_selection
        indices, _ = self.__static_bit_indices(selector, angle_type.width, name, iterators, destination=destination)
        # Unlike a bit-register index, even a single angle bit is bit[1].
        return _StaticAngleBitSelection(source, angle_type, tuple(indices), runtime)

    def __runtime_angle_bit_range(
            self, source: ast.Identifier | ast.IndexExpression, angle_type: _AngleType,
            selector, iterators: set[str] | dict[str, ValueType] | None
            ) -> _RuntimeAngleBitRangeSelection | None:
        if not (isinstance(selector, list) and len(selector) == 1
                and isinstance(selector[0], ast.RangeDefinition)):
            return None
        bounds = selector[0]
        if bounds.start is None or bounds.end is None:
            raise InvalidLoopRangeException('Angle bit range requires both bounds')
        # Source order is start, optional step, stop. Even constants stay as
        # ASTs in a runtime range: domains and bounds are checked on entry,
        # never while validating a skipped enclosing loop.
        start_type, start_runtime = self.__angle_bit_index_info(bounds.start, iterators)
        step_type, step_runtime = ((ValueType.INT, False) if bounds.step is None
                                  else self.__angle_bit_index_info(bounds.step, iterators))
        stop_type, stop_runtime = self.__angle_bit_index_info(bounds.end, iterators)
        if not (start_runtime or step_runtime or stop_runtime):
            return None  # Preserve the existing static range path.
        return _RuntimeAngleBitRangeSelection(
            source, angle_type, bounds, start_type, step_type, stop_type)

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
    def __qcx_bit_name(variable_name: str, size: int, index: int | str) -> str:
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
        runtime_range = self.__runtime_bit_range_measurement_info(measurement, target)
        if runtime_range is not None:
            self.__emit_runtime_bit_range_measurement(runtime_range)
            return
        runtime_target = self.__runtime_bit_set_measurement_info(measurement, target)
        if runtime_target is not None:
            self.__emit_runtime_bit_set_measurement(runtime_target)
            return
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
            if self.__angle_bit_cast_info(expression, iterators) is not None:
                return None  # Executable register reinterpretation, not an angle copy.
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
        if self.__angle_bit_selection(expression) is not None:
            raise InvalidBitOperandException('An angle bit selection requires a bit-register context')
        if isinstance(expression, ast.IndexExpression):
            captured = self.__captured_angle_array_values.get(id(expression))
            if captured is not None:
                return captured[1], ValueType.ANGLE, ValueKind.LVALUE
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
                    storage_type = ValueType.UINT if element_type == ValueType.ANGLE else element_type
                    result = self.__add_new_temporary_variable(storage_type, uint_type)
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

    def __runtime_angle_range_register_assignment_info(
            self, target: ast.Identifier | ast.IndexedIdentifier, expression: ast.Expression,
            iterators: set[str] | dict[str, ValueType] | None = None
            ) -> _RuntimeAngleRangeRegisterAssignment | None:
        # This narrow path is for direct whole-register copies. Nested
        # expressions and selected destinations retain their existing paths.
        if not isinstance(target, ast.Identifier):
            return None
        selection = self.__angle_bit_selection(expression, iterators, allow_runtime_range=True)
        if not isinstance(selection, _RuntimeAngleBitRangeSelection):
            return None
        width = self.__runtime_register_assignment_destination_width(target, iterators)
        capacity = min(width, 2 * selection.angle_type.width, self.MAX_RUNTIME_BIT_RANGE_POSITIONS)
        return _RuntimeAngleRangeRegisterAssignment(target, width, selection, capacity)

    def __runtime_angle_range_expression_assignment_info(
            self, target: ast.Identifier | ast.IndexedIdentifier, expression: ast.Expression,
            iterators: set[str] | dict[str, ValueType] | None = None) -> _RuntimeBitRegisterAssignment | None:
        # Direct copies retain their separate count-before-extraction path.
        if (not isinstance(target, ast.Identifier) or isinstance(expression, ast.IndexExpression)
                or not self.__contains_angle_bit_range(expression, iterators)):
            return None
        shapes = self.__runtime_bit_register_expression_shapes(expression, iterators, allow_angle_ranges=True)
        if shapes[id(expression)] < 0:
            return None
        width = self.__runtime_register_assignment_destination_width(target, iterators)
        value = self.__runtime_bit_register_expression_info(expression, shapes, iterators, allow_angle_ranges=True)
        return _RuntimeBitRegisterAssignment(target, width, value)

    def __runtime_register_assignment_info(
            self, target: ast.Identifier | ast.IndexedIdentifier, expression: ast.Expression,
            iterators: set[str] | dict[str, ValueType] | None = None
            ) -> _RuntimeBitRegisterAssignment | _RuntimeBitRangeExpressionAssignment | None:
        # Ordinary '=' assignments and declaration initializers share RHS snapshots.
        if not isinstance(target, ast.Identifier):
            runtime_range = self.__runtime_range_expression_assignment_info(target, expression, iterators)
            if runtime_range is not None:
                return runtime_range
            runtime_set = self.__runtime_set_expression_assignment_info(target, expression, iterators)
            if runtime_set is not None:
                return runtime_set
            return self.__runtime_static_selection_assignment_info(target, expression, iterators)
        shapes = self.__runtime_bit_register_expression_shapes(expression, iterators)
        if shapes[id(expression)] < 0:
            return None  # Preserve fixed-width and scalar assignment analysis.
        width = self.__runtime_register_assignment_destination_width(target, iterators)
        value = self.__runtime_bit_register_expression_info(expression, shapes, iterators)
        # The destination width is a required result length, never an inferred
        # width for dynamic leaves or permission to pad/truncate their values.
        return _RuntimeBitRegisterAssignment(target, width, value)

    def __runtime_static_selection_assignment_info(
            self, target: ast.Identifier | ast.IndexedIdentifier, expression: ast.Expression,
            iterators: set[str] | dict[str, ValueType] | None = None) -> _RuntimeBitRegisterAssignment | None:
        if not isinstance(target, ast.IndexedIdentifier) or not self.__is_register_bit_selection(target):
            return None  # Whole-register and scalar destinations have separate paths.
        shapes = self.__runtime_bit_register_expression_shapes(expression, iterators)
        if shapes[id(expression)] < 0:
            return None  # Preserve the existing fixed-width selection paths.
        self.__runtime_register_assignment_destination_width(target.name, iterators)
        selection = self.__static_bit_selection(
            target, iterators, destination=True,
            max_indices=min(self.QCX_INT_MAX, self.MAX_RUNTIME_BIT_RANGE_POSITIONS))
        value = self.__runtime_bit_register_expression_info(expression, shapes, iterators)
        return _RuntimeBitRegisterAssignment(target, selection.width, value, selection)

    def __runtime_angle_range_static_selection_assignment_info(
            self, target: ast.Identifier | ast.IndexedIdentifier, expression: ast.Expression,
            iterators: set[str] | dict[str, ValueType] | None = None) -> _RuntimeBitRegisterAssignment | None:
        # Only the new angle-range RHS and static register selections belong
        # here. Other destinations and RHS contexts retain their own paths.
        if (not isinstance(target, ast.IndexedIdentifier) or not self.__is_register_bit_selection(target)
                or not self.__contains_angle_bit_range(expression, iterators)):
            return None
        shapes = self.__runtime_bit_register_expression_shapes(expression, iterators, allow_angle_ranges=True)
        if shapes[id(expression)] < 0:
            return None
        if (self.__runtime_bit_range_destination(target, iterators) is not None
                or self.__runtime_bit_set_selection(target, iterators, destination=True) is not None):
            return None
        self.__runtime_register_assignment_destination_width(target.name, iterators)
        selection = self.__static_bit_selection(
            target, iterators, destination=True,
            max_indices=min(self.QCX_INT_MAX, self.MAX_RUNTIME_BIT_RANGE_POSITIONS))
        value = self.__runtime_bit_register_expression_info(expression, shapes, iterators, allow_angle_ranges=True)
        return _RuntimeBitRegisterAssignment(target, selection.width, value, selection)

    def __runtime_set_expression_assignment_info(
            self, target: ast.Identifier | ast.IndexedIdentifier, expression: ast.Expression,
            iterators: set[str] | dict[str, ValueType] | None = None) -> _RuntimeBitRegisterAssignment | None:
        # A set has a known entry count even when its destination positions
        # require runtime capture. Other target forms keep their own paths.
        if (not isinstance(target, ast.IndexedIdentifier) or len(target.indices) != 1
                or not isinstance(target.indices[0], ast.DiscreteSet)):
            return None
        shapes = self.__runtime_bit_register_expression_shapes(expression, iterators)
        if shapes[id(expression)] < 0:
            return None  # Preserve fixed-width RHS handling, including scalar counts.
        self.__runtime_register_assignment_destination_width(target.name, iterators)
        selection = self.__runtime_bit_set_selection(target, iterators, destination=True)
        if selection is None:
            return None  # A wholly static set belongs to static-selection analysis.
        if not 1 <= selection.width <= self.MAX_RUNTIME_BIT_RANGE_POSITIONS:
            raise InvalidBitOperandException('Runtime expression assignment selection width exceeds storage limit')
        value = self.__runtime_bit_register_expression_info(expression, shapes, iterators)
        return _RuntimeBitRegisterAssignment(target, selection.width, value, selection)

    def __runtime_angle_range_set_expression_assignment_info(
            self, target: ast.Identifier | ast.IndexedIdentifier, expression: ast.Expression,
            iterators: set[str] | dict[str, ValueType] | None = None) -> _RuntimeBitRegisterAssignment | None:
        if (not isinstance(target, ast.IndexedIdentifier) or len(target.indices) != 1
                or not isinstance(target.indices[0], ast.DiscreteSet)
                or not self.__contains_angle_bit_range(expression, iterators)):
            return None
        shapes = self.__runtime_bit_register_expression_shapes(expression, iterators, allow_angle_ranges=True)
        if shapes[id(expression)] < 0:
            return None
        self.__runtime_register_assignment_destination_width(target.name, iterators)
        selection = self.__runtime_bit_set_selection(target, iterators, destination=True)
        if selection is None:
            return None  # Static destinations retain their existing path.
        if not 1 <= selection.width <= self.MAX_RUNTIME_BIT_RANGE_POSITIONS:
            raise InvalidBitOperandException('Runtime expression assignment selection width exceeds storage limit')
        value = self.__runtime_bit_register_expression_info(expression, shapes, iterators, allow_angle_ranges=True)
        return _RuntimeBitRegisterAssignment(target, selection.width, value, selection)

    def __runtime_range_expression_assignment_info(
            self, target: ast.Identifier | ast.IndexedIdentifier, expression: ast.Expression,
            iterators: set[str] | dict[str, ValueType] | None = None) -> _RuntimeBitRangeExpressionAssignment | None:
        if (not isinstance(target, ast.IndexedIdentifier) or len(target.indices) != 1
                or not isinstance(target.indices[0], list) or len(target.indices[0]) != 1
                or not isinstance(target.indices[0][0], ast.RangeDefinition)):
            return None
        shapes = self.__runtime_bit_register_expression_shapes(expression, iterators)
        if shapes[id(expression)] < 0:
            return None  # Fixed-width RHS expressions retain their existing path.
        self.__runtime_register_assignment_destination_width(target.name, iterators)
        selection = self.__runtime_bit_range_destination(target, iterators)
        if selection is None:
            return None  # Wholly static ranges retain static-selection analysis.
        capacity = min(selection.source_width, self.MAX_RUNTIME_BIT_RANGE_POSITIONS)
        if not 1 <= capacity <= self.QCX_INT_MAX:
            raise InvalidBitOperandException('Runtime range expression assignment capacity must fit QCX INT indexing')
        # Analyze the RHS independently: destination storage/count cannot
        # constrain a dynamic leaf's length or suppress its evaluation.
        value = self.__runtime_bit_register_expression_info(expression, shapes, iterators)
        return _RuntimeBitRangeExpressionAssignment(target, selection, capacity, value)

    def __runtime_angle_range_range_expression_assignment_info(
            self, target: ast.Identifier | ast.IndexedIdentifier, expression: ast.Expression,
            iterators: set[str] | dict[str, ValueType] | None = None) -> _RuntimeBitRangeExpressionAssignment | None:
        if (not isinstance(target, ast.IndexedIdentifier) or len(target.indices) != 1
                or not isinstance(target.indices[0], list) or len(target.indices[0]) != 1
                or not isinstance(target.indices[0][0], ast.RangeDefinition)
                or not self.__contains_angle_bit_range(expression, iterators)):
            return None
        shapes = self.__runtime_bit_register_expression_shapes(expression, iterators, allow_angle_ranges=True)
        if shapes[id(expression)] < 0:
            return None
        self.__runtime_register_assignment_destination_width(target.name, iterators)
        selection = self.__runtime_bit_range_destination(target, iterators)
        if selection is None:
            return None  # Static destinations retain their existing path.
        capacity = min(selection.source_width, self.MAX_RUNTIME_BIT_RANGE_POSITIONS)
        if not 1 <= capacity <= self.QCX_INT_MAX:
            raise InvalidBitOperandException('Runtime range expression assignment capacity must fit QCX INT indexing')
        # Destination storage bounds neither the length nor the evaluation of
        # an angle-range leaf or any other RHS operand.
        value = self.__runtime_bit_register_expression_info(expression, shapes, iterators, allow_angle_ranges=True)
        return _RuntimeBitRangeExpressionAssignment(target, selection, capacity, value)

    def __runtime_register_compound_assignment_info(
            self, statement: ast.ClassicalAssignment,
            iterators: set[str] | dict[str, ValueType] | None = None
            ) -> _RuntimeBitRegisterAssignment | _RuntimeBitRangeExpressionAssignment | None:
        operator = statement.op.name
        if operator not in ('&=', '|=', '^='):
            return None
        if self.__runtime_bit_range_source(statement.rvalue, iterators) is not None:
            return None  # Preserve the direct-range path and its check timing.
        info = self.__runtime_register_assignment_info(statement.lvalue, statement.rvalue, iterators)
        if info is None:
            return None  # Fixed-width and scalar RHS expressions keep their paths.
        # Original LHS bits need a bounded snapshot as well as the independent
        # RHS tree. Runtime-range destinations already have a bounded capacity.
        if (isinstance(info, _RuntimeBitRegisterAssignment)
                and not 1 <= info.width <= self.MAX_RUNTIME_BIT_RANGE_POSITIONS):
            raise InvalidBitOperandException('Runtime register compound snapshot exceeds storage limit')
        return dataclasses.replace(info, operator=operator)

    def __runtime_angle_range_register_compound_assignment_info(
            self, statement: ast.ClassicalAssignment,
            iterators: set[str] | dict[str, ValueType] | None = None) -> _RuntimeBitRegisterAssignment | None:
        if (statement.op.name not in ('&=', '|=', '^=') or not isinstance(statement.lvalue, ast.Identifier)
                or not self.__contains_angle_bit_range(statement.rvalue, iterators)):
            return None
        shapes = self.__runtime_bit_register_expression_shapes(statement.rvalue, iterators, allow_angle_ranges=True)
        if shapes[id(statement.rvalue)] < 0:
            return None
        width = self.__runtime_register_assignment_destination_width(statement.lvalue, iterators)
        if not 1 <= width <= self.MAX_RUNTIME_BIT_RANGE_POSITIONS:
            raise InvalidBitOperandException('Runtime register compound snapshot exceeds storage limit')
        value = self.__runtime_bit_register_expression_info(statement.rvalue, shapes, iterators, allow_angle_ranges=True)
        return _RuntimeBitRegisterAssignment(statement.lvalue, width, value, operator=statement.op.name)

    def __runtime_register_assignment_destination_width(
            self, target: ast.Identifier,
            iterators: set[str] | dict[str, ValueType] | None) -> int:
        name = target.name
        if ((iterators is not None and name in iterators)
                or any(name in scope for scope in self.__loop_bindings)):
            raise UnsupportedOpenQASMError('assignment to a for-loop iterator')
        self.__reject_constant_bit_target(target, iterators)
        source_type = self.__classical_source_types.get(name)
        if source_type is None and name not in self.__constant_source_types:
            raise NoVariableNameException(name)
        if not isinstance(source_type, ast.BitType) or source_type.size is None:
            raise InvalidBitOperandException('Runtime register assignment requires a mutable bit-register destination')
        if (not self.__is_initialization_process
                and self.__capitalize_variable_name(name) not in self.__sized_bit_variables):
            raise NoVariableNameException(name)
        # Width was resolved in declaration scope. Do not re-evaluate the
        # type's size expression under an iterator that shadows a constant.
        width = self.__bit_register_source_sizes[name]
        if not 1 <= width <= self.QCX_INT_MAX:
            raise InvalidBitOperandException('Runtime register assignment destination width must fit QCX INT indexing')
        return width

    def __runtime_register_comparison_info(
            self, expression: ast.BinaryExpression,
            iterators: set[str] | dict[str, ValueType] | None = None) -> _RuntimeBitRegisterComparison | None:
        # Probe only register-valued branches. Scalar distance arguments are
        # opaque, even when they contain popcount calls on runtime ranges.
        left_shapes = self.__runtime_bit_register_expression_shapes(expression.lhs, iterators)
        right_shapes = self.__runtime_bit_register_expression_shapes(expression.rhs, iterators)
        if left_shapes[id(expression.lhs)] < 0 and right_shapes[id(expression.rhs)] < 0:
            return None  # Preserve the existing scalar and fixed-width paths.
        left = self.__runtime_bit_register_expression_info(expression.lhs, left_shapes, iterators)
        right = self.__runtime_bit_register_expression_info(expression.rhs, right_shapes, iterators)
        if expression.op.name not in ('==', '!='):
            raise UnsupportedOpenQASMError('ordered comparisons of bit-register values')
        # Only fixed leaves have known widths. Capacities bound storage, never
        # lengths; dynamic length equality must be asserted by the consumer.
        return _RuntimeBitRegisterComparison(expression, left, right)

    def __register_comparison_info(
            self, expression: ast.BinaryExpression,
            iterators: set[str] | dict[str, ValueType] | None = None) -> int | _RuntimeBitRegisterComparison | None:
        runtime = self.__runtime_register_comparison_info(expression, iterators)
        return runtime if runtime is not None else self.__register_comparison_width(expression, iterators)

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

        # Register shift/rotation counts may contain angle quotients without
        # making the comparison operands scalar angles.
        register_info = self.__register_comparison_info(condition)
        if register_info is not None:
            if isinstance(register_info, _RuntimeBitRegisterComparison):
                self.__emit_runtime_register_equality_condition(register_info, true_label, false_label)
            else:
                self.__emit_register_equality_condition(condition, true_label, false_label)
            return
        if self.__expression_mentions_angle(condition):
            angle_info = self.__angle_comparison_info(condition)
            if angle_info is not None:
                self.__emit_angle_comparison_condition(
                    condition, angle_info, operators[condition.op], true_label, false_label)
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

    def __emit_angle_comparison_condition(
            self, expression: ast.BinaryExpression, info: tuple[_AngleType, _AngleType, bool],
            operator: str, true_label: str, false_label: str) -> None:
        lhs_type, rhs_type, _ = info
        common = _AngleType(max(lhs_type.width, rhs_type.width), lhs_type.is_sized or rhs_type.is_sized)
        previous_kind = self.__expression_kind
        previous_used = self.__used_temporary_variables.copy()
        try:
            self.__expression_kind = ExpressionKind.ARITHMETIC
            # Snapshot left before evaluating right. Widen the fractional
            # angle representation with low zero bits, never UINT promotion.
            lhs = self.__angle_arithmetic_operand(expression.lhs, common)
            rhs = self.__angle_arithmetic_operand(expression.rhs, common)
            if isinstance(lhs, int):
                # JUMPIF requires a variable on its left, even for constants.
                temporary = self.__add_new_temporary_variable(
                    ValueType.UINT, _UnsignedIntegerType(common.width, common.is_sized))
                self.__qcx_lines.append(f'LET {temporary} := {lhs}')
                lhs = temporary
            self.__qcx_lines.extend([
                f'JUMPIF {true_label} {lhs} {operator} {rhs}', f'JUMP {false_label}'])
        finally:
            self.__expression_kind = previous_kind
            for temporary in self.__used_temporary_variables - previous_used:
                self.__release_temporary_variable(temporary)

    def __emit_runtime_register_equality_condition(
            self, info: _RuntimeBitRegisterComparison, true_label: str, false_label: str) -> None:
        previous_used = self.__used_temporary_variables.copy()
        previous_declarations = self.__declared_temporary_variables.copy()
        prefix = f'QASM2QCX_REGISTER_EQUALITY_{self.__register_equality_index}'
        self.__register_equality_index += 1
        mismatch = false_label if info.expression.op.name == '==' else true_label
        match = true_label if info.expression.op.name == '==' else false_label
        try:
            with contextlib.ExitStack() as operands:
                left, left_count = operands.enter_context(
                    self.__runtime_bit_register_expression_value(info.left, f'{prefix}_LEFT'))
                right, right_count = operands.enter_context(
                    self.__runtime_bit_register_expression_value(info.right, f'{prefix}_RIGHT'))
                # Complete both operands, including all runtime checks, before
                # checking their lengths or exiting on a mismatching bit.
                index = self.__add_new_temporary_variable(ValueType.INT)
                self.__qcx_lines.extend([
                    f'ASSERT {left_count} == {right_count}', f'LET {index} := 0',
                    f'@{prefix}_COMPARE', f'JUMPIF {match} {index} >= {left_count}',
                    f'JUMPIF {mismatch} {left}:{index} \\= {right}:{index}',
                    f'LET {index} += 1', f'JUMP {prefix}_COMPARE',
                ])
                self.__check_runtime_register_output()
            self.__hoist_temporary_declarations(previous_declarations)
        finally:
            for temporary in self.__used_temporary_variables - previous_used:
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
        selection = self.__angle_bit_selection(expression)
        if selection is not None:
            if selection.runtime:
                raise NoConstantExpressionException
            constant = self.__const_angle_values[selection.source.name]
            return (sum(((constant.value >> source) & 1) << target
                        for target, source in enumerate(selection.indices))
                    if self.__evaluate_constant else 0)
        if isinstance(expression, ast.Cast):
            uint_info = self.__uint_bit_cast_info(expression)
            if uint_info is not None and uint_info.result_type == ValueType.BIT_REGISTER:
                self.__visit_constant_uint_bit_cast(expression, uint_info)
                return int(self.__value)
            info = self.__angle_bit_cast_info(expression)
            if info is not None and info[0] == ValueType.BIT_REGISTER:
                self.__visit_constant_angle_bit_cast(expression, info)
                return int(self.__value)
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
            angle_comparison = None
            if expression.op.name in ('==', '!=', '<', '<=', '>', '>='):
                width = self.__register_comparison_info(expression)
                if isinstance(width, _RuntimeBitRegisterComparison):
                    raise NoConstantExpressionException
                if width is not None:
                    lhs = self.__constant_bit_register_value(expression.lhs, width)
                    rhs = self.__constant_bit_register_value(expression.rhs, width)
                    equal = lhs == rhs
                    self.__value = int(equal if expression.op.name == '==' else not equal) if self.__evaluate_constant else 0
                    self.__value_type, self.__value_kind = ValueType.BOOL, ValueKind.LITERAL
                    return
                if self.__expression_mentions_angle(expression):
                    angle_comparison = self.__angle_comparison_info(expression)
                    if angle_comparison is not None and angle_comparison[2]:
                        raise NoConstantExpressionException
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
                if angle_comparison is not None:
                    lhs_angle, rhs_angle, _ = angle_comparison
                    width = max(lhs_angle.width, rhs_angle.width)
                    lhs_value = int(lhs_value) << (width - lhs_angle.width)
                    rhs_value = int(rhs_value) << (width - rhs_angle.width)
                    result_type = ValueType.UINT
                else:
                    _, _, result_type = self.__comparison_types(
                        lhs_type, rhs_type, expression.op)
                if angle_comparison is None and ValueType.UINT in (lhs_type, rhs_type):
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
        selection = self.__angle_bit_selection(expression, iterators)
        if selection is not None:
            return ValueType.BIT_REGISTER, selection.runtime
        if self.__runtime_bit_set_selection(expression, iterators) is not None:
            return ValueType.BIT_REGISTER, True
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
                if source_type == ValueType.ANGLE:
                    self.__angle_operand_info(argument, iterators)
                elif source_type != ValueType.UINT or uint_type is None or not uint_type.is_sized:
                    raise InvalidFunctionCallException(f'Function {name} requires bit[n], sized uint[n], or angle')
            distance_type, distance_runtime = self.__loop_expression_info(distance, iterators)
            signature = self.__resolve_builtin_function(name, (source_type, distance_type))
            self.__validate_bitwise_indices(distance, iterators)
            return signature, source_runtime or distance_runtime
        if overloads[0].population_count:
            argument = expression.arguments[0]
            # Runtime ranges and their supported operations have no inferred
            # register width. Validate metadata without evaluating positions;
            # capture and counting happen when reached.
            if self.__runtime_bit_range_popcount_info(argument, iterators) is not None:
                return overloads[0], True
            width = self.__bit_register_expression_width(argument, iterators)
            if width is None:
                source_type, uint_type, runtime = self.__integer_operand_info(argument, iterators)
                if source_type == ValueType.ANGLE:
                    self.__angle_operand_info(argument, iterators)
                    return self.__resolve_builtin_function(name, (ValueType.ANGLE,)), runtime
                if source_type != ValueType.UINT or uint_type is None or not uint_type.is_sized:
                    raise InvalidFunctionCallException(
                        'Function popcount requires a bit-register operand, sized uint[n], or angle')
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
        uint_info = self.__uint_bit_cast_info(expression, iterators)
        if uint_info is not None:
            return uint_info.result_type, uint_info.runtime
        info = self.__angle_bit_cast_info(expression, iterators)
        if info is not None:
            return info[0], info[2]
        if isinstance(expression.type, ast.UintType):
            self.__unsigned_integer_type(expression.type, iterators)
        operand_type, runtime = self.__loop_expression_info(expression.argument, iterators)
        if operand_type == ValueType.BIT_REGISTER:
            # Supported same-width reinterpretations were resolved above. Register
            # values must not fall through scalar numeric/Boolean cast typing.
            raise UnsupportedOpenQASMError('bit-register cast outside supported angle conversion')
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
        if expression.op.name == '~' and self.__expression_mentions_angle(expression, iterators):
            value_type, _, runtime = self.__angle_operand_info(expression, iterators)
            return value_type, runtime
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
        if expression.op.name in ('<<', '>>') and self.__expression_mentions_angle(expression, iterators):
            info = self.__angle_shift_info(expression, iterators)
            if info is not None:
                return ValueType.ANGLE, info[2]
        if expression.op.name in ('&', '|', '^') and self.__expression_mentions_angle(expression, iterators):
            value_type, _, runtime = self.__angle_operand_info(expression, iterators)
            return value_type, runtime
        if expression.op.name in ('==', '!=', '<', '<=', '>', '>='):
            register_info = self.__register_comparison_info(expression, iterators)
            if isinstance(register_info, _RuntimeBitRegisterComparison):
                return ValueType.BOOL, True
            if register_info is not None:
                lhs_runtime = self.__register_expression_runtime(expression.lhs, iterators)
                rhs_runtime = self.__register_expression_runtime(expression.rhs, iterators)
                return ValueType.BOOL, lhs_runtime or rhs_runtime
        if (expression.op.name in ('==', '!=', '<', '<=', '>', '>=')
                and self.__expression_mentions_angle(expression, iterators)):
            info = self.__angle_comparison_info(expression, iterators)
            if info is not None:
                return ValueType.BOOL, info[2]
        if expression.op.name in ('*', '/') and self.__expression_mentions_angle(expression, iterators):
            info = self.__angle_multiplicative_info(expression, iterators)
            if info is not None:
                return info.result_type, info.runtime
        if expression.op.name in ('+', '-') and self.__expression_mentions_angle(expression, iterators):
            value_type, _, runtime = self.__angle_operand_info(expression, iterators)
            return value_type, runtime
        if expression.op.name == '**':
            value_type, _, runtime = self.__power_expression_info(expression, iterators)
            return value_type, runtime
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
        selection = self.__angle_bit_selection(expression, iterators)
        if selection is not None:
            return selection.runtime
        if isinstance(expression, ast.Cast):
            uint_info = self.__uint_bit_cast_info(expression, iterators)
            if uint_info is not None:
                return uint_info.runtime
            info = self.__angle_bit_cast_info(expression, iterators)
            if info is not None:
                return info[2]
        if isinstance(expression, ast.BitstringLiteral):
            return False
        if isinstance(expression, ast.Identifier):
            if expression.name in self.__const_bit_register_values:
                self.__constant_bit_register(expression.name)
                return False
            return True
        if isinstance(expression, ast.IndexExpression):
            if self.__runtime_bit_set_selection(expression, iterators) is not None:
                return True
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
            if isinstance(variable_type.base_type, ast.AngleType):
                variable_name = self.__capitalize_variable_name(name)
                if not self.__is_initialization_process and variable_name not in self.__array_element_types:
                    raise NoVariableNameException(name)
                self.__validate_bitwise_indices(expression, iterators)
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

    def __angle_selection_loop_info(
            self, source: ast.QASMNode, iteration_type: ast.QASMNode,
            iterators: set[str] | dict[str, ValueType] | None = None
            ) -> _AngleBitLoopSelection | None:
        # Reuse selection metadata without reading the source or capturing
        # either index. Resolve it before introducing the loop's own binding.
        selection = self.__angle_bit_selection(source, iterators, allow_runtime_range=True)
        if selection is not None and (
                not isinstance(iteration_type, ast.BitType) or iteration_type.size is not None):
            raise UnsupportedOpenQASMError('angle bit-selection iteration requires a scalar bit iterator')
        if (selection is not None and not isinstance(selection, _RuntimeAngleBitRangeSelection)
                and selection.width > self.QCX_INT_MAX):
            raise InvalidLoopRangeException('Angle bit-selection loop size exceeds QCX INT range')
        return selection

    def __bit_selection_loop_info(
            self, source: ast.QASMNode, iteration_type: ast.QASMNode,
            iterators: set[str] | dict[str, ValueType] | None = None
            ) -> _BitSelectionLoop | None:
        # Whole registers use collection-loop metadata. Only named register
        # selections belong here; angle selections retain their separate path.
        if not (isinstance(source, ast.IndexExpression) and isinstance(source.collection, ast.Identifier)):
            return None
        name = source.collection.name
        source_type = self.__constant_source_types.get(name) or self.__classical_source_types.get(name)
        if not isinstance(source_type, ast.BitType):
            return None
        if source_type.size is None:
            raise UnsupportedOpenQASMError('bit-selection iteration requires a bit-register source')
        if ((iterators is not None and name in iterators)
                or any(name in scope for scope in self.__loop_bindings)):
            raise UnsupportedOpenQASMError('bit selection of a for-loop iterator')
        constant = self.__constant_bit_register(name) if name in self.__const_bit_register_values else None
        if (not self.__is_initialization_process and constant is None
                and self.__capitalize_variable_name(name) not in self.__sized_bit_variables):
            raise NoVariableNameException(name)
        size = constant.width if constant is not None else self.__bit_register_source_sizes[name]
        selection = None
        if isinstance(source.index, ast.DiscreteSet):
            entries = source.index.values
            info = tuple(self.__bit_index_info(index, iterators) for index in entries)
            if any(runtime for _, runtime in info):
                if len(entries) == 1:
                    selection = _RuntimeSingletonBitSelection(name, size, entries[0], info[0][0])
                else:
                    selection = _RuntimeBitSetSelection(
                        name, size, tuple(entries), tuple(value_type for value_type, _ in info))
        elif (isinstance(source.index, list) and len(source.index) == 1
              and isinstance(source.index[0], ast.RangeDefinition)):
            bounds = source.index[0]
            if bounds.start is None or bounds.end is None:
                raise InvalidLoopRangeException('Bit-register range requires both bounds')
            start_type, start_runtime = self.__bit_index_info(bounds.start, iterators)
            step_type, step_runtime = ((ValueType.INT, False) if bounds.step is None
                                      else self.__bit_index_info(bounds.step, iterators))
            stop_type, stop_runtime = self.__bit_index_info(bounds.end, iterators)
            if start_runtime or step_runtime or stop_runtime:
                selection = _RuntimeBitRangeSelection(name, size, bounds, start_type, step_type, stop_type)
        if selection is not None and not 1 <= size <= self.QCX_INT_MAX:
            raise InvalidLoopRangeException('Runtime selection source width must fit QCX INT indexing')
        # The shared resolver rejects active scalar iterators and retains
        # static index order/repetitions without inspecting mutable bit values.
        if selection is None:
            selection = self.__static_bit_selection(source, iterators)
            if not selection.is_register:
                raise UnsupportedOpenQASMError('bit-selection iteration requires a register selection, not a scalar bit')
        if not isinstance(iteration_type, ast.BitType) or iteration_type.size is not None:
            raise UnsupportedOpenQASMError('bit-selection iteration requires a scalar bit iterator')
        if isinstance(selection, _RuntimeBitRangeSelection):
            # The validated raw progression can contain up to 2*size entries,
            # including negative aliases. Its table capacity/count must fit INT.
            if 2 * size > self.QCX_INT_MAX:
                raise InvalidLoopRangeException('Runtime bit-range position capacity must fit QCX INT indexing')
        elif selection.width > self.QCX_INT_MAX:
            raise InvalidLoopRangeException('Bit-selection loop size exceeds QCX INT range')
        return selection

    def __validate_loop_header(
            self, statement: ast.ForInLoop,
            iterators: set[str] | dict[str, ValueType] | None = None) -> None:
        if self.__angle_selection_loop_info(statement.set_declaration, statement.type, iterators) is not None:
            return
        if self.__bit_selection_loop_info(statement.set_declaration, statement.type, iterators) is not None:
            return
        typed_collection_loop = (
            isinstance(statement.set_declaration, ast.Identifier)
            and (isinstance(statement.type, (ast.FloatType, ast.UintType, ast.BoolType, ast.ComplexType, ast.AngleType))
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
        if (self.__angle_selection_loop_info(bounds, statement.type, iterators) is not None
                or self.__bit_selection_loop_info(bounds, statement.type, iterators) is not None):
            return None  # BIT iterators do not carry UINT width metadata.
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
                     | set(self.__const_bit_register_values)
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
                self.__validate_loop_header(child, iterators)
                uint_type = self.__validate_nested_loop_bounds(child, iterators)
                value_type = self.__source_value_type(child.type)
                # Width metadata must follow the same nested shadowing as
                # value types, even when no enclosing iteration executes.
                angle_type = self.__angle_type(child.type, iterators) if value_type == ValueType.ANGLE else None
                binding = _RuntimeIteratorBinding('QASM2QCX_VALIDATION_ITERATOR', value_type, uint_type, angle_type)
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
        if target_type == ValueType.ANGLE and statement.op.name in ('&=', '|=', '^=', '<<=', '>>='):
            self.__validate_angle_assignment(statement, iterators)
        elif statement.op.name in ('<<=', '>>='):
            expression = ast.BinaryExpression(ast.BinaryOperator[statement.op.name[:-1]], target, statement.rvalue)
            self.__validate_shift_types(expression, target_type, rhs_type, iterators)
        else:
            self.__bitwise_type(target_type, rhs_type)
        self.__validate_bitwise_indices(target, iterators)
        self.__validate_bitwise_indices(statement.rvalue, iterators)

    def __validate_bitwise_indices(
            self, expression: ast.Expression, iterators: dict[str, ValueType]) -> None:
        if (isinstance(expression, ast.Cast)
                and self.__runtime_range_angle_cast_info(expression, iterators) is not None):
            return  # Angle range casts validate register operands as a unit.
        if isinstance(expression, ast.Cast) and isinstance(expression.type, ast.UintType):
            info = self.__uint_bit_cast_info(expression, iterators)
            if info is not None and (info.range_selection is not None or info.range_tree is not None):
                return  # The cast checks range types; its count is checked at runtime.
        if (self.__angle_bit_selection(expression, iterators) is not None
                or self.__runtime_bit_set_selection(expression, iterators) is not None):
            return
        if (isinstance(expression, ast.BinaryExpression)
                and expression.op.name in ('==', '!=', '<', '<=', '>', '>=')
                and self.__register_comparison_info(expression, iterators) is not None):
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
        if (statement.op.name == '=' and self.__runtime_angle_range_register_assignment_info(
                statement.lvalue, statement.rvalue, iterators) is not None):
            return True
        if (statement.op.name == '=' and self.__runtime_angle_range_expression_assignment_info(
                statement.lvalue, statement.rvalue, iterators) is not None):
            return True
        if (statement.op.name == '=' and self.__runtime_angle_range_static_selection_assignment_info(
                statement.lvalue, statement.rvalue, iterators) is not None):
            return True
        if (statement.op.name == '=' and self.__runtime_angle_range_set_expression_assignment_info(
                statement.lvalue, statement.rvalue, iterators) is not None):
            return True
        if (statement.op.name == '=' and self.__runtime_angle_range_range_expression_assignment_info(
                statement.lvalue, statement.rvalue, iterators) is not None):
            return True
        if self.__runtime_angle_range_register_compound_assignment_info(statement, iterators) is not None:
            return True
        if self.__runtime_bit_range_copy_info(statement, iterators) is not None:
            return True
        if self.__runtime_register_compound_assignment_info(statement, iterators) is not None:
            return True
        if (statement.op.name == '='
                and self.__runtime_register_assignment_info(statement.lvalue, statement.rvalue, iterators) is not None):
            return True
        if self.__validate_angle_bit_assignment(statement, iterators):
            return True
        if self.__runtime_bit_set_assignment_info(statement, iterators) is not None:
            return True
        if self.__runtime_bit_range_assignment_info(statement, iterators) is not None:
            return True
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
                and not isinstance(statement.rvalue, (ast.UnaryExpression, ast.BinaryExpression, ast.FunctionCall, ast.Cast))
                and not self.__is_constant_bit_source(statement.rvalue)
                and self.__angle_bit_selection(statement.rvalue, iterators) is None
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
            if (self.__angle_selection_loop_info(node.set_declaration, node.type, iterators) is None
                    and self.__bit_selection_loop_info(node.set_declaration, node.type, iterators) is None):
                self.__validate_bitwise_types(node.set_declaration, iterators)
            return
        if (isinstance(node, ast.IndexExpression)
                and (self.__angle_bit_selection(node, iterators) is not None
                     or self.__runtime_bit_set_selection(node, iterators) is not None)):
            return  # The source/index and static bit bounds were validated together.
        if (isinstance(node, ast.IndexExpression) and isinstance(node.collection, ast.Identifier)
                and (node.collection.name in iterators
                     or any(node.collection.name in scope for scope in self.__loop_bindings))):
            # Scalar iterators cannot be indexed, even in a plain copy in a
            # skipped body. Sized angle bit selections were handled above.
            self.__loop_index_info(node, iterators)
        if isinstance(node, ast.ClassicalAssignment):
            if (node.op.name == '=' and self.__runtime_angle_range_register_assignment_info(
                    node.lvalue, node.rvalue, iterators) is not None):
                return
            if (node.op.name == '=' and self.__runtime_angle_range_expression_assignment_info(
                    node.lvalue, node.rvalue, iterators) is not None):
                return
            if (node.op.name == '=' and self.__runtime_angle_range_static_selection_assignment_info(
                    node.lvalue, node.rvalue, iterators) is not None):
                return
            if (node.op.name == '=' and self.__runtime_angle_range_set_expression_assignment_info(
                    node.lvalue, node.rvalue, iterators) is not None):
                return
            if (node.op.name == '=' and self.__runtime_angle_range_range_expression_assignment_info(
                    node.lvalue, node.rvalue, iterators) is not None):
                return
            if self.__runtime_angle_range_register_compound_assignment_info(node, iterators) is not None:
                return
            if self.__runtime_bit_range_copy_info(node, iterators) is not None:
                return
            if self.__runtime_register_compound_assignment_info(node, iterators) is not None:
                return
            if (node.op.name == '=' and self.__runtime_range_expression_assignment_info(
                    node.lvalue, node.rvalue, iterators) is not None):
                return
            if self.__validate_angle_bit_assignment(node, iterators):
                return
            if self.__runtime_bit_range_assignment_info(node, iterators) is not None:
                return
            if (self.__angle_bit_selection(node.rvalue, iterators) is not None
                    or self.__runtime_bit_set_selection(node.rvalue, iterators) is not None):
                if not self.__validate_register_operator_assignment(node, iterators):
                    raise InvalidBitOperandException('A bit selection requires a bit-register target')
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
                    and self.__runtime_bit_range_measurement_info(node.measure, target, iterators) is not None):
                return
            if (isinstance(node, ast.QuantumMeasurementStatement)
                    and self.__runtime_bit_set_measurement_info(node.measure, target, iterators) is not None):
                return
            if (isinstance(node, ast.QuantumMeasurementStatement)
                    and (self.__is_angle_identifier(self.__operand_name(target), iterators)
                         or self.__capitalize_variable_name(self.__operand_name(target)) in self.__angle_array_types)):
                raise InvalidBitOperandException('A measurement target must be a bit variable')
        if isinstance(node, ast.ClassicalAssignment):
            self.__validate_angle_assignment(node, iterators)
        # Register values need register-aware contexts, not accidental use as
        # scalar loop bounds, casts, or indices.
        if isinstance(node, ast.Cast) and (self.__uint_bit_cast_info(node, iterators) is not None
                                         or self.__angle_bit_cast_info(node, iterators) is not None):
            # The cast validates its entire source in the appropriate scalar
            # or register context. A second scalar walk would reject valid
            # register shifts and bitwise expressions in skipped bodies.
            return
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
                and self.__register_comparison_info(node, iterators) is not None):
            return
        # Logical values and casts also need type checking when a body is
        # empty; this inspects types without executing skipped arithmetic.
        if (isinstance(node, ast.FunctionCall)
                or (isinstance(node, ast.UnaryExpression) and node.op.name in ('~', '-', '!'))
                or (isinstance(node, ast.Cast)
                    and (isinstance(node.type, (ast.BoolType, ast.BitType, ast.AngleType))
                         or self.__expression_mentions_angle(node, iterators)
                         or self.__bit_register_expression_width(node.argument, iterators) is not None))
                or (isinstance(node, ast.BinaryExpression) and node.op.name in (
                    '&', '|', '^', '<<', '>>', '**', '%', '&&', '||', '==', '!=', '<', '<=', '>', '>='))
                or (isinstance(node, ast.BinaryExpression) and self.__expression_mentions_angle(node, iterators))):
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
        if isinstance(source, ast.IndexExpression):
            if self.__angle_selection_loop_info(source, statement.type) is not None:
                self.__visit_angle_selection_for(statement)
            else:
                self.__visit_bit_selection_for(statement)
            return
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

    def __visit_bit_selection_for(self, statement: ast.ForInLoop) -> None:
        # Resolve the source and static selectors before binding the iterator,
        # so its name may shadow either the register or a selector constant.
        selection = self.__bit_selection_loop_info(statement.set_declaration, statement.type)
        binding = _RuntimeIteratorBinding('QASM2QCX_VALIDATION_ITERATOR', ValueType.BIT)
        with self.__iterator_binding(statement.identifier.name, binding):
            self.__validate_loop_body(statement.block, self.__loop_iterator_types())
        self.__validate_loop_syntax(statement)
        if isinstance(selection, (_RuntimeBitRangeSelection, _RuntimeBitSetSelection)):
            if not self.__is_initialization_process:
                self.__emit_bit_selection_for(statement, selection)
            return
        if isinstance(selection, _RuntimeSingletonBitSelection):
            if not self.__is_initialization_process:
                self.__emit_runtime_singleton_bit_for(statement, selection)
            return
        if selection.source_name in self.__const_bit_register_values:
            source = self.__constant_bit_register(selection.source_name)
            loop_index = self.__loop_index
            self.__loop_index += 1
            previous = (self.__expression_kind, self.__value, self.__value_type,
                        self.__value_kind, self.__evaluate_constant)
            try:
                self.__visit_constant_bit_register_for(statement, source, loop_index, selection.indices)
            finally:
                (self.__expression_kind, self.__value, self.__value_type,
                 self.__value_kind, self.__evaluate_constant) = previous
        elif not self.__is_initialization_process:
            self.__emit_bit_selection_for(statement, selection)

    def __emit_runtime_singleton_bit_for(
            self, statement: ast.ForInLoop, selection: _RuntimeSingletonBitSelection) -> None:
        previous = (self.__expression_kind, self.__value, self.__value_type,
                    self.__value_kind, self.__evaluate_constant)
        previous_used = self.__used_temporary_variables.copy()
        previous_temporaries = self.__declared_temporary_variables.copy()
        loop_index = self.__loop_index
        self.__loop_index += 1
        next_label = f'QASM2QCX_LOOP_{loop_index}_NEXT'
        end_label = f'QASM2QCX_LOOP_{loop_index}_END'
        try:
            constant = (self.__constant_bit_register(selection.source_name)
                        if selection.source_name in self.__const_bit_register_values else None)
            index = self.__capture_array_index(selection.index, selection.source_width)
            iterator = self.__add_new_temporary_variable(ValueType.BIT)
            # Both index capture and the value copy precede the scoped binding.
            # The body cannot change its copied bit by modifying either source.
            if constant is None:
                source = self.__capitalize_variable_name(selection.source_name)
                operand = source if selection.source_width == 1 else f'{source}:{index}'
                self.__qcx_lines.append(f'LET {iterator} := {operand}')
            else:
                self.__copy_constant_bit_at_index(constant, index, iterator)
            with self.__iterator_binding(statement.identifier.name, _RuntimeIteratorBinding(iterator, ValueType.BIT)), \
                    self.__loop_context(end_label, next_label):
                for child in statement.block:
                    self.visit(child)
            # A singleton selection has one visit; either transfer exits it.
            self.__qcx_lines.extend([f'@{next_label}', f'@{end_label}'])
            self.__hoist_temporary_declarations(previous_temporaries)
            if constant is not None or self.__inside_expanded_loop():
                self.__check_loop_output_limit()
        finally:
            (self.__expression_kind, self.__value, self.__value_type,
             self.__value_kind, self.__evaluate_constant) = previous
            for temporary in self.__used_temporary_variables - previous_used:
                self.__release_temporary_variable(temporary)

    def __copy_constant_bit_at_index(self, source: _ConstantBitRegister, index: str, destination: str) -> None:
        previous_used = self.__used_temporary_variables.copy()
        try:
            words = self.__prepare_constant_bit_lookup(source)
            self.__copy_prepared_constant_bit(words, index, destination)
        finally:
            for temporary in self.__used_temporary_variables - previous_used:
                self.__release_temporary_variable(temporary)

    def __prepare_constant_bit_lookup(self, source: _ConstantBitRegister) -> str | int:
        # The caller retains this storage across all extractions. Each uses
        # separate scratch storage, so nested lookups cannot overwrite it.
        word_width = self.QCX_UINT_WIDTH
        word_count = (source.width + word_width - 1) // word_width
        if len(self.__qcx_lines) + word_count > self.MAX_LOOP_OUTPUT_LINES:
            raise InvalidLoopRangeException('Constant bit lookup exceeds QCX output limit')
        if word_count == 1:
            return source.value
        words = self.__add_temporary_array(word_count, 'CONST_BITS', ValueType.UINT)
        word_bytes = word_width // 8
        packed = source.value.to_bytes(word_count * word_bytes, byteorder='little')
        for word in range(word_count):
            # Decode fixed-size words instead of repeatedly shifting the
            # whole large integer, which would be quadratic.
            start = word * word_bytes
            bits = int.from_bytes(packed[start:start + word_bytes], byteorder='little')
            self.__qcx_lines.append(f'LET {words}:{word} := {bits}')
        return words

    def __copy_prepared_constant_bit(self, words: str | int, index: str, destination: str) -> None:
        # The index is normalized and checked by the caller. Even the final
        # partially filled word therefore cannot expose padding bits.
        previous_used = self.__used_temporary_variables.copy()
        try:
            value = self.__add_new_temporary_variable(ValueType.UINT)
            if isinstance(words, int):
                self.__qcx_lines.append(f'LET {value} := {words}')
                offset = index
            else:
                word_index = self.__add_new_temporary_variable(ValueType.INT)
                offset = self.__add_new_temporary_variable(ValueType.INT)
                self.__qcx_lines.extend([
                    f'LET {word_index} := {index}', f'LET {word_index} /= {self.QCX_UINT_WIDTH}',
                    f'LET {offset} := {index}', f'LET {offset} %= {self.QCX_UINT_WIDTH}',
                    f'LET {value} := {words}:{word_index}',
                ])
            self.__qcx_lines.extend([
                f'LET {value} >>= {offset}', f'LET {value} &= 1', f'LET {destination} := :INT:{value}',
            ])
        finally:
            for temporary in self.__used_temporary_variables - previous_used:
                self.__release_temporary_variable(temporary)

    def __emit_bit_selection_for(
            self, statement: ast.ForInLoop,
            selection: _StaticBitSelection | _RuntimeBitSetSelection | _RuntimeBitRangeSelection) -> None:
        previous = (self.__expression_kind, self.__value, self.__value_type,
                    self.__value_kind, self.__evaluate_constant)
        previous_used = self.__used_temporary_variables.copy()
        previous_temporaries = self.__declared_temporary_variables.copy()
        loop_index = self.__loop_index
        self.__loop_index += 1
        body_label = f'QASM2QCX_LOOP_{loop_index}_BODY'
        next_label = f'QASM2QCX_LOOP_{loop_index}_NEXT'
        end_label = f'QASM2QCX_LOOP_{loop_index}_END'
        try:
            constant = (self.__constant_bit_register(selection.source_name)
                        if selection.source_name in self.__const_bit_register_values else None)
            source = self.__capitalize_variable_name(selection.source_name)
            size = constant.width if constant is not None else self.__bit_variable_name_size_map[source]
            iterator = self.__add_new_temporary_variable(ValueType.BIT)
            index = self.__add_new_temporary_variable(ValueType.INT)
            bit_index = self.__add_new_temporary_variable(ValueType.INT)
            if isinstance(selection, _RuntimeBitRangeSelection):
                label = f'QASM2QCX_BIT_RANGE_{self.__array_index}'
                self.__array_index += 1
                positions, count = self.__capture_bit_range_positions(selection.bounds, size, label)
            else:
                positions = self.__add_temporary_int_array(selection.width, 'POSITIONS')
                count = selection.width
                for position, selected_index in enumerate(selection.indices):
                    if isinstance(selection, _RuntimeBitSetSelection):
                        # Capture every occurrence in source order before binding
                        # the iterator or executing any body statement. The table
                        # owns a copy, so capture storage can be reused immediately.
                        captured = self.__capture_array_index(selected_index, size)
                        self.__qcx_lines.append(f'LET {positions}:{position} := {captured}')
                        self.__release_temporary_variable(captured)
                    else:
                        self.__qcx_lines.append(f'LET {positions}:{position} := {selected_index}')
            # QCX cannot nest array subscripts. Load the selected position
            # first, then copy that live bit into the scoped scalar iterator.
            # bit[1] has scalar INT storage, so its read needs no subscript.
            words = self.__prepare_constant_bit_lookup(constant) if constant is not None else None
            self.__qcx_lines.extend([
                f'LET {index} := 0', f'@{body_label}',
                f'LET {bit_index} := {positions}:{index}',
            ])
            if words is None:
                operand = source if size == 1 else f'{source}:{bit_index}'
                self.__qcx_lines.append(f'LET {iterator} := {operand}')
            else:
                self.__copy_prepared_constant_bit(words, bit_index, iterator)
            with self.__iterator_binding(statement.identifier.name, _RuntimeIteratorBinding(iterator, ValueType.BIT)), \
                    self.__loop_context(end_label, next_label):
                for child in statement.block:
                    self.visit(child)
            self.__qcx_lines.append(f'@{next_label}')
            if isinstance(count, int):
                self.__qcx_lines.extend([
                    f'JUMPIF {end_label} {index} == {count - 1}', f'LET {index} += 1', f'JUMP {body_label}',
                ])
            else:
                # count fits INT and index is at most count-1 before advancing.
                self.__qcx_lines.extend([
                    f'LET {index} += 1', f'JUMPIF {body_label} {index} < {count}',
                ])
            self.__qcx_lines.append(f'@{end_label}')
            self.__hoist_temporary_declarations(previous_temporaries)
            if constant is not None or isinstance(selection, _RuntimeBitRangeSelection) or self.__inside_expanded_loop():
                self.__check_loop_output_limit()
        finally:
            (self.__expression_kind, self.__value, self.__value_type,
             self.__value_kind, self.__evaluate_constant) = previous
            for temporary in self.__used_temporary_variables - previous_used:
                self.__release_temporary_variable(temporary)

    def __visit_angle_selection_for(self, statement: ast.ForInLoop) -> None:
        selection = self.__angle_selection_loop_info(statement.set_declaration, statement.type)
        # The header has validated the selection before this binding shadows
        # source names. Body validation needs a BIT type, never a made-up bit.
        binding = _RuntimeIteratorBinding('QASM2QCX_VALIDATION_ITERATOR', ValueType.BIT)
        with self.__iterator_binding(statement.identifier.name, binding):
            self.__validate_loop_body(statement.block, self.__loop_iterator_types())
        self.__validate_loop_syntax(statement)
        if self.__is_initialization_process:
            return
        self.__emit_angle_selection_for(statement, selection)

    def __angle_selection_loop_source(self, source: ast.Identifier | ast.IndexExpression) -> str | int:
        if isinstance(source, ast.IndexExpression):
            # Retain the element address, not an entry-value snapshot. Its
            # captured index remains reserved until the loop finishes.
            return self.__array_access(source)[0]
        self.visit(source)
        if self.__value_type != ValueType.ANGLE:
            raise InvalidBitOperandException('An angle bit-selection loop requires an angle source')
        return self.__value

    def __capture_bit_range_integer(
            self, expression: ast.Expression, label: str) -> tuple[str, str]:
        value, value_type, value_kind = self.__condition_operand(expression)
        if value_type not in (ValueType.INT, ValueType.UINT):
            raise InvalidLoopRangeException('Bit range operands must be integers')
        high = self.__add_new_temporary_variable(ValueType.INT)
        low = self.__add_new_temporary_variable(ValueType.UINT)
        # Represent n exactly as high * (UINT_MAX + 1) + low. Captured native
        # operands have high -1 or 0; subsequent addition can produce -2 or 1.
        # This preserves mathematical range advancement across zero and past
        # either native endpoint without signed overflow or UINT narrowing.
        self.__qcx_lines.append(f'LET {high} := 0')
        if value_kind == ValueKind.LITERAL:
            minimum = self.QCX_INT_MIN if value_type == ValueType.INT else 0
            maximum = self.QCX_INT_MAX if value_type == ValueType.INT else self.QCX_UINT_MAX
            if not minimum <= int(value) <= maximum:
                # A nonrepresentable literal must fail only when reached; do
                # not put a literal rejected by the QCX parser on skipped paths.
                self.__qcx_lines.append(f'ASSERT {high} == 1')
                value = 0
            if int(value) < 0:
                self.__qcx_lines.append(f'LET {high} := -1')
        elif value_type == ValueType.INT:
            self.__qcx_lines.extend([
                f'JUMPIF {label} {value} >= 0', f'LET {high} := -1', f'@{label}',
            ])
        self.__emit_assignment(low, ':=', ValueType.UINT, value, value_type, value_kind)
        return high, low

    def __emit_bit_range_comparison(
            self, lhs: tuple[str, str], operator: str, rhs: tuple[str, str],
            true_label: str, false_label: str) -> None:
        # Lexicographic comparison of the signed high part and unsigned low
        # part. No mixed signed/unsigned QCX comparison or lossy cast is needed.
        self.__qcx_lines.extend([
            f'JUMPIF {true_label} {lhs[0]} {operator} {rhs[0]}',
            f'JUMPIF {false_label} {lhs[0]} \\= {rhs[0]}',
            f'JUMPIF {true_label} {lhs[1]} {operator} {rhs[1]}',
            f'JUMP {false_label}',
        ])

    def __capture_bit_range_positions(
            self, bounds: ast.RangeDefinition, width: int, label: str, *,
            capacity: int | None = None) -> tuple[str, str]:
        capacity = min(2 * width, self.MAX_RUNTIME_BIT_RANGE_POSITIONS) if capacity is None else capacity
        if not 1 <= capacity <= self.QCX_INT_MAX:
            raise InvalidLoopRangeException('Bit-range position capacity must fit QCX INT indexing')
        if capacity > self.MAX_RUNTIME_BIT_RANGE_POSITIONS:
            raise InvalidLoopRangeException('Bit-range position capacity exceeds storage limit')
        current = self.__capture_bit_range_integer(bounds.start, label + '_START_NONNEGATIVE')
        step = self.__capture_bit_range_integer(
            bounds.step if bounds.step is not None else ast.IntegerLiteral(1), label + '_STEP_NONNEGATIVE')
        stop = self.__capture_bit_range_integer(bounds.end, label + '_STOP_NONNEGATIVE')
        # Every valid raw position is in [-width, width). A strictly monotonic
        # nonzero-step range can therefore contain at most 2*width entries,
        # including aliases such as -1 and width-1, which both must be kept.
        # Cap allocation independently of the source width; oversized reached
        # selections fail before their first body, without unbounded storage.
        positions = self.__add_temporary_int_array(capacity, 'POSITIONS')
        count = self.__add_new_temporary_variable(ValueType.INT)
        normalized = self.__add_new_temporary_variable(ValueType.UINT)
        previous_low = self.__add_new_temporary_variable(ValueType.UINT)
        entry, positive, append, negative, store, finish = (
            label + suffix for suffix in ('_ENTRY', '_POSITIVE', '_APPEND', '_NEGATIVE', '_STORE', '_FINISH'))
        self.__qcx_lines.extend([
            f'LET {count} := 0',
            # For native INT/UINT captures, low == 0 iff the step is zero.
            f'ASSERT {step[1]} \\= 0', f'@{entry}',
            f'JUMPIF {positive} {step[0]} == 0',
        ])
        self.__emit_bit_range_comparison(current, '<', stop, finish, append)
        self.__qcx_lines.append(f'@{positive}')
        self.__emit_bit_range_comparison(current, '>', stop, finish, append)
        self.__qcx_lines.extend([
            f'@{append}', f'ASSERT {current[0]} >= -1', f'ASSERT {current[0]} <= 0',
            f'LET {normalized} := {current[1]}', f'JUMPIF {negative} {current[0]} < 0',
            f'ASSERT {normalized} < {width}', f'JUMP {store}',
            f'@{negative}', f'ASSERT {normalized} >= {self.QCX_UINT_MAX + 1 - width}',
            f'LET {normalized} += {width}', f'@{store}',
            f'ASSERT {count} < {capacity}', f'LET {positions}:{count} := :INT:{normalized}',
            f'LET {count} += 1',
            # UINT addition wraps by definition. Its carry corrects the high
            # part, so the pair still holds the exact mathematical candidate.
            f'LET {previous_low} := {current[1]}', f'LET {current[1]} += {step[1]}',
            f'LET {current[0]} += {step[0]}',
            f'JUMPIF {entry} {current[1]} >= {previous_low}',
            f'LET {current[0]} += 1', f'JUMP {entry}', f'@{finish}', f'ASSERT {count} > 0',
        ])
        return positions, count

    @contextlib.contextmanager
    def __captured_angle_bit_range(
            self, selection: _RuntimeAngleBitRangeSelection) -> Iterator[tuple[str | int, str, str]]:
        previous = (self.__expression_kind, self.__value, self.__value_type,
                    self.__value_kind, self.__evaluate_constant)
        previous_used = self.__used_temporary_variables.copy()
        label = f'QASM2QCX_ANGLE_RANGE_{self.__array_index}'
        self.__array_index += 1
        try:
            self.__expression_kind = ExpressionKind.ARITHMETIC
            source = self.__angle_selection_loop_source(selection.source)
            positions, count = self.__capture_bit_range_positions(selection.bounds, selection.angle_type.width, label)
            yield source, positions, count
        finally:
            (self.__expression_kind, self.__value, self.__value_type,
             self.__value_kind, self.__evaluate_constant) = previous
            for temporary in self.__used_temporary_variables - previous_used:
                self.__release_temporary_variable(temporary)

    def __emit_angle_selection_for(
            self, statement: ast.ForInLoop,
            selection: _AngleBitLoopSelection) -> None:
        previous = (self.__expression_kind, self.__value, self.__value_type, self.__value_kind)
        previous_used = self.__used_temporary_variables.copy()
        previous_temporaries = self.__declared_temporary_variables.copy()
        loop_index = self.__loop_index
        self.__loop_index += 1
        try:
            self.__expression_kind = ExpressionKind.ARITHMETIC
            if isinstance(selection, _RuntimeAngleBitRangeSelection):
                with self.__captured_angle_bit_range(selection) as (source, positions, count):
                    self.__emit_angle_selection_loop_body(statement, source, positions, count, loop_index)
            else:
                source = self.__angle_selection_loop_source(selection.source)
                # Capture and check indices before introducing the iterator,
                # including same-name bindings. Keep selection order/repeats.
                indices = self.__capture_angle_bit_indices(selection)
                positions = self.__add_temporary_int_array(selection.width, 'POSITIONS')
                for position, bit_index in enumerate(indices):
                    self.__qcx_lines.append(f'LET {positions}:{position} := {bit_index}')
                self.__emit_angle_selection_loop_body(statement, source, positions, selection.width, loop_index)
            self.__hoist_temporary_declarations(previous_temporaries)
            if self.__inside_expanded_loop():
                self.__check_loop_output_limit()
        finally:
            self.__expression_kind, self.__value, self.__value_type, self.__value_kind = previous
            for temporary in self.__used_temporary_variables - previous_used:
                self.__release_temporary_variable(temporary)

    def __emit_angle_selection_loop_body(
            self, statement: ast.ForInLoop, source: str | int, positions: str,
            count: int | str, loop_index: int) -> None:
        body_label = f'QASM2QCX_LOOP_{loop_index}_BODY'
        next_label = f'QASM2QCX_LOOP_{loop_index}_NEXT'
        end_label = f'QASM2QCX_LOOP_{loop_index}_END'
        iterator = self.__add_new_temporary_variable(ValueType.BIT)
        index = self.__add_new_temporary_variable(ValueType.INT)
        scratch = self.__add_new_temporary_variable(ValueType.UINT)
        self.__qcx_lines.extend([
            f'LET {index} := 0', f'@{body_label}', f'LET {scratch} := {source}',
            f'LET {scratch} >>= {positions}:{index}', f'LET {scratch} &= 1',
            f'LET {iterator} := :INT:{scratch}',
        ])
        with self.__iterator_binding(statement.identifier.name, _RuntimeIteratorBinding(iterator, ValueType.BIT)), \
                self.__loop_context(end_label, next_label):
            for child in statement.block:
                self.visit(child)
        self.__qcx_lines.append(f'@{next_label}')
        if isinstance(count, int):
            # Preserve fixed-width output and avoid incrementing at INT_MAX.
            self.__qcx_lines.extend([
                f'JUMPIF {end_label} {index} == {count - 1}', f'LET {index} += 1', f'JUMP {body_label}',
            ])
        else:
            # The runtime count is positive and at most twice the angle width.
            self.__qcx_lines.extend([f'LET {index} += 1', f'JUMPIF {body_label} {index} < {count}'])
        self.__qcx_lines.append(f'@{end_label}')

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
                    ast.IntType, ast.UintType, ast.FloatType, ast.BoolType, ast.ComplexType, ast.AngleType)):
                raise UnsupportedOpenQASMError('iteration over classical array other than int, uint, float, bool, complex, or angle')
            value_type = self.__source_value_type(variable_type.base_type)
            if self.__source_value_type(iteration_type) != value_type:
                raise UnsupportedOpenQASMError('array iteration requires matching element and iterator types')
            uint_type = None
            if value_type == ValueType.UINT:
                uint_type = self.__unsigned_integer_type(iteration_type, iterators)
                if uint_type.width != self.__uint_variable_types[variable_name].width:
                    raise UnsupportedOpenQASMError('UINT array iteration requires matching effective widths')
            if value_type == ValueType.ANGLE:
                angle_type = self.__angle_type(iteration_type, iterators)
                if angle_type.width != self.__angle_array_types[variable_name].width:
                    raise UnsupportedOpenQASMError('angle array iteration requires matching effective widths')
            if (not self.__is_initialization_process
                    and variable_name not in self.__array_element_types):
                raise NoVariableNameException(name)
            return variable_name, self.__array_source_sizes[name], value_type, uint_type
        raise UnsupportedOpenQASMError('for-loop identifier other than an int/uint/float/bool/complex/angle array or bit register')

    def __visit_collection_for(self, statement: ast.ForInLoop) -> None:
        source, size, value_type, uint_type = self.__collection_loop_info(
            statement.set_declaration, statement.type)
        angle_type = self.__angle_type(statement.type) if value_type == ValueType.ANGLE else None
        binding = _RuntimeIteratorBinding(f'QASM2QCX_LOOP_{self.__loop_index}_ITERATOR', value_type, uint_type, angle_type)
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
        # Angle values use raw UINT storage while the scoped binding retains
        # their source type, width, and sizedness for expression lowering.
        storage_type = ValueType.UINT if angle_type is not None else value_type
        storage_uint_type = (_UnsignedIntegerType(angle_type.width, angle_type.is_sized)
                             if angle_type is not None else uint_type)
        iterator = self.__add_new_temporary_variable(storage_type, storage_uint_type)
        index = self.__add_new_temporary_variable(ValueType.INT)
        try:
            # QCX stores a singleton bit register as a scalar INT, unlike
            # classical arrays, whose scalar accesses retain explicit indexing.
            operand = source if value_type == ValueType.BIT and size == 1 else f'{source}:{index}'
            self.__qcx_lines.extend([f'LET {index} := 0', f'@{body_label}',
                                    f'LET {iterator} := {operand}'])
            # Read the current element on each iteration, not a loop-entry
            # snapshot. Keep its copied value and index reserved across the body.
            with self.__iterator_binding(statement.identifier.name, dataclasses.replace(binding, storage=iterator)), \
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
            self, statement: ast.ForInLoop, source: _ConstantBitRegister, loop_index: int,
            indices: tuple[int, ...] | None = None) -> None:
        count = source.width if indices is None else len(indices)
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
                        source_index = index if indices is None else indices[index]
                        self.__qcx_lines.append(f'LET {iterator} := {(source.value >> source_index) & 1}')
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

        uint_info = self.__uint_bit_cast_info(expression)
        if uint_info is not None:
            if uint_info.result_type == ValueType.BIT_REGISTER:
                raise InvalidBitOperandException('A register cast requires a bit-register expression context')
            if self.__expression_kind == ExpressionKind.CONST_ARITHMETIC:
                self.__visit_constant_uint_bit_cast(expression, uint_info)
            else:
                if self.__expression_kind not in (ExpressionKind.ARITHMETIC, ExpressionKind.CONDITIONAL):
                    raise UnsupportedOpenQASMError('UINT/bit-register cast outside a value expression')
                self.__visit_runtime_bit_to_uint_cast(expression, uint_info)
            return
        info = self.__angle_bit_cast_info(expression)
        if info is not None:
            if info[0] == ValueType.BIT_REGISTER:
                raise InvalidBitOperandException('A register cast requires a bit-register expression context')
            if self.__expression_kind != ExpressionKind.CONST_ARITHMETIC:
                if self.__expression_kind not in (ExpressionKind.ARITHMETIC, ExpressionKind.CONDITIONAL):
                    raise UnsupportedOpenQASMError('angle/bit-register cast outside a value expression')
                self.__visit_runtime_bit_to_angle_cast(expression)
            else:
                self.__visit_constant_angle_bit_cast(expression, info)
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
        if expression.name.name in ('sin', 'cos', 'tan') and signature.operand_types == (ValueType.ANGLE,):
            self.__visit_angle_trigonometric(expression, runtime)
            return
        if (signature.rotation or signature.population_count) and signature.operand_types[0] == ValueType.ANGLE:
            self.__visit_angle_bit_function(expression, signature, runtime)
            return
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

    @staticmethod
    def __fold_angle_trigonometric(name: str, bits: int, angle_type: _AngleType) -> float:
        denominator = 1 << angle_type.width
        # Choose the nearest quarter turn using integer arithmetic, including
        # width one. Keep the small signed distance before converting to REAL;
        # subtracting rounded radians would lose low bits near axes and poles.
        quadrant = (4 * bits + denominator // 2) // denominator
        residual = 4 * bits - quadrant * denominator
        quadrant %= 4
        if residual == 0:
            if name == 'sin':
                return (0.0, 1.0, 0.0, -1.0)[quadrant]
            if name == 'cos':
                return (1.0, 0.0, -1.0, 0.0)[quadrant]
            if quadrant % 2:
                raise InvalidConstantFunctionException('Function tan: exact angle tangent pole')
            return 0.0  # Canonical positive zero at exact axes.
        reduced = math.ldexp(float(residual), -angle_type.width) * (math.pi / 2)
        if name == 'tan':
            tangent = math.tan(reduced)
            return -1.0 / tangent if quadrant % 2 else tangent
        sine, cosine = math.sin(reduced), math.cos(reduced)
        return ((sine, cosine, -sine, -cosine) if name == 'sin'
                else (cosine, -sine, -cosine, sine))[quadrant]

    def __visit_angle_trigonometric(self, expression: ast.FunctionCall, runtime: bool) -> None:
        if self.__expression_kind != ExpressionKind.CONST_ARITHMETIC:
            if self.__expression_kind not in (ExpressionKind.ARITHMETIC, ExpressionKind.CONDITIONAL):
                raise UnsupportedOpenQASMError('function call outside a value expression')
            self.__visit_runtime_angle_trigonometric(expression)
            return
        if runtime:
            raise NoConstantExpressionException
        previous = (self.__expression_kind, self.__value, self.__value_type,
                    self.__value_kind, self.__evaluate_constant)
        previous_used = self.__used_temporary_variables.copy()
        try:
            argument = expression.arguments[0]
            _, angle_type, _ = self.__angle_operand_info(argument)
            self.visit(argument)
            if self.__value_kind != ValueKind.LITERAL:
                raise NoConstantExpressionException
            result = (self.__fold_angle_trigonometric(expression.name.name, int(self.__value), angle_type)
                      if self.__evaluate_constant else 0.0)
            self.__value, self.__value_type, self.__value_kind = result, ValueType.FLOAT, ValueKind.LITERAL
        except Exception:
            (self.__expression_kind, self.__value, self.__value_type,
             self.__value_kind, self.__evaluate_constant) = previous
            raise
        finally:
            for temporary in self.__used_temporary_variables - previous_used:
                self.__release_temporary_variable(temporary)

    def __visit_runtime_angle_trigonometric(self, expression: ast.FunctionCall) -> None:
        previous = (self.__expression_kind, self.__value, self.__value_type, self.__value_kind)
        previous_used = self.__used_temporary_variables.copy()
        keep = None
        prefix = f'QASM2QCX_ANGLE_TRIG_{self.__angle_trigonometric_index}'
        self.__angle_trigonometric_index += 1
        try:
            self.__expression_kind = ExpressionKind.ARITHMETIC
            argument = expression.arguments[0]
            _, angle_type, _ = self.__angle_operand_info(argument)
            bits = self.__angle_arithmetic_operand(argument, angle_type)
            quadrant = self.__add_new_temporary_variable(ValueType.INT)
            residual = self.__add_new_temporary_variable(ValueType.INT)
            reduced = self.__add_new_temporary_variable(ValueType.FLOAT)
            result = self.__add_new_temporary_variable(ValueType.FLOAT)
            if angle_type.width == 1:
                self.__qcx_lines.extend([
                    f'LET {quadrant} := :INT:{bits}', f'LET {quadrant} *= 2', f'LET {residual} := 0',
                ])
            else:
                # Split before converting to INT: a full-width angle can
                # exceed INT_MAX, but its quadrant and offset always fit.
                offset = self.__add_new_temporary_variable(ValueType.UINT)
                quarter = 1 << (angle_type.width - 2)
                self.__qcx_lines.extend([
                    f'LET {offset} := {bits}', f'LET {offset} >>= {angle_type.width - 2}',
                    f'LET {quadrant} := :INT:{offset}',
                    f'LET {offset} := {bits}', f'LET {offset} &= {quarter - 1}',
                    f'LET {residual} := :INT:{offset}',
                ])
                if angle_type.width >= 3:
                    self.__qcx_lines.extend([
                        f'JUMPIF {prefix}_REDUCED {residual} < {quarter // 2}',
                        f'LET {residual} -= {quarter}', f'LET {quadrant} += 1',
                        f'LET {quadrant} &= 3', f'@{prefix}_REDUCED',
                    ])
            self.__qcx_lines.extend([
                f'LET {reduced} := :REAL:{residual}',
                # The power-of-two scale is exact in every REAL precision;
                # native PI avoids limiting wider backends to host binary64.
                f'LET {reduced} *= {math.ldexp(1.0, 1 - angle_type.width)!r}',
                f'LET {reduced} *= :PI',
            ])
            name = expression.name.name
            if name == 'tan':
                self.__qcx_lines.extend([
                    f'JUMPIF {prefix}_NONAXIS {residual} \\= 0',
                    f'ASSERT {quadrant} \\= 1', f'ASSERT {quadrant} \\= 3',
                    f'LET {result} := 0.0', f'JUMP {prefix}_DONE', f'@{prefix}_NONAXIS',
                    f'APPLY {result} <- TAN {reduced}',
                    f'LET {quadrant} &= 1', f'JUMPIF {prefix}_DONE {quadrant} == 0',
                    f'LET {reduced} := -1.0', f'LET {reduced} /= {result}',
                    f'LET {result} := {reduced}',
                ])
            else:
                for index in range(3):
                    self.__qcx_lines.append(f'JUMPIF {prefix}_Q{index} {quadrant} == {index}')
                self.__qcx_lines.append(f'JUMP {prefix}_Q3')
                functions = ('SIN', 'COS', 'SIN', 'COS') if name == 'sin' else ('COS', 'SIN', 'COS', 'SIN')
                axes = (0.0, 1.0, 0.0, -1.0) if name == 'sin' else (1.0, 0.0, -1.0, 0.0)
                for index, (function, axis) in enumerate(zip(functions, axes)):
                    negative = index >= 2 if name == 'sin' else index in (1, 2)
                    self.__qcx_lines.extend([
                        f'@{prefix}_Q{index}',
                        f'JUMPIF {prefix}_AXIS{index} {residual} == 0',
                        f'APPLY {result} <- {function} {reduced}',
                    ])
                    if negative:
                        self.__qcx_lines.append(f'LET {result} *= -1.0')
                    self.__qcx_lines.extend([
                        f'JUMP {prefix}_DONE', f'@{prefix}_AXIS{index}',
                        f'LET {result} := {axis!r}', f'JUMP {prefix}_DONE',
                    ])
            self.__qcx_lines.append(f'@{prefix}_DONE')
            self.__expression_kind = previous[0]
            self.__value, self.__value_type, self.__value_kind = result, ValueType.FLOAT, ValueKind.TEMPORARY
            keep = result
        except Exception:
            self.__expression_kind, self.__value, self.__value_type, self.__value_kind = previous
            raise
        finally:
            for temporary in self.__used_temporary_variables - previous_used:
                if temporary != keep:
                    self.__release_temporary_variable(temporary)

    def __visit_angle_bit_function(
            self, expression: ast.FunctionCall, signature: _BuiltinFunctionSignature, runtime: bool) -> None:
        if self.__expression_kind != ExpressionKind.CONST_ARITHMETIC:
            if self.__expression_kind not in (ExpressionKind.ARITHMETIC, ExpressionKind.CONDITIONAL):
                raise UnsupportedOpenQASMError('function call outside a value expression')
            if signature.rotation:
                _, angle_type, _ = self.__angle_operand_info(expression.arguments[0])
                self.__visit_runtime_uint_rotation(expression, angle_type)
            else:
                self.__visit_population_count(expression, signature, runtime)
            return
        if runtime:
            raise NoConstantExpressionException
        previous = (self.__expression_kind, self.__value, self.__value_type,
                    self.__value_kind, self.__evaluate_constant)
        previous_used = self.__used_temporary_variables.copy()
        try:
            argument = expression.arguments[0]
            _, angle_type, _ = self.__angle_operand_info(argument)
            self.visit(argument)
            if self.__value_kind != ValueKind.LITERAL:
                raise NoConstantExpressionException
            bits = int(self.__value) & angle_type.mask if self.__evaluate_constant else 0
            if signature.rotation:
                distance = self.__constant_rotation_distance(expression.arguments[1])
                value = (self.__fold_rotation(bits, angle_type.width, distance, expression.name.name)
                         if self.__evaluate_constant else 0)
                self.__uint_expression_types.pop(id(expression), None)
                self.__angle_expression_types[id(expression)] = (expression, angle_type)
            else:
                value = bits.bit_count() if self.__evaluate_constant else 0
                self.__uint_expression_types[id(expression)] = (
                    expression, _UnsignedIntegerType(self.QCX_UINT_WIDTH, False))
            self.__value, self.__value_type, self.__value_kind = value, signature.result_type, ValueKind.LITERAL
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
            self.__angle_expression_types.pop(id(expression), None)
            self.__uint_expression_types[id(expression)] = (expression, uint_type)
            self.__expression_kind = previous[0]
        except Exception:
            self.__expression_kind, self.__value, self.__value_type, self.__value_kind = previous
            raise

    def __visit_runtime_uint_rotation(
            self, expression: ast.FunctionCall, angle_type: _AngleType | None = None) -> None:
        previous = (self.__expression_kind, self.__value, self.__value_type, self.__value_kind)
        previous_used = self.__used_temporary_variables.copy()
        keep = None
        prefix = f'QASM2QCX_ROTATION_{self.__rotation_index}'
        self.__rotation_index += 1
        try:
            self.__expression_kind = ExpressionKind.ARITHMETIC
            argument, distance_expression = expression.arguments
            if angle_type is None:
                _, uint_type, _ = self.__integer_operand_info(argument, None)
            else:
                uint_type = _UnsignedIntegerType(angle_type.width, angle_type.is_sized)
            value, value_type, value_kind = self.__condition_operand(argument)
            # Snapshot the value before evaluating the distance, and keep all
            # nested operand/index storage reserved until the result is ready.
            result = self.__add_new_temporary_variable(ValueType.UINT, uint_type)
            # Angle storage already contains the UINT bit pattern; this is
            # internal reuse, not a source-language angle-to-UINT conversion.
            operand = str(value) if angle_type is not None else self.__converted_operand(
                value, value_type, value_kind, ValueType.UINT)[0]
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
            if angle_type is None:
                self.__angle_expression_types.pop(id(expression), None)
                self.__uint_expression_types[id(expression)] = (expression, uint_type)
            else:
                self.__uint_expression_types.pop(id(expression), None)
                self.__angle_expression_types[id(expression)] = (expression, angle_type)
            self.__expression_kind = previous[0]
            result_type = ValueType.ANGLE if angle_type is not None else ValueType.UINT
            self.__value, self.__value_type, self.__value_kind = result, result_type, ValueKind.TEMPORARY
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
            angle_type = (self.__angle_operand_info(argument)[1]
                          if signature.operand_types == (ValueType.ANGLE,) else None)
            unsigned_operand = signature.operand_types[0] in (ValueType.UINT, ValueType.ANGLE)
            if self.__expression_kind == ExpressionKind.CONST_ARITHMETIC:
                if unsigned_operand:
                    if angle_type is None:
                        _, uint_type, _ = self.__integer_operand_info(argument, None)
                    else:
                        uint_type = _UnsignedIntegerType(angle_type.width, angle_type.is_sized)
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
                result = self.__emit_uint_population_count(argument, angle_type)
                kind = ValueKind.TEMPORARY
            else:
                self.__expression_kind = ExpressionKind.ARITHMETIC
                range_info = self.__runtime_bit_range_popcount_info(argument)
                if range_info is not None:
                    if range_info.tree is not None:
                        result = self.__emit_popcount_tree_population_count(range_info.tree)
                    elif range_info.pair is not None:
                        result = self.__emit_bit_range_pair_population_count(range_info)
                    elif range_info.bitwise is not None:
                        result = self.__emit_bit_range_bitwise_population_count(range_info)
                    elif range_info.mixed_operations:
                        result = self.__emit_bit_range_population_count(
                            range_info.selection, mixed_operations=range_info.mixed_operations)
                    elif len(range_info.rotations) > 1:
                        result = self.__emit_bit_range_population_count(
                            range_info.selection, range_info.rotation_complemented, rotations=range_info.rotations)
                    else:
                        result = self.__emit_bit_range_population_count(
                            range_info.selection, range_info.complemented, range_info.shift,
                            range_info.outer_complemented, range_info.rotation, range_info.shifts, range_info.operations)
                else:
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

    def __check_range_pair_popcount_output(self) -> None:
        if len(self.__qcx_lines) + 1 > self.MAX_LOOP_OUTPUT_LINES:
            raise InvalidFunctionCallException('Runtime range-pair popcount exceeds QCX output limit')

    def __check_runtime_register_output(self, extra: int = 0) -> None:
        if len(self.__qcx_lines) + extra > self.MAX_LOOP_OUTPUT_LINES:
            raise InvalidFunctionCallException('Runtime register expression exceeds QCX output limit')

    @contextlib.contextmanager
    def __runtime_bit_register_expression_value(
            self, tree: _RuntimeBitRegisterExpression, prefix: str) -> Iterator[tuple[str, str]]:
        previous = (self.__expression_kind, self.__value, self.__value_type,
                    self.__value_kind, self.__evaluate_constant)
        previous_used = self.__used_temporary_variables.copy()
        owned = None
        try:
            self.__check_runtime_register_output(1)
            if not tree.children:
                if isinstance(tree.selection, _RuntimeAngleBitRangeSelection):
                    snapshot, count = self.__capture_angle_range_expression_snapshot(
                        tree.selection, prefix, tree.capacity)
                elif tree.selection is not None:
                    snapshot, count = self.__capture_popcount_range_snapshot(
                        tree.selection, prefix, capacity=tree.capacity)
                else:
                    self.__check_runtime_register_output(tree.width + 3)
                    with self.__bit_register_expression_value(tree.expression) as values:
                        if len(values) != tree.width:
                            raise InvalidBitOperandException('Runtime register expression fixed-leaf width mismatch')
                        snapshot = self.__add_temporary_int_array(tree.capacity, 'POPCOUNT_TREE_BITS')
                        count = self.__add_new_temporary_variable(ValueType.INT)
                        self.__qcx_lines.append(f'LET {count} := {tree.width}')
                        for index, value in enumerate(values):
                            self.__qcx_lines.append(f'LET {snapshot}:{index} := {value}')
            else:
                with contextlib.ExitStack() as children:
                    # Do not propagate a parent's width into its children:
                    # complete each subtree, including its distances, first.
                    values = [children.enter_context(self.__runtime_bit_register_expression_value(child, f'{prefix}_CHILD_{index}'))
                              for index, child in enumerate(tree.children)]
                    count = self.__add_new_temporary_variable(ValueType.INT)
                    self.__qcx_lines.append(f'LET {count} := {values[0][1]}')
                    if len(values) == 2:
                        self.__qcx_lines.append(f'ASSERT {count} == {values[1][1]}')
                        snapshot = self.__add_temporary_int_array(tree.capacity, 'POPCOUNT_TREE_BITS')
                        index = self.__add_new_temporary_variable(ValueType.INT)
                        bit = self.__add_new_temporary_variable(ValueType.BIT)
                        self.__qcx_lines.extend([
                            f'LET {index} := 0', f'@{prefix}_COMBINE', f'JUMPIF {prefix}_DONE {index} >= {count}',
                        ])
                        self.__emit_popcount_bitwise_bit(
                            bit, f'{values[0][0]}:{index}', f'{values[1][0]}:{index}', tree.expression.op.name, None, None)
                        self.__qcx_lines.extend([
                            f'LET {snapshot}:{index} := {bit}', f'LET {index} += 1',
                            f'JUMP {prefix}_COMBINE', f'@{prefix}_DONE',
                        ])
                    else:
                        operations = (_RuntimeBitRangePopcountOperation(tree.expression, tree.count_type),)
                        snapshot = self.__transform_popcount_snapshot(
                            values[0][0], count, tree.capacity, operations, prefix)
                # Children are now released; the parent's copied count and
                # private bits remain live for its consumer.
            self.__check_runtime_register_output()
            keep = {snapshot, count}
            for temporary in self.__used_temporary_variables - previous_used - keep:
                self.__release_temporary_variable(temporary)
            owned = keep
            yield snapshot, count
        finally:
            (self.__expression_kind, self.__value, self.__value_type,
             self.__value_kind, self.__evaluate_constant) = previous
            temporaries = (self.__used_temporary_variables - previous_used if owned is None
                           else self.__used_temporary_variables & owned)
            for temporary in temporaries:
                self.__release_temporary_variable(temporary)

    def __emit_popcount_tree_population_count(self, tree: _RuntimeBitRegisterExpression) -> str:
        previous_declarations = self.__declared_temporary_variables.copy()
        prefix = f'QASM2QCX_POPCOUNT_{self.__population_count_index}_TREE'
        self.__population_count_index += 1
        with self.__runtime_bit_register_expression_value(tree, prefix) as (snapshot, count):
            result = self.__add_new_temporary_variable(ValueType.UINT)
            index = self.__add_new_temporary_variable(ValueType.INT)
            self.__qcx_lines.extend([
                f'LET {result} := 0', f'LET {index} := 0', f'@{prefix}_COUNT',
                f'JUMPIF {prefix}_COUNT_DONE {index} >= {count}',
                f'LET {result} += :UINT:{snapshot}:{index}', f'LET {index} += 1',
                f'JUMP {prefix}_COUNT', f'@{prefix}_COUNT_DONE',
            ])
            self.__check_runtime_register_output()
        self.__hoist_temporary_declarations(previous_declarations)
        return result

    def __capture_angle_range_expression_snapshot(
            self, selection: _RuntimeAngleBitRangeSelection, label: str, capacity: int) -> tuple[str, str]:
        self.__expression_kind = ExpressionKind.ARITHMETIC
        source = self.__angle_arithmetic_operand(selection.source, selection.angle_type)
        positions, count = self.__capture_bit_range_positions(
            selection.bounds, selection.angle_type.width, label, capacity=capacity)
        self.__check_runtime_register_output(12)
        snapshot = self.__add_temporary_int_array(capacity, 'ANGLE_EXPRESSION_BITS')
        index = self.__add_new_temporary_variable(ValueType.INT)
        position = self.__add_new_temporary_variable(ValueType.INT)
        scratch = self.__add_new_temporary_variable(ValueType.UINT)
        self.__qcx_lines.extend([
            f'LET {index} := 0', f'@{label}_SNAPSHOT',
            f'JUMPIF {label}_SNAPSHOT_DONE {index} >= {count}',
            f'LET {position} := {positions}:{index}', f'LET {scratch} := {source}',
            f'LET {scratch} >>= {position}', f'LET {scratch} &= 1',
            f'LET {snapshot}:{index} := :INT:{scratch}', f'LET {index} += 1',
            f'JUMP {label}_SNAPSHOT', f'@{label}_SNAPSHOT_DONE',
        ])
        return snapshot, count

    def __capture_popcount_range_snapshot(
            self, selection: _RuntimeBitRangeSelection, label: str,
            expected_count: str | None = None, *, capacity: int | None = None) -> tuple[str, str]:
        # The enclosing popcount visitor owns all scratch storage. Counts and
        # snapshot buffers remain reserved through both captures and counting.
        self.__check_range_pair_popcount_output()
        capacity = min(2 * selection.source_width, self.MAX_RUNTIME_BIT_RANGE_POSITIONS) if capacity is None else capacity
        positions, count = self.__capture_bit_range_positions(
            selection.bounds, selection.source_width, label, capacity=capacity)
        if expected_count is not None:
            self.__qcx_lines.append(f'ASSERT {count} == {expected_count}')
        self.__check_range_pair_popcount_output()
        constant = (self.__constant_bit_register(selection.source_name)
                    if selection.source_name in self.__const_bit_register_values else None)
        words = self.__prepare_constant_bit_lookup(constant) if constant is not None else None
        snapshot = self.__add_temporary_int_array(capacity, 'POPCOUNT_BITS')
        slot, position, bit = (self.__add_new_temporary_variable(ValueType.INT) for _ in range(3))
        self.__qcx_lines.extend([
            f'LET {slot} := 0', f'@{label}_SNAPSHOT',
            f'JUMPIF {label}_SNAPSHOT_DONE {slot} >= {count}',
            f'LET {position} := {positions}:{slot}',
        ])
        if words is None:
            source = self.__capitalize_variable_name(selection.source_name)
            operand = source if selection.source_width == 1 else f'{source}:{position}'
            self.__qcx_lines.append(f'LET {bit} := {operand}')
        else:
            self.__copy_prepared_constant_bit(words, position, bit)
        self.__qcx_lines.extend([
            f'LET {snapshot}:{slot} := {bit}', f'LET {slot} += 1',
            f'JUMP {label}_SNAPSHOT', f'@{label}_SNAPSHOT_DONE',
        ])
        self.__check_range_pair_popcount_output()
        return snapshot, count

    def __capture_popcount_operand(
            self, operand: _RuntimeBitRangePopcountOperand, label: str,
            expected_count: str | None = None, *, width: int | None = None) -> tuple[str, str]:
        capacity = min(2 * operand.selection.source_width, self.MAX_RUNTIME_BIT_RANGE_POSITIONS) if width is None else width
        snapshot, count = self.__capture_popcount_range_snapshot(
            operand.selection, label, expected_count, capacity=capacity)
        if not operand.operations:
            return snapshot, count
        prefix = label + '_OPERAND'
        return self.__transform_popcount_snapshot(snapshot, count, capacity, operand.operations, prefix), count

    def __transform_popcount_snapshot(
            self, snapshot: str, count: str, capacity: int,
            operations: tuple[_RuntimeBitRangePopcountOperation, ...], prefix: str) -> str:
        # Capture every distance before mapping any bit, including later
        # distances after a shift that discards the entire operand.
        distances = self.__capture_bit_range_popcount_distances(operations, count, prefix)
        result = self.__add_temporary_int_array(capacity, 'POPCOUNT_OPERAND_BITS')
        slot, index, bit, remaining = (self.__add_new_temporary_variable(ValueType.INT) for _ in range(4))
        self.__qcx_lines.extend([
            f'LET {slot} := 0', f'@{prefix}_LOOP', f'JUMPIF {prefix}_DONE {slot} >= {count}',
            f'LET {index} := {slot}',
        ])
        self.__emit_popcount_operand_bit(snapshot, count, operations, distances, index, bit, remaining, prefix)
        self.__qcx_lines.extend([
            f'@{prefix}_STORE', f'LET {result}:{slot} := {bit}', f'LET {slot} += 1',
            f'JUMP {prefix}_LOOP', f'@{prefix}_DONE',
        ])
        self.__check_range_pair_popcount_output()
        return result

    def __emit_popcount_operand_bit(
            self, snapshot: str, count: str, operations: tuple[_RuntimeBitRangePopcountOperation, ...],
            distances: tuple[str | None, ...], index: str, bit: str, remaining: str, prefix: str) -> None:
        # Trace one output position backward. Every index stays in [0,count),
        # and subtraction-based wrapping avoids index+count or negating INT_MIN.
        inverted = False
        fills = []
        for number in reversed(range(len(operations))):
            node = operations[number].expression
            if isinstance(node, ast.UnaryExpression):
                inverted = not inverted
                continue
            distance = distances[number]
            label = f'{prefix}_MAP_{number}'
            if isinstance(node, ast.FunctionCall):
                self.__qcx_lines.extend([
                    f'JUMPIF {label}_WRAP {index} < {distance}',
                    f'LET {index} -= {distance}', f'JUMP {label}_NEXT', f'@{label}_WRAP',
                    f'LET {remaining} := {count}', f'LET {remaining} -= {distance}',
                    f'LET {index} += {remaining}', f'@{label}_NEXT',
                ])
            else:
                if node.op.name == '<<':
                    self.__qcx_lines.extend([
                        f'JUMPIF {label}_FILL {index} < {distance}', f'LET {index} -= {distance}',
                    ])
                else:
                    self.__qcx_lines.extend([
                        f'LET {remaining} := {count}', f'LET {remaining} -= {distance}',
                        f'JUMPIF {label}_FILL {index} >= {remaining}', f'LET {index} += {distance}',
                    ])
                # A fill bit sees only complements outside this particular
                # shift, and never reads an original source position.
                fills.append((label, inverted))
        self.__qcx_lines.append(f'LET {bit} := {snapshot}:{index}')
        if inverted:
            self.__qcx_lines.append(f'LET {bit} ^= 1')
        self.__qcx_lines.append(f'JUMP {prefix}_STORE')
        for label, complemented in fills:
            self.__qcx_lines.extend([
                f'@{label}_FILL', f'LET {bit} := {int(complemented)}', f'JUMP {prefix}_STORE',
            ])

    def __emit_popcount_bitwise_bit(
            self, bit: str, left: str | int, right: str | int, operator: str,
            complements: _RuntimeBitRangePopcountComplements | None,
            right_bit: str | None) -> None:
        # Captures contain individual 0/1 bits. XOR with one flips exactly
        # that bit, without extending a signed native complement's width.
        self.__qcx_lines.append(f'LET {bit} := {left}')
        if complements is not None:
            if complements.left:
                self.__qcx_lines.append(f'LET {bit} ^= 1')
            if complements.right:
                assert right_bit is not None
                self.__qcx_lines.extend([f'LET {right_bit} := {right}', f'LET {right_bit} ^= 1'])
                right = right_bit
        self.__qcx_lines.append(f'LET {bit} {operator}= {right}')
        if complements is not None and complements.outer:
            self.__qcx_lines.append(f'LET {bit} ^= 1')

    def __emit_bit_range_pair_population_count(self, info: _RuntimeBitRangePopcountInfo) -> str:
        self.__check_range_pair_popcount_output()
        previous_declarations = self.__declared_temporary_variables.copy()
        prefix = f'QASM2QCX_POPCOUNT_{self.__population_count_index}'
        self.__population_count_index += 1
        # Complete the LHS, including its bit reads, before starting the RHS.
        if info.operand_chains is not None:
            left, count = self.__capture_popcount_operand(info.operand_chains[0], prefix + '_LEFT')
            right, _ = self.__capture_popcount_operand(info.operand_chains[1], prefix + '_RIGHT', expected_count=count)
        else:
            left, count = self.__capture_popcount_range_snapshot(info.selection, prefix + '_LEFT')
            right, _ = self.__capture_popcount_range_snapshot(
                info.pair.right_selection, prefix + '_RIGHT', expected_count=count)
        rotated = any(isinstance(record.expression, ast.FunctionCall) for record in info.result_operations)
        if rotated:
            result, starts, ends, intervals, inverted = self.__emit_bit_range_popcount_mixed_intervals(
                info.result_operations, info.selection.source_width, count, prefix)
            interval_slot, slot, end = (self.__add_new_temporary_variable(ValueType.INT) for _ in range(3))
        elif info.result_operations:
            result, slot, end, inverted = self.__emit_bit_range_popcount_interleaved_window(
                info.result_operations, count, prefix)
        else:
            result = self.__add_new_temporary_variable(ValueType.UINT)
            slot = self.__add_new_temporary_variable(ValueType.INT)
            end, inverted = count, False
        bit = self.__add_new_temporary_variable(ValueType.BIT)
        complements = info.binary_complements
        right_bit = (self.__add_new_temporary_variable(ValueType.BIT)
                     if complements is not None and complements.right else None)
        # Counts/indices fit INT and the accumulator never exceeds the count.
        # Runtime loops keep code size independent of the buffer capacities,
        # apart from existing constant-word lookup preparation.
        if not info.result_operations:
            self.__qcx_lines.extend([f'LET {result} := 0', f'LET {slot} := 0'])
        if rotated:
            self.__qcx_lines.extend([
                f'LET {interval_slot} := 0', f'@{prefix}_INTERVAL',
                f'JUMPIF {prefix}_DONE {interval_slot} >= {intervals}',
                f'LET {slot} := {starts}:{interval_slot}', f'LET {end} := {ends}:{interval_slot}',
            ])
        interval_done = prefix + ('_INTERVAL_NEXT' if rotated else '_DONE')
        self.__qcx_lines.extend([
            f'@{prefix}_PAIR_COUNT', f'JUMPIF {interval_done} {slot} >= {end}',
        ])
        self.__emit_popcount_bitwise_bit(
            bit, f'{left}:{slot}', f'{right}:{slot}', info.pair.expression.op.name, complements, right_bit)
        self.__qcx_lines.extend([
            f'LET {result} {"-=" if inverted else "+="} :UINT:{bit}', f'LET {slot} += 1',
            f'JUMP {prefix}_PAIR_COUNT',
        ])
        if rotated:
            self.__qcx_lines.extend([
                f'@{interval_done}', f'LET {interval_slot} += 1', f'JUMP {prefix}_INTERVAL',
            ])
        self.__qcx_lines.append(f'@{prefix}_DONE')
        self.__check_range_pair_popcount_output()
        self.__hoist_temporary_declarations(previous_declarations)
        return result

    def __emit_bit_range_bitwise_population_count(self, info: _RuntimeBitRangePopcountInfo) -> str:
        record = info.bitwise
        complements = info.binary_complements
        extra = (int(complements.left) + 2 * int(complements.right) + int(complements.outer)
                 if complements is not None else 0)
        windowed = bool(info.result_operations)
        rotated = any(isinstance(record.expression, ast.FunctionCall) for record in info.result_operations)
        # Range snapshots need two instructions per bit; combining/counting
        # needs three more. Reject excessive fixed widths before allocation.
        def check_output(extra=0):
            if len(self.__qcx_lines) + extra > self.MAX_LOOP_OUTPUT_LINES:
                raise InvalidFunctionCallException('Runtime range bitwise popcount exceeds QCX output limit')
        check_output((5 + extra + 3 * windowed) * record.width + 2)
        previous_declarations = self.__declared_temporary_variables.copy()
        with contextlib.ExitStack() as operands:
            def capture_fixed():
                values = operands.enter_context(self.__bit_register_expression_value(record.fixed_operand))
                if len(values) != record.width:
                    raise InvalidBitOperandException('Runtime range bitwise popcount operand width mismatch')
                check_output(sum(isinstance(value, str) for value in values))
                snapshot = []
                for value in values:
                    if isinstance(value, str):
                        bit = self.__add_new_temporary_variable(ValueType.BIT)
                        self.__qcx_lines.append(f'LET {bit} := {value}')
                        snapshot.append(bit)
                    else:
                        snapshot.append(value)
                check_output()
                return tuple(snapshot)

            # Capture complete operands at their written evaluation points.
            # Range capture checks the count before lookup or any bit read;
            # borrowed fixed-register reads must be frozen before the RHS.
            def capture_range():
                if info.operand_chains is None:
                    return operands.enter_context(self.__runtime_bit_range_value(info.selection, record.width))
                position = 0 if record.range_on_left else 1
                prefix = f'QASM2QCX_POPCOUNT_{self.__population_count_index}'
                self.__population_count_index += 1
                label = prefix + ('_LEFT' if record.range_on_left else '_RIGHT')
                snapshot, _ = self.__capture_popcount_operand(
                    info.operand_chains[position], label, str(record.width), width=record.width)
                return tuple(f'{snapshot}:{index}' for index in range(record.width))
            if record.range_on_left:
                left = capture_range()
                right = capture_fixed()
            else:
                left = capture_fixed()
                right = capture_range()
            if windowed:
                # QCX comparisons require a variable on the left, including
                # the negative-distance assertion in the shared window/interval helpers.
                count = self.__add_new_temporary_variable(ValueType.INT)
                self.__qcx_lines.append(f'LET {count} := {record.width}')
                prefix = f'QASM2QCX_POPCOUNT_{self.__population_count_index}'
                self.__population_count_index += 1
                if rotated:
                    result, starts, ends, intervals, inverted = self.__emit_bit_range_popcount_mixed_intervals(
                        info.result_operations, record.width, count, prefix)
                    interval_slot, first, end = (self.__add_new_temporary_variable(ValueType.INT) for _ in range(3))
                else:
                    result, first, end, inverted = self.__emit_bit_range_popcount_interleaved_window(
                        info.result_operations, count, prefix)
            else:
                inverted = False
            check_output((3 + extra + 3 * windowed) * record.width + 3
                         + int(complements is not None and complements.right))
            if not windowed:
                result = self.__add_new_temporary_variable(ValueType.UINT)
            bit = self.__add_new_temporary_variable(ValueType.BIT)
            right_bit = (self.__add_new_temporary_variable(ValueType.BIT)
                         if complements is not None and complements.right else None)
            if not windowed:
                self.__qcx_lines.append(f'LET {result} := 0')
            if rotated:
                self.__qcx_lines.extend([
                    f'LET {interval_slot} := 0', f'@{prefix}_INTERVAL',
                    f'JUMPIF {prefix}_DONE {interval_slot} >= {intervals}',
                    f'LET {first} := {starts}:{interval_slot}', f'LET {end} := {ends}:{interval_slot}',
                ])
            interval_done = prefix + ('_INTERVAL_NEXT' if rotated else '_DONE') if windowed else None
            for index, (lhs, rhs) in enumerate(zip(left, right)):
                if windowed:
                    self.__qcx_lines.extend([
                        f'JUMPIF {prefix}_BIT_{index}_NEXT {first} > {index}',
                        f'JUMPIF {interval_done} {end} <= {index}',
                    ])
                self.__emit_popcount_bitwise_bit(bit, lhs, rhs, record.expression.op.name, complements, right_bit)
                self.__qcx_lines.append(f'LET {result} {"-=" if inverted else "+="} :UINT:{bit}')
                if windowed:
                    self.__qcx_lines.append(f'@{prefix}_BIT_{index}_NEXT')
            if windowed:
                if rotated:
                    self.__qcx_lines.extend([
                        f'@{interval_done}', f'LET {interval_slot} += 1', f'JUMP {prefix}_INTERVAL',
                    ])
                self.__qcx_lines.append(f'@{prefix}_DONE')
            check_output()
        # The enclosing population-count visitor releases all scratch,
        # including these fixed snapshots, and keeps only the UINT result.
        self.__hoist_temporary_declarations(previous_declarations)
        return result

    def __emit_bit_range_population_count(
            self, selection: _RuntimeBitRangeSelection, complemented: bool = False,
            shift: ast.BinaryExpression | None = None, outer_complemented: bool = False,
            rotation: ast.FunctionCall | None = None,
            shifts: tuple[_RuntimeBitRangePopcountShift, ...] = (),
            operations: tuple[_RuntimeBitRangePopcountOperation, ...] = (),
            rotations: tuple[_RuntimeBitRangePopcountRotation, ...] = (),
            mixed_operations: tuple[_RuntimeBitRangePopcountOperation, ...] = ()) -> str:
        # The enclosing population-count visitor owns all scratch storage and
        # preserves only the returned accumulator, including after failures.
        if len(self.__qcx_lines) + 1 > self.MAX_LOOP_OUTPUT_LINES:
            raise InvalidFunctionCallException('Runtime range popcount exceeds QCX output limit')
        previous_declarations = self.__declared_temporary_variables.copy()
        prefix = f'QASM2QCX_POPCOUNT_{self.__population_count_index}'
        self.__population_count_index += 1
        # Capture/check the complete range before reading any source bit or
        # preparing constant words. No fixed selected width is required.
        positions, count = self.__capture_bit_range_positions(
            selection.bounds, selection.source_width, prefix + '_RANGE')
        rotation_calls = (tuple(record.expression for record in rotations) if rotations else
                          (() if rotation is None else (rotation,)))
        for call in rotation_calls:
            # Each rotation only permutes positions, but every distance and
            # checked INT conversion must execute, innermost to outermost.
            value, value_type, kind = self.__condition_operand(call.arguments[1])
            operand, _ = self.__converted_operand(value, value_type, kind, ValueType.INT)
            distance = self.__add_new_temporary_variable(ValueType.INT)
            self.__qcx_lines.append(f'LET {distance} := {operand}')
        if rotation_calls:
            # No zeros are introduced: complements commute with rotation.
            # Reuse unshifted counting with the combined parity, without
            # normalizing/negating the distance or materializing rotated bits.
            complemented = complemented != outer_complemented
            outer_complemented = False
        windowed = shift is not None or len(shifts) > 1 or bool(operations) or bool(mixed_operations)
        if mixed_operations:
            result, starts, ends, intervals, complemented = self.__emit_bit_range_popcount_mixed_intervals(
                mixed_operations, selection.source_width, count, prefix)
            slot = self.__add_new_temporary_variable(ValueType.INT)
            index = self.__add_new_temporary_variable(ValueType.INT)
            end = self.__add_new_temporary_variable(ValueType.INT)
        elif operations:
            result, index, end, complemented = self.__emit_bit_range_popcount_interleaved_window(
                operations, count, prefix)
            outer_complemented = False
            self.__qcx_lines.append(f'JUMPIF {prefix}_DONE {index} >= {end}')
        elif windowed:
            result = self.__add_new_temporary_variable(ValueType.UINT)
            # An oversized shift contributes only filled zeros, or ones after
            # an outer complement. Initialize before the early-result jump.
            initial = f':UINT:{count}' if outer_complemented else '0'
            self.__qcx_lines.append(f'LET {result} := {initial}')
            if len(shifts) > 1:
                index, end = self.__emit_bit_range_popcount_nested_shift_window(shifts, count, prefix)
                # All distances have been evaluated, even if the survivor
                # interval emptied earlier. Only now may counting be skipped.
                self.__qcx_lines.append(f'JUMPIF {prefix}_DONE {index} >= {end}')
            else:
                index, end = self.__emit_bit_range_popcount_shift_window(shift, count, prefix + '_DONE')
            if complemented:
                # For surviving length m=end-index and original ones k:
                # inner only starts at m and subtracts k; both complements
                # start at n-m and add k. All intermediates stay within n.
                initial = count if outer_complemented else end
                subtract = end if outer_complemented else index
                self.__qcx_lines.extend([
                    f'LET {result} := :UINT:{initial}', f'LET {result} -= :UINT:{subtract}',
                ])
                if outer_complemented:
                    self.__qcx_lines.append(f'LET {result} += :UINT:{index}')
        constant = (self.__constant_bit_register(selection.source_name)
                    if selection.source_name in self.__const_bit_register_values else None)
        words = self.__prepare_constant_bit_lookup(constant) if constant is not None else None
        if not windowed:
            result = self.__add_new_temporary_variable(ValueType.UINT)
            index = self.__add_new_temporary_variable(ValueType.INT)
            end = count
        position = self.__add_new_temporary_variable(ValueType.INT)
        bit = self.__add_new_temporary_variable(ValueType.BIT)
        # One pass over original surviving bits suffices for all parities.
        operation = '-=' if complemented != outer_complemented else '+='
        if not windowed:
            initial = f':UINT:{count}' if complemented else '0'
            self.__qcx_lines.extend([f'LET {result} := {initial}', f'LET {index} := 0'])
        if mixed_operations:
            self.__qcx_lines.extend([
                f'LET {slot} := 0', f'@{prefix}_INTERVAL',
                f'JUMPIF {prefix}_DONE {slot} >= {intervals}',
                f'LET {index} := {starts}:{slot}', f'LET {end} := {ends}:{slot}',
            ])
        interval_done = prefix + ('_INTERVAL_NEXT' if mixed_operations else '_DONE')
        self.__qcx_lines.extend([
            f'@{prefix}_LOOP', f'JUMPIF {interval_done} {index} >= {end}',
            f'LET {position} := {positions}:{index}',
        ])
        if words is None:
            source = self.__capitalize_variable_name(selection.source_name)
            operand = source if selection.source_width == 1 else f'{source}:{position}'
            self.__qcx_lines.append(f'LET {bit} := {operand}')
        else:
            previous_temporaries = self.__declared_temporary_variables.copy()
            self.__copy_prepared_constant_bit(words, position, bit)
            # Lookup scratch declarations must execute once, outside the
            # counting loop. Keep their assignments inside each iteration.
            self.__hoist_temporary_declarations(previous_temporaries)
        # The position buffer bounds count within native INT capacity, so
        # neither the count nor incrementing index up to count can overflow.
        # Every accumulator update remains between zero and the selected length.
        self.__qcx_lines.extend([
            f'LET {result} {operation} :UINT:{bit}', f'LET {index} += 1',
            f'JUMP {prefix}_LOOP',
        ])
        if mixed_operations:
            self.__qcx_lines.extend([
                f'@{interval_done}', f'LET {slot} += 1', f'JUMP {prefix}_INTERVAL',
            ])
        self.__qcx_lines.append(f'@{prefix}_DONE')
        if len(self.__qcx_lines) > self.MAX_LOOP_OUTPUT_LINES:
            raise InvalidFunctionCallException('Runtime range popcount exceeds QCX output limit')
        if windowed:
            # The zero-result jump skips lookup and loop setup. Their storage
            # must still exist for later expressions that reuse these names.
            self.__hoist_temporary_declarations(previous_declarations)
        return result

    def __capture_bit_range_popcount_distances(
            self, operations: tuple[_RuntimeBitRangePopcountOperation, ...], count: str,
            prefix: str) -> tuple[str | None, ...]:
        # Evaluate all distances in execution order, even after complete bit
        # loss. Shifts save min(distance, n); rotations save a normalized
        # left-rotation distance after the ordinary checked INT conversion.
        distances = []
        for number, operation in enumerate(operations):
            if isinstance(operation.expression, ast.UnaryExpression):
                distances.append(None)
                continue
            if isinstance(operation.expression, ast.FunctionCall):
                value, value_type, kind = self.__condition_operand(operation.expression.arguments[1])
                operand, _ = self.__converted_operand(value, value_type, kind, ValueType.INT)
                distance = self.__add_new_temporary_variable(ValueType.INT)
                label = f'{prefix}_DISTANCE_{number}'
                self.__qcx_lines.extend([
                    f'LET {distance} := {operand}', f'LET {distance} %= {count}',
                    f'JUMPIF {label}_NORMALIZED {distance} >= 0',
                    f'LET {distance} += {count}', f'@{label}_NORMALIZED',
                ])
                if operation.expression.name.name == 'rotr':
                    remaining = self.__add_new_temporary_variable(ValueType.INT)
                    self.__qcx_lines.extend([
                        f'JUMPIF {label}_READY {distance} == 0',
                        f'LET {remaining} := {count}', f'LET {remaining} -= {distance}',
                        f'LET {distance} := {remaining}', f'@{label}_READY',
                    ])
                distances.append(distance)
                continue
            value, value_type, kind = self.__condition_operand(operation.expression.rhs)
            distance = self.__add_new_temporary_variable(ValueType.INT)
            if kind == ValueKind.LITERAL:
                raw = int(value)
                if raw < 0:
                    self.__qcx_lines.append(f'ASSERT {count} < 0')
                    operand = '0'
                else:
                    operand = count if raw >= self.QCX_INT_MAX else str(raw)
                self.__qcx_lines.append(f'LET {distance} := {operand}')
                if 0 <= raw < self.QCX_INT_MAX:
                    label = f'{prefix}_DISTANCE_{number}_CAPTURED'
                    self.__qcx_lines.extend([
                        f'JUMPIF {label} {distance} <= {count}',
                        f'LET {distance} := {count}', f'@{label}',
                    ])
            else:
                captured = self.__add_new_temporary_variable(value_type)
                self.__qcx_lines.append(f'LET {captured} := {value}')
                if value_type == ValueType.INT:
                    self.__qcx_lines.append(f'ASSERT {captured} >= 0')
                limit = f':UINT:{count}' if value_type == ValueType.UINT else count
                operand = f':INT:{captured}' if value_type == ValueType.UINT else captured
                label = f'{prefix}_DISTANCE_{number}_CAPTURED'
                self.__qcx_lines.extend([
                    f'LET {distance} := {count}', f'JUMPIF {label} {captured} >= {limit}',
                    f'LET {distance} := {operand}', f'@{label}',
                ])
            distances.append(distance)
        return tuple(distances)

    def __emit_bit_range_popcount_mixed_intervals(
            self, operations: tuple[_RuntimeBitRangePopcountOperation, ...], width: int,
            count: str, prefix: str) -> tuple[str, str, str, str, bool]:
        distances = self.__capture_bit_range_popcount_distances(operations, count, prefix)
        # Inverse shifts only clip/translate disjoint intervals. An inverse
        # rotation has one circular cut, splitting at most one interval.
        # There cannot be more nonempty disjoint intervals than positions.
        rotations = sum(isinstance(record.expression, ast.FunctionCall) for record in operations)
        capacity = min(1 + rotations, 2 * width, self.MAX_RUNTIME_BIT_RANGE_POSITIONS)
        starts, ends, next_starts, next_ends = (
            self.__add_temporary_int_array(capacity, 'POPCOUNT_INTERVALS') for _ in range(4))
        length, next_length, slot, start, end, removed, remaining = (
            self.__add_new_temporary_variable(ValueType.INT) for _ in range(7))
        result = self.__add_new_temporary_variable(ValueType.UINT)
        self.__qcx_lines.extend([
            f'LET {result} := 0', f'LET {length} := 1',
            f'LET {starts}:0 := 0', f'LET {ends}:0 := {count}',
        ])
        complemented = False
        for number in reversed(range(len(operations))):
            operation = operations[number].expression
            if isinstance(operation, ast.UnaryExpression):
                complemented = not complemented
                continue
            label = f'{prefix}_REVERSE_{number}'
            self.__qcx_lines.extend([
                f'LET {next_length} := 0', f'LET {slot} := 0', f'@{label}_LOOP',
                f'JUMPIF {label}_DONE {slot} >= {length}',
                f'LET {start} := {starts}:{slot}', f'LET {end} := {ends}:{slot}',
            ])
            if isinstance(operation, ast.FunctionCall):
                self.__emit_bit_range_popcount_reverse_rotation(
                    distances[number], count, start, end, remaining,
                    next_starts, next_ends, next_length, label)
            else:
                if complemented:
                    self.__qcx_lines.extend([f'LET {removed} := {end}', f'LET {removed} -= {start}'])
                self.__emit_bit_range_popcount_reverse_shift(
                    operation, distances[number], count, start, end, remaining, label)
                if complemented:
                    # Only the suffix parity acts on this shift's zero fill.
                    # Count those positions once, and remove them from tracing.
                    self.__qcx_lines.extend([
                        f'LET {remaining} := {end}', f'LET {remaining} -= {start}',
                        f'LET {removed} -= {remaining}', f'LET {result} += :UINT:{removed}',
                    ])
            self.__qcx_lines.extend([
                f'JUMPIF {label}_SKIP {start} >= {end}',
                f'LET {next_starts}:{next_length} := {start}',
                f'LET {next_ends}:{next_length} := {end}', f'LET {next_length} += 1',
                f'@{label}_SKIP', f'LET {slot} += 1', f'JUMP {label}_LOOP', f'@{label}_DONE',
            ])
            starts, ends, next_starts, next_ends = next_starts, next_ends, starts, ends
            length, next_length = next_length, length
        if complemented:
            # Original surviving bits contribute length-k. Fill contributions
            # and these lengths cover disjoint subsets of the n output bits.
            self.__qcx_lines.extend([
                f'LET {slot} := 0', f'@{prefix}_LENGTHS',
                f'JUMPIF {prefix}_LENGTHS_DONE {slot} >= {length}',
                f'LET {remaining} := {ends}:{slot}', f'LET {remaining} -= {starts}:{slot}',
                f'LET {result} += :UINT:{remaining}', f'LET {slot} += 1',
                f'JUMP {prefix}_LENGTHS', f'@{prefix}_LENGTHS_DONE',
            ])
        return result, starts, ends, length, complemented

    def __emit_bit_range_popcount_reverse_rotation(
            self, distance: str, count: str, start: str, end: str, remaining: str,
            starts: str, ends: str, length: str, label: str) -> None:
        # Trace a normalized left rotation backward by subtracting distance
        # modulo n. Never compute index+n or negate an unnormalized INT_MIN.
        self.__qcx_lines.extend([
            f'LET {remaining} := {count}', f'LET {remaining} -= {distance}',
            f'JUMPIF {label}_LOW {end} <= {distance}',
            f'JUMPIF {label}_HIGH {start} >= {distance}',
            # The interval straddles the circular cut. Save its wrapped half;
            # the caller appends the other half through the common path.
            f'LET {start} += {remaining}', f'LET {starts}:{length} := {start}',
            f'LET {ends}:{length} := {count}', f'LET {length} += 1',
            f'LET {start} := 0', f'LET {end} -= {distance}', f'JUMP {label}_NEXT',
            f'@{label}_LOW', f'LET {start} += {remaining}', f'LET {end} += {remaining}',
            f'JUMP {label}_NEXT', f'@{label}_HIGH',
            f'LET {start} -= {distance}', f'LET {end} -= {distance}', f'@{label}_NEXT',
        ])

    def __emit_bit_range_popcount_interleaved_window(
            self, operations: tuple[_RuntimeBitRangePopcountOperation, ...], count: str,
            prefix: str) -> tuple[str, str, str, bool]:
        distances = self.__capture_bit_range_popcount_distances(operations, count, prefix)
        result = self.__add_new_temporary_variable(ValueType.UINT)
        index = self.__add_new_temporary_variable(ValueType.INT)
        end = self.__add_new_temporary_variable(ValueType.INT)
        removed = self.__add_new_temporary_variable(ValueType.INT)
        remaining = self.__add_new_temporary_variable(ValueType.INT)
        self.__qcx_lines.extend([f'LET {result} := 0', f'LET {index} := 0', f'LET {end} := {count}'])
        complemented = False
        for number in reversed(range(len(operations))):
            operation = operations[number].expression
            if isinstance(operation, ast.UnaryExpression):
                complemented = not complemented
                continue
            if complemented:
                self.__qcx_lines.extend([f'LET {removed} := {end}', f'LET {removed} -= {index}'])
            self.__emit_bit_range_popcount_reverse_shift(
                operation, distances[number], count, index, end, remaining, f'{prefix}_REVERSE_{number}')
            if complemented:
                # Positions traced to this shift's zero fill become ones.
                # Count each such position once, then stop tracing it.
                self.__qcx_lines.extend([
                    f'LET {remaining} := {end}', f'LET {remaining} -= {index}',
                    f'LET {removed} -= {remaining}', f'LET {result} += :UINT:{removed}',
                ])
        if complemented:
            # Surviving original bits contribute length-k instead of k.
            self.__qcx_lines.extend([
                f'LET {remaining} := {end}', f'LET {remaining} -= {index}',
                f'LET {result} += :UINT:{remaining}',
            ])
        return result, index, end, complemented

    def __emit_bit_range_popcount_reverse_shift(
            self, shift: ast.BinaryExpression, distance: str, count: str,
            index: str, end: str, remaining: str, label: str) -> None:
        # Trace [index,end) backward to this shift's input. All coordinates
        # and the captured distance lie in [0,n]; the interval never grows.
        if shift.op.name == '<<':
            self.__qcx_lines.extend([
                f'JUMPIF {label}_EMPTY {end} <= {distance}',
                f'LET {end} -= {distance}',
                f'JUMPIF {label}_LOWER {index} >= {distance}',
                f'LET {index} := 0', f'JUMP {label}_NEXT',
                f'@{label}_LOWER', f'LET {index} -= {distance}', f'JUMP {label}_NEXT',
            ])
        else:
            self.__qcx_lines.extend([
                f'LET {remaining} := {count}', f'LET {remaining} -= {distance}',
                f'JUMPIF {label}_EMPTY {index} >= {remaining}',
                f'LET {index} += {distance}',
                f'JUMPIF {label}_UPPER {end} <= {remaining}',
                f'LET {end} := {remaining}',
                f'@{label}_UPPER', f'LET {end} += {distance}', f'JUMP {label}_NEXT',
            ])
        self.__qcx_lines.extend([
            f'@{label}_EMPTY', f'LET {index} := 0', f'LET {end} := 0', f'@{label}_NEXT',
        ])

    def __emit_bit_range_popcount_nested_shift_window(
            self, shifts: tuple[_RuntimeBitRangePopcountShift, ...], count: str, prefix: str) -> tuple[str, str]:
        # The surviving original interval starts at index, has length bits,
        # and currently starts at offset in the n-bit intermediate register.
        # index+length <= n and offset+length <= n. No cumulative signed
        # displacement is needed, even for arbitrarily long shift chains.
        index = self.__add_new_temporary_variable(ValueType.INT)
        length = self.__add_new_temporary_variable(ValueType.INT)
        offset = self.__add_new_temporary_variable(ValueType.INT)
        self.__qcx_lines.extend([f'LET {index} := 0', f'LET {length} := {count}', f'LET {offset} := 0'])
        for number, shift in enumerate(shifts):
            label = f'{prefix}_SHIFT_{number}'
            value, value_type, kind = self.__condition_operand(shift.expression.rhs)
            if kind == ValueKind.LITERAL:
                distance = int(value)
                if distance < 0:
                    self.__qcx_lines.append(f'ASSERT {index} < 0')
                    continue
                if distance >= self.QCX_INT_MAX:
                    # Do not put an unrepresentable literal in QCX or skip
                    # generation/evaluation of any later distance expression.
                    self.__qcx_lines.extend([f'LET {length} := 0', f'LET {offset} := 0'])
                    continue
                operand = str(distance)
            else:
                captured = self.__add_new_temporary_variable(value_type)
                self.__qcx_lines.append(f'LET {captured} := {value}')
                if value_type == ValueType.INT:
                    self.__qcx_lines.append(f'ASSERT {captured} >= 0')
                limit = f':UINT:{count}' if value_type == ValueType.UINT else count
                self.__qcx_lines.append(f'JUMPIF {label}_EMPTY {captured} >= {limit}')
                # UINT narrowing follows the oversized guard and is safe.
                operand = f':INT:{captured}' if value_type == ValueType.UINT else captured
            self.__qcx_lines.append(f'JUMPIF {label}_NEXT {length} == 0')
            remaining = self.__add_new_temporary_variable(ValueType.INT)
            if shift.expression.op.name == '<<':
                self.__qcx_lines.extend([
                    f'LET {remaining} := {count}', f'LET {remaining} -= {offset}',
                    f'JUMPIF {label}_EMPTY {remaining} <= {operand}',
                    # The preceding subtraction/guard proves offset+d < n.
                    f'LET {offset} += {operand}',
                    f'LET {remaining} := {count}', f'LET {remaining} -= {offset}',
                    f'JUMPIF {label}_NEXT {length} <= {remaining}',
                    f'LET {length} := {remaining}', f'JUMP {label}_NEXT',
                ])
            else:
                self.__qcx_lines.extend([
                    f'JUMPIF {label}_SLIDE {offset} >= {operand}',
                    f'LET {remaining} := {operand}', f'LET {remaining} -= {offset}',
                    f'JUMPIF {label}_EMPTY {remaining} >= {length}',
                    # trim < length, so index+trim remains within n.
                    f'LET {index} += {remaining}', f'LET {length} -= {remaining}',
                    f'LET {offset} := 0', f'JUMP {label}_NEXT',
                    f'@{label}_SLIDE', f'LET {offset} -= {operand}', f'JUMP {label}_NEXT',
                ])
            self.__qcx_lines.extend([
                f'@{label}_EMPTY', f'LET {length} := 0', f'LET {offset} := 0', f'@{label}_NEXT',
            ])
        end = self.__add_new_temporary_variable(ValueType.INT)
        self.__qcx_lines.extend([f'LET {end} := {index}', f'LET {end} += {length}'])
        return index, end

    def __emit_bit_range_popcount_shift_window(
            self, shift: ast.BinaryExpression, count: str, done: str) -> tuple[str, str]:
        # The enclosing visitor owns scratch storage, including count-expression
        # results. All range positions have already passed validation here.
        index = self.__add_new_temporary_variable(ValueType.INT)
        end = self.__add_new_temporary_variable(ValueType.INT)
        self.__qcx_lines.extend([f'LET {index} := 0', f'LET {end} := {count}'])
        value, value_type, kind = self.__condition_operand(shift.rhs)
        if kind == ValueKind.LITERAL:
            distance = int(value)
            if distance < 0:
                # Do not emit an out-of-range negative literal on skipped paths.
                self.__qcx_lines.append(f'ASSERT {index} < 0')
                return index, end
            if distance >= self.QCX_INT_MAX:
                # Even arbitrarily large literals exceed every supported count.
                self.__qcx_lines.append(f'JUMP {done}')
                return index, end
            operand = str(distance)
            self.__qcx_lines.append(f'JUMPIF {done} {count} <= {operand}')
        else:
            captured = self.__add_new_temporary_variable(value_type)
            self.__qcx_lines.append(f'LET {captured} := {value}')
            if value_type == ValueType.INT:
                self.__qcx_lines.append(f'ASSERT {captured} >= 0')
            limit = f':UINT:{count}' if value_type == ValueType.UINT else count
            self.__qcx_lines.append(f'JUMPIF {done} {captured} >= {limit}')
            # Conversion is safe only after the oversized-count guard.
            operand = f':INT:{captured}' if value_type == ValueType.UINT else captured
        if shift.op.name == '<<':
            self.__qcx_lines.append(f'LET {end} -= {operand}')
        else:
            self.__qcx_lines.append(f'LET {index} := {operand}')
        return index, end

    def __emit_uint_population_count(self, argument: ast.Expression, angle_type: _AngleType | None = None) -> str:
        # The enclosing visitor owns all scratch storage and preserves only
        # the returned native UINT accumulator, including on failure.
        if angle_type is None:
            _, uint_type, _ = self.__integer_operand_info(argument, None)
        else:
            uint_type = _UnsignedIntegerType(angle_type.width, angle_type.is_sized)
        value, value_type, value_kind = self.__condition_operand(argument)
        operand = str(value) if angle_type is not None else self.__converted_operand(
            value, value_type, value_kind, ValueType.UINT)[0]
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

    def __expression_mentions_angle(
            self, expression: ast.Expression,
            iterators: set[str] | dict[str, ValueType] | None = None) -> bool:
        if self.__angle_bit_selection(expression, iterators) is not None:
            return False  # Operations outside the selection use register semantics.
        if (isinstance(expression, ast.Cast) and isinstance(expression.type, ast.UintType)
                and (self.__runtime_angle_range_uint_cast_info(expression, iterators) is not None
                     or self.__runtime_angle_selection_uint_cast_info(expression, iterators) is not None)):
            return False  # The cast consumes the angle selection and yields UINT.
        if isinstance(expression, ast.Cast) and isinstance(expression.type, ast.AngleType):
            return True
        if isinstance(expression, ast.Cast) and isinstance(expression.type, ast.BitType) and expression.type.size is not None:
            return False  # Register operations outside the cast use bit semantics.
        if isinstance(expression, ast.IndexExpression) and isinstance(expression.collection, ast.Identifier):
            source_type = self.__classical_source_types.get(expression.collection.name)
            if isinstance(source_type, ast.ArrayType) and isinstance(source_type.base_type, ast.AngleType):
                return True
        if isinstance(expression, ast.Identifier):
            if iterators is not None and expression.name in iterators:
                return isinstance(iterators, dict) and iterators[expression.name] == ValueType.ANGLE
            for scope in reversed(self.__loop_bindings):
                if expression.name in scope:
                    binding = scope[expression.name]
                    return isinstance(binding, _RuntimeIteratorBinding) and binding.value_type == ValueType.ANGLE
            return (expression.name in self.__const_angle_values
                    or isinstance(self.__classical_source_types.get(expression.name), ast.AngleType))
        for field in dataclasses.fields(expression):
            value = getattr(expression, field.name)
            children = value if isinstance(value, list) else [value]
            if any(isinstance(child, ast.Expression) and self.__expression_mentions_angle(child, iterators)
                   for child in children):
                return True
        return False

    def __is_angle_identifier(self, name: str, iterators: dict[str, ValueType]) -> bool:
        # Do not ask scalar expression lookup about unrelated register/array
        # names. Active iterators can shadow either angle kind.
        if name in iterators or any(name in scope for scope in self.__loop_bindings):
            return self.__loop_identifier_info(ast.Identifier(name), iterators)[0] == ValueType.ANGLE
        return ((name in self.__const_angle_values
                 or isinstance(self.__classical_source_types.get(name), ast.AngleType))
                and self.__loop_identifier_info(ast.Identifier(name), iterators)[0] == ValueType.ANGLE)

    def __validate_angle_assignment(
            self, statement: ast.ClassicalAssignment, iterators: dict[str, ValueType]) -> None:
        name = self.__operand_name(statement.lvalue)
        array_type = self.__classical_source_types.get(name)
        angle_array = isinstance(array_type, ast.ArrayType) and isinstance(array_type.base_type, ast.AngleType)
        lhs = None
        if angle_array:
            if not isinstance(statement.lvalue, ast.IndexedIdentifier) or len(statement.lvalue.indices) != 1:
                raise UnsupportedOpenQASMError('whole or multidimensional angle-array assignment')
            lhs = ast.IndexExpression(statement.lvalue.name, statement.lvalue.indices[0])
            self.__loop_index_info(lhs, iterators)
        if angle_array or self.__is_angle_identifier(name, iterators):
            expression = self.__angle_assignment_expression(statement, lhs)
            source_type, _ = self.__loop_expression_info(expression, iterators)
            if source_type not in (ValueType.FLOAT, ValueType.ANGLE):
                raise UnsupportedOpenQASMError('angle conversion requires float or angle input')
        elif self.__expression_mentions_angle(statement.rvalue, iterators):
            if self.__loop_expression_info(statement.rvalue, iterators)[0] == ValueType.ANGLE:
                raise NoImplicitCastException

    def __runtime_bit_set_measurement_info(
            self, measurement: ast.QuantumMeasurement,
            target: ast.Identifier | ast.IndexedIdentifier | None,
            iterators: set[str] | dict[str, ValueType] | None = None) -> _RuntimeBitSetMeasurementInfo | None:
        if target is None:
            return None
        destination = self.__runtime_bit_set_selection(target, iterators, destination=True)
        if destination is None:
            return None
        name, indices = self.__static_measurement_qubit_source(
            measurement, iterators, expected_width=destination.width)
        return _RuntimeBitSetMeasurementInfo(destination, name, indices)

    def __runtime_bit_range_measurement_info(
            self, measurement: ast.QuantumMeasurement,
            target: ast.Identifier | ast.IndexedIdentifier | None,
            iterators: set[str] | dict[str, ValueType] | None = None) -> _RuntimeBitRangeMeasurementInfo | None:
        if target is None:
            return None
        destination = self.__runtime_bit_range_destination(target, iterators)
        if destination is None:
            if (isinstance(target, ast.IndexedIdentifier) and len(target.indices) == 1
                    and isinstance(target.indices[0], list) and len(target.indices[0]) == 1
                    and isinstance(target.indices[0][0], ast.RangeDefinition)):
                name = target.name.name
                source_type = self.__constant_source_types.get(name) or self.__classical_source_types.get(name)
                if source_type is None:
                    raise NoVariableNameException(name)
                if not isinstance(source_type, ast.BitType):
                    raise InvalidBitOperandException('A measurement target must be a bit variable')
            return None
        # At least one M is required per source qubit. Bound static expansion
        # before materializing whole registers or enormous valid source ranges.
        name, indices = self.__static_measurement_qubit_source(
            measurement, iterators, max_width=min(self.QCX_INT_MAX, self.MAX_LOOP_OUTPUT_LINES))
        return _RuntimeBitRangeMeasurementInfo(destination, name, indices)

    def __static_measurement_qubit_source(
            self, measurement: ast.QuantumMeasurement,
            iterators: set[str] | dict[str, ValueType] | None = None, *,
            expected_width: int | None = None, max_width: int | None = None) -> tuple[str, tuple[int, ...]]:
        qubit = measurement.qubit
        if not isinstance(qubit, (ast.Identifier, ast.IndexedIdentifier)):
            raise UnsupportedOpenQASMError('computed measurement qubit source')
        name = self.__operand_name(qubit)
        if ((iterators is not None and name in iterators)
                or any(name in scope for scope in self.__loop_bindings)):
            raise InvalidQubitOperandException('For-loop iterator is not a qubit operand')
        if (name not in self.__quantum_registers
                or (not self.__is_initialization_process and name not in self.__declared_quantum_registers)):
            raise InvalidQubitOperandException(f'Qubit register {name} is not declared')
        size = self.__quantum_registers[name]
        if isinstance(qubit, ast.Identifier):
            if name not in self.__sized_quantum_registers:
                raise InvalidBitOperandException('Measurement operands must both be scalars or both be registers')
            if expected_width is not None and size != expected_width:
                raise MeasurementSizeMismatchException(size, expected_width)
            if max_width is not None and size > max_width:
                raise InvalidQubitOperandException('Qubit selection exceeds supported size limit')
            indices = range(size)
        else:
            if len(qubit.indices) != 1:
                raise UnsupportedOpenQASMError('multidimensional measurement qubit indexing')
            indices, is_register = self.__static_bit_indices(
                qubit.indices[0], size, name, iterators, kind='qubit', max_indices=max_width)
            if not is_register:
                raise InvalidBitOperandException('Measurement operands must both be scalars or both be registers')
            if expected_width is not None and len(indices) != expected_width:
                raise MeasurementSizeMismatchException(len(indices), expected_width)
        return name, tuple(indices)

    def __runtime_bit_range_source(
            self, source: ast.Expression,
            iterators: set[str] | dict[str, ValueType] | None = None) -> _RuntimeBitRangeSelection | None:
        if (not isinstance(source, ast.IndexExpression) or not isinstance(source.collection, ast.Identifier)
                or not isinstance(source.index, list) or len(source.index) != 1
                or not isinstance(source.index[0], ast.RangeDefinition)):
            return None
        name, bounds = source.collection.name, source.index[0]
        source_type = self.__constant_source_types.get(name) or self.__classical_source_types.get(name)
        if not isinstance(source_type, ast.BitType):
            return None
        if ((iterators is not None and name in iterators)
                or any(name in scope for scope in self.__loop_bindings)):
            raise UnsupportedOpenQASMError('bit selection of a for-loop iterator')
        if bounds.start is None or bounds.end is None:
            raise InvalidBitOperandException('Bit-register range requires both bounds')
        # Inspect all operands without evaluating mixed runtime domains or
        # guessing the selected width, including in skipped loop bodies.
        start_type, start_runtime = self.__bit_index_info(bounds.start, iterators)
        step_type, step_runtime = ((ValueType.INT, False) if bounds.step is None
                                  else self.__bit_index_info(bounds.step, iterators))
        stop_type, stop_runtime = self.__bit_index_info(bounds.end, iterators)
        if not (start_runtime or step_runtime or stop_runtime):
            return None
        if source_type.size is None:
            raise InvalidBitOperandException('Runtime range copy requires a bit-register source')
        constant = self.__constant_bit_register(name) if name in self.__const_bit_register_values else None
        if (not self.__is_initialization_process and constant is None
                and self.__capitalize_variable_name(name) not in self.__sized_bit_variables):
            raise NoVariableNameException(name)
        size = constant.width if constant is not None else self.__bit_register_source_sizes[name]
        if not 1 <= size <= self.QCX_INT_MAX:
            raise InvalidBitOperandException('Runtime selection source width must fit QCX INT indexing')
        return _RuntimeBitRangeSelection(name, size, bounds, start_type, step_type, stop_type)

    def __runtime_bit_range_popcount_info(
            self, argument: ast.Expression,
            iterators: set[str] | dict[str, ValueType] | None = None) -> _RuntimeBitRangePopcountInfo | None:
        shapes = self.__runtime_bit_register_expression_shapes(argument, iterators)
        if shapes[id(argument)] >= 2:
            tree = self.__runtime_bit_register_expression_info(argument, shapes, iterators)
            selection = tree.first_selection
            if selection is not None:
                return _RuntimeBitRangePopcountInfo(selection, False, tree=tree)
        original = argument
        outer_complemented = False
        outer_nodes = []
        while isinstance(argument, ast.UnaryExpression) and argument.op.name == '~':
            outer_nodes.append(argument)
            outer_complemented = not outer_complemented
            argument = argument.expression
        if isinstance(argument, ast.BinaryExpression) and argument.op.name in ('&', '|', '^'):
            return self.__runtime_bit_range_popcount_bitwise_info(
                argument, iterators, original=original, outer_complemented=outer_complemented)
        if isinstance(argument, ast.FunctionCall) and argument.name.name in ('rotl', 'rotr'):
            info = self.__runtime_bit_range_popcount_rotation_chain_info(argument, outer_complemented, iterators)
            return info if info is not None else self.__runtime_bit_range_popcount_mixed_info(original, iterators)
        if isinstance(argument, ast.BinaryExpression) and argument.op.name in ('<<', '>>'):
            source = argument
            nodes = list(outer_nodes)
            while True:
                if isinstance(source, ast.BinaryExpression) and source.op.name in ('<<', '>>'):
                    nodes.append(source)
                    source = source.lhs
                elif isinstance(source, ast.UnaryExpression) and source.op.name == '~':
                    nodes.append(source)
                    source = source.expression
                else:
                    break
            selection = self.__runtime_bit_range_source(source, iterators)
            if selection is None:
                if isinstance(source, ast.BinaryExpression) and source.op.name in ('&', '|', '^'):
                    info = self.__runtime_bit_range_popcount_bitwise_info(
                        source, iterators, original=source, outer_complemented=False)
                    if info is not None:
                        return self.__runtime_bit_range_popcount_result_info(info, nodes, iterators)
                return self.__runtime_bit_range_popcount_mixed_info(original, iterators)
            # Validate every original count in execution order without reading
            # its value, including after a shift that could discard all bits.
            operations = tuple(_RuntimeBitRangePopcountOperation(
                node, self.__bit_index_info(node.rhs, iterators, role='register shift count')[0]
                if isinstance(node, ast.BinaryExpression) else None) for node in reversed(nodes))
            shift_positions = [i for i, operation in enumerate(operations)
                               if isinstance(operation.expression, ast.BinaryExpression)]
            first_shift, last_shift = shift_positions[0], shift_positions[-1]
            interleaved = any(isinstance(operation.expression, ast.UnaryExpression)
                              for operation in operations[first_shift+1:last_shift])
            # Before the first shift, every operation is a source complement.
            complemented = bool(first_shift % 2)
            shifts = tuple(_RuntimeBitRangePopcountShift(operation.expression, operation.count_type)
                           for operation in operations if isinstance(operation.expression, ast.BinaryExpression))
            single = shifts[0] if len(shifts) == 1 else None
            return _RuntimeBitRangePopcountInfo(
                selection, complemented,
                shift=single.expression if single is not None else None,
                count_type=single.count_type if single is not None else None,
                outer_complemented=outer_complemented, shifts=shifts,
                operations=operations if interleaved else ())
        # Even complement parity still validates every bound and the source.
        # Other operators cannot turn a nested raw range into this operand.
        selection = self.__runtime_bit_range_source(argument, iterators)
        return None if selection is None else _RuntimeBitRangePopcountInfo(selection, outer_complemented)

    def __runtime_bit_register_expression_shapes(
            self, expression: ast.Expression,
            iterators: set[str] | dict[str, ValueType] | None, *,
            allow_angle_ranges: bool = False) -> dict[int, int]:
        # A structural probe only: do not validate or evaluate bounds/counts.
        # Depth >= 2 identifies a runtime range inside a nested register binary
        # subtree. Fixed subtrees and scalar distance expressions are opaque,
        # so the existing simple-expression paths keep their validation policy.
        active = set(iterators or ()) | {name for scope in self.__loop_bindings for name in scope}
        constants = set(self.__constant_source_types) - active
        shapes = {}
        def scan(node):
            depth = -1
            if isinstance(node, ast.UnaryExpression) and node.op.name == '~':
                depth = scan(node.expression)
            elif isinstance(node, ast.BinaryExpression) and node.op.name in ('&', '|', '^', '<<', '>>'):
                depth = scan(node.lhs)
                if node.op.name in ('&', '|', '^'):
                    depth = max(depth, scan(node.rhs))
                    if depth >= 0:
                        depth += 1
            elif isinstance(node, ast.FunctionCall) and node.name.name in ('rotl', 'rotr') and node.arguments:
                depth = scan(node.arguments[0])
            elif (isinstance(node, ast.IndexExpression)
                  and isinstance(node.index, list) and len(node.index) == 1
                  and isinstance(node.index[0], ast.RangeDefinition)):
                collection = node.collection
                array_element = (isinstance(collection, ast.IndexExpression)
                                 and isinstance(collection.collection, ast.Identifier))
                name = (collection.name if isinstance(collection, ast.Identifier)
                        else collection.collection.name if array_element else None)
                source_type = self.__constant_source_types.get(name) or self.__classical_source_types.get(name)
                is_bit_register = isinstance(collection, ast.Identifier) and isinstance(source_type, ast.BitType)
                is_angle = False
                if allow_angle_ranges and name is not None:
                    if name in active:
                        if isinstance(iterators, dict) and name in iterators:
                            is_angle = not array_element and iterators[name] == ValueType.ANGLE
                        else:
                            binding = next((scope[name] for scope in reversed(self.__loop_bindings) if name in scope), None)
                            is_angle = (not array_element and isinstance(binding, _RuntimeIteratorBinding)
                                        and binding.value_type == ValueType.ANGLE)
                    elif array_element:
                        is_angle = isinstance(source_type, ast.ArrayType) and isinstance(source_type.base_type, ast.AngleType)
                    else:
                        is_angle = isinstance(source_type, ast.AngleType)
                if (is_bit_register or is_angle) and self.__loop_bound_names(node.index[0]) - constants:
                    depth = 0
            shapes[id(node)] = depth
            return depth
        scan(expression)
        return shapes

    def __runtime_bit_register_expression_info(
            self, expression: ast.Expression, shapes: dict[int, int],
            iterators: set[str] | dict[str, ValueType] | None, *,
            allow_angle_ranges: bool = False) -> _RuntimeBitRegisterExpression:
        if shapes.get(id(expression), -1) < 0:
            width = self.__bounded_bit_register_expression_width(expression, iterators)
            return _RuntimeBitRegisterExpression(expression, width, width=width)
        if isinstance(expression, ast.FunctionCall):
            self.__builtin_function_overloads(expression.name.name, len(expression.arguments))
            child = self.__runtime_bit_register_expression_info(
                expression.arguments[0], shapes, iterators, allow_angle_ranges=allow_angle_ranges)
            count_type = self.__runtime_bit_range_popcount_rotation_distance_type(expression, iterators)
            return _RuntimeBitRegisterExpression(expression, child.capacity, children=(child,), count_type=count_type)
        if isinstance(expression, ast.UnaryExpression):
            child = self.__runtime_bit_register_expression_info(
                expression.expression, shapes, iterators, allow_angle_ranges=allow_angle_ranges)
            return _RuntimeBitRegisterExpression(expression, child.capacity, children=(child,))
        if isinstance(expression, ast.BinaryExpression):
            left = self.__runtime_bit_register_expression_info(
                expression.lhs, shapes, iterators, allow_angle_ranges=allow_angle_ranges)
            if expression.op.name in ('<<', '>>'):
                count_type, _ = self.__bit_index_info(expression.rhs, iterators, role='register shift count')
                return _RuntimeBitRegisterExpression(expression, left.capacity, children=(left,), count_type=count_type)
            right = self.__runtime_bit_register_expression_info(
                expression.rhs, shapes, iterators, allow_angle_ranges=allow_angle_ranges)
            # Equality is checked when this node is reached, after both children
            # have completed. min(capacities) bounds a valid result, not its length.
            return _RuntimeBitRegisterExpression(expression, min(left.capacity, right.capacity), children=(left, right))
        if allow_angle_ranges:
            angle_selection = self.__angle_bit_selection(expression, iterators, allow_runtime_range=True)
            if isinstance(angle_selection, _RuntimeAngleBitRangeSelection):
                capacity = min(2 * angle_selection.angle_type.width, self.MAX_RUNTIME_BIT_RANGE_POSITIONS, self.QCX_INT_MAX)
                return _RuntimeBitRegisterExpression(expression, capacity, selection=angle_selection)
        selection = self.__runtime_bit_range_source(expression, iterators)
        if selection is None:
            raise InvalidBitOperandException('Runtime register expression requires a bit-register leaf')
        capacity = min(2 * selection.source_width, self.MAX_RUNTIME_BIT_RANGE_POSITIONS, self.QCX_INT_MAX)
        return _RuntimeBitRegisterExpression(expression, capacity, selection=selection)

    def __runtime_bit_range_popcount_result_info(
            self, info: _RuntimeBitRangePopcountInfo,
            nodes: list[ast.UnaryExpression | ast.BinaryExpression | ast.FunctionCall],
            iterators: set[str] | dict[str, ValueType] | None) -> _RuntimeBitRangePopcountInfo:
        # Binary metadata has already validated the operands and their widths.
        # Keep every wrapper, including even complement groups, and validate
        # distances once in execution order without reading their values.
        return dataclasses.replace(info, result_operations=self.__runtime_bit_range_popcount_operations(nodes, iterators))

    def __runtime_bit_range_popcount_operations(
            self, nodes: list[ast.UnaryExpression | ast.BinaryExpression | ast.FunctionCall],
            iterators: set[str] | dict[str, ValueType] | None) -> tuple[_RuntimeBitRangePopcountOperation, ...]:
        operations = []
        for node in reversed(nodes):
            if isinstance(node, ast.BinaryExpression):
                value_type, _ = self.__bit_index_info(node.rhs, iterators, role='register shift count')
            elif isinstance(node, ast.FunctionCall):
                value_type = self.__runtime_bit_range_popcount_rotation_distance_type(node, iterators)
            else:
                value_type = None
            operations.append(_RuntimeBitRangePopcountOperation(node, value_type))
        return tuple(operations)

    def __runtime_bit_range_popcount_operand_chains_info(
            self, expression: ast.BinaryExpression,
            iterators: set[str] | dict[str, ValueType] | None, *,
            original: ast.Expression, outer_complemented: bool) -> _RuntimeBitRangePopcountInfo | None:
        def peel(node):
            nodes = []
            while True:
                if isinstance(node, ast.UnaryExpression) and node.op.name == '~':
                    nodes.append(node)
                    node = node.expression
                elif isinstance(node, ast.BinaryExpression) and node.op.name in ('<<', '>>'):
                    nodes.append(node)
                    node = node.lhs
                elif isinstance(node, ast.FunctionCall) and node.name.name in ('rotl', 'rotr'):
                    if len(node.arguments) != 2:
                        return node, nodes
                    nodes.append(node)
                    node = node.arguments[0]
                else:
                    return node, nodes
        leaves = [peel(expression.lhs), peel(expression.rhs)]
        # Ordinary direct/complemented ranges and fixed register expressions
        # retain their existing paths. Only a transformed range-shaped leaf
        # needs the extended operand description; runtime status is checked below.
        if not any(isinstance(leaf, ast.IndexExpression) and isinstance(leaf.index, list)
                   and len(leaf.index) == 1 and isinstance(leaf.index[0], ast.RangeDefinition)
                   and any(not isinstance(node, ast.UnaryExpression) for node in nodes)
                   for leaf, nodes in leaves):
            return None
        operands, widths = [], []
        for operand, (leaf, nodes) in zip((expression.lhs, expression.rhs), leaves):
            selection = self.__runtime_bit_range_source(leaf, iterators)
            if selection is None:
                # Validate a fixed LHS before inspecting RHS range bounds or
                # distances. No values, lengths, or bits are evaluated here.
                widths.append(self.__bounded_bit_register_expression_width(operand, iterators))
                operands.append(None)
            else:
                operations = self.__runtime_bit_range_popcount_operations(nodes, iterators)
                operands.append(_RuntimeBitRangePopcountOperand(operand, selection, operations))
                widths.append(None)
        if not any(operand is not None and any(not isinstance(record.expression, ast.UnaryExpression)
                                              for record in operand.operations) for operand in operands):
            return None
        left, right = operands
        # Complements inside operands belong to their complete operation chains;
        # only the binary result's complement parity belongs to this record.
        complements = (_RuntimeBitRangePopcountComplements(original, False, False, outer_complemented)
                       if original is not expression else None)
        if left is not None and right is not None:
            return _RuntimeBitRangePopcountInfo(
                left.selection, False, pair=_RuntimeBitRangePopcountPair(expression, right.selection),
                binary_complements=complements, operand_chains=(left, right))
        selection = left.selection if left is not None else right.selection
        fixed = expression.rhs if left is not None else expression.lhs
        width = widths[1] if left is not None else widths[0]
        return _RuntimeBitRangePopcountInfo(selection, False,
            bitwise=_RuntimeBitRangePopcountBitwise(expression, fixed, width, left is not None),
            binary_complements=complements, operand_chains=(left, right))

    def __runtime_bit_range_popcount_bitwise_info(
            self, expression: ast.BinaryExpression,
            iterators: set[str] | dict[str, ValueType] | None, *,
            original: ast.Expression, outer_complemented: bool) -> _RuntimeBitRangePopcountInfo | None:
        chains = self.__runtime_bit_range_popcount_operand_chains_info(
            expression, iterators, original=original, outer_complemented=outer_complemented)
        if chains is not None:
            return chains
        def peel(node):
            complemented = False
            while isinstance(node, ast.UnaryExpression) and node.op.name == '~':
                complemented = not complemented
                node = node.expression
            return node, complemented
        left, left_complemented = peel(expression.lhs)
        right, right_complemented = peel(expression.rhs)
        selection = self.__runtime_bit_range_source(left, iterators)
        if selection is not None:
            right_selection = self.__runtime_bit_range_source(right, iterators)
            if right_selection is not None:
                # Validate both ranges in written order without inferring a
                # selected width or comparing their runtime lengths here.
                complements = (None if original is expression and left is expression.lhs and right is expression.rhs
                               else _RuntimeBitRangePopcountComplements(
                                   original, left_complemented, right_complemented, outer_complemented))
                return _RuntimeBitRangePopcountInfo(
                    selection, False, pair=_RuntimeBitRangePopcountPair(expression, right_selection),
                    binary_complements=complements)
            range_on_left = True
        else:
            selection = self.__runtime_bit_range_source(right, iterators)
            if selection is None:
                return None
            range_on_left = False
        range_wrapped = left is not expression.lhs if range_on_left else right is not expression.rhs
        extended = original is not expression or range_wrapped
        # Existing fixed-operand complements keep their original reader/path.
        # Extended forms expose the bare fixed operand and record all three
        # parities; the original AST remains available in the complement info.
        fixed = (right if range_on_left else left) if extended else (
            expression.rhs if range_on_left else expression.lhs)
        width = self.__bounded_bit_register_expression_width(fixed, iterators)
        complements = (_RuntimeBitRangePopcountComplements(
            original, left_complemented, right_complemented, outer_complemented) if extended else None)
        return _RuntimeBitRangePopcountInfo(
            selection, False, bitwise=_RuntimeBitRangePopcountBitwise(expression, fixed, width, range_on_left),
            binary_complements=complements)

    def __bounded_bit_register_expression_width(
            self, expression: ast.Expression,
            iterators: set[str] | dict[str, ValueType] | None) -> int:
        limit = min(self.QCX_INT_MAX, self.MAX_RUNTIME_BIT_RANGE_POSITIONS)
        # Bound static selection expansion before generic width inference.
        # This is register storage, so native UINT width is not a restriction.
        def check_selections(node):
            if isinstance(node, ast.IndexExpression) and self.__is_register_bit_selection(node):
                selection = self.__runtime_bit_set_selection(node, iterators)
                if selection is None:
                    self.__static_bit_selection(node, iterators, max_indices=limit)
            elif isinstance(node, ast.UnaryExpression) and node.op.name == '~':
                check_selections(node.expression)
            elif isinstance(node, ast.BinaryExpression) and node.op.name in ('&', '|', '^', '<<', '>>'):
                check_selections(node.lhs)
                if node.op.name in ('&', '|', '^'):
                    check_selections(node.rhs)
            elif isinstance(node, ast.FunctionCall) and node.name.name in ('rotl', 'rotr'):
                self.__builtin_function_overloads(node.name.name, len(node.arguments))
                check_selections(node.arguments[0])
        check_selections(expression)
        width = self.__bit_register_expression_width(expression, iterators)
        if width is None or not 1 <= width <= limit:
            raise InvalidBitOperandException('Runtime register expression requires a bounded known-width register operand')
        if not self.__is_initialization_process:
            active = set(iterators or ()) | {name for scope in self.__loop_bindings for name in scope}
            for name in self.__loop_bound_names(expression) - active:
                if name in self.__constant_source_types:
                    if name not in self.__declared_constant_variables:
                        raise NoVariableNameException(name)
                elif name in self.__classical_source_types:
                    self.__type_of(self.__capitalize_variable_name(name))
        return width

    def __runtime_bit_range_popcount_mixed_info(
            self, argument: ast.Expression,
            iterators: set[str] | dict[str, ValueType] | None) -> _RuntimeBitRangePopcountInfo | None:
        source = argument
        nodes = []
        has_shift = has_rotation = False
        while True:
            if isinstance(source, ast.UnaryExpression) and source.op.name == '~':
                nodes.append(source)
                source = source.expression
            elif isinstance(source, ast.BinaryExpression) and source.op.name in ('<<', '>>'):
                nodes.append(source)
                has_shift = True
                source = source.lhs
            elif isinstance(source, ast.FunctionCall) and source.name.name in ('rotl', 'rotr'):
                self.__builtin_function_overloads(source.name.name, len(source.arguments))
                nodes.append(source)
                has_rotation = True
                source = source.arguments[0]
            else:
                break
        if has_rotation and isinstance(source, ast.BinaryExpression) and source.op.name in ('&', '|', '^'):
            info = self.__runtime_bit_range_popcount_bitwise_info(
                source, iterators, original=source, outer_complemented=False)
            if info is not None:
                return self.__runtime_bit_range_popcount_result_info(info, nodes, iterators)
        if not (has_shift and has_rotation):
            return None
        selection = self.__runtime_bit_range_source(source, iterators)
        if selection is None:
            return None
        operations = []
        # Validate distances in execution order, without evaluating domains or
        # folding complements across shifts that introduce zero-filled bits.
        for node in reversed(nodes):
            if isinstance(node, ast.BinaryExpression):
                value_type, _ = self.__bit_index_info(node.rhs, iterators, role='register shift count')
            elif isinstance(node, ast.FunctionCall):
                value_type = self.__runtime_bit_range_popcount_rotation_distance_type(node, iterators)
            else:
                value_type = None
            operations.append(_RuntimeBitRangePopcountOperation(node, value_type))
        return _RuntimeBitRangePopcountInfo(selection, False, mixed_operations=tuple(operations))

    def __runtime_bit_range_popcount_rotation_distance_type(
            self, node: ast.FunctionCall, iterators: set[str] | dict[str, ValueType] | None) -> ValueType:
        distance = node.arguments[1]
        distance_type, _ = self.__loop_expression_info(distance, iterators)
        self.__resolve_builtin_function(node.name.name, (ValueType.BIT_REGISTER, distance_type))
        self.__validate_bitwise_indices(distance, iterators)
        if not self.__is_initialization_process:
            active = set(iterators or ()) | {name for scope in self.__loop_bindings for name in scope}
            for name in self.__loop_bound_names(distance) - active:
                if name in self.__constant_source_types:
                    if name not in self.__declared_constant_variables:
                        raise NoVariableNameException(name)
                elif name in self.__classical_source_types:
                    self.__type_of(self.__capitalize_variable_name(name))
        return distance_type

    def __runtime_bit_range_popcount_rotation_chain_info(
            self, argument: ast.FunctionCall, outer_complemented: bool,
            iterators: set[str] | dict[str, ValueType] | None) -> _RuntimeBitRangePopcountInfo | None:
        source = argument
        nodes = []
        complemented = False
        total_complemented = outer_complemented
        while True:
            if isinstance(source, ast.FunctionCall) and source.name.name in ('rotl', 'rotr'):
                self.__builtin_function_overloads(source.name.name, len(source.arguments))
                nodes.append(source)
                source = source.arguments[0]
                # Only the final group belongs to the original range itself.
                complemented = False
            elif isinstance(source, ast.UnaryExpression) and source.op.name == '~':
                complemented = not complemented
                total_complemented = not total_complemented
                source = source.expression
            else:
                break
        selection = self.__runtime_bit_range_source(source, iterators)
        if selection is None:
            return None
        rotations = []
        for node in reversed(nodes):
            # Preserve ordinary rotation promotions and source-order visibility
            # without evaluating, converting, or normalizing any distance.
            distance_type = self.__runtime_bit_range_popcount_rotation_distance_type(node, iterators)
            rotations.append(_RuntimeBitRangePopcountRotation(node, distance_type))
        single = rotations[0] if len(rotations) == 1 else None
        return _RuntimeBitRangePopcountInfo(
            selection, complemented, rotation=single.expression if single is not None else None,
            count_type=single.distance_type if single is not None else None,
            outer_complemented=outer_complemented, rotations=tuple(rotations),
            rotation_complemented=total_complemented)

    def __runtime_bit_range_copy_info(
            self, statement: ast.ClassicalAssignment,
            iterators: set[str] | dict[str, ValueType] | None = None
            ) -> _RuntimeBitRangeCopyInfo | _RuntimeBitRangeToRangeCopyInfo | None:
        selection = self.__runtime_bit_range_source(statement.rvalue, iterators)
        if selection is None:
            return None
        operator = statement.op.name
        if operator not in ('=', '&=', '|=', '^='):
            raise UnsupportedOpenQASMError(f'runtime bit-register range RHS assignment operator {operator}')
        target = statement.lvalue
        destination_range = self.__runtime_bit_range_destination(target, iterators)
        if destination_range is not None:
            # A successful destination is distinct, so its count cannot exceed
            # its containing width. Reuse this bounded capacity for both ranges
            # and the source snapshot without guessing either runtime count.
            capacity = min(destination_range.source_width, self.MAX_RUNTIME_BIT_RANGE_POSITIONS)
            if not 1 <= capacity <= self.QCX_INT_MAX:
                raise InvalidBitOperandException('Runtime range copy capacity must fit QCX INT indexing')
            return _RuntimeBitRangeToRangeCopyInfo(destination_range, selection, capacity, operator)
        if not isinstance(target, (ast.Identifier, ast.IndexedIdentifier)):
            raise UnsupportedOpenQASMError('runtime range copy requires a named bit-register destination')
        name = target.name if isinstance(target, ast.Identifier) else target.name.name
        self.__reject_constant_bit_target(target, iterators)
        if ((iterators is not None and name in iterators)
                or any(name in scope for scope in self.__loop_bindings)):
            raise UnsupportedOpenQASMError('assignment to a for-loop iterator')
        if name not in self.__bit_register_source_sizes:
            raise InvalidBitOperandException('Runtime range copy requires a mutable bit-register destination')
        if (not self.__is_initialization_process
                and self.__capitalize_variable_name(name) not in self.__sized_bit_variables):
            raise NoVariableNameException(name)
        destination_width = self.__bit_register_source_sizes[name]
        if not 1 <= destination_width <= self.QCX_INT_MAX:
            raise InvalidBitOperandException('Runtime range copy destination width must fit QCX INT indexing')
        indices = None
        runtime_destination = None
        if isinstance(target, ast.IndexedIdentifier):
            if not self.__is_register_bit_selection(target):
                raise InvalidBitOperandException('Runtime range copy requires a register selection, not a scalar bit')
            runtime_destination = self.__runtime_bit_set_selection(target, iterators, destination=True)
            if runtime_destination is not None:
                # Retain every occurrence and type, including constants in a
                # mixed set. Bounds, arithmetic and duplicates are checked only
                # when the destination is captured on a reached execution path.
                if not 1 <= runtime_destination.width <= self.MAX_RUNTIME_BIT_RANGE_POSITIONS:
                    raise InvalidBitOperandException('Runtime range copy selection exceeds storage limit')
            else:
                # Resolve static destinations without inventing iterator values
                # and bound expansion before materializing their position list.
                destination = self.__static_bit_selection(
                    target, iterators, destination=True,
                    max_indices=min(self.QCX_INT_MAX, self.MAX_RUNTIME_BIT_RANGE_POSITIONS))
                indices = destination.indices
                if not destination.is_register or not 1 <= len(indices) <= self.QCX_INT_MAX:
                    raise InvalidBitOperandException('Runtime range copy selection width must fit QCX INT indexing')
        # Source aliases may repeat. Only the runtime count must equal width;
        # copies need no 2*source_width position capacity or distinctness test.
        return _RuntimeBitRangeCopyInfo(selection, name, destination_width, indices, runtime_destination, operator)

    def __runtime_bit_range_destination(
            self, target: ast.Identifier | ast.IndexedIdentifier,
            iterators: set[str] | dict[str, ValueType] | None = None) -> _RuntimeBitRangeSelection | None:
        if (not isinstance(target, ast.IndexedIdentifier) or len(target.indices) != 1
                or not isinstance(target.indices[0], list) or len(target.indices[0]) != 1
                or not isinstance(target.indices[0][0], ast.RangeDefinition)):
            return None
        name, bounds = target.name.name, target.indices[0][0]
        source_type = self.__constant_source_types.get(name) or self.__classical_source_types.get(name)
        if not isinstance(source_type, ast.BitType):
            return None
        if ((iterators is not None and name in iterators)
                or any(name in scope for scope in self.__loop_bindings)):
            raise UnsupportedOpenQASMError('bit selection of a for-loop iterator')
        self.__reject_constant_bit_target(target, iterators)
        if bounds.start is None or bounds.end is None:
            raise InvalidBitOperandException('Bit-register range requires both bounds')
        # Inspect every operand's type and visibility, but do not evaluate
        # constants in a mixed runtime range, even in an empty enclosing loop.
        start_type, start_runtime = self.__bit_index_info(bounds.start, iterators)
        step_type, step_runtime = ((ValueType.INT, False) if bounds.step is None
                                  else self.__bit_index_info(bounds.step, iterators))
        stop_type, stop_runtime = self.__bit_index_info(bounds.end, iterators)
        if not (start_runtime or step_runtime or stop_runtime):
            return None
        if source_type.size is None:
            raise InvalidBitOperandException('Runtime range assignment requires a bit-register destination')
        if (not self.__is_initialization_process
                and self.__capitalize_variable_name(name) not in self.__sized_bit_variables):
            raise NoVariableNameException(name)
        size = self.__bit_register_source_sizes[name]
        if not 1 <= size <= self.QCX_INT_MAX:
            raise InvalidBitOperandException('Runtime range destination width must fit QCX INT indexing')
        return _RuntimeBitRangeSelection(name, size, bounds, start_type, step_type, stop_type)

    def __runtime_bit_range_assignment_info(
            self, statement: ast.ClassicalAssignment,
            iterators: set[str] | dict[str, ValueType] | None = None
            ) -> _RuntimeBitRangeAssignmentInfo | _RuntimeBitRangeShiftAssignmentInfo | None:
        selection = self.__runtime_bit_range_destination(statement.lvalue, iterators)
        if selection is None:
            return None
        operator = statement.op.name
        if operator in ('<<=', '>>='):
            count_type, _ = self.__bit_index_info(statement.rvalue, iterators, role='range shift count')
            # Successful destinations are distinct, so at most source_width
            # entries can be written. Unlike iteration, shifts need no extra
            # room for negative aliases. Larger sources may still use short
            # selections when the configured position limit is smaller.
            capacity = min(selection.source_width, self.MAX_RUNTIME_BIT_RANGE_POSITIONS)
            if not 1 <= capacity <= self.QCX_INT_MAX:
                raise InvalidBitOperandException('Runtime range shift capacity must fit QCX INT indexing')
            return _RuntimeBitRangeShiftAssignmentInfo(selection, operator, count_type, capacity)
        if operator not in ('=', '&=', '|=', '^='):
            raise UnsupportedOpenQASMError(f'runtime bit-register range assignment operator {operator}')
        width = self.__bit_register_expression_width(statement.rvalue, iterators)
        if width is None or not 1 <= width <= self.QCX_INT_MAX:
            raise InvalidBitOperandException('A runtime range assignment requires a known-width bit register RHS')
        if not self.__is_initialization_process:
            active = set(iterators or ()) | {name for scope in self.__loop_bindings for name in scope}
            for bound in self.__loop_bound_names(statement.rvalue) - active:
                if bound in self.__constant_source_types:
                    if bound not in self.__declared_constant_variables:
                        raise NoVariableNameException(bound)
                elif bound in self.__classical_source_types:
                    self.__type_of(self.__capitalize_variable_name(bound))
        return _RuntimeBitRangeAssignmentInfo(selection, width, operator)

    def __runtime_bit_set_assignment_info(
            self, statement: ast.ClassicalAssignment,
            iterators: set[str] | dict[str, ValueType] | None = None) -> _RuntimeBitSetAssignmentInfo | None:
        selection = self.__runtime_bit_set_selection(statement.lvalue, iterators, destination=True)
        if selection is None:
            return None
        operator = statement.op.name
        if operator not in ('=', '&=', '|=', '^=', '<<=', '>>='):
            raise UnsupportedOpenQASMError(f'runtime bit-register set assignment operator {operator}')
        if operator in ('<<=', '>>='):
            count_type, _ = self.__loop_expression_info(statement.rvalue, iterators)
            if count_type not in (ValueType.INT, ValueType.UINT):
                raise InvalidBitOperandException('A register selection shift count must be a scalar integer')
            self.__validate_bitwise_indices(statement.rvalue, iterators)
            return _RuntimeBitSetAssignmentInfo(selection, operator, count_type)
        width = self.__bit_register_expression_width(statement.rvalue, iterators)
        if width != selection.width:
            raise InvalidBitOperandException('A runtime bit selection assignment requires a matching-width bit register')
        return _RuntimeBitSetAssignmentInfo(selection, operator)

    def __validate_angle_bit_assignment(
            self, statement: ast.ClassicalAssignment,
            iterators: set[str] | dict[str, ValueType] | None = None) -> bool:
        return self.__angle_bit_assignment_info(statement, iterators) is not None

    def __angle_bit_assignment_info(
            self, statement: ast.ClassicalAssignment,
            iterators: set[str] | dict[str, ValueType] | None = None) -> _AngleBitAssignmentInfo | None:
        selection = self.__angle_bit_selection(statement.lvalue, iterators, destination=True)
        if selection is None:
            return None
        operator = statement.op.name
        if operator not in ('=', '&=', '|=', '^=', '<<=', '>>='):
            raise UnsupportedOpenQASMError(f'angle bit selection assignment operator {operator}')
        if operator in ('<<=', '>>='):
            # Validate count types and nested accesses without evaluating an
            # index, count, arithmetic domain, or the destination's value.
            count_type, _ = self.__loop_expression_info(statement.rvalue, iterators)
            if count_type not in (ValueType.INT, ValueType.UINT):
                raise InvalidBitOperandException('An angle selection shift count must be a scalar integer')
            self.__validate_bitwise_indices(statement.rvalue, iterators)
            return _AngleBitAssignmentInfo(selection, operator, count_type)
        width = self.__bit_register_expression_width(statement.rvalue, iterators)
        if width != selection.width:
            raise InvalidBitOperandException('An angle bit selection assignment requires a matching-width bit register')
        return _AngleBitAssignmentInfo(selection, operator)

    @staticmethod
    def __angle_assignment_expression(
            statement: ast.ClassicalAssignment, lhs: ast.Expression | None = None) -> ast.Expression:
        if lhs is None and isinstance(statement.lvalue, ast.IndexedIdentifier):
            raise UnsupportedOpenQASMError('angle indexing')
        if statement.op.name == '=':
            return statement.rvalue
        if statement.op.name not in ('+=', '-=', '*=', '/=', '&=', '|=', '^=', '<<=', '>>='):
            raise UnsupportedOpenQASMError(f'angle assignment operator {statement.op.name}')
        # Preserve the binary operation's promoted width before converting
        # back to the destination. Pre-narrowing the RHS could lose a borrow.
        # Multiplicative typing also checks UINT factor widths. Angle/angle
        # produces UINT, which the destination conversion rejects.
        return ast.BinaryExpression(ast.BinaryOperator[statement.op.name[:-1]],
                                    statement.lvalue if lhs is None else lhs, statement.rvalue)

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

    def __uint_bit_range_expression_info(
            self, expression: ast.Expression,
            iterators: set[str] | dict[str, ValueType] | None
            ) -> tuple[_RuntimeBitRangeSelection, tuple[_BitRangeCastOperation, ...]] | None:
        # Peel only width-preserving complement/shift nodes. Their range leaf
        # must be recognized before any scalar-count validation is performed;
        # ordinary scalar UINT casts retain their existing typing path.
        nodes = []
        while True:
            if isinstance(expression, ast.UnaryExpression) and expression.op.name == '~':
                nodes.append(expression)
                expression = expression.expression
            elif isinstance(expression, ast.BinaryExpression) and expression.op.name in ('<<', '>>'):
                nodes.append(expression)
                expression = expression.lhs
            else:
                break
        selection = self.__runtime_bit_range_source(expression, iterators)
        if selection is None:
            return None
        operations = []
        for node in reversed(nodes):
            count_type = None
            if isinstance(node, ast.BinaryExpression):
                count_type, _ = self.__loop_expression_info(node.rhs, iterators)
                if count_type not in (ValueType.INT, ValueType.UINT):
                    raise InvalidBitOperandException('A register shift count must be a scalar integer')
                self.__validate_bitwise_indices(node.rhs, iterators)
            operations.append(_BitRangeCastOperation(node, count_type))
        return selection, tuple(operations)

    def __contains_uint_bit_range(
            self, expression: ast.Expression,
            iterators: set[str] | dict[str, ValueType] | None, *,
            allow_runtime_angle_ranges: bool = False) -> bool:
        # Follow register operands only. Ranges inside scalar counts or nested
        # casts have their own contexts and cannot make this a register tree.
        if isinstance(expression, ast.UnaryExpression) and expression.op.name == '~':
            return self.__contains_uint_bit_range(
                expression.expression, iterators, allow_runtime_angle_ranges=allow_runtime_angle_ranges)
        if (isinstance(expression, ast.FunctionCall) and expression.name.name in ('rotl', 'rotr')
                and expression.arguments):
            return self.__contains_uint_bit_range(
                expression.arguments[0], iterators, allow_runtime_angle_ranges=allow_runtime_angle_ranges)
        if isinstance(expression, ast.BinaryExpression):
            if expression.op.name in ('<<', '>>'):
                return self.__contains_uint_bit_range(
                    expression.lhs, iterators, allow_runtime_angle_ranges=allow_runtime_angle_ranges)
            if expression.op.name in ('&', '|', '^'):
                return (self.__contains_uint_bit_range(
                            expression.lhs, iterators, allow_runtime_angle_ranges=allow_runtime_angle_ranges)
                        or self.__contains_uint_bit_range(
                            expression.rhs, iterators, allow_runtime_angle_ranges=allow_runtime_angle_ranges))
        if allow_runtime_angle_ranges:
            selection = self.__angle_bit_selection(expression, iterators, allow_runtime_range=True)
            if isinstance(selection, _RuntimeAngleBitRangeSelection):
                return True
        return self.__runtime_bit_range_source(expression, iterators) is not None

    def __uint_bit_range_tree_info(
            self, expression: ast.Expression, width: int,
            iterators: set[str] | dict[str, ValueType] | None, *,
            allow_runtime_angle_selections: bool = False,
            allow_runtime_angle_ranges: bool = False) -> _BitRangeCastExpression:
        if isinstance(expression, ast.FunctionCall) and expression.name.name in ('rotl', 'rotr'):
            # The required cast width supplies the register context; generic
            # function inference cannot infer a runtime range's width. Keep
            # the original distance node and defer its value/conversion checks.
            self.__builtin_function_overloads(expression.name.name, len(expression.arguments))
            argument, distance = expression.arguments
            child = self.__uint_bit_range_tree_info(
                argument, width, iterators, allow_runtime_angle_selections=allow_runtime_angle_selections,
                allow_runtime_angle_ranges=allow_runtime_angle_ranges)
            distance_type, _ = self.__loop_expression_info(distance, iterators)
            self.__resolve_builtin_function(expression.name.name, (ValueType.BIT_REGISTER, distance_type))
            self.__validate_bitwise_indices(distance, iterators)
            return _BitRangeCastExpression(expression, width, (child,), count_type=distance_type)
        if isinstance(expression, ast.UnaryExpression) and expression.op.name == '~':
            child = self.__uint_bit_range_tree_info(
                expression.expression, width, iterators, allow_runtime_angle_selections=allow_runtime_angle_selections,
                allow_runtime_angle_ranges=allow_runtime_angle_ranges)
            return _BitRangeCastExpression(expression, width, (child,))
        if isinstance(expression, ast.BinaryExpression) and expression.op.name in ('&', '|', '^', '<<', '>>'):
            lhs = self.__uint_bit_range_tree_info(
                expression.lhs, width, iterators, allow_runtime_angle_selections=allow_runtime_angle_selections,
                allow_runtime_angle_ranges=allow_runtime_angle_ranges)
            if expression.op.name in ('<<', '>>'):
                count_type, _ = self.__loop_expression_info(expression.rhs, iterators)
                if count_type not in (ValueType.INT, ValueType.UINT):
                    raise InvalidBitOperandException('A register shift count must be a scalar integer')
                self.__validate_bitwise_indices(expression.rhs, iterators)
                return _BitRangeCastExpression(expression, width, (lhs,), count_type=count_type)
            rhs = self.__uint_bit_range_tree_info(
                expression.rhs, width, iterators, allow_runtime_angle_selections=allow_runtime_angle_selections,
                allow_runtime_angle_ranges=allow_runtime_angle_ranges)
            return _BitRangeCastExpression(expression, width, (lhs, rhs))
        if allow_runtime_angle_ranges:
            angle_selection = self.__angle_bit_selection(expression, iterators, allow_runtime_range=True)
            if isinstance(angle_selection, _RuntimeAngleBitRangeSelection):
                return _BitRangeCastExpression(expression, width, selection=angle_selection)
        selection = self.__runtime_bit_range_source(expression, iterators)
        if selection is not None:
            return _BitRangeCastExpression(expression, width, selection=selection)
        # Bound static expansion before asking existing register inference to
        # validate fixed leaves. No runtime values or register bits are read.
        self.__validate_uint_bit_register_source(
            expression, iterators, allow_runtime_angle_selections=allow_runtime_angle_selections)
        source_width = self.__bit_register_expression_width(expression, iterators)
        if source_width is None:
            raise InvalidBitOperandException('A bitwise UINT cast requires register operands')
        if source_width != width:
            raise InvalidBitOperandException(
                f'UINT/bit-register cast requires equal widths, got {source_width} and {width}')
        return _BitRangeCastExpression(expression, width)

    def __contains_angle_bit_range(
            self, expression: ast.Expression,
            iterators: set[str] | dict[str, ValueType] | None = None) -> bool:
        # Follow register operands only; scalar counts and nested casts have
        # their own contexts and do not determine this cast's register width.
        if isinstance(expression, ast.UnaryExpression) and expression.op.name == '~':
            return self.__contains_angle_bit_range(expression.expression, iterators)
        if (isinstance(expression, ast.FunctionCall) and expression.name.name in ('rotl', 'rotr')
                and expression.arguments):
            return self.__contains_angle_bit_range(expression.arguments[0], iterators)
        if isinstance(expression, ast.BinaryExpression):
            if expression.op.name in ('<<', '>>'):
                return self.__contains_angle_bit_range(expression.lhs, iterators)
            if expression.op.name in ('&', '|', '^'):
                return (self.__contains_angle_bit_range(expression.lhs, iterators)
                        or self.__contains_angle_bit_range(expression.rhs, iterators))
        return isinstance(self.__angle_bit_selection(expression, iterators, allow_runtime_range=True),
                          _RuntimeAngleBitRangeSelection)

    def __contains_runtime_angle_bit_selection(
            self, expression: ast.Expression,
            iterators: set[str] | dict[str, ValueType] | None = None) -> bool:
        # Scalar counts and nested casts have their own expression contexts.
        if isinstance(expression, ast.UnaryExpression) and expression.op.name == '~':
            return self.__contains_runtime_angle_bit_selection(expression.expression, iterators)
        if (isinstance(expression, ast.FunctionCall) and expression.name.name in ('rotl', 'rotr')
                and expression.arguments):
            return self.__contains_runtime_angle_bit_selection(expression.arguments[0], iterators)
        if isinstance(expression, ast.BinaryExpression):
            if expression.op.name in ('<<', '>>'):
                return self.__contains_runtime_angle_bit_selection(expression.lhs, iterators)
            if expression.op.name in ('&', '|', '^'):
                return (self.__contains_runtime_angle_bit_selection(expression.lhs, iterators)
                        or self.__contains_runtime_angle_bit_selection(expression.rhs, iterators))
        selection = self.__angle_bit_selection(expression, iterators, allow_runtime_range=True)
        return isinstance(selection, (_RuntimeAngleBitSelection, _RuntimeAngleBitSetSelection))

    def __runtime_angle_selection_uint_cast_info(
            self, expression: ast.Cast,
            iterators: set[str] | dict[str, ValueType] | None = None) -> _UIntBitCastInfo | None:
        if (not isinstance(expression.type, ast.UintType)
                or not self.__contains_runtime_angle_bit_selection(expression.argument, iterators)):
            return None
        return self.__uint_bit_cast_info(expression, iterators, allow_runtime_angle_selections=True,
                                         allow_runtime_angle_ranges=True)

    def __runtime_angle_range_uint_cast_info(
            self, expression: ast.Cast,
            iterators: set[str] | dict[str, ValueType] | None = None) -> _UIntBitCastInfo | None:
        if (not isinstance(expression.type, ast.UintType)
                or not self.__contains_angle_bit_range(expression.argument, iterators)):
            return None
        return self.__uint_bit_cast_info(expression, iterators, allow_runtime_angle_selections=True,
                                         allow_runtime_angle_ranges=True)

    def __uint_bit_cast_info(
            self, expression: ast.Cast,
            iterators: set[str] | dict[str, ValueType] | None = None, *,
            allow_runtime_angle_selections: bool = False,
            allow_runtime_angle_ranges: bool = False) -> _UIntBitCastInfo | None:
        # Infer bit patterns and UINT widths without executing the operand or
        # consulting mutable expression-result caches. Numeric casts and the
        # existing angle/bit conversions retain their separate paths.
        if not allow_runtime_angle_selections and not allow_runtime_angle_ranges:
            angle_selection = self.__runtime_angle_selection_uint_cast_info(expression, iterators)
            if angle_selection is not None:
                return angle_selection
            angle_range = self.__runtime_angle_range_uint_cast_info(expression, iterators)
            if angle_range is not None:
                return angle_range
        range_selection = None
        range_operations = ()
        range_tree = None
        if isinstance(expression.type, ast.UintType):
            range_info = self.__uint_bit_range_expression_info(expression.argument, iterators)
            if range_info is None and self.__contains_uint_bit_range(
                    expression.argument, iterators, allow_runtime_angle_ranges=allow_runtime_angle_ranges):
                uint_type = self.__unsigned_integer_type(expression.type, iterators)
                range_tree = self.__uint_bit_range_tree_info(
                    expression.argument, uint_type.width, iterators,
                    allow_runtime_angle_selections=allow_runtime_angle_selections,
                    allow_runtime_angle_ranges=allow_runtime_angle_ranges)
                source_width = None
            elif range_info is None:
                # Bound static selection expansion before generic width inference
                # can materialize a range that cannot fit any native UINT cast.
                self.__validate_uint_bit_register_source(
                    expression.argument, iterators, allow_runtime_angle_selections=allow_runtime_angle_selections)
                source_width = self.__bit_register_expression_width(expression.argument, iterators)
                if source_width is None:
                    return None
            else:
                range_selection, range_operations = range_info
                # Width-preserving operations do not determine a range's count.
                # Preserve its operands until capture checks the target width.
                source_width = None
            if range_tree is None:
                uint_type = self.__unsigned_integer_type(expression.type, iterators)
            target_width = uint_type.width
            runtime = (True if range_selection is not None or range_tree is not None
                       else self.__register_expression_runtime(expression.argument, iterators))
            result_type = ValueType.UINT
        elif isinstance(expression.type, ast.BitType) and expression.type.size is not None:
            source_type, uint_type, runtime = self.__integer_operand_info(expression.argument, iterators)
            if source_type != ValueType.UINT:
                return None
            source_width = uint_type.width
            target_width = self.__unsigned_integer_type(ast.UintType(expression.type.size), iterators).width
            self.__validate_bitwise_indices(expression.argument, iterators)
            result_type = ValueType.BIT_REGISTER
        else:
            return None
        if source_width is not None and source_width != target_width:
            raise InvalidBitOperandException(
                f'UINT/bit-register cast requires equal widths, got {source_width} and {target_width}')
        # Register width inference and scalar type lookup alone do not always
        # enforce source-order visibility. Check names without reading values.
        if not self.__is_initialization_process:
            active = set(iterators or ()) | {name for scope in self.__loop_bindings for name in scope}
            for name in self.__loop_bound_names(expression.argument) - active:
                if name in self.__constant_source_types:
                    if name not in self.__declared_constant_variables:
                        raise NoVariableNameException(name)
                elif name in self.__classical_source_types:
                    self.__type_of(self.__capitalize_variable_name(name))
        return _UIntBitCastInfo(result_type, uint_type, runtime, range_selection, range_operations, range_tree)

    def __validate_uint_bit_register_source(
            self, expression: ast.Expression,
            iterators: set[str] | dict[str, ValueType] | None, *,
            allow_runtime_angle_selections: bool = False) -> None:
        # Runtime bit-register sets have a fixed width; their index values are
        # captured by the existing snapshot reader. Supported runtime range
        # chains are checked separately; other nested ranges and dynamic
        # angle-bit selections remain outside the UINT subset. Angle casts
        # reuse this analyzer with their existing fixed-width selection support.
        if isinstance(expression, ast.IndexExpression):
            selection = self.__angle_bit_selection(expression, iterators)
            if selection is not None:
                if (not allow_runtime_angle_selections
                        and isinstance(selection, (_RuntimeAngleBitSelection, _RuntimeAngleBitSetSelection))):
                    raise UnsupportedOpenQASMError('runtime-selected UINT/bit-register cast operand')
            elif self.__is_register_bit_selection(expression):
                runtime_selection = self.__runtime_bit_set_selection(expression, iterators)
                if runtime_selection is not None:
                    if runtime_selection.width > self.QCX_UINT_WIDTH:
                        raise InvalidBitOperandException('Runtime set UINT cast width exceeds native QCX UINT width')
                else:
                    self.__static_bit_selection(expression, iterators, max_indices=self.QCX_UINT_WIDTH)
        elif isinstance(expression, ast.UnaryExpression) and expression.op.name == '~':
            self.__validate_uint_bit_register_source(
                expression.expression, iterators, allow_runtime_angle_selections=allow_runtime_angle_selections)
        elif isinstance(expression, ast.BinaryExpression) and expression.op.name in ('&', '|', '^', '<<', '>>'):
            self.__validate_uint_bit_register_source(
                expression.lhs, iterators, allow_runtime_angle_selections=allow_runtime_angle_selections)
            if expression.op.name not in ('<<', '>>'):
                self.__validate_uint_bit_register_source(
                    expression.rhs, iterators, allow_runtime_angle_selections=allow_runtime_angle_selections)
        elif (isinstance(expression, ast.FunctionCall) and expression.name.name in ('rotl', 'rotr')
              and expression.arguments):
            self.__validate_uint_bit_register_source(
                expression.arguments[0], iterators, allow_runtime_angle_selections=allow_runtime_angle_selections)
        elif isinstance(expression, ast.Cast) and isinstance(expression.type, (ast.BitType, ast.AngleType)):
            if self.__runtime_range_angle_cast_info(expression, iterators) is not None:
                return  # This scalar angle cast owns its range validation.
            if isinstance(expression.type, ast.AngleType):
                # Bound static selections before generic width inference, while
                # allowing the angle cast's own runtime selection operands.
                self.__validate_uint_bit_register_source(
                    expression.argument, iterators, allow_runtime_angle_selections=True)
                info = self.__angle_bit_cast_info(expression, iterators)
                if info is not None and info[0] == ValueType.ANGLE:
                    return  # This scalar angle cast validates its register operand.
            self.__validate_uint_bit_register_source(
                expression.argument, iterators, allow_runtime_angle_selections=allow_runtime_angle_selections)

    def __visit_constant_uint_bit_cast(self, expression: ast.Cast, info: _UIntBitCastInfo) -> None:
        if info.runtime:
            raise NoConstantExpressionException
        previous = (self.__expression_kind, self.__value, self.__value_type,
                    self.__value_kind, self.__evaluate_constant)
        previous_used = self.__used_temporary_variables.copy()
        try:
            if info.result_type == ValueType.UINT:
                value = self.__constant_bit_register_value(expression.argument, info.width)
            else:
                self.visit(expression.argument)
                if self.__value_kind != ValueKind.LITERAL or self.__value_type != ValueType.UINT:
                    raise NoConstantExpressionException
                value = int(self.__value)
            self.__value = value if self.__evaluate_constant else 0
            self.__value_type, self.__value_kind = info.result_type, ValueKind.LITERAL
            if info.result_type == ValueType.UINT:
                self.__uint_expression_types[id(expression)] = (expression, info.uint_type)
        except Exception:
            (self.__expression_kind, self.__value, self.__value_type,
             self.__value_kind, self.__evaluate_constant) = previous
            raise
        finally:
            for temporary in self.__used_temporary_variables - previous_used:
                self.__release_temporary_variable(temporary)

    def __check_uint_bit_cast_output(self, extra: int = 0) -> None:
        if len(self.__qcx_lines) + extra > self.MAX_LOOP_OUTPUT_LINES:
            raise InvalidBitOperandException('UINT/bit-register cast exceeds QCX output limit')

    def __pack_bit_register_cast(
            self, argument: ast.Expression, info: _UIntBitCastInfo) -> tuple[str | int, ValueKind]:
        # The scalar consumer owns the result and error cleanup. Packing itself
        # does not register UINT or ANGLE type metadata for the original cast.
        self.__check_uint_bit_cast_output(1)
        if info.range_tree is not None:
            source = self.__runtime_bit_range_tree_value(info.range_tree)
        elif info.range_selection is not None:
            source = self.__runtime_bit_range_cast_value(argument, info)
        else:
            source = self.__bit_register_expression_value(argument)
        with source as values:
            literal = sum(value << index for index, value in enumerate(values) if isinstance(value, int))
            if all(isinstance(value, int) for value in values):
                result, kind = literal, ValueKind.LITERAL
            else:
                result = self.__add_new_temporary_variable(ValueType.UINT, info.uint_type)
                self.__qcx_lines.append(f'LET {result} := {literal}')
                scratch = None
                for index, value in enumerate(values):
                    if isinstance(value, int):
                        continue
                    if scratch is None:
                        scratch = self.__add_new_temporary_variable(ValueType.UINT)
                    self.__qcx_lines.append(f'LET {scratch} := :UINT:{value}')
                    if index:
                        self.__qcx_lines.append(f'LET {scratch} <<= {index}')
                    self.__qcx_lines.append(f'LET {result} |= {scratch}')
                kind = ValueKind.TEMPORARY
        self.__check_uint_bit_cast_output()
        return result, kind

    def __visit_runtime_bit_to_uint_cast(self, expression: ast.Cast, info: _UIntBitCastInfo) -> None:
        previous = (self.__expression_kind, self.__value, self.__value_type,
                    self.__value_kind, self.__evaluate_constant)
        previous_used = self.__used_temporary_variables.copy()
        keep = None
        try:
            self.__expression_kind = ExpressionKind.ARITHMETIC
            result, kind = self.__pack_bit_register_cast(expression.argument, info)
            self.__expression_kind = previous[0]
            self.__value, self.__value_type, self.__value_kind = result, ValueType.UINT, kind
            self.__uint_expression_types[id(expression)] = (expression, info.uint_type)
            keep = result if kind == ValueKind.TEMPORARY else None
        except Exception:
            (self.__expression_kind, self.__value, self.__value_type,
             self.__value_kind, self.__evaluate_constant) = previous
            raise
        finally:
            for temporary in self.__used_temporary_variables - previous_used:
                if temporary != keep:
                    self.__release_temporary_variable(temporary)

    @contextlib.contextmanager
    def __uint_to_bit_register_value(
            self, expression: ast.Cast, info: _UIntBitCastInfo) -> Iterator[tuple[str | int, ...]]:
        self.__check_uint_bit_cast_output(1)
        previous = (self.__expression_kind, self.__value, self.__value_type,
                    self.__value_kind, self.__evaluate_constant)
        previous_used = self.__used_temporary_variables.copy()
        owned = None
        try:
            self.__expression_kind = ExpressionKind.ARITHMETIC
            self.visit(expression.argument)
            if self.__value_type != ValueType.UINT or self.__value_kind not in (
                    ValueKind.LITERAL, ValueKind.LVALUE, ValueKind.TEMPORARY):
                raise UninitializedValueException
            if self.__value_kind == ValueKind.LITERAL:
                bits = int(self.__value)
                values = tuple((bits >> index) & 1 for index in range(info.width))
            else:
                snapshot = self.__add_new_temporary_variable(ValueType.UINT, info.uint_type)
                self.__qcx_lines.append(f'LET {snapshot} := {self.__value}')
                scratch = self.__add_new_temporary_variable(ValueType.UINT)
                values = []
                for index in range(info.width):
                    bit = self.__add_new_temporary_variable(ValueType.BIT)
                    self.__qcx_lines.append(f'LET {scratch} := {snapshot}')
                    if index:
                        self.__qcx_lines.append(f'LET {scratch} >>= {index}')
                    self.__qcx_lines.extend([f'LET {scratch} &= 1', f'LET {bit} := :INT:{scratch}'])
                    values.append(bit)
                values = tuple(values)
            self.__check_uint_bit_cast_output()
            owned = self.__used_temporary_variables - previous_used
            yield values
        finally:
            (self.__expression_kind, self.__value, self.__value_type,
             self.__value_kind, self.__evaluate_constant) = previous
            # A consumer may allocate its own result during yield. Its storage
            # is not owned by this operand snapshot or its extracted bits.
            for temporary in owned if owned is not None else self.__used_temporary_variables - previous_used:
                self.__release_temporary_variable(temporary)

    def __runtime_range_angle_cast_info(
            self, expression: ast.Cast,
            iterators: set[str] | dict[str, ValueType] | None = None, *,
            allow_runtime_angle_ranges: bool = True) -> _RuntimeRangeAngleCastInfo | None:
        if (not isinstance(expression.type, ast.AngleType)
                or not self.__contains_uint_bit_range(
                    expression.argument, iterators, allow_runtime_angle_ranges=allow_runtime_angle_ranges)):
            return None  # Existing fixed-register and numeric casts keep their paths.
        angle_type = self.__angle_type(expression.type, iterators)
        # Build only a metadata view; keep the original cast and source nodes.
        # The effective angle width is already validated, so do not evaluate
        # its original width expression again in the UINT metadata analyzer.
        size = ast.IntegerLiteral(angle_type.width) if angle_type.is_sized else None
        packing = self.__uint_bit_cast_info(
            ast.Cast(ast.UintType(size), expression.argument), iterators, allow_runtime_angle_selections=True,
            allow_runtime_angle_ranges=allow_runtime_angle_ranges)
        if packing is None:
            return None
        return _RuntimeRangeAngleCastInfo(expression, angle_type, packing)

    def __angle_bit_cast_info(
            self, expression: ast.Cast,
            iterators: set[str] | dict[str, ValueType] | None = None
            ) -> tuple[ValueType, int, bool] | None:
        # Width/type inference must not read bits, evaluate angle operands,
        # or emit instructions, including in skipped and empty bodies.
        runtime_range = self.__runtime_range_angle_cast_info(expression, iterators)
        if runtime_range is not None:
            return runtime_range.result_type, runtime_range.width, runtime_range.runtime
        if isinstance(expression.type, ast.AngleType):
            source_width = self.__bit_register_expression_width(expression.argument, iterators)
            if source_width is None:
                return None
            target_width = self.__angle_type(expression.type, iterators).width
            runtime = self.__register_expression_runtime(expression.argument, iterators)
            result_type = ValueType.ANGLE
        elif isinstance(expression.type, ast.BitType) and expression.type.size is not None:
            source_type, angle_type, runtime = self.__angle_operand_info(expression.argument, iterators)
            if source_type != ValueType.ANGLE:
                return None
            # A bit register that holds an angle pattern has the same width
            # limits and constant-width requirements as the source angle.
            target_width = self.__angle_type(ast.AngleType(expression.type.size), iterators).width
            source_width = angle_type.width
            result_type = ValueType.BIT_REGISTER
        else:
            return None
        if source_width != target_width:
            raise InvalidBitOperandException(
                f'Angle/bit-register cast requires equal widths, got {source_width} and {target_width}')
        return result_type, target_width, runtime

    def __visit_constant_angle_bit_cast(
            self, expression: ast.Cast, info: tuple[ValueType, int, bool]) -> None:
        result_type, width, runtime = info
        if runtime:
            raise NoConstantExpressionException
        previous = (self.__expression_kind, self.__value, self.__value_type,
                    self.__value_kind, self.__evaluate_constant)
        previous_used = self.__used_temporary_variables.copy()
        try:
            if result_type == ValueType.ANGLE:
                value = self.__constant_bit_register_value(expression.argument, width)
                angle_type = self.__angle_type(expression.type)
                self.__angle_expression_types[id(expression)] = (expression, angle_type)
            else:
                self.visit(expression.argument)
                if self.__value_kind != ValueKind.LITERAL or self.__value_type != ValueType.ANGLE:
                    raise NoConstantExpressionException
                value = int(self.__value)
            self.__value = value if self.__evaluate_constant else 0
            self.__value_type, self.__value_kind = result_type, ValueKind.LITERAL
        except Exception:
            (self.__expression_kind, self.__value, self.__value_type,
             self.__value_kind, self.__evaluate_constant) = previous
            raise
        finally:
            for temporary in self.__used_temporary_variables - previous_used:
                self.__release_temporary_variable(temporary)

    def __visit_runtime_range_angle_cast(self, info: _RuntimeRangeAngleCastInfo) -> None:
        previous = (self.__expression_kind, self.__value, self.__value_type,
                    self.__value_kind, self.__evaluate_constant)
        previous_used = self.__used_temporary_variables.copy()
        keep = None
        try:
            self.__expression_kind = ExpressionKind.ARITHMETIC
            result, kind = self.__pack_bit_register_cast(info.expression.argument, info.packing)
            self.__angle_expression_types[id(info.expression)] = (info.expression, info.angle_type)
            self.__expression_kind = previous[0]
            self.__value, self.__value_type, self.__value_kind = result, ValueType.ANGLE, kind
            keep = result if kind == ValueKind.TEMPORARY else None
        except Exception:
            (self.__expression_kind, self.__value, self.__value_type,
             self.__value_kind, self.__evaluate_constant) = previous
            raise
        finally:
            for temporary in self.__used_temporary_variables - previous_used:
                if temporary != keep:
                    self.__release_temporary_variable(temporary)

    def __visit_runtime_bit_to_angle_cast(self, expression: ast.Cast) -> None:
        runtime_range = self.__runtime_range_angle_cast_info(expression)
        if runtime_range is not None:
            self.__visit_runtime_range_angle_cast(runtime_range)
            return
        previous = (self.__expression_kind, self.__value, self.__value_type, self.__value_kind)
        previous_used = self.__used_temporary_variables.copy()
        keep = None
        try:
            self.__expression_kind = ExpressionKind.ARITHMETIC
            angle_type = self.__angle_type(expression.type)
            with self.__bit_register_expression_value(expression.argument) as values:
                result = self.__add_new_temporary_variable(
                    ValueType.UINT, _UnsignedIntegerType(angle_type.width, True))
                literal = sum(value << index for index, value in enumerate(values) if isinstance(value, int))
                self.__qcx_lines.append(f'LET {result} := {literal}')
                scratch = None
                for index, value in enumerate(values):
                    if isinstance(value, int):
                        continue
                    if scratch is None:
                        scratch = self.__add_new_temporary_variable(ValueType.UINT)
                    self.__qcx_lines.append(f'LET {scratch} := :UINT:{value}')
                    if index:
                        self.__qcx_lines.append(f'LET {scratch} <<= {index}')
                    self.__qcx_lines.append(f'LET {result} |= {scratch}')
            self.__angle_expression_types[id(expression)] = (expression, angle_type)
            self.__expression_kind = previous[0]
            self.__value, self.__value_type, self.__value_kind = result, ValueType.ANGLE, ValueKind.TEMPORARY
            keep = result
        except Exception:
            self.__expression_kind, self.__value, self.__value_type, self.__value_kind = previous
            raise
        finally:
            for temporary in self.__used_temporary_variables - previous_used:
                if temporary != keep:
                    self.__release_temporary_variable(temporary)

    @contextlib.contextmanager
    def __angle_to_bit_register_value(self, expression: ast.Cast) -> Iterator[tuple[str | int, ...]]:
        _, angle_type, _ = self.__angle_operand_info(expression.argument)
        with self.__angle_bits_value(expression.argument, angle_type, range(angle_type.width)) as values:
            yield values

    @contextlib.contextmanager
    def __runtime_angle_bit_value(
            self, selection: _RuntimeAngleBitSelection | _RuntimeAngleBitSetSelection, *,
            captured: tuple[str, tuple[int | str, ...]] | None = None) -> Iterator[tuple[str, ...]]:
        previous = (self.__expression_kind, self.__value, self.__value_type, self.__value_kind)
        previous_used = self.__used_temporary_variables.copy()
        owned = None
        try:
            self.__expression_kind = ExpressionKind.ARITHMETIC
            # Capture the array element (if any) before evaluating its bit
            # indices. Keep the complete source alive across index arithmetic.
            if captured is None:
                bits = self.__angle_arithmetic_operand(selection.source, selection.angle_type)
                indices = self.__capture_angle_bit_indices(selection)
            else:
                bits, indices = captured
            scratch = self.__add_new_temporary_variable(ValueType.UINT)
            values = []
            for index in indices:
                bit = self.__add_new_temporary_variable(ValueType.BIT)
                self.__qcx_lines.extend([
                    f'LET {scratch} := {bits}', f'LET {scratch} >>= {index}',
                    f'LET {scratch} &= 1', f'LET {bit} := :INT:{scratch}',
                ])
                values.append(bit)
            owned = self.__used_temporary_variables - previous_used
            yield tuple(values)
        finally:
            self.__expression_kind, self.__value, self.__value_type, self.__value_kind = previous
            temporaries = owned if owned is not None else self.__used_temporary_variables - previous_used
            for temporary in temporaries:
                self.__release_temporary_variable(temporary)

    def __capture_angle_bit_indices(self, selection: _AngleBitSelection) -> tuple[int | str, ...]:
        if isinstance(selection, _StaticAngleBitSelection):
            return selection.indices
        expressions = ((selection.index,) if isinstance(selection, _RuntimeAngleBitSelection)
                       else selection.indices)
        return tuple(self.__capture_array_index(index, selection.angle_type.width) for index in expressions)

    @contextlib.contextmanager
    def __angle_bits_value(
            self, source: ast.Expression, angle_type: _AngleType,
            indices: Iterable[int], *, captured: str | None = None) -> Iterator[tuple[str | int, ...]]:
        previous = (self.__expression_kind, self.__value, self.__value_type, self.__value_kind)
        previous_used = self.__used_temporary_variables.copy()
        owned = None
        try:
            self.__expression_kind = ExpressionKind.ARITHMETIC
            # Capture the complete angle once before extracting any bits.
            # A compound destination already has a snapshot and reserved
            # index; extracting its bits must not evaluate the source again.
            bits = captured if captured is not None else self.__angle_arithmetic_operand(source, angle_type)
            if isinstance(bits, int):
                owned = self.__used_temporary_variables - previous_used
                yield tuple((bits >> index) & 1 for index in indices)
                return
            scratch = self.__add_new_temporary_variable(ValueType.UINT)
            values = []
            for index in indices:
                bit = self.__add_new_temporary_variable(ValueType.BIT)
                self.__qcx_lines.append(f'LET {scratch} := {bits}')
                if index:
                    self.__qcx_lines.append(f'LET {scratch} >>= {index}')
                self.__qcx_lines.extend([f'LET {scratch} &= 1', f'LET {bit} := :INT:{scratch}'])
                values.append(bit)
            owned = self.__used_temporary_variables - previous_used
            yield tuple(values)
        finally:
            self.__expression_kind, self.__value, self.__value_type, self.__value_kind = previous
            # The consumer may allocate its result while these bits are
            # yielded. Release only this context's storage, not that result.
            temporaries = owned if owned is not None else self.__used_temporary_variables - previous_used
            for temporary in temporaries:
                self.__release_temporary_variable(temporary)

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
        for scope in reversed(self.__loop_bindings):
            if name in scope:
                binding = scope[name]
                return binding.angle_type if isinstance(binding, _RuntimeIteratorBinding) else None
        constant = self.__const_angle_values.get(name)
        return (constant.angle_type if constant is not None
                else self.__angle_variable_types.get(self.__capitalize_variable_name(name)))

    def __angle_operand_info(
            self, expression: ast.Expression,
            iterators: set[str] | dict[str, ValueType] | None = None
            ) -> tuple[ValueType, _AngleType | None, bool]:
        # Shared, non-evaluating inference for arithmetic and skipped bodies.
        # Recurse through arithmetic and bitwise nodes, rather than re-analyzing both
        # subtrees separately for value types, widths, and runtime dependence.
        if isinstance(expression, ast.FunctionCall) and expression.name.name in ('rotl', 'rotr'):
            signature, runtime = self.__function_call_info(expression, iterators)
            angle_type = (self.__angle_operand_info(expression.arguments[0], iterators)[1]
                          if signature.result_type == ValueType.ANGLE else None)
            return signature.result_type, angle_type, runtime
        if (isinstance(expression, ast.BinaryExpression) and expression.op.name in ('<<', '>>')
                and self.__expression_mentions_angle(expression, iterators)):
            info = self.__angle_shift_info(expression, iterators)
            if info is not None:
                return ValueType.ANGLE, info[0], info[2]
        if (isinstance(expression, ast.BinaryExpression) and expression.op.name in ('*', '/')
                and self.__expression_mentions_angle(expression, iterators)):
            info = self.__angle_multiplicative_info(expression, iterators)
            if info is not None:
                return info.result_type, info.result if info.result_type == ValueType.ANGLE else None, info.runtime
        if isinstance(expression, ast.BinaryExpression) and expression.op.name in ('+', '-', '&', '|', '^'):
            lhs_type, lhs_angle, lhs_runtime = self.__angle_operand_info(expression.lhs, iterators)
            rhs_type, rhs_angle, rhs_runtime = self.__angle_operand_info(expression.rhs, iterators)
            runtime = lhs_runtime or rhs_runtime
            if ValueType.ANGLE not in (lhs_type, rhs_type):
                result_type = (self.__bitwise_type(lhs_type, rhs_type) if expression.op.name in ('&', '|', '^')
                               else self.__promoted_type(lhs_type, rhs_type))
                return result_type, None, runtime
            if lhs_angle is None or rhs_angle is None:
                raise UnsupportedOpenQASMError('angle binary arithmetic and bitwise operations require two angle operands')
            # Match UINT's sizedness convention, but preserve angle values by
            # appending low zeros when widening, not by integer zero-extension.
            angle_type = _AngleType(max(lhs_angle.width, rhs_angle.width),
                                    lhs_angle.is_sized or rhs_angle.is_sized)
            return ValueType.ANGLE, angle_type, runtime
        if isinstance(expression, ast.UnaryExpression) and expression.op.name in ('-', '~'):
            value_type, angle_type, runtime = self.__angle_operand_info(expression.expression, iterators)
            if value_type == ValueType.ANGLE:
                return value_type, angle_type, runtime
            if expression.op.name == '~':
                return self.__bitwise_type(value_type), None, runtime
            value_type, runtime = self.__negation_expression_info(expression, iterators)
            return value_type, None, runtime
        value_type, runtime = self.__loop_expression_info(expression, iterators)
        if value_type != ValueType.ANGLE:
            return value_type, None, runtime
        if isinstance(expression, ast.Identifier):
            angle_type = self.__identifier_angle_type(expression.name)
        elif isinstance(expression, ast.IndexExpression) and isinstance(expression.collection, ast.Identifier):
            angle_type = self.__angle_array_types.get(self.__capitalize_variable_name(expression.collection.name))
        elif isinstance(expression, ast.Cast) and isinstance(expression.type, ast.AngleType):
            angle_type = self.__angle_type(expression.type, iterators)
        else:
            angle_type = None
        if angle_type is None:
            raise UnsupportedOpenQASMError('angle expression without width metadata')
        return value_type, angle_type, runtime

    def __angle_shift_info(
            self, expression: ast.BinaryExpression,
            iterators: set[str] | dict[str, ValueType] | None = None
            ) -> tuple[_AngleType, ValueType, bool] | None:
        lhs_type, angle_type, lhs_runtime = self.__angle_operand_info(expression.lhs, iterators)
        # Resolve integer promotion precisely: narrow UINT mixed with INT can
        # produce a signed count that must be checked before any UINT capture.
        count_type, _, count_runtime = self.__integer_operand_info(expression.rhs, iterators)
        if lhs_type != ValueType.ANGLE:
            # Angle quotients can supply ordinary integer shift counts.
            return None
        if count_type not in (ValueType.INT, ValueType.UINT):
            raise UnsupportedOpenQASMError('angle shift count must be a scalar int or uint')
        return angle_type, count_type, lhs_runtime or count_runtime

    def __visit_angle_shift(
            self, expression: ast.BinaryExpression, info: tuple[_AngleType, ValueType, bool]) -> None:
        angle_type, count_type, runtime = info
        if runtime and self.__expression_kind == ExpressionKind.CONST_ARITHMETIC:
            raise NoConstantExpressionException
        previous_used = self.__used_temporary_variables.copy()
        result = None
        keep = False
        try:
            value = self.__angle_arithmetic_operand(expression.lhs, angle_type)
            self.visit(expression.rhs)
            if self.__value_kind == ValueKind.LITERAL:
                count = int(self.__value)
            elif self.__value_kind in (ValueKind.LVALUE, ValueKind.TEMPORARY):
                count = self.__add_new_temporary_variable(
                    count_type, self.__uint_expression_type(expression.rhs) if count_type == ValueType.UINT else None)
                self.__qcx_lines.append(f'LET {count} := {self.__value}')
            else:
                raise UninitializedValueException
            if isinstance(count, int) and (not self.__evaluate_constant or count >= angle_type.width):
                # Evaluate both operands first. An oversized count is bounded
                # work, including huge literals that do not fit native storage.
                self.__value, self.__value_kind = 0, ValueKind.LITERAL
            elif isinstance(value, int) and isinstance(count, int) and count >= 0:
                bits = value << count if expression.op.name == '<<' else value >> count
                self.__value, self.__value_kind = bits & angle_type.mask, ValueKind.LITERAL
            else:
                if self.__expression_kind == ExpressionKind.CONST_ARITHMETIC:
                    if isinstance(count, int) and count < 0:
                        raise InvalidShiftException('Angle shift count must not be negative')
                    raise NoConstantExpressionException
                result = self.__emit_angle_shift(value, count, count_type, angle_type, expression.op.name)
                self.__value, self.__value_kind = result, ValueKind.TEMPORARY
            self.__value_type = ValueType.ANGLE
            self.__angle_expression_types[id(expression)] = (expression, angle_type)
            keep = result is not None
        finally:
            for temporary in self.__used_temporary_variables - previous_used:
                if not keep or temporary != result:
                    self.__release_temporary_variable(temporary)

    def __emit_angle_shift(
            self, value: int | str, count: int | str, count_type: ValueType,
            angle_type: _AngleType, operator: str) -> str:
        storage = _UnsignedIntegerType(angle_type.width, True)
        result = self.__add_new_temporary_variable(ValueType.UINT, storage)
        if isinstance(count, int) and count >= 0:
            self.__qcx_lines.extend([f'LET {result} := {value}', f'LET {result} {operator}= {count}'])
        else:
            if isinstance(count, int):
                temporary = self.__add_new_temporary_variable(count_type)
                self.__qcx_lines.append(f'LET {temporary} := {count}')
                count = temporary
            if count_type == ValueType.INT:
                self.__qcx_lines.append(f'ASSERT {count} >= 0')
            end = f'QASM2QCX_ANGLE_SHIFT_END_{self.__angle_shift_index}'
            self.__angle_shift_index += 1
            self.__qcx_lines.extend([
                f'LET {result} := 0', f'JUMPIF {end} {count} >= {angle_type.width}',
                f'LET {result} := {value}', f'LET {result} {operator}= {count}'])
            # Guard before the native shift, including UINT_MAX counts. No
            # signed conversion or narrowing of the count is required.
            self.__normalize_uint_storage(result, storage)
            self.__qcx_lines.append(f'@{end}')
            return result
        self.__normalize_uint_storage(result, storage)
        return result

    def __visit_angle_bitwise(self, expression: ast.UnaryExpression | ast.BinaryExpression) -> None:
        _, angle_type, runtime = self.__angle_operand_info(expression)
        if runtime and self.__expression_kind == ExpressionKind.CONST_ARITHMETIC:
            raise NoConstantExpressionException
        previous_used = self.__used_temporary_variables.copy()
        result = None
        keep = False
        try:
            operands = ([expression.expression] if isinstance(expression, ast.UnaryExpression)
                        else [expression.lhs, expression.rhs])
            values = [self.__angle_arithmetic_operand(operand, angle_type) for operand in operands]
            unary = isinstance(expression, ast.UnaryExpression)
            if all(isinstance(value, int) for value in values):
                if not self.__evaluate_constant:
                    value = 0
                elif unary:
                    value = ~values[0]
                elif expression.op.name == '&':
                    value = values[0] & values[1]
                elif expression.op.name == '|':
                    value = values[0] | values[1]
                else:
                    value = values[0] ^ values[1]
                self.__value, self.__value_kind = value & angle_type.mask, ValueKind.LITERAL
            else:
                if self.__expression_kind == ExpressionKind.CONST_ARITHMETIC:
                    raise NoConstantExpressionException
                storage = _UnsignedIntegerType(angle_type.width, True)
                result = self.__add_new_temporary_variable(ValueType.UINT, storage)
                operator = '^=' if unary else expression.op.name + '='
                rhs = angle_type.mask if unary else values[1]
                self.__qcx_lines.extend([f'LET {result} := {values[0]}', f'LET {result} {operator} {rhs}'])
                self.__normalize_uint_storage(result, storage)
                self.__value, self.__value_kind = result, ValueKind.TEMPORARY
            self.__value_type = ValueType.ANGLE
            self.__angle_expression_types[id(expression)] = (expression, angle_type)
            keep = result is not None
        finally:
            for temporary in self.__used_temporary_variables - previous_used:
                if not keep or temporary != result:
                    self.__release_temporary_variable(temporary)

    def __angle_comparison_info(
            self, expression: ast.BinaryExpression,
            iterators: set[str] | dict[str, ValueType] | None = None
            ) -> tuple[_AngleType, _AngleType, bool] | None:
        # Infer without evaluating operands, including in skipped bodies.
        lhs_type, lhs_angle, lhs_runtime = self.__angle_operand_info(expression.lhs, iterators)
        rhs_type, rhs_angle, rhs_runtime = self.__angle_operand_info(expression.rhs, iterators)
        if ValueType.ANGLE not in (lhs_type, rhs_type):
            # An integer quotient can mention angles without being an angle.
            return None
        if lhs_angle is None or rhs_angle is None:
            raise UnsupportedOpenQASMError('angle comparison requires two angle operands')
        return lhs_angle, rhs_angle, lhs_runtime or rhs_runtime

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
        if info.runtime and self.__expression_kind == ExpressionKind.CONST_ARITHMETIC:
            raise NoConstantExpressionException
        previous_used = self.__used_temporary_variables.copy()
        result = None
        keep = False
        try:
            values = [self.__angle_multiplicative_operand(operand, metadata, info.result.width)
                      for operand, metadata in ((expression.lhs, info.lhs), (expression.rhs, info.rhs))]
            defer_zero = (expression.op.name == '/' and values[1] == 0
                          and self.__expression_kind == ExpressionKind.ARITHMETIC)
            if all(isinstance(value, int) for value in values) and not defer_zero:
                if not self.__evaluate_constant:
                    value = 0
                elif expression.op.name == '*':
                    value = values[0] * values[1]
                elif values[1] != 0:
                    value = values[0] // values[1]
                else:
                    raise ZeroDivisorException('/')
                self.__value, self.__value_kind = value & info.result.mask, ValueKind.LITERAL
            else:
                if self.__expression_kind == ExpressionKind.CONST_ARITHMETIC:
                    raise NoConstantExpressionException
                storage = (info.result if info.result_type == ValueType.UINT
                           else _UnsignedIntegerType(info.result.width, True))
                result = self.__add_new_temporary_variable(ValueType.UINT, storage)
                operator = '*=' if expression.op.name == '*' else '/='
                self.__qcx_lines.extend([f'LET {result} := {values[0]}', f'LET {result} {operator} {values[1]}'])
                # Native UINT division diagnoses zero before modifying this
                # private result. A skipped classical path never executes it.
                self.__normalize_uint_storage(result, storage)
                self.__value, self.__value_kind = result, ValueKind.TEMPORARY
            self.__value_type = info.result_type
            metadata_map = (self.__angle_expression_types if info.result_type == ValueType.ANGLE
                            else self.__uint_expression_types)
            metadata_map[id(expression)] = (expression, info.result)
            keep = result is not None
        finally:
            for temporary in self.__used_temporary_variables - previous_used:
                if not keep or temporary != result:
                    self.__release_temporary_variable(temporary)

    def __angle_multiplicative_operand(
            self, expression: ast.Expression, metadata: _AngleType | _UnsignedIntegerType,
            width: int) -> int | str:
        if isinstance(metadata, _AngleType):
            return self.__angle_arithmetic_operand(expression, _AngleType(width, metadata.is_sized))
        # UINT is an integer factor, not a turn fraction: never shift its bits
        # to align it with an angle. The typing step checked matching widths.
        self.visit(expression)
        if self.__value_type != ValueType.UINT:
            raise UnsupportedOpenQASMError('angle scaling requires an unsigned integer factor')
        if self.__value_kind == ValueKind.LITERAL:
            return metadata.normalize(int(self.__value))
        if self.__value_kind not in (ValueKind.LVALUE, ValueKind.TEMPORARY):
            raise UninitializedValueException
        snapshot = self.__add_new_temporary_variable(ValueType.UINT, metadata)
        self.__qcx_lines.append(f'LET {snapshot} := {self.__value}')
        return snapshot

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
        if isinstance(expression, ast.IndexExpression):
            return self.__angle_operand_info(expression)[1]
        if (isinstance(expression, (ast.BinaryExpression, ast.UnaryExpression))
                or isinstance(expression, ast.FunctionCall) and expression.name.name in ('rotl', 'rotr')):
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
        angle_range = self.__runtime_angle_range_register_assignment_info(target, expression)
        if angle_range is not None:
            self.__emit_runtime_angle_range_register_assignment(angle_range)
            return
        angle_expression = self.__runtime_angle_range_expression_assignment_info(target, expression)
        if angle_expression is not None:
            self.__emit_runtime_register_assignment(angle_expression)
            return
        runtime = self.__runtime_register_assignment_info(target, expression)
        if runtime is not None:
            self.__emit_runtime_register_assignment(runtime)
            return
        with self.__bit_assignment_target(target) as (target_names, target_is_register):
            self.__emit_bit_value_assignment(target_names, target_is_register, expression)

    def __emit_runtime_angle_range_register_assignment(
            self, info: _RuntimeAngleRangeRegisterAssignment) -> None:
        previous = (self.__expression_kind, self.__value, self.__value_type,
                    self.__value_kind, self.__evaluate_constant)
        previous_used = self.__used_temporary_variables.copy()
        previous_declarations = self.__declared_temporary_variables.copy()
        prefix = f'QASM2QCX_ANGLE_RANGE_ASSIGNMENT_{self.__register_assignment_index}'
        self.__register_assignment_index += 1
        try:
            self.__expression_kind = ExpressionKind.ARITHMETIC
            self.__check_runtime_register_output(1)
            # Retain the source snapshot before evaluating bounds, including
            # an array-element address, and validate all positions and count
            # before extracting or writing any selected bit.
            source = self.__angle_arithmetic_operand(info.selection.source, info.selection.angle_type)
            positions, count = self.__capture_bit_range_positions(
                info.selection.bounds, info.selection.angle_type.width, prefix, capacity=info.capacity)
            self.__check_runtime_register_output(19)
            bits = self.__add_temporary_int_array(info.capacity, 'ANGLE_RANGE_ASSIGNMENT_BITS')
            index = self.__add_new_temporary_variable(ValueType.INT)
            position = self.__add_new_temporary_variable(ValueType.INT)
            scratch = self.__add_new_temporary_variable(ValueType.UINT)
            bit = self.__add_new_temporary_variable(ValueType.INT)
            destination = self.__capitalize_variable_name(info.target.name)
            target = destination if info.width == 1 else f'{destination}:{index}'
            self.__qcx_lines.extend([
                f'ASSERT {count} == {info.width}', f'LET {index} := 0',
                f'@{prefix}_SNAPSHOT', f'JUMPIF {prefix}_SNAPSHOT_DONE {index} >= {count}',
                f'LET {position} := {positions}:{index}', f'LET {scratch} := {source}',
                f'LET {scratch} >>= {position}', f'LET {scratch} &= 1',
                f'LET {bit} := :INT:{scratch}', f'LET {bits}:{index} := {bit}',
                f'LET {index} += 1', f'JUMP {prefix}_SNAPSHOT', f'@{prefix}_SNAPSHOT_DONE',
                f'LET {index} := 0', f'@{prefix}_WRITE',
                f'JUMPIF {prefix}_END {index} >= {count}',
                f'LET {target} := {bits}:{index}', f'LET {index} += 1',
                f'JUMP {prefix}_WRITE', f'@{prefix}_END',
            ])
            self.__check_runtime_register_output()
            self.__hoist_temporary_declarations(previous_declarations)
        finally:
            (self.__expression_kind, self.__value, self.__value_type,
             self.__value_kind, self.__evaluate_constant) = previous
            for temporary in self.__used_temporary_variables - previous_used:
                self.__release_temporary_variable(temporary)

    def __snapshot_register_assignment_destination(
            self, info: _RuntimeBitRegisterAssignment | _RuntimeBitRangeExpressionAssignment,
            positions: tuple[int | str, ...] | str, count: int | str, prefix: str) -> str:
        capacity = info.capacity if isinstance(info, _RuntimeBitRangeExpressionAssignment) else info.width
        original = self.__add_temporary_int_array(capacity, 'REGISTER_ASSIGNMENT_ORIGINAL')
        target = self.__capitalize_variable_name(self.__operand_name(info.target))
        size = self.__bit_variable_name_size_map[target]
        if isinstance(positions, str) or info.selection is None:
            index = self.__add_new_temporary_variable(ValueType.INT)
            reads = []
            position = index
            if isinstance(positions, str):
                position = self.__add_new_temporary_variable(ValueType.INT)
                reads.append(f'LET {position} := {positions}:{index}')
            reads.append(f'LET {original}:{index} := {self.__qcx_bit_name(target, size, position)}')
            lines = [
                f'LET {index} := 0', f'@{prefix}_ORIGINAL',
                f'JUMPIF {prefix}_ORIGINAL_DONE {index} >= {count}', *reads,
                f'LET {index} += 1', f'JUMP {prefix}_ORIGINAL', f'@{prefix}_ORIGINAL_DONE',
            ]
        else:
            lines = [f'@{prefix}_ORIGINAL']
            lines.extend(f'LET {original}:{offset} := {self.__qcx_bit_name(target, size, position)}'
                         for offset, position in enumerate(positions))
            lines.append(f'@{prefix}_ORIGINAL_DONE')
        self.__check_runtime_register_output(len(lines))
        self.__qcx_lines.extend(lines)
        return original

    def __emit_runtime_register_assignment(
            self, info: _RuntimeBitRegisterAssignment | _RuntimeBitRangeExpressionAssignment) -> None:
        previous = (self.__expression_kind, self.__value, self.__value_type,
                    self.__value_kind, self.__evaluate_constant)
        previous_used = self.__used_temporary_variables.copy()
        previous_declarations = self.__declared_temporary_variables.copy()
        prefix = f'QASM2QCX_REGISTER_ASSIGNMENT_{self.__register_assignment_index}'
        self.__register_assignment_index += 1
        destination = self.__capitalize_variable_name(self.__operand_name(info.target))
        try:
            positions: tuple[int | str, ...] = ()
            if isinstance(info, _RuntimeBitRangeExpressionAssignment):
                self.__expression_kind = ExpressionKind.ARITHMETIC
                self.__check_runtime_register_output(9)
                destination_positions, destination_count = self.__capture_bit_range_positions(
                    info.selection.bounds, info.selection.source_width, f'{prefix}_DESTINATION', capacity=info.capacity)
                self.__check_distinct_bit_range_positions(
                    destination_positions, destination_count, info.selection.source_width, f'{prefix}_DESTINATION')
                self.__check_runtime_register_output(9)
            elif isinstance(info.selection, _RuntimeBitSetSelection):
                self.__expression_kind = ExpressionKind.ARITHMETIC
                comparisons = info.width * (info.width - 1) // 2
                self.__check_runtime_register_output(comparisons + info.width)
                positions = tuple(self.__capture_array_index(index, info.selection.source_width)
                                  for index in info.selection.indices)
                self.__check_runtime_register_output(comparisons)
                self.__check_distinct_bit_indices(positions)
            elif info.selection is not None:
                positions = info.selection.indices
            expected_count = destination_count if isinstance(info, _RuntimeBitRangeExpressionAssignment) else info.width
            original = None
            if info.binary_operator is not None:
                original = self.__snapshot_register_assignment_destination(
                    info, destination_positions if isinstance(info, _RuntimeBitRangeExpressionAssignment) else positions,
                    expected_count, prefix)
            with self.__runtime_bit_register_expression_value(info.value, f'{prefix}_VALUE') as (snapshot, count):
                # No destination bit is written until the complete RHS and
                # all nested checks have finished and its final length agrees.
                guard = [f'ASSERT {count} == {expected_count}']
                if original is not None:
                    index = self.__add_new_temporary_variable(ValueType.INT)
                    self.__check_runtime_register_output(7)
                    self.__qcx_lines.extend(guard)
                    self.__combine_bit_range_snapshots(original, snapshot, info.operator, index, count, prefix)
                    snapshot = original
                    guard = []  # The final length was checked before combining.
                if isinstance(info, _RuntimeBitRangeExpressionAssignment):
                    index = self.__add_new_temporary_variable(ValueType.INT)
                    position = self.__add_new_temporary_variable(ValueType.INT)
                    self.__check_runtime_register_output(8 + len(guard))
                    target = self.__qcx_bit_name(destination, info.selection.source_width, position)
                    self.__qcx_lines.extend([
                        *guard, f'LET {index} := 0',
                        f'@{prefix}_WRITE', f'JUMPIF {prefix}_END {index} >= {destination_count}',
                        f'LET {position} := {destination_positions}:{index}',
                        f'LET {target} := {snapshot}:{index}', f'LET {index} += 1',
                        f'JUMP {prefix}_WRITE', f'@{prefix}_END',
                    ])
                elif info.selection is None:
                    index = self.__add_new_temporary_variable(ValueType.INT)
                    self.__check_runtime_register_output(7 + len(guard))
                    target = destination if info.width == 1 else f'{destination}:{index}'
                    self.__qcx_lines.extend([
                        *guard, f'LET {index} := 0',
                        f'@{prefix}_WRITE', f'JUMPIF {prefix}_END {index} >= {info.width}',
                        f'LET {target} := {snapshot}:{index}', f'LET {index} += 1',
                        f'JUMP {prefix}_WRITE', f'@{prefix}_END',
                    ])
                else:
                    self.__check_runtime_register_output(info.width + 2 + len(guard))
                    self.__qcx_lines.extend([*guard, f'@{prefix}_WRITE'])
                    size = self.__bit_variable_name_size_map[destination]
                    for offset, position in enumerate(positions):
                        target = self.__qcx_bit_name(destination, size, position)
                        self.__qcx_lines.append(f'LET {target} := {snapshot}:{offset}')
                    self.__qcx_lines.append(f'@{prefix}_END')
            self.__hoist_temporary_declarations(previous_declarations)
        finally:
            (self.__expression_kind, self.__value, self.__value_type,
             self.__value_kind, self.__evaluate_constant) = previous
            for temporary in self.__used_temporary_variables - previous_used:
                self.__release_temporary_variable(temporary)

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
        return self.__add_temporary_int_array(width, 'BITS')

    def __add_temporary_int_array(self, width: int, purpose: str) -> str:
        return self.__add_temporary_array(width, purpose, ValueType.INT)

    def __add_temporary_array(self, width: int, purpose: str, value_type: ValueType) -> str:
        if value_type not in (ValueType.INT, ValueType.UINT):
            raise ValueError('temporary arrays require INT or UINT storage')
        # Preserve INT-array names. A distinct UINT namespace prevents reuse
        # of storage that was already declared with a different element type.
        prefix = f'QASM2QCX_{purpose}_{"UINT_" if value_type == ValueType.UINT else ""}{width}_'
        index = 0
        name = prefix + str(index)
        while name in self.__used_temporary_variables or name in self.__reserved_variable_names:
            index += 1
            name = prefix + str(index)
        self.__used_temporary_variables.add(name)
        if value_type == ValueType.UINT:
            self.__temporary_uint_types[name] = _UnsignedIntegerType(self.QCX_UINT_WIDTH, False)
        if name not in self.__declared_temporary_variables:
            self.__declared_temporary_variables.add(name)
            self.__qcx_lines.append(f'VAR {name} {value_type.name} {width}')
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
    def __runtime_bit_set_value(self, selection: _RuntimeBitSetSelection) -> Iterator[tuple[str, ...]]:
        if len(self.__qcx_lines) + selection.width > self.MAX_LOOP_OUTPUT_LINES:
            raise InvalidBitOperandException('Runtime bit selection exceeds QCX output limit')
        previous = (self.__expression_kind, self.__value, self.__value_type,
                    self.__value_kind, self.__evaluate_constant)
        previous_used = self.__used_temporary_variables.copy()
        owned = None
        try:
            self.__expression_kind = ExpressionKind.ARITHMETIC
            constant = (self.__constant_bit_register(selection.source_name)
                        if selection.source_name in self.__const_bit_register_values else None)
            source = self.__capitalize_variable_name(selection.source_name)
            # Capture every occurrence before reading bits, including repeated
            # indices and constants in a mixed runtime selection.
            indices = tuple(self.__capture_array_index(index, selection.source_width) for index in selection.indices)
            words = self.__prepare_constant_bit_lookup(constant) if constant is not None else None
            values = []
            for index in indices:
                bit = self.__add_new_temporary_variable(ValueType.BIT)
                if words is None:
                    operand = source if selection.source_width == 1 else f'{source}:{index}'
                    self.__qcx_lines.append(f'LET {bit} := {operand}')
                else:
                    self.__copy_prepared_constant_bit(words, index, bit)
                values.append(bit)
            if len(self.__qcx_lines) > self.MAX_LOOP_OUTPUT_LINES:
                raise InvalidBitOperandException('Runtime bit selection exceeds QCX output limit')
            # Only release this access's storage after yield: its consumer may
            # allocate a result that must remain live beyond this context.
            owned = self.__used_temporary_variables - previous_used
            yield tuple(values)
        finally:
            (self.__expression_kind, self.__value, self.__value_type,
             self.__value_kind, self.__evaluate_constant) = previous
            temporaries = owned if owned is not None else self.__used_temporary_variables - previous_used
            for temporary in temporaries:
                self.__release_temporary_variable(temporary)

    @contextlib.contextmanager
    def __runtime_bit_range_tree_value(
            self, tree: _BitRangeCastExpression) -> Iterator[tuple[str | int, ...]]:
        self.__check_uint_bit_cast_output(1)
        previous = (self.__expression_kind, self.__value, self.__value_type,
                    self.__value_kind, self.__evaluate_constant)
        previous_used = self.__used_temporary_variables.copy()
        owned = None
        try:
            self.__expression_kind = ExpressionKind.ARITHMETIC
            if not tree.children:
                if isinstance(tree.selection, _RuntimeAngleBitRangeSelection):
                    source = self.__runtime_angle_bit_range_value(tree.selection, tree.width)
                elif tree.selection is not None:
                    source = self.__runtime_bit_range_value(tree.selection, tree.width)
                else:
                    source = self.__bit_register_expression_value(tree.expression)
                with source as bits:
                    if len(bits) != tree.width:
                        raise InvalidBitOperandException('Bitwise UINT cast operand width mismatch')
                    values = list(bits)
                    if tree.selection is None:
                        # Freeze fixed register reads at this operand's position,
                        # before any access or arithmetic in the right subtree.
                        for index, bit in enumerate(bits):
                            if isinstance(bit, str):
                                snapshot = self.__add_new_temporary_variable(ValueType.BIT)
                                self.__qcx_lines.append(f'LET {snapshot} := {bit}')
                                values[index] = snapshot
                    self.__check_uint_bit_cast_output()
                    owned = self.__used_temporary_variables - previous_used
                    yield tuple(values)
                return
            with contextlib.ExitStack() as result:
                with contextlib.ExitStack() as operands:
                    # Complete each register child, including nested shifts
                    # and rotations, before starting the next child. A parent
                    # rotation evaluates its distance only after this capture.
                    children = [operands.enter_context(self.__runtime_bit_range_tree_value(child))
                                for child in tree.children]
                    bindings = []
                    try:
                        for child, bits in zip(tree.children, children):
                            key = id(child.expression)
                            bindings.append((key, self.__captured_bit_register_values.get(key)))
                            self.__captured_bit_register_values[key] = (child.expression, bits)
                        values = result.enter_context(self.__bit_register_expression_value(tree.expression))
                    finally:
                        for key, old in reversed(bindings):
                            if old is None:
                                self.__captured_bit_register_values.pop(key, None)
                            else:
                                self.__captured_bit_register_values[key] = old
                # The parent result is private now; release child storage before
                # yielding so later consumers may safely reuse it.
                self.__check_uint_bit_cast_output()
                owned = self.__used_temporary_variables - previous_used
                yield values
        finally:
            (self.__expression_kind, self.__value, self.__value_type,
             self.__value_kind, self.__evaluate_constant) = previous
            temporaries = (self.__used_temporary_variables - previous_used if owned is None
                           else self.__used_temporary_variables & owned)
            for temporary in temporaries:
                self.__release_temporary_variable(temporary)

    @contextlib.contextmanager
    def __runtime_bit_range_cast_value(
            self, expression: ast.Expression, info: _UIntBitCastInfo) -> Iterator[tuple[str | int, ...]]:
        previous_used = self.__used_temporary_variables.copy()
        owned = None
        try:
            with self.__runtime_bit_range_value(info.range_selection, info.width) as snapshot:
                if not info.range_operations:
                    owned = self.__used_temporary_variables - previous_used
                    yield snapshot
                    return
                operation = info.range_operations[0].expression
                leaf = operation.expression if isinstance(operation, ast.UnaryExpression) else operation.lhs
                # Bind this occurrence only. Count expressions, including nested
                # casts selecting the same source, keep their own evaluations.
                previous = self.__captured_bit_register_values.get(id(leaf))
                self.__captured_bit_register_values[id(leaf)] = (leaf, snapshot)
                try:
                    with self.__bit_register_expression_value(expression) as values:
                        self.__check_uint_bit_cast_output()
                        owned = self.__used_temporary_variables - previous_used
                        yield values
                finally:
                    if previous is None:
                        self.__captured_bit_register_values.pop(id(leaf), None)
                    else:
                        self.__captured_bit_register_values[id(leaf)] = previous
        finally:
            # Helpers normally release their storage. Also clean up partially
            # allocated intermediates on failure, preserving consumer results.
            temporaries = (self.__used_temporary_variables - previous_used if owned is None
                           else self.__used_temporary_variables & owned)
            for temporary in temporaries:
                self.__release_temporary_variable(temporary)

    @contextlib.contextmanager
    def __runtime_angle_bit_range_value(
            self, selection: _RuntimeAngleBitRangeSelection, width: int) -> Iterator[tuple[str, ...]]:
        self.__check_uint_bit_cast_output(5 * width + 1)
        previous = (self.__expression_kind, self.__value, self.__value_type,
                    self.__value_kind, self.__evaluate_constant)
        previous_used = self.__used_temporary_variables.copy()
        owned = None
        label = f'QASM2QCX_ANGLE_CAST_RANGE_{self.__array_index}'
        self.__array_index += 1
        try:
            self.__expression_kind = ExpressionKind.ARITHMETIC
            # Capture the source (including an array address) before its range
            # operands. Keep the full angle snapshot alive across bound/count
            # arithmetic, and validate every position before extracting bits.
            snapshot = self.__angle_arithmetic_operand(selection.source, selection.angle_type)
            positions, count = self.__capture_bit_range_positions(
                selection.bounds, selection.angle_type.width, label, capacity=width)
            self.__qcx_lines.append(f'ASSERT {count} == {width}')
            position = self.__add_new_temporary_variable(ValueType.INT)
            scratch = self.__add_new_temporary_variable(ValueType.UINT)
            values = []
            for offset in range(width):
                bit = self.__add_new_temporary_variable(ValueType.BIT)
                self.__qcx_lines.extend([
                    f'LET {position} := {positions}:{offset}', f'LET {scratch} := {snapshot}',
                    f'LET {scratch} >>= {position}', f'LET {scratch} &= 1',
                    f'LET {bit} := :INT:{scratch}',
                ])
                values.append(bit)
            self.__check_uint_bit_cast_output()
            owned = self.__used_temporary_variables - previous_used
            yield tuple(values)
        finally:
            (self.__expression_kind, self.__value, self.__value_type,
             self.__value_kind, self.__evaluate_constant) = previous
            temporaries = owned if owned is not None else self.__used_temporary_variables - previous_used
            for temporary in temporaries:
                self.__release_temporary_variable(temporary)

    @contextlib.contextmanager
    def __runtime_bit_range_value(
            self, selection: _RuntimeBitRangeSelection, width: int) -> Iterator[tuple[str, ...]]:
        self.__check_uint_bit_cast_output(2 * width + 1)
        previous = (self.__expression_kind, self.__value, self.__value_type,
                    self.__value_kind, self.__evaluate_constant)
        previous_used = self.__used_temporary_variables.copy()
        owned = None
        label = f'QASM2QCX_RANGE_UINT_{self.__array_index}'
        self.__array_index += 1
        try:
            self.__expression_kind = ExpressionKind.ARITHMETIC
            # The target width bounds storage independently of the source and
            # raw range length. Validate all positions and the exact count
            # before preparing a lookup table or reading any selected bits.
            positions, count = self.__capture_bit_range_positions(
                selection.bounds, selection.source_width, label, capacity=width)
            self.__qcx_lines.append(f'ASSERT {count} == {width}')
            constant = (self.__constant_bit_register(selection.source_name)
                        if selection.source_name in self.__const_bit_register_values else None)
            words = self.__prepare_constant_bit_lookup(constant) if constant is not None else None
            source = self.__capitalize_variable_name(selection.source_name)
            position = self.__add_new_temporary_variable(ValueType.INT)
            values = []
            for offset in range(width):
                self.__qcx_lines.append(f'LET {position} := {positions}:{offset}')
                bit = self.__add_new_temporary_variable(ValueType.BIT)
                if words is None:
                    operand = source if selection.source_width == 1 else f'{source}:{position}'
                    self.__qcx_lines.append(f'LET {bit} := {operand}')
                else:
                    self.__copy_prepared_constant_bit(words, position, bit)
                values.append(bit)
            self.__check_uint_bit_cast_output()
            # The UINT result is allocated by the consumer during yield and
            # must survive release of this snapshot's private storage.
            owned = self.__used_temporary_variables - previous_used
            yield tuple(values)
        finally:
            (self.__expression_kind, self.__value, self.__value_type,
             self.__value_kind, self.__evaluate_constant) = previous
            temporaries = owned if owned is not None else self.__used_temporary_variables - previous_used
            for temporary in temporaries:
                self.__release_temporary_variable(temporary)

    @contextlib.contextmanager
    def __bit_register_expression_value(
            self, expression: ast.Expression) -> Iterator[tuple[str | int, ...]]:
        # The caller validates widths first. Each result owns its temporary
        # bits until the assignment finishes; no destination is written here.
        captured = self.__captured_bit_register_values.get(id(expression))
        if captured is not None:
            yield captured[1]
            return
        selection = self.__angle_bit_selection(expression)
        if selection is not None:
            if isinstance(selection, (_RuntimeAngleBitSelection, _RuntimeAngleBitSetSelection)):
                with self.__runtime_angle_bit_value(selection) as values:
                    yield values
                return
            with self.__angle_bits_value(selection.source, selection.angle_type, selection.indices) as values:
                yield values
            return
        if isinstance(expression, ast.Cast):
            uint_info = self.__uint_bit_cast_info(expression)
            if uint_info is not None and uint_info.result_type == ValueType.BIT_REGISTER:
                with self.__uint_to_bit_register_value(expression, uint_info) as values:
                    yield values
                return
            info = self.__angle_bit_cast_info(expression)
            if info is not None and info[0] == ValueType.BIT_REGISTER:
                with self.__angle_to_bit_register_value(expression) as values:
                    yield values
                return
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
            runtime_selection = self.__runtime_bit_set_selection(expression)
            if runtime_selection is not None:
                with self.__runtime_bit_set_value(runtime_selection) as values:
                    yield values
                return
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
        selection = self.__angle_bit_selection(expression) or self.__runtime_bit_set_selection(expression)
        if selection is not None and not target_is_register:
            raise InvalidBitOperandException('A bit selection requires a bit-register target')
        if target_is_register and (constant_source or selection is not None or isinstance(
                expression, (ast.UnaryExpression, ast.BinaryExpression, ast.FunctionCall, ast.Cast))):
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
            if value_kind == ValueKind.LITERAL and not -size <= int(value) < size:
                # Preserve a reached bounds failure without emitting a huge
                # literal that the QCX parser would reject on skipped paths.
                value, value_type = size, ValueType.INT
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
                ast.IntType, ast.UintType, ast.FloatType, ast.BoolType, ast.ComplexType, ast.AngleType)):
            raise UnsupportedOpenQASMError('classical array base type other than int, uint, float, bool, complex, or angle')
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
        # Angle arrays use UINT storage without becoming UINT expressions or
        # scalar angle variables. Keep their element width in separate metadata.
        storage_type = ast.UintType() if element_type == ValueType.ANGLE else statement.type.base_type
        self.__declare_classical_variable(storage_type, variable_name, size)
        self.__array_element_types[variable_name] = element_type
        if initializer is None:
            return
        for index, expression in enumerate(initializer.values):
            if element_type == ValueType.ANGLE:
                self.__emit_angle_copy(f'{variable_name}:{index}', self.__angle_array_types[variable_name], expression)
                continue
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
            if isinstance(base_type, ast.AngleType):
                metadata = (self.__angle_array_types if isinstance(statement.type, ast.ArrayType)
                            else self.__angle_variable_types)
                metadata[self.__capitalize_variable_name(statement.identifier.name)] = self.__angle_type(base_type)
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

        if statement.op.name == '=':
            angle_range = self.__runtime_angle_range_register_assignment_info(statement.lvalue, statement.rvalue)
            if angle_range is not None:
                self.__emit_runtime_angle_range_register_assignment(angle_range)
                return
            angle_expression = self.__runtime_angle_range_expression_assignment_info(statement.lvalue, statement.rvalue)
            if angle_expression is not None:
                self.__emit_runtime_register_assignment(angle_expression)
                return
            angle_selection = self.__runtime_angle_range_static_selection_assignment_info(statement.lvalue, statement.rvalue)
            if angle_selection is not None:
                self.__emit_runtime_register_assignment(angle_selection)
                return
            angle_set = self.__runtime_angle_range_set_expression_assignment_info(statement.lvalue, statement.rvalue)
            if angle_set is not None:
                self.__emit_runtime_register_assignment(angle_set)
                return
            angle_range_destination = self.__runtime_angle_range_range_expression_assignment_info(
                statement.lvalue, statement.rvalue)
            if angle_range_destination is not None:
                self.__emit_runtime_register_assignment(angle_range_destination)
                return
        angle_compound = self.__runtime_angle_range_register_compound_assignment_info(statement)
        if angle_compound is not None:
            self.__emit_runtime_register_assignment(angle_compound)
            return
        bit_range_copy = self.__runtime_bit_range_copy_info(statement)
        if bit_range_copy is not None:
            if isinstance(bit_range_copy, _RuntimeBitRangeToRangeCopyInfo):
                self.__emit_runtime_bit_range_to_range_copy(bit_range_copy)
            else:
                self.__emit_runtime_bit_range_copy(bit_range_copy)
            return
        compound = self.__runtime_register_compound_assignment_info(statement)
        if compound is not None:
            self.__emit_runtime_register_assignment(compound)
            return
        angle_bits = self.__angle_bit_assignment_info(statement)
        if angle_bits is not None:
            self.__emit_angle_bit_assignment(angle_bits.selection, statement.rvalue, angle_bits.binary_operator)
            return
        if statement.op.name == '=':
            runtime_range = self.__runtime_range_expression_assignment_info(statement.lvalue, statement.rvalue)
            if runtime_range is not None:
                self.__emit_runtime_register_assignment(runtime_range)
                return
            runtime_set = self.__runtime_set_expression_assignment_info(statement.lvalue, statement.rvalue)
            if runtime_set is not None:
                self.__emit_runtime_register_assignment(runtime_set)
                return
        bit_set = self.__runtime_bit_set_assignment_info(statement)
        if bit_set is not None:
            self.__emit_runtime_bit_set_assignment(bit_set.selection, statement.rvalue, bit_set.binary_operator)
            return
        bit_range = self.__runtime_bit_range_assignment_info(statement)
        if bit_range is not None:
            if isinstance(bit_range, _RuntimeBitRangeShiftAssignmentInfo):
                self.__emit_runtime_bit_range_shift_assignment(bit_range, statement.rvalue)
            else:
                self.__emit_runtime_bit_range_assignment(bit_range, statement.rvalue)
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
            element_type = self.__array_element_types[variable_name]
            self.__validate_array_assignment_operator(element_type, statement.op.name)
            operand, index = self.__array_access(statement.lvalue)
            previous_temporaries = self.__used_temporary_variables.copy()
            try:
                if element_type == ValueType.ANGLE:
                    self.__emit_angle_array_assignment(statement, operand, self.__angle_array_types[variable_name])
                else:
                    self.__emit_classical_value_assignment(
                        statement, operand, element_type, runtime_array=index is not None)
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

    @contextlib.contextmanager
    def __angle_bit_assignment_values(
            self, selection: _AngleBitSelection, snapshot: str,
            expression: ast.Expression, operator: str | None, *,
            indices: tuple[int | str, ...]) -> Iterator[tuple[str | int, ...]]:
        if operator is None:
            with self.__bit_register_expression_value(expression) as bits:
                yield bits
            return
        if isinstance(selection, (_RuntimeAngleBitSelection, _RuntimeAngleBitSetSelection)):
            selector = ([selection.index] if isinstance(selection, _RuntimeAngleBitSelection)
                        else ast.DiscreteSet(list(selection.indices)))
            lhs = ast.IndexExpression(selection.source, selector)
            old_values = self.__runtime_angle_bit_value(selection, captured=(snapshot, indices))
        else:
            lhs = ast.IndexExpression(selection.source, ast.DiscreteSet(
                [ast.IntegerLiteral(bit_index) for bit_index in selection.indices]))
            old_values = self.__angle_bits_value(
                selection.source, selection.angle_type, selection.indices, captured=snapshot)
        with old_values as old_bits:
            with self.__bit_register_operation_values(lhs, old_bits, operator, expression) as bits:
                yield bits

    @contextlib.contextmanager
    def __bit_register_operation_values(
            self, lhs: ast.IndexExpression, old_bits: tuple[str | int, ...],
            operator: str, expression: ast.Expression) -> Iterator[tuple[str | int, ...]]:
        # Bind only the synthetic LHS read. Explicit RHS accesses retain their
        # own captures and checks, even when they select the same source bits.
        previous = self.__captured_bit_register_values.get(id(lhs))
        self.__captured_bit_register_values[id(lhs)] = (lhs, old_bits)
        try:
            operation = ast.BinaryExpression(ast.BinaryOperator[operator], lhs, expression)
            with self.__bit_register_expression_value(operation) as bits:
                yield bits
        finally:
            if previous is None:
                self.__captured_bit_register_values.pop(id(lhs), None)
            else:
                self.__captured_bit_register_values[id(lhs)] = previous

    @contextlib.contextmanager
    def __runtime_bit_assignment_values(
            self, selection: _RuntimeBitSetSelection | _RuntimeBitRangeSelection,
            target: str, indices: tuple[str, ...],
            expression: ast.Expression, operator: str | None) -> Iterator[tuple[str | int, ...]]:
        if operator is None:
            with self.__bit_register_expression_value(expression) as bits:
                yield bits
            return
        old_bits = []
        try:
            for index in indices:
                bit = self.__add_new_temporary_variable(ValueType.BIT)
                old_bits.append(bit)
                operand = target if selection.source_width == 1 else f'{target}:{index}'
                self.__qcx_lines.append(f'LET {bit} := {operand}')
            selector = (ast.DiscreteSet(list(selection.indices)) if isinstance(selection, _RuntimeBitSetSelection)
                        else [selection.bounds])
            lhs = ast.IndexExpression(ast.Identifier(selection.source_name), selector)
            with self.__bit_register_operation_values(lhs, tuple(old_bits), operator, expression) as bits:
                yield bits
        finally:
            for bit in old_bits:
                self.__release_temporary_variable(bit)

    def __emit_runtime_bit_set_assignment(
            self, selection: _RuntimeBitSetSelection, expression: ast.Expression,
            operator: str | None = None) -> None:
        # Pairwise checks depend on the selection length, not the register
        # width, so they also cover positions beyond a native UINT bitmap.
        comparisons = selection.width * (selection.width - 1) // 2
        if len(self.__qcx_lines) + comparisons + selection.width > self.MAX_LOOP_OUTPUT_LINES:
            raise InvalidBitOperandException('Runtime bit assignment exceeds QCX output limit')
        previous = (self.__expression_kind, self.__value, self.__value_type,
                    self.__value_kind, self.__evaluate_constant)
        previous_used = self.__used_temporary_variables.copy()
        try:
            self.__expression_kind = ExpressionKind.ARITHMETIC
            target = self.__capitalize_variable_name(selection.source_name)
            indices = tuple(self.__capture_array_index(index, selection.source_width) for index in selection.indices)
            self.__check_distinct_bit_indices(indices)
            # Static/whole-register RHS operands can be direct storage names.
            # Snapshot every nonliteral value even if dynamic destination
            # addresses prevent detecting an overlap during conversion.
            with self.__runtime_bit_assignment_values(selection, target, indices, expression, operator) as values:
                snapshot = []
                for value in values:
                    if isinstance(value, int):
                        snapshot.append(value)
                    else:
                        bit = self.__add_new_temporary_variable(ValueType.BIT)
                        self.__qcx_lines.append(f'LET {bit} := {value}')
                        snapshot.append(bit)
                if len(self.__qcx_lines) + selection.width > self.MAX_LOOP_OUTPUT_LINES:
                    raise InvalidBitOperandException('Runtime bit assignment exceeds QCX output limit')
                for index, value in zip(indices, snapshot):
                    operand = target if selection.source_width == 1 else f'{target}:{index}'
                    self.__qcx_lines.append(f'LET {operand} := {value}')
        finally:
            (self.__expression_kind, self.__value, self.__value_type,
             self.__value_kind, self.__evaluate_constant) = previous
            for temporary in self.__used_temporary_variables - previous_used:
                self.__release_temporary_variable(temporary)

    def __emit_runtime_bit_range_to_range_copy(self, info: _RuntimeBitRangeToRangeCopyInfo) -> None:
        bitwise = info.binary_operator is not None
        write_lines = 6
        original_lines = 6 if bitwise else 0
        if len(self.__qcx_lines) + original_lines + write_lines > self.MAX_LOOP_OUTPUT_LINES:
            raise InvalidBitOperandException('Runtime range-to-range copy exceeds QCX output limit')
        previous = (self.__expression_kind, self.__value, self.__value_type,
                    self.__value_kind, self.__evaluate_constant)
        previous_used = self.__used_temporary_variables.copy()
        previous_temporaries = self.__declared_temporary_variables.copy()
        label = f'QASM2QCX_RANGE_TO_RANGE_COPY_{self.__array_index}'
        self.__array_index += 1
        try:
            self.__expression_kind = ExpressionKind.ARITHMETIC
            destination_positions, destination_count = self.__capture_bit_range_positions(
                info.destination.bounds, info.destination.source_width, label + '_DESTINATION', capacity=info.capacity)
            self.__check_distinct_bit_range_positions(
                destination_positions, destination_count, info.destination.source_width, label + '_DESTINATION')
            if len(self.__qcx_lines) + original_lines + write_lines > self.MAX_LOOP_OUTPUT_LINES:
                raise InvalidBitOperandException('Runtime range-to-range copy exceeds QCX output limit')
            original = (self.__snapshot_runtime_range_destination(
                info, destination_positions, destination_count, label) if bitwise else None)
            source_positions, source_count = self.__capture_bit_range_positions(
                info.source.bounds, info.source.source_width, label + '_SOURCE', capacity=info.capacity)
            self.__qcx_lines.append(f'ASSERT {source_count} == {destination_count}')
            snapshot = self.__add_temporary_int_array(info.capacity, 'RANGE_COPY_SOURCE')
            index = self.__add_new_temporary_variable(ValueType.INT)
            position = self.__add_new_temporary_variable(ValueType.INT)
            constant = (self.__constant_bit_register(info.source.source_name)
                        if info.source.source_name in self.__const_bit_register_values else None)
            words = self.__prepare_constant_bit_lookup(constant) if constant is not None else None
            self.__qcx_lines.extend([
                f'LET {index} := 0', f'@{label}_SNAPSHOT', f'LET {position} := {source_positions}:{index}',
            ])
            if words is None:
                source = self.__capitalize_variable_name(info.source.source_name)
                operand = source if info.source.source_width == 1 else f'{source}:{position}'
                self.__qcx_lines.append(f'LET {snapshot}:{index} := {operand}')
            else:
                self.__copy_prepared_constant_bit(words, position, f'{snapshot}:{index}')
            # Repeated source positions are valid. Finish all reads before
            # touching the destination, including overlapping self-copies.
            self.__qcx_lines.extend([
                f'LET {index} += 1', f'JUMPIF {label}_SNAPSHOT {index} < {source_count}',
            ])
            if bitwise:
                self.__combine_bit_range_snapshots(original, snapshot, info.operator, index, source_count, label)
                snapshot = original
            if len(self.__qcx_lines) + write_lines > self.MAX_LOOP_OUTPUT_LINES:
                raise InvalidBitOperandException('Runtime range-to-range copy exceeds QCX output limit')
            target = self.__capitalize_variable_name(info.destination.source_name)
            operand = target if info.destination.source_width == 1 else f'{target}:{position}'
            self.__qcx_lines.extend([
                f'LET {index} := 0', f'@{label}_WRITE', f'LET {position} := {destination_positions}:{index}',
                f'LET {operand} := {snapshot}:{index}', f'LET {index} += 1',
                f'JUMPIF {label}_WRITE {index} < {destination_count}',
            ])
            self.__hoist_temporary_declarations(previous_temporaries)
        finally:
            (self.__expression_kind, self.__value, self.__value_type,
             self.__value_kind, self.__evaluate_constant) = previous
            for temporary in self.__used_temporary_variables - previous_used:
                self.__release_temporary_variable(temporary)

    def __snapshot_runtime_range_destination(
            self, info: _RuntimeBitRangeToRangeCopyInfo, positions: str, count: str, label: str) -> str:
        original = self.__add_temporary_int_array(info.capacity, 'RANGE_BITWISE_ORIGINAL')
        index = self.__add_new_temporary_variable(ValueType.INT)
        position = self.__add_new_temporary_variable(ValueType.INT)
        target = self.__capitalize_variable_name(info.destination.source_name)
        operand = target if info.destination.source_width == 1 else f'{target}:{position}'
        self.__qcx_lines.extend([
            f'LET {index} := 0', f'@{label}_ORIGINAL', f'LET {position} := {positions}:{index}',
            f'LET {original}:{index} := {operand}', f'LET {index} += 1',
            f'JUMPIF {label}_ORIGINAL {index} < {count}',
        ])
        return original

    def __combine_bit_range_snapshots(
            self, original: str, source: str, operator: str, index: str, count: str, label: str) -> None:
        # Retain both operand snapshots until the complete result is computed;
        # the original-value buffer becomes the private result buffer.
        self.__qcx_lines.extend([
            f'LET {index} := 0', f'@{label}_BITWISE',
            f'LET {original}:{index} {operator} {source}:{index}',
            f'LET {index} += 1', f'JUMPIF {label}_BITWISE {index} < {count}',
        ])

    def __snapshot_bit_range_destination(
            self, info: _RuntimeBitRangeCopyInfo, label: str,
            indices: tuple[int | str, ...] | None = None) -> str:
        original = self.__add_temporary_int_array(info.width, 'RANGE_BITWISE_ORIGINAL')
        target = self.__capitalize_variable_name(info.destination_name)
        destinations = info.destination_indices if indices is None else indices
        if destinations is None:
            index = self.__add_new_temporary_variable(ValueType.INT)
            operand = target if info.destination_width == 1 else f'{target}:{index}'
            self.__qcx_lines.extend([
                f'LET {index} := 0', f'@{label}_ORIGINAL', f'LET {original}:{index} := {operand}',
                f'LET {index} += 1', f'JUMPIF {label}_ORIGINAL {index} < {info.width}',
            ])
        else:
            for offset, position in enumerate(destinations):
                operand = target if info.destination_width == 1 else f'{target}:{position}'
                self.__qcx_lines.append(f'LET {original}:{offset} := {operand}')
        return original

    def __emit_runtime_bit_range_copy(self, info: _RuntimeBitRangeCopyInfo) -> None:
        bitwise = info.binary_operator is not None
        if bitwise and not 1 <= info.width <= self.MAX_RUNTIME_BIT_RANGE_POSITIONS:
            raise InvalidBitOperandException('Runtime range bitwise snapshot exceeds storage limit')
        runtime_destination = info.runtime_destination
        write_lines = (5 if runtime_destination is None and info.destination_indices is None else info.width)
        comparisons = info.width * (info.width - 1) // 2 if runtime_destination is not None else 0
        original_lines = (5 if runtime_destination is None and info.destination_indices is None
                          else info.width) if bitwise else 0
        if len(self.__qcx_lines) + comparisons + max(8, write_lines + original_lines) > self.MAX_LOOP_OUTPUT_LINES:
            raise InvalidBitOperandException('Runtime range copy exceeds QCX output limit')
        previous = (self.__expression_kind, self.__value, self.__value_type,
                    self.__value_kind, self.__evaluate_constant)
        previous_used = self.__used_temporary_variables.copy()
        previous_temporaries = self.__declared_temporary_variables.copy()
        label = f'QASM2QCX_RANGE_COPY_{self.__array_index}'
        self.__array_index += 1
        try:
            self.__expression_kind = ExpressionKind.ARITHMETIC
            destinations = info.destination_indices
            if runtime_destination is not None:
                # Reserve all captured addresses throughout source evaluation.
                # Check destination failures before evaluating source bounds.
                destinations = tuple(self.__capture_array_index(expression, info.destination_width)
                                     for expression in runtime_destination.indices)
                if len(self.__qcx_lines) + comparisons + original_lines + write_lines > self.MAX_LOOP_OUTPUT_LINES:
                    raise InvalidBitOperandException('Runtime range copy exceeds QCX output limit')
                self.__check_distinct_bit_indices(destinations)
            # Resolve and validate destination addresses before reading their
            # original values. Keep those values reserved throughout the RHS.
            original = (self.__snapshot_bit_range_destination(info, label, destinations) if bitwise else None)
            selection = info.selection
            # Bound private storage by the known destination width. Excess
            # entries, empty ranges and bad indices fail before any source read.
            positions, count = self.__capture_bit_range_positions(
                selection.bounds, selection.source_width, label, capacity=info.width)
            self.__qcx_lines.append(f'ASSERT {count} == {info.width}')
            snapshot = self.__add_temporary_int_array(info.width, 'RANGE_COPY_SOURCE')
            index = self.__add_new_temporary_variable(ValueType.INT)
            position = self.__add_new_temporary_variable(ValueType.INT)
            constant = (self.__constant_bit_register(selection.source_name)
                        if selection.source_name in self.__const_bit_register_values else None)
            words = self.__prepare_constant_bit_lookup(constant) if constant is not None else None
            self.__qcx_lines.extend([
                f'LET {index} := 0', f'@{label}_SNAPSHOT',
                f'LET {position} := {positions}:{index}',
            ])
            if words is None:
                source = self.__capitalize_variable_name(selection.source_name)
                operand = source if selection.source_width == 1 else f'{source}:{position}'
                self.__qcx_lines.append(f'LET {snapshot}:{index} := {operand}')
            else:
                self.__copy_prepared_constant_bit(words, position, f'{snapshot}:{index}')
            # Reads may repeat a normalized source position. Snapshot the
            # complete selection in its original order before writing, so
            # self-copies and overlapping source/destination storage are safe.
            self.__qcx_lines.extend([
                f'LET {index} += 1', f'JUMPIF {label}_SNAPSHOT {index} < {count}',
            ])
            if bitwise:
                self.__combine_bit_range_snapshots(original, snapshot, info.operator, index, count, label)
                snapshot = original
            if len(self.__qcx_lines) + write_lines > self.MAX_LOOP_OUTPUT_LINES:
                raise InvalidBitOperandException('Runtime range copy exceeds QCX output limit')
            target = self.__capitalize_variable_name(info.destination_name)
            if destinations is None:
                operand = target if info.destination_width == 1 else f'{target}:{index}'
                self.__qcx_lines.extend([
                    f'LET {index} := 0', f'@{label}_WRITE',
                    f'LET {operand} := {snapshot}:{index}',
                    f'LET {index} += 1', f'JUMPIF {label}_WRITE {index} < {count}',
                ])
            else:
                # Both static positions and captured set indices can be used
                # directly. Singleton slices retain their containing storage.
                for offset, destination in enumerate(destinations):
                    operand = target if info.destination_width == 1 else f'{target}:{destination}'
                    self.__qcx_lines.append(f'LET {operand} := {snapshot}:{offset}')
            self.__hoist_temporary_declarations(previous_temporaries)
        finally:
            (self.__expression_kind, self.__value, self.__value_type,
             self.__value_kind, self.__evaluate_constant) = previous
            for temporary in self.__used_temporary_variables - previous_used:
                self.__release_temporary_variable(temporary)

    def __emit_runtime_bit_range_assignment(
            self, info: _RuntimeBitRangeAssignmentInfo, expression: ast.Expression) -> None:
        selection, width = info.selection, info.rhs_width
        comparisons = width * (width - 1) // 2
        if len(self.__qcx_lines) + comparisons + 2 * width + 1 > self.MAX_LOOP_OUTPUT_LINES:
            raise InvalidBitOperandException('Runtime range assignment exceeds QCX output limit')
        previous = (self.__expression_kind, self.__value, self.__value_type,
                    self.__value_kind, self.__evaluate_constant)
        previous_used = self.__used_temporary_variables.copy()
        label = f'QASM2QCX_BIT_RANGE_ASSIGNMENT_{self.__array_index}'
        self.__array_index += 1
        try:
            self.__expression_kind = ExpressionKind.ARITHMETIC
            # A successful assignment has exactly the known RHS width. Bound
            # the table by that width; excess entries fail before any RHS read.
            positions, count = self.__capture_bit_range_positions(
                selection.bounds, selection.source_width, label, capacity=width)
            self.__qcx_lines.append(f'ASSERT {count} == {width}')
            indices = []
            for position in range(width):
                index = self.__add_new_temporary_variable(ValueType.INT)
                indices.append(index)
                self.__qcx_lines.append(f'LET {index} := {positions}:{position}')
            if len(self.__qcx_lines) + comparisons + width > self.MAX_LOOP_OUTPUT_LINES:
                raise InvalidBitOperandException('Runtime range assignment exceeds QCX output limit')
            self.__check_distinct_bit_indices(tuple(indices))
            target = self.__capitalize_variable_name(selection.source_name)
            with self.__runtime_bit_assignment_values(
                    selection, target, tuple(indices), expression, info.binary_operator) as values:
                snapshot = []
                for value in values:
                    if isinstance(value, int):
                        snapshot.append(value)
                    else:
                        bit = self.__add_new_temporary_variable(ValueType.BIT)
                        self.__qcx_lines.append(f'LET {bit} := {value}')
                        snapshot.append(bit)
                if len(self.__qcx_lines) + width > self.MAX_LOOP_OUTPUT_LINES:
                    raise InvalidBitOperandException('Runtime range assignment exceeds QCX output limit')
                for index, value in zip(indices, snapshot):
                    operand = target if selection.source_width == 1 else f'{target}:{index}'
                    self.__qcx_lines.append(f'LET {operand} := {value}')
        finally:
            (self.__expression_kind, self.__value, self.__value_type,
             self.__value_kind, self.__evaluate_constant) = previous
            for temporary in self.__used_temporary_variables - previous_used:
                self.__release_temporary_variable(temporary)

    def __check_distinct_bit_range_positions(self, positions: str, count: str, width: int, label: str) -> None:
        index, first, stride, divisor, remainder, period = (
            self.__add_new_temporary_variable(ValueType.INT) for _ in range(6))
        # The normalized arithmetic progression repeats after
        # width/gcd(width, step mod width) entries. Derive the stride from
        # its first two captured positions without reevaluating the raw step.
        # This avoids a quadratic duplicate scan for large runtime selections.
        self.__qcx_lines.extend([
            f'JUMPIF {label}_DISTINCT_END {count} <= 1',
            f'LET {index} := 0', f'LET {first} := {positions}:{index}',
            f'LET {index} := 1', f'LET {stride} := {positions}:{index}', f'LET {stride} -= {first}',
            f'JUMPIF {label}_DISTINCT_NONNEGATIVE {stride} >= 0', f'LET {stride} += {width}',
            f'@{label}_DISTINCT_NONNEGATIVE', f'LET {divisor} := {width}', f'@{label}_DISTINCT_GCD',
            f'JUMPIF {label}_DISTINCT_PERIOD {stride} == 0',
            f'LET {remainder} := {divisor}', f'LET {remainder} %= {stride}',
            f'LET {divisor} := {stride}', f'LET {stride} := {remainder}', f'JUMP {label}_DISTINCT_GCD',
            f'@{label}_DISTINCT_PERIOD', f'LET {period} := {width}', f'LET {period} /= {divisor}',
            f'ASSERT {count} <= {period}', f'@{label}_DISTINCT_END',
        ])

    def __capture_runtime_range_shift_distance(
            self, expression: ast.Expression, capacity: int, count: str, label: str) -> str:
        value, value_type, value_kind = self.__condition_operand(expression)
        captured = self.__add_new_temporary_variable(value_type)
        if value_kind == ValueKind.LITERAL:
            # Large nonnegative counts always produce zeros. Representable
            # negative literals fail only when reached. Neither case needs
            # its original magnitude in the generated QCX.
            value = -1 if int(value) < 0 else min(int(value), capacity)
        self.__qcx_lines.append(f'LET {captured} := {value}')
        if value_type == ValueType.INT:
            self.__qcx_lines.append(f'ASSERT {captured} >= 0')
            limit = count
        else:
            limit = self.__add_new_temporary_variable(ValueType.UINT)
            self.__qcx_lines.append(f'LET {limit} := :UINT:{count}')
        self.__qcx_lines.append(f'JUMPIF {label}_SHIFT_END {captured} >= {limit}')
        distance = self.__add_new_temporary_variable(ValueType.INT)
        operand = f':INT:{captured}' if value_type == ValueType.UINT else captured
        self.__qcx_lines.append(f'LET {distance} := {operand}')
        return distance

    def __runtime_bit_range_shift_result(
            self, info: _RuntimeBitRangeShiftAssignmentInfo, positions: str, count: str,
            expression: ast.Expression, label: str) -> str:
        original = self.__add_temporary_int_array(info.capacity, 'RANGE_SHIFT_SOURCE')
        result = self.__add_temporary_int_array(info.capacity, 'RANGE_SHIFT_RESULT')
        index = self.__add_new_temporary_variable(ValueType.INT)
        position = self.__add_new_temporary_variable(ValueType.INT)
        target = self.__capitalize_variable_name(info.selection.source_name)
        operand = target if info.selection.source_width == 1 else f'{target}:{position}'
        # Snapshot every selected bit and initialize the complete zero result
        # before evaluating a possibly self-referential shift distance.
        self.__qcx_lines.extend([
            f'LET {index} := 0', f'@{label}_SNAPSHOT', f'JUMPIF {label}_SNAPSHOT_END {index} >= {count}',
            f'LET {position} := {positions}:{index}', f'LET {original}:{index} := {operand}',
            f'LET {result}:{index} := 0', f'LET {index} += 1', f'JUMP {label}_SNAPSHOT',
            f'@{label}_SNAPSHOT_END',
        ])
        distance = self.__capture_runtime_range_shift_distance(expression, info.capacity, count, label)
        source_index = self.__add_new_temporary_variable(ValueType.INT)
        target_index = self.__add_new_temporary_variable(ValueType.INT)
        # Advance two bounded indices instead of adding a distance to each
        # candidate. Each index increments at most to count, which fits INT.
        self.__qcx_lines.extend([
            f'LET {source_index} := {0 if info.operator == "<<=" else distance}',
            f'LET {target_index} := {distance if info.operator == "<<=" else 0}',
            f'@{label}_SHIFT_COPY', f'JUMPIF {label}_SHIFT_END {source_index} >= {count}',
            f'JUMPIF {label}_SHIFT_END {target_index} >= {count}',
            f'LET {result}:{target_index} := {original}:{source_index}',
            f'LET {source_index} += 1', f'LET {target_index} += 1', f'JUMP {label}_SHIFT_COPY',
            f'@{label}_SHIFT_END',
        ])
        return result

    def __emit_runtime_bit_range_shift_assignment(
            self, info: _RuntimeBitRangeShiftAssignmentInfo, expression: ast.Expression) -> None:
        if len(self.__qcx_lines) + 8 > self.MAX_LOOP_OUTPUT_LINES:
            raise InvalidBitOperandException('Runtime range shift exceeds QCX output limit')
        previous = (self.__expression_kind, self.__value, self.__value_type,
                    self.__value_kind, self.__evaluate_constant)
        previous_used = self.__used_temporary_variables.copy()
        label = f'QASM2QCX_RANGE_SHIFT_{self.__array_index}'
        self.__array_index += 1
        try:
            self.__expression_kind = ExpressionKind.ARITHMETIC
            positions, count = self.__capture_bit_range_positions(
                info.selection.bounds, info.selection.source_width, label, capacity=info.capacity)
            self.__check_distinct_bit_range_positions(positions, count, info.selection.source_width, label)
            result = self.__runtime_bit_range_shift_result(info, positions, count, expression, label)
            index = self.__add_new_temporary_variable(ValueType.INT)
            position = self.__add_new_temporary_variable(ValueType.INT)
            if len(self.__qcx_lines) + 8 > self.MAX_LOOP_OUTPUT_LINES:
                raise InvalidBitOperandException('Runtime range shift exceeds QCX output limit')
            target = self.__capitalize_variable_name(info.selection.source_name)
            operand = target if info.selection.source_width == 1 else f'{target}:{position}'
            self.__qcx_lines.extend([
                f'LET {index} := 0', f'@{label}_WRITE', f'JUMPIF {label}_END {index} >= {count}',
                f'LET {position} := {positions}:{index}', f'LET {operand} := {result}:{index}',
                f'LET {index} += 1', f'JUMP {label}_WRITE', f'@{label}_END',
            ])
        finally:
            (self.__expression_kind, self.__value, self.__value_type,
             self.__value_kind, self.__evaluate_constant) = previous
            for temporary in self.__used_temporary_variables - previous_used:
                self.__release_temporary_variable(temporary)

    def __emit_runtime_bit_range_measurement(self, info: _RuntimeBitRangeMeasurementInfo) -> None:
        # Two lines per measured qubit, one count assertion, and a write loop.
        remaining_lines = 2 * info.width + 9
        if len(self.__qcx_lines) + remaining_lines > self.MAX_LOOP_OUTPUT_LINES:
            raise InvalidBitOperandException('Runtime range measurement exceeds QCX output limit')
        previous = (self.__expression_kind, self.__value, self.__value_type,
                    self.__value_kind, self.__evaluate_constant)
        previous_used = self.__used_temporary_variables.copy()
        label = f'QASM2QCX_RANGE_MEASUREMENT_{self.__array_index}'
        self.__array_index += 1
        try:
            self.__expression_kind = ExpressionKind.ARITHMETIC
            positions, count = self.__capture_bit_range_positions(
                info.destination.bounds, info.destination.source_width, label, capacity=info.width)
            self.__qcx_lines.append(f'ASSERT {count} == {info.width}')
            self.__check_distinct_bit_range_positions(positions, count, info.destination.source_width, label)
            # The private position table remains reserved and unchanged across
            # every M and destination write. All addresses are already checked.
            outcomes = self.__add_temporary_int_array(info.width, 'RANGE_MEASUREMENT_OUTCOMES')
            index = self.__add_new_temporary_variable(ValueType.INT)
            position = self.__add_new_temporary_variable(ValueType.INT)
            if len(self.__qcx_lines) + 2 * info.width + 8 > self.MAX_LOOP_OUTPUT_LINES:
                raise InvalidBitOperandException('Runtime range measurement exceeds QCX output limit')
            for entry, qubit in enumerate(info.qubit_indices):
                self.__qcx_lines.extend([
                    f'M {self.__to_qubit_index(info.qubit_source_name, qubit)}',
                    f'LET {outcomes}:{entry} := :OUTCOME',
                ])
            target = self.__capitalize_variable_name(info.destination.source_name)
            operand = target if info.destination.source_width == 1 else f'{target}:{position}'
            self.__qcx_lines.extend([
                f'LET {index} := 0', f'@{label}_WRITE', f'JUMPIF {label}_END {index} >= {count}',
                f'LET {position} := {positions}:{index}', f'LET {operand} := {outcomes}:{index}',
                f'LET {index} += 1', f'JUMP {label}_WRITE', f'@{label}_END',
            ])
        finally:
            (self.__expression_kind, self.__value, self.__value_type,
             self.__value_kind, self.__evaluate_constant) = previous
            for temporary in self.__used_temporary_variables - previous_used:
                self.__release_temporary_variable(temporary)

    def __emit_runtime_bit_set_measurement(self, info: _RuntimeBitSetMeasurementInfo) -> None:
        selection = info.destination
        comparisons = selection.width * (selection.width - 1) // 2
        remaining_lines = comparisons + 3 * selection.width
        if len(self.__qcx_lines) + remaining_lines > self.MAX_LOOP_OUTPUT_LINES:
            raise InvalidBitOperandException('Runtime bit measurement exceeds QCX output limit')
        previous = (self.__expression_kind, self.__value, self.__value_type,
                    self.__value_kind, self.__evaluate_constant)
        previous_used = self.__used_temporary_variables.copy()
        try:
            self.__expression_kind = ExpressionKind.ARITHMETIC
            target = self.__capitalize_variable_name(selection.source_name)
            # Capture every address while all classical bits still hold their
            # original values. Invalid destinations must fail before collapse.
            indices = tuple(self.__capture_array_index(index, selection.source_width) for index in selection.indices)
            outcomes = tuple(self.__add_new_temporary_variable(ValueType.BIT) for _ in indices)
            # Include declarations and capture code in the full budget check,
            # before emitting any measurement or destination write.
            if len(self.__qcx_lines) + remaining_lines > self.MAX_LOOP_OUTPUT_LINES:
                raise InvalidBitOperandException('Runtime bit measurement exceeds QCX output limit')
            self.__check_distinct_bit_indices(indices)
            for qubit_index, outcome in zip(info.qubit_indices, outcomes):
                self.__qcx_lines.extend([
                    f'M {self.__to_qubit_index(info.qubit_source_name, qubit_index)}',
                    f'LET {outcome} := :OUTCOME',
                ])
            for index, outcome in zip(indices, outcomes):
                operand = target if selection.source_width == 1 else f'{target}:{index}'
                self.__qcx_lines.append(f'LET {operand} := {outcome}')
        finally:
            (self.__expression_kind, self.__value, self.__value_type,
             self.__value_kind, self.__evaluate_constant) = previous
            for temporary in self.__used_temporary_variables - previous_used:
                self.__release_temporary_variable(temporary)

    def __check_distinct_bit_indices(self, indices: tuple[str, ...]) -> None:
        for position, index in enumerate(indices):
            for previous in indices[:position]:
                self.__qcx_lines.append(f'ASSERT {index} \\= {previous}')

    def __emit_angle_bit_assignment(
            self, selection: _AngleBitSelection, expression: ast.Expression,
            operator: str | None = None) -> None:
        previous = (self.__expression_kind, self.__value, self.__value_type, self.__value_kind)
        previous_used = self.__used_temporary_variables.copy()
        try:
            self.__expression_kind = ExpressionKind.ARITHMETIC
            if isinstance(selection.source, ast.IndexExpression):
                # Capture and check the destination element once. Keep its
                # index reserved while the RHS evaluates its own accesses.
                target, _ = self.__array_access(selection.source)
            else:
                target = self.__capitalize_variable_name(selection.source.name)
                if self.__type_of(target) != ValueType.ANGLE:
                    raise InvalidBitOperandException('An angle bit selection target must have angle type')
            indices = self.__capture_angle_bit_indices(selection)
            if isinstance(selection, _RuntimeAngleBitSetSelection):
                self.__check_distinct_angle_bit_indices(indices)
            result = self.__add_new_temporary_variable(
                ValueType.UINT, _UnsignedIntegerType(selection.angle_type.width, True))
            self.__qcx_lines.append(f'LET {result} := {target}')
            with self.__angle_bit_assignment_values(selection, result, expression, operator, indices=indices) as bits:
                if isinstance(selection, _StaticAngleBitSelection):
                    self.__emit_static_angle_bit_update(result, selection, bits)
                else:
                    for index, value in zip(indices, bits):
                        self.__emit_runtime_angle_bit_update(result, selection.angle_type, index, value)
            # Commit once, after every RHS bit has been captured and all
            # calculations have succeeded, including overlapping self-copies.
            self.__qcx_lines.append(f'LET {target} := {result}')
        finally:
            self.__expression_kind, self.__value, self.__value_type, self.__value_kind = previous
            for temporary in self.__used_temporary_variables - previous_used:
                self.__release_temporary_variable(temporary)

    def __check_distinct_angle_bit_indices(self, indices: tuple[int | str, ...]) -> None:
        # Every position was bounds-checked before this pass. A native UINT
        # bitmap detects normalized duplicates in linear time, before the RHS.
        previous_used = self.__used_temporary_variables.copy()
        try:
            seen = self.__add_new_temporary_variable(ValueType.UINT)
            mask = self.__add_new_temporary_variable(ValueType.UINT)
            overlap = self.__add_new_temporary_variable(ValueType.UINT)
            self.__qcx_lines.append(f'LET {seen} := 0')
            for index in indices:
                self.__qcx_lines.extend([
                    f'LET {mask} := 1', f'LET {mask} <<= {index}',
                    f'LET {overlap} := {seen}', f'LET {overlap} &= {mask}',
                    f'ASSERT {overlap} == 0', f'LET {seen} |= {mask}',
                ])
        finally:
            for temporary in self.__used_temporary_variables - previous_used:
                self.__release_temporary_variable(temporary)

    def __emit_runtime_angle_bit_update(
            self, result: str, angle_type: _AngleType, index: str, value: str | int) -> None:
        mask = self.__add_new_temporary_variable(ValueType.UINT)
        scratch = self.__add_new_temporary_variable(ValueType.UINT)
        operand = str(value) if isinstance(value, int) else f':UINT:{value}'
        self.__qcx_lines.extend([
            f'LET {mask} := 1', f'LET {mask} <<= {index}',
            f'LET {mask} ^= {angle_type.mask}', f'LET {result} &= {mask}',
            f'LET {scratch} := {operand}', f'LET {scratch} <<= {index}',
            f'LET {result} |= {scratch}',
        ])
        self.__release_temporary_variable(mask)
        self.__release_temporary_variable(scratch)

    def __emit_static_angle_bit_update(
            self, result: str, selection: _StaticAngleBitSelection, bits: tuple[str | int, ...]) -> None:
        selected_mask = sum(1 << index for index in selection.indices)
        self.__qcx_lines.append(f'LET {result} &= {selection.angle_type.mask ^ selected_mask}')
        literal = sum(value << index for index, value in zip(selection.indices, bits)
                      if isinstance(value, int))
        if literal:
            self.__qcx_lines.append(f'LET {result} |= {literal}')
        scratch = None
        for index, value in zip(selection.indices, bits):
            if isinstance(value, int):
                continue
            if scratch is None:
                scratch = self.__add_new_temporary_variable(ValueType.UINT)
            self.__qcx_lines.append(f'LET {scratch} := :UINT:{value}')
            if index:
                self.__qcx_lines.append(f'LET {scratch} <<= {index}')
            self.__qcx_lines.append(f'LET {result} |= {scratch}')

    @staticmethod
    def __validate_array_assignment_operator(element_type: ValueType, operator: str) -> None:
        if element_type == ValueType.ANGLE and operator not in ('=', '+=', '-=', '*=', '/=', '&=', '|=', '^=', '<<=', '>>='):
            raise UnsupportedOpenQASMError(f'angle array assignment operator {operator}')
        if element_type == ValueType.BOOL and operator != '=':
            raise UnsupportedOpenQASMError(f'Boolean array assignment operator {operator}')
        if element_type == ValueType.FLOAT and operator not in ('=', '+=', '-=', '*=', '/='):
            raise UnsupportedOpenQASMError(f'floating-point array assignment operator {operator}')
        if element_type == ValueType.COMPLEX and operator not in ('=', '+=', '-=', '*=', '/='):
            raise UnsupportedOpenQASMError(f'complex array assignment operator {operator}')

    def __emit_angle_array_assignment(
            self, statement: ast.ClassicalAssignment, target: str, angle_type: _AngleType) -> None:
        # The destination index has already been captured and checked. Bind
        # only the synthetic LHS read to its old value; explicit RHS element
        # reads retain their own evaluation and bounds checks.
        lhs = ast.IndexExpression(statement.lvalue.name, statement.lvalue.indices[0])
        expression = self.__angle_assignment_expression(statement, lhs)
        previous_used = self.__used_temporary_variables.copy()
        try:
            if statement.op.name != '=':
                snapshot = self.__add_new_temporary_variable(ValueType.UINT, _UnsignedIntegerType(angle_type.width, True))
                self.__qcx_lines.append(f'LET {snapshot} := {target}')
                self.__captured_angle_array_values[id(lhs)] = (lhs, snapshot)
            self.__emit_angle_copy(target, angle_type, expression)
        finally:
            self.__captured_angle_array_values.pop(id(lhs), None)
            for temporary in self.__used_temporary_variables - previous_used:
                self.__release_temporary_variable(temporary)

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
