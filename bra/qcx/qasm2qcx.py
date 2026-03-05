import sys
import itertools
import math
import enum

import openqasm3.parser
import openqasm3.ast as ast
import openqasm3.visitor as visitor

ValueType = enum.Enum('ValueType', [('INT', 1), ('FLOAT', 2), ('COMPLEX', 3)])
ValueKind = enum.Enum('ValueKind', [('LITERAL', 1), ('TEMPORARY', 2), ('LVALUE', 3)])
# CONST_ARITHMETIC => ARITHMETIC
ExpressionKind = enum.Enum('ExpressionKind', [('ARITHMETIC', 1), ('CONST_ARITHMETIC', 2), ('CONDITIONAL', 3)])

class WrongBroadcastingException(Exception):
    def __str__(self):
        return 'Wrong broadcasting; see https://openqasm.com/language/gates.html#broadcasting'

class WrongClassicalDeclarationException(Exception):
    def __init__(self, variable_name: str):
        self.__variable_name = variable_name

    def __str__(self):
        return f'Wrong declaration of classical variable: {self.__variable_name}'

class UninitializedValueException(Exception):
    def __str__(self):
        return 'Wrong value type'

class NoImplicitCastException(Exception):
    def __str__(self):
        return 'No implicit cast'

class NoVariableNameException(Exception):
    def __init__(self, variable_name: str):
        self.__variable_name = variable_name

    def __str__(self):
        return f'No variable {self.__variable_name} is found'

class WrongParameterTypeException(Exception):
    def __init__(self, value: str):
        self.__value = value

    def __str__(self):
        return f'The type of variable {self.__value} is wrong'

class NoConstantExpressionException(Exception):
    def __str__(self):
        return 'No constant expression'

class WrongConstantVariableException(Exception):
    def __str__(self):
        return 'Wrong constant variable'

class QASM2QCXConverter(visitor.QASMVisitor):
    # TODO: implement gphase(\gamma)
    default_gates_qcx_map: dict[str, str] = {'U': 'U3'}

    # TODO: implement cu(\theta, \phi, \lambda, \gamma) and id
    stdgates_qcx_map: dict[str, str] = {
            'p': 'U1', 'phase': 'U1', 'u1': 'U1', 'u2': 'U2', 'u3': 'U3',
            'x': 'X', 'y': 'Y', 'z': 'Z', 'h': 'H', 's': 'S', 'sdg': 'S+', 't': 'T', 'tdg': 'T+',
            'sx': 'SX', 'rx': 'EX', 'ry': 'EY', 'rz': 'EZ',
            'cx': 'CX', 'CX': 'CX', 'cy': 'CY', 'cz': 'CZ', 'cp': 'CU1', 'cphase': 'CU1',
            'crx': 'CEX', 'cry': 'CEY', 'crz': 'CEZ', 'ch': 'CH', 'swap': 'SWAP', 'ccx': 'CCX', 'cswap': 'CSWAP'}

    def __init__(self, qasm_ast_root: ast.Program) -> None:
        self.__value: str | int | float | complex | None = None
        self.__value_type: ValueType | None = None
        self.__value_kind: ValueKind | None = None
        self.__expression_kind: ExpressionKind | None = None
        self.__declared_temporary_variables: set[str] = set()
        self.__used_temporary_variables: set[str] = set()

        self.__int_variable_name_size_map: dict[str, int] = {}
        self.__float_variable_name_size_map: dict[str, int] = {}
        self.__complex_variable_name_size_map: dict[str, int] = {}

        self.__const_int_variable_name_values_map: dict[str, [int]] = {}
        self.__const_float_variable_name_values_map: dict[str, [float]] = {}
        self.__const_complex_variable_name_values_map: dict[str, [complex]] = {}

        self.__is_stdgates_included: bool = False
        self.__quantum_registers: dict[str, int] = {}

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
        if identifier_name in self.__int_variable_name_size_map:
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

    def __add_new_temporary_variable(self, value_type: ValueType) -> str:
        match value_type:
            case ValueType.INT:
                type_str = 'INT'
            case ValueType.FLOAT:
                type_str = 'REAL'
            case ValueType.COMPLEX:
                type_str = 'COMPLEX'

        temporary_variable_index: int = 0
        temporary_variable: str = f'__{type_str}{temporary_variable_index}'
        while temporary_variable in self.__used_temporary_variables:
            temporary_variable_index += 1
            temporary_variable = f'__{type_str}{temporary_variable_index}'

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

        if expression.name in self.__const_int_variable_name_values_map:
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
                return

            self.__value = self.__capitalize_variable_name(expression.name)
            self.__value_type = self.__type_of(self.__value)
            self.__value_kind = ValueKind.LVALUE

    def visit_UnaryExpression(self, expression: ast.UnaryExpression) -> None:
        if self.__expression_kind is None:
            return

        self.visit(expression.expression)
        if self.__value is None or self.__value_type is None or self.__value_kind is None:
            raise UninitializedValueException

        if self.__value_kind == ValueKind.LITERAL:
            if expression.op == ast.UnaryOperator['~']:
                pass # TODO
            elif expression.op == ast.UnaryOperator['!']:
                pass # TODO
            elif expression.op == ast.UnaryOperator['-']:
                self.__value = -self.__value
        else:
            match self.__expression_kind:
                # lhs/rhs_value_kind != ValueKind.LITERAL => no CONST_ARITHMETIC
                case ExpressionKind.CONST_ARITHMETIC:
                    raise # TODO

                case ExpressionKind.ARITHMETIC:
                    if self.__value_kind == ValueKind.LVALUE:
                        temporary_variable: str = self.__add_new_temporary_variable(self.__value_type)
                        self.__value = tempoary_variable
                        self.__value_kind = ValueKind.TEMPORARY

                    if expression.op == ast.UnaryOperator['~']:
                        pass # TODO
                    elif expression.op == ast.UnaryOperator['!']:
                        pass # TODO
                    elif expression.op == ast.UnaryOperator['-']:
                        match self.__value_type:
                            case ValueType.INT:
                                self.__qcx_lines.append(f'LET {self.__value} *= -1')
                            case ValueType.FLOAT:
                                self.__qcx_lines.append(f'LET {self.__value} *= -1.0')
                            case ValueType.COMPLEX:
                                self.__qcx_lines.append(f'LET {self.__value} *= :COMPLEX:-1.0')
                            case _:
                                pass
                    else:
                        raise # TODO

    def visit_BinaryExpression(self, expression: ast.UnaryExpression) -> None:
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

        if lhs_value_kind == ValueKind.LITERAL and rhs_value_kind == ValueKind.LITERAL:
            if lhs_value_type in [ValueType.INT, ValueType.FLOAT, ValueType.COMPLEX] and rhs_value_type in [ValueType.INT, ValueType.FLOAT, ValueType.COMPLEX]:
                # TODO: % and **, and other operations
                if expression.op == ast.BinaryOperator['+']:
                    self.__value = lhs_value + rhs_value
                elif expression.op == ast.BinaryOperator['-']:
                    self.__value = lhs_value - rhs_value
                elif expression.op == ast.BinaryOperator['*']:
                    self.__value = lhs_value * rhs_value
                elif expression.op == ast.BinaryOperator['/']:
                    self.__value = lhs_value / rhs_value
                else:
                    raise # TODO

                if lhs_value_type == ValueType.COMPLEX or rhs_value_type == ValueType.COMPLEX:
                    self.__value_type = ValueType.COMPLEX
                elif lhs_value_type == ValueType.FLOAT or rhs_value_type == ValueType.FLOAT:
                    self.__value_type = ValueType.FLOAT

        elif lhs_value_kind == ValueKind.LITERAL:
            match self.__expression_kind:
                # lhs/rhs_value_kind != ValueKind.LITERAL => no CONST_ARITHMETIC
                case ExpressionKind.CONST_ARITHMETIC:
                    raise # TODO

                case ExpressionKind.ARITHMETIC:
                    if lhs_value_type in [ValueType.INT, ValueType.FLOAT, ValueType.COMPLEX] and rhs_value_type in [ValueType.INT, ValueType.FLOAT, ValueType.COMPLEX]:
                        # TODO: % and **, and other operations
                        if expression.op == ast.BinaryOperator['+']:
                            operator = '+='
                        elif expression.op == ast.BinaryOperator['-']:
                            operator = '-='
                        elif expression.op == ast.BinaryOperator['*']:
                            operator = '*='
                        elif expression.op == ast.BinaryOperator['/']:
                            operator = '/='
                        else:
                            operator = ''

                        if rhs_value_type == ValueType.INT:
                            lhs = int(lhs_value)
                        elif rhs_value_type == ValueType.FLOAT:
                            lhs = float(lhs_value)
                        else: # rhs_value_type == ValueType.COMPLEX
                            lhs = complex(lhs_value)

                        if rhs_value_kind == ValueKind.TEMPORARY:
                            self.__qcx_lines.append(f'LET {rhs_value} {operator} {lhs}')
                            self.__value = rhs_value
                            self.__value_type = rhs_value_type
                            self.__value_kind = ValueKind.TEMPORARY
                        else: # rhs_value_kind == ValueKind.LVALUE
                            temporary_variable: str = self.__add_new_temporary_variable(rhs_value_type)
                            self.__qcx_lines.append(f'LET {temporary_variable} := {lhs}')
                            self.__qcx_lines.append(f'LET {temporary_variable} {operator} {rhs_value}')
                            self.__value = temporary_variable
                            self.__value_type = rhs_value_type
                            self.__value_kind = ValueKind.TEMPORARY

                case ExpressionKind.CONDITIONAL:
                    pass # TODO

        elif rhs_value_kind == ValueKind.LITERAL:
            match self.__expression_kind:
                # lhs/rhs_value_kind != ValueKind.LITERAL => no CONST_ARITHMETIC
                case ExpressionKind.CONST_ARITHMETIC:
                    raise # TODO

                case ExpressionKind.ARITHMETIC:
                    if lhs_value_type in [ValueType.INT, ValueType.FLOAT, ValueType.COMPLEX] and rhs_value_type in [ValueType.INT, ValueType.FLOAT, ValueType.COMPLEX]:
                        # TODO: % and **, and other operations
                        if expression.op == ast.BinaryOperator['+']:
                            operator = '+='
                        elif expression.op == ast.BinaryOperator['-']:
                            operator = '-='
                        elif expression.op == ast.BinaryOperator['*']:
                            operator = '*='
                        elif expression.op == ast.BinaryOperator['/']:
                            operator = '/='
                        else:
                            operator = ''

                        if lhs_value_type == ValueType.INT:
                            rhs = int(rhs_value)
                        elif lhs_value_type == ValueType.FLOAT:
                            rhs = float(rhs_value)
                        else: # lhs_value_type == ValueType.COMPLEX
                            rhs = complex(rhs_value)

                        if lhs_value_kind == ValueKind.TEMPORARY:
                            self.__qcx_lines.append(f'LET {lhs_value} {operator} {rhs}')
                            self.__value = lhs_value
                            self.__value_type = lhs_value_type
                            self.__value_kind = ValueKind.TEMPORARY
                        else: # lhs_value_kind == ValueKind.LVALUE
                            temporary_variable: str = self.__add_new_temporary_variable(lhs_value_type)
                            self.__qcx_lines.append(f'LET {temporary_variable} := {rhs}')
                            self.__qcx_lines.append(f'LET {temporary_variable} {operator} {lhs_value}')
                            self.__value = temporary_variable
                            self.__value_type = lhs_value_type
                            self.__value_kind = ValueKind.TEMPORARY

                case ExpressionKind.CONDITIONAL:
                    pass # TODO

        else:
            match self.__expression_kind:
                # lhs/rhs_value_kind != ValueKind.LITERAL => no CONST_ARITHMETIC
                case ExpressionKind.CONST_ARITHMETIC:
                    raise # TODO

                case ExpressionKind.ARITHMETIC:
                    if lhs_value_type in [ValueType.INT, ValueType.FLOAT, ValueType.COMPLEX] and rhs_value_type in [ValueType.INT, ValueType.FLOAT, ValueType.COMPLEX]:
                        if lhs_value_type != rhs_value_type:
                            raise NoImplicitCastException

                        # TODO: % and **, and other operations
                        if expression.op == ast.BinaryOperator['+']:
                            operator = '+='
                        elif expression.op == ast.BinaryOperator['-']:
                            operator = '-='
                        elif expression.op == ast.BinaryOperator['*']:
                            operator = '*='
                        elif expression.op == ast.BinaryOperator['/']:
                            operator = '/='
                        else:
                            operator = ''

                        if lhs_value_kind == ValueKind.TEMPORARY:
                            self.__qcx_lines.append(f'LET {lhs_value} {operator} {rhs_value}')
                            if rhs_value_kind == ValueKind.TEMPORARY:
                                self.__release_temporary_variable(rhs_value)
                        elif rhs_value_kind == ValueKind.TEMPORARY:
                            self.__qcx_lines.append(f'LET {rhs_value} {operator} {lhs_value}')
                            self.__value = rhs_value
                            self.__value_kind = ValueKind.TEMPORARY
                        else:
                            temporary_variable: str = self.__add_new_temporary_variable(lhs_value_type)
                            self.__qcx_lines.append(f'LET {temporary_variable} := {lhs_value}')
                            self.__qcx_lines.append(f'LET {temporary_variable} {operator} {rhs_value}')
                            self.__value = temporary_variable
                            self.__value_kind = ValueKind.TEMPORARY

                case ExpressionKind.CONDITIONAL:
                    pass # TODO

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

    # TODO
    # def visit_BooleanLiteral(self, expression: ast.BooleanLiteral):
    #    pass

    # TODO
    # def visit_BitstringLiteral(self, expression: ast.BitstringLiteral):
    #    pass

    # TODO
    # def visit_StringLiteral(self, expression: ast.StringLiteral):
    #    pass

    # TODO
    # def visit_ArrayLiteral(self, expression: ast.ArrayLiteral):
    #    pass

    def visit_Include(self, statement: ast.Include) -> None:
        if not self.__is_initialization_process:
            return

        self.__is_stdgates_included = statement.filename == 'stdgates.inc'

    def visit_QubitDeclaration(self, statement: ast.QubitDeclaration) -> None:
        if not self.__is_initialization_process:
            return

        if statement.size is None:
            self.__quantum_registers[statement.qubit.name] = 1
            return

        self.__expression_kind = ExpressionKind.CONST_ARITHMETIC
        self.visit(statement.size)
        self.__expression_kind = None
        self.__quantum_registers[statement.qubit.name] = int(self.__value)

    def __to_qubit_index(self, quantum_register_name: str, index: int) -> int:
        quantum_register_index = self.__quantum_register_names.index(quantum_register_name)
        return self.__first_qubit_indices[quantum_register_index] + index

    def __convert_parameter(self, parameter: (str, ValueType, ValueKind), qcx_name: str) -> (str, ValueKind):
        if qcx_name not in ['EX', 'EY', 'EZ', 'CEX', 'CEY', 'CEZ']:
            return parameter[0]

        if parameter[2] == ValueKind.LITERAL:
            return str(-0.5 * float(parameter[0])), ValueKind.LITERAL

        if parameter[1] == ValueType.FLOAT:
            if parameter[2] == ValueKind.TEMPORARY:
                self.__qcx_lines.append(f'LET {parameter[0]} *= -0.5')
                return parameter[0], ValueKind.TEMPORARY

            temporary_variable: str = self.__add_new_temporary_variable(parameter[1])
            self.__qcx_lines.append(f'LET {temporary_variable} := {parameter[0]}')
            self.__qcx_lines.append(f'LET {temporary_variable} *= -0.5')
        else:
            temporary_variable: str = self.__add_new_temporary_variable(parameter[1])
            self.__qcx_lines.append(f'LET {temporary_variable} := :REAL:{parameter[0]}')
            self.__qcx_lines.append(f'LET {temporary_variable} *= -0.5')

        if parameter[2] == ValueKind.TEMPORARY:
            self.__release_temporary_variable(parameter[0])

        return temporary_variable, ValueKind.TEMPORARY

    def visit_QuantumGate(self, statement: ast.QuantumGate) -> None:
        if self.__is_initialization_process:
            return

        qasm_gate_name: str = statement.name.name

        if qasm_gate_name in QASM2QCXConverter.default_gates_qcx_map:
            qcx_gate_name = QASM2QCXConverter.default_gates_qcx_map[qasm_gate_name]
        elif self.__is_stdgates_included and qasm_gate_name in QASM2QCXConverter.stdgates_qcx_map:
            qcx_gate_name = QASM2QCXConverter.stdgates_qcx_map[qasm_gate_name]
        else:
            return

        loop_size = 1
        for qubit in filter(lambda qubit: not isinstance(qubit, ast.IndexedIdentifier), statement.qubits):
            if loop_size > 1:
                if loop_size == self.__quantum_registers[qubit.name]:
                    continue
                raise WrongBroadcastingException
            loop_size = self.__quantum_registers[qubit.name]

        parameters: [(str, ValueType, ValueKind)] = []
        for argument in statement.arguments:
            self.__expression_kind = ExpressionKind.ARITHMETIC
            self.visit(argument)
            self.__expression_kind = None

            if self.__value_type == ValueType.COMPLEX:
                raise WrongParameterTypeException(self.__value)

            parameters.append((self.__value, self.__value_type, self.__value_kind))

        temporary_variables: str = []
        for index in range(loop_size):
            qcx_line = f'{qcx_gate_name} {" ".join([str(self.__to_qubit_index(qubit.name.name, qubit.indices[0][0].value)) if isinstance(qubit, ast.IndexedIdentifier) else str(self.__to_qubit_index(qubit.name, index)) for qubit in statement.qubits])}'

            for parameter in parameters:
                converted_parameter, value_kind = self.__convert_parameter(parameter, qcx_gate_name)
                qcx_line += f' {converted_parameter}'
                if value_kind == ValueKind.TEMPORARY:
                    temporary_variables.append(converted_parameter)

            self.__qcx_lines.append(qcx_line)

            for temporary_variable in temporary_variables:
                self.__release_temporary_variable(temporary_variable)

            temporary_variables = []

    def __make_constant_variable(self, variable_name: str, variable_type, num_elements: int = 1) -> None:
        if num_elements <= 0:
            raise #TODO

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
                    raise #TODO

                self.__const_int_variable_name_values_map[variable_name][0] = self.__value

            case ast.FloatType():
                if variable_name not in self.__const_float_variable_name_values_map:
                    raise WrongConstantVariableException(variable_name)

                if self.__value_type == ValueType.COMPLEX:
                    raise #TODO

                self.__const_float_variable_name_values_map[variable_name][0] = self.__value

            #case ast.AngleType():
            #    pass

            #case ast.BitType():
            #    pass

            #case ast.BoolType():
            #    pass

            case ast.ComplexType():
                if variable_name not in self.__const_complex_variable_name_values_map:
                    raise WrongConstantVariableException(variable_name)

                self.__const_complex_variable_name_values_map[variable_name][0] = self.__value

    def visit_ConstantDeclaration(self, statement: ast.ConstantDeclaration) -> None:
        if not self.__is_initialization_process:
            return

        variable_type = statement.type
        variable_name = statement.identifier.name

        match variable_type:
            case ast.ArrayType():
                self.__make_constant_variable(variable_name, variable_type.base_type, math.prod(variable_type.dimensions))

            case _:
                self.__make_constant_variable(variable_name, variable_type)

        if statement.init_expression is None:
            return

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
            raise # TODO

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

            #case ast.AngleType():
            #    if variable_name in self.__angle_variable_name_size_map:
            #        raise WrongClassicalDeclarationException(variable_name)

            #    if variable_type.size is None:
            #        self.__angle_variable_name_size_map[variable_name] = 64
            #    else:
            #        array_size = int(variable_type.size)
            #        self.__angle_variable_name_size_map[variable_name] = array_size
            #    self.__qcx_lines.append(f'VAR {variable_name} INT' + (f' {num_elements}' if num_elements > 1 else ''))

            #case ast.BitType():
            #    if variable_name in self.__bit_variable_name_size_map:
            #        raise WrongClassicalDeclarationException(variable_name)

            #    if variable_type.size is None:
            #        self.__bit_variable_name_size_map[variable_name] = 1
            #        self.__qcx_lines.append(f'VAR {variable_name} INT {num_elements}')
            #    else:
            #        self.__expression_kind = ExpressionKind.ARITHMETIC
            #        self.visit(statement.size)
            #        self.__expression_kind = None

            #        if self.__value_type != ValueType.INT:
            #            raise

            #        self.__qcx_lines.append(f'VAR {variable_name} INT {num_elements}')

            #        if parameter[2] == ValueKind.TEMPORARY:
            #            self.__release_temporary_variable(parameter[0])

            #    self.__bit_variable_name_size_map[variable_name] = 1
            #    self.__qcx_lines.append(f'VAR {variable_name} INT {}' + (f' {num_elements}' if num_elements > 1 else ''))

            #case ast.BoolType():
            #    if variable_name in self.__bool_variable_name_size_map:
            #        raise WrongClassicalDeclarationException(variable_name)

            #    self.__int_variable_name_size_map[variable_name] = 1
            #    self.__qcx_lines.append(f'VAR {variable_name} INT' + (f' {num_elements}' if num_elements > 1 else ''))

            case ast.ComplexType():
                if variable_name in self.__complex_variable_name_size_map:
                    raise WrongClassicalDeclarationException(variable_name)

                self.__complex_variable_name_size_map[variable_name] = num_elements
                self.__qcx_lines.append(f'VAR {variable_name} COMPLEX' + (f' {num_elements}' if num_elements > 1 else ''))

    def visit_ClassicalDeclaration(self, statement: ast.ClassicalDeclaration) -> None:
        if self.__is_initialization_process:
            return

        variable_type = statement.type
        variable_name = self.__capitalize_variable_name(statement.identifier.name)

        match variable_type:
            case ast.ArrayType():
                self.__declare_classical_variable(variable_type.base_type, variable_name, math.prod(variable_type.dimensions))

            case _:
                self.__declare_classical_variable(variable_type, variable_name)

        if statement.init_expression is None:
            return

        match statement.init_expression:
            case ast.Expression():
                self.__expression_kind = ExpressionKind.ARITHMETIC
                self.visit(statement.init_expression)
                self.__expression_kind = None

                self.__qcx_lines.append(f'LET {variable_name} := {self.__value}')

                if self.__value_kind == ValueKind.TEMPORARY:
                    self.__release_temporary_variable(self.__value)

            case ast.QuantumMeasurement():
                pass # TODO

            case ast.QuantumCallExpression():
                pass # TODO

    def visit_ClassicalAssignment(self, statement: ast.ClassicalAssignment) -> None:
        if self.__is_initialization_process:
            return

        variable_name: str = self.__capitalize_variable_name(statement.lvalue.name)
        variable_type: ValueType = self.__type_of(variable_name)
        if isinstance(statement.lvalue, ast.IndexedIdentifier):
            # TODO: present impl. assumes size(indices) == 1, indices[0] is list[Expression], and size(indices[0]) == 1
            self.__expression_kind = ExpressionKind.ARITHMETIC
            self.visit(statement.lvalue.indices[0][0])
            self.__expression_kind = None
            variable_name = f'{variable_name}:{self.__value}'

        # TODO: %= and other operations
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
            operator = ''

        self.__expression_kind = ExpressionKind.ARITHMETIC
        self.visit(statement.rvalue)
        self.__expression_kind = None
        rhs_value = self.__value
        rhs_value_type: ValueType = self.__value_type
        rhs_value_kind: ValueKind = self.__value_kind

        if rhs_value_kind == ValueKind.LITERAL:
            if variable_type == ValueType.INT:
                rhs = int(rhs_value)
            elif variable_type == ValueType.FLOAT:
                rhs = float(rhs_value)
            elif variable_type == ValueType.COMPLEX:
                rhs = complex(rhs_value)
            else:
                rhs = None
            self.__qcx_lines.append(f'LET {variable_name} {operator} {rhs}')
        else:
            if rhs_value_type == variable_type:
                self.__qcx_lines.append(f'LET {variable_name} {operator} {rhs_value}')
            else:
                temporary_variable: str = self.__add_new_temporary_variable(variable_type)
                if variable_type == ValueType.INT:
                    cast_type: str = 'INT'
                elif variable_type == ValueType.FLOAT:
                    cast_type: str = 'REAL'
                elif variable_type == ValueType.COMPLEX:
                    cast_type: str = 'COMPLEX'
                else:
                    cast_type: str = ''
                self.__qcx_lines.append(f'LET {temporary_variable} := :{cast_type}:{rhs_value}')
                self.__qcx_lines.append(f'LET {variable_name} {operator} {temporary_variable}')
                self.__release_temporary_variable(temporary_variable)

            if rhs_value_kind == ValueKind.TEMPORARY:
                self.__release_temporary_variable(rhs_value)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        exit('usage: qasm2qcx.py <OpenQASM file name>')

    qasm_filename: str = sys.argv[1]
    with open(qasm_filename) as qasm_file:
        qasm_ast_root: ast.Program = openqasm3.parse(qasm_file.read())

        converter: QASM2QCXConverter = QASM2QCXConverter(qasm_ast_root)
        converter.visit(qasm_ast_root)

        for qcx_line in converter:
            print(qcx_line)

