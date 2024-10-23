/// @file IRGenQASM3Visitor.h
/// @brief Visitor header file that traverses the OpenQASM 3 AST to generate IRs.
/// @note Include doxygen comments only where used.
///
/// Copyright (c) RIKEN, Japan. All rights reserved.

#ifndef VISITOR_IR_GEN_VISITOR_H
#define VISITOR_IR_GEN_VISITOR_H

#include "BaseQASM3Visitor.h"

namespace qip::frontend::openqasm3 {

/// @class IRGenQASM3Visitor
/// @brief Visitor class to generate IRs for quantum circuits.
class IRGenQASM3Visitor : public BaseQASM3Visitor {
private:
  std::ostream &vStream; // visitor output stream

public:
  /// @brief Constructor of the IRGenQASM3Visitor class
  /// @param [in] sList AST Statement List
  /// @param [in] os ostream
  IRGenQASM3Visitor(QASM::ASTStatementList *sList, std::ostream &os)
      : BaseQASM3Visitor(sList), vStream(os) {}

  IRGenQASM3Visitor(std::ostream &os) : BaseQASM3Visitor(), vStream(os) {}

  void visit(const QASM::ASTForStatementNode *) override;

  void visit(const QASM::ASTForLoopNode *) override;

  void visit(const QASM::ASTIfStatementNode *) override;

  void visit(const QASM::ASTElseStatementNode *) override;

  void visit(const QASM::ASTSwitchStatementNode *) override;

  void visit(const QASM::ASTWhileStatementNode *) override;

  void visit(const QASM::ASTWhileLoopNode *) override;

  void visit(const QASM::ASTReturnStatementNode *) override;

  void visit(const QASM::ASTResultNode *) override;

  void visit(const QASM::ASTFunctionDeclarationNode *) override;

  void visit(const QASM::ASTFunctionDefinitionNode *) override;

  void visit(const QASM::ASTGateDeclarationNode *) override;

  /// @brief Obtain information on Generic gate application
  /// @details Set the ID to the gate operation information.
  ///          Call the visit function to obtain the number of qubits and the qubit number.
  ///          Get the argument Double value.
  /// @param [in] node ASTGenericGateOpNode
  void visit(const QASM::ASTGenericGateOpNode *node) override;

  /// @brief Obtain information on gate application
  /// @details Obtain the number of qubits and the qubit number from the ASTGateNode and set them to the gate operation information.
  /// @param [in] node ASTGateNode
  void visit(const QASM::ASTGateNode *node) override;

  /// @brief Obtain information on the application of hadamard gate
  /// @details Set “_HGate” for gate operation information.
  ///          Call the visit function to obtain the number of qubits and the qubit number.
  /// @param [in] node ASTHGateOpNode
  void visit(const QASM::ASTHGateOpNode *node) override;

  void visit(const QASM::ASTUGateOpNode *) override;

  /// @brief Obtain information on CNOT gate application
  /// @details Set “_CXGate” for gate operation information.
  ///          Call the visit function to obtain the number of qubits and the qubit number.
  /// @param [in] node ASTHGateOpNode
  void visit(const QASM::ASTCXGateOpNode *node) override;

  void visit(const QASM::ASTResetNode *) override;

  /// @brief Get measure information
  /// @details Call the visit function to obtain information about the target qubit.
  /// @param [in] node ASTMeasureNode
  ///
  /// @note Intermediate code generation is not implemented.
  /// @attention measure is a feature that does not exist in ket and is therefore not supported.
  void visit(const QASM::ASTMeasureNode *node) override;

  void visit(const QASM::ASTDelayStatementNode *) override;

  void visit(const QASM::ASTDelayNode *) override;

  void visit(const QASM::ASTBarrierNode *) override;

  /// @brief Obtain information on qubits and classical bits
  /// @details Call the visit function to obtain information on the quantum and classical bits.
  /// @param [in] node ASTDeclarationNode
  void visit(const QASM::ASTDeclarationNode *node) override;

  void visit(const QASM::ASTKernelDeclarationNode *) override;

  /// @brief Get the number of qubits
  /// @param [in] node ASTQubitContainerNode
  void visit(const QASM::ASTQubitContainerNode *node) override;

  void visit(const QASM::ASTQubitNode *) override;

  /// @brief Obtain information on classical bits
  /// @param [in] node ASTCBitNode
  /// @attention CBit used in measure that does not exist in ket is not supported.
  void visit(const QASM::ASTCBitNode *node) override;

  void visit(const QASM::ASTDurationNode *) override;

  void visit(const QASM::ASTStretchStatementNode *) override;

  void visit(const QASM::ASTStretchNode *) override;

  void visit(const QASM::ASTIdentifierRefNode *) override;

  void visit(const QASM::ASTIdentifierNode *) override;

  void visit(const QASM::ASTBinaryOpNode *) override;

  void visit(const QASM::ASTIntNode *) override;

  void visit(const QASM::ASTMPIntegerNode *) override;

  void visit(const QASM::ASTFloatNode *) override;

  void visit(const QASM::ASTMPDecimalNode *) override;

  void visit(const QASM::ASTMPComplexNode *) override;

  /// @brief Get information on Angle
  /// @param [in] node ASTAngleNode
  void visit(const QASM::ASTAngleNode *node) override;

  void visit(const QASM::ASTBoolNode *) override;

  void visit(const QASM::ASTCastExpressionNode *) override;

  void visit(const QASM::ASTKernelNode *) override;

  void visit(const QASM::ASTDeclarationList *list) override;

  void visit(const QASM::ASTFunctionCallNode *node) override;

  void visit(const QASM::ASTVoidNode *) override;

  void visit(const QASM::ASTOperatorNode *) override;

  void visit(const QASM::ASTOperandNode *) override;

  void visit(const QASM::ASTUnaryOpNode *) override;
};

} // namespace qip::frontend::openqasm3

#endif // VISITOR_IR_GEN_VISITOR_H
