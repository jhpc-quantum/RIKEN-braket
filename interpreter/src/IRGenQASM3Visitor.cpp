/// @file IRGenQASM3Visitor.cpp
/// @brief Visitor to traverse the AST of OpenQASM 3 and generate IR.
/// @note Include doxygen comments only where used.

#include "IRGenQASM3Visitor.h"

#include "BaseQASM3Visitor.h"

#include "qipTypes.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <qasm/AST/ASTBarrier.h>
#include <qasm/AST/ASTCBit.h>
#include <qasm/AST/ASTCastExpr.h>
#include <qasm/AST/ASTDeclarationList.h>
#include <qasm/AST/ASTDelay.h>
#include <qasm/AST/ASTDuration.h>
#include <qasm/AST/ASTFunctionCallExpr.h>
#include <qasm/AST/ASTFunctions.h>
#include <qasm/AST/ASTGateOpList.h>
#include <qasm/AST/ASTGates.h>
#include <qasm/AST/ASTIdentifier.h>
#include <qasm/AST/ASTIfConditionals.h>
#include <qasm/AST/ASTIntegerList.h>
#include <qasm/AST/ASTKernel.h>
#include <qasm/AST/ASTLoops.h>
#include <qasm/AST/ASTMeasure.h>
#include <qasm/AST/ASTQubit.h>
#include <qasm/AST/ASTReset.h>
#include <qasm/AST/ASTResult.h>
#include <qasm/AST/ASTReturn.h>
#include <qasm/AST/ASTStatement.h>
#include <qasm/AST/ASTStretch.h>
#include <qasm/AST/ASTSwitchStatement.h>
#include <qasm/AST/ASTSymbolTable.h>
#include <qasm/AST/ASTTypeEnums.h>
#include <qasm/AST/ASTTypes.h>
#include <sstream>
#include <stdexcept>
#include <string>

namespace qip {

extern qipIrTy qasmir;  ///< Variable that holds information on the gate of a quantum circuit

}

using namespace QASM;

namespace qip::frontend::openqasm3 {

void IRGenQASM3Visitor::visit(const ASTForStatementNode *node) {
  assert(0 && "ASTForStatementNode");
}

void IRGenQASM3Visitor::visit(const ASTForLoopNode *node) {
  assert(0 && "ASTForLoopNode");
}

void IRGenQASM3Visitor::visit(const ASTIfStatementNode *node) {
  assert(0 && "ASTIfStatementNode");
}

void IRGenQASM3Visitor::visit(const ASTElseStatementNode *node) {
  assert(0 && "ASTElseStatementNode");
}

void IRGenQASM3Visitor::visit(const ASTSwitchStatementNode *node) {
  assert(0 && "ASTSwitchStatementNode");
}

void IRGenQASM3Visitor::visit(const ASTWhileStatementNode *node) {
  assert(0 && "ASTWhileStatementNode");
}

void IRGenQASM3Visitor::visit(const ASTWhileLoopNode *node) {
  assert(0 && "ASTWhileLoopNode");
}

void IRGenQASM3Visitor::visit(const ASTReturnStatementNode *node) {
  assert(0 && "ASTReturnStatementNode");
}

void IRGenQASM3Visitor::visit(const ASTResultNode *node) {
  assert(0 && "ASTResultNode");
}

void IRGenQASM3Visitor::visit(const ASTFunctionDeclarationNode *node) {
  assert(0 && "ASTFunctionDeclarationNode");
}

void IRGenQASM3Visitor::visit(const ASTFunctionDefinitionNode *node) {
  assert(0 && "ASTFunctionDefinitionNode");
}

void IRGenQASM3Visitor::visit(const ASTGateDeclarationNode *node) {
}

void IRGenQASM3Visitor::visit(const ASTGenericGateOpNode *node) {
  assert(0 && "ASTGenericGateOpNode");
}

void IRGenQASM3Visitor::visit(const ASTGateNode *node) {
  // number of parameters
  const size_t numParams = node->ParamsSize();
  // number of quantum bit
  const size_t numQubits = node->QubitsSize();

  // Obtain parameter information.
  for (size_t i = 0; i < numParams; i++) {
    visit(node->GetParam(i));
  }

  // Obtaining information on qubits.
  for (size_t i = 0; i < numQubits; i++) {
    visit(node->GetQubit(i));
  }

  // Set the number of qubits to an integer parameter number.
  qasmir.gate[qasmir.ngates].niarg = node->GetNumQCParams();

  // Set the integer parameter to a qubit number.
  for (size_t i = 0; i < node->GetNumQCParams(); i++) {
    auto *paramId = node->GetQCParams()[i]->GetIdentifier();
    assert(paramId);

    auto qcname = paramId->GetName();
    size_t pos = qcname.find(":");
    qasmir.gate[qasmir.ngates].iarg[i] = std::stoi(qcname.substr(pos+1));
  }
}

void IRGenQASM3Visitor::visit(const ASTHGateOpNode *node) {
  qasmir.gate[qasmir.ngates].id = HGate;

  const ASTGateNode *gateNode = node->GetGateNode();
  visit(gateNode);

  qasmir.ngates++;
}

void IRGenQASM3Visitor::visit(const ASTUGateOpNode *node) {
  assert(0 && "ASTUGateOpNode");
}

void IRGenQASM3Visitor::visit(const ASTCXGateOpNode *node) {
  qasmir.gate[qasmir.ngates].id = CXGate;

  const ASTGateNode *gateNode = node->GetGateNode();
  visit(gateNode);

  qasmir.ngates++;
}

void IRGenQASM3Visitor::visit(const ASTResetNode *node) {
  assert(0 && "ASTResetNode");
}

void IRGenQASM3Visitor::visit(const ASTMeasureNode *node) {
  const ASTQubitContainerNode *qubitNode = node->GetTarget();
  visit(qubitNode);
  if (const ASTCBitNode *bits = node->GetResult())
    visit(bits);
}

void IRGenQASM3Visitor::visit(const ASTDelayStatementNode *node) {
  assert(0 && "ASTDelayStatementNode");
}

void IRGenQASM3Visitor::visit(const ASTDelayNode *node) {
  assert(0 && "ASTDelayNode");
}

void IRGenQASM3Visitor::visit(const ASTBarrierNode *node) {
  assert(0 && "ASTBarrierNode");
}

void IRGenQASM3Visitor::visit(const ASTDeclarationNode *node) {
  // if it's a function, process it directly
  if (const auto *funcDecl =
          dynamic_cast<const ASTFunctionDeclarationNode *>(node)) {
    visit(funcDecl);
    return;
  }
  // otherwise, lookup node in sym table if it exists
  const ASTIdentifierNode *idNode = node->GetIdentifier();
  ASTSymbolTableEntry *symTableEntry =
      ASTSymbolTable::Instance().Lookup(idNode);
  if (symTableEntry)
    BaseQASM3Visitor::visit(symTableEntry);
  // finally resort to printing the identifier.
  else
    visit(idNode);
}

void IRGenQASM3Visitor::visit(const ASTKernelDeclarationNode *node) {
  assert(0 && "ASTKernelDeclarationNode");
}

void IRGenQASM3Visitor::visit(const ASTQubitContainerNode *node) {
  // Get the number of declared qubits.
  qasmir.qubits = node->Size();
}

void IRGenQASM3Visitor::visit(const ASTQubitNode *node) {
  assert(0 && "ASTQubitNode");
}

void IRGenQASM3Visitor::visit(const ASTCBitNode *node) {
  if (const auto *nodeGateOp =
          dynamic_cast<const ASTMeasureNode *>(node->GetGateQOp())) {
    visit(nodeGateOp);
  }
}

void IRGenQASM3Visitor::visit(const ASTDurationNode *node) {
  assert(0 && "ASTDurationNode");
}

void IRGenQASM3Visitor::visit(const ASTStretchStatementNode *node) {
  assert(0 && "ASTStretchStatementNode");
}

void IRGenQASM3Visitor::visit(const ASTStretchNode *node) {
  assert(0 && "ASTStretchNode");
}

void IRGenQASM3Visitor::visit(const ASTIdentifierRefNode *node) {
  assert(0 && "ASTIdentifierRefNode");
}

void IRGenQASM3Visitor::visit(const ASTIdentifierNode *node) {
  assert(0 && "ASTIdentifierNode");
}

void IRGenQASM3Visitor::visit(const ASTBinaryOpNode *node) {
  assert(0 && "ASTBinaryOpNode");
}

void IRGenQASM3Visitor::visit(const ASTUnaryOpNode *node) {
  assert(0 && "ASTUnaryOpNode");
}

void IRGenQASM3Visitor::visit(const ASTIntNode *node) {
  assert(0 && "ASTIntNode");
}

void IRGenQASM3Visitor::visit(const ASTMPIntegerNode *node) {
  assert(0 && "ASTMPIntegerNode");
}

void IRGenQASM3Visitor::visit(const ASTFloatNode *node) {
  assert(0 && "ASTFloatNode");
}

void IRGenQASM3Visitor::visit(const ASTMPDecimalNode *node) {
  assert(0 && "ASTMPDecimalNode");
}

void IRGenQASM3Visitor::visit(const ASTMPComplexNode *node) {
  assert(0 && "ASTMPComplexNode");
}

void IRGenQASM3Visitor::visit(const ASTAngleNode *node) {
  assert(0 && "ASTAngleNode");
}

void IRGenQASM3Visitor::visit(const ASTBoolNode *node) {
  assert(0 && "ASTBoolNode");
}

void IRGenQASM3Visitor::visit(const ASTCastExpressionNode *node) {
  assert(0 && "ASTCastExpressionNode");
}

void IRGenQASM3Visitor::visit(const ASTKernelNode *node) {
  assert(0 && "ASTKernelNode");
}

void IRGenQASM3Visitor::visit(const ASTDeclarationList *list) {
  assert(0 && "ASTDeclarationList");
}

void IRGenQASM3Visitor::visit(const ASTFunctionCallNode *node) {
  assert(0 && "ASTFunctionCallNode");
}

void IRGenQASM3Visitor::visit(const QASM::ASTVoidNode *) {
  assert(0 && "QASM::ASTVoidNode");
}

void IRGenQASM3Visitor::visit(const QASM::ASTOperatorNode *node) {
  assert(0 && "QASM::ASTOperatorNode");
}

void IRGenQASM3Visitor::visit(const QASM::ASTOperandNode *node) {
  assert(0 && "QASM::ASTOperandNode");
}

} // namespace qip::frontend::openqasm3
