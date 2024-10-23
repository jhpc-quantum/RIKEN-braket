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

#define SET_ERROR_INFO(msg)  std::cerr << "Unsupported : " << "\"" << msg << "\"" << "\n"; \
  qip::parseErrFlag = true;

namespace qip {

extern qipIrTy qasmir;  ///< Variable that holds information on the gate of a quantum circuit
extern bool parseErrFlag;  ///< parse error flag

}

using namespace QASM;

namespace qip::frontend::openqasm3 {

void IRGenQASM3Visitor::visit(const ASTForStatementNode *node) {
  SET_ERROR_INFO("For Statement");
}

void IRGenQASM3Visitor::visit(const ASTForLoopNode *node) {
  SET_ERROR_INFO("For Loop");
}

void IRGenQASM3Visitor::visit(const ASTIfStatementNode *node) {
  SET_ERROR_INFO("If Statement");
}

void IRGenQASM3Visitor::visit(const ASTElseStatementNode *node) {
  SET_ERROR_INFO("Else Statement");
}

void IRGenQASM3Visitor::visit(const ASTSwitchStatementNode *node) {
  SET_ERROR_INFO("Switch Statement");
}

void IRGenQASM3Visitor::visit(const ASTWhileStatementNode *node) {
  SET_ERROR_INFO("While Statement");
}

void IRGenQASM3Visitor::visit(const ASTWhileLoopNode *node) {
  SET_ERROR_INFO("While Loop");
}

void IRGenQASM3Visitor::visit(const ASTReturnStatementNode *node) {
  SET_ERROR_INFO("Return Statement");
}

void IRGenQASM3Visitor::visit(const ASTResultNode *node) {
  SET_ERROR_INFO("Result");
}

void IRGenQASM3Visitor::visit(const ASTFunctionDeclarationNode *node) {
  SET_ERROR_INFO("Function Declaration");
}

void IRGenQASM3Visitor::visit(const ASTFunctionDefinitionNode *node) {
  SET_ERROR_INFO("Function Definition");
}

void IRGenQASM3Visitor::visit(const ASTGateDeclarationNode *node) {
}

void IRGenQASM3Visitor::visit(const ASTGenericGateOpNode *node) {
  const ASTGateNode *gateNode = node->GetGateNode();
  const std::string &gateName = gateNode->GetName();

  if (gateName == "cz") {
    qasmir.gate[qasmir.ngates].id = CZGate;
  }
  else if (gateName == "s") {
    qasmir.gate[qasmir.ngates].id = SGate;
  }
  else if (gateName == "sdg") {
    qasmir.gate[qasmir.ngates].id = SdgGate;
  }
  else if (gateName == "rx") {
    qasmir.gate[qasmir.ngates].id = RXGate;
  }
  else if (gateName == "ry") {
    qasmir.gate[qasmir.ngates].id = RYGate;
  }
  else if (gateName == "rz") {
    qasmir.gate[qasmir.ngates].id = RZGate;
  }
  else if (gateName == "x") {
    qasmir.gate[qasmir.ngates].id = XGate;
  }
  else if (gateName == "u1") {
    qasmir.gate[qasmir.ngates].id = U1Gate;
  }
  else {
    SET_ERROR_INFO("Generic Gate");
  }

  visit(gateNode);

  qasmir.ngates++;
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

  // Set the number of arguments to a real parameter number.
  const size_t nParams = node->GetNumParams();
  qasmir.gate[qasmir.ngates].nrarg = nParams;

  // Set the value of the argument to the real parameter.
  for (size_t i = 0; i < nParams; i++) {
    if (!node->GetIdentifier()) {
      assert(0 && "ASTGateNode");
    }
    auto param = node->GetParam(i);
    double value = 0.0;
    if (!param->IsNan()) {
      value = param->AsDouble();
    }
    qasmir.gate[qasmir.ngates].rarg[i] = value;
  }

  // Set the number of qubits to an integer parameter number.
  const size_t nQCParams = node->GetNumQCParams();
  qasmir.gate[qasmir.ngates].niarg = nQCParams;

  // Set the integer parameter to a qubit number.
  for (size_t i = 0; i < nQCParams; i++) {
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
  SET_ERROR_INFO("U Gate");
}

void IRGenQASM3Visitor::visit(const ASTCXGateOpNode *node) {
  qasmir.gate[qasmir.ngates].id = CXGate;

  const ASTGateNode *gateNode = node->GetGateNode();
  visit(gateNode);

  qasmir.ngates++;
}

void IRGenQASM3Visitor::visit(const ASTResetNode *node) {
  SET_ERROR_INFO("Reset");
}

void IRGenQASM3Visitor::visit(const ASTMeasureNode *node) {
  SET_ERROR_INFO("Measure");
}

void IRGenQASM3Visitor::visit(const ASTDelayStatementNode *node) {
  SET_ERROR_INFO("Delay Statement");
}

void IRGenQASM3Visitor::visit(const ASTDelayNode *node) {
  SET_ERROR_INFO("Delay");
}

void IRGenQASM3Visitor::visit(const ASTBarrierNode *node) {
  SET_ERROR_INFO("Barrier");
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
  SET_ERROR_INFO("Kernel Declaration");
}

void IRGenQASM3Visitor::visit(const ASTQubitContainerNode *node) {
  // Get the number of declared qubits.
  qasmir.qubits = node->Size();
}

void IRGenQASM3Visitor::visit(const ASTQubitNode *node) {
  SET_ERROR_INFO("Qubit");
}

void IRGenQASM3Visitor::visit(const ASTCBitNode *node) {
  SET_ERROR_INFO("C Bit");
}

void IRGenQASM3Visitor::visit(const ASTDurationNode *node) {
  SET_ERROR_INFO("Duration");
}

void IRGenQASM3Visitor::visit(const ASTStretchStatementNode *node) {
  SET_ERROR_INFO("Stretch Statement");
}

void IRGenQASM3Visitor::visit(const ASTStretchNode *node) {
  SET_ERROR_INFO("Stretch");
}

void IRGenQASM3Visitor::visit(const ASTIdentifierRefNode *node) {
  SET_ERROR_INFO("Identifier Ref");
}

void IRGenQASM3Visitor::visit(const ASTIdentifierNode *node) {
  SET_ERROR_INFO("Identifier");
}

void IRGenQASM3Visitor::visit(const ASTBinaryOpNode *node) {
  SET_ERROR_INFO("Binary");
}

void IRGenQASM3Visitor::visit(const ASTUnaryOpNode *node) {
  SET_ERROR_INFO("Unary");
}

void IRGenQASM3Visitor::visit(const ASTIntNode *node) {
  SET_ERROR_INFO("Int");
}

void IRGenQASM3Visitor::visit(const ASTMPIntegerNode *node) {
  SET_ERROR_INFO("MP Integer");
}

void IRGenQASM3Visitor::visit(const ASTFloatNode *node) {
  SET_ERROR_INFO("Float");
}

void IRGenQASM3Visitor::visit(const ASTMPDecimalNode *node) {
  SET_ERROR_INFO("MP Decimal");
}

void IRGenQASM3Visitor::visit(const ASTMPComplexNode *node) {
  SET_ERROR_INFO("MP Complex");
}

void IRGenQASM3Visitor::visit(const ASTAngleNode *node) {
}

void IRGenQASM3Visitor::visit(const ASTBoolNode *node) {
  SET_ERROR_INFO("Bool");
}

void IRGenQASM3Visitor::visit(const ASTCastExpressionNode *node) {
  SET_ERROR_INFO("Cast Expression");
}

void IRGenQASM3Visitor::visit(const ASTKernelNode *node) {
  SET_ERROR_INFO("Kernel");
}

void IRGenQASM3Visitor::visit(const ASTDeclarationList *list) {
  SET_ERROR_INFO("Declaration List");
}

void IRGenQASM3Visitor::visit(const ASTFunctionCallNode *node) {
  SET_ERROR_INFO("Function Call");
}

void IRGenQASM3Visitor::visit(const QASM::ASTVoidNode *) {
  SET_ERROR_INFO("Void");
}

void IRGenQASM3Visitor::visit(const QASM::ASTOperatorNode *node) {
  SET_ERROR_INFO("Operator");
}

void IRGenQASM3Visitor::visit(const QASM::ASTOperandNode *node) {
  SET_ERROR_INFO("Operand");
}

} // namespace qip::frontend::openqasm3
