#ifndef VISITOR_IR_GEN_VISITOR_H
#define VISITOR_IR_GEN_VISITOR_H

#include "BaseQASM3Visitor.h"

namespace qip::frontend::openqasm3 {

/// @class IRGenQASM3Visitor
/// @brief 量子回路IR生成
class IRGenQASM3Visitor : public BaseQASM3Visitor {
private:
  std::ostream &vStream; // visitor output stream

public:
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

  void visit(const QASM::ASTGenericGateOpNode *) override;

  /// @brief ゲート適用の情報取得
  /// @details ASTGateNodeから量子ビット数と量子ビット番号を取得し、ゲート操作情報に設定する。
  /// @param [in] node ASTGateNode
  void visit(const QASM::ASTGateNode *) override;

  /// @brief アダマールゲート適用の情報取得
  /// @details ゲート操作情報に"_HGate"を設定する。
  ///          量子ビット数と量子ビット番号を取得するvisit関数を呼び出す。
  /// @param [in] node ASTHGateOpNode
  void visit(const QASM::ASTHGateOpNode *) override;

  void visit(const QASM::ASTUGateOpNode *) override;

  /// @brief CNOTゲート適用の情報取得
  /// @details ゲート操作情報に"_CXGate"を設定する。
  ///          量子ビット数と量子ビット番号を取得するvisit関数を呼び出す。
  /// @param [in] node ASTHGateOpNode
  void visit(const QASM::ASTCXGateOpNode *) override;

  void visit(const QASM::ASTResetNode *) override;

  /// @brief measureの情報取得
  /// @details ターゲットの量子ビットの情報を取得するvisit関数を呼び出す。
  /// @param [in] node ASTMeasureNode
  ///
  /// @note 中間コード生成は未実装
  void visit(const QASM::ASTMeasureNode *) override;

  void visit(const QASM::ASTDelayStatementNode *) override;

  void visit(const QASM::ASTDelayNode *) override;

  void visit(const QASM::ASTBarrierNode *) override;

  /// @brief 量子ビット、古典ビットの情報取得
  /// @details 量子ビット、古典ビットの情報を取得するvisit関数を呼び出す。
  /// @param [in] node ASTDeclarationNode
  void visit(const QASM::ASTDeclarationNode *) override;

  void visit(const QASM::ASTKernelDeclarationNode *) override;

  /// @brief 量子ビット数の取得
  /// @param [in] node ASTQubitContainerNode
  void visit(const QASM::ASTQubitContainerNode *) override;

  void visit(const QASM::ASTQubitNode *) override;

  /// @brief 古典ビットの情報取得
  /// @param [in] node ASTCBitNode
  void visit(const QASM::ASTCBitNode *) override;

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

  void visit(const QASM::ASTAngleNode *) override;

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
