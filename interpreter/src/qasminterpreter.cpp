#include "IRGenQASM3Visitor.h"
#include "qipTypes.h"
#include "qipKet.hpp"

#include <qasm/AST/AST.h>
#include <qasm/AST/ASTObjectTracker.h>
#include <qasm/Frontend/QasmParser.h>

#include <iostream>


/// @brief Usage出力
static void Usage() {
  std::cerr << "Usage: qasminterpreter ";
  std::cerr << "[-I<include-dir> [ -I<include-dir> ...]] ";
  std::cerr << "\n                  <translation-unit>" << std::endl;
}
namespace qip {

/// @brief 量子回路のゲートの情報を保持する変数宣言
/// @note 現時点では、複数回qubit宣言には対応していない。
qipIrTy qasmir;
/// @brief ketの呼び出しを行うためのクラス宣言
ketInfo ki;

}

/// @brief 量子回路IRをダンプする
static void printSqcIr() {
  std::cout << "qipIrTy" << std::endl;
  std::cout << " <qubits> : " << qip::qasmir.qubits << std::endl;
  std::cout << " <ngates> : " << qip::qasmir.ngates << std::endl;
  for (int i = 0; i < qip::qasmir.ngates; i++) {
    std::cout << " <gate[" << i << "]>" << std::endl;
    std::cout << "  <id> : " << qip::qasmir.gate[i].id << std::endl;
    int niarg = qip::qasmir.gate[i].niarg;
    std::cout << "  <niarg> : " << niarg << std::endl;
    int nrarg = qip::qasmir.gate[i].nrarg;
    std::cout << "  <nrarg> : " << nrarg << std::endl;
    for (int j = 0; j < niarg; j++) {
      std::cout << "  <iarg[" << j << "]> : " << qip::qasmir.gate[i].iarg[j] << std::endl;
    }
    for (int j = 0; j < nrarg; j++) {
      std::cout << " <rarg[" << j << "]> : " << qip::qasmir.gate[i].rarg[j] << std::endl;
    }
  }
}

int main(int argc, char *argv[]) {
  std::ios::sync_with_stdio(false);

  if (argc < 2) {
    Usage();
    return 1;
  }

  // MPI実行の準備
  qip::ki.environment  = new yampi::environment(argc, argv, yampi::thread_support::funneled);
  qip::ki.communicator = new yampi::communicator(yampi::tags::world_communicator);
  qip::ki.rank         = qip::ki.communicator->rank(*(qip::ki.environment));
  qip::ki.nprocs       = qip::ki.communicator->size(*(qip::ki.environment));
  MPI_Comm_rank(MPI_COMM_WORLD, &(qip::ki.myrank));
  qip::ki.root         = yampi::rank{0};

  // Enabling the ASTObjectTracker is optional.
  // Nothing bad will happen if it's not enabled. By default, the memory
  // allocated by the AST Generator is handed over unmanaged, and will be
  // automatically released at program exit.
  // The ASTObjectTracker manages the memory dynamically allocated by the
  // AST Generator. It is enabled here for illustration purposes.
  QASM::ASTObjectTracker::Instance().Enable();

  // Parse
  QASM::ASTParser Parser;
  auto root = std::unique_ptr<QASM::ASTRoot>(nullptr);
  Parser.ParseCommandLineArguments(argc, argv);
  root.reset(Parser.ParseAST());

  // visitor
  auto *statementList = QASM::ASTStatementBuilder::Instance().List();
  qip::frontend::openqasm3::IRGenQASM3Visitor visitor(std::cout);

  // ASTを辿り、量子回路IRを生成
  visitor.setStatementList(statementList);
  visitor.walkAST();

  // 量子回路IRのダンプ
  printSqcIr();

  // ket呼び出し
  // 初期化
  qip::initialize();

  // ゲート適用
  qip::addGate();

  // 終了処理
  qip::finalize();

  // If the ASTObjectTracker is not enabled, this is a no-op.
  QASM::ASTObjectTracker::Instance().Release();

  return 0;
}
