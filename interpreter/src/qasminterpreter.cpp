#include "IRGenQASM3Visitor.h"
#include "qipTypes.h"
#include "qipKet.hpp"

#include <qasm/AST/AST.h>
#include <qasm/AST/ASTObjectTracker.h>
#include <qasm/Frontend/QasmParser.h>

#include <iostream>


/// @brief Usage
static void Usage() {
  std::cerr << "Usage: qasminterpreter ";
  std::cerr << "[-I<include-dir> [ -I<include-dir> ...]] ";
  std::cerr << "\n                  <translation-unit>" << std::endl;
}
namespace qip {

/// @brief Declare variables that hold information about the gates of the quantum circuit
/// @note At this time, multiple qubit declarations are not supported.
qipIrTy qasmir;
/// @brief Class declaration for calling ket
ketInfo ki;

}

/// @brief Dumping a quantum circuit IR
/// @note At this time, runtime options to control dump output are not supported.
/// @todo The intermediate code dump should be tied to the input code.
///       - The id should show the gate type
///       - Quantum bit numbers should be displayed in the specified order
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

  // Prepare for MPI execution
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

  // Trace AST and generate quantum circuit IR.
  visitor.setStatementList(statementList);
  visitor.walkAST();

  // Quantum circuit IR dump output.
  printSqcIr();

  // Call ket
  // initialization
  qip::initialize();

  // Apply gate
  qip::addGate();

  // Finalize
  qip::finalize();

  // If the ASTObjectTracker is not enabled, this is a no-op.
  QASM::ASTObjectTracker::Instance().Release();

  return 0;
}
