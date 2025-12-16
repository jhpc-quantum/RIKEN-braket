/// @file qasminterpreter.cpp
/// @brief Performs the main processing of the OpenQASM 3 interpreter.
///
/// Copyright (c) RIKEN, Japan. All rights reserved.

#include "IRGenQASM3Visitor.h"
#include "qipTypes.h"
#include "qipKet.hpp"

#include <qasm/AST/AST.h>
#include <qasm/AST/ASTObjectTracker.h>
#include <qasm/Frontend/QasmParser.h>

#include <iostream>
#include <utility>
#include <cstdint>
#include <cstdlib>
#include <string>


/// @brief Usage
static void Usage() {
  std::cerr << "Usage: qasminterpreter ";
  std::cerr << "[-I<include-dir> [ -I<include-dir> ...]] ";
  std::cerr << "[--print-only n] ";
  std::cerr << "\n                  <translation-unit>" << std::endl;
}
namespace qip {

/// @brief Declare variables that hold information about the gates of the quantum circuit
/// @note At this time, multiple qubit declarations are not supported.
qipIrTy qasmir;
/// @brief Class declaration for calling ket
ketInfo ki;
/// @brief Parse error flag
bool parseErrFlag = false;
}

namespace fs = std::filesystem;

/// @brief Get output file path
/// @param [in] argc number of command line arguments
/// @param [in] argv command line argument
/// @return output file path
/// @note Generate output file paths according to the following rules.
///       - The output file should be in json format
///       - Output file are created in the current directory
///       - The file name of the output file is the input file with the extension changed to “json”
std::string getOutputFile(int argc, char *const argv[]) {
  // Get input filename
  std::string inputFile;
  bool push = false;
  for (int I = 1; I < argc; ++I) {
    if ((argv[I][0] == '-') && (argv[I][1] == 'I')) {
      if (argv[I][2] == '.' || argv[I][2] == '/') {
        push = false;
      } else {
        push = true;
      }
    } else {
      if (push) {
        push = false;
      } else {
        inputFile = argv[I];
      }
    }
  }
  // Generate json file path
  fs::path pInput = inputFile;
  pInput.replace_extension(".json");
  std::string outputFile = "./" + pInput.filename().string<char>();

  return outputFile;
}

std::pair<bool, std::uint64_t> get_print_only(int argc, char* const argv[])
{
  auto exists = false;
  auto is_value_next = false;
  auto print_index = std::uint64_t{};
  for (int I = 1; I < argc; ++I) {
    if (std::string{argv[I]} == "--print-only")
      is_value_next = true;
    else
    {
      if (not is_value_next)
        continue;

      print_index = static_cast<std::uint64_t>(std::atoll(argv[I]));
      exists = true;
      is_value_next = false;
    }
  }

  return {exists, print_index};
}

/// @brief Dumping a quantum circuit IR
/// @note At this time, runtime options to control dump output are not supported.
/// @todo The intermediate code dump should be tied to the input code.
///       - The id should show the gate type
///       - Quantum bit numbers should be displayed in the specified order
///       - Specify dump output as an option
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
      std::cout << "  <rarg[" << j << "]> : " << std::fixed << std::setprecision(20) << qip::qasmir.gate[i].rarg[j] << std::endl;
    }
  }
}

/// @brief Output message and abort when an error occurs
void checkPerseError() {
  if (qip::parseErrFlag) {
    std::cerr << "Error detected in parse. Processing is aborted.\n";
    std::cerr << std::flush;
    abort();
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

  checkPerseError();

  // Quantum circuit IR dump output.
  printSqcIr();

  // Call ket
  // initialization
  qip::initialize();

  // Apply gate
  qip::addGate();

  // Get the path of the output file
  std::string outputFile = getOutputFile(argc, argv);

  // Get print_index if required
  auto const [exists_print_index, print_index] = get_print_only(argc, argv);

  // Output amplitudes
  if (exists_print_index)
    qip::outputAmplitudes(outputFile, print_index);
  else
    qip::outputAmplitudes(outputFile);
  /*
  // Output spin expectation
  qip::outputSpinExpectation(outputFile);
  */

  // Finalize
  qip::finalize();

  // If the ASTObjectTracker is not enabled, this is a no-op.
  QASM::ASTObjectTracker::Instance().Release();

  return 0;
}
