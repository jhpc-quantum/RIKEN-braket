/// @file qipKet.cpp
/// @brief Process of calling ket.

#include"qipKet.hpp"

namespace qip {

extern qipIrTy qasmir;  ///< Variable that holds information on the gate of a quantum circuit
extern ketInfo ki;  ///< Class declaration for calling ket

}

void qip::initialize() {

  // Perform calculations for parallel execution
  auto const numGqubits = ket::utility::integer_log2<bitIntegerTye>(ki.nprocs);
  ki.nqubits = qasmir.qubits;
  auto const numQubits = bitIntegerTye{(unsigned int) qasmir.qubits};
  auto const numLqubits = numQubits - numGqubits;
  auto const initialStateValue = stateIntegerTy{0u};

  ki.permutation = new ket::mpi::qubit_permutation <stateIntegerTy, bitIntegerTye>{numQubits};
  ki.localState =
      new ket::mpi::state<complexTy, false, yampi::allocator<complexTy>>{numLqubits, initialStateValue,
                                                                               *(qip::ki.permutation),
                                                                               *(qip::ki.communicator),
                                                                               *(qip::ki.environment)};
}

void qip::finalize() {
  // delete
  if (ki.permutation) {
    delete ki.permutation;
  }
  if (ki.localState) {
    delete ki.localState;
  }
  MPI_Finalize();
}

void qip::addGate() {
  int n = qip::qasmir.ngates;

  // Apply all gates
  for (int i = 0; i < n; i++) {
    switch (qip::qasmir.gate[i].id) {
    case HGate:
      addHGate(&qip::qasmir.gate[i]);
      break;
    case CXGate:
      addCXGate(&qip::qasmir.gate[i]);
      break;
    default:
      assert(0 && "Unsupported Gate");
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

void qip::addHGate(gateInfoTy *ginfo) {
  auto buffer = std::vector < complexTy > {};

  // hadamard
  ket::mpi::gate::hadamard(*(ki.localState),
                           qubitTy{bitIntegerTye{(unsigned int) (ginfo->iarg[0])}},
                           *(ki.permutation),
                           buffer,
                           *(ki.communicator),
                           *(ki.environment));
}

void qip::addCXGate(gateInfoTy *ginfo) {
  auto buffer = std::vector < complexTy > {};

  // target bit
  qubitTy target_qubit{bitIntegerTye{(unsigned int) (ginfo->iarg[1])}};
  // control bit
  ket::control <qubitTy> control_qubit{qubitTy{bitIntegerTye{(unsigned int) (ginfo->iarg[0])}}};

  // CNOT
  ket::mpi::gate::controlled_not(*(ki.localState),
                                 target_qubit,
                                 control_qubit,
                                 *(ki.permutation),
                                 buffer,
                                 *(ki.communicator),
                                 *(ki.environment));
}