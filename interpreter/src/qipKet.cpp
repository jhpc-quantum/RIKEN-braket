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
    case CZGate:
      addCZGate(&qip::qasmir.gate[i]);
      break;
    case SGate:
      addSGate(&qip::qasmir.gate[i]);
      break;
    case SdgGate:
      addSdgGate(&qip::qasmir.gate[i]);
      break;
    case RXGate:
      addRXGate(&qip::qasmir.gate[i]);
      break;
    case RYGate:
      addRYGate(&qip::qasmir.gate[i]);
      break;
    case RZGate:
      addRZGate(&qip::qasmir.gate[i]);
      break;
    case XGate:
      addXGate(&qip::qasmir.gate[i]);
      break;
    case U1Gate:
      addU1Gate(&qip::qasmir.gate[i]);
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

void qip::addCZGate(gateInfoTy *ginfo) {
  auto buffer = std::vector < complexTy > {};

  // target bit
  qubitTy target_qubit{bitIntegerTye{(unsigned int) (ginfo->iarg[1])}};
  // control bit
  ket::control <qubitTy> control_qubit{qubitTy{bitIntegerTye{(unsigned int) (ginfo->iarg[0])}}};

  // cz
  ket::mpi::gate::controlled_phase_shift(*(ki.localState),
                                         M_PI, target_qubit,
                                         control_qubit,
                                         *(ki.permutation),
                                         buffer,
                                         *(ki.communicator),
                                         *(ki.environment));

}

void qip::addSGate(gateInfoTy *ginfo) {
  auto buffer = std::vector < complexTy > {};

  // target bit
  qubitTy target_qubit{bitIntegerTye{(unsigned int) (ginfo->iarg[1])}};
  // control bit
  ket::control <qubitTy> control_qubit{qubitTy{bitIntegerTye{(unsigned int) (ginfo->iarg[0])}}};

  // s
  ket::mpi::gate::phase_shift(*(ki.localState),
                              (M_PI*0.5),
                              qubitTy{bitIntegerTye{(unsigned int)(ginfo->iarg[0])}},
                              *(ki.permutation),
                              buffer,
                              *(ki.communicator),
                              *(ki.environment));

}

void qip::addSdgGate(gateInfoTy *ginfo) {
  auto buffer = std::vector < complexTy > {};

  // target bit
  qubitTy target_qubit{bitIntegerTye{(unsigned int) (ginfo->iarg[1])}};
  // control bit
  ket::control <qubitTy> control_qubit{qubitTy{bitIntegerTye{(unsigned int) (ginfo->iarg[0])}}};

  // sdg
  ket::mpi::gate::phase_shift(*(ki.localState),
                             (-M_PI*0.5),
                              qubitTy{bitIntegerTye{(unsigned int)(ginfo->iarg[0])}},
                              *(ki.permutation),
                              buffer,
                              *(ki.communicator),
                              *(ki.environment));

}

void qip::addRXGate(gateInfoTy *ginfo) {
  auto buffer = std::vector < complexTy > {};

  // target bit
  qubitTy target_qubit{bitIntegerTye{(unsigned int) (ginfo->iarg[1])}};
  // control bit
  ket::control <qubitTy> control_qubit{qubitTy{bitIntegerTye{(unsigned int) (ginfo->iarg[0])}}};

  double theta = ginfo->rarg[0];

  // rx
  ket::mpi::gate::phase_shift3(*(ki.localState),
                               theta,
                               -M_PI/2.0,
                               M_PI/2.0,
                               target_qubit,
                               *(ki.permutation),
                               buffer,
                               *(ki.communicator),
                               *(ki.environment));

}

void qip::addRYGate(gateInfoTy *ginfo) {
  auto buffer = std::vector < complexTy > {};

  // target bit
  qubitTy target_qubit{bitIntegerTye{(unsigned int) (ginfo->iarg[1])}};
  // control bit
  ket::control <qubitTy> control_qubit{qubitTy{bitIntegerTye{(unsigned int) (ginfo->iarg[0])}}};

  double theta = ginfo->rarg[0];

  // ry
  ket::mpi::gate::phase_shift3(*(ki.localState),
                               theta,
                               0.0,
                               0.0,
                               target_qubit,
                               *(ki.permutation),
                               buffer,
                               *(ki.communicator),
                               *(ki.environment));

}

/// @attention RZ gate is not supported. The process of applying coefficients is not yet implemented.
void qip::addRZGate(gateInfoTy *ginfo) {
  std::cerr << "RZ gate is not supported.\n";
  std::cout << std::flush;
  abort();
  // The rz gate shall be unsupported.
  // Because it is unclear how to realize “gate rz(λ) a { gphase(-λ/2); U(0, 0, λ) a; }” in ket.
#if 0
  auto buffer = std::vector < complexTy > {};

  // target bit
  qubitTy target_qubit{bitIntegerTye{(unsigned int) (ginfo->iarg[1])}};
  // control bit
  ket::control <qubitTy> control_qubit{qubitTy{bitIntegerTye{(unsigned int) (ginfo->iarg[0])}}};

  double theta = ginfo->rarg[0];

  // rz
  ket::mpi::gate::phase_shift(*(ki.localState),
                              theta,
                              target_qubit,
                              *(ki.permutation),
                              buffer,
                              *(ki.communicator),
                              *(ki.environment));
  complexTy c{cos(theta/2.0),-sin(theta/2.0)};
  ket::mpi::gate::mult(*(ki.localState), c, *(ki.permutation), buffer, *(ki.communicator), *(ki.environment));
#endif
}

void qip::addXGate(gateInfoTy *ginfo) {
  auto buffer = std::vector < complexTy > {};

  // target bit
  qubitTy target_qubit{bitIntegerTye{(unsigned int) (ginfo->iarg[1])}};
  // control bit
  ket::control <qubitTy> control_qubit{qubitTy{bitIntegerTye{(unsigned int) (ginfo->iarg[0])}}};

  // x
  ket::mpi::gate::pauli_x(*(ki.localState),
                          qubitTy{bitIntegerTye{(unsigned int)(ginfo->iarg[0])}},
                          *(ki.permutation),
                          buffer,
                          *(ki.communicator),
                          *(ki.environment));

}

void qip::addU1Gate(gateInfoTy *ginfo) {
  auto buffer = std::vector < complexTy > {};

  // target bit
  qubitTy target_qubit{bitIntegerTye{(unsigned int) (ginfo->iarg[1])}};
  // control bit
  ket::control <qubitTy> control_qubit{qubitTy{bitIntegerTye{(unsigned int) (ginfo->iarg[0])}}};

  double theta = ginfo->rarg[0];

  // u1
  ket::mpi::gate::phase_shift(*(ki.localState),
                              theta, target_qubit,
                              *(ki.permutation),
                              buffer,
                              *(ki.communicator),
                              *(ki.environment));

}

namespace bpt = boost::property_tree;
typedef bpt::ptree JSON;
namespace boost { namespace property_tree {
/// @brief Output numbers in json format
/// @param [in] path File path
/// @param [in] ptree Property tree 
    inline void write_jsonEx(const std::string & path, const JSON & ptree)
    {
        std::ostringstream oss;
        bpt::write_json(oss, ptree);
        std::regex reg("\\\"([0-9]+\\.{0,1}[0-9]*)\\\"");
        std::string result = std::regex_replace(oss.str(), reg, "$1");

        std::ofstream file;
        file.open(path);
        file << result;
        file.close();
    }
} }

using spin_type = std::array<double, 3u>;
using spins_allocator_type = yampi::allocator<spin_type>;

/// @note Spin expectation calculation is performed with reference to the bra process.
/// @note The information to be output in json format is as follows.
///       - "qubit" : Quantum bit number
///       - "Qx", "Qy", "Qz" : Spin Expectation
/// Example)
/// ```
/// {
///     "Expectation values of spins": [
///         {
///             "qubit": 0,
///             "Qx": 0.5,
///             "Qy": 0.5,
///             "Qz": 0.0
///         },
///         {
///             "qubit": 1,
///             "Qx": 0.5,
///             "Qy": 0.5,
///             "Qz": 0.0
///         }
///     ]
/// }
/// ```
void qip::outputSpinExpectation(std::string outputFile) {
  const unsigned numQubits = qasmir.qubits;
  auto buffer = std::vector < complexTy > {};
  auto expectation_values = ket::mpi::all_spin_expectation_values<spins_allocator_type>(
          *(ki.localState), *(qip::ki.permutation), numQubits, buffer, *(qip::ki.communicator), *(ki.environment));

  qint i = 0;
  JSON pt;
  JSON child;
  for (auto const& spin: expectation_values) {
    {
      JSON info;
      info.put("qubit", i);
      info.put("Qx", 0.5 - static_cast<double>(spin[0u]));
      info.put("Qy", 0.5 - static_cast<double>(spin[1u]));
      info.put("Qz", 0.5 - static_cast<double>(spin[2u]));
      child.push_back(std::make_pair("", info));
    }
    i++;
  }
  pt.add_child("Expectation values of spins", child);
  write_jsonEx(outputFile, pt);

}
