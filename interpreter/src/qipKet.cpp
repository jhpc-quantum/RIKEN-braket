/// @file qipKet.cpp
/// @brief Process of calling ket.
///
/// Copyright (c) RIKEN, Japan. All rights reserved.

#include"qipKet.hpp"

#include <cmath>
#include <sstream>
#include <string>
#include <vector>
#include <ket/mpi/print_amplitudes.hpp>
#include <bra/state.hpp>
#include <yampi/rank.hpp>
//#include <boost/algorithm/string/trim.hpp>
//#include <boost/algorithm/string/split.hpp>
//#include <boost/algorithm/string/classification.hpp>
#include <cstdint>

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
                           *(ki.permutation),
                           buffer,
                           *(ki.communicator),
                           *(ki.environment),
                           qubitTy{bitIntegerTye{(unsigned int) (ginfo->iarg[0])}});
}

void qip::addCXGate(gateInfoTy *ginfo) {
  auto buffer = std::vector < complexTy > {};

  // target bit
  qubitTy target_qubit{bitIntegerTye{(unsigned int) (ginfo->iarg[1])}};
  // control bit
  ket::control <qubitTy> control_qubit{qubitTy{bitIntegerTye{(unsigned int) (ginfo->iarg[0])}}};

  // CNOT
  ket::mpi::gate::pauli_x(*(ki.localState),
                          *(ki.permutation),
                          buffer,
                          *(ki.communicator),
                          *(ki.environment)
                          target_qubit,
                          control_qubit);
}

void qip::addCZGate(gateInfoTy *ginfo) {
  auto buffer = std::vector < complexTy > {};

  // target bit
  ket::control <qubitTy> control_qubit1{qubitTy{bitIntegerTye{(unsigned int) (ginfo->iarg[1])}}};
  // control bit
  ket::control <qubitTy> control_qubit2{qubitTy{bitIntegerTye{(unsigned int) (ginfo->iarg[0])}}};

  // cz
  ket::mpi::gate::pauli_z(*(ki.localState),
                          *(ki.permutation),
                          buffer,
                          *(ki.communicator),
                          *(ki.environment),
                          control_qubit1,
                          control_qubit2);

}

void qip::addSGate(gateInfoTy *ginfo) {
  auto buffer = std::vector < complexTy > {};

  // s
  ket::mpi::gate::phase_shift(*(ki.localState),
                              *(ki.permutation),
                              buffer,
                              *(ki.communicator),
                              *(ki.environment),
                              (M_PI*0.5),
                              ket::control<qubitTy>{bitIntegerTye{(unsigned int)(ginfo->iarg[0])}});

}

void qip::addSdgGate(gateInfoTy *ginfo) {
  auto buffer = std::vector < complexTy > {};

  // sdg
  ket::mpi::gate::phase_shift(*(ki.localState),
                              *(ki.permutation),
                              buffer,
                              *(ki.communicator),
                              *(ki.environment),
                              (-M_PI*0.5),
                              ket::control<qubitTy>{bitIntegerTye{(unsigned int)(ginfo->iarg[0])}});

}

void qip::addRXGate(gateInfoTy *ginfo) {
  auto buffer = std::vector < complexTy > {};

  double theta = ginfo->rarg[0];

  // rx
  ket::mpi::gate::exponential_pauli_x(*(ki.localState),
                                      *(ki.permutation),
                                      buffer,
                                      *(ki.communicator),
                                      *(ki.environment),
                                      -theta/2.0,
                                      qubitTy{bitIntegerTye{(unsigned int)(ginfo->iarg[0])}});

}

void qip::addRYGate(gateInfoTy *ginfo) {
  auto buffer = std::vector < complexTy > {};

  double theta = ginfo->rarg[0];

  // ry
  ket::mpi::gate::exponential_pauli_y(*(ki.localState),
                                      *(ki.permutation),
                                      buffer,
                                      *(ki.communicator),
                                      *(ki.environment),
                                      -theta/2.0,
                                      qubitTy{bitIntegerTye{(unsigned int)(ginfo->iarg[0])}});

}

void qip::addRZGate(gateInfoTy *ginfo) {
  auto buffer = std::vector < complexTy > {};

  double theta = ginfo->rarg[0];

  // rz
  ket::mpi::gate::exponential_pauli_z(*(ki.localState),
                                      *(ki.permutation),
                                      buffer,
                                      *(ki.communicator),
                                      *(ki.environment),
                                      -theta/2.0,
                                      qubitTy{bitIntegerTye{(unsigned int)(ginfo->iarg[0])}});

}

void qip::addXGate(gateInfoTy *ginfo) {
  auto buffer = std::vector < complexTy > {};

  // x
  ket::mpi::gate::pauli_x(*(ki.localState),
                          *(ki.permutation),
                          buffer,
                          *(ki.communicator),
                          *(ki.environment),
                          qubitTy{bitIntegerTye{(unsigned int)(ginfo->iarg[0])}});

}

void qip::addU1Gate(gateInfoTy *ginfo) {
  auto buffer = std::vector < complexTy > {};

  double theta = ginfo->rarg[0];

  // u1
  ket::mpi::gate::phase_shift(*(ki.localState),
                              *(ki.permutation),
                              buffer,
                              *(ki.communicator),
                              *(ki.environment),
                              theta,
                              ket::control<qubitTy>{bitIntegerTye{(unsigned int)(ginfo->iarg[0])}});
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

/// @note Amplitudes is printed with reference to the bra process.
/// @note The information to be output in json format is as follows.
///       - "bits" : Bit string
///       - "real", "imag" : Real and imaginary part of amplitude
/// Example)
/// ```
/// {
///     "Amplitudes": [
///         {
///             "bits": "000",
///             "real": 1.0,
///             "imag": 0.0
///         },
///         {
///             "bits": "001",
///             "real": 0.0,
///             "imag": 1.0
///         }
///     ]
/// }
/// ```
void qip::outputAmplitudes(std::string const& outputFile) {
  const unsigned numQubits = qasmir.qubits;

  using std::begin;
  auto const present_rank = (qip::ki.communicator)->rank(*(ki.environment));

  std::ofstream ofs;
  using namespace yampi::literals::rank_literals;
  if (present_rank == 0_r)
  {
    ofs.open(outputFile);
    ofs << "{\n    \"Amplitudes\": [\n" << std::flush;
  }

  ket::mpi::println_amplitudes(
    ofs, *(ki.localState), *(qip::ki.permutation), 0_r, *(qip::ki.communicator), *(ki.environment),
    [numQubits](auto const qubit_value, auto const& amplitude)
    {
      std::ostringstream oss;
      using std::real;
      using std::imag;
      oss << "        {\n            \"bits\": " << ::bra::state_detail::integer_to_bits_string(qubit_value, numQubits) << ",\n            \"real\": " << real(amplitude) << ",\n            \"imag\": " << imag(amplitude) << "\n        }";
      return oss.str();
    }, std::string{",\n"});

  if (present_rank == 0_r)
    ofs << "\n    ]\n}" << std::flush;
  /*
  std::ostringstream oss;
  using namespace yampi::literals::rank_literals;
  ket::mpi::println_amplitudes(
    oss, *(ki.localState), *(qip::ki.permutation), 0_r, *(qip::ki.communicator), *(ki.environment),
    [numQubits](auto const qubit_value, auto const& amplitude)
    {
      std::ostringstream oss;
      using std::real;
      using std::imag;
      oss << ::bra::state_detail::integer_to_bits_string(qubit_value, numQubits) << ' ' << real(amplitude) << ' ' << imag(amplitude);
      return oss.str();
    }, std::string{"\n"});

  if (present_rank != 0_r)
    return;

  std::istringstream iss{oss.str()};

  auto line = std::string{};
  auto columns = std::vector<std::string>{};
  columns.reserve(3u);

  JSON pt;
  JSON child;
  while (std::getline(iss, line))
  {
    if (line.empty())
      continue;

    boost::algorithm::trim(line);
    if (line.empty())
      continue;

    boost::algorithm::split(columns, line, boost::algorithm::is_space(), boost::algorithm::token_compress_on);

    if (columns.size() != 3u)
      continue;

    JSON info;
    info.put("bits", columns[0]);
    info.put("real", columns[1]);
    info.put("imag", columns[2]);
    child.push_back(std::make_pair("", info));
  }
  pt.add_child("Amplitudes", child);
  write_jsonEx(outputFile, pt);
  */
}

void qip::outputAmplitudes(std::string const& outputFile, std::uint64_t const print_index) {
  const unsigned numQubits = qasmir.qubits;

  using std::begin;
  auto const first = begin(*(ki.localState));
  auto const present_rank = (qip::ki.communicator)->rank(*(ki.environment));

  std::ofstream ofs;
  using namespace yampi::literals::rank_literals;
  if (present_rank == 0_r)
  {
    ofs.open(outputFile);
    ofs << "{\n    \"Amplitudes\": [\n" << std::flush;
  }

  auto const rank_index
    = ::ket::mpi::utility::qubit_value_to_rank_index(
        ket::mpi::utility::policy::make_simple_mpi(), *(ki.localState),
        ket::mpi::permutate_bits(*(qip::ki.permutation), print_index),
        *(qip::ki.communicator), *(ki.environment));

  if (present_rank == 0_r)
  {
    auto amplitude = complexTy{};

    if (present_rank == rank_index.first)
      amplitude = *(first + rank_index.second);
    else
      yampi::receive(yampi::ignore_status, yampi::make_buffer(amplitude), rank_index.first, yampi::tag{static_cast<int>(rank_index.second)}, *(qip::ki.communicator), *(ki.environment));

    using std::real;
    using std::imag;
    ofs << "        {\n            \"bits\": " << ::bra::state_detail::integer_to_bits_string(print_index, numQubits) << ",\n            \"real\": " << real(amplitude) << ",\n            \"imag\": " << imag(amplitude) << "\n        }" << std::flush;
  }
  else if (present_rank == rank_index.first)
    yampi::send(yampi::make_buffer(*(first + rank_index.second)), 0_r, yampi::tag{static_cast<int>(rank_index.second)}, *(qip::ki.communicator), *(ki.environment));

  if (present_rank == 0_r)
    ofs << "\n    ]\n}" << std::flush;
}
