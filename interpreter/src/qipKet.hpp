/// @file qipKet.hpp
/// @brief Header file for the process of calling ket.

#ifndef _QIPKET_HPP_
#define _QIPKET_HPP_

#include <iostream>
#include <vector>
#include <random>
#include <omp.h>
#include <stdio.h>

#include <boost/optional.hpp>
#include <boost/range/empty.hpp>
#include <boost/range/size.hpp>

#include <ket/qubit.hpp>
#include <ket/utility/integer_log2.hpp>
#include <ket/utility/parallel/loop_n.hpp>
#include <ket/mpi/state.hpp>
#include <ket/mpi/qubit_permutation.hpp>
#include <ket/mpi/gate/controlled_not.hpp>
#include <ket/mpi/gate/hadamard.hpp>
#include <ket/mpi/gate/controlled_phase_shift.hpp>
#include <ket/mpi/gate/phase_shift.hpp>
#include <yampi/allocator.hpp>
#include <yampi/rank.hpp>
#include <yampi/communicator.hpp>
#include <yampi/environment.hpp>

#include "qipTypes.h"

namespace qip {

using bitIntegerTye = unsigned int;
using stateIntegerTy = std::uint64_t;
using complexTy = std::complex<double>;
using qubitTy = ket::qubit<stateIntegerTy, bitIntegerTye>;
using permutatedQubitTy = ket::mpi::permutated<qubitTy>;

/// @brief Class for making ket calls
class ketInfo {
public:
  qint              nqubits;         ///< number of quantum bit
  yampi::environment  *environment;  ///< environment
  yampi::communicator *communicator; ///< communicator
  yampi::rank rank;  ///< rank
  yampi::rank root;  ///< root rank
  ket::mpi::state<complexTy, false, yampi::allocator<complexTy>> *localState;  ///< local state
  ket::mpi::qubit_permutation<stateIntegerTy, bitIntegerTye> *permutation;     ///< qubit permutation
  int nprocs;
  int myrank;
};

/// @brief initialization
void initialize();

/// @brief Finalize
void finalize();

/// @brief Gate Application
void addGate();

/// @brief Application of hadamard Gate
/// @param [in] ginfo Gate operation information
void addHGate(gateInfoTy *ginfo);

/// @brief Application of CNOT Gate
/// @param [in] ginfo Gate operation information
void addCXGate(gateInfoTy *ginfo);

/// @brief Application of CZ Gate
/// @param [in] ginfo Gate operation information
void addCZGate(gateInfoTy *ginfo);

/// @brief Application of S Gate
/// @param [in] ginfo Gate operation information
void addSGate(gateInfoTy *ginfo);

/// @brief Application of Sdg Gate
/// @param [in] ginfo Gate operation information
void addSdgGate(gateInfoTy *ginfo);

/// @brief Application of RX Gate
/// @param [in] ginfo Gate operation information
void addRXGate(gateInfoTy *ginfo);

}
#endif // _QIPKET_HPP_
