
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

/// @brief ket呼び出しを行うためのクラス
class ketInfo {
public:
  qint              nqubits;         ///< 量子ビット数
  yampi::environment  *environment;
  yampi::communicator *communicator;
  yampi::rank rank;
  yampi::rank root;
  ket::mpi::state<complexTy, false, yampi::allocator<complexTy>> *localState;
  ket::mpi::qubit_permutation<stateIntegerTy, bitIntegerTye> *permutation;
  int nprocs;
  int myrank;
};

/// @brief 初期化
void initialize();

/// @brief 終了処理
void finalize();

/// @brief ゲート適用
void addGate();

/// @brief アダマールゲート適用
/// @param [in] ginfo ゲート操作の情報
void addHGate(gateInfoTy *ginfo);

/// @brief CNOTゲート適用
/// @param [in] ginfo ゲート操作の情報
void addCXGate(gateInfoTy *ginfo);

}
#endif // _QIPKET_HPP_
