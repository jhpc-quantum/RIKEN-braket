#include <mpi.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include <ket/runtime_control.hpp>
#include <ket/mpi/state.hpp>
#include <ket/mpi/qubit_permutation.hpp>
#include <ket/mpi/utility/simple_mpi.hpp>
#include <ket/mpi/utility/unit_mpi.hpp>
#include <yampi/predefined_datatype.hpp>
#include <ket/utility/loop_n.hpp>
#include <ket/gate/hadamard.hpp>
#include <ket/mpi/gate/hadamard.hpp>
#include <ket/gate/not_.hpp>
#include <ket/mpi/gate/not_.hpp>
#include <ket/mpi/gate/controlled_not.hpp>
#include <ket/mpi/gate/toffoli.hpp>
#include <ket/mpi/gate/controlled_phase_shift.hpp>
#include <ket/gate/pauli_x.hpp>
#include <ket/mpi/gate/pauli_x.hpp>
#include <ket/gate/pauli_y.hpp>
#include <ket/mpi/gate/pauli_y.hpp>
#include <ket/gate/pauli_z.hpp>
#include <ket/mpi/gate/pauli_z.hpp>
#include <ket/gate/sqrt_pauli_x.hpp>
#include <ket/mpi/gate/sqrt_pauli_x.hpp>
#include <ket/gate/sqrt_pauli_y.hpp>
#include <ket/mpi/gate/sqrt_pauli_y.hpp>
#include <ket/gate/sqrt_pauli_z.hpp>
#include <ket/mpi/gate/sqrt_pauli_z.hpp>
#include <ket/gate/x_rotation_half_pi.hpp>
#include <ket/mpi/gate/x_rotation_half_pi.hpp>
#include <ket/gate/y_rotation_half_pi.hpp>
#include <ket/mpi/gate/y_rotation_half_pi.hpp>
#include <ket/gate/phase_shift.hpp>
#include <ket/mpi/gate/phase_shift.hpp>
#include <ket/gate/controlled_v.hpp>
#include <ket/mpi/gate/controlled_v.hpp>
#include <ket/gate/swap.hpp>
#include <ket/mpi/gate/swap.hpp>
#include <ket/gate/exponential_swap.hpp>
#include <ket/mpi/gate/exponential_swap.hpp>
#include <ket/gate/exponential_pauli_x.hpp>
#include <ket/mpi/gate/exponential_pauli_x.hpp>
#include <ket/gate/exponential_pauli_y.hpp>
#include <ket/mpi/gate/exponential_pauli_y.hpp>
#include <ket/gate/exponential_pauli_z.hpp>
#include <ket/mpi/gate/exponential_pauli_z.hpp>

#ifndef KET_MPI_TEST_OPERATION
# define KET_MPI_TEST_OPERATION -1
#endif

#ifndef KET_MPI_TEST_GROUP
# define KET_MPI_TEST_GROUP 1
#endif
#ifndef KET_MPI_TEST_PAGE_QUBITS
# define KET_MPI_TEST_PAGE_QUBITS 2u
#endif

namespace {
using Complex = std::complex<double>;
using UInt = std::uint64_t;
using Qubit = ket::qubit<UInt, unsigned int>;
using Control = ket::runtime_control<Qubit>;
using Positive = ket::control<Qubit>;
using Negative = ket::control<Qubit, 0u>;
using Permutation = ket::mpi::qubit_permutation<UInt, unsigned int>;
constexpr auto local_qubits = 6u;
constexpr auto local_size = std::size_t{1u} << local_qubits;
char const* const names[] = {"hadamard", "adj_hadamard", "not_", "adj_not_", "pauli_x", "adj_pauli_x", "pauli_y", "adj_pauli_y", "pauli_z", "adj_pauli_z", "sqrt_pauli_x", "adj_sqrt_pauli_x", "sqrt_pauli_y", "adj_sqrt_pauli_y", "sqrt_pauli_z", "adj_sqrt_pauli_z", "x_rotation_half_pi", "adj_x_rotation_half_pi", "y_rotation_half_pi", "adj_y_rotation_half_pi", "phase_shift", "adj_phase_shift", "phase_shift_coeff", "adj_phase_shift_coeff", "phase_shift2", "adj_phase_shift2", "phase_shift3", "adj_phase_shift3", "controlled_v", "adj_controlled_v", "controlled_v_coeff", "adj_controlled_v_coeff", "swap", "adj_swap", "exponential_swap", "adj_exponential_swap", "exponential_swap_coeff", "adj_exponential_swap_coeff", "exponential_pauli_x", "adj_exponential_pauli_x", "exponential_pauli_y", "adj_exponential_pauli_y", "exponential_pauli_z", "adj_exponential_pauli_z", "exponential_pauli_x_coeff", "adj_exponential_pauli_x_coeff", "exponential_pauli_y_coeff", "adj_exponential_pauli_y_coeff", "exponential_pauli_z_coeff", "adj_exponential_pauli_z_coeff", "phase_shift", "adj_phase_shift", "phase_shift_coeff", "adj_phase_shift_coeff", "pauli_z", "adj_pauli_z", "sqrt_pauli_z", "adj_sqrt_pauli_z"};
unsigned int const target_counts[] = {1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0};

#ifdef KET_MPI_TEST_ITERATORS
# define KET_MPI_RUNTIME_GATE ::ket::mpi::gate::runtime
# define KET_MPI_TARGET_RANGE t.begin(), t.end()
# define KET_MPI_CONTROL_RANGE c.begin(), c.end()
#else
# define KET_MPI_RUNTIME_GATE ::ket::mpi::gate::runtime::ranges
# define KET_MPI_TARGET_RANGE t
# define KET_MPI_CONTROL_RANGE c
#endif
#ifdef KET_MPI_TEST_DATATYPE
# define KET_MPI_DATATYPE yampi::predefined_datatype<Complex>{},
#else
# define KET_MPI_DATATYPE
#endif

auto mpi_policy()
#ifdef KET_MPI_TEST_UNIT
  -> ket::mpi::utility::policy::unit_mpi<UInt, unsigned, unsigned>
{ return ket::mpi::utility::policy::make_unit_mpi<UInt>(1u, 2u); }
#else
  -> ket::mpi::utility::policy::simple_mpi
{ return ket::mpi::utility::policy::make_simple_mpi(); }
#endif

void reference(int const op, std::vector<Complex>& state,
               std::vector<Qubit> const& t, std::vector<Positive> const& c) {
  auto const seq = ket::utility::policy::make_sequential();
  switch (op) {
#if KET_MPI_TEST_GROUP == 1 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 0)
  case 0: ket::gate::runtime::qubit_ranges::hadamard(seq, state.begin(), state.end(), t[0], c); break;
#endif
#if KET_MPI_TEST_GROUP == 1 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 1)
  case 1: ket::gate::runtime::qubit_ranges::adj_hadamard(seq, state.begin(), state.end(), t[0], c); break;
#endif
#if KET_MPI_TEST_GROUP == 1 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 2)
  case 2: ket::gate::runtime::qubit_ranges::not_(seq, state.begin(), state.end(), t[0], c); break;
#endif
#if KET_MPI_TEST_GROUP == 1 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 3)
  case 3: ket::gate::runtime::qubit_ranges::adj_not_(seq, state.begin(), state.end(), t[0], c); break;
#endif
#if KET_MPI_TEST_GROUP == 1 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 4)
  case 4: ket::gate::runtime::qubit_ranges::pauli_x(seq, state.begin(), state.end(), t, c); break;
#endif
#if KET_MPI_TEST_GROUP == 1 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 5)
  case 5: ket::gate::runtime::qubit_ranges::adj_pauli_x(seq, state.begin(), state.end(), t, c); break;
#endif
#if KET_MPI_TEST_GROUP == 1 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 6)
  case 6: ket::gate::runtime::qubit_ranges::pauli_y(seq, state.begin(), state.end(), t, c); break;
#endif
#if KET_MPI_TEST_GROUP == 1 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 7)
  case 7: ket::gate::runtime::qubit_ranges::adj_pauli_y(seq, state.begin(), state.end(), t, c); break;
#endif
#if KET_MPI_TEST_GROUP == 1 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 8)
  case 8: ket::gate::runtime::qubit_ranges::pauli_z(seq, state.begin(), state.end(), t, c); break;
#endif
#if KET_MPI_TEST_GROUP == 1 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 9)
  case 9: ket::gate::runtime::qubit_ranges::adj_pauli_z(seq, state.begin(), state.end(), t, c); break;
#endif
#if KET_MPI_TEST_GROUP == 1 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 10)
  case 10: ket::gate::runtime::qubit_ranges::sqrt_pauli_x(seq, state.begin(), state.end(), t[0], c); break;
#endif
#if KET_MPI_TEST_GROUP == 1 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 11)
  case 11: ket::gate::runtime::qubit_ranges::adj_sqrt_pauli_x(seq, state.begin(), state.end(), t[0], c); break;
#endif
#if KET_MPI_TEST_GROUP == 1 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 12)
  case 12: ket::gate::runtime::qubit_ranges::sqrt_pauli_y(seq, state.begin(), state.end(), t[0], c); break;
#endif
#if KET_MPI_TEST_GROUP == 1 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 13)
  case 13: ket::gate::runtime::qubit_ranges::adj_sqrt_pauli_y(seq, state.begin(), state.end(), t[0], c); break;
#endif
#if KET_MPI_TEST_GROUP == 1 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 14)
  case 14: ket::gate::runtime::qubit_ranges::sqrt_pauli_z(seq, state.begin(), state.end(), t, c); break;
#endif
#if KET_MPI_TEST_GROUP == 1 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 15)
  case 15: ket::gate::runtime::qubit_ranges::adj_sqrt_pauli_z(seq, state.begin(), state.end(), t, c); break;
#endif
#if KET_MPI_TEST_GROUP == 1 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 16)
  case 16: ket::gate::runtime::qubit_ranges::x_rotation_half_pi(seq, state.begin(), state.end(), t[0], c); break;
#endif
#if KET_MPI_TEST_GROUP == 1 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 17)
  case 17: ket::gate::runtime::qubit_ranges::adj_x_rotation_half_pi(seq, state.begin(), state.end(), t[0], c); break;
#endif
#if KET_MPI_TEST_GROUP == 1 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 18)
  case 18: ket::gate::runtime::qubit_ranges::y_rotation_half_pi(seq, state.begin(), state.end(), t[0], c); break;
#endif
#if KET_MPI_TEST_GROUP == 1 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 19)
  case 19: ket::gate::runtime::qubit_ranges::adj_y_rotation_half_pi(seq, state.begin(), state.end(), t[0], c); break;
#endif
#if KET_MPI_TEST_GROUP == 2 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 20)
  case 20: ket::gate::runtime::qubit_ranges::phase_shift(seq, state.begin(), state.end(), 0.37, t[0], c); break;
#endif
#if KET_MPI_TEST_GROUP == 2 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 21)
  case 21: ket::gate::runtime::qubit_ranges::adj_phase_shift(seq, state.begin(), state.end(), 0.37, t[0], c); break;
#endif
#if KET_MPI_TEST_GROUP == 2 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 22)
  case 22: ket::gate::runtime::qubit_ranges::phase_shift_coeff(seq, state.begin(), state.end(), Complex{0.8, 0.6}, t[0], c); break;
#endif
#if KET_MPI_TEST_GROUP == 2 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 23)
  case 23: ket::gate::runtime::qubit_ranges::adj_phase_shift_coeff(seq, state.begin(), state.end(), Complex{0.8, 0.6}, t[0], c); break;
#endif
#if KET_MPI_TEST_GROUP == 2 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 24)
  case 24: ket::gate::runtime::qubit_ranges::phase_shift2(seq, state.begin(), state.end(), 0.37, -0.21, t[0], c); break;
#endif
#if KET_MPI_TEST_GROUP == 2 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 25)
  case 25: ket::gate::runtime::qubit_ranges::adj_phase_shift2(seq, state.begin(), state.end(), 0.37, -0.21, t[0], c); break;
#endif
#if KET_MPI_TEST_GROUP == 2 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 26)
  case 26: ket::gate::runtime::qubit_ranges::phase_shift3(seq, state.begin(), state.end(), 0.19, 0.37, -0.21, t[0], c); break;
#endif
#if KET_MPI_TEST_GROUP == 2 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 27)
  case 27: ket::gate::runtime::qubit_ranges::adj_phase_shift3(seq, state.begin(), state.end(), 0.19, 0.37, -0.21, t[0], c); break;
#endif
#if KET_MPI_TEST_GROUP == 2 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 28)
  case 28: ket::gate::runtime::qubit_ranges::controlled_v(seq, state.begin(), state.end(), 0.37, t[0], c); break;
#endif
#if KET_MPI_TEST_GROUP == 2 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 29)
  case 29: ket::gate::runtime::qubit_ranges::adj_controlled_v(seq, state.begin(), state.end(), 0.37, t[0], c); break;
#endif
#if KET_MPI_TEST_GROUP == 2 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 30)
  case 30: ket::gate::runtime::qubit_ranges::controlled_v_coeff(seq, state.begin(), state.end(), Complex{0.8, 0.6}, t[0], c); break;
#endif
#if KET_MPI_TEST_GROUP == 2 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 31)
  case 31: ket::gate::runtime::qubit_ranges::adj_controlled_v_coeff(seq, state.begin(), state.end(), Complex{0.8, 0.6}, t[0], c); break;
#endif
#if KET_MPI_TEST_GROUP == 2 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 32)
  case 32: ket::gate::runtime::qubit_ranges::swap(seq, state.begin(), state.end(), t[0], t[1], c); break;
#endif
#if KET_MPI_TEST_GROUP == 2 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 33)
  case 33: ket::gate::runtime::qubit_ranges::adj_swap(seq, state.begin(), state.end(), t[0], t[1], c); break;
#endif
#if KET_MPI_TEST_GROUP == 2 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 34)
  case 34: ket::gate::runtime::qubit_ranges::exponential_swap(seq, state.begin(), state.end(), 0.37, t[0], t[1], c); break;
#endif
#if KET_MPI_TEST_GROUP == 2 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 35)
  case 35: ket::gate::runtime::qubit_ranges::adj_exponential_swap(seq, state.begin(), state.end(), 0.37, t[0], t[1], c); break;
#endif
#if KET_MPI_TEST_GROUP == 2 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 36)
  case 36: ket::gate::runtime::qubit_ranges::exponential_swap_coeff(seq, state.begin(), state.end(), Complex{0.8, 0.6}, t[0], t[1], c); break;
#endif
#if KET_MPI_TEST_GROUP == 2 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 37)
  case 37: ket::gate::runtime::qubit_ranges::adj_exponential_swap_coeff(seq, state.begin(), state.end(), Complex{0.8, 0.6}, t[0], t[1], c); break;
#endif
#if KET_MPI_TEST_GROUP == 3 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 38)
  case 38: ket::gate::runtime::qubit_ranges::exponential_pauli_x(seq, state.begin(), state.end(), 0.37, t, c); break;
#endif
#if KET_MPI_TEST_GROUP == 3 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 39)
  case 39: ket::gate::runtime::qubit_ranges::adj_exponential_pauli_x(seq, state.begin(), state.end(), 0.37, t, c); break;
#endif
#if KET_MPI_TEST_GROUP == 3 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 40)
  case 40: ket::gate::runtime::qubit_ranges::exponential_pauli_y(seq, state.begin(), state.end(), 0.37, t, c); break;
#endif
#if KET_MPI_TEST_GROUP == 3 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 41)
  case 41: ket::gate::runtime::qubit_ranges::adj_exponential_pauli_y(seq, state.begin(), state.end(), 0.37, t, c); break;
#endif
#if KET_MPI_TEST_GROUP == 3 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 42)
  case 42: ket::gate::runtime::qubit_ranges::exponential_pauli_z(seq, state.begin(), state.end(), 0.37, t, c); break;
#endif
#if KET_MPI_TEST_GROUP == 3 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 43)
  case 43: ket::gate::runtime::qubit_ranges::adj_exponential_pauli_z(seq, state.begin(), state.end(), 0.37, t, c); break;
#endif
#if KET_MPI_TEST_GROUP == 3 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 44)
  case 44: ket::gate::runtime::qubit_ranges::exponential_pauli_x_coeff(seq, state.begin(), state.end(), Complex{0.8, 0.6}, t, c); break;
#endif
#if KET_MPI_TEST_GROUP == 3 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 45)
  case 45: ket::gate::runtime::qubit_ranges::adj_exponential_pauli_x_coeff(seq, state.begin(), state.end(), Complex{0.8, 0.6}, t, c); break;
#endif
#if KET_MPI_TEST_GROUP == 3 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 46)
  case 46: ket::gate::runtime::qubit_ranges::exponential_pauli_y_coeff(seq, state.begin(), state.end(), Complex{0.8, 0.6}, t, c); break;
#endif
#if KET_MPI_TEST_GROUP == 3 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 47)
  case 47: ket::gate::runtime::qubit_ranges::adj_exponential_pauli_y_coeff(seq, state.begin(), state.end(), Complex{0.8, 0.6}, t, c); break;
#endif
#if KET_MPI_TEST_GROUP == 3 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 48)
  case 48: ket::gate::runtime::qubit_ranges::exponential_pauli_z_coeff(seq, state.begin(), state.end(), Complex{0.8, 0.6}, t, c); break;
#endif
#if KET_MPI_TEST_GROUP == 3 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 49)
  case 49: ket::gate::runtime::qubit_ranges::adj_exponential_pauli_z_coeff(seq, state.begin(), state.end(), Complex{0.8, 0.6}, t, c); break;
#endif
#if KET_MPI_TEST_GROUP == 3 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 50)
  case 50: ket::gate::runtime::qubit_ranges::phase_shift(seq, state.begin(), state.end(), 0.37, c); break;
#endif
#if KET_MPI_TEST_GROUP == 3 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 51)
  case 51: ket::gate::runtime::qubit_ranges::adj_phase_shift(seq, state.begin(), state.end(), 0.37, c); break;
#endif
#if KET_MPI_TEST_GROUP == 3 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 52)
  case 52: ket::gate::runtime::qubit_ranges::phase_shift_coeff(seq, state.begin(), state.end(), Complex{0.8, 0.6}, c); break;
#endif
#if KET_MPI_TEST_GROUP == 3 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 53)
  case 53: ket::gate::runtime::qubit_ranges::adj_phase_shift_coeff(seq, state.begin(), state.end(), Complex{0.8, 0.6}, c); break;
#endif
#if KET_MPI_TEST_GROUP == 3 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 54)
  case 54: ket::gate::runtime::qubit_ranges::pauli_z(seq, state.begin(), state.end(), c); break;
#endif
#if KET_MPI_TEST_GROUP == 3 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 55)
  case 55: ket::gate::runtime::qubit_ranges::adj_pauli_z(seq, state.begin(), state.end(), c); break;
#endif
#if KET_MPI_TEST_GROUP == 3 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 56)
  case 56: ket::gate::runtime::qubit_ranges::sqrt_pauli_z(seq, state.begin(), state.end(), c); break;
#endif
#if KET_MPI_TEST_GROUP == 3 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 57)
  case 57: ket::gate::runtime::qubit_ranges::adj_sqrt_pauli_z(seq, state.begin(), state.end(), c); break;
#endif
  }
}
template <typename LocalState, typename Controls>
void distributed_ranges(int const op, LocalState& state, Permutation& permutation,
                 std::vector<Complex>& buffer,
                 yampi::communicator const& comm, yampi::environment const& env,
                 std::vector<Qubit> const& t, Controls const& c) {
  auto const mpi = mpi_policy();
  auto const seq = ket::utility::policy::make_sequential();
  switch (op) {
#if KET_MPI_TEST_GROUP == 1 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 0)
  case 0: KET_MPI_RUNTIME_GATE::hadamard(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, t[0], KET_MPI_CONTROL_RANGE); break;
#endif
#if KET_MPI_TEST_GROUP == 1 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 1)
  case 1: KET_MPI_RUNTIME_GATE::adj_hadamard(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, t[0], KET_MPI_CONTROL_RANGE); break;
#endif
#if KET_MPI_TEST_GROUP == 1 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 2)
  case 2: KET_MPI_RUNTIME_GATE::not_(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, t[0], KET_MPI_CONTROL_RANGE); break;
#endif
#if KET_MPI_TEST_GROUP == 1 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 3)
  case 3: KET_MPI_RUNTIME_GATE::adj_not_(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, t[0], KET_MPI_CONTROL_RANGE); break;
#endif
#if KET_MPI_TEST_GROUP == 1 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 4)
  case 4: KET_MPI_RUNTIME_GATE::pauli_x(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, KET_MPI_TARGET_RANGE, KET_MPI_CONTROL_RANGE); break;
#endif
#if KET_MPI_TEST_GROUP == 1 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 5)
  case 5: KET_MPI_RUNTIME_GATE::adj_pauli_x(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, KET_MPI_TARGET_RANGE, KET_MPI_CONTROL_RANGE); break;
#endif
#if KET_MPI_TEST_GROUP == 1 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 6)
  case 6: KET_MPI_RUNTIME_GATE::pauli_y(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, KET_MPI_TARGET_RANGE, KET_MPI_CONTROL_RANGE); break;
#endif
#if KET_MPI_TEST_GROUP == 1 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 7)
  case 7: KET_MPI_RUNTIME_GATE::adj_pauli_y(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, KET_MPI_TARGET_RANGE, KET_MPI_CONTROL_RANGE); break;
#endif
#if KET_MPI_TEST_GROUP == 1 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 8)
  case 8: KET_MPI_RUNTIME_GATE::pauli_z(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, KET_MPI_TARGET_RANGE, KET_MPI_CONTROL_RANGE); break;
#endif
#if KET_MPI_TEST_GROUP == 1 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 9)
  case 9: KET_MPI_RUNTIME_GATE::adj_pauli_z(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, KET_MPI_TARGET_RANGE, KET_MPI_CONTROL_RANGE); break;
#endif
#if KET_MPI_TEST_GROUP == 1 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 10)
  case 10: KET_MPI_RUNTIME_GATE::sqrt_pauli_x(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, t[0], KET_MPI_CONTROL_RANGE); break;
#endif
#if KET_MPI_TEST_GROUP == 1 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 11)
  case 11: KET_MPI_RUNTIME_GATE::adj_sqrt_pauli_x(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, t[0], KET_MPI_CONTROL_RANGE); break;
#endif
#if KET_MPI_TEST_GROUP == 1 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 12)
  case 12: KET_MPI_RUNTIME_GATE::sqrt_pauli_y(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, t[0], KET_MPI_CONTROL_RANGE); break;
#endif
#if KET_MPI_TEST_GROUP == 1 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 13)
  case 13: KET_MPI_RUNTIME_GATE::adj_sqrt_pauli_y(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, t[0], KET_MPI_CONTROL_RANGE); break;
#endif
#if KET_MPI_TEST_GROUP == 1 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 14)
  case 14: KET_MPI_RUNTIME_GATE::sqrt_pauli_z(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, KET_MPI_TARGET_RANGE, KET_MPI_CONTROL_RANGE); break;
#endif
#if KET_MPI_TEST_GROUP == 1 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 15)
  case 15: KET_MPI_RUNTIME_GATE::adj_sqrt_pauli_z(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, KET_MPI_TARGET_RANGE, KET_MPI_CONTROL_RANGE); break;
#endif
#if KET_MPI_TEST_GROUP == 1 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 16)
  case 16: KET_MPI_RUNTIME_GATE::x_rotation_half_pi(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, t[0], KET_MPI_CONTROL_RANGE); break;
#endif
#if KET_MPI_TEST_GROUP == 1 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 17)
  case 17: KET_MPI_RUNTIME_GATE::adj_x_rotation_half_pi(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, t[0], KET_MPI_CONTROL_RANGE); break;
#endif
#if KET_MPI_TEST_GROUP == 1 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 18)
  case 18: KET_MPI_RUNTIME_GATE::y_rotation_half_pi(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, t[0], KET_MPI_CONTROL_RANGE); break;
#endif
#if KET_MPI_TEST_GROUP == 1 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 19)
  case 19: KET_MPI_RUNTIME_GATE::adj_y_rotation_half_pi(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, t[0], KET_MPI_CONTROL_RANGE); break;
#endif
#if KET_MPI_TEST_GROUP == 2 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 20)
  case 20: KET_MPI_RUNTIME_GATE::phase_shift(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, 0.37, t[0], KET_MPI_CONTROL_RANGE); break;
#endif
#if KET_MPI_TEST_GROUP == 2 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 21)
  case 21: KET_MPI_RUNTIME_GATE::adj_phase_shift(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, 0.37, t[0], KET_MPI_CONTROL_RANGE); break;
#endif
#if KET_MPI_TEST_GROUP == 2 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 22)
  case 22: KET_MPI_RUNTIME_GATE::phase_shift_coeff(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, Complex{0.8, 0.6}, t[0], KET_MPI_CONTROL_RANGE); break;
#endif
#if KET_MPI_TEST_GROUP == 2 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 23)
  case 23: KET_MPI_RUNTIME_GATE::adj_phase_shift_coeff(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, Complex{0.8, 0.6}, t[0], KET_MPI_CONTROL_RANGE); break;
#endif
#if KET_MPI_TEST_GROUP == 2 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 24)
  case 24: KET_MPI_RUNTIME_GATE::phase_shift2(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, 0.37, -0.21, t[0], KET_MPI_CONTROL_RANGE); break;
#endif
#if KET_MPI_TEST_GROUP == 2 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 25)
  case 25: KET_MPI_RUNTIME_GATE::adj_phase_shift2(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, 0.37, -0.21, t[0], KET_MPI_CONTROL_RANGE); break;
#endif
#if KET_MPI_TEST_GROUP == 2 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 26)
  case 26: KET_MPI_RUNTIME_GATE::phase_shift3(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, 0.19, 0.37, -0.21, t[0], KET_MPI_CONTROL_RANGE); break;
#endif
#if KET_MPI_TEST_GROUP == 2 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 27)
  case 27: KET_MPI_RUNTIME_GATE::adj_phase_shift3(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, 0.19, 0.37, -0.21, t[0], KET_MPI_CONTROL_RANGE); break;
#endif
#if KET_MPI_TEST_GROUP == 2 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 28)
  case 28: KET_MPI_RUNTIME_GATE::controlled_v(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, 0.37, t[0], KET_MPI_CONTROL_RANGE); break;
#endif
#if KET_MPI_TEST_GROUP == 2 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 29)
  case 29: KET_MPI_RUNTIME_GATE::adj_controlled_v(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, 0.37, t[0], KET_MPI_CONTROL_RANGE); break;
#endif
#if KET_MPI_TEST_GROUP == 2 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 30)
  case 30: KET_MPI_RUNTIME_GATE::controlled_v_coeff(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, Complex{0.8, 0.6}, t[0], KET_MPI_CONTROL_RANGE); break;
#endif
#if KET_MPI_TEST_GROUP == 2 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 31)
  case 31: KET_MPI_RUNTIME_GATE::adj_controlled_v_coeff(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, Complex{0.8, 0.6}, t[0], KET_MPI_CONTROL_RANGE); break;
#endif
#if KET_MPI_TEST_GROUP == 2 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 32)
  case 32: KET_MPI_RUNTIME_GATE::swap(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, t[0], t[1], KET_MPI_CONTROL_RANGE); break;
#endif
#if KET_MPI_TEST_GROUP == 2 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 33)
  case 33: KET_MPI_RUNTIME_GATE::adj_swap(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, t[0], t[1], KET_MPI_CONTROL_RANGE); break;
#endif
#if KET_MPI_TEST_GROUP == 2 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 34)
  case 34: KET_MPI_RUNTIME_GATE::exponential_swap(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, 0.37, t[0], t[1], KET_MPI_CONTROL_RANGE); break;
#endif
#if KET_MPI_TEST_GROUP == 2 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 35)
  case 35: KET_MPI_RUNTIME_GATE::adj_exponential_swap(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, 0.37, t[0], t[1], KET_MPI_CONTROL_RANGE); break;
#endif
#if KET_MPI_TEST_GROUP == 2 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 36)
  case 36: KET_MPI_RUNTIME_GATE::exponential_swap_coeff(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, Complex{0.8, 0.6}, t[0], t[1], KET_MPI_CONTROL_RANGE); break;
#endif
#if KET_MPI_TEST_GROUP == 2 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 37)
  case 37: KET_MPI_RUNTIME_GATE::adj_exponential_swap_coeff(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, Complex{0.8, 0.6}, t[0], t[1], KET_MPI_CONTROL_RANGE); break;
#endif
#if KET_MPI_TEST_GROUP == 3 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 38)
  case 38: KET_MPI_RUNTIME_GATE::exponential_pauli_x(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, 0.37, KET_MPI_TARGET_RANGE, KET_MPI_CONTROL_RANGE); break;
#endif
#if KET_MPI_TEST_GROUP == 3 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 39)
  case 39: KET_MPI_RUNTIME_GATE::adj_exponential_pauli_x(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, 0.37, KET_MPI_TARGET_RANGE, KET_MPI_CONTROL_RANGE); break;
#endif
#if KET_MPI_TEST_GROUP == 3 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 40)
  case 40: KET_MPI_RUNTIME_GATE::exponential_pauli_y(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, 0.37, KET_MPI_TARGET_RANGE, KET_MPI_CONTROL_RANGE); break;
#endif
#if KET_MPI_TEST_GROUP == 3 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 41)
  case 41: KET_MPI_RUNTIME_GATE::adj_exponential_pauli_y(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, 0.37, KET_MPI_TARGET_RANGE, KET_MPI_CONTROL_RANGE); break;
#endif
#if KET_MPI_TEST_GROUP == 3 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 42)
  case 42: KET_MPI_RUNTIME_GATE::exponential_pauli_z(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, 0.37, KET_MPI_TARGET_RANGE, KET_MPI_CONTROL_RANGE); break;
#endif
#if KET_MPI_TEST_GROUP == 3 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 43)
  case 43: KET_MPI_RUNTIME_GATE::adj_exponential_pauli_z(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, 0.37, KET_MPI_TARGET_RANGE, KET_MPI_CONTROL_RANGE); break;
#endif
#if KET_MPI_TEST_GROUP == 3 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 44)
  case 44: KET_MPI_RUNTIME_GATE::exponential_pauli_x_coeff(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, Complex{0.8, 0.6}, KET_MPI_TARGET_RANGE, KET_MPI_CONTROL_RANGE); break;
#endif
#if KET_MPI_TEST_GROUP == 3 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 45)
  case 45: KET_MPI_RUNTIME_GATE::adj_exponential_pauli_x_coeff(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, Complex{0.8, 0.6}, KET_MPI_TARGET_RANGE, KET_MPI_CONTROL_RANGE); break;
#endif
#if KET_MPI_TEST_GROUP == 3 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 46)
  case 46: KET_MPI_RUNTIME_GATE::exponential_pauli_y_coeff(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, Complex{0.8, 0.6}, KET_MPI_TARGET_RANGE, KET_MPI_CONTROL_RANGE); break;
#endif
#if KET_MPI_TEST_GROUP == 3 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 47)
  case 47: KET_MPI_RUNTIME_GATE::adj_exponential_pauli_y_coeff(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, Complex{0.8, 0.6}, KET_MPI_TARGET_RANGE, KET_MPI_CONTROL_RANGE); break;
#endif
#if KET_MPI_TEST_GROUP == 3 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 48)
  case 48: KET_MPI_RUNTIME_GATE::exponential_pauli_z_coeff(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, Complex{0.8, 0.6}, KET_MPI_TARGET_RANGE, KET_MPI_CONTROL_RANGE); break;
#endif
#if KET_MPI_TEST_GROUP == 3 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 49)
  case 49: KET_MPI_RUNTIME_GATE::adj_exponential_pauli_z_coeff(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, Complex{0.8, 0.6}, KET_MPI_TARGET_RANGE, KET_MPI_CONTROL_RANGE); break;
#endif
#if KET_MPI_TEST_GROUP == 3 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 50)
  case 50: KET_MPI_RUNTIME_GATE::phase_shift(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, 0.37, KET_MPI_CONTROL_RANGE); break;
#endif
#if KET_MPI_TEST_GROUP == 3 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 51)
  case 51: KET_MPI_RUNTIME_GATE::adj_phase_shift(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, 0.37, KET_MPI_CONTROL_RANGE); break;
#endif
#if KET_MPI_TEST_GROUP == 3 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 52)
  case 52: KET_MPI_RUNTIME_GATE::phase_shift_coeff(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, Complex{0.8, 0.6}, KET_MPI_CONTROL_RANGE); break;
#endif
#if KET_MPI_TEST_GROUP == 3 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 53)
  case 53: KET_MPI_RUNTIME_GATE::adj_phase_shift_coeff(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, Complex{0.8, 0.6}, KET_MPI_CONTROL_RANGE); break;
#endif
#if KET_MPI_TEST_GROUP == 3 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 54)
  case 54: KET_MPI_RUNTIME_GATE::pauli_z(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, KET_MPI_CONTROL_RANGE); break;
#endif
#if KET_MPI_TEST_GROUP == 3 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 55)
  case 55: KET_MPI_RUNTIME_GATE::adj_pauli_z(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, KET_MPI_CONTROL_RANGE); break;
#endif
#if KET_MPI_TEST_GROUP == 3 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 56)
  case 56: KET_MPI_RUNTIME_GATE::sqrt_pauli_z(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, KET_MPI_CONTROL_RANGE); break;
#endif
#if KET_MPI_TEST_GROUP == 3 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 57)
  case 57: KET_MPI_RUNTIME_GATE::adj_sqrt_pauli_z(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, KET_MPI_CONTROL_RANGE); break;
#endif
  }
}

template <typename LocalState, typename Control0, typename Control1>
void distributed_variadic(int const op, LocalState& state, Permutation& permutation,
                 std::vector<Complex>& buffer,
                 yampi::communicator const& comm, yampi::environment const& env,
                 std::vector<Qubit> const& t, Control0 const c0, Control1 const c1) {
  auto const mpi = mpi_policy();
  auto const seq = ket::utility::policy::make_sequential();
  switch (op) {
#if KET_MPI_TEST_GROUP == 1 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 0)
  case 0: ket::mpi::gate::hadamard(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, t[0], c0, c1); break;
#endif
#if KET_MPI_TEST_GROUP == 1 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 1)
  case 1: ket::mpi::gate::adj_hadamard(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, t[0], c0, c1); break;
#endif
#if KET_MPI_TEST_GROUP == 1 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 2)
#ifdef KET_MPI_TEST_ALIASES
  case 2: ket::mpi::gate::toffoli(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, t[0], c0, c1); break;
#else
  case 2: ket::mpi::gate::not_(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, t[0], c0, c1); break;
#endif
#endif
#if KET_MPI_TEST_GROUP == 1 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 3)
#ifdef KET_MPI_TEST_ALIASES
  case 3: ket::mpi::gate::adj_controlled_not(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, t[0], c0, c1); break;
#else
  case 3: ket::mpi::gate::adj_not_(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, t[0], c0, c1); break;
#endif
#endif
#if KET_MPI_TEST_GROUP == 1 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 4)
  case 4: ket::mpi::gate::pauli_x(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, t[0], t[1], c0, c1); break;
#endif
#if KET_MPI_TEST_GROUP == 1 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 5)
  case 5: ket::mpi::gate::adj_pauli_x(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, t[0], t[1], c0, c1); break;
#endif
#if KET_MPI_TEST_GROUP == 1 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 6)
  case 6: ket::mpi::gate::pauli_y(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, t[0], t[1], c0, c1); break;
#endif
#if KET_MPI_TEST_GROUP == 1 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 7)
  case 7: ket::mpi::gate::adj_pauli_y(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, t[0], t[1], c0, c1); break;
#endif
#if KET_MPI_TEST_GROUP == 1 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 8)
  case 8: ket::mpi::gate::pauli_z(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, t[0], t[1], c0, c1); break;
#endif
#if KET_MPI_TEST_GROUP == 1 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 9)
  case 9: ket::mpi::gate::adj_pauli_z(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, t[0], t[1], c0, c1); break;
#endif
#if KET_MPI_TEST_GROUP == 1 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 10)
  case 10: ket::mpi::gate::sqrt_pauli_x(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, t[0], c0, c1); break;
#endif
#if KET_MPI_TEST_GROUP == 1 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 11)
  case 11: ket::mpi::gate::adj_sqrt_pauli_x(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, t[0], c0, c1); break;
#endif
#if KET_MPI_TEST_GROUP == 1 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 12)
  case 12: ket::mpi::gate::sqrt_pauli_y(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, t[0], c0, c1); break;
#endif
#if KET_MPI_TEST_GROUP == 1 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 13)
  case 13: ket::mpi::gate::adj_sqrt_pauli_y(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, t[0], c0, c1); break;
#endif
#if KET_MPI_TEST_GROUP == 1 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 14)
  case 14: ket::mpi::gate::sqrt_pauli_z(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, t[0], t[1], c0, c1); break;
#endif
#if KET_MPI_TEST_GROUP == 1 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 15)
  case 15: ket::mpi::gate::adj_sqrt_pauli_z(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, t[0], t[1], c0, c1); break;
#endif
#if KET_MPI_TEST_GROUP == 1 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 16)
  case 16: ket::mpi::gate::x_rotation_half_pi(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, t[0], c0, c1); break;
#endif
#if KET_MPI_TEST_GROUP == 1 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 17)
  case 17: ket::mpi::gate::adj_x_rotation_half_pi(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, t[0], c0, c1); break;
#endif
#if KET_MPI_TEST_GROUP == 1 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 18)
  case 18: ket::mpi::gate::y_rotation_half_pi(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, t[0], c0, c1); break;
#endif
#if KET_MPI_TEST_GROUP == 1 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 19)
  case 19: ket::mpi::gate::adj_y_rotation_half_pi(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, t[0], c0, c1); break;
#endif
#if KET_MPI_TEST_GROUP == 2 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 20)
#ifdef KET_MPI_TEST_ALIASES
  case 20: ket::mpi::gate::controlled_phase_shift(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, 0.37, t[0], c0, c1); break;
#else
  case 20: ket::mpi::gate::phase_shift(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, 0.37, t[0], c0, c1); break;
#endif
#endif
#if KET_MPI_TEST_GROUP == 2 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 21)
#ifdef KET_MPI_TEST_ALIASES
  case 21: ket::mpi::gate::adj_controlled_phase_shift(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, 0.37, t[0], c0, c1); break;
#else
  case 21: ket::mpi::gate::adj_phase_shift(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, 0.37, t[0], c0, c1); break;
#endif
#endif
#if KET_MPI_TEST_GROUP == 2 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 22)
#ifdef KET_MPI_TEST_ALIASES
  case 22: ket::mpi::gate::controlled_phase_shift_coeff(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, Complex{0.8, 0.6}, t[0], c0, c1); break;
#else
  case 22: ket::mpi::gate::phase_shift_coeff(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, Complex{0.8, 0.6}, t[0], c0, c1); break;
#endif
#endif
#if KET_MPI_TEST_GROUP == 2 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 23)
#ifdef KET_MPI_TEST_ALIASES
  case 23: ket::mpi::gate::adj_controlled_phase_shift_coeff(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, Complex{0.8, 0.6}, t[0], c0, c1); break;
#else
  case 23: ket::mpi::gate::adj_phase_shift_coeff(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, Complex{0.8, 0.6}, t[0], c0, c1); break;
#endif
#endif
#if KET_MPI_TEST_GROUP == 2 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 24)
  case 24: ket::mpi::gate::phase_shift2(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, 0.37, -0.21, t[0], c0, c1); break;
#endif
#if KET_MPI_TEST_GROUP == 2 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 25)
  case 25: ket::mpi::gate::adj_phase_shift2(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, 0.37, -0.21, t[0], c0, c1); break;
#endif
#if KET_MPI_TEST_GROUP == 2 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 26)
  case 26: ket::mpi::gate::phase_shift3(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, 0.19, 0.37, -0.21, t[0], c0, c1); break;
#endif
#if KET_MPI_TEST_GROUP == 2 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 27)
  case 27: ket::mpi::gate::adj_phase_shift3(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, 0.19, 0.37, -0.21, t[0], c0, c1); break;
#endif
#if KET_MPI_TEST_GROUP == 2 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 28)
  case 28: ket::mpi::gate::controlled_v(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, 0.37, t[0], c0, c1); break;
#endif
#if KET_MPI_TEST_GROUP == 2 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 29)
  case 29: ket::mpi::gate::adj_controlled_v(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, 0.37, t[0], c0, c1); break;
#endif
#if KET_MPI_TEST_GROUP == 2 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 30)
  case 30: ket::mpi::gate::controlled_v_coeff(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, Complex{0.8, 0.6}, t[0], c0, c1); break;
#endif
#if KET_MPI_TEST_GROUP == 2 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 31)
  case 31: ket::mpi::gate::adj_controlled_v_coeff(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, Complex{0.8, 0.6}, t[0], c0, c1); break;
#endif
#if KET_MPI_TEST_GROUP == 2 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 32)
  case 32: ket::mpi::gate::swap(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, t[0], t[1], c0, c1); break;
#endif
#if KET_MPI_TEST_GROUP == 2 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 33)
  case 33: ket::mpi::gate::adj_swap(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, t[0], t[1], c0, c1); break;
#endif
#if KET_MPI_TEST_GROUP == 2 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 34)
  case 34: ket::mpi::gate::exponential_swap(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, 0.37, t[0], t[1], c0, c1); break;
#endif
#if KET_MPI_TEST_GROUP == 2 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 35)
  case 35: ket::mpi::gate::adj_exponential_swap(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, 0.37, t[0], t[1], c0, c1); break;
#endif
#if KET_MPI_TEST_GROUP == 2 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 36)
  case 36: ket::mpi::gate::exponential_swap_coeff(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, Complex{0.8, 0.6}, t[0], t[1], c0, c1); break;
#endif
#if KET_MPI_TEST_GROUP == 2 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 37)
  case 37: ket::mpi::gate::adj_exponential_swap_coeff(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, Complex{0.8, 0.6}, t[0], t[1], c0, c1); break;
#endif
#if KET_MPI_TEST_GROUP == 3 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 38)
  case 38: ket::mpi::gate::exponential_pauli_x(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, 0.37, t[0], t[1], c0, c1); break;
#endif
#if KET_MPI_TEST_GROUP == 3 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 39)
  case 39: ket::mpi::gate::adj_exponential_pauli_x(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, 0.37, t[0], t[1], c0, c1); break;
#endif
#if KET_MPI_TEST_GROUP == 3 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 40)
  case 40: ket::mpi::gate::exponential_pauli_y(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, 0.37, t[0], t[1], c0, c1); break;
#endif
#if KET_MPI_TEST_GROUP == 3 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 41)
  case 41: ket::mpi::gate::adj_exponential_pauli_y(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, 0.37, t[0], t[1], c0, c1); break;
#endif
#if KET_MPI_TEST_GROUP == 3 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 42)
  case 42: ket::mpi::gate::exponential_pauli_z(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, 0.37, t[0], t[1], c0, c1); break;
#endif
#if KET_MPI_TEST_GROUP == 3 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 43)
  case 43: ket::mpi::gate::adj_exponential_pauli_z(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, 0.37, t[0], t[1], c0, c1); break;
#endif
#if KET_MPI_TEST_GROUP == 3 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 44)
  case 44: ket::mpi::gate::exponential_pauli_x_coeff(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, Complex{0.8, 0.6}, t[0], t[1], c0, c1); break;
#endif
#if KET_MPI_TEST_GROUP == 3 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 45)
  case 45: ket::mpi::gate::adj_exponential_pauli_x_coeff(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, Complex{0.8, 0.6}, t[0], t[1], c0, c1); break;
#endif
#if KET_MPI_TEST_GROUP == 3 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 46)
  case 46: ket::mpi::gate::exponential_pauli_y_coeff(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, Complex{0.8, 0.6}, t[0], t[1], c0, c1); break;
#endif
#if KET_MPI_TEST_GROUP == 3 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 47)
  case 47: ket::mpi::gate::adj_exponential_pauli_y_coeff(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, Complex{0.8, 0.6}, t[0], t[1], c0, c1); break;
#endif
#if KET_MPI_TEST_GROUP == 3 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 48)
  case 48: ket::mpi::gate::exponential_pauli_z_coeff(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, Complex{0.8, 0.6}, t[0], t[1], c0, c1); break;
#endif
#if KET_MPI_TEST_GROUP == 3 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 49)
  case 49: ket::mpi::gate::adj_exponential_pauli_z_coeff(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, Complex{0.8, 0.6}, t[0], t[1], c0, c1); break;
#endif
#if KET_MPI_TEST_GROUP == 3 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 50)
#ifdef KET_MPI_TEST_ALIASES
  case 50: ket::mpi::gate::controlled_phase_shift(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, 0.37, c0, c1); break;
#else
  case 50: ket::mpi::gate::phase_shift(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, 0.37, c0, c1); break;
#endif
#endif
#if KET_MPI_TEST_GROUP == 3 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 51)
#ifdef KET_MPI_TEST_ALIASES
  case 51: ket::mpi::gate::adj_controlled_phase_shift(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, 0.37, c0, c1); break;
#else
  case 51: ket::mpi::gate::adj_phase_shift(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, 0.37, c0, c1); break;
#endif
#endif
#if KET_MPI_TEST_GROUP == 3 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 52)
#ifdef KET_MPI_TEST_ALIASES
  case 52: ket::mpi::gate::controlled_phase_shift_coeff(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, Complex{0.8, 0.6}, c0, c1); break;
#else
  case 52: ket::mpi::gate::phase_shift_coeff(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, Complex{0.8, 0.6}, c0, c1); break;
#endif
#endif
#if KET_MPI_TEST_GROUP == 3 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 53)
#ifdef KET_MPI_TEST_ALIASES
  case 53: ket::mpi::gate::adj_controlled_phase_shift_coeff(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, Complex{0.8, 0.6}, c0, c1); break;
#else
  case 53: ket::mpi::gate::adj_phase_shift_coeff(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, Complex{0.8, 0.6}, c0, c1); break;
#endif
#endif
#if KET_MPI_TEST_GROUP == 3 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 54)
  case 54: ket::mpi::gate::pauli_z(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, c0, c1); break;
#endif
#if KET_MPI_TEST_GROUP == 3 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 55)
  case 55: ket::mpi::gate::adj_pauli_z(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, c0, c1); break;
#endif
#if KET_MPI_TEST_GROUP == 3 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 56)
  case 56: ket::mpi::gate::sqrt_pauli_z(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, c0, c1); break;
#endif
#if KET_MPI_TEST_GROUP == 3 && (KET_MPI_TEST_OPERATION == -1 || KET_MPI_TEST_OPERATION == 57)
  case 57: ket::mpi::gate::adj_sqrt_pauli_z(mpi, seq, state, permutation, buffer, KET_MPI_DATATYPE comm, env, c0, c1); break;
#endif
  }
}

template <typename LocalState, typename Controls>
void distributed(int const op, LocalState& state, Permutation& permutation,
                 std::vector<Complex>& buffer,
                 yampi::communicator const& comm, yampi::environment const& env,
                 std::vector<Qubit> const& t, Controls const& c) {
#if defined(KET_MPI_TEST_HOMOGENEOUS)
  if (ket::control_value(c[0]) == 0u and ket::control_value(c[1]) == 0u) {
    std::vector<Negative> negative{Negative{ket::remove_control(c[0])}, Negative{ket::remove_control(c[1])}};
    distributed_ranges(op, state, permutation, buffer, comm, env, t, negative);
  } else if (ket::control_value(c[0]) == 1u and ket::control_value(c[1]) == 1u) {
    std::vector<Positive> positive{Positive{ket::remove_control(c[0])}, Positive{ket::remove_control(c[1])}};
    distributed_ranges(op, state, permutation, buffer, comm, env, t, positive);
  } else distributed_ranges(op, state, permutation, buffer, comm, env, t, c);
#elif defined(KET_MPI_TEST_VARIADIC)
# if KET_MPI_TEST_VARIADIC == 1
  auto const q0 = ket::remove_control(c[0]), q1 = ket::remove_control(c[1]);
  if (ket::control_value(c[0]) == 0u) {
    if (ket::control_value(c[1]) == 0u)
      distributed_variadic(op, state, permutation, buffer, comm, env, t, Negative{q0}, Negative{q1});
    else distributed_variadic(op, state, permutation, buffer, comm, env, t, Negative{q0}, Positive{q1});
  } else {
    if (ket::control_value(c[1]) == 0u)
      distributed_variadic(op, state, permutation, buffer, comm, env, t, Positive{q0}, Negative{q1});
    else distributed_variadic(op, state, permutation, buffer, comm, env, t, Positive{q0}, Positive{q1});
  }
# else
  distributed_variadic(op, state, permutation, buffer, comm, env, t, c[0], c[1]);
# endif
#else
  distributed_ranges(op, state, permutation, buffer, comm, env, t, c);
#endif
}

template <typename LocalState, typename Controls>
bool check(int const op, LocalState& state, Controls const& controls,
           std::vector<Qubit> const& targets, std::vector<Complex> expected,
           unsigned const total_qubits, yampi::communicator const& comm,
           yampi::environment const& env) {
  auto permutation = Permutation{total_qubits};
  auto const rank = static_cast<int>(comm.rank(env));
  std::copy(expected.begin() + rank * local_size,
            expected.begin() + (rank + 1) * local_size, state.begin());
  std::vector<Positive> positive;
  for (auto const control : controls) {
    auto const wire = ket::remove_control(control);
    positive.emplace_back(wire);
    if (ket::control_value(control) == 0u)
      ket::gate::not_(expected.begin(), expected.end(), wire);
  }
  reference(op, expected, targets, positive);
  for (auto const control : controls)
    if (ket::control_value(control) == 0u)
      ket::gate::not_(expected.begin(), expected.end(), ket::remove_control(control));
  std::vector<Complex> buffer;
  distributed(op, state, permutation, buffer, comm, env, targets, controls);
  // Page iterators represent the logical local order even after page exchanges.
  std::vector<Complex> local(state.begin(), state.end()), gathered(expected.size());
  MPI_Allgather(local.data(), static_cast<int>(local_size * 2u), MPI_DOUBLE,
                gathered.data(), static_cast<int>(local_size * 2u), MPI_DOUBLE, MPI_COMM_WORLD);
  for (UInt index = 0; index < gathered.size(); ++index) {
    auto const logical_index = ket::mpi::inverse_permutate_bits(permutation, index);
    if (std::abs(gathered[index] - expected[logical_index]) > 1e-11) {
      if (rank == 0)
      {
        std::cerr << names[op] << " failed at " << index << "; targets";
        for (auto q : targets) std::cerr << ' ' << static_cast<unsigned>(q);
        std::cerr << "; controls";
        for (auto c : controls) std::cerr << ' ' << static_cast<unsigned>(ket::remove_control(c)) << ':' << ket::control_value(c);
        std::cerr << "; page " << ket::mpi::page::is_on_page(permutation[Qubit{4u}], state) << "; permutation";
        for (auto i = 0u; i < total_qubits; ++i) std::cerr << ' ' << static_cast<unsigned>(permutation[Qubit{i}].qubit());
        std::cerr << "; actual " << gathered[index] << "; expected " << expected[logical_index] << '\n';
      }
      return false;
    }
  }
  for (auto const control : controls)
    if (ket::mpi::control_value(permutation[control]) != ket::control_value(control))
      return false;
  return true;
}

bool permutation_api() {
  auto p = Permutation{8u};
  p.permutate(Qubit{1u}, Qubit{6u});
  Negative const negative{1u};
  auto const dynamic = ket::make_runtime_control(Qubit{1u}, 0u);
  auto const mapped = p[negative];
  if (ket::mpi::remove_control(mapped).qubit() != Qubit{6u}
      or ket::mpi::control_value(mapped) != 0u
      or ket::mpi::control_value(p.at(dynamic)) != 0u
      or ket::mpi::inverse(p)[mapped] != negative
      or ket::control_value(ket::mpi::inverse(p).at(p.at(dynamic))) != 0u)
    return false;
  p.permutate(negative, Positive{2u});
  p.permutate(Qubit{3u}, dynamic);
  p.permutate(dynamic, Negative{4u});
  return ket::control_value(ket::mpi::inverse(p)[p[dynamic]]) == 0u;
}
} // namespace

int main(int argc, char** argv) {
  yampi::environment env(argc, argv);
  yampi::communicator comm{yampi::tags::world_communicator};
  auto const ranks = comm.size(env);
  auto rank_bits = 0u;
  for (auto count = static_cast<int>(ranks); count > 1; count >>= 1) ++rank_bits;
  if ((1 << rank_bits) != static_cast<int>(ranks) or rank_bits == 0u) return EXIT_FAILURE;
  auto const total_qubits = local_qubits + rank_bits;
  auto const size = std::size_t{1u} << total_qubits;
  auto const basis_tests = argc > 1 ? std::min(size, std::size_t{std::stoul(argv[1])}) : 0u;
  auto layouts = std::vector<std::array<unsigned, 4>>{
    {{0, 1, 2, 3}}, {{4, 0, 2, 1}}, {{0, 1, 4, 5}},
    {{6, 1, 4, 2}}, {{0, 1, 6, 4}}, {{4, 1, 6, 2}}, {{6, 0, 4, 5}}
  };
  if (rank_bits > 1u) { layouts.push_back({{0, 1, 7, 6}}); layouts.push_back({{7, 1, 6, 4}}); }
  bool passed = permutation_api();
  std::size_t comparisons = 0;
  for (int op = 0; op < static_cast<int>(sizeof(names) / sizeof(*names)); ++op)
    if ((KET_MPI_TEST_OPERATION == -1 or op == KET_MPI_TEST_OPERATION))
    if ((op < 20 ? 1 : op < 38 ? 2 : 3) == KET_MPI_TEST_GROUP)
    for (auto const& layout : layouts)
      for (auto polarity = 0u; polarity < 4u; ++polarity)
        for (auto basis = std::size_t{0}; basis <= basis_tests; ++basis) {
          std::vector<Complex> initial(size);
          if (basis < basis_tests) initial[basis] = Complex{1, 0};
          else for (auto i = std::size_t{0}; i < size; ++i)
            initial[i] = Complex{0.013 * static_cast<double>(i + 1), -0.017 * static_cast<double>((i * 7 + 3) % 13)};
          std::vector<Qubit> targets;
          for (auto i = 0u; i < target_counts[op]; ++i) targets.emplace_back(layout[i]);
          std::vector<Control> controls;
          controls.emplace_back(Qubit{layout[2]}, polarity & 1u);
          controls.emplace_back(Qubit{layout[3]}, (polarity >> 1u) & 1u);
          auto state = std::vector<Complex>(local_size);
          passed = check(op, state, controls, targets, initial, total_qubits, comm, env) and passed;
          auto permutation = Permutation{total_qubits};
          auto page_state = ket::mpi::state<Complex, true>{local_qubits, unsigned{KET_MPI_TEST_PAGE_QUBITS}, UInt{0}, permutation, comm, env};
          passed = check(op, page_state, controls, targets, initial, total_qubits, comm, env) and passed;
          comparisons += 2;
        }
  if (static_cast<int>(comm.rank(env)) == 0)
    std::cout << comparisons << " MPI control state comparisons " << (passed ? "passed" : "FAILED") << '\n';
  return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
