#include <mpi.h>

// Example:
//   mpicxx -std=c++14 -DNDEBUG -Iket/include -I../yampi/include \
//     ket/test/mpi/fused_runtime_gate_numerical.cpp -o /tmp/fused_runtime_gate_numerical
//   mpiexec -n 2 /tmp/fused_runtime_gate_numerical

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <ket/control.hpp>
#include <ket/gate/exponential_pauli_x.hpp>
#include <ket/gate/exponential_pauli_z.hpp>
#include <ket/gate/exponential_swap.hpp>
#include <ket/gate/fused/exponential_pauli_x.hpp>
#include <ket/gate/fused/exponential_pauli_z.hpp>
#include <ket/gate/fused/exponential_swap.hpp>
#include <ket/gate/fused/hadamard.hpp>
#include <ket/gate/fused/pauli_x.hpp>
#include <ket/gate/fused/swap.hpp>
#include <ket/gate/hadamard.hpp>
#include <ket/gate/pauli_x.hpp>
#include <ket/gate/swap.hpp>
#include <ket/mpi/gate/gate.hpp>
#include <ket/mpi/qubit_permutation.hpp>
#include <ket/mpi/state.hpp>
#include <ket/mpi/utility/simple_mpi.hpp>
#include <ket/qubit.hpp>
#include <ket/utility/loop_n.hpp>
#include <yampi/communicator.hpp>
#include <yampi/environment.hpp>

namespace
{
  using complex_type = std::complex<double>;
  using state_integer_type = std::uint64_t;
  using bit_integer_type = unsigned int;
  using qubit_type = ket::qubit<state_integer_type, bit_integer_type>;
  using control_qubit_type = ket::control<qubit_type>;
  using permutation_type = ket::mpi::qubit_permutation<state_integer_type, bit_integer_type>;

  using namespace ket::literals::control_literals;
  using namespace ket::literals::qubit_literals;
  using namespace yampi::literals::rank_literals;

  constexpr auto total_qubits = bit_integer_type{4u};
  constexpr auto local_qubits = bit_integer_type{3u};
  constexpr auto total_state_size = std::size_t{1u} << total_qubits;
  constexpr auto local_state_size = std::size_t{1u} << local_qubits;

  auto initial_state() -> std::vector<complex_type>
  {
    auto result = std::vector<complex_type>(total_state_size);
    for (auto index = std::size_t{0u}; index < result.size(); ++index)
      result[index] = complex_type{
        0.125 * static_cast<double>(index + 1u),
        -0.0625 * static_cast<double>((index * 3u + 1u) % 7u)};
    return result;
  }

  auto local_slice(std::vector<complex_type> const& full_state, yampi::rank const rank)
    -> std::vector<complex_type>
  {
    auto const rank_index = static_cast<int>(rank);
    auto result = std::vector<complex_type>(local_state_size);
    std::copy(
      full_state.begin() + rank_index * local_state_size,
      full_state.begin() + (rank_index + 1) * local_state_size,
      result.begin());
    return result;
  }

  auto gather_state(std::vector<complex_type> const& local_state)
    -> std::vector<complex_type>
  {
    auto result = std::vector<complex_type>(total_state_size);
    MPI_Allgather(
      static_cast<void const*>(local_state.data()), static_cast<int>(local_state.size() * 2u), MPI_DOUBLE,
      static_cast<void*>(result.data()), static_cast<int>(local_state.size() * 2u), MPI_DOUBLE,
      MPI_COMM_WORLD);
    return result;
  }

  auto max_error(
    std::vector<complex_type> const& gathered_permuted_state,
    std::vector<complex_type> const& reference_state,
    permutation_type const& permutation)
    -> double
  {
    auto result = 0.0;
    for (auto permuted_index = state_integer_type{0u}; permuted_index < gathered_permuted_state.size(); ++permuted_index)
    {
      auto const unpermuted_index = ket::mpi::inverse_permutate_bits(permutation, permuted_index);
      result = std::max(result, std::abs(gathered_permuted_state[permuted_index] - reference_state[unpermuted_index]));
    }
    return result;
  }

  auto make_controls(std::initializer_list<control_qubit_type> const control_qubits)
    -> std::vector<control_qubit_type>
  { return std::vector<control_qubit_type>{control_qubits}; }

  auto local_fused_controls(std::initializer_list<control_qubit_type> const control_qubits)
    -> std::vector<control_qubit_type>
  { return make_controls(control_qubits); }

  template <typename MpiFusedOperation, typename ReferenceOperation>
  auto run_case(
    std::string const& name,
    yampi::communicator const& communicator, yampi::environment const& environment,
    MpiFusedOperation const& mpi_fused_operation, ReferenceOperation const& reference_operation)
    -> bool
  {
    auto const rank = communicator.rank(environment);

    auto reference_state = initial_state();
    auto local_state = local_slice(reference_state, rank);
    auto buffer = std::vector<complex_type>(local_state.size());
    auto permutation = permutation_type{total_qubits};

    reference_operation(reference_state);
    mpi_fused_operation(local_state, permutation, buffer, communicator, environment);

    auto const error = max_error(gather_state(local_state), reference_state, permutation);
    if (error < 1e-12)
      return true;

    if (rank == 0_r)
      std::cerr << name << " failed: max error = " << error << '\n';
    return false;
  }
}

int main(int argc, char** argv)
{
  yampi::environment environment{argc, argv};
  auto communicator = yampi::communicator{yampi::tags::world_communicator};

  auto const rank = communicator.rank(environment);
  auto const size = communicator.size(environment);

  if (size != 2)
  {
    if (rank == 0_r)
      std::cerr << "fused_runtime_gate_numerical requires exactly 2 MPI processes\n";
    return EXIT_FAILURE;
  }

  auto failed = false;
  auto const run = [&failed](bool const passed) { failed = failed or not passed; };

  run(run_case(
    "mpi generic gate + fused hadamard",
    communicator, environment,
    [](auto& local_state, auto& permutation, auto& buffer, auto const& comm, auto const& env)
    {
      auto const qubits = std::vector<qubit_type>{3_q};
      auto const controls = make_controls({0_cq});
      ket::mpi::gate::runtime::ranges::gate(
        ket::mpi::utility::policy::make_simple_mpi(), ket::utility::policy::make_sequential(),
        local_state, permutation, buffer, comm, env,
        [](auto const first, auto const fused_index, auto const& unsorted, auto const& sorted, int const)
        { ket::gate::fused::runtime::ranges::hadamard(first, fused_index, unsorted, sorted, 0_q, local_fused_controls({1_cq})); },
        qubits, controls);
    },
    [](auto& reference_state)
    { ket::gate::runtime::ranges::hadamard(ket::utility::policy::make_sequential(), reference_state, 3_q, make_controls({0_cq})); }));

  run(run_case(
    "mpi generic gate + fused pauli_x",
    communicator, environment,
    [](auto& local_state, auto& permutation, auto& buffer, auto const& comm, auto const& env)
    {
      auto const qubits = std::vector<qubit_type>{0_q, 3_q};
      auto const controls = make_controls({1_cq});
      ket::mpi::gate::runtime::ranges::gate(
        ket::mpi::utility::policy::make_simple_mpi(), ket::utility::policy::make_sequential(),
        local_state, permutation, buffer, comm, env,
        [](auto const first, auto const fused_index, auto const& unsorted, auto const& sorted, int const)
        { ket::gate::fused::runtime::ranges::pauli_x(first, fused_index, unsorted, sorted, std::vector<qubit_type>{0_q, 1_q}, local_fused_controls({2_cq})); },
        qubits, controls);
    },
    [](auto& reference_state)
    { ket::gate::runtime::ranges::pauli_x(ket::utility::policy::make_sequential(), reference_state, std::vector<qubit_type>{0_q, 3_q}, make_controls({1_cq})); }));

  run(run_case(
    "mpi generic gate + fused swap",
    communicator, environment,
    [](auto& local_state, auto& permutation, auto& buffer, auto const& comm, auto const& env)
    {
      auto const qubits = std::vector<qubit_type>{0_q, 3_q};
      auto const controls = make_controls({1_cq});
      ket::mpi::gate::runtime::ranges::gate(
        ket::mpi::utility::policy::make_simple_mpi(), ket::utility::policy::make_sequential(),
        local_state, permutation, buffer, comm, env,
        [](auto const first, auto const fused_index, auto const& unsorted, auto const& sorted, int const)
        { ket::gate::fused::runtime::ranges::swap(first, fused_index, unsorted, sorted, 0_q, 1_q, local_fused_controls({2_cq})); },
        qubits, controls);
    },
    [](auto& reference_state)
    { ket::gate::runtime::ranges::swap(ket::utility::policy::make_sequential(), reference_state, 0_q, 3_q, make_controls({1_cq})); }));

  run(run_case(
    "mpi generic gate + fused exponential_pauli_x",
    communicator, environment,
    [](auto& local_state, auto& permutation, auto& buffer, auto const& comm, auto const& env)
    {
      auto const qubits = std::vector<qubit_type>{0_q, 3_q};
      auto const controls = make_controls({1_cq});
      ket::mpi::gate::runtime::ranges::gate(
        ket::mpi::utility::policy::make_simple_mpi(), ket::utility::policy::make_sequential(),
        local_state, permutation, buffer, comm, env,
        [](auto const first, auto const fused_index, auto const& unsorted, auto const& sorted, int const)
        { ket::gate::fused::runtime::ranges::exponential_pauli_x(first, fused_index, unsorted, sorted, 0.25, std::vector<qubit_type>{0_q, 1_q}, local_fused_controls({2_cq})); },
        qubits, controls);
    },
    [](auto& reference_state)
    { ket::gate::runtime::ranges::exponential_pauli_x(ket::utility::policy::make_sequential(), reference_state, 0.25, std::vector<qubit_type>{0_q, 3_q}, make_controls({1_cq})); }));

  run(run_case(
    "mpi generic gate + fused exponential_pauli_z",
    communicator, environment,
    [](auto& local_state, auto& permutation, auto& buffer, auto const& comm, auto const& env)
    {
      auto const qubits = std::vector<qubit_type>{0_q, 3_q};
      auto const controls = make_controls({1_cq});
      ket::mpi::gate::runtime::ranges::gate(
        ket::mpi::utility::policy::make_simple_mpi(), ket::utility::policy::make_sequential(),
        local_state, permutation, buffer, comm, env,
        [](auto const first, auto const fused_index, auto const& unsorted, auto const& sorted, int const)
        { ket::gate::fused::runtime::ranges::exponential_pauli_z(first, fused_index, unsorted, sorted, 0.25, std::vector<qubit_type>{0_q, 1_q}, local_fused_controls({2_cq})); },
        qubits, controls);
    },
    [](auto& reference_state)
    { ket::gate::runtime::ranges::exponential_pauli_z(ket::utility::policy::make_sequential(), reference_state, 0.25, std::vector<qubit_type>{0_q, 3_q}, make_controls({1_cq})); }));

  run(run_case(
    "mpi generic gate + fused exponential_swap",
    communicator, environment,
    [](auto& local_state, auto& permutation, auto& buffer, auto const& comm, auto const& env)
    {
      auto const qubits = std::vector<qubit_type>{0_q, 3_q};
      auto const controls = make_controls({1_cq});
      ket::mpi::gate::runtime::ranges::gate(
        ket::mpi::utility::policy::make_simple_mpi(), ket::utility::policy::make_sequential(),
        local_state, permutation, buffer, comm, env,
        [](auto const first, auto const fused_index, auto const& unsorted, auto const& sorted, int const)
        { ket::gate::fused::runtime::ranges::exponential_swap(first, fused_index, unsorted, sorted, 0.25, 0_q, 1_q, local_fused_controls({2_cq})); },
        qubits, controls);
    },
    [](auto& reference_state)
    { ket::gate::runtime::ranges::exponential_swap(ket::utility::policy::make_sequential(), reference_state, 0.25, 0_q, 3_q, make_controls({1_cq})); }));

  if (rank == 0_r and not failed)
    std::cout << "runtime MPI fused gate numerical tests passed\n";

  return failed ? EXIT_FAILURE : EXIT_SUCCESS;
}
