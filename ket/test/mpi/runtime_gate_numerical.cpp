#include <mpi.h>

// Example:
//   mpicxx -std=c++14 -DNDEBUG -Iket/include -I../yampi/include \
//     ket/test/mpi/runtime_gate_numerical.cpp -o /tmp/runtime_gate_numerical
//   mpiexec -n 2 /tmp/runtime_gate_numerical

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <ket/control.hpp>
#include <ket/gate/controlled_v.hpp>
#include <ket/gate/exponential_pauli_x.hpp>
#include <ket/gate/exponential_pauli_y.hpp>
#include <ket/gate/exponential_pauli_z.hpp>
#include <ket/gate/exponential_swap.hpp>
#include <ket/gate/hadamard.hpp>
#include <ket/gate/pauli_x.hpp>
#include <ket/gate/pauli_y.hpp>
#include <ket/gate/pauli_z.hpp>
#include <ket/gate/phase_shift.hpp>
#include <ket/gate/sqrt_pauli_x.hpp>
#include <ket/gate/sqrt_pauli_y.hpp>
#include <ket/gate/sqrt_pauli_z.hpp>
#include <ket/gate/swap.hpp>
#include <ket/gate/x_rotation_half_pi.hpp>
#include <ket/gate/y_rotation_half_pi.hpp>
#include <ket/mpi/gate/controlled_v.hpp>
#include <ket/mpi/gate/exponential_pauli_x.hpp>
#include <ket/mpi/gate/exponential_pauli_y.hpp>
#include <ket/mpi/gate/exponential_pauli_z.hpp>
#include <ket/mpi/gate/exponential_swap.hpp>
#include <ket/mpi/gate/hadamard.hpp>
#include <ket/mpi/gate/pauli_x.hpp>
#include <ket/mpi/gate/pauli_y.hpp>
#include <ket/mpi/gate/pauli_z.hpp>
#include <ket/mpi/gate/phase_shift.hpp>
#include <ket/mpi/gate/sqrt_pauli_x.hpp>
#include <ket/mpi/gate/sqrt_pauli_y.hpp>
#include <ket/mpi/gate/sqrt_pauli_z.hpp>
#include <ket/mpi/gate/swap.hpp>
#include <ket/mpi/gate/x_rotation_half_pi.hpp>
#include <ket/mpi/gate/y_rotation_half_pi.hpp>
#include <ket/mpi/qubit_permutation.hpp>
#include <ket/mpi/state.hpp>
#include <ket/mpi/utility/simple_mpi.hpp>
#include <ket/qubit.hpp>
#include <ket/utility/exp_i.hpp>
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
      result = std::max(
        result,
        std::abs(gathered_permuted_state[permuted_index] - reference_state[unpermuted_index]));
    }

    return result;
  }

  auto report_failure(
    std::string const& name, double const error, yampi::rank const rank)
    -> void
  {
    if (rank == 0_r)
      std::cerr << name << " failed: max error = " << error << '\n';
  }

  template <typename MpiOperation, typename ReferenceOperation>
  auto run_vector_case(
    std::string const& name,
    yampi::communicator const& communicator, yampi::environment const& environment,
    MpiOperation const& mpi_operation, ReferenceOperation const& reference_operation)
    -> bool
  {
    auto const rank = communicator.rank(environment);

    auto reference_state = initial_state();
    auto local_state = local_slice(reference_state, rank);
    auto buffer = std::vector<complex_type>(local_state.size());
    auto permutation = permutation_type{total_qubits};

    reference_operation(reference_state);
    mpi_operation(local_state, permutation, buffer, communicator, environment);

    auto const error = max_error(gather_state(local_state), reference_state, permutation);
    auto const passed = error < 1e-12;
    if (not passed)
      report_failure(name, error, rank);

    return passed;
  }

  template <typename MpiOperation, typename ReferenceOperation>
  auto run_page_case(
    std::string const& name,
    yampi::communicator const& communicator, yampi::environment const& environment,
    MpiOperation const& mpi_operation, ReferenceOperation const& reference_operation)
    -> bool
  {
    auto const rank = communicator.rank(environment);

    auto reference_state = initial_state();
    auto const local_vector = local_slice(reference_state, rank);
    auto local_state = ket::mpi::state<complex_type, true>{
      {
        local_vector[0], local_vector[1], local_vector[2], local_vector[3],
        local_vector[4], local_vector[5], local_vector[6], local_vector[7]},
      bit_integer_type{1u}};
    auto buffer = std::vector<complex_type>(local_vector.size());
    auto permutation = permutation_type{total_qubits};

    reference_operation(reference_state);
    mpi_operation(local_state, permutation, buffer, communicator, environment);

    auto gathered_local_state = std::vector<complex_type>{local_state.begin(), local_state.end()};
    auto const error = max_error(gather_state(gathered_local_state), reference_state, permutation);
    auto const passed = error < 1e-12;
    if (not passed)
      report_failure(name, error, rank);

    return passed;
  }

  auto make_controls(std::initializer_list<control_qubit_type> const control_qubits)
    -> std::vector<control_qubit_type>
  { return std::vector<control_qubit_type>{control_qubits}; }

  template <typename LocalState>
  auto runtime_hadamard(
    LocalState& local_state, permutation_type& permutation, std::vector<complex_type>& buffer,
    yampi::communicator const& communicator, yampi::environment const& environment,
    qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
    -> void
  {
    ket::mpi::gate::runtime::ranges::hadamard(
      ket::mpi::utility::policy::make_simple_mpi(), ket::utility::policy::make_sequential(),
      local_state, permutation, buffer, communicator, environment, target_qubit, control_qubits);
  }

  template <typename LocalState>
  auto runtime_pauli_x(
    LocalState& local_state, permutation_type& permutation, std::vector<complex_type>& buffer,
    yampi::communicator const& communicator, yampi::environment const& environment,
    std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
    -> void
  {
    ket::mpi::gate::runtime::ranges::pauli_x(
      ket::mpi::utility::policy::make_simple_mpi(), ket::utility::policy::make_sequential(),
      local_state, permutation, buffer, communicator, environment, target_qubits, control_qubits);
  }

  template <typename LocalState>
  auto runtime_pauli_y(
    LocalState& local_state, permutation_type& permutation, std::vector<complex_type>& buffer,
    yampi::communicator const& communicator, yampi::environment const& environment,
    std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
    -> void
  {
    ket::mpi::gate::runtime::ranges::pauli_y(
      ket::mpi::utility::policy::make_simple_mpi(), ket::utility::policy::make_sequential(),
      local_state, permutation, buffer, communicator, environment, target_qubits, control_qubits);
  }

  template <typename LocalState>
  auto runtime_swap(
    LocalState& local_state, permutation_type& permutation, std::vector<complex_type>& buffer,
    yampi::communicator const& communicator, yampi::environment const& environment,
    qubit_type const target_qubit1, qubit_type const target_qubit2,
    std::vector<control_qubit_type> const& control_qubits)
    -> void
  {
    ket::mpi::gate::runtime::ranges::swap(
      ket::mpi::utility::policy::make_simple_mpi(), ket::utility::policy::make_sequential(),
      local_state, permutation, buffer, communicator, environment,
      target_qubit1, target_qubit2, control_qubits);
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
      std::cerr << "runtime_gate_numerical requires exactly 2 MPI processes\n";
    return EXIT_FAILURE;
  }

  auto failed = false;
  auto const run = [&failed](bool const passed) { failed = failed or not passed; };

  run(run_vector_case(
    "runtime::ranges::hadamard nonlocal target",
    communicator, environment,
    [](auto& local_state, auto& permutation, auto& buffer, auto const& comm, auto const& env)
    { runtime_hadamard(local_state, permutation, buffer, comm, env, 3_q, make_controls({0_cq})); },
    [](auto& reference_state)
    {
      auto const controls = make_controls({0_cq});
      ket::gate::runtime::ranges::hadamard(
        ket::utility::policy::make_sequential(), reference_state, 3_q, controls);
    }));

  run(run_page_case(
    "runtime::ranges::hadamard page target",
    communicator, environment,
    [](auto& local_state, auto& permutation, auto& buffer, auto const& comm, auto const& env)
    { runtime_hadamard(local_state, permutation, buffer, comm, env, 2_q, make_controls({0_cq})); },
    [](auto& reference_state)
    {
      auto const controls = make_controls({0_cq});
      ket::gate::runtime::ranges::hadamard(
        ket::utility::policy::make_sequential(), reference_state, 2_q, controls);
    }));

  run(run_vector_case(
    "runtime::ranges::pauli_x multi-target",
    communicator, environment,
    [](auto& local_state, auto& permutation, auto& buffer, auto const& comm, auto const& env)
    {
      auto const target_qubits = std::vector<qubit_type>{0_q, 3_q};
      runtime_pauli_x(local_state, permutation, buffer, comm, env, target_qubits, make_controls({1_cq}));
    },
    [](auto& reference_state)
    {
      auto const target_qubits = std::vector<qubit_type>{0_q, 3_q};
      auto const controls = make_controls({1_cq});
      ket::gate::runtime::ranges::pauli_x(
        ket::utility::policy::make_sequential(), reference_state, target_qubits, controls);
    }));

  run(run_vector_case(
    "runtime::ranges::pauli_y multi-target",
    communicator, environment,
    [](auto& local_state, auto& permutation, auto& buffer, auto const& comm, auto const& env)
    {
      auto const target_qubits = std::vector<qubit_type>{0_q, 3_q};
      runtime_pauli_y(local_state, permutation, buffer, comm, env, target_qubits, make_controls({1_cq}));
    },
    [](auto& reference_state)
    {
      auto const target_qubits = std::vector<qubit_type>{0_q, 3_q};
      auto const controls = make_controls({1_cq});
      ket::gate::runtime::ranges::pauli_y(
        ket::utility::policy::make_sequential(), reference_state, target_qubits, controls);
    }));

  run(run_vector_case(
    "runtime::ranges::pauli_z target-control diagonal",
    communicator, environment,
    [](auto& local_state, auto& permutation, auto& buffer, auto const& comm, auto const& env)
    {
      auto const target_qubits = std::vector<qubit_type>{3_q};
      auto const controls = make_controls({0_cq});
      ket::mpi::gate::runtime::ranges::pauli_z(
        ket::mpi::utility::policy::make_simple_mpi(), ket::utility::policy::make_sequential(),
        local_state, permutation, buffer, comm, env, target_qubits, controls);
    },
    [](auto& reference_state)
    {
      auto const target_qubits = std::vector<qubit_type>{3_q};
      auto const controls = make_controls({0_cq});
      ket::gate::runtime::ranges::pauli_z(
        ket::utility::policy::make_sequential(), reference_state, target_qubits, controls);
    }));

  run(run_vector_case(
    "runtime::ranges::phase_shift_coeff diagonal",
    communicator, environment,
    [](auto& local_state, auto& permutation, auto& buffer, auto const& comm, auto const& env)
    {
      auto const controls = make_controls({0_cq});
      ket::mpi::gate::runtime::ranges::phase_shift_coeff(
        ket::mpi::utility::policy::make_simple_mpi(), ket::utility::policy::make_sequential(),
        local_state, permutation, buffer, comm, env, ket::utility::exp_i<complex_type>(0.375), 3_q, controls);
    },
    [](auto& reference_state)
    {
      auto const controls = make_controls({0_cq});
      ket::gate::runtime::ranges::phase_shift_coeff(
        ket::utility::policy::make_sequential(), reference_state,
        ket::utility::exp_i<complex_type>(0.375), 3_q, controls);
    }));

  run(run_vector_case(
    "runtime::ranges::controlled_v",
    communicator, environment,
    [](auto& local_state, auto& permutation, auto& buffer, auto const& comm, auto const& env)
    {
      auto const controls = make_controls({0_cq, 1_cq});
      ket::mpi::gate::runtime::ranges::controlled_v(
        ket::mpi::utility::policy::make_simple_mpi(), ket::utility::policy::make_sequential(),
        local_state, permutation, buffer, comm, env, 0.25, 3_q, controls);
    },
    [](auto& reference_state)
    {
      auto const controls = make_controls({0_cq, 1_cq});
      ket::gate::runtime::ranges::controlled_v(
        ket::utility::policy::make_sequential(), reference_state, 0.25, 3_q, controls);
    }));

  run(run_vector_case(
    "runtime::ranges::sqrt_pauli_z diagonal",
    communicator, environment,
    [](auto& local_state, auto& permutation, auto& buffer, auto const& comm, auto const& env)
    {
      auto const target_qubits = std::vector<qubit_type>{3_q};
      auto const controls = make_controls({0_cq});
      ket::mpi::gate::runtime::ranges::sqrt_pauli_z(
        ket::mpi::utility::policy::make_simple_mpi(), ket::utility::policy::make_sequential(),
        local_state, permutation, buffer, comm, env, target_qubits, controls);
    },
    [](auto& reference_state)
    {
      auto const target_qubits = std::vector<qubit_type>{3_q};
      auto const controls = make_controls({0_cq});
      ket::gate::runtime::ranges::sqrt_pauli_z(
        ket::utility::policy::make_sequential(), reference_state, target_qubits, controls);
    }));

  run(run_vector_case(
    "runtime::ranges::swap",
    communicator, environment,
    [](auto& local_state, auto& permutation, auto& buffer, auto const& comm, auto const& env)
    { runtime_swap(local_state, permutation, buffer, comm, env, 0_q, 3_q, make_controls({1_cq})); },
    [](auto& reference_state)
    {
      auto const controls = make_controls({1_cq});
      ket::gate::runtime::ranges::swap(
        ket::utility::policy::make_sequential(), reference_state, 0_q, 3_q, controls);
    }));

  run(run_vector_case(
    "runtime::ranges::x_rotation_half_pi",
    communicator, environment,
    [](auto& local_state, auto& permutation, auto& buffer, auto const& comm, auto const& env)
    {
      auto const controls = make_controls({0_cq});
      ket::mpi::gate::runtime::ranges::x_rotation_half_pi(
        ket::mpi::utility::policy::make_simple_mpi(), ket::utility::policy::make_sequential(),
        local_state, permutation, buffer, comm, env, 3_q, controls);
    },
    [](auto& reference_state)
    {
      auto const controls = make_controls({0_cq});
      ket::gate::runtime::ranges::x_rotation_half_pi(
        ket::utility::policy::make_sequential(), reference_state, 3_q, controls);
    }));

  run(run_vector_case(
    "runtime::ranges::y_rotation_half_pi",
    communicator, environment,
    [](auto& local_state, auto& permutation, auto& buffer, auto const& comm, auto const& env)
    {
      auto const controls = make_controls({0_cq});
      ket::mpi::gate::runtime::ranges::y_rotation_half_pi(
        ket::mpi::utility::policy::make_simple_mpi(), ket::utility::policy::make_sequential(),
        local_state, permutation, buffer, comm, env, 3_q, controls);
    },
    [](auto& reference_state)
    {
      auto const controls = make_controls({0_cq});
      ket::gate::runtime::ranges::y_rotation_half_pi(
        ket::utility::policy::make_sequential(), reference_state, 3_q, controls);
    }));

  run(run_vector_case(
    "runtime::ranges::exponential_pauli_x",
    communicator, environment,
    [](auto& local_state, auto& permutation, auto& buffer, auto const& comm, auto const& env)
    {
      auto const target_qubits = std::vector<qubit_type>{0_q, 3_q};
      auto const controls = make_controls({1_cq});
      ket::mpi::gate::runtime::ranges::exponential_pauli_x(
        ket::mpi::utility::policy::make_simple_mpi(), ket::utility::policy::make_sequential(),
        local_state, permutation, buffer, comm, env, 0.25, target_qubits, controls);
    },
    [](auto& reference_state)
    {
      auto const target_qubits = std::vector<qubit_type>{0_q, 3_q};
      auto const controls = make_controls({1_cq});
      ket::gate::runtime::ranges::exponential_pauli_x(
        ket::utility::policy::make_sequential(), reference_state, 0.25, target_qubits, controls);
    }));

  run(run_vector_case(
    "runtime::ranges::exponential_pauli_y",
    communicator, environment,
    [](auto& local_state, auto& permutation, auto& buffer, auto const& comm, auto const& env)
    {
      auto const target_qubits = std::vector<qubit_type>{0_q, 3_q};
      auto const controls = make_controls({1_cq});
      ket::mpi::gate::runtime::ranges::exponential_pauli_y(
        ket::mpi::utility::policy::make_simple_mpi(), ket::utility::policy::make_sequential(),
        local_state, permutation, buffer, comm, env, 0.25, target_qubits, controls);
    },
    [](auto& reference_state)
    {
      auto const target_qubits = std::vector<qubit_type>{0_q, 3_q};
      auto const controls = make_controls({1_cq});
      ket::gate::runtime::ranges::exponential_pauli_y(
        ket::utility::policy::make_sequential(), reference_state, 0.25, target_qubits, controls);
    }));

  run(run_vector_case(
    "runtime::ranges::exponential_pauli_z diagonal",
    communicator, environment,
    [](auto& local_state, auto& permutation, auto& buffer, auto const& comm, auto const& env)
    {
      auto const target_qubits = std::vector<qubit_type>{0_q, 3_q};
      auto const controls = make_controls({1_cq});
      ket::mpi::gate::runtime::ranges::exponential_pauli_z(
        ket::mpi::utility::policy::make_simple_mpi(), ket::utility::policy::make_sequential(),
        local_state, permutation, buffer, comm, env, 0.25, target_qubits, controls);
    },
    [](auto& reference_state)
    {
      auto const target_qubits = std::vector<qubit_type>{0_q, 3_q};
      auto const controls = make_controls({1_cq});
      ket::gate::runtime::ranges::exponential_pauli_z(
        ket::utility::policy::make_sequential(), reference_state, 0.25, target_qubits, controls);
    }));

  run(run_vector_case(
    "runtime::ranges::exponential_swap",
    communicator, environment,
    [](auto& local_state, auto& permutation, auto& buffer, auto const& comm, auto const& env)
    {
      auto const controls = make_controls({1_cq});
      ket::mpi::gate::runtime::ranges::exponential_swap(
        ket::mpi::utility::policy::make_simple_mpi(), ket::utility::policy::make_sequential(),
        local_state, permutation, buffer, comm, env, 0.25, 0_q, 3_q, controls);
    },
    [](auto& reference_state)
    {
      auto const controls = make_controls({1_cq});
      ket::gate::runtime::ranges::exponential_swap(
        ket::utility::policy::make_sequential(), reference_state, 0.25, 0_q, 3_q, controls);
    }));

  if (rank == 0_r and not failed)
    std::cout << "runtime MPI gate numerical tests passed\n";

  return failed ? EXIT_FAILURE : EXIT_SUCCESS;
}
