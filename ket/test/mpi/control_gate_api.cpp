#include <algorithm>
#include <cstdint>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <type_traits>
#include <vector>
#include <ket/runtime_control.hpp>
#include <ket/mpi/permutated.hpp>
#include <ket/mpi/qubit_permutation.hpp>
#include <ket/mpi/page/transpage_iterator.hpp>
#include <ket/mpi/gate/hadamard.hpp>
#include <ket/mpi/gate/phase_shift.hpp>
#include <ket/mpi/gate/controlled_v.hpp>
#include <ket/mpi/gate/identity.hpp>
#include <ket/mpi/gate/toffoli.hpp>

// Instantiate the convenience interfaces too: they must preserve values when
// forwarding to the overloads that take explicit MPI and parallel policies.
void convenience_api_smoke(
  std::vector<std::complex<double>>& state,
  ket::mpi::qubit_permutation<std::uint32_t, unsigned>& permutation,
  std::vector<std::complex<double>>& buffer,
  yampi::communicator const& communicator, yampi::environment const& environment)
{
  using Qubit = ket::qubit<std::uint32_t, unsigned>;
  auto const negative = ket::control<Qubit, 0u>{1u};
  auto const positive = ket::control<Qubit>{2u};
  auto const dynamic = ket::make_runtime_control(Qubit{2u}, 1u);
  ket::mpi::gate::hadamard(state, permutation, buffer, communicator, environment, Qubit{0u}, negative, dynamic);
  ket::mpi::gate::phase_shift(state, permutation, buffer, communicator, environment, 0.3, negative, positive);
  ket::mpi::gate::controlled_v(state, permutation, buffer, communicator, environment, 0.3, Qubit{0u}, negative, positive);
  ket::mpi::gate::identity(state, permutation, buffer, communicator, environment, Qubit{0u}, negative, dynamic);
  ket::mpi::gate::identity(state, permutation, buffer, communicator, environment, negative, dynamic);
  ket::mpi::gate::toffoli(state, permutation, buffer, communicator, environment, Qubit{0u}, negative, positive);
}

int main()
{
  using Qubit = ket::qubit<std::uint32_t, unsigned>;
  using Negative = ket::control<Qubit, 0u>;
  using Positive = ket::control<Qubit, 1u>;
  using Permutated = ket::mpi::permutated<Qubit>;
  constexpr auto negative = ket::mpi::make_permutated(Negative{2u});
  constexpr auto dynamic = ket::mpi::make_permutated(ket::make_runtime_control(Qubit{2u}, 0u));
  static_assert(negative == ket::mpi::make_permutated(Positive{2u}), "Positional equality");
  static_assert(ket::mpi::control_value(dynamic + 1u) == 0u, "Position shifts preserve values");
  static_assert(ket::mpi::remove_control(negative).qubit() == Qubit{2u}, "Remove condition explicitly");

  auto permutation = ket::mpi::qubit_permutation<std::uint32_t, unsigned>{8u};
  permutation.permutate(Negative{2u}, Positive{6u});
  if (ket::mpi::remove_control(permutation[Negative{2u}]).qubit() != Qubit{6u}
      or ket::mpi::control_value(permutation.at(ket::make_runtime_control(Qubit{2u}, 0u))) != 0u
      or ket::control_value(ket::mpi::inverse(permutation)[permutation[Negative{2u}]]) != 0u)
    return EXIT_FAILURE;
  try { (void)permutation.at(Negative{8u}); return EXIT_FAILURE; }
  catch (std::out_of_range const&) { }

  // Exhaust all occupied nonpage subsets, including an empty range, and all
  // page counts that can be mapped to distinct unoccupied nonpage wires.
  for (auto occupied = 0u; occupied < 16u; ++occupied)
  {
    std::vector<Permutated> nonpage;
    for (auto bit = 0u; bit < 4u; ++bit)
      if ((occupied & (1u << bit)) != 0u) nonpage.emplace_back(Qubit{bit});
    for (auto pages = 0u; pages <= 4u - nonpage.size(); ++pages)
    {
      auto const mapped = ket::mpi::page::generate_mapped_permutated_nonpage_qubits(
        nonpage.begin(), nonpage.end(), Permutated{Qubit{4u}}, pages);
      if (mapped.size() != pages or not std::is_sorted(mapped.begin(), mapped.end()))
        return EXIT_FAILURE;
      for (auto i = std::size_t{0}; i < mapped.size(); ++i)
        if (mapped[i] >= Permutated{Qubit{4u}}
            or std::binary_search(nonpage.begin(), nonpage.end(), mapped[i])
            or (i != 0u and mapped[i] == mapped[i - 1u]))
          return EXIT_FAILURE;
    }
  }
  std::cout << "MPI control permutation and page-mapping API tests passed\n";
}
