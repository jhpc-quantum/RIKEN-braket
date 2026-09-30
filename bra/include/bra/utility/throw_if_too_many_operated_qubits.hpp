#ifndef BRA_UTILITY_THROW_IF_TOO_MANY_OPERATED_QUBITS_HPP
# define BRA_UTILITY_THROW_IF_TOO_MANY_OPERATED_QUBITS_HPP

# ifndef BRA_NO_MPI
#   include <cstddef>

#   include <bra/state.hpp>

namespace bra
{
  template <typename MpiPolicy, typename LocalState, typename Communicator, typename Environment>
  auto throw_if_too_many_operated_qubits(
    std::size_t const num_operated_qubits,
    MpiPolicy const& mpi_policy, LocalState const& local_state,
    Communicator const& communicator, Environment const& environment) -> void
  {
    auto const num_local_qubits
      = ket::mpi::utility::policy::num_local_qubits(mpi_policy, local_state, communicator, environment);
    if (num_operated_qubits > num_local_qubits)
      throw ::bra::too_many_operated_qubits_error{num_operated_qubits, num_local_qubits};
  }
} // namespace bra

# endif // BRA_NO_MPI

#endif // BRA_UTILITY_THROW_IF_TOO_MANY_OPERATED_QUBITS_HPP
