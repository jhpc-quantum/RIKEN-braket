#ifndef KET_MPI_GATE_DETAIL_ASSERT_ALL_QUBITS_ARE_LOCAL_HPP
# define KET_MPI_GATE_DETAIL_ASSERT_ALL_QUBITS_ARE_LOCAL_HPP

# include <cassert>

# include <ket/qubit.hpp>
# ifndef NDEBUG
#   include <ket/meta/state_integer_of.hpp>
#   include <ket/meta/bit_integer_of.hpp>
# endif // NDEBUG
# include <ket/mpi/permutated.hpp>
# ifndef NDEBUG
#   include <ket/mpi/utility/simple_mpi.hpp>
# endif // NDEBUG

# include <yampi/communicator.hpp>
# include <yampi/environment.hpp>

namespace ket
{
  namespace mpi
  {
    namespace gate
    {
      namespace detail
      {
        template <typename BitInteger>
        inline auto assert_all_qubits_are_local(BitInteger const num_local_qubits) -> void
        { }

        template <typename BitInteger, typename PermutatedQubit, typename... PermutatedQubits>
        inline auto assert_all_qubits_are_local(
          BitInteger const num_local_qubits,
          PermutatedQubit const permutated_qubit, PermutatedQubits const... permutated_qubits)
        -> void
        {
          static_assert(std::is_same<BitInteger, ::ket::meta::bit_integer_t<PermutatedQubit>>::value, "BitInteger should be the same as bit_integer_type of PermutatedQubit");
# ifndef NDEBUG
          using state_integer_type = ::ket::meta::state_integer_t<PermutatedQubit>;
          auto const least_nonlocal_permutated_qubit = ::ket::mpi::make_permutated(::ket::make_qubit<state_integer_type>(num_local_qubits));
# endif // NDEBUG
          assert(::ket::mpi::remove_control(permutated_qubit) < least_nonlocal_permutated_qubit);
          ::ket::mpi::gate::detail::assert_all_qubits_are_local(num_local_qubits, permutated_qubits...);
        }

        template <typename MpiPolicy, typename RandomAccessRange>
        inline auto assert_all_qubits_are_local(
          MpiPolicy const& mpi_policy, RandomAccessRange& local_state,
          yampi::communicator const& communicator, yampi::environment const& environment)
        -> void
        { }

        template <typename MpiPolicy, typename RandomAccessRange, typename PermutatedQubit, typename... PermutatedQubits>
        inline auto assert_all_qubits_are_local(
          MpiPolicy const& mpi_policy, RandomAccessRange& local_state,
          yampi::communicator const& communicator, yampi::environment const& environment,
          PermutatedQubit const permutated_qubit, PermutatedQubits const... permutated_qubits)
        -> void
        {
          using bit_integer_type = ::ket::meta::bit_integer_t<PermutatedQubit>;
          ::ket::mpi::gate::detail::assert_all_qubits_are_local(
            static_cast<bit_integer_type>(::ket::mpi::utility::policy::num_local_qubits(mpi_policy, local_state, communicator, environment)),
            permutated_qubits...);
        }
      } // namespace detail
    } // namespace gate
  } // namespace mpi
} // namespace ket

#endif // KET_MPI_GATE_DETAIL_ASSERT_ALL_QUBITS_ARE_LOCAL_HPP
