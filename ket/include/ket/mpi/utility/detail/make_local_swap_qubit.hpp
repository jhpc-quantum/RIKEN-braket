#ifndef KET_MPI_UTILITY_DETAIL_MAKE_LOCAL_SWAP_QUBIT_HPP
# define KET_MPI_UTILITY_DETAIL_MAKE_LOCAL_SWAP_QUBIT_HPP

# include <algorithm>
# include <iterator>
# include <type_traits>

# include <yampi/communicator.hpp>
# include <yampi/environment.hpp>

# include <ket/qubit.hpp>
# include <ket/utility/contains.hpp>
# include <ket/utility/meta/ranges.hpp>
# include <ket/mpi/permutated.hpp>
# include <ket/mpi/qubit_permutation.hpp>
# include <ket/mpi/utility/detail/swap_permutated_local_qubits.hpp>


namespace ket
{
  namespace mpi
  {
    namespace utility
    {
      namespace detail
      {
        template <
          typename ParallelPolicy, typename LocalState,
          typename StateInteger, typename BitInteger, typename Allocator,
          typename UnswappableQubits>
        inline auto make_local_swap_qubit(
          ParallelPolicy const parallel_policy,
          LocalState& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          StateInteger const num_data_blocks, StateInteger const data_block_size,
          yampi::communicator const& communicator, yampi::environment const& environment,
          UnswappableQubits const& unswappable_qubits,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_local_swap_qubit)
        -> ::ket::qubit<StateInteger, BitInteger>
        {
          using qubit_type = ket::qubit<StateInteger, BitInteger>;
          static_assert(
            std::is_same< ::ket::utility::meta::range_value_t<UnswappableQubits>, qubit_type >::value,
            "value_type of UnswappableQubits must be the same to qubit_type");

          auto const local_swap_qubit = ::ket::mpi::inverse(permutation)[permutated_local_swap_qubit];

          using std::begin;
          using std::end;
          if (not ::ket::utility::contains(begin(unswappable_qubits), end(unswappable_qubits), local_swap_qubit))
            return local_swap_qubit;

          auto permutated_other_qubit = permutated_local_swap_qubit;
          auto other_qubit = qubit_type{};
          do
          {
            --permutated_other_qubit;
            other_qubit = ::ket::mpi::inverse(permutation)[permutated_other_qubit];
          }
          while (::ket::utility::contains(begin(unswappable_qubits), end(unswappable_qubits), other_qubit));

          ::ket::mpi::utility::detail::swap_permutated_local_qubits(
            parallel_policy, local_state,
            permutated_local_swap_qubit, permutated_other_qubit,
            num_data_blocks, data_block_size, communicator, environment);
          ::ket::mpi::permutate(permutation, local_swap_qubit, other_qubit);

          return ::ket::mpi::inverse(permutation)[permutated_local_swap_qubit];
        }

        template <
          typename ParallelPolicy, typename LocalState,
          typename StateInteger, typename BitInteger, typename Allocator,
          typename UnswappableQubits>
        [[deprecated]] inline auto make_local_swap_qubit(
          ParallelPolicy const parallel_policy,
          LocalState& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          UnswappableQubits const& unswappable_qubits,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_local_swap_qubit,
          StateInteger const num_data_blocks, StateInteger const data_block_size,
          yampi::communicator const& communicator, yampi::environment const& environment)
        -> ::ket::qubit<StateInteger, BitInteger>
        {
          return ::ket::mpi::utility::detail::make_local_swap_qubit(
            parallel_policy, local_state, permutation,
            num_data_blocks, data_block_size, communicator, environment,
            unswappable_qubits, permutated_local_swap_qubit);
        }
      } // namespace detail
    } // namespace utility
  } // namespace mpi
} // namespace ket


#endif // KET_MPI_UTILITY_DETAIL_MAKE_LOCAL_SWAP_QUBIT_HPP
