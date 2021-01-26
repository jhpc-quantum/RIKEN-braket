#ifndef KET_MPI_UTILITY_DETAIL_SWAP_PERMUTATED_LOCAL_QUBITS_HPP
# define KET_MPI_UTILITY_DETAIL_SWAP_PERMUTATED_LOCAL_QUBITS_HPP

# include <algorithm>
# include <type_traits>

# include <yampi/communicator.hpp>
# include <yampi/environment.hpp>

# include <ket/qubit.hpp>
# include <ket/utility/integer_exp2.hpp>
# include <ket/utility/loop_n.hpp>
# include <ket/utility/begin.hpp>


namespace ket
{
  namespace mpi
  {
    namespace utility
    {
      namespace dispatch
      {
        template <typename MpiPolicy, typename LoalState_>
        struct swap_permutated_local_qubits;
      } // namespace dispatch

      namespace detail
      {
        template <
          typename MpiPolicy, typename ParallelPolicy, typename LocalState,
          typename StateInteger, typename BitInteger>
        inline void swap_permutated_local_qubits(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          LocalState& local_state,
          ket::qubit<StateInteger, BitInteger> const permutated_qubit1,
          ket::qubit<StateInteger, BitInteger> const permutated_qubit2,
          yampi::communicator const& communicator, yampi::environment const& environment)
        {
          using swap_permutated_local_qubits_
            = ::ket::mpi::utility::dispatch::swap_permutated_local_qubits<MpiPolicy, typename std::remove_cv<LocalState>::type>;
          swap_permutated_local_qubits_::call(
            mpi_policy, parallel_policy, local_state, permutated_qubit1, permutated_qubit2,
            communicator, environment);
        }
      } // namespace detail
    } // namespace utility
  } // namespace mpi
} // namespace ket


#endif // KET_MPI_UTILITY_DETAIL_SWAP_PERMUTATED_LOCAL_QUBITS_HPP
