#ifndef KET_MPI_UTILITY_DETAIL_SWAP_PERMUTATED_LOCAL_QUBITS_HPP
# define KET_MPI_UTILITY_DETAIL_SWAP_PERMUTATED_LOCAL_QUBITS_HPP

# include <algorithm>
# include <type_traits>

# include <yampi/communicator.hpp>
# include <yampi/environment.hpp>

# include <ket/qubit.hpp>
# include <ket/utility/integer_exp2.hpp>
# include <ket/utility/integer_log2.hpp>
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
        template <typename LoalState_>
        struct swap_permutated_local_qubits
        {
          template <
            typename MpiPolicy, typename ParallelPolicy,
            typename LocalState, typename StateInteger, typename BitInteger>
          static void call(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            LocalState& local_state,
            ket::qubit<StateInteger, BitInteger> const permutated_qubit1,
            ket::qubit<StateInteger, BitInteger> const permutated_qubit2,
            StateInteger const num_data_blocks, StateInteger const data_block_size,
            yampi::communicator const& communicator, yampi::environment const& environment)
          {
            auto const minmax_qubits = std::minmax(permutated_qubit1, permutated_qubit2);
            // || implies the border of local qubits and unit qubits
            // 0000||00000001000
            auto const min_qubit_mask = ::ket::utility::integer_exp2<StateInteger>(minmax_qubits.first);
            // 0000||00010000000
            auto const max_qubit_mask = ::ket::utility::integer_exp2<StateInteger>(minmax_qubits.second);
            // 0000||000|111|
            auto const middle_bits_mask
              = ::ket::utility::integer_exp2<StateInteger>(minmax_qubits.second - minmax_qubits.first) - StateInteger{1u};

            auto const local_state_first = ::ket::utility::begin(local_state);
            auto const num_local_qubits = ::ket::utility::integer_log2<BitInteger>(data_block_size);
            for (auto data_block_index = StateInteger{0u}; data_block_index < num_data_blocks; ++data_block_index)
            {
              // ****||00000000000
              auto const data_block_mask = data_block_index << num_local_qubits;

              using ket::utility::loop_n;
              loop_n(
                parallel_policy,
                (data_block_size >> minmax_qubits.first) >> 2u,
                [local_state_first, data_block_mask, &minmax_qubits,
                 min_qubit_mask, max_qubit_mask, middle_bits_mask](
                  // xxx|xxx|
                  StateInteger const value_wo_qubits, int const)
                {
                  // ****||xxx0xxx0000
                  auto const base_index
                    = ((value_wo_qubits bitand middle_bits_mask) << (minmax_qubits.first + BitInteger{1u}))
                      bitor ((value_wo_qubits bitand compl middle_bits_mask) << (minmax_qubits.first + BitInteger{2u}))
                      bitor data_block_mask;
                  // ****||xxx1xxx0000
                  auto const index1 = base_index bitor max_qubit_mask;
                  // ****||xxx0xxx1000
                  auto const index2 = base_index bitor min_qubit_mask;

                  std::swap_ranges(
                    local_state_first + index1,
                    local_state_first + (index1 bitor min_qubit_mask),
                    local_state_first + index2);
                });
            }
          }
        }; // struct swap_permutated_local_qubits<LocalState_>
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
          StateInteger const num_data_blocks, StateInteger const data_block_size,
          yampi::communicator const& communicator, yampi::environment const& environment)
        {
          using swap_permutated_local_qubits_
            = ::ket::mpi::utility::dispatch::swap_permutated_local_qubits<typename std::remove_cv<LocalState>::type>;
          swap_permutated_local_qubits_::call(
            mpi_policy, parallel_policy, local_state, permutated_qubit1, permutated_qubit2,
            num_data_blocks, data_block_size, communicator, environment);
        }
      } // namespace detail
    } // namespace utility
  } // namespace mpi
} // namespace ket


#endif // KET_MPI_UTILITY_DETAIL_SWAP_PERMUTATED_LOCAL_QUBITS_HPP
