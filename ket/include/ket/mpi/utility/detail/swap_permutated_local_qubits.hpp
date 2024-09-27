#ifndef KET_MPI_UTILITY_DETAIL_SWAP_PERMUTATED_LOCAL_QUBITS_HPP
# define KET_MPI_UTILITY_DETAIL_SWAP_PERMUTATED_LOCAL_QUBITS_HPP

# include <algorithm>
# include <iterator>
# include <type_traits>

# include <yampi/communicator.hpp>
# include <yampi/environment.hpp>

# include <ket/qubit.hpp>
# include <ket/utility/integer_exp2.hpp>
# include <ket/utility/integer_log2.hpp>
# include <ket/utility/loop_n.hpp>
# include <ket/mpi/permutated.hpp>


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
          template <typename ParallelPolicy, typename LocalState, typename StateInteger, typename BitInteger>
          static void call(
            ParallelPolicy const parallel_policy,
            LocalState& local_state,
            ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_qubit1,
            ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_qubit2,
            StateInteger const num_data_blocks, StateInteger const data_block_size,
            yampi::communicator const& communicator, yampi::environment const& environment)
          {
            auto const minmax_permutated_qubits = std::minmax(permutated_qubit1, permutated_qubit2);
            // || implies the border of local qubits and unit qubits
            // 0000||00000001000
            auto const min_permutated_qubit_mask = ::ket::utility::integer_exp2<StateInteger>(minmax_permutated_qubits.first);
            // 0000||00010000000
            auto const max_permutated_qubit_mask = ::ket::utility::integer_exp2<StateInteger>(minmax_permutated_qubits.second);
            // 0000||000|111|
            auto const middle_bits_mask
              = ::ket::utility::integer_exp2<StateInteger>(minmax_permutated_qubits.second - minmax_permutated_qubits.first - BitInteger{1u}) - StateInteger{1u};

            using std::begin;
            auto const local_state_first = begin(local_state);
            auto const num_local_qubits = ::ket::utility::integer_log2<BitInteger>(data_block_size);
            for (auto data_block_index = StateInteger{0u}; data_block_index < num_data_blocks; ++data_block_index)
            {
              // ****||00000000000
              auto const data_block_mask = data_block_index << num_local_qubits;

              ::ket::utility::loop_n(
                parallel_policy,
                (data_block_size >> minmax_permutated_qubits.first) >> 2u,
                [local_state_first, data_block_mask, &minmax_permutated_qubits,
                 min_permutated_qubit_mask, max_permutated_qubit_mask, middle_bits_mask](
                  // xxx|xxx|
                  StateInteger const value_wo_qubits, int const)
                {
                  // ****||xxx0xxx0000
                  auto const base_index
                    = ((value_wo_qubits bitand middle_bits_mask) << (minmax_permutated_qubits.first + BitInteger{1u}))
                      bitor ((value_wo_qubits bitand compl middle_bits_mask) << (minmax_permutated_qubits.first + BitInteger{2u}))
                      bitor data_block_mask;
                  // ****||xxx1xxx0000
                  auto const index1 = base_index bitor max_permutated_qubit_mask;
                  // ****||xxx0xxx1000
                  auto const index2 = base_index bitor min_permutated_qubit_mask;

                  std::swap_ranges(
                    local_state_first + index1,
                    local_state_first + (index1 bitor min_permutated_qubit_mask),
                    local_state_first + index2);
                });
            }
          }
        }; // struct swap_permutated_local_qubits<LocalState_>
      } // namespace dispatch

      namespace detail
      {
        template <typename ParallelPolicy, typename LocalState, typename StateInteger, typename BitInteger>
        inline auto swap_permutated_local_qubits(
          ParallelPolicy const parallel_policy,
          LocalState& local_state,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_qubit1,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_qubit2,
          StateInteger const num_data_blocks, StateInteger const data_block_size,
          yampi::communicator const& communicator, yampi::environment const& environment)
        -> void
        {
          using swap_permutated_local_qubits_
            = ::ket::mpi::utility::dispatch::swap_permutated_local_qubits<std::remove_cv_t<LocalState>>;
          swap_permutated_local_qubits_::call(
            parallel_policy, local_state, permutated_qubit1, permutated_qubit2,
            num_data_blocks, data_block_size, communicator, environment);
        }
      } // namespace detail
    } // namespace utility
  } // namespace mpi
} // namespace ket


#endif // KET_MPI_UTILITY_DETAIL_SWAP_PERMUTATED_LOCAL_QUBITS_HPP
