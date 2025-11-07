#ifndef KET_MPI_UTILITY_DETAIL_FOR_EACH_IN_DIAGONAL_LOOP_HPP
# define KET_MPI_UTILITY_DETAIL_FOR_EACH_IN_DIAGONAL_LOOP_HPP
# ifdef KET_USE_DIAGONAL_LOOP

#   include <cstddef>
#   include <array>
#   include <algorithm>
#   include <numeric>
#   include <iterator>
#   include <utility>
#   include <type_traits>

#   include <ket/qubit.hpp>
#   include <ket/control.hpp>
#   include <ket/utility/loop_n.hpp>
#   include <ket/mpi/permutated.hpp>


namespace ket
{
  namespace mpi
  {
    namespace utility
    {
      namespace dispatch
      {
        template <typename LocalState_>
        struct for_each_in_diagonal_loop
        {
          template <
            typename ParallelPolicy, typename LocalState, typename StateInteger, typename BitInteger,
            std::size_t num_local_control_qubits, typename Function>
          static auto call(
            ParallelPolicy const parallel_policy,
            LocalState&& local_state,
            StateInteger const data_block_index, StateInteger const data_block_size,
            StateInteger const last_local_qubit_value,
            std::array< ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > >, num_local_control_qubits > local_permutated_control_qubits,
            Function&& function)
          -> void
          {
            using std::begin;
            using std::end;
            std::sort(begin(local_permutated_control_qubits), end(local_permutated_control_qubits));

            impl(
              parallel_policy, std::forward<LocalState>(local_state),
              data_block_index, data_block_size, last_local_qubit_value,
              local_permutated_control_qubits, std::forward<Function>(function));
          }

         private:
          template <
            typename ParallelPolicy, typename LocalState, typename StateInteger, typename BitInteger,
            std::size_t num_local_control_qubits, typename Function>
          static auto impl(
            ParallelPolicy const parallel_policy,
            LocalState&& local_state,
            StateInteger const data_block_index, StateInteger const data_block_size,
            StateInteger const last_local_qubit_value,
            std::array< ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > >, num_local_control_qubits > const& sorted_local_permutated_control_qubits,
            Function&& function)
          -> void
          {
            constexpr auto zero_state_integer = StateInteger{0u};

            using permutated_control_qubit_type
              = ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > >;
            using std::begin;
            using std::end;
            // 000101000100
            auto const mask
              = std::accumulate(
                  begin(sorted_local_permutated_control_qubits), end(sorted_local_permutated_control_qubits),
                  zero_state_integer,
                  [](StateInteger const& partial_mask, permutated_control_qubit_type const& permutated_control_qubit)
                  {
                    constexpr auto one_state_integer = StateInteger{1u};
                    return partial_mask bitor (one_state_integer << permutated_control_qubit);
                  });

            auto const last_integer = last_local_qubit_value >> num_local_control_qubits;

            auto const first = begin(std::forward<LocalState>(local_state));
            auto const first_index = data_block_index * data_block_size;
            ::ket::utility::loop_n(
              parallel_policy, last_integer,
              [&function, &sorted_local_permutated_control_qubits, mask, first, first_index](StateInteger state_integer, int const)
              {
                constexpr auto one_state_integer = StateInteger{1u};

                // xxx0x0xxx0xx
                for (permutated_control_qubit_type const& permutated_control_qubit: sorted_local_permutated_control_qubits)
                {
                  auto const lower_mask = (one_state_integer << permutated_control_qubit) - one_state_integer;
                  auto const upper_mask = compl lower_mask;
                  state_integer = (state_integer bitand lower_mask) bitor ((state_integer bitand upper_mask) << 1u);
                }

                // xxx1x1xxx1xx
                state_integer |= mask;

                function(first + first_index + state_integer, first_index + state_integer);
              });
          }
        }; // struct for_each_in_diagonal_loop<LocalState_>
      } // namespace dispatch

      namespace detail
      {
        template <
          typename ParallelPolicy, typename LocalState,
          typename StateInteger, typename BitInteger, std::size_t num_local_control_qubits,
          typename Function>
        inline auto for_each_in_diagonal_loop(
          ParallelPolicy const parallel_policy,
          LocalState&& local_state,
          StateInteger const data_block_index, StateInteger const data_block_size,
          StateInteger const last_local_qubit_value,
          std::array< ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > >, num_local_control_qubits > const& local_permutated_control_qubits,
          Function&& function)
        -> void
        {
          using for_each_in_diagonal_loop_type
            = ::ket::mpi::utility::dispatch::for_each_in_diagonal_loop<std::remove_cv_t<std::remove_reference_t<LocalState>>>;
          return for_each_in_diagonal_loop_type::call(
            parallel_policy, std::forward<LocalState>(local_state), data_block_index, data_block_size, last_local_qubit_value,
            local_permutated_control_qubits, std::forward<Function>(function));
        }
      } // namespace detail
    } // namespace utility
  } // namespace mpi
} // namespace ket


# endif // KET_USE_DIAGONAL_LOOP
#endif // KET_MPI_UTILITY_DETAIL_FOR_EACH_IN_DIAGONAL_LOOP_HPP
