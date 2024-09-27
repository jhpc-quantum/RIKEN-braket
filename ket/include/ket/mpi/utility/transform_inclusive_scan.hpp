#ifndef KET_MPI_UTILITY_TRANSFORM_INCLUSIVE_SCAN_HPP
# define KET_MPI_UTILITY_TRANSFORM_INCLUSIVE_SCAN_HPP

# include <iterator>
# include <type_traits>

# include <yampi/environment.hpp>

# include <ket/utility/loop_n.hpp>
# include <ket/utility/meta/ranges.hpp>


namespace ket
{
  namespace mpi
  {
    namespace utility
    {
      namespace dispatch
      {
        template <typename LocalState_>
        struct transform_inclusive_scan
        {
          template <
            typename ParallelPolicy, typename LocalState, typename ForwardIterator,
            typename BinaryOperation, typename UnaryOperation>
          static auto call(
            ParallelPolicy const parallel_policy,
            LocalState const& local_state, ForwardIterator const d_first,
            BinaryOperation binary_operation, UnaryOperation unary_operation,
            yampi::environment const&)
          -> ::ket::utility::meta::range_value_t<LocalState>
          {
            ::ket::utility::ranges::transform_inclusive_scan(
              parallel_policy, local_state, d_first, binary_operation, unary_operation);

            using std::end;
            return *std::prev(end(local_state));
          }

          template <
            typename ParallelPolicy, typename LocalState, typename ForwardIterator,
            typename BinaryOperation, typename UnaryOperation, typename Value>
          static auto call(
            ParallelPolicy const parallel_policy,
            LocalState const& local_state, ForwardIterator const d_first,
            BinaryOperation binary_operation, UnaryOperation unary_operation,
            Value const initial_value, yampi::environment const&)
          -> ::ket::utility::meta::range_value_t<LocalState>
          {
            ::ket::utility::ranges::transform_inclusive_scan(
              parallel_policy,
              local_state, d_first, binary_operation, unary_operation, initial_value);

            using std::end;
            return *std::prev(end(local_state));
          }
        }; // struct transform_inclusive_scan<LocalState_>
      } // namespace dispatch

      template <
        typename Value, typename ParallelPolicy,
        typename LocalState, typename ForwardIterator,
        typename BinaryOperation, typename UnaryOperation>
      inline std::enable_if_t< ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value, Value > transform_inclusive_scan(
        ParallelPolicy const parallel_policy,
        LocalState const& local_state, ForwardIterator const d_first,
        BinaryOperation binary_operation, UnaryOperation unary_operation,
        yampi::environment const& environment)
      {
        return static_cast<Value>(
          ::ket::mpi::utility::dispatch::transform_inclusive_scan<LocalState>::call(
            parallel_policy,
            local_state, d_first, binary_operation, unary_operation, environment));
      }

      template <
        typename Value, typename LocalState, typename OutputIterator,
        typename BinaryOperation, typename UnaryOperation>
      inline Value transform_inclusive_scan(
        LocalState const& local_state, OutputIterator const d_first,
        BinaryOperation binary_operation, UnaryOperation unary_operation,
        yampi::environment const& environment)
      {
        return static_cast<Value>(
          ::ket::mpi::utility::dispatch::transform_inclusive_scan<LocalState>::call(
            ::ket::utility::policy::make_sequential(),
            local_state, d_first, binary_operation, unary_operation, environment));
      }

      template <
        typename Value1, typename ParallelPolicy,
        typename LocalState, typename ForwardIterator,
        typename BinaryOperation, typename UnaryOperation, typename Value2>
      inline Value1 transform_inclusive_scan(
        ParallelPolicy const parallel_policy,
        LocalState const& local_state, ForwardIterator const d_first,
        BinaryOperation binary_operation, UnaryOperation unary_operation,
        Value2 const initial_value,
        yampi::environment const& environment)
      {
        return static_cast<Value1>(
          ::ket::mpi::utility::dispatch::transform_inclusive_scan<LocalState>::call(
            parallel_policy,
            local_state, d_first, binary_operation, unary_operation,
            initial_value, environment));
      }

      template <
        typename Value1, typename LocalState, typename OutputIterator,
        typename BinaryOperation, typename UnaryOperation, typename Value2>
      inline std::enable_if_t<not ::ket::utility::policy::meta::is_loop_n_policy<LocalState>::value, Value1> transform_inclusive_scan(
        LocalState const& local_state, OutputIterator const d_first,
        BinaryOperation binary_operation, UnaryOperation unary_operation,
        Value2 const initial_value,
        yampi::environment const& environment)
      {
        return static_cast<Value1>(
          ::ket::mpi::utility::dispatch::transform_inclusive_scan<LocalState>::call(
            ::ket::utility::policy::make_sequential(),
            local_state, d_first, binary_operation, unary_operation,
            initial_value, environment));
      }
    } // namespace utility
  } // namespace mpi
} // namespace ket


#endif // KET_MPI_UTILITY_TRANSFORM_INCLUSIVE_SCAN_HPP
