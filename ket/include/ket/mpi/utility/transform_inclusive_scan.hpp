#ifndef KET_MPI_UTILITY_TRANSFORM_INCLUSIVE_SCAN_HPP
# define KET_MPI_UTILITY_TRANSFORM_INCLUSIVE_SCAN_HPP

# include <iterator>
# include <type_traits>

# include <boost/range/value_type.hpp>

# include <yampi/environment.hpp>

# include <ket/utility/loop_n.hpp>


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
            typename ParallelPolicy,
            typename LocalState, typename ForwardIterator,
            typename BinaryOperation, typename UnaryOperation>
          static typename boost::range_value<LocalState>::type call(
            ParallelPolicy const parallel_policy,
            LocalState const& local_state, ForwardIterator const d_first,
            BinaryOperation binary_operation, UnaryOperation unary_operation,
            yampi::environment const&)
          {
            ::ket::utility::ranges::transform_inclusive_scan(
              parallel_policy,
              local_state, d_first, binary_operation, unary_operation);
            return *std::prev(std::end(local_state));
          }

          template <
            typename ParallelPolicy,
            typename LocalState, typename ForwardIterator,
            typename BinaryOperation, typename UnaryOperation,
            typename Value>
          static typename boost::range_value<LocalState>::type call(
            ParallelPolicy const parallel_policy,
            LocalState const& local_state, ForwardIterator const d_first,
            BinaryOperation binary_operation, UnaryOperation unary_operation,
            Value const initial_value, yampi::environment const&)
          {
            ::ket::utility::ranges::transform_inclusive_scan(
              parallel_policy,
              local_state, d_first,
              binary_operation, unary_operation, initial_value);
            return *std::prev(std::end(local_state));
          }
        }; // struct transform_inclusive_scan<LocalState_>
      } // namespace dispatch

      template <
        typename Value, typename ParallelPolicy,
        typename LocalState, typename ForwardIterator,
        typename BinaryOperation, typename UnaryOperation>
      inline typename std::enable_if<
        ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
        Value>::type
      transform_inclusive_scan(
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
      inline typename std::enable_if<
        not ::ket::utility::policy::meta::is_loop_n_policy<LocalState>::value,
        Value1>::type
      transform_inclusive_scan(
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
