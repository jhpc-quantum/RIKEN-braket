#ifndef KET_MPI_UTILITY_TRANSFORM_INCLUSIVE_SCAN_SELF_HPP
# define KET_MPI_UTILITY_TRANSFORM_INCLUSIVE_SCAN_SELF_HPP

# include <boost/config.hpp>

# include <boost/range/begin.hpp>
# include <boost/range/end.hpp>
# include <boost/range/value_type.hpp>
# include <boost/utility.hpp> // boost::prior

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
        struct transform_inclusive_scan_self
        {
          template <
            typename ParallelPolicy,
            typename LocalState, typename BinaryOperation, typename UnaryOperation>
          static typename boost::range_value<LocalState>::type call(
            ParallelPolicy const parallel_policy, LocalState& local_state,
            BinaryOperation binary_operation, UnaryOperation unary_operation)
          {
            ::ket::utility::ranges::transform_inclusive_scan(
              parallel_policy,
              local_state, boost::begin(local_state), binary_operation, unary_operation);
            return *boost::prior(boost::end(local_state));
          }

          template <
            typename ParallelPolicy,
            typename LocalState, typename BinaryOperation, typename UnaryOperation,
            typename Value>
          static typename boost::range_value<LocalState>::type call(
            ParallelPolicy const parallel_policy, LocalState& local_state,
            BinaryOperation binary_operation, UnaryOperation unary_operation,
            Value const initial_value)
          {
            ::ket::utility::ranges::transform_inclusive_scan(
              parallel_policy,
              local_state, boost::begin(local_state),
              binary_operation, unary_operation, initial_value);
            return *boost::prior(boost::end(local_state));
          }
        };
      } // namespace dispatch

      template <
        typename ParallelPolicy,
        typename LocalState, typename BinaryOperation, typename UnaryOperation>
      inline typename boost::range_value<LocalState>::type
      transform_inclusive_scan_self(
        ParallelPolicy const parallel_policy, LocalState& local_state,
        BinaryOperation binary_operation, UnaryOperation unary_operation)
      {
        return ::ket::mpi::utility::dispatch::transform_inclusive_scan_self<LocalState>::call(
          parallel_policy,
          local_state, binary_operation, unary_operation);
      }

      template <
        typename ParallelPolicy,
        typename LocalState, typename BinaryOperation, typename UnaryOperation, typename Value>
      inline typename boost::range_value<LocalState>::type
      transform_inclusive_scan_self(
        ParallelPolicy const parallel_policy, LocalState& local_state,
        BinaryOperation binary_operation, UnaryOperation unary_operation,
        Value const initial_value)
      {
        return ::ket::mpi::utility::dispatch::transform_inclusive_scan_self<LocalState>::call(
          parallel_policy,
          local_state, binary_operation, unary_operation, initial_value);
      }
    } // namespace utility
  } // namespace mpi
} // namespace ket


#endif

