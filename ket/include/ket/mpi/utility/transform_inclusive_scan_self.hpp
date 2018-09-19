#ifndef KET_MPI_UTILITY_TRANSFORM_INCLUSIVE_SCAN_SELF_HPP
# define KET_MPI_UTILITY_TRANSFORM_INCLUSIVE_SCAN_SELF_HPP

# include <boost/config.hpp>

# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   include <type_traits>
# else
#   include <boost/utility/enable_if.hpp>
# endif

# include <boost/range/begin.hpp>
# include <boost/range/end.hpp>
# include <boost/range/value_type.hpp>
# include <boost/utility.hpp> // boost::prior

# include <ket/utility/loop_n.hpp>

# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   define KET_enable_if std::enable_if
# else
#   define KET_enable_if boost::enable_if_c
# endif


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
        typename Value, typename ParallelPolicy,
        typename LocalState, typename BinaryOperation, typename UnaryOperation>
      inline typename KET_enable_if<
        ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
        Value>::type
      transform_inclusive_scan_self(
        ParallelPolicy const parallel_policy, LocalState& local_state,
        BinaryOperation binary_operation, UnaryOperation unary_operation)
      {
        return static_cast<Value>(
          ::ket::mpi::utility::dispatch::transform_inclusive_scan_self<LocalState>::call(
            parallel_policy,
            local_state, binary_operation, unary_operation));
      }

      template <
        typename Value, typename LocalState, typename BinaryOperation, typename UnaryOperation>
      inline Value transform_inclusive_scan_self(
        LocalState& local_state,
        BinaryOperation binary_operation, UnaryOperation unary_operation)
      {
        return static_cast<Value>(
          ::ket::mpi::utility::dispatch::transform_inclusive_scan_self<LocalState>::call(
            ::ket::utility::policy::make_sequential(),
            local_state, binary_operation, unary_operation));
      }

      template <
        typename Value1, typename ParallelPolicy,
        typename LocalState, typename BinaryOperation, typename UnaryOperation, typename Value2>
      inline Value1 transform_inclusive_scan_self(
        ParallelPolicy const parallel_policy, LocalState& local_state,
        BinaryOperation binary_operation, UnaryOperation unary_operation,
        Value2 const initial_value)
      {
        return static_cast<Value1>(
          ::ket::mpi::utility::dispatch::transform_inclusive_scan_self<LocalState>::call(
            parallel_policy,
            local_state, binary_operation, unary_operation, initial_value));
      }

      template <
        typename Value1, typename LocalState,
        typename BinaryOperation, typename UnaryOperation, typename Value2>
      inline typename KET_enable_if<
        not ::ket::utility::policy::meta::is_loop_n_policy<LocalState>::value,
        Value1>::type
      transform_inclusive_scan_self(
        LocalState& local_state,
        BinaryOperation binary_operation, UnaryOperation unary_operation,
        Value2 const initial_value)
      {
        return static_cast<Value1>(
          ::ket::mpi::utility::dispatch::transform_inclusive_scan_self<LocalState>::call(
            ::ket::utility::policy::make_sequential(),
            local_state, binary_operation, unary_operation, initial_value));
      }
    } // namespace utility
  } // namespace mpi
} // namespace ket


# undef KET_enable_if

#endif

