#ifndef KET_MPI_UTILITY_TRANSFORM_INCLUSIVE_SCAN_SELF_HPP
# define KET_MPI_UTILITY_TRANSFORM_INCLUSIVE_SCAN_SELF_HPP

# include <iterator>

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
        struct transform_inclusive_scan_self
        {
          template <
            typename ParallelPolicy,
            typename LocalState, typename BinaryOperation, typename UnaryOperation>
          static auto call(
            ParallelPolicy const parallel_policy, LocalState& local_state,
            BinaryOperation binary_operation, UnaryOperation unary_operation,
            yampi::environment const&)
          -> ::ket::utility::meta::range_value_t<LocalState>
          {
            using std::begin;
            ::ket::utility::ranges::transform_inclusive_scan(
              parallel_policy, local_state, begin(local_state), binary_operation, unary_operation);

            using std::end;
            return *std::prev(end(local_state));
          }

          template <
            typename ParallelPolicy,
            typename LocalState, typename BinaryOperation, typename UnaryOperation, typename Value>
          static auto call(
            ParallelPolicy const parallel_policy, LocalState& local_state,
            BinaryOperation binary_operation, UnaryOperation unary_operation,
            Value const initial_value, yampi::environment const&)
          -> ::ket::utility::meta::range_value_t<LocalState>
          {
            using std::begin;
            ::ket::utility::ranges::transform_inclusive_scan(
              parallel_policy,
              local_state, begin(local_state), binary_operation, unary_operation, initial_value);

            using std::end;
            return *std::prev(end(local_state));
          }
        };
      } // namespace dispatch

      template <
        typename ParallelPolicy,
        typename LocalState, typename BinaryOperation, typename UnaryOperation>
      inline auto transform_inclusive_scan_self(
        ParallelPolicy const parallel_policy, LocalState& local_state,
        BinaryOperation binary_operation, UnaryOperation unary_operation,
        yampi::environment const& environment)
      -> ::ket::utility::meta::range_value_t<LocalState>
      {
        return ::ket::mpi::utility::dispatch::transform_inclusive_scan_self<LocalState>::call(
          parallel_policy,
          local_state, binary_operation, unary_operation, environment);
      }

      template <
        typename ParallelPolicy,
        typename LocalState, typename BinaryOperation, typename UnaryOperation, typename Value>
      inline auto transform_inclusive_scan_self(
        ParallelPolicy const parallel_policy, LocalState& local_state,
        BinaryOperation binary_operation, UnaryOperation unary_operation,
        Value const initial_value,
        yampi::environment const& environment)
      -> ::ket::utility::meta::range_value_t<LocalState>
      {
        return ::ket::mpi::utility::dispatch::transform_inclusive_scan_self<LocalState>::call(
          parallel_policy,
          local_state, binary_operation, unary_operation, initial_value, environment);
      }
    } // namespace utility
  } // namespace mpi
} // namespace ket


#endif // KET_MPI_UTILITY_TRANSFORM_INCLUSIVE_SCAN_SELF_HPP
