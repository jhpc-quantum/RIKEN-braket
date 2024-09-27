#ifndef KET_MPI_UTILITY_FILL_HPP
# define KET_MPI_UTILITY_FILL_HPP

# include <yampi/communicator.hpp>
# include <yampi/environment.hpp>

# include <ket/utility/loop_n.hpp>
# include <ket/mpi/utility/simple_mpi.hpp>
# include <ket/mpi/utility/for_each_local_range.hpp>


namespace ket
{
  namespace mpi
  {
    namespace utility
    {
      template <typename MpiPolicy, typename ParallelPolicy, typename LocalState, typename Value>
      inline auto fill(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        LocalState& local_state, Value const& value,
        yampi::communicator const& communicator, yampi::environment const& environment)
      -> LocalState&
      {
        return ::ket::mpi::utility::for_each_local_range(
          mpi_policy, local_state, communicator, environment,
          [parallel_policy, &value](auto const first, auto const last)
          { ::ket::utility::fill(parallel_policy, first, last, value); });
      }

      template <typename LocalState, typename Value>
      inline auto fill(
        LocalState& local_state, Value const& value,
        yampi::communicator const& communicator, yampi::environment const& environment)
      -> LocalState&
      {
        return ::ket::mpi::utility::fill(
          ::ket::mpi::utility::policy::make_simple_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, value, communicator, environment);
      }

      template <typename ParallelPolicy, typename LocalState, typename Value>
      inline auto fill(
        ParallelPolicy const parallel_policy,
        LocalState& local_state, Value const& value,
        yampi::communicator const& communicator, yampi::environment const& environment)
      -> LocalState&
      {
        return ::ket::mpi::utility::fill(
          ::ket::mpi::utility::policy::make_simple_mpi(),
          parallel_policy, local_state, value, communicator, environment);
      }
    } // namespace utility
  } // namespace mpi
} // namespace ket


#endif // KET_MPI_UTILITY_FILL_HPP
