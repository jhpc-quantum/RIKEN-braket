#ifndef KET_MPI_UTILITY_FILL_HPP
# define KET_MPI_UTILITY_FILL_HPP

# include <boost/config.hpp>

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
      namespace fill_detail
      {
# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
        template <typename ParallelPolicy, typename Value>
        struct fill
        {
          ParallelPolicy parallel_policy_;
          Value const& value_;

          fill(ParallelPolicy const parallel_policy, Value const& value)
            : parallel_policy_{parallel_policy}, value_{value}
          { }

          typedef void result_type;
          template <typename Iterator>
          void operator()(Iterator const first, Iterator const last) const
          { ::ket::utility::fill(parallel_policy_, first, last, value_); }
        }; // struct fill<ParallelPolicy, Value>

        template <typename ParallelPolicy, typename Value>
        inline ::ket::mpi::utility::fill_detail::fill<ParallelPolicy, Value> make_fill(
          ParallelPolicy const parallel_policy, Value const& value)
        {
          return ::ket::mpi::utility::fill_detail::fill<ParallelPolicy, Value>{
            parallel_policy, value};
        }
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
      } // namespace fill_detail

      template <typename MpiPolicy, typename ParallelPolicy, typename LocalState, typename Value>
      inline LocalState& fill(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        LocalState& local_state, Value const& value,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
        return ::ket::mpi::utility::for_each_local_range(
          mpi_policy, local_state, communicator, environment,
          [parallel_policy, &value](auto const first, auto const last)
          { ::ket::utility::fill(parallel_policy, first, last, value); });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
        return ::ket::mpi::utility::for_each_local_range(
          mpi_policy, local_state, communicator, environment,
          ::ket::mpi::utility::fill_detail::make_fill(parallel_policy, value));
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
      }

      template <typename LocalState, typename Value>
      inline LocalState& fill(
        LocalState& local_state, Value const& value,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        return ::ket::mpi::utility::fill(
          ::ket::mpi::utility::policy::make_simple_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, value, communicator, environment);
      }
    } // namespace utility
  } // namespace mpi
} // namespace ket


#endif // KET_MPI_UTILITY_FILL_HPP
