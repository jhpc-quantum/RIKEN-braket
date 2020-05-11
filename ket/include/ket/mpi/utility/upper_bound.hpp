#ifndef KET_MPI_UTILITY_UPPER_BOUND_HPP
# define KET_MPI_UTILITY_UPPER_BOUND_HPP

# include <boost/config.hpp>

# include <boost/range/difference_type.hpp>

# include <ket/utility/begin.hpp>
# include <ket/utility/end.hpp>


namespace ket
{
  namespace mpi
  {
    namespace utility
    {
      namespace dispatch
      {
        template <typename LocalState_>
        struct upper_bound
        {
          template <typename LocalState, typename Value, typename Compare>
          static typename boost::range_difference<LocalState>::type call(
            LocalState const& local_state, Value const& value, Compare compare)
          {
            return
              std::upper_bound(
                ::ket::utility::begin(local_state), ::ket::utility::end(local_state), value, compare)
              - ::ket::utility::begin(local_state);
          }
        };
      } // namespace dispatch

      template <typename LocalState, typename Value, typename Compare>
      inline typename boost::range_difference<LocalState>::type upper_bound(
        LocalState const& local_state, Value const& value, Compare compare)
      {
        return ::ket::mpi::utility::dispatch::upper_bound<LocalState>::call(
          local_state, value, compare);
      }
    } // namespace utility
  } // namespace mpi
} // namespace ket


#endif

