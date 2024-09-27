#ifndef KET_MPI_UTILITY_UPPER_BOUND_HPP
# define KET_MPI_UTILITY_UPPER_BOUND_HPP

# include <yampi/environment.hpp>

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
        struct upper_bound
        {
          template <typename LocalState, typename Value, typename Compare>
          static auto call(
            LocalState const& local_state, Value const& value, Compare compare,
            yampi::environment const&)
          -> ::ket::utility::meta::range_difference_t<LocalState>
          {
            using std::begin;
            using std::end;
            return std::upper_bound(begin(local_state), end(local_state), value, compare) - begin(local_state);
          }
        }; // struct upper_bound<LocalState_>
      } // namespace dispatch

      template <typename LocalState, typename Value, typename Compare>
      inline auto upper_bound(
        LocalState const& local_state, Value const& value, Compare compare,
        yampi::environment const& environment)
      -> ::ket::utility::meta::range_difference_t<LocalState>
      { return ::ket::mpi::utility::dispatch::upper_bound<LocalState>::call(local_state, value, compare, environment); }
    } // namespace utility
  } // namespace mpi
} // namespace ket


#endif // KET_MPI_UTILITY_UPPER_BOUND_HPP
