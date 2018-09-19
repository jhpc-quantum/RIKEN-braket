#ifndef KET_MPI_UTILITY_UPPER_BOUND_HPP
# define KET_MPI_UTILITY_UPPER_BOUND_HPP

# include <boost/config.hpp>

# ifdef KET_PREFER_POINTER_TO_VECTOR_ITERATOR
#   ifndef BOOST_NO_CXX11_ADDRESSOF
#     include <memory>
#   else
#     include <boost/core/addressof.hpp>
#   endif
# endif

# include <boost/range/begin.hpp>
# include <boost/range/end.hpp>
# include <boost/range/difference_type.hpp>

# ifdef KET_PREFER_POINTER_TO_VECTOR_ITERATOR
#   ifndef BOOST_NO_CXX11_ADDRESSOF
#     define KET_addressof std::addressof
#   else
#     define KET_addressof boost::addressof
#   endif
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
        struct upper_bound
        {
          template <typename LocalState, typename Value>
          static typename boost::range_difference<LocalState>::type call(
            LocalState const& local_state, Value const& value)
          {
            return
              std::upper_bound(
                boost::begin(local_state), boost::end(local_state), value)
              - boost::begin(local_state);
          }

          template <typename LocalState, typename Value, typename Compare>
          static typename boost::range_difference<LocalState>::type call(
            LocalState const& local_state, Value const& value, Compare compare)
          {
            return
              std::upper_bound(
                boost::begin(local_state), boost::end(local_state), value, compare)
              - boost::begin(local_state);
          }

# ifdef KET_PREFER_POINTER_TO_VECTOR_ITERATOR
          template <typename Value, typename Allocator>
          static typename boost::range_difference< std::vector<Value, Allocator> >::type call(
            std::vector<Value, Allocator> const& local_state, Value const& value)
          {
            return
              std::upper_bound(
                KET_addressof(local_state.front()),
                KET_addressof(local_state.front()) + local_state.size(),
                value)
              - KET_addressof(local_state.front());
          }

          template <typename Value, typename Allocator, typename Compare>
          static typename boost::range_difference< std::vector<Value, Allocator> >::type call(
            std::vector<Value, Allocator> const& local_state, Value const& value, Compare compare)
          {
            return
              std::upper_bound(
                KET_addressof(local_state.front()),
                KET_addressof(local_state.front()) + local_state.size(),
                value, compare)
              - KET_addressof(local_state.front());
          }
# endif // KET_PREFER_POINTER_TO_VECTOR_ITERATOR
        };
      } // namespace dispatch

      template <typename LocalState, typename Value>
      inline typename boost::range_difference<LocalState>::type upper_bound(
        LocalState const& local_state, Value const& value)
      {
        return ::ket::mpi::utility::dispatch::upper_bound<LocalState>::call(
          local_state, value);
      }

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


# ifdef KET_PREFER_POINTER_TO_VECTOR_ITERATOR
#   undef KET_addressof
# endif

#endif

