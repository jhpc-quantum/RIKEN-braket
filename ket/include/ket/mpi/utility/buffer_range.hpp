#ifndef KET_MPI_UTILITY_BUFFER_RANGE_HPP
# define KET_MPI_UTILITY_BUFFER_RANGE_HPP

# include <vector>
# include <iterator>

# include <boost/range/iterator_range.hpp>

# include <ket/utility/meta/ranges.hpp>


namespace ket
{
  namespace mpi
  {
    namespace utility
    {
      namespace dispatch
      {
        template <typename LocalState>
        struct buffer_range
        {
          template <typename Allocator>
          static auto call(LocalState&, std::vector< ::ket::utility::meta::range_value_t<LocalState>, Allocator >& buffer)
          -> boost::iterator_range<typename std::vector< ::ket::utility::meta::range_value_t<LocalState>, Allocator >::iterator>
          { using std::begin; using std::end; return {begin(buffer), end(buffer)}; }

          template <typename Allocator>
          static auto call(LocalState const&, std::vector< ::ket::utility::meta::range_value_t<LocalState>, Allocator > const& buffer)
          -> boost::iterator_range<typename std::vector< ::ket::utility::meta::range_value_t<LocalState>, Allocator >::const_iterator>
          { using std::begin; using std::end; return {begin(buffer), end(buffer)}; }

          template <typename Allocator>
          static auto call_begin(LocalState&, std::vector< ::ket::utility::meta::range_value_t<LocalState>, Allocator >& buffer)
          -> typename std::vector< ::ket::utility::meta::range_value_t<LocalState>, Allocator >::iterator
          { using std::begin; return begin(buffer); }

          template <typename Allocator>
          static auto call_begin(LocalState const&, std::vector< ::ket::utility::meta::range_value_t<LocalState>, Allocator > const& buffer)
          -> typename std::vector< ::ket::utility::meta::range_value_t<LocalState>, Allocator >::const_iterator
          { using std::begin; return begin(buffer); }

          template <typename Allocator>
          static auto call_end(LocalState&, std::vector< ::ket::utility::meta::range_value_t<LocalState>, Allocator >& buffer)
          -> typename std::vector< ::ket::utility::meta::range_value_t<LocalState>, Allocator >::iterator
          { using std::end; return end(buffer); }

          template <typename Allocator>
          static auto call_end(LocalState const&, std::vector< ::ket::utility::meta::range_value_t<LocalState>, Allocator > const& buffer)
          -> typename std::vector< ::ket::utility::meta::range_value_t<LocalState>, Allocator >::const_iterator
          { using std::end; return end(buffer); }
        }; // struct buffer_range<LocalState>
      } // namespace dispatch

      template <typename LocalState, typename Allocator>
      inline auto buffer_range(LocalState& local_state, std::vector< ::ket::utility::meta::range_value_t<LocalState>, Allocator >& buffer)
        -> decltype(::ket::mpi::utility::dispatch::buffer_range<LocalState>::call(local_state, buffer))
      { return ::ket::mpi::utility::dispatch::buffer_range<LocalState>::call(local_state, buffer); }

      template <typename LocalState, typename Allocator>
      inline auto buffer_range(LocalState const& local_state, std::vector< ::ket::utility::meta::range_value_t<LocalState>, Allocator > const& buffer)
        -> decltype(::ket::mpi::utility::dispatch::buffer_range<LocalState>::call(local_state, buffer))
      { return ::ket::mpi::utility::dispatch::buffer_range<LocalState>::call(local_state, buffer); }

      template <typename LocalState, typename Allocator>
      inline auto buffer_begin(LocalState& local_state, std::vector< ::ket::utility::meta::range_value_t<LocalState>, Allocator >& buffer)
        -> decltype(::ket::mpi::utility::dispatch::buffer_range<LocalState>::call_begin(local_state, buffer))
      { return ::ket::mpi::utility::dispatch::buffer_range<LocalState>::call_begin(local_state, buffer); }

      template <typename LocalState, typename Allocator>
      inline auto buffer_begin(LocalState const& local_state, std::vector< ::ket::utility::meta::range_value_t<LocalState>, Allocator > const& buffer)
        -> decltype(::ket::mpi::utility::dispatch::buffer_range<LocalState>::call_begin(local_state, buffer))
      { return ::ket::mpi::utility::dispatch::buffer_range<LocalState>::call_begin(local_state, buffer); }

      template <typename LocalState, typename Allocator>
      inline auto buffer_end(LocalState& local_state, std::vector< ::ket::utility::meta::range_value_t<LocalState>, Allocator >& buffer)
        -> decltype(::ket::mpi::utility::dispatch::buffer_range<LocalState>::call_begin(local_state, buffer))
      { return ::ket::mpi::utility::dispatch::buffer_range<LocalState>::call_end(local_state, buffer); }

      template <typename LocalState, typename Allocator>
      inline auto buffer_end(LocalState const& local_state, std::vector< ::ket::utility::meta::range_value_t<LocalState>, Allocator > const& buffer)
        -> decltype(::ket::mpi::utility::dispatch::buffer_range<LocalState>::call_begin(local_state, buffer))
      { return ::ket::mpi::utility::dispatch::buffer_range<LocalState>::call_end(local_state, buffer); }
    } // namespace utility
  } // namespace mpi
} // namespace ket


#endif // KET_MPI_UTILITY_BUFFER_RANGE_HPP
