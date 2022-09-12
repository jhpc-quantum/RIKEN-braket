#ifndef KET_MPI_UTILITY_BUFFER_RANGE_HPP
# define KET_MPI_UTILITY_BUFFER_RANGE_HPP


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
          static boost::iterator_range<typename std::vector<typename boost::range_value<LocalState>::type, Allocator>::iterator> call(
            LocalState&, std::vector<typename boost::range_value<LocalState>::type, Allocator>& buffer)
          { return boost::make_iterator_range(std::begin(buffer), std::end(buffer)); }

          template <typename Allocator>
          static boost::iterator_range<typename std::vector<typename boost::range_value<LocalState>::type, Allocator>::const_iterator> call(
            LocalState const&, std::vector<typename boost::range_value<LocalState>::type, Allocator> const& buffer)
          { return boost::make_iterator_range(std::begin(buffer), std::end(buffer)); }

          template <typename Allocator>
          static typename std::vector<typename boost::range_value<LocalState>::type, Allocator>::iterator call_begin(
            LocalState&, std::vector<typename boost::range_value<LocalState>::type, Allocator>& buffer)
          { return std::begin(buffer); }

          template <typename Allocator>
          static typename std::vector<typename boost::range_value<LocalState>::type, Allocator>::const_iterator call_begin(
            LocalState const&, std::vector<typename boost::range_value<LocalState>::type, Allocator> const& buffer)
          { return std::begin(buffer); }

          template <typename Allocator>
          static typename std::vector<typename boost::range_value<LocalState>::type, Allocator>::iterator call_end(
            LocalState&, std::vector<typename boost::range_value<LocalState>::type, Allocator>& buffer)
          { return std::end(buffer); }

          template <typename Allocator>
          static typename std::vector<typename boost::range_value<LocalState>::type, Allocator>::const_iterator call_end(
            LocalState const&, std::vector<typename boost::range_value<LocalState>::type, Allocator> const& buffer)
          { return std::end(buffer); }
        }; // struct buffer_range<LocalState_>
      } // namespace dispatch

      template <typename LocalState, typename Allocator>
      inline auto buffer_range(
        LocalState& local_state, std::vector<typename boost::range_value<LocalState>::type, Allocator>& buffer)
        -> decltype(::ket::mpi::utility::dispatch::buffer_range<LocalState>::call_begin(local_state, buffer))
      { return ::ket::mpi::utility::dispatch::buffer_range<LocalState>::call(local_state, buffer); }

      template <typename LocalState, typename Allocator>
      inline auto buffer_range(
        LocalState const& local_state, std::vector<typename boost::range_value<LocalState>::type, Allocator> const& buffer)
        -> decltype(::ket::mpi::utility::dispatch::buffer_range<LocalState>::call_begin(local_state, buffer))
      { return ::ket::mpi::utility::dispatch::buffer_range<LocalState>::call(local_state, buffer); }

      template <typename LocalState, typename Allocator>
      inline auto buffer_begin(
        LocalState& local_state, std::vector<typename boost::range_value<LocalState>::type, Allocator>& buffer)
        -> decltype(::ket::mpi::utility::dispatch::buffer_range<LocalState>::call_begin(local_state, buffer))
      { return ::ket::mpi::utility::dispatch::buffer_range<LocalState>::call_begin(local_state, buffer); }

      template <typename LocalState, typename Allocator>
      inline auto buffer_begin(
        LocalState const& local_state, std::vector<typename boost::range_value<LocalState>::type, Allocator> const& buffer)
        -> decltype(::ket::mpi::utility::dispatch::buffer_range<LocalState>::call_begin(local_state, buffer))
      { return ::ket::mpi::utility::dispatch::buffer_range<LocalState>::call_begin(local_state, buffer); }

      template <typename LocalState, typename Allocator>
      inline auto buffer_end(
        LocalState& local_state, std::vector<typename boost::range_value<LocalState>::type, Allocator>& buffer)
        -> decltype(::ket::mpi::utility::dispatch::buffer_range<LocalState>::call_begin(local_state, buffer))
      { return ::ket::mpi::utility::dispatch::buffer_range<LocalState>::call_end(local_state, buffer); }

      template <typename LocalState, typename Allocator>
      inline auto buffer_end(
        LocalState const& local_state, std::vector<typename boost::range_value<LocalState>::type, Allocator> const& buffer)
        -> decltype(::ket::mpi::utility::dispatch::buffer_range<LocalState>::call_begin(local_state, buffer))
      { return ::ket::mpi::utility::dispatch::buffer_range<LocalState>::call_end(local_state, buffer); }
    } // namespace utility
  } // namespace mpi
} // namespace ket


#endif // KET_MPI_UTILITY_BUFFER_RANGE_HPP
