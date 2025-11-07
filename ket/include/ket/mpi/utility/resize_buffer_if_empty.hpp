#ifndef KET_MPI_UTILITY_RESIZE_BUFFER_IF_EMPTY_HPP
# define KET_MPI_UTILITY_RESIZE_BUFFER_IF_EMPTY_HPP

# include <cstddef>
# include <vector>

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
        struct resize_buffer_if_empty
        {
          template <typename LocalState, typename BufferAllocator>
          static auto call(
            LocalState const&,
            std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
            std::size_t const new_size)
          -> void
          {
            if (not buffer.empty())
              return;

            buffer.resize(new_size);
          }
        }; // struct resize_buffer_if_empty<LocalState_>
      } // namespace dispatch

      template <typename LocalState, typename BufferAllocator>
      inline auto resize_buffer_if_empty(
        LocalState const& local_state,
        std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
        std::size_t const new_size)
      -> void
      { ::ket::mpi::utility::dispatch::resize_buffer_if_empty<std::remove_cv_t<std::remove_reference_t<LocalState>>>::call(local_state, buffer, new_size); }
    } // namespace utility
  } // namespace mpi
} // namespace ket


#endif // KET_MPI_UTILITY_RESIZE_BUFFER_IF_EMPTY_HPP
