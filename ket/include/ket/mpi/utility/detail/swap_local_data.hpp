#ifndef KET_MPI_UTILITY_DETAIL_SWAP_LOCAL_DATA_HPP
# define KET_MPI_UTILITY_DETAIL_SWAP_LOCAL_DATA_HPP

# include <algorithm>
# include <iterator>
# include <type_traits>


namespace ket
{
  namespace mpi
  {
    namespace utility
    {
      namespace dispatch
      {
        template <typename LocalState_>
        struct swap_local_data
        {
          template <typename LocalState, typename StateInteger>
          static auto call(
            LocalState& local_state,
            StateInteger const data_block_index1, StateInteger const local_first_index1, StateInteger const local_last_index1,
            StateInteger const data_block_index2, StateInteger const local_first_index2,
            StateInteger const data_block_size)
          -> void
          {
            using std::begin;
            auto const first1 = begin(local_state) + data_block_index1 * data_block_size + local_first_index1;
            auto const last1 = begin(local_state) + data_block_index1 * data_block_size + local_last_index1;
            auto const first2 = begin(local_state) + data_block_index2 * data_block_size + local_first_index2;

            std::swap_ranges(first1, last1, first2);
          }
        }; // struct swap_local_data<LocalState_>
      } // namespace dispatch

      namespace detail
      {
        template <typename LocalState, typename StateInteger>
        inline auto swap_local_data(
          LocalState& local_state,
          StateInteger const data_block_index1, StateInteger const local_first_index1, StateInteger const local_last_index1,
          StateInteger const data_block_index2, StateInteger const local_first_index2,
          StateInteger const data_block_size)
        -> void
        {
          using local_state_type = std::remove_cv_t<std::remove_reference_t<LocalState>>;
          ::ket::mpi::utility::dispatch::swap_local_data<local_state_type>::call(
            local_state,
            data_block_index1, local_first_index1, local_last_index1,
            data_block_index2, local_first_index2, data_block_size);
        }
      } // namespace detail
    } // namespace utility
  } // namespace mpi
} // namespace ket


#endif // KET_MPI_UTILITY_DETAIL_SWAP_LOCAL_DATA_HPP
