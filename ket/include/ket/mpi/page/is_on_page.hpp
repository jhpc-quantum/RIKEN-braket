#ifndef KET_MPI_PAGE_IS_ON_PAGE_HPP
# define KET_MPI_PAGE_IS_ON_PAGE_HPP

# include <ket/qubit.hpp>
# include <ket/mpi/qubit_permutation.hpp>


namespace ket
{
  namespace mpi
  {
    namespace page
    {
      template <typename StateInteger, typename BitInteger, typename LocalState, typename Allocator>
      inline constexpr bool is_on_page(
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        LocalState const& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator> const& permutation)
      { return false; }
    } // namespace page
  } // namespace mpi
} // namespace ket


#endif // KET_MPI_PAGE_IS_ON_PAGE_HPP
