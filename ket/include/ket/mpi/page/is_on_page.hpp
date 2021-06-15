#ifndef KET_MPI_PAGE_IS_ON_PAGE_HPP
# define KET_MPI_PAGE_IS_ON_PAGE_HPP

# include <ket/qubit.hpp>
# include <ket/control.hpp>
# include <ket/mpi/permutated.hpp>


namespace ket
{
  namespace mpi
  {
    namespace page
    {
      template <typename StateInteger, typename BitInteger, typename LocalState>
      inline constexpr bool is_on_page(
        ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const,
        LocalState const&)
      { return false; }

      template <typename StateInteger, typename BitInteger, typename LocalState>
      inline constexpr bool is_on_page(
        ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const,
        LocalState const&)
      { return false; }
    } // namespace page
  } // namespace mpi
} // namespace ket


#endif // KET_MPI_PAGE_IS_ON_PAGE_HPP
