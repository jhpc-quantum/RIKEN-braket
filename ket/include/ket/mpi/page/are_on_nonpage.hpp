#ifndef KET_MPI_PAGE_ARE_ON_NONPAGE_HPP
# define KET_MPI_PAGE_ARE_ON_NONPAGE_HPP

# include <ket/qubit.hpp>
# include <ket/control.hpp>
# include <ket/mpi/page/is_on_page.hpp>


namespace ket
{
  namespace mpi
  {
    namespace page
    {
      template <typename LocalState, typename StateInteger, typename BitInteger, typename Allocator>
      inline constexpr bool are_on_nonpage(
        LocalState const& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator> const& permutation)
      { return true; }

      template <typename LocalState, typename StateInteger, typename BitInteger, typename Allocator, typename Qubit, typename... Qubits>
      inline constexpr bool are_on_nonpage(
        LocalState const& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator> const& permutation,
        Qubit const qubit, Qubits const... qubits)
      {
        return
          (not ::ket::mpi::page::is_on_page(permutation[qubit], local_state))
          and ::ket::mpi::page::are_on_nonpage(local_state, permutation, qubits...);
      }
    } // namespace page
  } // namepsace mpi
} // namespace ket


#endif // KET_MPI_PAGE_ARE_ON_NONPAGE_HPP
