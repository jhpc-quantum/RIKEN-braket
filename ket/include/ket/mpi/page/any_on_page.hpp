#ifndef KET_MPI_PAGE_ANY_ON_PAGE_HPP
# define KET_MPI_PAGE_ANY_ON_PAGE_HPP

# include <ket/mpi/page/is_on_page.hpp>


namespace ket
{
  namespace mpi
  {
    namespace page
    {
      template <typename LocalState>
      inline constexpr bool any_on_page(LocalState const& local_state)
      { return false; }

      template <typename LocalState, typename PermutatedQubit, typename... PermutatedQubits>
      inline constexpr bool any_on_page(
        LocalState const& local_state, PermutatedQubit const permutated_qubit, PermutatedQubits const... permutated_qubits)
      {
        return ::ket::mpi::page::is_on_page(permutated_qubit, local_state)
          or ::ket::mpi::page::any_on_page(local_state, permutated_qubits...);
      }
    } // namespace page
  } // namepsace mpi
} // namespace ket


#endif // KET_MPI_PAGE_ANY_ON_PAGE_HPP
