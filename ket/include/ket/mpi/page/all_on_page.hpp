#ifndef KET_MPI_PAGE_ALL_ON_PAGE_HPP
# define KET_MPI_PAGE_ALL_ON_PAGE_HPP

# include <ket/mpi/page/is_on_page.hpp>


namespace ket
{
  namespace mpi
  {
    namespace page
    {
      template <typename LocalState>
      inline constexpr bool all_on_page(LocalState const& local_state)
      { return true; }

      template <typename PermutatedQubit, typename... PermutatedQubits, typename LocalState>
      inline constexpr bool all_on_page(
        PermutatedQubit const permutated_qubit, PermutatedQubits const... permutated_qubits, LocalState const& local_state)
      {
        return ::ket::mpi::page::is_on_page(permutated_qubit, local_state)
          and ::ket::mpi::page::all_on_page(permutated_qubits..., local_state);
      }
    } // namespace page
  } // namepsace mpi
} // namespace ket


#endif // KET_MPI_PAGE_ALL_ON_PAGE_HPP
