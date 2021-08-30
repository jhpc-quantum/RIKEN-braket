#ifndef KET_MPI_PAGE_IS_ON_PAGE_ARRAY_HPP
# define KET_MPI_PAGE_IS_ON_PAGE_ARRAY_HPP

# include <array>

# include <ket/mpi/page/is_on_page.hpp>


namespace ket
{
  namespace mpi
  {
    namespace page
    {
      namespace is_on_page_array_detail
      {
        template <std::size_t num_permutated_qubits, typename LocalState>
        inline void is_on_page_array(std::array<bool, num_permutated_qubits>&, LocalState const&)
        { }

        template <std::size_t num_permutated_qubits, typename PermutatedQubit, typename... PermutatedQubits, typename LocalState>
        inline void is_on_page_array(
          std::array<bool, num_permutated_qubits>& result,
          PermutatedQubit const permutated_qubit, PermutatedQubits const... permutated_qubits, LocalState const& local_state)
        {
          result[num_permutated_qubits - sizeof...(PermutatedQubits) - 1u] = ::ket::mpi::page::is_on_page(permutated_qubit, local_state);
          ::ket::mpi::page::is_on_page_array_detail::is_on_page_array(result, permutated_qubits..., local_state);
        }
      } // namespace is_on_page_array_detail

      template <typename PermutatedQubit, typename... PermutatedQubits, typename LocalState>
      inline std::array<bool, sizeof...(PermutatedQubits) + 1u> is_on_page_array(
        PermutatedQubit const permutated_qubit, PermutatedQubits const... permutated_qubits, LocalState const& local_state)
      {
        std::array<bool, sizeof...(PermutatedQubits) + 1u> result;
        ::ket::mpi::page::is_on_page_array_detail::is_on_page_array(result, permutated_qubit, permutated_qubits..., local_state);
        return result;
      }
    } // namespace page
  } // namepsace mpi
} // namespace ket


#endif // KET_MPI_PAGE_IS_ON_PAGE_ARRAY_HPP
