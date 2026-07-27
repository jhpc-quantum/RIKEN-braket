#ifndef KET_MPI_PAGE_ANY_ON_PAGE_HPP
# define KET_MPI_PAGE_ANY_ON_PAGE_HPP

# include <algorithm>
# include <iterator>

# include <ket/mpi/page/is_on_page.hpp>


namespace ket
{
  namespace mpi
  {
    namespace page
    {
      namespace runtime
      {
        template <typename LocalState, typename PermutatedQubitIterator>
        inline auto any_on_page(
          LocalState const& local_state, PermutatedQubitIterator const permutated_qubit_first, PermutatedQubitIterator const permutated_qubit_last)
        -> bool
        {
          using permutated_qubit_type = typename std::iterator_traits<PermutatedQubitIterator>::value_type;
          return std::any_of(
            permutated_qubit_first, permutated_qubit_last,
            [&local_state](permutated_qubit_type const permutated_qubit)
            { return ::ket::mpi::page::is_on_page(permutated_qubit, local_state); });
        }

        template <typename LocalState, typename StateInteger, typename BitInteger, typename PermutatedControlQubitIterator>
        inline auto any_on_page(
          LocalState const& local_state,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_target_qubit,
          PermutatedControlQubitIterator const permutated_qubit_first, PermutatedControlQubitIterator const permutated_qubit_last)
        -> bool
        {
          if (::ket::mpi::page::is_on_page(permutated_target_qubit, local_state))
            return true;

          return ::ket::mpi::page::runtime::any_on_page(local_state, permutated_qubit_first, permutated_qubit_last);
        }

        template <typename LocalState, typename PermutatedQubitIterator, typename PermutatedControlQubitIterator>
        inline auto any_on_page(
          LocalState const& local_state,
          PermutatedQubitIterator const permutated_target_qubit_first, PermutatedQubitIterator const permutated_target_qubit_last,
          PermutatedControlQubitIterator const permutated_control_qubit_first, PermutatedControlQubitIterator const permutated_control_qubit_last)
        -> bool
        {
          return ::ket::mpi::page::runtime::any_on_page(local_state, permutated_target_qubit_first, permutated_target_qubit_last)
            or ::ket::mpi::page::runtime::any_on_page(local_state, permutated_control_qubit_first, permutated_control_qubit_last);
        }

        namespace ranges
        {
          template <typename LocalState, typename PermutatedQubits>
          inline auto any_on_page(LocalState const& local_state, PermutatedQubits const& permutated_qubits)
          -> bool
          {
            using std::begin;
            using std::end;
            return ::ket::mpi::page::runtime::any_on_page(local_state, begin(permutated_qubits), end(permutated_qubits));
          }

          template <typename LocalState, typename StateInteger, typename BitInteger, typename PermutatedControlQubits>
          inline auto any_on_page(
            LocalState const& local_state,
            ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_target_qubit,
            PermutatedControlQubits const& permutated_control_qubits)
          -> bool
          {
            using std::begin;
            using std::end;
            return ::ket::mpi::page::runtime::any_on_page(
              local_state, permutated_target_qubit, begin(permutated_control_qubits), end(permutated_control_qubits));
          }

          template <typename LocalState, typename PermutatedQubits, typename PermutatedControlQubits>
          inline auto any_on_page(
            LocalState const& local_state,
            PermutatedQubits const& permutated_target_qubits, PermutatedControlQubits const& permutated_control_qubits)
          -> bool
          {
            using std::begin;
            using std::end;
            return ::ket::mpi::page::runtime::any_on_page(
              local_state,
              begin(permutated_target_qubits), end(permutated_target_qubits),
              begin(permutated_control_qubits), end(permutated_control_qubits));
          }
        } // namespace ranges
      } // namespace runtime

      template <typename LocalState>
      inline constexpr auto any_on_page(LocalState const& local_state) -> bool
      { return false; }

      template <typename LocalState, typename PermutatedQubit, typename... PermutatedQubits>
      inline constexpr auto any_on_page(
        LocalState const& local_state, PermutatedQubit const permutated_qubit, PermutatedQubits const... permutated_qubits)
      -> bool
      {
        return ::ket::mpi::page::is_on_page(permutated_qubit, local_state)
          or ::ket::mpi::page::any_on_page(local_state, permutated_qubits...);
      }
    } // namespace page
  } // namepsace mpi
} // namespace ket


#endif // KET_MPI_PAGE_ANY_ON_PAGE_HPP
