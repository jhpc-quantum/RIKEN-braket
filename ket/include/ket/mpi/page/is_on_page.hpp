#ifndef KET_MPI_PAGE_IS_ON_PAGE_HPP
# define KET_MPI_PAGE_IS_ON_PAGE_HPP

# include <boost/config.hpp>

# include <ket/qubit.hpp>
# include <ket/mpi/qubit_permutation.hpp>
# include <ket/mpi/state.hpp>


namespace ket
{
  namespace mpi
  {
    namespace page
    {
      template <typename StateInteger, typename BitInteger, typename LocalState, typename Allocator>
      inline BOOST_CONSTEXPR bool is_on_page(
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        LocalState const& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator> const& permutation)
      { return false; }

      template <
        typename StateInteger, typename BitInteger,
        typename Complex, typename StateAllocator,
        typename PermutationAllocator>
      inline bool is_on_page(
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::state<Complex, 0, StateAllocator> const& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, PermutationAllocator> const& permutation)
      { return false; }

      template <
        typename StateInteger, typename BitInteger,
        typename Complex, int num_page_qubits_, typename StateAllocator,
        typename PermutationAllocator>
      inline bool is_on_page(
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator> const& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, PermutationAllocator> const& permutation)
      { return local_state.is_page_qubit(permutation[qubit]); }
    }
  }
}


#endif

