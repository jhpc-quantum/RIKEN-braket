#ifndef KET_MPI_GATE_PAGE_PAULI_X_HPP
# define KET_MPI_GATE_PAGE_PAULI_X_HPP

# include <boost/config.hpp>

# include <algorithm>

# include <ket/qubit.hpp>
# include <ket/mpi/qubit_permutation.hpp>
# include <ket/mpi/gate/page/detail/one_page_qubit_gate.hpp>


namespace ket
{
  namespace mpi
  {
    namespace gate
    {
      namespace page
      {
        namespace pauli_x_detail
        {
# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
          struct pauli_x
          {
            template <typename Iterator, typename StateInteger>
            void operator()(Iterator const zero_first, Iterator const one_first, StateInteger const index, int const) const
            { std::iter_swap(zero_first + index, one_first + index); }
          }; // struct pauli_x
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        } // namespace pauli_x_detail

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange,
          typename StateInteger, typename BitInteger, typename Allocator>
        inline RandomAccessRange& pauli_x(
          MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::qubit<StateInteger, BitInteger> const qubit,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, Allocator> const& permutation)
        {
# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<0u>(
            mpi_policy, parallel_policy, local_state, qubit, permutation,
            [](auto const zero_first, auto const one_first, StateInteger const index, int const)
            { std::iter_swap(zero_first + index, one_first + index); });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<0u>(
            mpi_policy, parallel_policy, local_state, qubit, permutation,
            ::ket::mpi::gate::page::pauli_x_detail::pauli_x{});
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        }

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange,
          typename StateInteger, typename BitInteger, typename Allocator>
        inline RandomAccessRange& adj_pauli_x(
          MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::qubit<StateInteger, BitInteger> const qubit,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, Allocator> const& permutation)
        {
          return ::ket::mpi::gate::page::pauli_x(
            mpi_policy, parallel_policy, local_state, qubit, permutation);
        }
      } // namespace page
    } // namespace gate
  } // namespace mpi
} // namespace ket


#endif // KET_MPI_GATE_PAGE_PAULI_X_HPP
