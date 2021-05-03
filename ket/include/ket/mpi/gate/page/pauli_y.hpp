#ifndef KET_MPI_GATE_PAGE_PAULI_Y_HPP
# define KET_MPI_GATE_PAGE_PAULI_Y_HPP

# include <boost/config.hpp>

# include <algorithm>

# include <boost/range/value_type.hpp>

# include <ket/qubit.hpp>
# include <ket/utility/imaginary_unit.hpp>
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
        namespace pauli_y_detail
        {
# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
          template <typename Complex>
          struct pauli_y
          {
            template <typename Iterator>
            void operator()(Iterator const zero_iter, Iterator const one_iter) const
            {
              std::iter_swap(zero_iter, one_iter);
              *zero_iter *= -::ket::utility::imaginary_unit<Complex>();
              *one_iter *= ::ket::utility::imaginary_unit<Complex>();
            }
          }; // struct pauli_y<Complex>
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        } // namespace pauli_y_detail

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange,
          typename StateInteger, typename BitInteger, typename Allocator>
        inline RandomAccessRange& pauli_y(
          MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::qubit<StateInteger, BitInteger> const qubit,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, Allocator> const& permutation)
        {
          using complex_type = typename boost::range_value<RandomAccessRange>::type;

# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate(
            mpi_policy, parallel_policy, local_state, qubit, permutation,
            [](auto const zero_iter, auto const one_iter)
            {
              std::iter_swap(zero_iter, one_iter);
              *zero_iter *= -::ket::utility::imaginary_unit<complex_type>();
              *one_iter *= ::ket::utility::imaginary_unit<complex_type>();
            });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate(
            mpi_policy, parallel_policy, local_state, qubit, permutation,
            ::ket::mpi::gate::page::pauli_y_detail::pauli_y<complex_type>{});
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        }

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange,
          typename StateInteger, typename BitInteger, typename Allocator>
        inline RandomAccessRange& adj_pauli_y(
          MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::qubit<StateInteger, BitInteger> const qubit,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, Allocator> const& permutation)
        {
          return ::ket::mpi::gate::page::pauli_y(
            mpi_policy, parallel_policy, local_state, qubit, permutation);
        }
      } // namespace page
    } // namespace gate
  } // namespace mpi
} // namespace ket


#endif // KET_MPI_GATE_PAGE_PAULI_Y_HPP
