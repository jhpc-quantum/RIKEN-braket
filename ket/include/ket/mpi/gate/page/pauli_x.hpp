#ifndef KET_MPI_GATE_PAGE_PAULI_X_HPP
# define KET_MPI_GATE_PAGE_PAULI_X_HPP

# include <boost/config.hpp>

# include <cassert>
# include <algorithm>

# include <ket/qubit.hpp>
# include <ket/utility/integer_exp2.hpp>
# include <ket/mpi/permutated.hpp>
# include <ket/mpi/page/is_on_page.hpp>
# include <ket/mpi/gate/page/detail/one_page_qubit_gate.hpp>
# include <ket/mpi/gate/page/detail/two_page_qubits_gate.hpp>


namespace ket
{
  namespace mpi
  {
    namespace gate
    {
      namespace page
      {
        // 1_p: the qubit of X is on page
        // X_i (NOT_i)
        // X_1 (a_0 |0> + a_1 |1>) = a_1 |0> + a_0 |1>
        namespace pauli_x_detail
        {
# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
          struct pauli_x1
          {
            template <typename Iterator, typename StateInteger>
            void operator()(Iterator const zero_first, Iterator const one_first, StateInteger const index, int const) const
            { std::iter_swap(zero_first + index, one_first + index); }
          }; // struct pauli_x1
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        } // namespace pauli_x_detail

        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger>
        inline RandomAccessRange& pauli_x1(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_qubit)
        {
# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<0u>(
            parallel_policy, local_state, permutated_qubit,
            [](auto const zero_first, auto const one_first, StateInteger const index, int const)
            { std::iter_swap(zero_first + index, one_first + index); });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<0u>(
            parallel_policy, local_state, permutated_qubit,
            ::ket::mpi::gate::page::pauli_x_detail::pauli_x1{});
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        }

        // 2_2p: both of qubits of XX are on page
        // XX_{ij} = X_i X_j
        // XX_{1,2} (a_{00} |00> + a_{01} |01> + a_{10} |10> + a{11} |11>)
        //   = a_{11} |00> + a_{10} |01> + a_{01} |10> + a_{00} |11>
        namespace pauli_x_detail
        {
# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
          struct pauli_x2_2p
          {
            template <typename Iterator, typename StateInteger>
            void operator()(
              Iterator const first_00, Iterator const first_01,
              Iterator const first_10, Iterator const first_11,
              StateInteger const index, int const) const
            {
              std::iter_swap(first_00 + index, first_11 + index);
              std::iter_swap(first_01 + index, first_10 + index);
            }
          }; // struct pauli_x2_2p
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        } // namespace pauli_x_detail

        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger>
        inline RandomAccessRange& pauli_x2_2p(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_qubit1,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_qubit2)
        {
# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::two_page_qubits_gate<0u>(
            parallel_policy, local_state, permutated_qubit1, permutated_qubit2,
            [](auto const first_00, auto const first_01, auto const first_10, auto const first_11, StateInteger const index, int const)
            {
              std::iter_swap(first_00 + index, first_11 + index);
              std::iter_swap(first_01 + index, first_10 + index);
            });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::two_page_qubits_gate<0u>(
            parallel_policy, local_state, permutated_qubit1, permutated_qubit2,
            ::ket::mpi::gate::page::pauli_x_detail::pauli_x2_2p{});
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        }

        // 2_p: only one qubit of XX is on page
        // XX_{ij} = X_i X_j
        // XX_{1,2} (a_{00} |00> + a_{01} |01> + a_{10} |10> + a{11} |11>)
        //   = a_{11} |00> + a_{10} |01> + a_{01} |10> + a_{00} |11>
        namespace pauli_x_detail
        {
# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
          struct pauli_x2_p
          {
            template <typename Iterator, typename StateInteger>
            void operator()(
              Iterator const first_00, Iterator const first_01,
              Iterator const first_10, Iterator const first_11,
              StateInteger const index, int const) const
            {
              auto const zero_index
                = ((index_wo_nonpage_qubit bitand nonpage_upper_bits_mask) << 1u)
                  bitor (index_wo_nonpage_qubit bitand nonpage_lower_bits_mask);
              auto const one_index = zero_index bitor nonpage_permutated_qubit_mask;

              auto const iter_00 = zero_first + zero_index;
              auto const iter_01_or_10 = zero_first + one_index;
              auto const iter_10_or_01 = one_first + zero_index;
              auto const iter_11 = one_first + one_index;

              std::iter_swap(iter_00, iter_11);
              std::iter_swap(iter_01_or_10, iter_10_or_10);
            }
          }; // struct pauli_x2_p
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        } // namespace pauli_x_detail

        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger>
        inline RandomAccessRange& pauli_x2_p(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const page_permutated_qubit,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const nonpage_permutated_qubit)
        {
          assert(::ket::mpi::page::is_on_page(page_permutated_qubit, local_state));
          assert(not ::ket::mpi::page::is_on_page(nonpage_permutated_qubit, local_state));

          auto const nonpage_permutated_qubit_mask
            = ::ket::utility::integer_exp2<StateInteger>(nonpage_permutated_qubit);
          auto const nonpage_lower_bits_mask = nonpage_permutated_qubit_mask - StateInteger{1u};
          auto const nonpage_upper_bits_mask = compl nonpage_lower_bits_mask;

# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<1u>(
            parallel_policy, local_state, page_permutated_qubit,
            [nonpage_permutated_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask](
              auto const zero_first, auto const one_first, StateInteger const index_wo_nonpage_qubit, int const)
            {
              auto const zero_index
                = ((index_wo_nonpage_qubit bitand nonpage_upper_bits_mask) << 1u)
                  bitor (index_wo_nonpage_qubit bitand nonpage_lower_bits_mask);
              auto const one_index = zero_index bitor nonpage_permutated_qubit_mask;

              auto const iter_00 = zero_first + zero_index;
              auto const iter_01_or_10 = zero_first + one_index;
              auto const iter_10_or_01 = one_first + zero_index;
              auto const iter_11 = one_first + one_index;

              std::iter_swap(iter_00, iter_11);
              std::iter_swap(iter_01_or_10, iter_10_or_01);
            });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<1u>(
            parallel_policy, local_state, page_permutated_qubit,
            ::ket::mpi::gate::page::pauli_x_detail::pauli_x2_p{});
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        }
      } // namespace page
    } // namespace gate
  } // namespace mpi
} // namespace ket


#endif // KET_MPI_GATE_PAGE_PAULI_X_HPP
