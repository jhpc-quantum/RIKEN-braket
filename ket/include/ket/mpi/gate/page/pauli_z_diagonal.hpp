#ifndef KET_MPI_GATE_PAGE_PAULI_Z_STANDARD_HPP
# define KET_MPI_GATE_PAGE_PAULI_Z_STANDARD_HPP

# include <boost/config.hpp>

# include <cassert>
# include <algorithm>
# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
#  include <iterator>
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS

# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
#  include <boost/range/value_type.hpp>
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS

# include <ket/qubit.hpp>
# include <ket/utility/integer_exp2.hpp>
# include <ket/utility/meta/real_of.hpp>
# include <ket/mpi/permutated.hpp>
# include <ket/mpi/page/is_on_page.hpp>
# include <ket/mpi/gate/page/detail/one_page_qubit_gate.hpp>
# include <ket/mpi/gate/page/detail/two_page_qubits_gate.hpp>
# include <ket/mpi/gate/page/detail/pauli_z2_p_diagonal.hpp>
# include <ket/mpi/gate/page/detail/pauli_cz_tp_diagonal.hpp>
# include <ket/mpi/gate/page/detail/pauli_cz_cp_diagonal.hpp>


namespace ket
{
  namespace mpi
  {
    namespace gate
    {
      namespace page
      {
        // 1_p: the qubit of Z is on page
        // Z_i
        // Z_1 (a_0 |0> + a_1 |1>) = a_0 |0> - a_1 |1>
        namespace pauli_z_detail
        {
# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
          struct pauli_z1
          {
            template <typename Iterator, typename StateInteger>
            void operator()(Iterator const zero_first, Iterator const one_first, StateInteger const index, int const) const
            {
              using complex_type = typename std::iterator_traits<Iterator>::value_type;
              using real_type = typename ::ket::utility::meta::real_of<complex_type>::type;
              *(one_first + index) *= static_cast<real_type>(-1);
            }
          }; // struct pauli_z1
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        } // namespace pauli_z_detail

        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger>
        inline RandomAccessRange& pauli_z1(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_qubit)
        {
# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<0u>(
            parallel_policy, local_state, permutated_qubit,
            [](auto const zero_first, auto const one_first, StateInteger const index, int const)
            {
              using complex_type = typename boost::range_value<RandomAccessRange>::type;
              using real_type = typename ::ket::utility::meta::real_of<complex_type>::type;
              *(one_first + index) *= static_cast<real_type>(-1);
            });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<0u>(
            parallel_policy, local_state, permutated_qubit,
            ::ket::mpi::gate::page::pauli_z_detail::pauli_z1{});
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        }

        // 2_2p: both of qubits of ZZ are on page
        // ZZ_i = Z_i Z_j
        // ZZ_{1,2} (a_{00} |00> + a_{01} |01> + a_{10} |10> + a{11} |11>)
        //   = a_{00} |00> - a_{01} |01> - a_{10} |10> + a{11} |11>
        namespace pauli_z_detail
        {
# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
          struct pauli_z2_2p
          {
            template <typename Iterator, typename StateInteger>
            void operator()(
              Iterator const first_00, Iterator const first_01,
              Iterator const first_10, Iterator const first_11,
              StateInteger const index, int const) const
            {
              using complex_type = typename std::iterator_traits<Iterator>::value_type;
              using real_type = typename ::ket::utility::meta::real_of<complex_type>::type;
              *(first_01 + index) *= real_type{-1.0};
              *(first_10 + index) *= real_type{-1.0};
            }
          }; // struct pauli_z2_2p
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        } // namespace pauli_z_detail

        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger>
        inline RandomAccessRange& pauli_z2_2p(
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
              using complex_type = typename boost::range_value<RandomAccessRange>::type;
              using real_type = typename ::ket::utility::meta::real_of<complex_type>::type;
              *(first_01 + index) *= real_type{-1.0};
              *(first_10 + index) *= real_type{-1.0};
            });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::two_page_qubits_gate<0u>(
            parallel_policy, local_state, permutated_qubit1, permutated_qubit2,
            ::ket::mpi::gate::page::pauli_z_detail::pauli_z2_2p{});
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        }

        // 2_p: only one qubit of ZZ is on page
        // ZZ_i = Z_i Z_j
        // ZZ_{1,2} (a_{00} |00> + a_{01} |01> + a_{10} |10> + a{11} |11>)
        //   = a_{00} |00> - a_{01} |01> - a_{10} |10> + a{11} |11>
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger>
        inline RandomAccessRange& pauli_z2_p(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const page_permutated_qubit,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const nonpage_permutated_qubit,
          yampi::rank const rank)
        {
          return ::ket::mpi::gate::page::detail::pauli_z2_p(
            mpi_policy, parallel_policy, local_state,
            page_permutated_qubit, nonpage_permutated_qubit, rank);
        }

        // cz_tcp: both of target and control qubits of CZ are on page
        // CZ_{tc}, CZ1_{tc}, C1Z_{tc}, or C1Z1_{tc}
        // CZ_{1,2} (a_{00} |00> + a_{01} |01> + a_{10} |10> + a{11} |11>)
        //   = a_{00} |00> + a_{01} |01> + a_{10} |10> - a{11} |11>
        namespace pauli_z_detail
        {
# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
          struct pauli_cz_tcp
          {
            template <typename Iterator, typename StateInteger>
            void operator()(
              Iterator const, Iterator const, Iterator const, Iterator const first_11, StateInteger const index, int const) const
            {
              using complex_type = typename std::iterator_traits<Iterator>::value_type;
              using real_type = typename ::ket::utility::meta::real_of<complex_type>::type;
              *(first_11 + index) *= real_type{-1.0};
            }
          }; // struct pauli_cz_tcp
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        } // namespace pauli_z_detail

        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger>
        inline RandomAccessRange& pauli_cz_tcp(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_target_qubit,
          ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const permutated_control_qubit)
        {
# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::two_page_qubits_gate<0u>(
            parallel_policy, local_state, permutated_target_qubit, permutated_control_qubit,
            [](auto const, auto const, auto const, auto const first_11, StateInteger const index, int const)
            {
              using complex_type = typename boost::range_value<RandomAccessRange>::type;
              using real_type = typename ::ket::utility::meta::real_of<complex_type>::type;
              *(first_11 + index) *= real_type{-1.0};
            });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::two_page_qubits_gate<0u>(
            parallel_policy, local_state, permutated_target_qubit, permutated_control_qubit,
            ::ket::mpi::gate::page::pauli_z_detail::pauli_cz_tcp{});
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        }

        // cz_tp: only target qubit is on page
        // CZ_{tc}, CZ1_{tc}, C1Z_{tc}, or C1Z1_{tc}
        // CZ_{1,2} (a_{00} |00> + a_{01} |01> + a_{10} |10> + a{11} |11>)
        //   = a_{00} |00> + a_{01} |01> + a_{10} |10> - a{11} |11>
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger>
        inline RandomAccessRange& pauli_cz_tp(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_target_qubit,
          ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const permutated_control_qubit,
          yampi::rank const rank)
        {
          return ::ket::mpi::gate::page::detail::pauli_cz_tp(
            mpi_policy, parallel_policy, local_state, permutated_target_qubit, permutated_control_qubit, rank);
        }

        // cz_cp: only control qubit is on page
        // CZ_{tc}, CZ1_{tc}, C1Z_{tc}, or C1Z1_{tc}
        // CZ_{1,2} (a_{00} |00> + a_{01} |01> + a_{10} |10> + a{11} |11>)
        //   = a_{00} |00> + a_{01} |01> + a_{10} |10> - a{11} |11>
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger>
        inline RandomAccessRange& pauli_cz_cp(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_target_qubit,
          ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const permutated_control_qubit,
          yampi::rank const rank)
        {
          return ::ket::mpi::gate::page::detail::pauli_cz_cp(
            mpi_policy, parallel_policy, local_state, permutated_target_qubit, permutated_control_qubit, rank);
        }
      } // namespace page
    } // namespace gate
  } // namespace mpi
} // namespace ket


#endif // KET_MPI_GATE_PAGE_PAULI_Z_STANDARD_HPP
