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
# include <ket/control.hpp>
# include <ket/utility/integer_exp2.hpp>
# include <ket/utility/meta/real_of.hpp>
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
        namespace pauli_z_detail
        {
# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
          template <typename StateInteger>
          struct pauli_z2_p
          {
            StateInteger nonpage_permutated_qubit_mask_;
            StateInteger nonpage_lower_bits_mask_;
            StateInteger nonpage_upper_bits_mask_;

            pauli_z2_p(
              StateInteger const nonpage_permutated_qubit_mask,
              StateInteger const nonpage_lower_bits_mask,
              StateInteger const nonpage_upper_bits_mask)
              : nonpage_permutated_qubit_mask_{nonpage_permutated_qubit_mask},
                nonpage_lower_bits_mask_{nonpage_lower_bits_mask},
                nonpage_upper_bits_mask_{nonpage_upper_bits_mask}
            { }

            template <typename Iterator>
            void operator()(
              Iterator const zero_first, Iterator const one_first,
              StateInteger const index_wo_nonpage_qubit, int const) const
            {
              auto const zero_index
                = ((index_wo_nonpage_qubit bitand nonpage_upper_bits_mask_) << 1u)
                  bitor (index_wo_nonpage_qubit bitand nonpage_lower_bits_mask_);
              auto const one_index = zero_index bitor nonpage_permutated_qubit_mask_;

              auto const iter_01_or_10 = zero_first + one_index;
              auto const iter_10_or_01 = one_first + zero_index;

              using complex_type = typename std::iterator_traits<Iterator>::value_type;
              using real_type = typename ::ket::utility::meta::real_of<complex_type>::type;
              *iter_01_or_10 *= real_type{-1.0};
              *iter_10_or_01 *= real_type{-1.0};
            }
          }; // struct pauli_z2_p<StateInteger>
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        } // namespace pauli_z_detail

        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger>
        inline RandomAccessRange& pauli_z2_p(
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

              auto const iter_01_or_10 = zero_first + one_index;
              auto const iter_10_or_01 = one_first + zero_index;

              using complex_type = typename boost::range_value<RandomAccessRange>::type;
              using real_type = typename ::ket::utility::meta::real_of<complex_type>::type;
              *iter_01_or_10 *= real_type{-1.0};
              *iter_10_or_01 *= real_type{-1.0};
            });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<1u>(
            parallel_policy, local_state, page_permutated_qubit,
            ::ket::mpi::gate::page::pauli_z_detail::pauli_z2_p<StateInteger>{
              nonpage_permutated_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask});
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
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
        namespace pauli_z_detail
        {
# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
          template <typename StateInteger>
          struct pauli_cz_tp
          {
            StateInteger control_qubit_mask_;
            StateInteger nonpage_lower_bits_mask_;
            StateInteger nonpage_upper_bits_mask_;

            pauli_cz_tp(
              StateInteger const control_qubit_mask,
              StateInteger const nonpage_lower_bits_mask,
              StateInteger const nonpage_upper_bits_mask) noexcept
              : control_qubit_mask_{control_qubit_mask},
                nonpage_lower_bits_mask_{nonpage_lower_bits_mask},
                nonpage_upper_bits_mask_{nonpage_upper_bits_mask}
            { }

            template <typename Iterator>
            void operator()(
              Iterator const, Iterator const one_first,
              StateInteger const index_wo_nonpage_qubit, int const) const
            {
              auto const zero_index
                = ((index_wo_nonpage_qubit bitand nonpage_upper_bits_mask_) << 1u)
                  bitor (index_wo_nonpage_qubit bitand nonpage_lower_bits_mask_);
              auto const one_index = zero_index bitor control_qubit_mask_;

              using complex_type = typename std::iterator_traits<Iterator>::value_type;
              using real_type = typename ::ket::utility::meta::real_of<complex_type>::type;
              *(one_first + one_index) *= static_cast<real_type>(-1);
            }
          }; // struct pauli_cz_tp<StateInteger>

          template <typename StateInteger>
          inline ::ket::mpi::gate::page::pauli_z_detail::pauli_cz_tp<StateInteger>
          make_pauli_cz_tp(
            StateInteger const control_qubit_mask,
            StateInteger const nonpage_lower_bits_mask,
            StateInteger const nonpage_upper_bits_mask)
          { return {control_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask}; }
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        } // namespace pauli_z_detail

        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger>
        inline RandomAccessRange& pauli_cz_tp(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_target_qubit,
          ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const permutated_control_qubit)
        {
          assert(not ::ket::mpi::page::is_on_page(permutated_control_qubit, local_state));
          auto const control_qubit_mask
            = ::ket::utility::integer_exp2<StateInteger>(permutated_control_qubit);
          auto const nonpage_lower_bits_mask = control_qubit_mask - StateInteger{1u};
          auto const nonpage_upper_bits_mask = compl nonpage_lower_bits_mask;

# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<1u>(
            parallel_policy, local_state, permutated_target_qubit,
            [control_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask](
              auto const, auto const one_first, StateInteger const index_wo_nonpage_qubit, int const)
            {
              auto const zero_index
                = ((index_wo_nonpage_qubit bitand nonpage_upper_bits_mask) << 1u)
                  bitor (index_wo_nonpage_qubit bitand nonpage_lower_bits_mask);
              auto const one_index = zero_index bitor control_qubit_mask;

              using complex_type = typename boost::range_value<RandomAccessRange>::type;
              using real_type = typename ::ket::utility::meta::real_of<complex_type>::type;
              *(one_first + one_index) *= static_cast<real_type>(-1);
            });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<1u>(
            parallel_policy, local_state, permutated_target_qubit,
            ::ket::mpi::gate::page::pauli_z_detail::make_pauli_cz_tp(
              control_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask));
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        }

        // cz_cp: only control qubit is on page
        // CZ_{tc}, CZ1_{tc}, C1Z_{tc}, or C1Z1_{tc}
        // CZ_{1,2} (a_{00} |00> + a_{01} |01> + a_{10} |10> + a{11} |11>)
        //   = a_{00} |00> + a_{01} |01> + a_{10} |10> - a{11} |11>
        namespace pauli_z_detail
        {
# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
          template <typename StateInteger>
          struct pauli_cz_cp
          {
            StateInteger target_qubit_mask_;
            StateInteger nonpage_lower_bits_mask_;
            StateInteger nonpage_upper_bits_mask_;

            pauli_cz_cp(
              StateInteger const target_qubit_mask,
              StateInteger const nonpage_lower_bits_mask,
              StateInteger const nonpage_upper_bits_mask) noexcept
              : target_qubit_mask_{target_qubit_mask},
                nonpage_lower_bits_mask_{nonpage_lower_bits_mask},
                nonpage_upper_bits_mask_{nonpage_upper_bits_mask}
            { }

            template <typename Iterator>
            void operator()(
              Iterator const, Iterator const one_first,
              StateInteger const index_wo_nonpage_qubit, int const) const
            {
              auto const zero_index
                = ((index_wo_nonpage_qubit bitand nonpage_upper_bits_mask_) << 1u)
                  bitor (index_wo_nonpage_qubit bitand nonpage_lower_bits_mask_);
              auto const one_index = zero_index bitor target_qubit_mask_;

              using complex_type = typename std::iterator_traits<Iterator>::value_type;
              using real_type = typename ::ket::utility::meta::real_of<complex_type>::type;
              *(one_first + one_index) *= static_cast<real_type>(-1);
            }
          }; // struct pauli_cz_cp<StateInteger>

          template <typename StateInteger>
          inline ::ket::mpi::gate::page::pauli_y_detail::pauli_cy_cp<StateInteger>
          make_pauli_cz_cp(
            StateInteger const target_qubit_mask,
            StateInteger const nonpage_lower_bits_mask,
            StateInteger const nonpage_upper_bits_mask)
          { return {target_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask}; }
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        } // namespace pauli_z_detail

        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger>
        inline RandomAccessRange& pauli_cz_cp(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_target_qubit,
          ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const permutated_control_qubit)
        {
          assert(not ::ket::mpi::page::is_on_page(permutated_target_qubit, local_state));

          auto const target_qubit_mask
            = ::ket::utility::integer_exp2<StateInteger>(permutated_target_qubit);
          auto const nonpage_lower_bits_mask = target_qubit_mask - StateInteger{1u};
          auto const nonpage_upper_bits_mask = compl nonpage_lower_bits_mask;

# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<1u>(
            parallel_policy, local_state, permutated_control_qubit,
            [target_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask](
              auto const, auto const one_first, StateInteger const index_wo_nonpage_qubit, int const)
            {
              auto const zero_index
                = ((index_wo_nonpage_qubit bitand nonpage_upper_bits_mask) << 1u)
                  bitor (index_wo_nonpage_qubit bitand nonpage_lower_bits_mask);
              auto const one_index = zero_index bitor target_qubit_mask;

              using complex_type = typename boost::range_value<RandomAccessRange>::type;
              using real_type = typename ::ket::utility::meta::real_of<complex_type>::type;
              *(one_first + one_index) *= static_cast<real_type>(-1);
            });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<1u>(
            parallel_policy, local_state, permutated_control_qubit,
            ::ket::mpi::gate::page::pauli_z_detail::make_pauli_cz_cp(
              target_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask));
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        }
      } // namespace page
    } // namespace gate
  } // namespace mpi
} // namespace ket


#endif // KET_MPI_GATE_PAGE_PAULI_Z_STANDARD_HPP
