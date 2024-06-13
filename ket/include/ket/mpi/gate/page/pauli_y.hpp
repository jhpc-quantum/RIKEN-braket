#ifndef KET_MPI_GATE_PAGE_PAULI_Y_HPP
# define KET_MPI_GATE_PAGE_PAULI_Y_HPP

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
# include <ket/utility/imaginary_unit.hpp>
# include <ket/utility/meta/real_of.hpp>
# include <ket/mpi/permutated.hpp>
# include <ket/mpi/page/is_on_page.hpp>
# include <ket/mpi/gate/page/detail/pauli_cy_tcp.hpp>
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
        // 1_p: the qubit of Y is on page
        // Y_i
        // Y_1 (a_0 |0> + a_1 |1>) = -i a_1 |0> + i a_0 |1>
        namespace pauli_y_detail
        {
# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
          struct pauli_y1
          {
            template <typename Iterator, typename StateInteger>
            void operator()(Iterator const zero_first, Iterator const one_first, StateInteger const index, int const) const
            {
              auto const zero_iter = zero_first + index;
              auto const one_iter = one_first + index;

              std::iter_swap(zero_iter, one_iter);

              using complex_type = typename std::iterator_traits<Iterator>::value_type;
              *zero_iter *= -::ket::utility::imaginary_unit<complex_type>();
              *one_iter *= ::ket::utility::imaginary_unit<complex_type>();
            }
          }; // struct pauli_y1
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        } // namespace pauli_y_detail

        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger>
        inline RandomAccessRange& pauli_y1(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_qubit)
        {
# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<0u>(
            parallel_policy, local_state, permutated_qubit,
            [](auto const zero_first, auto const one_first, StateInteger const index, int const)
            {
              auto const zero_iter = zero_first + index;
              auto const one_iter = one_first + index;

              std::iter_swap(zero_iter, one_iter);

              using complex_type = typename boost::range_value<RandomAccessRange>::type;
              *zero_iter *= -::ket::utility::imaginary_unit<complex_type>();
              *one_iter *= ::ket::utility::imaginary_unit<complex_type>();
            });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<0u>(
            parallel_policy, local_state, permutated_qubit,
            ::ket::mpi::gate::page::pauli_y_detail::pauli_y1{});
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        }

        // 2_2p: both of qubits of YY are on page
        // YY_{ij} = Y_i Y_j
        // YY_{1,2} (a_{00} |00> + a_{01} |01> + a_{10} |10> + a{11} |11>)
        //   = -a_{11} |00> + a_{10} |01> + a_{01} |10> - a_{00} |11>
        namespace pauli_y_detail
        {
# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
          struct pauli_y2_2p
          {
            template <typename Iterator, typename StateInteger>
            void operator()(
              Iterator const first_00, Iterator const first_01,
              Iterator const first_10, Iterator const first_11,
              StateInteger const index, int const) const
            {
              auto iter_00 = first_00 + index;
              auto iter_01 = first_01 + index;
              auto iter_10 = first_10 + index;
              auto iter_11 = first_11 + index;

              std::iter_swap(iter_00, iter_11);
              std::iter_swap(iter_01, iter_10);

              using complex_type = typename std::iterator_traits<Iterator>::value_type;
              using real_type = typename ::ket::utility::meta::real_of<complex_type>::type;
              *iter_00 *= real_type{-1.0};
              *iter_11 *= real_type{-1.0};
            }
          }; // struct pauli_y2_2p
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        } // namespace pauli_y_detail

        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger>
        inline RandomAccessRange& pauli_y2_2p(
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
              auto iter_00 = first_00 + index;
              auto iter_01 = first_01 + index;
              auto iter_10 = first_10 + index;
              auto iter_11 = first_11 + index;

              std::iter_swap(iter_00, iter_11);
              std::iter_swap(iter_01, iter_10);

              using complex_type = typename boost::range_value<RandomAccessRange>::type;
              using real_type = typename ::ket::utility::meta::real_of<complex_type>::type;
              *iter_00 *= real_type{-1.0};
              *iter_11 *= real_type{-1.0};
            });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::two_page_qubits_gate<0u>(
            parallel_policy, local_state, permutated_qubit1, permutated_qubit2,
            ::ket::mpi::gate::page::pauli_y_detail::pauli_y2_2p{});
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        }

        // 2_p: only one qubit of YY is on page
        // YY_{ij} = Y_i Y_j
        // YY_{1,2} (a_{00} |00> + a_{01} |01> + a_{10} |10> + a{11} |11>)
        //   = -a_{11} |00> + a_{10} |01> + a_{01} |10> - a_{00} |11>
        namespace pauli_y_detail
        {
# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
          template <typename StateInteger>
          struct pauli_y2_p
          {
            StateInteger nonpage_permutated_qubit_mask_;
            StateInteger nonpage_lower_bits_mask_;
            StateInteger nonpage_upper_bits_mask_;

            pauli_y2_p(
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

              auto const iter_00 = zero_first + zero_index;
              auto const iter_01_or_10 = zero_first + one_index;
              auto const iter_10_or_01 = one_first + zero_index;
              auto const iter_11 = one_first + one_index;

              std::iter_swap(iter_00, iter_11);
              std::iter_swap(iter_01_or_10, iter_10_or_01);

              using complex_type = typename std::iterator_traits<Iterator>::value_type;
              using real_type = typename ::ket::utility::meta::real_of<complex_type>::type;
              *iter_00 *= real_type{-1.0};
              *iter_11 *= real_type{-1.0};
            }
          }; // struct pauli_y2_p<StateInteger>
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        } // namespace pauli_y_detail

        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger>
        inline RandomAccessRange& pauli_y2_p(
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

              using complex_type = typename boost::range_value<RandomAccessRange>::type;
              using real_type = typename ::ket::utility::meta::real_of<complex_type>::type;
              *iter_00 *= real_type{-1.0};
              *iter_11 *= real_type{-1.0};
            });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<1u>(
            parallel_policy, local_state, page_permutated_qubit,
            ::ket::mpi::gate::page::pauli_y_detail::pauli_y2_p<StateInteger>{
              nonpage_permutated_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask});
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        }

        // cy_tcp: both of target and control qubits of CY are on page
        // CY_{tc}, CY1_{tc}, C1Y_{tc}, or C1Y1_{tc}
        // CY_{1,2} (a_{00} |00> + a_{01} |01> + a_{10} |10> + a{11} |11>)
        //   = a_{00} |00> + a_{01} |01> - i a_{11} |10> + i a_{10} |11>
        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger>
        inline RandomAccessRange& pauli_cy_tcp(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_target_qubit,
          ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const permutated_control_qubit)
        {
          return ::ket::mpi::gate::page::detail::pauli_cy_tcp(
            parallel_policy, local_state, permutated_target_qubit, permutated_control_qubit);
        }

        // cy_tp: only target qubit is on page
        // CY_{tc}, CY1_{tc}, C1Y_{tc}, or C1Y1_{tc}
        // CY_{1,2} (a_{00} |00> + a_{01} |01> + a_{10} |10> + a{11} |11>)
        //   = a_{00} |00> + a_{01} |01> - i a_{11} |10> + i a_{10} |11>
        namespace pauli_y_detail
        {
# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
          template <typename StateInteger>
          struct pauli_cy_tp
          {
            StateInteger control_qubit_mask_;
            StateInteger nonpage_lower_bits_mask_;
            StateInteger nonpage_upper_bits_mask_;

            pauli_cy_tp(
              StateInteger const control_qubit_mask,
              StateInteger const nonpage_lower_bits_mask,
              StateInteger const nonpage_upper_bits_mask) noexcept
              : control_qubit_mask_{control_qubit_mask},
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
              auto const one_index = zero_index bitor control_qubit_mask_;
              auto const control_on_iter = zero_first + one_index;
              auto const target_control_on_iter = one_first + one_index;

              std::iter_swap(control_on_iter, target_control_on_iter);

              using complex_type = typename std::iterator_traits<Iterator>::value_type;
              *control_on_iter *= -::ket::utility::imaginary_unit<complex_type>();
              *target_control_on_iter *= ::ket::utility::imaginary_unit<complex_type>();
            }
          }; // struct pauli_cy_tp<StateInteger>

          template <typename StateInteger>
          inline ::ket::mpi::gate::page::pauli_y_detail::pauli_cy_tp<StateInteger>
          make_pauli_cy_tp(
            StateInteger const control_qubit_mask,
            StateInteger const nonpage_lower_bits_mask,
            StateInteger const nonpage_upper_bits_mask)
          { return {control_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask}; }
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        } // namespace pauli_y_detail

        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger>
        inline RandomAccessRange& pauli_cy_tp(
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
              auto const zero_first, auto const one_first, StateInteger const index_wo_nonpage_qubit, int const)
            {
              auto const zero_index
                = ((index_wo_nonpage_qubit bitand nonpage_upper_bits_mask) << 1u)
                  bitor (index_wo_nonpage_qubit bitand nonpage_lower_bits_mask);
              auto const one_index = zero_index bitor control_qubit_mask;
              auto const control_on_iter = zero_first + one_index;
              auto const target_control_on_iter = one_first + one_index;

              std::iter_swap(control_on_iter, target_control_on_iter);

              using complex_type = typename boost::range_value<RandomAccessRange>::type;
              *control_on_iter *= -::ket::utility::imaginary_unit<complex_type>();
              *target_control_on_iter *= ::ket::utility::imaginary_unit<complex_type>();
            });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<1u>(
            parallel_policy, local_state, permutated_target_qubit,
            ::ket::mpi::gate::page::pauli_y_detail::make_pauli_cy_tp(
              control_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask));
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        }

        // cy_cp: only control qubit is on page
        // CY_{tc}, CY1_{tc}, C1Y_{tc}, or C1Y1_{tc}
        // CY_{1,2} (a_{00} |00> + a_{01} |01> + a_{10} |10> + a{11} |11>)
        //   = a_{00} |00> + a_{01} |01> - i a_{11} |10> + i a_{10} |11>
        namespace pauli_y_detail
        {
# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
          template <typename StateInteger>
          struct pauli_cy_cp
          {
            StateInteger target_qubit_mask_;
            StateInteger nonpage_lower_bits_mask_;
            StateInteger nonpage_upper_bits_mask_;

            pauli_cy_cp(
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
              auto const control_on_iter = one_first + zero_index;
              auto const target_control_on_iter = one_first + one_index;

              std::iter_swap(control_on_iter, target_control_on_iter);

              using complex_type = typename std::iterator_traits<Iterator>::value_type;
              *control_on_iter *= -::ket::utility::imaginary_unit<complex_type>();
              *target_control_on_iter *= ::ket::utility::imaginary_unit<complex_type>();
            }
          }; // struct pauli_cy_cp<StateInteger>

          template <typename StateInteger>
          inline ::ket::mpi::gate::page::pauli_y_detail::pauli_cy_cp<StateInteger>
          make_pauli_cy_cp(
            StateInteger const target_qubit_mask,
            StateInteger const nonpage_lower_bits_mask,
            StateInteger const nonpage_upper_bits_mask)
          { return {target_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask}; }
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        } // namespace pauli_y_detail

        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger>
        inline RandomAccessRange& pauli_cy_cp(
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
              auto const control_on_iter = one_first + zero_index;
              auto const target_control_on_iter = one_first + one_index;

              std::iter_swap(control_on_iter, target_control_on_iter);

              using complex_type = typename boost::range_value<RandomAccessRange>::type;
              *control_on_iter *= -::ket::utility::imaginary_unit<complex_type>();
              *target_control_on_iter *= ::ket::utility::imaginary_unit<complex_type>();
            });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<1u>(
            parallel_policy, local_state, permutated_control_qubit,
            ::ket::mpi::gate::page::pauli_y_detail::make_pauli_cy_cp(
              target_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask));
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        }
      } // namespace page
    } // namespace gate
  } // namespace mpi
} // namespace ket


#endif // KET_MPI_GATE_PAGE_PAULI_Y_HPP
