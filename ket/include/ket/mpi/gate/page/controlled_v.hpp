#ifndef KET_MPI_GATE_PAGE_CONTROLLED_V_HPP
# define KET_MPI_GATE_PAGE_CONTROLLED_V_HPP

# include <boost/config.hpp>

# include <cassert>

# include <ket/qubit.hpp>
# include <ket/control.hpp>
# include <ket/utility/integer_exp2.hpp>
# include <ket/mpi/qubit_permutation.hpp>
# include <ket/mpi/page/is_on_page.hpp>
# include <ket/mpi/gate/page/detail/two_page_qubits_gate.hpp>
# include <ket/mpi/gate/page/detail/one_page_qubit_gate.hpp>


namespace ket
{
  namespace mpi
  {
    namespace gate
    {
      namespace page
      {
        // tcp: both of target qubit and control qubit are on page
        namespace controlled_v_detail
        {
# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
          template <typename Complex>
          struct controlled_v_coeff_tcp
          {
            Complex one_plus_phase_coefficient_;
            Complex one_minus_phase_coefficient_;

            controlled_v_coeff_tcp(
              Complex const& one_plus_phase_coefficient,
              Complex const& one_minus_phase_coefficient) noexcept
              : one_plus_phase_coefficient_{one_plus_phase_coefficient},
                one_minus_phase_coefficient_{one_minus_phase_coefficient}
            { }

            template <typename Iterator, typename StateInteger>
            void operator()(
              Iterator const, Iterator const,
              Iterator const first_10, Iterator const first_11,
              StateInteger const index, int const) const
            {
              auto const iter_10 = first_10 + index;
              auto const iter_11 = first_11 + index;
              auto const value_10 = *iter_10;

              using real_type = typename ::ket::utility::meta::real_of<Complex>::type;
              using boost::math::constants::half;
              *iter_10
                = half<real_type>()
                  * (one_plus_phase_coefficient_ * value_10
                     + one_minus_phase_coefficient_ * *iter_11);
              *iter_11
                = half<real_type>()
                  * (one_minus_phase_coefficient_ * value_10
                     + one_plus_phase_coefficient_ * *iter_11);
            }
          }; // struct controlled_v_coeff_tcp<Complex>

          template <typename Complex>
          inline ::ket::mpi::gate::page::controlled_v_detail::controlled_v_coeff_tcp<Complex>
          make_controlled_v_coeff_tcp(
            Complex const& one_plus_phase_coefficient,
            Complex const& one_minus_phase_coefficient)
          { return {one_plus_phase_coefficient, one_minus_phase_coefficient}; }
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        } // namespace controlled_v_detail

        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename Complex,
          typename StateInteger, typename BitInteger, typename PermutationAllocator>
        inline RandomAccessRange& controlled_v_coeff_tcp(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          Complex const& phase_coefficient,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, PermutationAllocator> const& permutation)
        {
          using real_type = typename ::ket::utility::meta::real_of<Complex>::type;
          auto const one_plus_phase_coefficient = real_type{1} + phase_coefficient;
          auto const one_minus_phase_coefficient = real_type{1} - phase_coefficient;

# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::two_page_qubits_gate<0u>(
            parallel_policy, local_state, target_qubit, control_qubit, permutation,
            [one_plus_phase_coefficient, one_minus_phase_coefficient](
              auto const, auto const, auto const first_10, auto const first_11,
              StateInteger const index, int const)
            {
              auto const iter_10 = first_10 + index;
              auto const iter_11 = first_11 + index;
              auto const value_10 = *iter_10;

              using boost::math::constants::half;
              *iter_10
                = half<real_type>()
                  * (one_plus_phase_coefficient * value_10
                     + one_minus_phase_coefficient * *iter_11);
              *iter_11
                = half<real_type>()
                  * (one_minus_phase_coefficient * value_10
                     + one_plus_phase_coefficient * *iter_11);
            });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::two_page_qubits_gate<0u>(
            parallel_policy, local_state, target_qubit, control_qubit, permutation,
            ::ket::mpi::gate::page::controlled_v_detail::make_controlled_v_coeff_tcp(
              one_plus_phase_coefficient, one_minus_phase_coefficient));
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        }

        // tp: only target qubit is on page
        namespace controlled_v_detail
        {
# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
          template <typename Complex, typename StateInteger>
          struct controlled_v_coeff_tp
          {
            Complex one_plus_phase_coefficient_;
            Complex one_minus_phase_coefficient_;
            StateInteger control_qubit_mask_;
            StateInteger nonpage_lower_bits_mask_;
            StateInteger nonpage_upper_bits_mask_;

            controlled_v_coeff_tp(
              Complex const& one_plus_phase_coefficient,
              Complex const& one_minus_phase_coefficient,
              StateInteger const control_qubit_mask,
              StateInteger const nonpage_lower_bits_mask,
              StateInteger const nonpage_upper_bits_mask) noexcept
              : one_plus_phase_coefficient_{one_plus_phase_coefficient},
                one_minus_phase_coefficient_{one_minus_phase_coefficient},
                control_qubit_mask_{control_qubit_mask},
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
              auto const control_on_value = *control_on_iter;

              using real_type = typename ::ket::utility::meta::real_of<Complex>::type;
              using boost::math::constants::half;
              *control_on_iter
                = half<real_type>()
                  * (one_plus_phase_coefficient_ * control_on_value
                     + one_minus_phase_coefficient_ * *target_control_on_iter);
              *target_control_on_iter
                = half<real_type>()
                  * (one_minus_phase_coefficient_ * control_on_value
                     + one_plus_phase_coefficient_ * *target_control_on_iter);
            }
          }; // struct controlled_v_coeff_tp<Complex, StateInteger>

          template <typename Complex, typename StateInteger>
          inline ::ket::mpi::gate::page::controlled_v_detail::controlled_v_coeff_tp<Complex, StateInteger>
          make_controlled_v_coeff_tp(
            Complex const& one_plus_phase_coefficient,
            Complex const& one_minus_phase_coefficient,
            StateInteger const control_qubit_mask,
            StateInteger const nonpage_lower_bits_mask,
            StateInteger const nonpage_upper_bits_mask)
          {
            return {
              one_plus_phase_coefficient, one_minus_phase_coefficient,
              control_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask};
          }
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        } // namespace controlled_v_detail

        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename Complex,
          typename StateInteger, typename BitInteger, typename PermutationAllocator>
        inline RandomAccessRange& controlled_v_coeff_tp(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          Complex const& phase_coefficient,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, PermutationAllocator> const& permutation)
        {
          assert(not ::ket::mpi::page::is_on_page(control_qubit.qubit(), local_state, permutation));

          using real_type = typename ::ket::utility::meta::real_of<Complex>::type;
          auto const one_plus_phase_coefficient = real_type{1} + phase_coefficient;
          auto const one_minus_phase_coefficient = real_type{1} - phase_coefficient;

          auto const control_qubit_mask
            = ::ket::utility::integer_exp2<StateInteger>(permutation[control_qubit.qubit()]);
          auto const nonpage_lower_bits_mask = control_qubit_mask - StateInteger{1u};
          auto const nonpage_upper_bits_mask = compl nonpage_lower_bits_mask;

# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<1u>(
            parallel_policy, local_state, target_qubit, permutation,
            [one_plus_phase_coefficient, one_minus_phase_coefficient,
             control_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask](
              auto const zero_first, auto const one_first, StateInteger const index_wo_nonpage_qubit, int const)
            {
              auto const zero_index
                = ((index_wo_nonpage_qubit bitand nonpage_upper_bits_mask) << 1u)
                  bitor (index_wo_nonpage_qubit bitand nonpage_lower_bits_mask);
              auto const one_index = zero_index bitor control_qubit_mask;

              auto const control_on_iter = zero_first + one_index;
              auto const target_control_on_iter = one_first + one_index;
              auto const control_on_value = *control_on_iter;

              using boost::math::constants::half;
              *control_on_iter
                = half<real_type>()
                  * (one_plus_phase_coefficient * control_on_value
                     + one_minus_phase_coefficient * *target_control_on_iter);
              *target_control_on_iter
                = half<real_type>()
                  * (one_minus_phase_coefficient * control_on_value
                     + one_plus_phase_coefficient * *target_control_on_iter);
            });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<1u>(
            parallel_policy, local_state, target_qubit, permutation,
            ::ket::mpi::gate::page::controlled_v_detail::make_controlled_v_coeff_tp(
              one_plus_phase_coefficient, one_minus_phase_coefficient,
              control_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask));
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        }

        // cp: only control qubit is on page
        namespace controlled_v_detail
        {
# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
          template <typename Complex, typename StateInteger>
          struct controlled_v_coeff_cp
          {
            Complex one_plus_phase_coefficient_;
            Complex one_minus_phase_coefficient_;
            StateInteger target_qubit_mask_;
            StateInteger nonpage_lower_bits_mask_;
            StateInteger nonpage_upper_bits_mask_;

            controlled_v_coeff_cp(
              Complex const& one_plus_phase_coefficient,
              Complex const& one_minus_phase_coefficient,
              StateInteger const target_qubit_mask,
              StateInteger const nonpage_lower_bits_mask,
              StateInteger const nonpage_upper_bits_mask) noexcept
              : one_plus_phase_coefficient_{one_plus_phase_coefficient},
                one_minus_phase_coefficient_{one_minus_phase_coefficient},
                target_qubit_mask_{target_qubit_mask},
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
              auto const control_on_value = *control_on_iter;

              using real_type = typename ::ket::utility::meta::real_of<Complex>::type;
              using boost::math::constants::half;
              *control_on_iter
                = half<real_type>()
                  * (one_plus_phase_coefficient_ * control_on_value
                     + one_minus_phase_coefficient_ * *target_control_on_iter);
              *target_control_on_iter
                = half<real_type>()
                  * (one_minus_phase_coefficient_ * control_on_value
                     + one_plus_phase_coefficient_ * *target_control_on_iter);
            }
          }; // struct controlled_v_coeff_cp<Complex, StateInteger>

          template <typename Complex, typename StateInteger>
          inline ::ket::mpi::gate::page::controlled_v_detail::controlled_v_coeff_cp<Complex, StateInteger>
          make_controlled_v_coeff_cp(
            Complex const& one_plus_phase_coefficient,
            Complex const& one_minus_phase_coefficient,
            StateInteger const target_qubit_mask,
            StateInteger const nonpage_lower_bits_mask,
            StateInteger const nonpage_upper_bits_mask)
          {
            return {
              one_plus_phase_coefficient, one_minus_phase_coefficient,
              target_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask};
          }
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        } // namespace controlled_v_detail

        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename Complex,
          typename StateInteger, typename BitInteger, typename PermutationAllocator>
        inline RandomAccessRange& controlled_v_coeff_cp(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          Complex const& phase_coefficient,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, PermutationAllocator> const& permutation)
        {
          assert(not ::ket::mpi::page::is_on_page(target_qubit, local_state, permutation));

          using real_type = typename ::ket::utility::meta::real_of<Complex>::type;
          auto const one_plus_phase_coefficient = real_type{1} + phase_coefficient;
          auto const one_minus_phase_coefficient = real_type{1} - phase_coefficient;

          auto const target_qubit_mask
            = ::ket::utility::integer_exp2<StateInteger>(permutation[target_qubit]);
          auto const nonpage_lower_bits_mask = target_qubit_mask - StateInteger{1u};
          auto const nonpage_upper_bits_mask = compl nonpage_lower_bits_mask;

# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<1u>(
            parallel_policy, local_state, control_qubit.qubit(), permutation,
            [one_plus_phase_coefficient, one_minus_phase_coefficient,
             target_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask](
              auto const, auto const one_first, StateInteger const index_wo_nonpage_qubit, int const)
            {
              auto const zero_index
                = ((index_wo_nonpage_qubit bitand nonpage_upper_bits_mask) << 1u)
                  bitor (index_wo_nonpage_qubit bitand nonpage_lower_bits_mask);
              auto const one_index = zero_index bitor target_qubit_mask;

              auto const control_on_iter = one_first + zero_index;
              auto const target_control_on_iter = one_first + one_index;
              auto const control_on_value = *control_on_iter;

              using boost::math::constants::half;
              *control_on_iter
                = half<real_type>()
                  * (one_plus_phase_coefficient * control_on_value
                     + one_minus_phase_coefficient * *target_control_on_iter);
              *target_control_on_iter
                = half<real_type>()
                  * (one_minus_phase_coefficient * control_on_value
                     + one_plus_phase_coefficient * *target_control_on_iter);
            });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<1u>(
            parallel_policy, local_state, control_qubit.qubit(), permutation,
            ::ket::mpi::gate::page::controlled_v_detail::make_controlled_v_coeff_cp(
              one_plus_phase_coefficient, one_minus_phase_coefficient,
              target_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask));
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        }
      } // namespace page
    } // namespace gate
  } // namespace mpi
} // namespace ket


#endif // KET_MPI_GATE_PAGE_CONTROLLED_V_HPP
