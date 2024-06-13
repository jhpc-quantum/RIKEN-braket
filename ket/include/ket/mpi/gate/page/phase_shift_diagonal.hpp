#ifndef KET_MPI_GATE_PAGE_PHASE_SHIFT_DIAGONAL_HPP
# define KET_MPI_GATE_PAGE_PHASE_SHIFT_DIAGONAL_HPP

# include <boost/config.hpp>

# include <cassert>
# include <cmath>
# include <type_traits>
# include <memory>

# include <boost/math/constants/constants.hpp>
# include <boost/range/value_type.hpp>

# include <yampi/rank.hpp>

# include <ket/qubit.hpp>
# include <ket/control.hpp>
# include <ket/utility/integer_exp2.hpp>
# include <ket/utility/exp_i.hpp>
# include <ket/utility/meta/real_of.hpp>
# include <ket/mpi/permutated.hpp>
# include <ket/mpi/page/is_on_page.hpp>
# include <ket/mpi/gate/page/detail/one_page_qubit_gate.hpp>
# include <ket/mpi/gate/page/detail/two_page_qubits_gate.hpp>
# include <ket/mpi/gate/page/detail/cphase_shift_coeff_tp_diagonal.hpp>
# include <ket/mpi/gate/page/detail/cphase_shift_coeff_cp_diagonal.hpp>


namespace ket
{
  namespace mpi
  {
    namespace gate
    {
      namespace page
      {
        namespace phase_shift_detail
        {
# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
          template <typename Complex>
          struct phase_shift_coeff
          {
            Complex phase_coefficient_;

            explicit phase_shift_coeff(Complex const& phase_coefficient) noexcept
              : phase_coefficient_{phase_coefficient}
            { }

            template <typename Iterator, typename StateInteger>
            void operator()(Iterator const, Iterator const one_first, StateInteger const index, int const) const
            { *(one_first + index) *= phase_coefficient_; }
          }; // struct phase_shift_coeff<Complex>

          template <typename Complex>
          inline ::ket::mpi::gate::page::phase_shift_detail::phase_shift_coeff<Complex>
          make_phase_shift_coeff(Complex const& phase_coefficient)
          { return ::ket::mpi::gate::page::phase_shift_detail::phase_shift_coeff<Complex>{phase_coefficient}; }
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        } // namespace phase_shift_detail

        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename Complex, typename StateInteger, typename BitInteger>
        inline RandomAccessRange& phase_shift_coeff(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          Complex const& phase_coefficient,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_qubit)
        {
# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<0u>(
            parallel_policy, local_state, permutated_qubit,
            [phase_coefficient](auto const, auto const one_first, StateInteger const index, int const)
            { *(one_first + index) *= phase_coefficient; });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<0u>(
            parallel_policy, local_state, permutated_qubit,
            ::ket::mpi::gate::page::phase_shift_detail::make_phase_shift_coeff(phase_coefficient));
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        }

        // cphase_shift_coeff_tcp: both of target and control qubits of CPhase(coeff) are on page
        // CU1_{tc}(theta) or C1U1_{tc}(theta)
        // CU1_{1,2}(theta) (a_{00} |00> + a_{01} |01> + a_{10} |10> + a{11} |11>)
        //   = a_{00} |00> + a_{01} |01> + a_{10} |10> + e^{i theta} a_{11} |11>
        namespace phase_shift_detail
        {
# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
          template <typename Complex>
          struct cphase_shift_coeff_tcp
          {
            Complex const* phase_coefficient_ptr_;

            cphase_shift_coeff_tcp(Complex const& phase_coefficient) noexcept
              : phase_coefficient_ptr_{std::addressof(phase_coefficient)}
            { }

            template <typename Iterator, typename StateInteger>
            void operator()(
              Iterator const, Iterator const, Iterator const, Iterator const first_11,
              StateInteger const index, int const) const
            { *(first_11 + index) *= *phase_coefficient_ptr_; }
          }; // struct cphase_shift_coeff_tcp<Complex>
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        } // namespace phase_shift_detail

        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename Complex, typename StateInteger, typename BitInteger>
        inline RandomAccessRange& cphase_shift_coeff_tcp(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state, Complex const& phase_coefficient,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_target_qubit,
          ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const permutated_control_qubit)
        {
# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::two_page_qubits_gate<0u>(
            parallel_policy, local_state, permutated_target_qubit, permutated_control_qubit,
            [&phase_coefficient](auto const, auto const, auto const, auto const first_11, StateInteger const index, int const)
            { *(first_11 + index) *= phase_coefficient; });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::two_page_qubits_gate<0u>(
            parallel_policy, local_state, permutated_target_qubit, permutated_control_qubit,
            ::ket::mpi::gate::page::phase_shift_detail::cphase_shift_coeff_tcp<Complex>{phase_coefficient});
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        }

        // cphase_shift_coeff_tp: only target qubit is on page
        // CU1_{tc}(theta) or C1U1_{tc}(theta)
        // CU1_{1,2}(theta) (a_{00} |00> + a_{01} |01> + a_{10} |10> + a{11} |11>)
        //   = a_{00} |00> + a_{01} |01> + a_{10} |10> + e^{i theta} a_{11} |11>
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename Complex, typename StateInteger, typename BitInteger>
        inline RandomAccessRange& cphase_shift_coeff_tp(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          Complex const& phase_coefficient,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_target_qubit,
          ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const permutated_control_qubit,
          yampi::rank const rank)
        {
          return ::ket::mpi::gate::page::detail::cphase_shift_coeff_tp(
            mpi_policy, parallel_policy, local_state,
            phase_coefficient, permutated_target_qubit, permutated_control_qubit, rank);
        }

        // cphase_shift_coeff_cp: only control qubit is on page
        // CU1_{tc}(theta) or C1U1_{tc}(theta)
        // CU1_{1,2}(theta) (a_{00} |00> + a_{01} |01> + a_{10} |10> + a{11} |11>)
        //   = a_{00} |00> + a_{01} |01> + a_{10} |10> + e^{i theta} a_{11} |11>
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename Complex, typename StateInteger, typename BitInteger>
        inline RandomAccessRange& cphase_shift_coeff_cp(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          Complex const& phase_coefficient,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_target_qubit,
          ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const permutated_control_qubit,
          yampi::rank const rank)
        {
          return ::ket::mpi::gate::page::detail::cphase_shift_coeff_cp(
            mpi_policy, parallel_policy, local_state,
            phase_coefficient, permutated_target_qubit, permutated_control_qubit, rank);
        }

        // generalized phase_shift
        namespace phase_shift_detail
        {
# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
          template <typename Complex>
          struct phase_shift2
          {
            Complex modified_phase_coefficient1_;
            Complex phase_coefficient2_;

            phase_shift2(
              Complex const& modified_phase_coefficient1,
              Complex const& phase_coefficient2) noexcept
              : modified_phase_coefficient1_{modified_phase_coefficient1},
                phase_coefficient2_{phase_coefficient2}
            { }

            template <typename Iterator, typename StateInteger>
            void operator()(Iterator const zero_first, Iterator const one_first, StateInteger const index, int const) const
            {
              auto const zero_iter = zero_first + index;
              auto const one_iter = one_first + index;
              auto const zero_iter_value = *zero_iter;

              using real_type = typename ::ket::utility::meta::real_of<Complex>::type;
              using boost::math::constants::one_div_root_two;
              *zero_iter -= phase_coefficient2_ * *one_iter;
              *zero_iter *= one_div_root_two<real_type>();
              *one_iter *= phase_coefficient2_;
              *one_iter += zero_iter_value;
              *one_iter *= modified_phase_coefficient1_;
            }
          }; // struct phase_shift2<Complex>

          template <typename Complex>
          inline ::ket::mpi::gate::page::phase_shift_detail::phase_shift2<Complex>
          make_phase_shift2(
            Complex const& modified_phase_coefficient1, Complex const& phase_coefficient2)
          { return {modified_phase_coefficient1, phase_coefficient2}; }
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        } // namespace phase_shift_detail

        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename Real, typename StateInteger, typename BitInteger>
        inline RandomAccessRange& phase_shift2(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          Real const phase1, Real const phase2,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_qubit)
        {
          using complex_type = typename boost::range_value<RandomAccessRange>::type;
          auto const phase_coefficient1 = ::ket::utility::exp_i<complex_type>(phase1);
          auto const phase_coefficient2 = ::ket::utility::exp_i<complex_type>(phase2);

          using boost::math::constants::one_div_root_two;
          auto const modified_phase_coefficient1 = one_div_root_two<Real>() * phase_coefficient1;

# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<0u>(
            parallel_policy, local_state, permutated_qubit,
            [modified_phase_coefficient1, phase_coefficient2](auto const zero_first, auto const one_first, StateInteger const index, int const)
            {
              auto const zero_iter = zero_first + index;
              auto const one_iter = one_first + index;
              auto const zero_iter_value = *zero_iter;

              using boost::math::constants::one_div_root_two;
              *zero_iter -= phase_coefficient2 * *one_iter;
              *zero_iter *= one_div_root_two<Real>();
              *one_iter *= phase_coefficient2;
              *one_iter += zero_iter_value;
              *one_iter *= modified_phase_coefficient1;
            });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<0u>(
            parallel_policy, local_state, permutated_qubit,
            ::ket::mpi::gate::page::phase_shift_detail::make_phase_shift2(modified_phase_coefficient1, phase_coefficient2));
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        }

        // cphase_shift2_tcp: both of target and control qubits of CPhase are on page
        // CU2_{tc}(theta, theta') or C1U2_{tc}(theta, theta')
        // CU2_{1,2}(theta, theta') (a_{00} |00> + a_{01} |01> + a_{10} |10> + a{11} |11>)
        //   = a_{00} |00> + a_{01} |01> + (a_{10} - e^{i theta'} a_{11})/sqrt(2) |10> + (e^{i theta} a_{10} + e^{i(theta + theta')} a_{11})/sqrt(2) |11>
        namespace phase_shift_detail
        {
# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
          template <typename Complex>
          struct cphase_shift2_tcp
          {
            Complex const* modified_phase_coefficient1_ptr_;
            Complex const* phase_coefficient2_ptr_;

            cphase_shift2_tcp(Complex const& modified_phase_coefficient1, Complex const& phase_coefficient2) noexcept
              : modified_phase_coefficient1_ptr_{std::addressof(modified_phase_coefficient1)},
                phase_coefficient2_ptr_{std::addressof(phase_coefficient2)}
            { }

            template <typename Iterator, typename StateInteger>
            void operator()(
              Iterator const, Iterator const, Iterator const first_10, Iterator const first_11,
              StateInteger const index, int const) const
            {
              auto const control_on_iter = first_10 + index;
              auto const target_control_on_iter = first_11 + index;
              auto const control_on_iter_value = *control_on_iter;

              using real_type = typename ::ket::utility::meta::real_of<Complex>::type;
              using boost::math::constants::one_div_root_two;
              *control_on_iter -= *phase_coefficient2_ptr_ * *target_control_on_iter;
              *control_on_iter *= one_div_root_two<real_type>();
              *target_control_on_iter *= *phase_coefficient2_ptr_;
              *target_control_on_iter += control_on_iter_value;
              *target_control_on_iter *= *modified_phase_coefficient1_ptr_;
            }
          }; // struct cphase_shift2_tcp<Complex>

          template <typename Complex>
          inline ::ket::mpi::gate::page::phase_shift_detail::cphase_shift2_tcp<Complex>
          make_cphase_shift2_tcp(Complex const& modified_phase_coefficient1, Complex const& phase_coefficient2)
          { return {modified_phase_coefficient1, phase_coefficient2}; }
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        } // namespace phase_shift_detail

        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename Real, typename StateInteger, typename BitInteger>
        inline RandomAccessRange& cphase_shift2_tcp(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state, Real const phase1, Real const phase2,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_target_qubit,
          ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const permutated_control_qubit)
        {
          using complex_type = typename boost::range_value<RandomAccessRange>::type;
          static_assert(
            (std::is_same<Real, typename ::ket::utility::meta::real_of<complex_type>::type>::value),
            "Real must be the same to \"real part\" of value_type of RandomAccessIterator");

          auto const phase_coefficient1 = ::ket::utility::exp_i<complex_type>(phase1);
          auto const phase_coefficient2 = ::ket::utility::exp_i<complex_type>(phase2);

          using boost::math::constants::one_div_root_two;
          auto const modified_phase_coefficient1 = one_div_root_two<Real>() * phase_coefficient1;

# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::two_page_qubits_gate<0u>(
            parallel_policy, local_state, permutated_target_qubit, permutated_control_qubit,
            [&modified_phase_coefficient1, &phase_coefficient2](
              auto const, auto const, auto const first_10, auto const first_11, StateInteger const index, int const)
            {
              auto const control_on_iter = first_10 + index;
              auto const target_control_on_iter = first_11 + index;
              auto const control_on_iter_value = *control_on_iter;

              using boost::math::constants::one_div_root_two;
              *control_on_iter -= phase_coefficient2 * *target_control_on_iter;
              *control_on_iter *= one_div_root_two<Real>();
              *target_control_on_iter *= phase_coefficient2;
              *target_control_on_iter += control_on_iter_value;
              *target_control_on_iter *= modified_phase_coefficient1;
            });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::two_page_qubits_gate<0u>(
            parallel_policy, local_state, permutated_target_qubit, permutated_control_qubit,
            ::ket::mpi::gate::page::phase_shift_detail::make_cphase_shift2_tcp(modified_phase_coefficient1, phase_coefficient2));
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        }

        // cphase_shift2_tp: only target qubit is on page
        // CU2_{tc}(theta, theta') or C1U2_{tc}(theta, theta')
        // CU2_{1,2}(theta, theta') (a_{00} |00> + a_{01} |01> + a_{10} |10> + a{11} |11>)
        //   = a_{00} |00> + a_{01} |01> + (a_{10} - e^{i theta'} a_{11})/sqrt(2) |10> + (e^{i theta} a_{10} + e^{i(theta + theta')} a_{11})/sqrt(2) |11>
        namespace phase_shift_detail
        {
# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
          template <typename Complex, typename StateInteger>
          struct cphase_shift2_tp
          {
            Complex const* modified_phase_coefficient1_ptr_;
            Complex const* phase_coefficient2_ptr_;
            StateInteger control_qubit_mask_;
            StateInteger nonpage_lower_bits_mask_;
            StateInteger nonpage_upper_bits_mask_;

            cphase_shift2_tp(
              Complex const& modified_phase_coefficient1, Complex const& phase_coefficient2,
              StateInteger const control_qubit_mask,
              StateInteger const nonpage_lower_bits_mask,
              StateInteger const nonpage_upper_bits_mask) noexcept
              : modified_phase_coefficient1_ptr_{std::addressof(modified_phase_coefficient1)},
                phase_coefficient2_ptr_{std::addressof(phase_coefficient2)},
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
              auto const control_on_iter_value = *control_on_iter;

              using real_type = typename ::ket::utility::meta::real_of<Complex>::type;
              using boost::math::constants::one_div_root_two;
              *control_on_iter -= *phase_coefficient2_ptr_ * *target_control_on_iter;
              *control_on_iter *= one_div_root_two<real_type>();
              *target_control_on_iter *= *phase_coefficient2_ptr_;
              *target_control_on_iter += control_on_iter_value;
              *target_control_on_iter *= *modified_phase_coefficient1_ptr_;
            }
          }; // struct cphase_shift2_tp<Complex, StateInteger>

          template <typename Complex, typename StateInteger>
          inline ::ket::mpi::gate::page::phase_shift_detail::cphase_shift2_tp<Complex, StateInteger>
          make_cphase_shift2_tp(
            Complex const& modified_phase_coefficient1, Complex const& phase_coefficient2,
            StateInteger const control_qubit_mask,
            StateInteger const nonpage_lower_bits_mask,
            StateInteger const nonpage_upper_bits_mask)
          { return {modified_phase_coefficient1, phase_coefficient2, control_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask}; }
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        } // namespace phase_shift_detail

        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename Real, typename StateInteger, typename BitInteger>
        inline RandomAccessRange& cphase_shift2_tp(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state, Real const phase1, Real const phase2,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_target_qubit,
          ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const permutated_control_qubit)
        {
          using complex_type = typename boost::range_value<RandomAccessRange>::type;
          static_assert(
            (std::is_same<Real, typename ::ket::utility::meta::real_of<complex_type>::type>::value),
            "Real must be the same to \"real part\" of value_type of RandomAccessIterator");

          auto const phase_coefficient1 = ::ket::utility::exp_i<complex_type>(phase1);
          auto const phase_coefficient2 = ::ket::utility::exp_i<complex_type>(phase2);

          using boost::math::constants::one_div_root_two;
          auto const modified_phase_coefficient1 = one_div_root_two<Real>() * phase_coefficient1;

          assert(not ::ket::mpi::page::is_on_page(permutated_control_qubit, local_state));
          auto const control_qubit_mask
            = ::ket::utility::integer_exp2<StateInteger>(permutated_control_qubit);
          auto const nonpage_lower_bits_mask = control_qubit_mask - StateInteger{1u};
          auto const nonpage_upper_bits_mask = compl nonpage_lower_bits_mask;

# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<1u>(
            parallel_policy, local_state, permutated_target_qubit,
            [&modified_phase_coefficient1, &phase_coefficient2, control_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask](
              auto const zero_first, auto const one_first, StateInteger const index_wo_nonpage_qubit, int const)
            {
              auto const zero_index
                = ((index_wo_nonpage_qubit bitand nonpage_upper_bits_mask) << 1u)
                  bitor (index_wo_nonpage_qubit bitand nonpage_lower_bits_mask);
              auto const one_index = zero_index bitor control_qubit_mask;
              auto const control_on_iter = zero_first + one_index;
              auto const target_control_on_iter = one_first + one_index;
              auto const control_on_iter_value = *control_on_iter;

              *control_on_iter -= phase_coefficient2 * *target_control_on_iter;
              *control_on_iter *= one_div_root_two<Real>();
              *target_control_on_iter *= phase_coefficient2;
              *target_control_on_iter += control_on_iter_value;
              *target_control_on_iter *= modified_phase_coefficient1;
            });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<1u>(
            parallel_policy, local_state, permutated_target_qubit,
            ::ket::mpi::gate::page::phase_shift_detail::make_cphase_shift2_tp(
              modified_phase_coefficient1, phase_coefficient2, control_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask));
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        }

        // cphase_shift2_cp: only control qubit is on page
        // CU2_{tc}(theta, theta') or C1U2_{tc}(theta, theta')
        // CU2_{1,2}(theta, theta') (a_{00} |00> + a_{01} |01> + a_{10} |10> + a{11} |11>)
        //   = a_{00} |00> + a_{01} |01> + (a_{10} - e^{i theta'} a_{11})/sqrt(2) |10> + (e^{i theta} a_{10} + e^{i(theta + theta')} a_{11})/sqrt(2) |11>
        namespace phase_shift_detail
        {
# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
          template <typename Complex, typename StateInteger>
          struct cphase_shift2_cp
          {
            Complex const* modified_phase_coefficient1_ptr_;
            Complex const* phase_coefficient2_ptr_;
            StateInteger target_qubit_mask_;
            StateInteger nonpage_lower_bits_mask_;
            StateInteger nonpage_upper_bits_mask_;

            cphase_shift2_cp(
              Complex const& modified_phase_coefficient1, Complex const& phase_coefficient2,
              StateInteger const target_qubit_mask,
              StateInteger const nonpage_lower_bits_mask,
              StateInteger const nonpage_upper_bits_mask) noexcept
              : modified_phase_coefficient1_ptr_{std::addressof(modified_phase_coefficient1)},
                phase_coefficient2_ptr_{std::addressof(phase_coefficient2)},
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
              auto const control_on_iter_value = *control_on_iter;

              using real_type = typename ::ket::utility::meta::real_of<Complex>::type;
              using boost::math::constants::one_div_root_two;
              *control_on_iter -= *phase_coefficient2_ptr_ * *target_control_on_iter;
              *control_on_iter *= one_div_root_two<real_type>();
              *target_control_on_iter *= *phase_coefficient2_ptr_;
              *target_control_on_iter += control_on_iter_value;
              *target_control_on_iter *= *modified_phase_coefficient1_ptr_;
            }
          }; // struct cphase_shift2_cp<Complex, StateInteger>

          template <typename Complex, typename StateInteger>
          inline ::ket::mpi::gate::page::phase_shift_detail::cphase_shift2_cp<Complex, StateInteger>
          make_cphase_shift2_cp(
            Complex const& modified_phase_coefficient1, Complex const& phase_coefficient2,
            StateInteger const target_qubit_mask,
            StateInteger const nonpage_lower_bits_mask,
            StateInteger const nonpage_upper_bits_mask)
          { return {modified_phase_coefficient1, phase_coefficient2, target_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask}; }
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        } // namespace phase_shift_detail

        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename Real, typename StateInteger, typename BitInteger>
        inline RandomAccessRange& cphase_shift2_cp(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state, Real const phase1, Real const phase2,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_target_qubit,
          ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const permutated_control_qubit)
        {
          using complex_type = typename boost::range_value<RandomAccessRange>::type;
          static_assert(
            (std::is_same<Real, typename ::ket::utility::meta::real_of<complex_type>::type>::value),
            "Real must be the same to \"real part\" of value_type of RandomAccessIterator");

          auto const phase_coefficient1 = ::ket::utility::exp_i<complex_type>(phase1);
          auto const phase_coefficient2 = ::ket::utility::exp_i<complex_type>(phase2);

          using boost::math::constants::one_div_root_two;
          auto const modified_phase_coefficient1 = one_div_root_two<Real>() * phase_coefficient1;

          assert(not ::ket::mpi::page::is_on_page(permutated_target_qubit, local_state));

          auto const target_qubit_mask
            = ::ket::utility::integer_exp2<StateInteger>(permutated_target_qubit);
          auto const nonpage_lower_bits_mask = target_qubit_mask - StateInteger{1u};
          auto const nonpage_upper_bits_mask = compl nonpage_lower_bits_mask;

# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<1u>(
            parallel_policy, local_state, permutated_control_qubit,
            [&modified_phase_coefficient1, &phase_coefficient2, target_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask](
              auto const, auto const one_first, StateInteger const index_wo_nonpage_qubit, int const)
            {
              auto const zero_index
                = ((index_wo_nonpage_qubit bitand nonpage_upper_bits_mask) << 1u)
                  bitor (index_wo_nonpage_qubit bitand nonpage_lower_bits_mask);
              auto const one_index = zero_index bitor target_qubit_mask;
              auto const control_on_iter = one_first + zero_index;
              auto const target_control_on_iter = one_first + one_index;
              auto const control_on_iter_value = *control_on_iter;

              *control_on_iter -= phase_coefficient2 * *target_control_on_iter;
              *control_on_iter *= one_div_root_two<Real>();
              *target_control_on_iter *= phase_coefficient2;
              *target_control_on_iter += control_on_iter_value;
              *target_control_on_iter *= modified_phase_coefficient1;
            });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<1u>(
            parallel_policy, local_state, permutated_control_qubit,
            ::ket::mpi::gate::page::phase_shift_detail::make_cphase_shift2_cp(
              modified_phase_coefficient1, phase_coefficient2, target_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask));
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        }

        namespace phase_shift_detail
        {
# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
          template <typename Complex>
          struct adj_phase_shift2
          {
            Complex phase_coefficient1_;
            Complex modified_phase_coefficient2_;

            adj_phase_shift2(
              Complex const& phase_coefficient1,
              Complex const& modified_phase_coefficient2) noexcept
              : phase_coefficient1_{phase_coefficient1},
                modified_phase_coefficient2_{modified_phase_coefficient2}
            { }

            template <typename Iterator, typename StateInteger>
            void operator()(Iterator const zero_first, Iterator const one_first, StateInteger const index, int const) const
            {
              auto const zero_iter = zero_first + index;
              auto const one_iter = one_first + index;
              auto const zero_iter_value = *zero_iter;

              using real_type = typename ::ket::utility::meta::real_of<Complex>::type;
              using boost::math::constants::one_div_root_two;
              *zero_iter += phase_coefficient1_ * *one_iter;
              *zero_iter *= one_div_root_two<real_type>();
              *one_iter *= phase_coefficient1_;
              *one_iter -= zero_iter_value;
              *one_iter *= modified_phase_coefficient2_;
            }
          }; // struct adj_phase_shift2<Complex>

          template <typename Complex>
          inline ::ket::mpi::gate::page::phase_shift_detail::adj_phase_shift2<Complex>
          make_adj_phase_shift2(
            Complex const& phase_coefficient1, Complex const& modified_phase_coefficient2)
          { return {phase_coefficient1, modified_phase_coefficient2}; }
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        } // namespace phase_shift_detail

        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename Real, typename StateInteger, typename BitInteger>
        inline RandomAccessRange& adj_phase_shift2(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          Real const phase1, Real const phase2,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_qubit)
        {
          using complex_type = typename boost::range_value<RandomAccessRange>::type;
          auto const phase_coefficient1 = ::ket::utility::exp_i<complex_type>(-phase1);
          auto const phase_coefficient2 = ::ket::utility::exp_i<complex_type>(-phase2);

          using boost::math::constants::one_div_root_two;
          auto const modified_phase_coefficient2 = one_div_root_two<Real>() * phase_coefficient2;

# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<0u>(
            parallel_policy, local_state, permutated_qubit,
            [phase_coefficient1, modified_phase_coefficient2](auto const zero_first, auto const one_first, StateInteger const index, int const)
            {
              auto const zero_iter = zero_first + index;
              auto const one_iter = one_first + index;
              auto const zero_iter_value = *zero_iter;

              *zero_iter += phase_coefficient1 * *one_iter;
              *zero_iter *= one_div_root_two<Real>();
              *one_iter *= phase_coefficient1;
              *one_iter -= zero_iter_value;
              *one_iter *= modified_phase_coefficient2;
            });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<0u>(
            parallel_policy, local_state, permutated_qubit,
            ::ket::mpi::gate::page::phase_shift_detail::make_adj_phase_shift2(phase_coefficient1, modified_phase_coefficient2));
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        }

        // adj_cphase_shift2_tcp: both of target and control qubits of Adj(CPhase) are on page
        // CU2+_{tc}(theta, theta') or C1U2+_{tc}(theta, theta')
        // CU2+_{1,2}(theta, theta') (a_{00} |00> + a_{01} |01> + a_{10} |10> + a{11} |11>)
        //   = a_{00} |00> + a_{01} |01> + (a_{10} + e^{-i theta} a_{11})/sqrt(2) |10> + (-e^{-i theta'} a_{10} + e^{-i(theta + theta')} a_{11})/sqrt(2) |11>
        namespace phase_shift_detail
        {
# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
          template <typename Complex>
          struct adj_cphase_shift2_tcp
          {
            Complex const* phase_coefficient1_ptr_;
            Complex const* modified_phase_coefficient2_ptr_;

            adj_cphase_shift2_tcp(Complex const& phase_coefficient1, Complex const& modified_phase_coefficient2) noexcept
              : phase_coefficient1_ptr_{std::addressof(phase_coefficient1)},
                modified_phase_coefficient2_ptr_{std::addressof(modified_phase_coefficient2)}
            { }

            template <typename Iterator, typename StateInteger>
            void operator()(
              Iterator const, Iterator const, Iterator const first_10, Iterator const first_11,
              StateInteger const index, int const) const
            {
              auto const control_on_iter = first_10 + index;
              auto const target_control_on_iter = first_11 + index;
              auto const control_on_iter_value = *control_on_iter;

              using real_type = typename ::ket::utility::meta::real_of<Complex>::type;
              using boost::math::constants::one_div_root_two;
              *control_on_iter += *phase_coefficient1_ptr_ * *target_control_on_iter;
              *control_on_iter *= one_div_root_two<real_type>();
              *target_control_on_iter *= *phase_coefficient1_ptr_;
              *target_control_on_iter -= control_on_iter_value;
              *target_control_on_iter *= *modified_phase_coefficient2_ptr_;
            }
          }; // struct adj_cphase_shift2_tcp<Complex>

          template <typename Complex>
          inline ::ket::mpi::gate::page::phase_shift_detail::adj_cphase_shift2_tcp<Complex>
          make_adj_cphase_shift2_tcp(Complex const& phase_coefficient1, Complex const& modified_phase_coefficient2)
          { return {phase_coefficient1, modified_phase_coefficient2}; }
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        } // namespace phase_shift_detail

        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename Real, typename StateInteger, typename BitInteger>
        inline RandomAccessRange& adj_cphase_shift2_tcp(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state, Real const phase1, Real const phase2,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_target_qubit,
          ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const permutated_control_qubit)
        {
          using complex_type = typename boost::range_value<RandomAccessRange>::type;
          static_assert(
            (std::is_same<Real, typename ::ket::utility::meta::real_of<complex_type>::type>::value),
            "Real must be the same to \"real part\" of value_type of RandomAccessIterator");

          auto const phase_coefficient1 = ::ket::utility::exp_i<complex_type>(-phase1);
          auto const phase_coefficient2 = ::ket::utility::exp_i<complex_type>(-phase2);

          using boost::math::constants::one_div_root_two;
          auto const modified_phase_coefficient2 = one_div_root_two<Real>() * phase_coefficient2;

# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::two_page_qubits_gate<0u>(
            parallel_policy, local_state, permutated_target_qubit, permutated_control_qubit,
            [&phase_coefficient1, &modified_phase_coefficient2](
              auto const, auto const, auto const first_10, auto const first_11, StateInteger const index, int const)
            {
              auto const control_on_iter = first_10 + index;
              auto const target_control_on_iter = first_11 + index;
              auto const control_on_iter_value = *control_on_iter;

              *control_on_iter += phase_coefficient1 * *target_control_on_iter;
              *control_on_iter *= one_div_root_two<Real>();
              *target_control_on_iter *= phase_coefficient1;
              *target_control_on_iter -= control_on_iter_value;
              *target_control_on_iter *= modified_phase_coefficient2;
            });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::two_page_qubits_gate<0u>(
            parallel_policy, local_state, permutated_target_qubit, permutated_control_qubit,
            ::ket::mpi::gate::page::phase_shift_detail::make_adj_cphase_shift2_tcp(phase_coefficient1, modified_phase_coefficient2));
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        }

        // adj_cphase_shift2_tp: only target qubit is on page
        // CU2+_{tc}(theta, theta') or C1U2+_{tc}(theta, theta')
        // CU2+_{1,2}(theta, theta') (a_{00} |00> + a_{01} |01> + a_{10} |10> + a{11} |11>)
        //   = a_{00} |00> + a_{01} |01> + (a_{10} + e^{-i theta} a_{11})/sqrt(2) |10> + (-e^{-i theta'} a_{10} + e^{-i(theta + theta')} a_{11})/sqrt(2) |11>
        namespace phase_shift_detail
        {
# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
          template <typename Complex, typename StateInteger>
          struct adj_cphase_shift2_tp
          {
            Complex const* phase_coefficient1_ptr_;
            Complex const* modified_phase_coefficient2_ptr_;
            StateInteger control_qubit_mask_;
            StateInteger nonpage_lower_bits_mask_;
            StateInteger nonpage_upper_bits_mask_;

            adj_cphase_shift2_tp(
              Complex const& phase_coefficient1, Complex const& modified_phase_coefficient2,
              StateInteger const control_qubit_mask,
              StateInteger const nonpage_lower_bits_mask,
              StateInteger const nonpage_upper_bits_mask) noexcept
              : phase_coefficient1_ptr_{std::addressof(phase_coefficient1)},
                modified_phase_coefficient2_ptr_{std::addressof(modified_phase_coefficient2)},
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
              auto const control_on_iter_value = *control_on_iter;

              using real_type = typename ::ket::utility::meta::real_of<Complex>::type;
              using boost::math::constants::one_div_root_two;
              *control_on_iter += *phase_coefficient1_ptr_ * *target_control_on_iter;
              *control_on_iter *= one_div_root_two<real_type>();
              *target_control_on_iter *= *phase_coefficient1_ptr_;
              *target_control_on_iter -= control_on_iter_value;
              *target_control_on_iter *= *modified_phase_coefficient2_ptr_;
            }
          }; // struct adj_cphase_shift2_tp<Complex, StateInteger>

          template <typename Complex, typename StateInteger>
          inline ::ket::mpi::gate::page::phase_shift_detail::adj_cphase_shift2_tp<Complex, StateInteger>
          make_adj_cphase_shift2_tp(
            Complex const& phase_coefficient1, Complex const& modified_phase_coefficient2,
            StateInteger const control_qubit_mask,
            StateInteger const nonpage_lower_bits_mask,
            StateInteger const nonpage_upper_bits_mask)
          { return {phase_coefficient1, modified_phase_coefficient2, control_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask}; }
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        } // namespace phase_shift_detail

        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename Real, typename StateInteger, typename BitInteger>
        inline RandomAccessRange& adj_cphase_shift2_tp(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state, Real const phase1, Real const phase2,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_target_qubit,
          ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const permutated_control_qubit)
        {
          using complex_type = typename boost::range_value<RandomAccessRange>::type;
          static_assert(
            (std::is_same<Real, typename ::ket::utility::meta::real_of<complex_type>::type>::value),
            "Real must be the same to \"real part\" of value_type of RandomAccessIterator");

          auto const phase_coefficient1 = ::ket::utility::exp_i<complex_type>(-phase1);
          auto const phase_coefficient2 = ::ket::utility::exp_i<complex_type>(-phase2);

          using boost::math::constants::one_div_root_two;
          auto const modified_phase_coefficient2 = one_div_root_two<Real>() * phase_coefficient2;

          assert(not ::ket::mpi::page::is_on_page(permutated_control_qubit, local_state));
          auto const control_qubit_mask
            = ::ket::utility::integer_exp2<StateInteger>(permutated_control_qubit);
          auto const nonpage_lower_bits_mask = control_qubit_mask - StateInteger{1u};
          auto const nonpage_upper_bits_mask = compl nonpage_lower_bits_mask;

# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<1u>(
            parallel_policy, local_state, permutated_target_qubit,
            [&phase_coefficient1, &modified_phase_coefficient2, control_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask](
              auto const zero_first, auto const one_first, StateInteger const index_wo_nonpage_qubit, int const)
            {
              auto const zero_index
                = ((index_wo_nonpage_qubit bitand nonpage_upper_bits_mask) << 1u)
                  bitor (index_wo_nonpage_qubit bitand nonpage_lower_bits_mask);
              auto const one_index = zero_index bitor control_qubit_mask;
              auto const control_on_iter = zero_first + one_index;
              auto const target_control_on_iter = one_first + one_index;
              auto const control_on_iter_value = *control_on_iter;

              *control_on_iter += phase_coefficient1 * *target_control_on_iter;
              *control_on_iter *= one_div_root_two<Real>();
              *target_control_on_iter *= phase_coefficient1;
              *target_control_on_iter -= control_on_iter_value;
              *target_control_on_iter *= modified_phase_coefficient2;
            });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<1u>(
            parallel_policy, local_state, permutated_target_qubit,
            ::ket::mpi::gate::page::phase_shift_detail::make_adj_cphase_shift2_tp(
              phase_coefficient1, modified_phase_coefficient2, control_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask));
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        }

        // adj_cphase_shift2_cp: only control qubit is on page
        // CU2+_{tc}(theta, theta') or C1U2+_{tc}(theta, theta')
        // CU2+_{1,2}(theta, theta') (a_{00} |00> + a_{01} |01> + a_{10} |10> + a{11} |11>)
        //   = a_{00} |00> + a_{01} |01> + (a_{10} + e^{-i theta} a_{11})/sqrt(2) |10> + (-e^{-i theta'} a_{10} + e^{-i(theta + theta')} a_{11})/sqrt(2) |11>
        namespace phase_shift_detail
        {
# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
          template <typename Complex, typename StateInteger>
          struct adj_cphase_shift2_cp
          {
            Complex const* phase_coefficient1_ptr_;
            Complex const* modified_phase_coefficient2_ptr_;
            StateInteger target_qubit_mask_;
            StateInteger nonpage_lower_bits_mask_;
            StateInteger nonpage_upper_bits_mask_;

            adj_cphase_shift2_cp(
              Complex const& phase_coefficient1, Complex const& modified_phase_coefficient2,
              StateInteger const target_qubit_mask,
              StateInteger const nonpage_lower_bits_mask,
              StateInteger const nonpage_upper_bits_mask) noexcept
              : phase_coefficient1_ptr_{std::addressof(phase_coefficient1)},
                modified_phase_coefficient2_ptr_{std::addressof(modified_phase_coefficient2)},
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
              auto const control_on_iter_value = *control_on_iter;

              using real_type = typename ::ket::utility::meta::real_of<Complex>::type;
              using boost::math::constants::one_div_root_two;
              *control_on_iter += *phase_coefficient1_ptr_ * *target_control_on_iter;
              *control_on_iter *= one_div_root_two<real_type>();
              *target_control_on_iter *= *phase_coefficient1_ptr_;
              *target_control_on_iter -= control_on_iter_value;
              *target_control_on_iter *= *modified_phase_coefficient2_ptr_;
            }
          }; // struct adj_cphase_shift2_cp<Complex, StateInteger>

          template <typename Complex, typename StateInteger>
          inline ::ket::mpi::gate::page::phase_shift_detail::adj_cphase_shift2_cp<Complex, StateInteger>
          make_adj_cphase_shift2_cp(
            Complex const& phase_coefficient1, Complex const& modified_phase_coefficient2,
            StateInteger const target_qubit_mask,
            StateInteger const nonpage_lower_bits_mask,
            StateInteger const nonpage_upper_bits_mask)
          { return {phase_coefficient1, modified_phase_coefficient2, target_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask}; }
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        } // namespace phase_shift_detail

        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename Real, typename StateInteger, typename BitInteger>
        inline RandomAccessRange& adj_cphase_shift2_cp(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state, Real const phase1, Real const phase2,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_target_qubit,
          ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const permutated_control_qubit)
        {
          using complex_type = typename boost::range_value<RandomAccessRange>::type;
          static_assert(
            (std::is_same<Real, typename ::ket::utility::meta::real_of<complex_type>::type>::value),
            "Real must be the same to \"real part\" of value_type of RandomAccessIterator");

          auto const phase_coefficient1 = ::ket::utility::exp_i<complex_type>(-phase1);
          auto const phase_coefficient2 = ::ket::utility::exp_i<complex_type>(-phase2);

          using boost::math::constants::one_div_root_two;
          auto const modified_phase_coefficient2 = one_div_root_two<Real>() * phase_coefficient2;

          assert(not ::ket::mpi::page::is_on_page(permutated_target_qubit, local_state));

          auto const target_qubit_mask
            = ::ket::utility::integer_exp2<StateInteger>(permutated_target_qubit);
          auto const nonpage_lower_bits_mask = target_qubit_mask - StateInteger{1u};
          auto const nonpage_upper_bits_mask = compl nonpage_lower_bits_mask;

# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<1u>(
            parallel_policy, local_state, permutated_control_qubit,
            [&phase_coefficient1, &modified_phase_coefficient2, target_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask](
              auto const, auto const one_first, StateInteger const index_wo_nonpage_qubit, int const)
            {
              auto const zero_index
                = ((index_wo_nonpage_qubit bitand nonpage_upper_bits_mask) << 1u)
                  bitor (index_wo_nonpage_qubit bitand nonpage_lower_bits_mask);
              auto const one_index = zero_index bitor target_qubit_mask;
              auto const control_on_iter = one_first + zero_index;
              auto const target_control_on_iter = one_first + one_index;
              auto const control_on_iter_value = *control_on_iter;

              *control_on_iter += phase_coefficient1 * *target_control_on_iter;
              *control_on_iter *= one_div_root_two<Real>();
              *target_control_on_iter *= phase_coefficient1;
              *target_control_on_iter -= control_on_iter_value;
              *target_control_on_iter *= modified_phase_coefficient2;
            });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<1u>(
            parallel_policy, local_state, permutated_control_qubit,
            ::ket::mpi::gate::page::phase_shift_detail::make_adj_cphase_shift2_cp(
              phase_coefficient1, modified_phase_coefficient2, target_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask));
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        }

        namespace phase_shift_detail
        {
# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
          template <typename Real, typename Complex>
          struct phase_shift3
          {
            Real sine_;
            Real cosine_;
            Complex phase_coefficient2_;
            Complex sine_phase_coefficient3_;
            Complex cosine_phase_coefficient3_;

            phase_shift3(
              Real const sine, Real const cosine,
              Complex const& phase_coefficient2,
              Complex const& sine_phase_coefficient3,
              Complex const& cosine_phase_coefficient3) noexcept
              : sine_{sine},
                cosine_{cosine},
                phase_coefficient2_{phase_coefficient2},
                sine_phase_coefficient3_{sine_phase_coefficient3},
                cosine_phase_coefficient3_{cosine_phase_coefficient3}
            { }

            template <typename Iterator, typename StateInteger>
            void operator()(Iterator const zero_first, Iterator const one_first, StateInteger const index, int const) const
            {
              auto const zero_iter = zero_first + index;
              auto const one_iter = one_first + index;
              auto const zero_iter_value = *zero_iter;

              *zero_iter *= cosine_;
              *zero_iter -= sine_phase_coefficient3_ * *one_iter;
              *one_iter *= cosine_phase_coefficient3_;
              *one_iter += sine_ * zero_iter_value;
              *one_iter *= phase_coefficient2_;
            }
          }; // struct phase_shift3<Real, Complex>

          template <typename Real, typename Complex>
          inline ::ket::mpi::gate::page::phase_shift_detail::phase_shift3<Real, Complex>
          make_phase_shift3(
            Real const sine, Real const cosine,
            Complex const& phase_coefficient2,
            Complex const& sine_phase_coefficient3,
            Complex const& cosine_phase_coefficient3)
          {
            return {
              sine, cosine, phase_coefficient2,
              sine_phase_coefficient3, cosine_phase_coefficient3};
          }
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        } // namespace phase_shift_detail

        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename Real, typename StateInteger, typename BitInteger>
        inline RandomAccessRange& phase_shift3(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          Real const phase1, Real const phase2, Real const phase3,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_qubit)
        {
          using std::cos;
          using std::sin;
          using boost::math::constants::half;
          auto const sine = sin(half<Real>() * phase1);
          auto const cosine = cos(half<Real>() * phase1);

          using complex_type = typename boost::range_value<RandomAccessRange>::type;
          auto const phase_coefficient2 = ::ket::utility::exp_i<complex_type>(phase2);
          auto const phase_coefficient3 = ::ket::utility::exp_i<complex_type>(phase3);

          auto const sine_phase_coefficient3 = sine * phase_coefficient3;
          auto const cosine_phase_coefficient3 = cosine * phase_coefficient3;

# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<0u>(
            parallel_policy, local_state, permutated_qubit,
            [sine, cosine, phase_coefficient2,
             sine_phase_coefficient3, cosine_phase_coefficient3](
              auto const zero_first, auto const one_first, StateInteger const index, int const)
            {
              auto const zero_iter = zero_first + index;
              auto const one_iter = one_first + index;
              auto const zero_iter_value = *zero_iter;

              *zero_iter *= cosine;
              *zero_iter -= sine_phase_coefficient3 * *one_iter;
              *one_iter *= cosine_phase_coefficient3;
              *one_iter += sine * zero_iter_value;
              *one_iter *= phase_coefficient2;
            });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<0u>(
            parallel_policy, local_state, permutated_qubit,
            ::ket::mpi::gate::page::phase_shift_detail::make_phase_shift3(
              sine, cosine, phase_coefficient2,
              sine_phase_coefficient3, cosine_phase_coefficient3));
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        }

        // cphase_shift3_tcp: both of target and control qubits of CPhase are on page
        // CU3_{tc}(theta, theta', theta'') or C1U3_{tc}(theta, theta', theta'')
        // CU3_{1,2}(theta, theta', theta'') (a_{00} |00> + a_{01} |01> + a_{10} |10> + a{11} |11>)
        //   = a_{00} |00> + a_{01} |01> + (cos(theta/2) a_{10} - e^{i theta''} sin(theta/2) a_{11}) |10>
        //     + (e^{i theta'} sin(theta/2) a_{10} + e^{i(theta' + theta'')} cos(theta/2) a_{11}) |11>
        namespace phase_shift_detail
        {
# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
          template <typename Real, typename Complex>
          struct cphase_shift3_tcp
          {
            Real sine_;
            Real cosine_;
            Complex const* phase_coefficient2_ptr_;
            Complex const* sine_phase_coefficient3_ptr_;
            Complex const* cosine_phase_coefficient3_ptr_;

            cphase_shift3_tcp(
              Real const sine, Real const cosine, Complex const& phase_coefficient2,
              Complex const& sine_phase_coefficient3, Complex const& cosine_phase_coefficient3) noexcept
              : sine_{sine}, cosine_{cosine},
                phase_coefficient2_ptr_{std::addressof(phase_coefficient2)},
                sine_phase_coefficient3_ptr_{std::addressof(sine_phase_coefficient3)},
                cosine_phase_coefficient3_ptr_{std::addressof(cosine_phase_coefficient3)}
            { }

            template <typename Iterator, typename StateInteger>
            void operator()(
              Iterator const, Iterator const, Iterator const first_10, Iterator const first_11,
              StateInteger const index, int const) const
            {
              auto const control_on_iter = first_10 + index;
              auto const target_control_on_iter = first_11 + index;
              auto const control_on_iter_value = *control_on_iter;

              *control_on_iter *= cosine_;
              *control_on_iter -= *sine_phase_coefficient3_ptr_ * *target_control_on_iter;
              *target_control_on_iter *= *cosine_phase_coefficient3_ptr_;
              *target_control_on_iter += sine_ * control_on_iter_value;
              *target_control_on_iter *= *phase_coefficient2_ptr_;
            }
          }; // struct cphase_shift3_tcp<Real, Complex>

          template <typename Real, typename Complex>
          inline ::ket::mpi::gate::page::phase_shift_detail::cphase_shift3_tcp<Real, Complex>
          make_cphase_shift3_tcp(
            Real const sine, Real const cosine, Complex const& phase_coefficient2,
            Complex const& sine_phase_coefficient3, Complex const& cosine_phase_coefficient3)
          { return {sine, cosine, phase_coefficient2, sine_phase_coefficient3, cosine_phase_coefficient3}; }
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        } // namespace phase_shift_detail

        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename Real, typename StateInteger, typename BitInteger>
        inline RandomAccessRange& cphase_shift3_tcp(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state, Real const phase1, Real const phase2, Real const phase3,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_target_qubit,
          ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const permutated_control_qubit)
        {
          using complex_type = typename boost::range_value<RandomAccessRange>::type;
          static_assert(
            (std::is_same<Real, typename ::ket::utility::meta::real_of<complex_type>::type>::value),
            "Real must be the same to \"real part\" of value_type of RandomAccessIterator");

          using std::cos;
          using std::sin;
          using boost::math::constants::half;
          auto const sine = sin(half<Real>() * phase1);
          auto const cosine = cos(half<Real>() * phase1);

          auto const phase_coefficient2 = ::ket::utility::exp_i<complex_type>(phase2);
          auto const phase_coefficient3 = ::ket::utility::exp_i<complex_type>(phase3);

          auto const sine_phase_coefficient3 = sine * phase_coefficient3;
          auto const cosine_phase_coefficient3 = cosine * phase_coefficient3;

# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::two_page_qubits_gate<0u>(
            parallel_policy, local_state, permutated_target_qubit, permutated_control_qubit,
            [sine, cosine, &phase_coefficient2, &sine_phase_coefficient3, &cosine_phase_coefficient3](
              auto const, auto const, auto const first_10, auto const first_11, StateInteger const index, int const)
            {
              auto const control_on_iter = first_10 + index;
              auto const target_control_on_iter = first_11 + index;
              auto const control_on_iter_value = *control_on_iter;

              *control_on_iter *= cosine;
              *control_on_iter -= sine_phase_coefficient3 * *target_control_on_iter;
              *target_control_on_iter *= cosine_phase_coefficient3;
              *target_control_on_iter += sine * control_on_iter_value;
              *target_control_on_iter *= phase_coefficient2;
            });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::two_page_qubits_gate<0u>(
            parallel_policy, local_state, permutated_target_qubit, permutated_control_qubit,
            ::ket::mpi::gate::page::phase_shift_detail::make_cphase_shift3_tcp(
              sine, cosine, phase_coefficient2, sine_phase_coefficient3, cosine_phase_coefficient3));
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        }

        // cphase_shift3_tp: only target qubit is on page
        // CU3_{tc}(theta, theta', theta'') or C1U3_{tc}(theta, theta', theta'')
        // CU3_{1,2}(theta, theta', theta'') (a_{00} |00> + a_{01} |01> + a_{10} |10> + a{11} |11>)
        //   = a_{00} |00> + a_{01} |01> + (cos(theta/2) a_{10} - e^{i theta''} sin(theta/2) a_{11}) |10>
        //     + (e^{i theta'} sin(theta/2) a_{10} + e^{i(theta' + theta'')} cos(theta/2) a_{11}) |11>
        namespace phase_shift_detail
        {
# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
          template <typename Real, typename Complex, typename StateInteger>
          struct cphase_shift3_tp
          {
            Real sine_;
            Real cosine_;
            Complex const* phase_coefficient2_ptr_;
            Complex const* sine_phase_coefficient3_ptr_;
            Complex const* cosine_phase_coefficient3_ptr_;
            StateInteger control_qubit_mask_;
            StateInteger nonpage_lower_bits_mask_;
            StateInteger nonpage_upper_bits_mask_;

            cphase_shift3_tp(
              Real const sine, Real const cosine, Complex const& phase_coefficient2,
              Complex const& sine_phase_coefficient3, Complex const& cosine_phase_coefficient3,
              StateInteger const control_qubit_mask,
              StateInteger const nonpage_lower_bits_mask,
              StateInteger const nonpage_upper_bits_mask) noexcept
              : sine_{sine}, cosine_{cosine},
                phase_coefficient2_ptr_{std::addressof(phase_coefficient2)},
                sine_phase_coefficient3_ptr_{std::addressof(sine_phase_coefficient3)},
                cosine_phase_coefficient3_ptr_{std::addressof(cosine_phase_coefficient3)},
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
              auto const control_on_iter_value = *control_on_iter;

              *control_on_iter *= cosine_;
              *control_on_iter -= *sine_phase_coefficient3_ptr_ * *target_control_on_iter;
              *target_control_on_iter *= *cosine_phase_coefficient3_ptr_;
              *target_control_on_iter += sine_ * control_on_iter_value;
              *target_control_on_iter *= *phase_coefficient2_ptr_;
            }
          }; // struct cphase_shift3_tp<Real, Complex, StateInteger>

          template <typename Real, typename Complex, typename StateInteger>
          inline ::ket::mpi::gate::page::phase_shift_detail::cphase_shift3_tp<Real, Complex, StateInteger>
          make_cphase_shift3_tp(
            Real const sine, Real const cosine, Complex const& phase_coefficient2,
            Complex const& sine_phase_coefficient3, Complex const& cosine_phase_coefficient3,
            StateInteger const control_qubit_mask,
            StateInteger const nonpage_lower_bits_mask,
            StateInteger const nonpage_upper_bits_mask)
          { return {sine, cosine, phase_coefficient2, sine_phase_coefficient3, cosine_phase_coefficient3, control_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask}; }
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        } // namespace phase_shift_detail

        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename Real, typename StateInteger, typename BitInteger>
        inline RandomAccessRange& cphase_shift3_tp(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state, Real const phase1, Real const phase2, Real const phase3,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_target_qubit,
          ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const permutated_control_qubit)
        {
          using complex_type = typename boost::range_value<RandomAccessRange>::type;
          static_assert(
            (std::is_same<Real, typename ::ket::utility::meta::real_of<complex_type>::type>::value),
            "Real must be the same to \"real part\" of value_type of RandomAccessIterator");

          using std::cos;
          using std::sin;
          using boost::math::constants::half;
          auto const sine = sin(half<Real>() * phase1);
          auto const cosine = cos(half<Real>() * phase1);

          auto const phase_coefficient2 = ::ket::utility::exp_i<complex_type>(phase2);
          auto const phase_coefficient3 = ::ket::utility::exp_i<complex_type>(phase3);

          auto const sine_phase_coefficient3 = sine * phase_coefficient3;
          auto const cosine_phase_coefficient3 = cosine * phase_coefficient3;

          assert(not ::ket::mpi::page::is_on_page(permutated_control_qubit, local_state));
          auto const control_qubit_mask
            = ::ket::utility::integer_exp2<StateInteger>(permutated_control_qubit);
          auto const nonpage_lower_bits_mask = control_qubit_mask - StateInteger{1u};
          auto const nonpage_upper_bits_mask = compl nonpage_lower_bits_mask;

# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<1u>(
            parallel_policy, local_state, permutated_target_qubit,
            [sine, cosine, &phase_coefficient2, &sine_phase_coefficient3, &cosine_phase_coefficient3,
             control_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask](
              auto const zero_first, auto const one_first, StateInteger const index_wo_nonpage_qubit, int const)
            {
              auto const zero_index
                = ((index_wo_nonpage_qubit bitand nonpage_upper_bits_mask) << 1u)
                  bitor (index_wo_nonpage_qubit bitand nonpage_lower_bits_mask);
              auto const one_index = zero_index bitor control_qubit_mask;
              auto const control_on_iter = zero_first + one_index;
              auto const target_control_on_iter = one_first + one_index;
              auto const control_on_iter_value = *control_on_iter;

              *control_on_iter *= cosine;
              *control_on_iter -= sine_phase_coefficient3 * *target_control_on_iter;
              *target_control_on_iter *= cosine_phase_coefficient3;
              *target_control_on_iter += sine * control_on_iter_value;
              *target_control_on_iter *= phase_coefficient2;
            });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<1u>(
            parallel_policy, local_state, permutated_target_qubit,
            ::ket::mpi::gate::page::phase_shift_detail::make_cphase_shift3_tp(
              sine, cosine, phase_coefficient2, sine_phase_coefficient3, cosine_phase_coefficient3,
              control_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask));
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        }

        // cphase_shift3_cp: only control qubit is on page
        // CU3_{tc}(theta, theta', theta'') or C1U3_{tc}(theta, theta', theta'')
        // CU3_{1,2}(theta, theta', theta'') (a_{00} |00> + a_{01} |01> + a_{10} |10> + a{11} |11>)
        //   = a_{00} |00> + a_{01} |01> + (cos(theta/2) a_{10} - e^{i theta''} sin(theta/2) a_{11}) |10>
        //     + (e^{i theta'} sin(theta/2) a_{10} + e^{i(theta' + theta'')} cos(theta/2) a_{11}) |11>
        namespace phase_shift_detail
        {
# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
          template <typename Real, typename Complex, typename StateInteger>
          struct cphase_shift3_cp
          {
            Real sine_;
            Real cosine_;
            Complex const* phase_coefficient2_ptr_;
            Complex const* sine_phase_coefficient3_ptr_;
            Complex const* cosine_phase_coefficient3_ptr_;
            StateInteger target_qubit_mask_;
            StateInteger nonpage_lower_bits_mask_;
            StateInteger nonpage_upper_bits_mask_;

            cphase_shift3_cp(
              Real const sine, Real const cosine,
              Complex const& phase_coefficient2, Complex const& sine_phase_coefficient3, Complex const& cosine_phase_coefficient3,
              StateInteger const target_qubit_mask,
              StateInteger const nonpage_lower_bits_mask,
              StateInteger const nonpage_upper_bits_mask) noexcept
              : sine_{sine}, cosine_{cosine},
                phase_coefficient2_ptr_{std::addressof(phase_coefficient2)},
                sine_phase_coefficient3_ptr_{std::addressof(sine_phase_coefficient3)},
                cosine_phase_coefficient3_ptr_{std::addressof(cosine_phase_coefficient3)},
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
              auto const control_on_iter_value = *control_on_iter;

              *control_on_iter *= cosine_;
              *control_on_iter -= *sine_phase_coefficient3_ptr_ * *target_control_on_iter;
              *target_control_on_iter *= *cosine_phase_coefficient3_ptr_;
              *target_control_on_iter += sine_ * control_on_iter_value;
              *target_control_on_iter *= *phase_coefficient2_ptr_;
            }
          }; // struct cphase_shift3_cp<Real, Complex, StateInteger>

          template <typename Real, typename Complex, typename StateInteger>
          inline ::ket::mpi::gate::page::phase_shift_detail::cphase_shift3_cp<Real, Complex, StateInteger>
          make_cphase_shift3_cp(
            Real const sine, Real const cosine,
            Complex const& phase_coefficient2, Complex const& sine_phase_coefficient3, Complex const& cosine_phase_coefficient3,
            StateInteger const target_qubit_mask,
            StateInteger const nonpage_lower_bits_mask,
            StateInteger const nonpage_upper_bits_mask)
          { return {sine, cosine, phase_coefficient2, sine_phase_coefficient3, cosine_phase_coefficient3, target_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask}; }
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        } // namespace phase_shift_detail

        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename Real, typename StateInteger, typename BitInteger>
        inline RandomAccessRange& cphase_shift3_cp(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state, Real const phase1, Real const phase2, Real const phase3,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_target_qubit,
          ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const permutated_control_qubit)
        {
          using complex_type = typename boost::range_value<RandomAccessRange>::type;
          static_assert(
            (std::is_same<Real, typename ::ket::utility::meta::real_of<complex_type>::type>::value),
            "Real must be the same to \"real part\" of value_type of RandomAccessIterator");

          using std::cos;
          using std::sin;
          using boost::math::constants::half;
          auto const sine = sin(half<Real>() * phase1);
          auto const cosine = cos(half<Real>() * phase1);

          auto const phase_coefficient2 = ::ket::utility::exp_i<complex_type>(phase2);
          auto const phase_coefficient3 = ::ket::utility::exp_i<complex_type>(phase3);

          auto const sine_phase_coefficient3 = sine * phase_coefficient3;
          auto const cosine_phase_coefficient3 = cosine * phase_coefficient3;

          assert(not ::ket::mpi::page::is_on_page(permutated_target_qubit, local_state));

          auto const target_qubit_mask
            = ::ket::utility::integer_exp2<StateInteger>(permutated_target_qubit);
          auto const nonpage_lower_bits_mask = target_qubit_mask - StateInteger{1u};
          auto const nonpage_upper_bits_mask = compl nonpage_lower_bits_mask;

# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<1u>(
            parallel_policy, local_state, permutated_control_qubit,
            [sine, cosine, &phase_coefficient2, &sine_phase_coefficient3, &cosine_phase_coefficient3,
             target_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask](
              auto const, auto const one_first, StateInteger const index_wo_nonpage_qubit, int const)
            {
              auto const zero_index
                = ((index_wo_nonpage_qubit bitand nonpage_upper_bits_mask) << 1u)
                  bitor (index_wo_nonpage_qubit bitand nonpage_lower_bits_mask);
              auto const one_index = zero_index bitor target_qubit_mask;
              auto const control_on_iter = one_first + zero_index;
              auto const target_control_on_iter = one_first + one_index;
              auto const control_on_iter_value = *control_on_iter;

              *control_on_iter *= cosine;
              *control_on_iter -= sine_phase_coefficient3 * *target_control_on_iter;
              *target_control_on_iter *= cosine_phase_coefficient3;
              *target_control_on_iter += sine * control_on_iter_value;
              *target_control_on_iter *= phase_coefficient2;
            });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<1u>(
            parallel_policy, local_state, permutated_control_qubit,
            ::ket::mpi::gate::page::phase_shift_detail::make_cphase_shift3_cp(
              sine, cosine, phase_coefficient2, sine_phase_coefficient3, cosine_phase_coefficient3,
              target_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask));
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        }

        namespace phase_shift_detail
        {
# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
          template <typename Real, typename Complex>
          struct adj_phase_shift3
          {
            Real sine_;
            Real cosine_;
            Complex sine_phase_coefficient2_;
            Complex cosine_phase_coefficient2_;
            Complex phase_coefficient3_;

            adj_phase_shift3(
              Real const sine, Real const cosine,
              Complex const& sine_phase_coefficient2,
              Complex const& cosine_phase_coefficient2,
              Complex const& phase_coefficient3)
              : sine_{sine},
                cosine_{cosine},
                sine_phase_coefficient2_{sine_phase_coefficient2},
                cosine_phase_coefficient2_{cosine_phase_coefficient2},
                phase_coefficient3_{phase_coefficient3}
            { }

            template <typename Iterator, typename StateInteger>
            void operator()(Iterator const zero_first, Iterator const one_first, StateInteger const index, int const) const
            {
              auto const zero_iter = zero_first + index;
              auto const one_iter = one_first + index;
              auto const zero_iter_value = *zero_iter;

              *zero_iter *= cosine_;
              *zero_iter += sine_phase_coefficient2_ * *one_iter;
              *one_iter *= cosine_phase_coefficient2_;
              *one_iter -= sine_ * zero_iter_value;
              *one_iter *= phase_coefficient3_;
            }
          }; // struct adj_phase_shift3<Real, Complex>

          template <typename Real, typename Complex>
          inline ::ket::mpi::gate::page::phase_shift_detail::adj_phase_shift3<Real, Complex>
          make_adj_phase_shift3(
            Real const sine, Real const cosine,
            Complex const& sine_phase_coefficient2,
            Complex const& cosine_phase_coefficient2,
            Complex const& phase_coefficient3)
          {
            return {
              sine, cosine, sine_phase_coefficient2, cosine_phase_coefficient2,
              phase_coefficient3};
          }
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        } // namespace phase_shift_detail

        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename Real, typename StateInteger, typename BitInteger>
        inline RandomAccessRange& adj_phase_shift3(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          Real const phase1, Real const phase2, Real const phase3,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_qubit)
        {
          using std::cos;
          using std::sin;
          using boost::math::constants::half;
          auto const sine = sin(half<Real>() * phase1);
          auto const cosine = cos(half<Real>() * phase1);

          using complex_type = typename boost::range_value<RandomAccessRange>::type;
          auto const phase_coefficient2 = ::ket::utility::exp_i<complex_type>(-phase2);
          auto const phase_coefficient3 = ::ket::utility::exp_i<complex_type>(-phase3);

          auto const sine_phase_coefficient2 = sine * phase_coefficient2;
          auto const cosine_phase_coefficient2 = cosine * phase_coefficient2;

# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<0u>(
            parallel_policy, local_state, permutated_qubit,
            [sine, cosine, sine_phase_coefficient2, cosine_phase_coefficient2,
             phase_coefficient3](
              auto const zero_first, auto const one_first, StateInteger const index, int const)
            {
              auto const zero_iter = zero_first + index;
              auto const one_iter = one_first + index;
              auto const zero_iter_value = *zero_iter;

              *zero_iter *= cosine;
              *zero_iter += sine_phase_coefficient2 * *one_iter;
              *one_iter *= cosine_phase_coefficient2;
              *one_iter -= sine * zero_iter_value;
              *one_iter *= phase_coefficient3;
            });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<0u>(
            parallel_policy, local_state, permutated_qubit,
            ::ket::mpi::gate::page::phase_shift_detail::make_adj_phase_shift3(
              sine, cosine, sine_phase_coefficient2, cosine_phase_coefficient2,
              phase_coefficient3));
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        }

        // adj_cphase_shift3_tcp: both of target and control qubits of Adj(CPhase) are on page
        // CU3+_{tc}(theta, theta', theta'') or C1U3+_{tc}(theta, theta', theta'')
        // CU3+_{1,2}(theta, theta', theta'') (a_{00} |00> + a_{01} |01> + a_{10} |10> + a{11} |11>)
        //   = a_{00} |00> + a_{01} |01> + (cos(theta/2) a_{10} + e^{-i theta'} sin(theta/2) a_{11}) |10>
        //     + (-e^{-i theta''} sin(theta/2) a_{10} + e^{-i(theta' + theta'')} cos(theta/2) a_{11}) |11>
        namespace phase_shift_detail
        {
# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
          template <typename Real, typename Complex>
          struct adj_cphase_shift3_tcp
          {
            Real sine_;
            Real cosine_;
            Complex const* sine_phase_coefficient2_ptr_;
            Complex const* cosine_phase_coefficient2_ptr_;
            Complex const* phase_coefficient3_ptr_;

            adj_cphase_shift3_tcp(
              Real const sine, Real const cosine,
              Complex const& sine_phase_coefficient2, Complex const& cosine_phase_coefficient2,
              Complex const& phase_coefficient3) noexcept
              : sine_{sine}, cosine_{cosine},
                sine_phase_coefficient2_ptr_{std::addressof(sine_phase_coefficient2)},
                cosine_phase_coefficient2_ptr_{std::addressof(cosine_phase_coefficient2)},
                phase_coefficient3_ptr_{std::addressof(phase_coefficient3)}
            { }

            template <typename Iterator, typename StateInteger>
            void operator()(
              Iterator const, Iterator const, Iterator const first_10, Iterator const first_11,
              StateInteger const index, int const) const
            {
              auto const control_on_iter = first_10 + index;
              auto const target_control_on_iter = first_11 + index;
              auto const control_on_iter_value = *control_on_iter;

              *control_on_iter *= cosine_;
              *control_on_iter += *sine_phase_coefficient2_ptr_ * *target_control_on_iter;
              *target_control_on_iter *= *cosine_phase_coefficient2_ptr_;
              *target_control_on_iter -= sine_ * control_on_iter_value;
              *target_control_on_iter *= *phase_coefficient3_ptr_;
            }
          }; // struct adj_cphase_shift3_tcp<Real, Complex>

          template <typename Real, typename Complex>
          inline ::ket::mpi::gate::page::phase_shift_detail::adj_cphase_shift3_tcp<Real, Complex>
          make_adj_cphase_shift3_tcp(
            Real const sine, Real const cosine,
            Complex const& sine_phase_coefficient2, Complex const& cosine_phase_coefficient2,
            Complex const& phase_coefficient3)
          { return {sine, cosine, sine_phase_coefficient2, cosine_phase_coefficient2, phase_coefficient3}; }
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        } // namespace phase_shift_detail

        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename Real, typename StateInteger, typename BitInteger>
        inline RandomAccessRange& adj_cphase_shift3_tcp(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state, Real const phase1, Real const phase2, Real const phase3,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_target_qubit,
          ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const permutated_control_qubit)
        {
          using complex_type = typename boost::range_value<RandomAccessRange>::type;
          static_assert(
            (std::is_same<Real, typename ::ket::utility::meta::real_of<complex_type>::type>::value),
            "Real must be the same to \"real part\" of value_type of RandomAccessIterator");

          using std::cos;
          using std::sin;
          using boost::math::constants::half;
          auto const sine = sin(half<Real>() * phase1);
          auto const cosine = cos(half<Real>() * phase1);

          auto const phase_coefficient2 = ::ket::utility::exp_i<complex_type>(-phase2);
          auto const phase_coefficient3 = ::ket::utility::exp_i<complex_type>(-phase3);

          auto const sine_phase_coefficient2 = sine * phase_coefficient2;
          auto const cosine_phase_coefficient2 = cosine * phase_coefficient2;

# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::two_page_qubits_gate<0u>(
            parallel_policy, local_state, permutated_target_qubit, permutated_control_qubit,
            [sine, cosine, &sine_phase_coefficient2, &cosine_phase_coefficient2, &phase_coefficient3](
              auto const, auto const, auto const first_10, auto const first_11, StateInteger const index, int const)
            {
              auto const control_on_iter = first_10 + index;
              auto const target_control_on_iter = first_11 + index;
              auto const control_on_iter_value = *control_on_iter;

              *control_on_iter *= cosine;
              *control_on_iter += sine_phase_coefficient2 * *target_control_on_iter;
              *target_control_on_iter *= cosine_phase_coefficient2;
              *target_control_on_iter -= sine * control_on_iter_value;
              *target_control_on_iter *= phase_coefficient3;
            });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::two_page_qubits_gate<0u>(
            parallel_policy, local_state, permutated_target_qubit, permutated_control_qubit,
            ::ket::mpi::gate::page::phase_shift_detail::make_adj_cphase_shift3_tcp(
              sine, cosine, sine_phase_coefficient2, cosine_phase_coefficient2, phase_coefficient3));
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        }

        // adj_cphase_shift3_tp: only target qubit is on page
        // CU3+_{tc}(theta, theta', theta'') or C1U3+_{tc}(theta, theta', theta'')
        // CU3+_{1,2}(theta, theta', theta'') (a_{00} |00> + a_{01} |01> + a_{10} |10> + a{11} |11>)
        //   = a_{00} |00> + a_{01} |01> + (cos(theta/2) a_{10} + e^{-i theta'} sin(theta/2) a_{11}) |10>
        //     + (-e^{-i theta''} sin(theta/2) a_{10} + e^{-i(theta' + theta'')} cos(theta/2) a_{11}) |11>
        namespace phase_shift_detail
        {
# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
          template <typename Real, typename Complex, typename StateInteger>
          struct adj_cphase_shift3_tp
          {
            Real sine_;
            Real cosine_;
            Complex const* sine_phase_coefficient2_ptr_;
            Complex const* cosine_phase_coefficient2_ptr_;
            Complex const* phase_coefficient3_ptr_;
            StateInteger control_qubit_mask_;
            StateInteger nonpage_lower_bits_mask_;
            StateInteger nonpage_upper_bits_mask_;

            adj_cphase_shift3_tp(
              Real const sine, Real const cosine,
              Complex const& sine_phase_coefficient2, Complex const& cosine_phase_coefficient2,
              Complex const& phase_coefficient3,
              StateInteger const control_qubit_mask,
              StateInteger const nonpage_lower_bits_mask,
              StateInteger const nonpage_upper_bits_mask) noexcept
              : sine_{sine}, cosine_{cosine},
                sine_phase_coefficient2_ptr_{std::addressof(sine_phase_coefficient2)},
                cosine_phase_coefficient2_ptr_{std::addressof(cosine_phase_coefficient2)},
                phase_coefficient3_ptr_{std::addressof(phase_coefficient3)},
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
              auto const control_on_iter_value = *control_on_iter;

              *control_on_iter *= cosine_;
              *control_on_iter += *sine_phase_coefficient2_ptr_ * *target_control_on_iter;
              *target_control_on_iter *= *cosine_phase_coefficient2_ptr_;
              *target_control_on_iter -= sine_ * control_on_iter_value;
              *target_control_on_iter *= *phase_coefficient3_ptr_;
            }
          }; // struct adj_cphase_shift3_tp<Real, Complex, StateInteger>

          template <typename Real, typename Complex, typename StateInteger>
          inline ::ket::mpi::gate::page::phase_shift_detail::adj_cphase_shift3_tp<Real, Complex, StateInteger>
          make_adj_cphase_shift3_tp(
            Real const sine, Real const cosine,
            Complex const& sine_phase_coefficient2, Complex const& cosine_phase_coefficient2,
            Complex const& phase_coefficient3,
            StateInteger const control_qubit_mask,
            StateInteger const nonpage_lower_bits_mask,
            StateInteger const nonpage_upper_bits_mask)
          { return {sine, cosine, sine_phase_coefficient2, cosine_phase_coefficient2, phase_coefficient3, control_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask}; }
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        } // namespace phase_shift_detail

        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename Real, typename StateInteger, typename BitInteger>
        inline RandomAccessRange& adj_cphase_shift3_tp(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state, Real const phase1, Real const phase2, Real const phase3,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_target_qubit,
          ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const permutated_control_qubit)
        {
          using complex_type = typename boost::range_value<RandomAccessRange>::type;
          static_assert(
            (std::is_same<Real, typename ::ket::utility::meta::real_of<complex_type>::type>::value),
            "Real must be the same to \"real part\" of value_type of RandomAccessIterator");

          using std::cos;
          using std::sin;
          using boost::math::constants::half;
          auto const sine = sin(half<Real>() * phase1);
          auto const cosine = cos(half<Real>() * phase1);

          auto const phase_coefficient2 = ::ket::utility::exp_i<complex_type>(-phase2);
          auto const phase_coefficient3 = ::ket::utility::exp_i<complex_type>(-phase3);

          auto const sine_phase_coefficient2 = sine * phase_coefficient2;
          auto const cosine_phase_coefficient2 = cosine * phase_coefficient2;

          assert(not ::ket::mpi::page::is_on_page(permutated_control_qubit, local_state));
          auto const control_qubit_mask
            = ::ket::utility::integer_exp2<StateInteger>(permutated_control_qubit);
          auto const nonpage_lower_bits_mask = control_qubit_mask - StateInteger{1u};
          auto const nonpage_upper_bits_mask = compl nonpage_lower_bits_mask;

# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<1u>(
            parallel_policy, local_state, permutated_target_qubit,
            [sine, cosine, &sine_phase_coefficient2, &cosine_phase_coefficient2, &phase_coefficient3,
             control_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask](
              auto const zero_first, auto const one_first, StateInteger const index_wo_nonpage_qubit, int const)
            {
              auto const zero_index
                = ((index_wo_nonpage_qubit bitand nonpage_upper_bits_mask) << 1u)
                  bitor (index_wo_nonpage_qubit bitand nonpage_lower_bits_mask);
              auto const one_index = zero_index bitor control_qubit_mask;
              auto const control_on_iter = zero_first + one_index;
              auto const target_control_on_iter = one_first + one_index;
              auto const control_on_iter_value = *control_on_iter;

              *control_on_iter *= cosine;
              *control_on_iter += sine_phase_coefficient2 * *target_control_on_iter;
              *target_control_on_iter *= cosine_phase_coefficient2;
              *target_control_on_iter -= sine * control_on_iter_value;
              *target_control_on_iter *= phase_coefficient3;
            });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<1u>(
            parallel_policy, local_state, permutated_target_qubit,
            ::ket::mpi::gate::page::phase_shift_detail::make_adj_cphase_shift3_tp(
              sine, cosine, sine_phase_coefficient2, cosine_phase_coefficient2, phase_coefficient3,
              control_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask));
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        }

        // adj_cphase_shift3_cp: only control qubit is on page
        // CU3+_{tc}(theta, theta', theta'') or C1U3+_{tc}(theta, theta', theta'')
        // CU3+_{1,2}(theta, theta', theta'') (a_{00} |00> + a_{01} |01> + a_{10} |10> + a{11} |11>)
        //   = a_{00} |00> + a_{01} |01> + (cos(theta/2) a_{10} + e^{-i theta'} sin(theta/2) a_{11}) |10>
        //     + (-e^{-i theta''} sin(theta/2) a_{10} + e^{-i(theta' + theta'')} cos(theta/2) a_{11}) |11>
        namespace phase_shift_detail
        {
# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
          template <typename Real, typename Complex, typename StateInteger>
          struct adj_cphase_shift3_cp
          {
            Real sine_;
            Real cosine_;
            Complex const* sine_phase_coefficient2_ptr_;
            Complex const* cosine_phase_coefficient2_ptr_;
            Complex const* phase_coefficient3_ptr_;
            StateInteger target_qubit_mask_;
            StateInteger nonpage_lower_bits_mask_;
            StateInteger nonpage_upper_bits_mask_;

            adj_cphase_shift3_cp(
              Real const sine, Real const cosine,
              Complex const& sine_phase_coefficient2, Complex const& cosine_phase_coefficient2,
              Complex const& phase_coefficient3,
              StateInteger const target_qubit_mask,
              StateInteger const nonpage_lower_bits_mask,
              StateInteger const nonpage_upper_bits_mask) noexcept
              : sine_{sine}, cosine_{cosine},
                sine_phase_coefficient2_ptr_{std::addressof(sine_phase_coefficient2)},
                cosine_phase_coefficient2_ptr_{std::addressof(cosine_phase_coefficient2)},
                phase_coefficient3_ptr_{std::addressof(phase_coefficient3)},
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
              auto const control_on_iter_value = *control_on_iter;

              *control_on_iter *= cosine_;
              *control_on_iter += *sine_phase_coefficient2_ptr_ * *target_control_on_iter;
              *target_control_on_iter *= *cosine_phase_coefficient2_ptr_;
              *target_control_on_iter -= sine_ * control_on_iter_value;
              *target_control_on_iter *= *phase_coefficient3_ptr_;
            }
          }; // struct adj_cphase_shift3_cp<Real, Complex, StateInteger>

          template <typename Real, typename Complex, typename StateInteger>
          inline ::ket::mpi::gate::page::phase_shift_detail::adj_cphase_shift3_cp<Real, Complex, StateInteger>
          make_adj_cphase_shift3_cp(
            Real const sine, Real const cosine,
            Complex const& sine_phase_coefficient2, Complex const& cosine_phase_coefficient2,
            Complex const& phase_coefficient3,
            StateInteger const target_qubit_mask,
            StateInteger const nonpage_lower_bits_mask,
            StateInteger const nonpage_upper_bits_mask)
          { return {sine, cosine, sine_phase_coefficient2, cosine_phase_coefficient2, phase_coefficient3, target_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask}; }
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        } // namespace phase_shift_detail

        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename Real, typename StateInteger, typename BitInteger>
        inline RandomAccessRange& adj_cphase_shift3_cp(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state, Real const phase1, Real const phase2, Real const phase3,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_target_qubit,
          ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const permutated_control_qubit)
        {
          using complex_type = typename boost::range_value<RandomAccessRange>::type;
          static_assert(
            (std::is_same<Real, typename ::ket::utility::meta::real_of<complex_type>::type>::value),
            "Real must be the same to \"real part\" of value_type of RandomAccessIterator");

          using std::cos;
          using std::sin;
          using boost::math::constants::half;
          auto const sine = sin(half<Real>() * phase1);
          auto const cosine = cos(half<Real>() * phase1);

          auto const phase_coefficient2 = ::ket::utility::exp_i<complex_type>(-phase2);
          auto const phase_coefficient3 = ::ket::utility::exp_i<complex_type>(-phase3);

          auto const sine_phase_coefficient2 = sine * phase_coefficient2;
          auto const cosine_phase_coefficient2 = cosine * phase_coefficient2;

          assert(not ::ket::mpi::page::is_on_page(permutated_target_qubit, local_state));

          auto const target_qubit_mask
            = ::ket::utility::integer_exp2<StateInteger>(permutated_target_qubit);
          auto const nonpage_lower_bits_mask = target_qubit_mask - StateInteger{1u};
          auto const nonpage_upper_bits_mask = compl nonpage_lower_bits_mask;

# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<1u>(
            parallel_policy, local_state, permutated_control_qubit,
            [sine, cosine, &sine_phase_coefficient2, &cosine_phase_coefficient2, &phase_coefficient3,
             target_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask](
              auto const, auto const one_first, StateInteger const index_wo_nonpage_qubit, int const)
            {
              auto const zero_index
                = ((index_wo_nonpage_qubit bitand nonpage_upper_bits_mask) << 1u)
                  bitor (index_wo_nonpage_qubit bitand nonpage_lower_bits_mask);
              auto const one_index = zero_index bitor target_qubit_mask;
              auto const control_on_iter = one_first + zero_index;
              auto const target_control_on_iter = one_first + one_index;
              auto const control_on_iter_value = *control_on_iter;

              *control_on_iter *= cosine;
              *control_on_iter += sine_phase_coefficient2 * *target_control_on_iter;
              *target_control_on_iter *= cosine_phase_coefficient2;
              *target_control_on_iter -= sine * control_on_iter_value;
              *target_control_on_iter *= phase_coefficient3;
            });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<1u>(
            parallel_policy, local_state, permutated_control_qubit,
            ::ket::mpi::gate::page::phase_shift_detail::make_adj_cphase_shift3_cp(
              sine, cosine, sine_phase_coefficient2, cosine_phase_coefficient2, phase_coefficient3,
              target_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask));
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        }
      } // namespace page
    } // namespace gate
  } // namespace mpi
} // namespace ket


#endif // KET_MPI_GATE_PAGE_PHASE_SHIFT_DIAGONAL_HPP
