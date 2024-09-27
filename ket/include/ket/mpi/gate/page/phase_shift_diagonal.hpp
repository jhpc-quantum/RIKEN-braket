#ifndef KET_MPI_GATE_PAGE_PHASE_SHIFT_DIAGONAL_HPP
# define KET_MPI_GATE_PAGE_PHASE_SHIFT_DIAGONAL_HPP

# include <cassert>
# include <cmath>
# include <type_traits>
# include <memory>

# include <boost/math/constants/constants.hpp>

# include <yampi/rank.hpp>

# include <ket/qubit.hpp>
# include <ket/control.hpp>
# include <ket/utility/integer_exp2.hpp>
# include <ket/utility/exp_i.hpp>
# include <ket/utility/meta/real_of.hpp>
# include <ket/utility/meta/ranges.hpp>
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
        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename Complex, typename StateInteger, typename BitInteger>
        inline auto phase_shift_coeff(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          Complex const& phase_coefficient,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_qubit)
        -> RandomAccessRange&
        {
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<0u>(
            parallel_policy, local_state, permutated_qubit,
            [phase_coefficient](auto const, auto const one_first, StateInteger const index, int const)
            { *(one_first + index) *= phase_coefficient; });
        }

        // cphase_shift_coeff_tcp: both of target and control qubits of CPhase(coeff) are on page
        // CU1_{tc}(theta) or C1U1_{tc}(theta)
        // CU1_{1,2}(theta) (a_{00} |00> + a_{01} |01> + a_{10} |10> + a{11} |11>)
        //   = a_{00} |00> + a_{01} |01> + a_{10} |10> + e^{i theta} a_{11} |11>
        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename Complex, typename StateInteger, typename BitInteger>
        inline auto cphase_shift_coeff_tcp(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state, Complex const& phase_coefficient,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_target_qubit,
          ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const permutated_control_qubit)
        -> RandomAccessRange&
        {
          return ::ket::mpi::gate::page::detail::two_page_qubits_gate<0u>(
            parallel_policy, local_state, permutated_target_qubit, permutated_control_qubit,
            [&phase_coefficient](auto const, auto const, auto const, auto const first_11, StateInteger const index, int const)
            { *(first_11 + index) *= phase_coefficient; });
        }

        // cphase_shift_coeff_tp: only target qubit is on page
        // CU1_{tc}(theta) or C1U1_{tc}(theta)
        // CU1_{1,2}(theta) (a_{00} |00> + a_{01} |01> + a_{10} |10> + a{11} |11>)
        //   = a_{00} |00> + a_{01} |01> + a_{10} |10> + e^{i theta} a_{11} |11>
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename Complex, typename StateInteger, typename BitInteger>
        inline auto cphase_shift_coeff_tp(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          Complex const& phase_coefficient,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_target_qubit,
          ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const permutated_control_qubit,
          yampi::rank const rank)
        -> RandomAccessRange&
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
        inline auto cphase_shift_coeff_cp(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          Complex const& phase_coefficient,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_target_qubit,
          ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const permutated_control_qubit,
          yampi::rank const rank)
        -> RandomAccessRange&
        {
          return ::ket::mpi::gate::page::detail::cphase_shift_coeff_cp(
            mpi_policy, parallel_policy, local_state,
            phase_coefficient, permutated_target_qubit, permutated_control_qubit, rank);
        }

        // generalized phase_shift
        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename Real, typename StateInteger, typename BitInteger>
        inline auto phase_shift2(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          Real const phase1, Real const phase2,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_qubit)
        -> RandomAccessRange&
        {
          using complex_type = ::ket::utility::meta::range_value_t<RandomAccessRange>;
          auto const phase_coefficient1 = ::ket::utility::exp_i<complex_type>(phase1);
          auto const phase_coefficient2 = ::ket::utility::exp_i<complex_type>(phase2);

          using boost::math::constants::one_div_root_two;
          auto const modified_phase_coefficient1 = one_div_root_two<Real>() * phase_coefficient1;

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
        }

        // cphase_shift2_tcp: both of target and control qubits of CPhase are on page
        // CU2_{tc}(theta, theta') or C1U2_{tc}(theta, theta')
        // CU2_{1,2}(theta, theta') (a_{00} |00> + a_{01} |01> + a_{10} |10> + a{11} |11>)
        //   = a_{00} |00> + a_{01} |01> + (a_{10} - e^{i theta'} a_{11})/sqrt(2) |10> + (e^{i theta} a_{10} + e^{i(theta + theta')} a_{11})/sqrt(2) |11>
        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename Real, typename StateInteger, typename BitInteger>
        inline auto cphase_shift2_tcp(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state, Real const phase1, Real const phase2,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_target_qubit,
          ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const permutated_control_qubit)
        -> RandomAccessRange&
        {
          using complex_type = ::ket::utility::meta::range_value_t<RandomAccessRange>;
          static_assert(
            std::is_same<Real, ::ket::utility::meta::real_t<complex_type>>::value,
            "Real must be the same to \"real part\" of value_type of RandomAccessIterator");

          auto const phase_coefficient1 = ::ket::utility::exp_i<complex_type>(phase1);
          auto const phase_coefficient2 = ::ket::utility::exp_i<complex_type>(phase2);

          using boost::math::constants::one_div_root_two;
          auto const modified_phase_coefficient1 = one_div_root_two<Real>() * phase_coefficient1;

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
        }

        // cphase_shift2_tp: only target qubit is on page
        // CU2_{tc}(theta, theta') or C1U2_{tc}(theta, theta')
        // CU2_{1,2}(theta, theta') (a_{00} |00> + a_{01} |01> + a_{10} |10> + a{11} |11>)
        //   = a_{00} |00> + a_{01} |01> + (a_{10} - e^{i theta'} a_{11})/sqrt(2) |10> + (e^{i theta} a_{10} + e^{i(theta + theta')} a_{11})/sqrt(2) |11>
        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename Real, typename StateInteger, typename BitInteger>
        inline auto cphase_shift2_tp(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state, Real const phase1, Real const phase2,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_target_qubit,
          ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const permutated_control_qubit)
        -> RandomAccessRange&
        {
          using complex_type = ::ket::utility::meta::range_value_t<RandomAccessRange>;
          static_assert(
            std::is_same<Real, ::ket::utility::meta::real_t<complex_type>>::value,
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
        }

        // cphase_shift2_cp: only control qubit is on page
        // CU2_{tc}(theta, theta') or C1U2_{tc}(theta, theta')
        // CU2_{1,2}(theta, theta') (a_{00} |00> + a_{01} |01> + a_{10} |10> + a{11} |11>)
        //   = a_{00} |00> + a_{01} |01> + (a_{10} - e^{i theta'} a_{11})/sqrt(2) |10> + (e^{i theta} a_{10} + e^{i(theta + theta')} a_{11})/sqrt(2) |11>
        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename Real, typename StateInteger, typename BitInteger>
        inline auto cphase_shift2_cp(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state, Real const phase1, Real const phase2,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_target_qubit,
          ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const permutated_control_qubit)
        -> RandomAccessRange&
        {
          using complex_type = ::ket::utility::meta::range_value_t<RandomAccessRange>;
          static_assert(
            std::is_same<Real, ::ket::utility::meta::real_t<complex_type>>::value,
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
        }

        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename Real, typename StateInteger, typename BitInteger>
        inline auto adj_phase_shift2(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          Real const phase1, Real const phase2,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_qubit)
        -> RandomAccessRange&
        {
          using complex_type = ::ket::utility::meta::range_value_t<RandomAccessRange>;
          auto const phase_coefficient1 = ::ket::utility::exp_i<complex_type>(-phase1);
          auto const phase_coefficient2 = ::ket::utility::exp_i<complex_type>(-phase2);

          using boost::math::constants::one_div_root_two;
          auto const modified_phase_coefficient2 = one_div_root_two<Real>() * phase_coefficient2;

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
        }

        // adj_cphase_shift2_tcp: both of target and control qubits of Adj(CPhase) are on page
        // CU2+_{tc}(theta, theta') or C1U2+_{tc}(theta, theta')
        // CU2+_{1,2}(theta, theta') (a_{00} |00> + a_{01} |01> + a_{10} |10> + a{11} |11>)
        //   = a_{00} |00> + a_{01} |01> + (a_{10} + e^{-i theta} a_{11})/sqrt(2) |10> + (-e^{-i theta'} a_{10} + e^{-i(theta + theta')} a_{11})/sqrt(2) |11>
        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename Real, typename StateInteger, typename BitInteger>
        inline auto adj_cphase_shift2_tcp(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state, Real const phase1, Real const phase2,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_target_qubit,
          ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const permutated_control_qubit)
        -> RandomAccessRange&
        {
          using complex_type = ::ket::utility::meta::range_value_t<RandomAccessRange>;
          static_assert(
            std::is_same<Real, ::ket::utility::meta::real_t<complex_type>>::value,
            "Real must be the same to \"real part\" of value_type of RandomAccessIterator");

          auto const phase_coefficient1 = ::ket::utility::exp_i<complex_type>(-phase1);
          auto const phase_coefficient2 = ::ket::utility::exp_i<complex_type>(-phase2);

          using boost::math::constants::one_div_root_two;
          auto const modified_phase_coefficient2 = one_div_root_two<Real>() * phase_coefficient2;

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
        }

        // adj_cphase_shift2_tp: only target qubit is on page
        // CU2+_{tc}(theta, theta') or C1U2+_{tc}(theta, theta')
        // CU2+_{1,2}(theta, theta') (a_{00} |00> + a_{01} |01> + a_{10} |10> + a{11} |11>)
        //   = a_{00} |00> + a_{01} |01> + (a_{10} + e^{-i theta} a_{11})/sqrt(2) |10> + (-e^{-i theta'} a_{10} + e^{-i(theta + theta')} a_{11})/sqrt(2) |11>
        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename Real, typename StateInteger, typename BitInteger>
        inline auto adj_cphase_shift2_tp(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state, Real const phase1, Real const phase2,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_target_qubit,
          ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const permutated_control_qubit)
        -> RandomAccessRange&
        {
          using complex_type = ::ket::utility::meta::range_value_t<RandomAccessRange>;
          static_assert(
            std::is_same<Real, ::ket::utility::meta::real_t<complex_type>>::value,
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
        }

        // adj_cphase_shift2_cp: only control qubit is on page
        // CU2+_{tc}(theta, theta') or C1U2+_{tc}(theta, theta')
        // CU2+_{1,2}(theta, theta') (a_{00} |00> + a_{01} |01> + a_{10} |10> + a{11} |11>)
        //   = a_{00} |00> + a_{01} |01> + (a_{10} + e^{-i theta} a_{11})/sqrt(2) |10> + (-e^{-i theta'} a_{10} + e^{-i(theta + theta')} a_{11})/sqrt(2) |11>
        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename Real, typename StateInteger, typename BitInteger>
        inline auto adj_cphase_shift2_cp(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state, Real const phase1, Real const phase2,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_target_qubit,
          ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const permutated_control_qubit)
        -> RandomAccessRange&
        {
          using complex_type = ::ket::utility::meta::range_value_t<RandomAccessRange>;
          static_assert(
            std::is_same<Real, ::ket::utility::meta::real_t<complex_type>>::value,
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
        }

        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename Real, typename StateInteger, typename BitInteger>
        inline auto phase_shift3(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          Real const phase1, Real const phase2, Real const phase3,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_qubit)
        -> RandomAccessRange&
        {
          using std::cos;
          using std::sin;
          using boost::math::constants::half;
          auto const sine = sin(half<Real>() * phase1);
          auto const cosine = cos(half<Real>() * phase1);

          using complex_type = ::ket::utility::meta::range_value_t<RandomAccessRange>;
          auto const phase_coefficient2 = ::ket::utility::exp_i<complex_type>(phase2);
          auto const phase_coefficient3 = ::ket::utility::exp_i<complex_type>(phase3);

          auto const sine_phase_coefficient3 = sine * phase_coefficient3;
          auto const cosine_phase_coefficient3 = cosine * phase_coefficient3;

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
        }

        // cphase_shift3_tcp: both of target and control qubits of CPhase are on page
        // CU3_{tc}(theta, theta', theta'') or C1U3_{tc}(theta, theta', theta'')
        // CU3_{1,2}(theta, theta', theta'') (a_{00} |00> + a_{01} |01> + a_{10} |10> + a{11} |11>)
        //   = a_{00} |00> + a_{01} |01> + (cos(theta/2) a_{10} - e^{i theta''} sin(theta/2) a_{11}) |10>
        //     + (e^{i theta'} sin(theta/2) a_{10} + e^{i(theta' + theta'')} cos(theta/2) a_{11}) |11>
        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename Real, typename StateInteger, typename BitInteger>
        inline auto cphase_shift3_tcp(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state, Real const phase1, Real const phase2, Real const phase3,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_target_qubit,
          ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const permutated_control_qubit)
        -> RandomAccessRange&
        {
          using complex_type = ::ket::utility::meta::range_value_t<RandomAccessRange>;
          static_assert(
            std::is_same<Real, ::ket::utility::meta::real_t<complex_type>>::value,
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
        }

        // cphase_shift3_tp: only target qubit is on page
        // CU3_{tc}(theta, theta', theta'') or C1U3_{tc}(theta, theta', theta'')
        // CU3_{1,2}(theta, theta', theta'') (a_{00} |00> + a_{01} |01> + a_{10} |10> + a{11} |11>)
        //   = a_{00} |00> + a_{01} |01> + (cos(theta/2) a_{10} - e^{i theta''} sin(theta/2) a_{11}) |10>
        //     + (e^{i theta'} sin(theta/2) a_{10} + e^{i(theta' + theta'')} cos(theta/2) a_{11}) |11>
        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename Real, typename StateInteger, typename BitInteger>
        inline auto cphase_shift3_tp(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state, Real const phase1, Real const phase2, Real const phase3,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_target_qubit,
          ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const permutated_control_qubit)
        -> RandomAccessRange&
        {
          using complex_type = ::ket::utility::meta::range_value_t<RandomAccessRange>;
          static_assert(
            std::is_same<Real, ::ket::utility::meta::real_t<complex_type>>::value,
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
        }

        // cphase_shift3_cp: only control qubit is on page
        // CU3_{tc}(theta, theta', theta'') or C1U3_{tc}(theta, theta', theta'')
        // CU3_{1,2}(theta, theta', theta'') (a_{00} |00> + a_{01} |01> + a_{10} |10> + a{11} |11>)
        //   = a_{00} |00> + a_{01} |01> + (cos(theta/2) a_{10} - e^{i theta''} sin(theta/2) a_{11}) |10>
        //     + (e^{i theta'} sin(theta/2) a_{10} + e^{i(theta' + theta'')} cos(theta/2) a_{11}) |11>
        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename Real, typename StateInteger, typename BitInteger>
        inline auto cphase_shift3_cp(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state, Real const phase1, Real const phase2, Real const phase3,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_target_qubit,
          ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const permutated_control_qubit)
        -> RandomAccessRange&
        {
          using complex_type = ::ket::utility::meta::range_value_t<RandomAccessRange>;
          static_assert(
            std::is_same<Real, ::ket::utility::meta::real_t<complex_type>>::value,
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
        }

        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename Real, typename StateInteger, typename BitInteger>
        inline auto adj_phase_shift3(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          Real const phase1, Real const phase2, Real const phase3,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_qubit)
        -> RandomAccessRange&
        {
          using std::cos;
          using std::sin;
          using boost::math::constants::half;
          auto const sine = sin(half<Real>() * phase1);
          auto const cosine = cos(half<Real>() * phase1);

          using complex_type = ::ket::utility::meta::range_value_t<RandomAccessRange>;
          auto const phase_coefficient2 = ::ket::utility::exp_i<complex_type>(-phase2);
          auto const phase_coefficient3 = ::ket::utility::exp_i<complex_type>(-phase3);

          auto const sine_phase_coefficient2 = sine * phase_coefficient2;
          auto const cosine_phase_coefficient2 = cosine * phase_coefficient2;

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
        }

        // adj_cphase_shift3_tcp: both of target and control qubits of Adj(CPhase) are on page
        // CU3+_{tc}(theta, theta', theta'') or C1U3+_{tc}(theta, theta', theta'')
        // CU3+_{1,2}(theta, theta', theta'') (a_{00} |00> + a_{01} |01> + a_{10} |10> + a{11} |11>)
        //   = a_{00} |00> + a_{01} |01> + (cos(theta/2) a_{10} + e^{-i theta'} sin(theta/2) a_{11}) |10>
        //     + (-e^{-i theta''} sin(theta/2) a_{10} + e^{-i(theta' + theta'')} cos(theta/2) a_{11}) |11>
        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename Real, typename StateInteger, typename BitInteger>
        inline auto adj_cphase_shift3_tcp(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state, Real const phase1, Real const phase2, Real const phase3,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_target_qubit,
          ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const permutated_control_qubit)
        -> RandomAccessRange&
        {
          using complex_type = ::ket::utility::meta::range_value_t<RandomAccessRange>;
          static_assert(
            std::is_same<Real, ::ket::utility::meta::real_t<complex_type>>::value,
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
        }

        // adj_cphase_shift3_tp: only target qubit is on page
        // CU3+_{tc}(theta, theta', theta'') or C1U3+_{tc}(theta, theta', theta'')
        // CU3+_{1,2}(theta, theta', theta'') (a_{00} |00> + a_{01} |01> + a_{10} |10> + a{11} |11>)
        //   = a_{00} |00> + a_{01} |01> + (cos(theta/2) a_{10} + e^{-i theta'} sin(theta/2) a_{11}) |10>
        //     + (-e^{-i theta''} sin(theta/2) a_{10} + e^{-i(theta' + theta'')} cos(theta/2) a_{11}) |11>
        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename Real, typename StateInteger, typename BitInteger>
        inline auto adj_cphase_shift3_tp(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state, Real const phase1, Real const phase2, Real const phase3,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_target_qubit,
          ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const permutated_control_qubit)
        -> RandomAccessRange&
        {
          using complex_type = ::ket::utility::meta::range_value_t<RandomAccessRange>;
          static_assert(
            std::is_same<Real, ::ket::utility::meta::real_t<complex_type>>::value,
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
        }

        // adj_cphase_shift3_cp: only control qubit is on page
        // CU3+_{tc}(theta, theta', theta'') or C1U3+_{tc}(theta, theta', theta'')
        // CU3+_{1,2}(theta, theta', theta'') (a_{00} |00> + a_{01} |01> + a_{10} |10> + a{11} |11>)
        //   = a_{00} |00> + a_{01} |01> + (cos(theta/2) a_{10} + e^{-i theta'} sin(theta/2) a_{11}) |10>
        //     + (-e^{-i theta''} sin(theta/2) a_{10} + e^{-i(theta' + theta'')} cos(theta/2) a_{11}) |11>
        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename Real, typename StateInteger, typename BitInteger>
        inline auto adj_cphase_shift3_cp(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state, Real const phase1, Real const phase2, Real const phase3,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_target_qubit,
          ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const permutated_control_qubit)
        -> RandomAccessRange&
        {
          using complex_type = ::ket::utility::meta::range_value_t<RandomAccessRange>;
          static_assert(
            std::is_same<Real, ::ket::utility::meta::real_t<complex_type>>::value,
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
        }
      } // namespace page
    } // namespace gate
  } // namespace mpi
} // namespace ket


#endif // KET_MPI_GATE_PAGE_PHASE_SHIFT_DIAGONAL_HPP
