#ifndef KET_MPI_GATE_PAGE_CONTROLLED_V_HPP
# define KET_MPI_GATE_PAGE_CONTROLLED_V_HPP

# include <cassert>

# include <boost/math/constants/constants.hpp>

# include <ket/qubit.hpp>
# include <ket/control.hpp>
# include <ket/utility/integer_exp2.hpp>
# include <ket/utility/meta/real_of.hpp>
# include <ket/mpi/permutated.hpp>
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
        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename Complex, typename StateInteger, typename BitInteger>
        inline auto controlled_v_coeff_tcp(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          Complex const& phase_coefficient,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_target_qubit,
          ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const permutated_control_qubit)
        -> RandomAccessRange&
        {
          using real_type = ::ket::utility::meta::real_t<Complex>;
          auto const one_plus_phase_coefficient = real_type{1} + phase_coefficient;
          auto const one_minus_phase_coefficient = real_type{1} - phase_coefficient;

          return ::ket::mpi::gate::page::detail::two_page_qubits_gate<0u>(
            parallel_policy, local_state, permutated_target_qubit, permutated_control_qubit,
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
        }

        // tp: only target qubit is on page
        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename Complex, typename StateInteger, typename BitInteger>
        inline auto controlled_v_coeff_tp(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          Complex const& phase_coefficient,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_target_qubit,
          ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const permutated_control_qubit)
        -> RandomAccessRange&
        {
          assert(not ::ket::mpi::page::is_on_page(permutated_control_qubit, local_state));

          using real_type = ::ket::utility::meta::real_t<Complex>;
          auto const one_plus_phase_coefficient = real_type{1} + phase_coefficient;
          auto const one_minus_phase_coefficient = real_type{1} - phase_coefficient;

          auto const permutated_control_qubit_mask
            = ::ket::utility::integer_exp2<StateInteger>(permutated_control_qubit);
          auto const nonpage_lower_bits_mask = permutated_control_qubit_mask - StateInteger{1u};
          auto const nonpage_upper_bits_mask = compl nonpage_lower_bits_mask;

          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<1u>(
            parallel_policy, local_state, permutated_target_qubit,
            [one_plus_phase_coefficient, one_minus_phase_coefficient,
             permutated_control_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask](
              auto const zero_first, auto const one_first, StateInteger const index_wo_nonpage_qubit, int const)
            {
              auto const zero_index
                = ((index_wo_nonpage_qubit bitand nonpage_upper_bits_mask) << 1u)
                  bitor (index_wo_nonpage_qubit bitand nonpage_lower_bits_mask);
              auto const one_index = zero_index bitor permutated_control_qubit_mask;

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
        }

        // cp: only control qubit is on page
        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename Complex, typename StateInteger, typename BitInteger>
        inline auto controlled_v_coeff_cp(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          Complex const& phase_coefficient,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_target_qubit,
          ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const permutated_control_qubit)
        -> RandomAccessRange&
        {
          assert(not ::ket::mpi::page::is_on_page(permutated_target_qubit, local_state));

          using real_type = ::ket::utility::meta::real_t<Complex>;
          auto const one_plus_phase_coefficient = real_type{1} + phase_coefficient;
          auto const one_minus_phase_coefficient = real_type{1} - phase_coefficient;

          auto const permutated_target_qubit_mask
            = ::ket::utility::integer_exp2<StateInteger>(permutated_target_qubit);
          auto const nonpage_lower_bits_mask = permutated_target_qubit_mask - StateInteger{1u};
          auto const nonpage_upper_bits_mask = compl nonpage_lower_bits_mask;

          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<1u>(
            parallel_policy, local_state, permutated_control_qubit,
            [one_plus_phase_coefficient, one_minus_phase_coefficient,
             permutated_target_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask](
              auto const, auto const one_first, StateInteger const index_wo_nonpage_qubit, int const)
            {
              auto const zero_index
                = ((index_wo_nonpage_qubit bitand nonpage_upper_bits_mask) << 1u)
                  bitor (index_wo_nonpage_qubit bitand nonpage_lower_bits_mask);
              auto const one_index = zero_index bitor permutated_target_qubit_mask;

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
        }
      } // namespace page
    } // namespace gate
  } // namespace mpi
} // namespace ket


#endif // KET_MPI_GATE_PAGE_CONTROLLED_V_HPP
