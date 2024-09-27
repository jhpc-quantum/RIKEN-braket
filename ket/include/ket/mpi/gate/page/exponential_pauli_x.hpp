#ifndef KET_MPI_GATE_PAGE_EXPONENTIAL_PAULI_X_HPP
# define KET_MPI_GATE_PAGE_EXPONENTIAL_PAULI_X_HPP

# include <cassert>
# include <cmath>
# include <memory>

# include <ket/qubit.hpp>
# include <ket/control.hpp>
# include <ket/utility/integer_exp2.hpp>
# include <ket/utility/imaginary_unit.hpp>
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
        // 1_p: the qubit of eX is on page
        // eX_i(theta) = exp(i theta X_i) = I cos(theta) + i X_i sin(theta)
        // eX_1(theta) (a_0 |0> + a_1 |1>) = (cos(theta) a_0 + i sin(theta) a_1) |0> + (i sin(theta) a_0 + cos(theta) a_1) |1>
        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename Complex, typename StateInteger, typename BitInteger>
        inline auto exponential_pauli_x_coeff1(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_qubit)
        -> RandomAccessRange&
        {
          assert(::ket::mpi::page::is_on_page(permutated_qubit, local_state));

          using std::real;
          using std::imag;
          auto const cos_theta = real(phase_coefficient);
          auto const i_sin_theta = ::ket::utility::imaginary_unit<Complex>() * imag(phase_coefficient);

          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<0u>(
            parallel_policy, local_state, permutated_qubit,
            [cos_theta, &i_sin_theta](
              auto const zero_first, auto const one_first, StateInteger const index, int const)
            {
              auto const zero_iter = zero_first + index;
              auto const one_iter = one_first + index;
              auto const zero_iter_value = *zero_iter;

              *zero_iter *= cos_theta;
              *zero_iter += *one_iter * i_sin_theta;
              *one_iter *= cos_theta;
              *one_iter += zero_iter_value * i_sin_theta;
            });
        }

        // 2_2p: both of qubits of eXX are on page
        // eXX_{ij}(theta) = exp(i theta X_i X_j) = I cos(theta) + i X_i X_j sin(theta)
        // eXX_{1,2}(theta) (a_{00} |00> + a_{01} |01> + a_{10} |10> + a{11} |11>)
        //   = (cos(theta) a_{00} + i sin(theta) a_{11}) |00> + (cos(theta) a_{01} + i sin(theta) a_{10}) |01>
        //     + (i sin(theta) a_{01} + cos(theta) a_{10}) |10> + (i sin(theta) a_{00} + cos(theta) a_{11}) |11>
        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename Complex, typename StateInteger, typename BitInteger>
        inline auto exponential_pauli_x_coeff2_2p(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const page_permutated_qubit1,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const page_permutated_qubit2)
        -> RandomAccessRange&
        {
          using std::real;
          using std::imag;
          auto const cos_theta = real(phase_coefficient);
          auto const i_sin_theta = ::ket::utility::imaginary_unit<Complex>() * imag(phase_coefficient);

          return ::ket::mpi::gate::page::detail::two_page_qubits_gate<0u>(
            parallel_policy, local_state, page_permutated_qubit1, page_permutated_qubit2,
            [cos_theta, &i_sin_theta](
              auto const first_00, auto const first_01, auto const first_10, auto const first_11,
              StateInteger const index, int const)
            {
              auto const iter_00 = first_00 + index;
              auto const iter_01 = first_01 + index;
              auto const iter_10 = first_10 + index;
              auto const iter_11 = first_11 + index;
              auto const value_00 = *iter_00;
              auto const value_01 = *iter_01;

              *iter_00 *= cos_theta;
              *iter_00 += *iter_11 * i_sin_theta;
              *iter_01 *= cos_theta;
              *iter_01 += *iter_10 * i_sin_theta;
              *iter_10 *= cos_theta;
              *iter_10 += value_01 * i_sin_theta;
              *iter_11 *= cos_theta;
              *iter_11 += value_00 * i_sin_theta;
            });
        }

        // 2_p: only one qubit of eXX is on page
        // eXX_{ij}(theta) = exp(i theta X_i X_j) = I cos(theta) + i X_i X_j sin(theta)
        // eXX_{1,2}(theta) (a_{00} |00> + a_{01} |01> + a_{10} |10> + a{11} |11>)
        //   = (cos(theta) a_{00} + i sin(theta) a_{11}) |00> + (cos(theta) a_{01} + i sin(theta) a_{10}) |01>
        //     + (i sin(theta) a_{01} + cos(theta) a_{10}) |10> + (i sin(theta) a_{00} + cos(theta) a_{11}) |11>
        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename Complex, typename StateInteger, typename BitInteger>
        inline auto exponential_pauli_x_coeff2_p(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const page_permutated_qubit,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const nonpage_permutated_qubit)
        -> RandomAccessRange&
        {
          assert(::ket::mpi::page::is_on_page(page_permutated_qubit, local_state));
          assert(not ::ket::mpi::page::is_on_page(nonpage_permutated_qubit, local_state));

          auto const nonpage_permutated_qubit_mask
            = ::ket::utility::integer_exp2<StateInteger>(nonpage_permutated_qubit);
          auto const nonpage_lower_bits_mask = nonpage_permutated_qubit_mask - StateInteger{1u};
          auto const nonpage_upper_bits_mask = compl nonpage_lower_bits_mask;

          using std::real;
          using std::imag;
          auto const cos_theta = real(phase_coefficient);
          auto const i_sin_theta = ::ket::utility::imaginary_unit<Complex>() * imag(phase_coefficient);

          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<1u>(
            parallel_policy, local_state, page_permutated_qubit,
            [cos_theta, &i_sin_theta, nonpage_permutated_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask](
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
              auto const value_00 = *iter_00;
              auto const value_01_or_10 = *iter_01_or_10;

              *iter_00 *= cos_theta;
              *iter_00 += *iter_11 * i_sin_theta;
              *iter_01_or_10 *= cos_theta;
              *iter_01_or_10 += *iter_10_or_01 * i_sin_theta;
              *iter_10_or_01 *= cos_theta;
              *iter_10_or_01 += value_01_or_10 * i_sin_theta;
              *iter_11 *= cos_theta;
              *iter_11 += value_00 * i_sin_theta;
            });
        }

        // cx_coeff_tcp: both of target and control qubits of CeX are on page
        // CeX_{tc}(theta) = C[exp(i theta X_t)]_c = C[I cos(theta) + i X_t sin(theta)]_c, C1eX_{tc}(theta), CeX1_{tc}(theta), or C1eX1_{tc}(theta)
        // CeX_{1,2}(theta) (a_{00} |00> + a_{01} |01> + a_{10} |10> + a{11} |11>)
        //   = a_{00} |00> + a_{01} |01> + (cos(theta) a_{10} + i sin(theta) a_{11}) |10> + (i sin(theta) a_{10} + cos(theta) a_{11}) |11>
        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename Complex, typename StateInteger, typename BitInteger>
        inline auto exponential_pauli_cx_coeff_tcp(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state, Complex const& phase_coefficient,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_target_qubit,
          ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const permutated_control_qubit)
        -> RandomAccessRange&
        {
          assert(::ket::mpi::page::is_on_page(permutated_target_qubit, local_state));
          assert(::ket::mpi::page::is_on_page(permutated_control_qubit, local_state));

          using std::real;
          using std::imag;
          auto const cos_theta = real(phase_coefficient);
          auto const i_sin_theta = ::ket::utility::imaginary_unit<Complex>() * imag(phase_coefficient);

          return ::ket::mpi::gate::page::detail::two_page_qubits_gate<0u>(
            parallel_policy, local_state, permutated_target_qubit, permutated_control_qubit,
            [cos_theta, &i_sin_theta](
              auto const, auto const, auto const first_10, auto const first_11, StateInteger const index, int const)
            {
              auto const control_on_iter = first_10 + index;
              auto const target_control_on_iter = first_11 + index;
              auto const control_on_iter_value = *control_on_iter;

              *control_on_iter *= cos_theta;
              *control_on_iter += *target_control_on_iter * i_sin_theta;
              *target_control_on_iter *= cos_theta;
              *target_control_on_iter += control_on_iter_value * i_sin_theta;
            });
        }

        // cx_coeff_tp: only target qubit is on page
        // CeX_{tc}(theta) = C[exp(i theta X_t)]_c = C[I cos(theta) + i X_t sin(theta)]_c, C1eX_{tc}(theta), CeX1_{tc}(theta), or C1eX1_{tc}(theta)
        // CeX_{1,2}(theta) (a_{00} |00> + a_{01} |01> + a_{10} |10> + a{11} |11>)
        //   = a_{00} |00> + a_{01} |01> + (cos(theta) a_{10} + i sin(theta) a_{11}) |10> + (i sin(theta) a_{10} + cos(theta) a_{11}) |11>
        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename Complex, typename StateInteger, typename BitInteger>
        inline auto exponential_pauli_cx_coeff_tp(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state, Complex const& phase_coefficient,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_target_qubit,
          ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const permutated_control_qubit)
        -> RandomAccessRange&
        {
          using std::real;
          using std::imag;
          auto const cos_theta = real(phase_coefficient);
          auto const i_sin_theta = ::ket::utility::imaginary_unit<Complex>() * imag(phase_coefficient);

          assert(not ::ket::mpi::page::is_on_page(permutated_control_qubit, local_state));
          auto const control_qubit_mask
            = ::ket::utility::integer_exp2<StateInteger>(permutated_control_qubit);
          auto const nonpage_lower_bits_mask = control_qubit_mask - StateInteger{1u};
          auto const nonpage_upper_bits_mask = compl nonpage_lower_bits_mask;

          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<1u>(
            parallel_policy, local_state, permutated_target_qubit,
            [cos_theta, &i_sin_theta, control_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask](
              auto const zero_first, auto const one_first, StateInteger const index_wo_nonpage_qubit, int const)
            {
              auto const zero_index
                = ((index_wo_nonpage_qubit bitand nonpage_upper_bits_mask) << 1u)
                  bitor (index_wo_nonpage_qubit bitand nonpage_lower_bits_mask);
              auto const one_index = zero_index bitor control_qubit_mask;
              auto const control_on_iter = zero_first + one_index;
              auto const target_control_on_iter = one_first + one_index;
              auto const control_on_iter_value = *control_on_iter;

              *control_on_iter *= cos_theta;
              *control_on_iter += *target_control_on_iter * i_sin_theta;
              *target_control_on_iter *= cos_theta;
              *target_control_on_iter += control_on_iter_value * i_sin_theta;
            });
        }

        // cx_coeff_cp: only control qubit is on page
        // CeX_{tc}(theta) = C[exp(i theta X_t)]_c = C[I cos(theta) + i X_t sin(theta)]_c, C1eX_{tc}(theta), CeX1_{tc}(theta), or C1eX1_{tc}(theta)
        // CeX_{1,2}(theta) (a_{00} |00> + a_{01} |01> + a_{10} |10> + a{11} |11>)
        //   = a_{00} |00> + a_{01} |01> + (cos(theta) a_{10} + i sin(theta) a_{11}) |10> + (i sin(theta) a_{10} + cos(theta) a_{11}) |11>
        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename Complex, typename StateInteger, typename BitInteger>
        inline auto exponential_pauli_cx_coeff_cp(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state, Complex const& phase_coefficient,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_target_qubit,
          ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const permutated_control_qubit)
        -> RandomAccessRange&
        {
          using std::real;
          using std::imag;
          auto const cos_theta = real(phase_coefficient);
          auto const i_sin_theta = ::ket::utility::imaginary_unit<Complex>() * imag(phase_coefficient);

          assert(not ::ket::mpi::page::is_on_page(permutated_target_qubit, local_state));

          auto const target_qubit_mask
            = ::ket::utility::integer_exp2<StateInteger>(permutated_target_qubit);
          auto const nonpage_lower_bits_mask = target_qubit_mask - StateInteger{1u};
          auto const nonpage_upper_bits_mask = compl nonpage_lower_bits_mask;

          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<1u>(
            parallel_policy, local_state, permutated_control_qubit,
            [cos_theta, &i_sin_theta, target_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask](
              auto const, auto const one_first, StateInteger const index_wo_nonpage_qubit, int const)
            {
              auto const zero_index
                = ((index_wo_nonpage_qubit bitand nonpage_upper_bits_mask) << 1u)
                  bitor (index_wo_nonpage_qubit bitand nonpage_lower_bits_mask);
              auto const one_index = zero_index bitor target_qubit_mask;
              auto const control_on_iter = one_first + zero_index;
              auto const target_control_on_iter = one_first + one_index;
              auto const control_on_iter_value = *control_on_iter;

              *control_on_iter *= cos_theta;
              *control_on_iter += *target_control_on_iter * i_sin_theta;
              *target_control_on_iter *= cos_theta;
              *target_control_on_iter += control_on_iter_value * i_sin_theta;
            });
        }
      } // namespace page
    } // namespace gate
  } // namespace mpi
} // namespace ket


#endif // KET_MPI_GATE_PAGE_EXPONENTIAL_PAULI_X_HPP
