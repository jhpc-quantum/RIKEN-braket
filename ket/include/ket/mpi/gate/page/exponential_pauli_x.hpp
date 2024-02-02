#ifndef KET_MPI_GATE_PAGE_EXPONENTIAL_PAULI_X_HPP
# define KET_MPI_GATE_PAGE_EXPONENTIAL_PAULI_X_HPP

# include <boost/config.hpp>

# include <cassert>
# include <cmath>

# include <ket/qubit.hpp>
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
        // eX_i(s) = exp(is X_i) = I cos s + i X_i sin s
        // eX_1(s) (a_0 |0> + a_1 |1>) = (cos s a_0 + i sin s a_1) |0> + (i sin s a_0 + cos s a_1) |1>
        namespace exponential_pauli_x_detail
        {
# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
          template <typename Real, typename Complex>
          struct exponential_pauli_x_coeff1
          {
            Real cos_theta_;
            Complex i_sin_theta_;

            exponential_pauli_x_coeff1(Real const cos_theta, Complex const& i_sin_theta) noexcept
              : cos_theta_{cos_theta}, i_sin_theta_{i_sin_theta}
            { }

            template <typename Iterator, typename StateInteger>
            void operator()(Iterator const zero_first, Iterator const one_first, StateInteger const index, int const) const
            {
              auto const zero_iter = zero_first + index;
              auto const one_iter = one_first + index;
              auto const zero_iter_value = *zero_iter;

              *zero_iter *= cos_theta_;
              *zero_iter += *one_iter * i_sin_theta_;
              *one_iter *= cos_theta_;
              *one_iter += zero_iter_value * i_sin_theta_;
            }
          }; // struct exponential_pauli_x_coeff1<Real, Complex>

          template <typename Real, typename Complex>
          inline ::ket::mpi::gate::page::exponential_pauli_x_detail::exponential_pauli_x_coeff1<Real, Complex>
          make_exponential_pauli_x_coeff1(Real const cos_theta, Complex const& i_sin_theta)
          { return {cos_theta, i_sin_theta}; }
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        } // namespace exponential_pauli_x_detail

        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename Complex, typename StateInteger, typename BitInteger>
        inline RandomAccessRange& exponential_pauli_x_coeff1(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_qubit)
        {
          assert(::ket::mpi::page::is_on_page(permutated_qubit, local_state));

          using std::real;
          using std::imag;
          auto const cos_theta = real(phase_coefficient);
          auto const i_sin_theta = ::ket::utility::imaginary_unit<Complex>() * imag(phase_coefficient);

# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
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
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<1u>(
            parallel_policy, local_state, permutated_qubit,
            ::ket::mpi::gate::page::exponential_pauli_x_detail::make_exponential_pauli_x_coeff1(cos_theta, i_sin_theta));
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        }

        // 2_2p: both of qubits of eXX are on page
        // eXX_{ij}(s) = exp(is X_i X_j) = I cos s + i X_i X_j sin s
        // eXX_{1,2}(s) (a_{00} |00> + a_{01} |01> + a_{10} |10> + a{11} |11>)
        //   = (cos s a_{00} + i sin s a_{11}) |00> + (cos s a_{01} + i sin s a_{10}) |01> + (i sin s a_{01} + cos s a_{10}) |10> + (i sin s a_{00} + cos s a_{11}) |11>
        namespace exponential_pauli_x_detail
        {
# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
          template <typename Real, typename Complex>
          struct exponential_pauli_x_coeff2_2p
          {
            Real cos_theta_;
            Complex i_sin_theta_;

            exponential_pauli_x_coeff2_2p(Real const cos_theta, Complex const& i_sin_theta) noexcept
              : cos_theta_{cos_theta}, i_sin_theta_{i_sin_theta}
            { }

            template <typename Iterator, typename StateInteger>
            void operator()(
              Iterator const first_00, Iterator const first_01,
              Iterator const first_10, Iterator const first_11,
              StateInteger const index, int const) const
            {
              auto const iter_00 = first_00 + index;
              auto const iter_01 = first_01 + index;
              auto const iter_10 = first_10 + index;
              auto const iter_11 = first_11 + index;
              auto const value_00 = *iter_00;
              auto const value_01 = *iter_01;

              *iter_00 *= cos_theta_;
              *iter_00 += *iter_11 * i_sin_theta_;
              *iter_01 *= cos_theta_;
              *iter_01 += *iter_10 * i_sin_theta_;
              *iter_10 *= cos_theta_;
              *iter_10 += value_01 * i_sin_theta_;
              *iter_11 *= cos_theta_;
              *iter_11 += value_00 * i_sin_theta_;
            }
          }; // struct exponential_pauli_x_coeff2_2p<Real, Complex>

          template <typename Real, typename Complex>
          inline ::ket::mpi::gate::page::exponential_pauli_x_detail::exponential_pauli_x_coeff2_2p<Real, Complex>
          make_exponential_pauli_x_coeff2_2p(Real const cos_theta, Complex const& i_sin_theta)
          { return {cos_theta, i_sin_theta}; }
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        } // namespace exponential_pauli_x_detail

        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename Complex, typename StateInteger, typename BitInteger>
        inline RandomAccessRange& exponential_pauli_x_coeff2_2p(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const page_permutated_qubit1,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const page_permutated_qubit2)
        {
          using std::real;
          using std::imag;
          auto const cos_theta = real(phase_coefficient);
          auto const i_sin_theta = ::ket::utility::imaginary_unit<Complex>() * imag(phase_coefficient);

# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
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
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::two_page_qubits_gate<0u>(
            parallel_policy, local_state, page_permutated_qubit1, page_permutated_qubit2,
            ::ket::mpi::gate::page::exponential_pauli_x_detail::make_exponential_pauli_x_coeff2_2p(cos_theta, i_sin_theta));
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        }

        // 2_p: only one qubit of eXX is on page
        // eXX_{ij}(s) = exp(is X_i X_j) = I cos s + i X_i X_j sin s
        // eXX_{1,2}(s) (a_{00} |00> + a_{01} |01> + a_{10} |10> + a{11} |11>)
        //   = (cos s a_{00} + i sin s a_{11}) |00> + (cos s a_{01} + i sin s a_{10}) |01> + (i sin s a_{01} + cos s a_{10}) |10> + (i sin s a_{00} + cos s a_{11}) |11>
        namespace exponential_pauli_x_detail
        {
# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
          template <typename Real, typename Complex, typename StateInteger>
          struct exponential_pauli_x_coeff2_p
          {
            Real cos_theta_;
            Complex i_sin_theta_;
            StateInteger nonpage_permutated_qubit_mask_;
            StateInteger nonpage_lower_bits_mask_;
            StateInteger nonpage_upper_bits_mask_;

            exponential_pauli_x_coeff2_p(
              Real const cos_theta, Complex const& i_sin_theta,
              StateInteger const nonpage_permutated_qubit_mask,
              StateInteger const nonpage_lower_bits_mask,
              StateInteger const nonpage_upper_bits_mask) noexcept
              : cos_theta_{cos_theta}, i_sin_theta_{i_sin_theta},
                nonpage_permutated_qubit_mask_{nonpage_permutated_qubit_mask},
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
              auto const value_00 = *iter_00;
              auto const value_01_or_10 = *iter_01_or_10;

              *iter_00 *= cos_theta_;
              *iter_00 += *iter_11 * i_sin_theta_;
              *iter_01_or_10 *= cos_theta_;
              *iter_01_or_10 += *iter_10_or_01 * i_sin_theta_;
              *iter_10_or_01 *= cos_theta_;
              *iter_10_or_01 += value_01_or_10 * i_sin_theta_;
              *iter_11 *= cos_theta_;
              *iter_11 += value_00 * i_sin_theta_;
            }
          }; // struct exponential_pauli_x_coeff2_p<Real, Complex, StateInteger>

          template <typename Real, typename Complex, typename StateInteger>
          inline ::ket::mpi::gate::page::exponential_pauli_x_detail::exponential_pauli_x_coeff2_p<Real, Complex, StateInteger>
          make_exponential_pauli_x_coeff2_p(
            Real const cos_theta, Complex const& i_sin_theta,
            StateInteger const nonpage_permutated_qubit_mask,
            StateInteger const nonpage_lower_bits_mask,
            StateInteger const nonpage_upper_bits_mask)
          { return {cos_theta, i_sin_theta, nonpage_permutated_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask}; }
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        } // namespace exponential_pauli_x_detail

        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename Complex, typename StateInteger, typename BitInteger>
        inline RandomAccessRange& exponential_pauli_x_coeff2_p(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const page_permutated_qubit,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const nonpage_permutated_qubit)
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

# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
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
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<1u>(
            parallel_policy, local_state, page_permutated_qubit,
            ::ket::mpi::gate::page::exponential_pauli_x_detail::make_exponential_pauli_x_coeff2_p(
              cos_theta, i_sin_theta, nonpage_permutated_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask));
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        }
      } // namespace page
    } // namespace gate
  } // namespace mpi
} // namespace ket


#endif // KET_MPI_GATE_PAGE_EXPONENTIAL_PAULI_X_HPP
