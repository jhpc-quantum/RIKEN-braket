#ifndef KET_MPI_GATE_PAGE_EXPONENTIAL_SWAP_HPP
# define KET_MPI_GATE_PAGE_EXPONENTIAL_SWAP_HPP

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
        // 2p: both of qubits are on page
        namespace exponential_swap_detail
        {
# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
          template <typename Complex>
          struct exponential_swap_coeff_2p
          {
            Complex phase_coefficient_;

            exponential_swap_coeff_2p(Complex const& phase_coefficient) noexcept
              : phase_coefficient_{phase_coefficient}
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
              auto const value_01 = *iter_01;

              *iter_00 *= phase_coefficient_;
              *iter_11 *= phase_coefficient_;

              using std::real;
              using std::imag;
              *iter_01 = real(phase_coefficient_) * value_01 + ::ket::utility::imaginary_unit<Complex>() * imag(phase_coefficient_) * *iter_10;
              *iter_10 = ::ket::utility::imaginary_unit<Complex>() * imag(phase_coefficient_) * value_01 + real(phase_coefficient_) * *iter_10;
            }
          }; // struct exponential_swap_coeff_2p<Complex>

          template <typename Complex>
          inline ::ket::mpi::gate::page::exponential_swap_detail::exponential_swap_coeff_2p<Complex>
          make_exponential_swap_coeff_2p(Complex const& phase_coefficient)
          { return {phase_coefficient}; }
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        } // namespace exponential_swap_detail

        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename Complex, typename StateInteger, typename BitInteger>
        inline RandomAccessRange& exponential_swap_coeff_2p(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const page_permutated_qubit1,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const page_permutated_qubit2)
        {
          using real_type = typename ::ket::utility::meta::real_of<Complex>::type;

# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::two_page_qubits_gate<0u>(
            parallel_policy, local_state, page_permutated_qubit1, page_permutated_qubit2,
            [&phase_coefficient](
              auto const first_00, auto const first_01, auto const first_10, auto const first_11,
              StateInteger const index, int const)
            {
              auto const iter_00 = first_00 + index;
              auto const iter_01 = first_01 + index;
              auto const iter_10 = first_10 + index;
              auto const iter_11 = first_11 + index;
              auto const value_01 = *iter_01;

              *iter_00 *= phase_coefficient;
              *iter_11 *= phase_coefficient;

              using std::real;
              using std::imag;
              *iter_01 = real(phase_coefficient) * value_01 + ::ket::utility::imaginary_unit<Complex>() * imag(phase_coefficient) * *iter_10;
              *iter_10 = ::ket::utility::imaginary_unit<Complex>() * imag(phase_coefficient) * value_01 + real(phase_coefficient) * *iter_10;
            });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::two_page_qubits_gate<0u>(
            parallel_policy, local_state, page_permutated_qubit1, page_permutated_qubit2,
            ::ket::mpi::gate::page::exponential_swap_detail::make_exponential_swap_coeff_2p(phase_coefficient));
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        }

        // p: only one qubit is on page
        namespace exponential_swap_detail
        {
# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
          template <typename Complex, typename StateInteger>
          struct exponential_swap_coeff_p
          {
            Complex phase_coefficient_; // exp(i theta) = cos(theta) + i sin(theta)
            StateInteger nonpage_permutated_qubit_mask_;
            StateInteger nonpage_lower_bits_mask_;
            StateInteger nonpage_upper_bits_mask_;

            exponential_swap_coeff_p(
              Complex const& phase_coefficient,
              StateInteger const nonpage_permutated_qubit_mask,
              StateInteger const nonpage_lower_bits_mask,
              StateInteger const nonpage_upper_bits_mask) noexcept
              : phase_coefficient_{phase_coefficient},
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
              auto const iter_01 = zero_first + one_index;
              auto const iter_10 = one_first + zero_index;
              auto const iter_11 = one_first + one_index;
              auto const value_01 = *iter_01;

              *iter_00 *= phase_coefficient_;
              *iter_11 *= phase_coefficient_;

              using std::real;
              using std::imag;
              *iter_01 = real(phase_coefficient_) * value_01 + ::ket::utility::imaginary_unit<Complex>() * imag(phase_coefficient_) * *iter_10;
              *iter_10 = ::ket::utility::imaginary_unit<Complex>() * imag(phase_coefficient_) * value_01 + real(phase_coefficient_) * *iter_10;
            }
          }; // struct exponential_swap_coeff_p<Complex, StateInteger>

          template <typename Complex, typename StateInteger>
          inline ::ket::mpi::gate::page::exponential_swap_detail::exponential_swap_coeff_p<Complex, StateInteger>
          make_exponential_swap_coeff_p(
            Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
            StateInteger const nonpage_permutated_qubit_mask,
            StateInteger const nonpage_lower_bits_mask,
            StateInteger const nonpage_upper_bits_mask)
          { return {phase_coefficient, nonpage_permutated_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask}; }
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        } // namespace exponential_swap_detail

        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename Complex, typename StateInteger, typename BitInteger>
        inline RandomAccessRange& exponential_swap_coeff_p(
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

# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<1u>(
            parallel_policy, local_state, page_permutated_qubit,
            [&phase_coefficient, nonpage_permutated_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask](
              auto const zero_first, auto const one_first, StateInteger const index_wo_nonpage_qubit, int const)
            {
              auto const zero_index
                = ((index_wo_nonpage_qubit bitand nonpage_upper_bits_mask) << 1u)
                  bitor (index_wo_nonpage_qubit bitand nonpage_lower_bits_mask);
              auto const one_index = zero_index bitor nonpage_permutated_qubit_mask;

              auto const iter_00 = zero_first + zero_index;
              auto const iter_01 = zero_first + one_index;
              auto const iter_10 = one_first + zero_index;
              auto const iter_11 = one_first + one_index;
              auto const value_01 = *iter_01;

              *iter_00 *= phase_coefficient;
              *iter_11 *= phase_coefficient;

              using std::real;
              using std::imag;
              *iter_01 = real(phase_coefficient) * value_01 + ::ket::utility::imaginary_unit<Complex>() * imag(phase_coefficient) * *iter_10;
              *iter_10 = ::ket::utility::imaginary_unit<Complex>() * imag(phase_coefficient) * value_01 + real(phase_coefficient) * *iter_10;
            });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<1u>(
            parallel_policy, local_state, page_permutated_qubit,
            ::ket::mpi::gate::page::exponential_swap_detail::make_exponential_swap_coeff_p(
              phase_coefficient, nonpage_permutated_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask));
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        }
      } // namespace page
    } // namespace gate
  } // namespace mpi
} // namespace ket


#endif // KET_MPI_GATE_PAGE_EXPONENTIAL_SWAP_HPP
