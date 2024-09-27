#ifndef KET_MPI_GATE_PAGE_EXPONENTIAL_SWAP_HPP
# define KET_MPI_GATE_PAGE_EXPONENTIAL_SWAP_HPP

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
        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename Complex, typename StateInteger, typename BitInteger>
        inline auto exponential_swap_coeff_2p(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const page_permutated_qubit1,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const page_permutated_qubit2)
        -> RandomAccessRange&
        {
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
        }

        // p: only one qubit is on page
        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename Complex, typename StateInteger, typename BitInteger>
        inline auto exponential_swap_coeff_p(
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
        }
      } // namespace page
    } // namespace gate
  } // namespace mpi
} // namespace ket


#endif // KET_MPI_GATE_PAGE_EXPONENTIAL_SWAP_HPP
