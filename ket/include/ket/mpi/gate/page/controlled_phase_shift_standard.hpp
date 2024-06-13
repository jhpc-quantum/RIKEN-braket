#ifndef KET_MPI_GATE_PAGE_CONTROLLED_PHASE_SHIFT_STANDARD_HPP
# define KET_MPI_GATE_PAGE_CONTROLLED_PHASE_SHIFT_STANDARD_HPP

# include <boost/config.hpp>

# include <cassert>

# include <ket/qubit.hpp>
# include <ket/control.hpp>
# include <ket/utility/integer_exp2.hpp>
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
        namespace controlled_phase_shift_detail
        {
# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
          // [[deprecated]]
          template <typename Complex>
          struct controlled_phase_shift_coeff_tcp
          {
            Complex phase_coefficient_;

            explicit controlled_phase_shift_coeff_tcp(Complex const& phase_coefficient) noexcept
              : phase_coefficient_{phase_coefficient}
            { }

            template <typename Iterator, typename StateInteger>
            void operator()(
              Iterator const, Iterator const, Iterator const, Iterator const first_11,
              StateInteger const index, int const) const
            { *(first_11 + index) *= phase_coefficient_; }
          }; // struct controlled_phase_shift_coeff_tcp<Complex>

          // [[deprecated]]
          template <typename Complex>
          inline ::ket::mpi::gate::page::controlled_phase_shift_detail::controlled_phase_shift_coeff_tcp<Complex>
          make_controlled_phase_shift_coeff_tcp(Complex const& phase_coefficient)
          { return ::ket::mpi::gate::page::controlled_phase_shift_detail::controlled_phase_shift_coeff_tcp<Complex>{phase_coefficient}; }
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        } // namespace controlled_phase_shift_detail

        // [[deprecated]]
        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename Complex, typename StateInteger, typename BitInteger>
        inline RandomAccessRange& controlled_phase_shift_coeff_tcp(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          Complex const& phase_coefficient,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_target_qubit,
          ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const permutated_control_qubit)
        {
# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::two_page_qubits_gate<0u>(
            parallel_policy, local_state, permutated_target_qubit, permutated_control_qubit,
            [phase_coefficient](
              auto const, auto const, auto const, auto const first_11,
              StateInteger const index, int const)
            { *(first_11 + index) *= phase_coefficient; });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::two_page_qubits_gate<0u>(
            parallel_policy, local_state, permutated_target_qubit, permutated_control_qubit,
            ::ket::mpi::gate::page::controlled_phase_shift_detail::make_controlled_phase_shift_coeff_tcp(phase_coefficient));
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        }

        // tp: only target qubit is on page
        namespace controlled_phase_shift_detail
        {
# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
          // [[deprecated]]
          template <typename Complex, typename StateInteger>
          struct controlled_phase_shift_coeff_tp
          {
            Complex phase_coefficient_;
            StateInteger permutated_control_qubit_mask_;
            StateInteger nonpage_lower_bits_mask_;
            StateInteger nonpage_upper_bits_mask_;

            controlled_phase_shift_coeff_tp(
              Complex const& phase_coefficient,
              StateInteger const permutated_control_qubit_mask,
              StateInteger const nonpage_lower_bits_mask,
              StateInteger const nonpage_upper_bits_mask) noexcept
              : phase_coefficient_{phase_coefficient},
                permutated_control_qubit_mask_{permutated_control_qubit_mask},
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
              auto const one_index = zero_index bitor permutated_control_qubit_mask_;
              *(one_first + one_index) *= phase_coefficient_;
            }
          }; // struct controlled_phase_shift_coeff_tp<Complex, StateInteger>

          // [[deprecated]]
          template <typename Complex, typename StateInteger>
          inline ::ket::mpi::gate::page::controlled_phase_shift_detail::controlled_phase_shift_coeff_tp<Complex, StateInteger>
          make_controlled_phase_shift_coeff_tp(
            Complex const& phase_coefficient,
            StateInteger const permutated_control_qubit_mask,
            StateInteger const nonpage_lower_bits_mask,
            StateInteger const nonpage_upper_bits_mask)
          { return {phase_coefficient, permutated_control_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask}; }
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        } // namespace controlled_phase_shift_detail

        // [[deprecated]]
        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename Complex, typename StateInteger, typename BitInteger>
        inline RandomAccessRange& controlled_phase_shift_coeff_tp(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          Complex const& phase_coefficient,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_target_qubit,
          ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const permutated_control_qubit)
        {
          assert(not ::ket::mpi::page::is_on_page(permutated_control_qubit, local_state));
          auto const permutated_control_qubit_mask
            = ::ket::utility::integer_exp2<StateInteger>(permutated_control_qubit);
          auto const nonpage_lower_bits_mask = permutated_control_qubit_mask - StateInteger{1u};
          auto const nonpage_upper_bits_mask = compl nonpage_lower_bits_mask;

# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<1u>(
            parallel_policy, local_state, permutated_target_qubit,
            [phase_coefficient, permutated_control_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask](
              auto const, auto const one_first, StateInteger const index_wo_nonpage_qubit, int const)
            {
              auto const zero_index
                = ((index_wo_nonpage_qubit bitand nonpage_upper_bits_mask) << 1u)
                  bitor (index_wo_nonpage_qubit bitand nonpage_lower_bits_mask);
              auto const one_index = zero_index bitor permutated_control_qubit_mask;
              *(one_first + one_index) *= phase_coefficient;
            });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<1u>(
            parallel_policy, local_state, permutated_target_qubit,
            ::ket::mpi::gate::page::controlled_phase_shift_detail::make_controlled_phase_shift_coeff_tp(
              phase_coefficient,
              permutated_control_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask));
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        }

        // cp: only control qubit is on page
        namespace controlled_phase_shift_detail
        {
# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
          // [[deprecated]]
          template <typename Complex, typename StateInteger>
          struct controlled_phase_shift_coeff_cp
          {
            Complex phase_coefficient_;
            StateInteger permutated_target_qubit_mask_;
            StateInteger nonpage_lower_bits_mask_;
            StateInteger nonpage_upper_bits_mask_;

            controlled_phase_shift_coeff_cp(
              Complex const& phase_coefficient,
              StateInteger const permutated_target_qubit_mask,
              StateInteger const nonpage_lower_bits_mask,
              StateInteger const nonpage_upper_bits_mask) noexcept
              : phase_coefficient_{phase_coefficient},
                permutated_target_qubit_mask_{permutated_target_qubit_mask},
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
              auto const one_index = zero_index bitor permutated_target_qubit_mask_;
              *(one_first + one_index) *= phase_coefficient_;
            }
          }; // struct controlled_phase_shift_coeff_cp<Complex, StateInteger>

          // [[deprecated]]
          template <typename Complex, typename StateInteger>
          inline ::ket::mpi::gate::page::controlled_phase_shift_detail::controlled_phase_shift_coeff_cp<Complex, StateInteger>
          make_controlled_phase_shift_coeff_cp(
            Complex const& phase_coefficient,
            StateInteger const permutated_target_qubit_mask,
            StateInteger const nonpage_lower_bits_mask,
            StateInteger const nonpage_upper_bits_mask)
          { return {phase_coefficient, permutated_target_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask}; }
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        } // namespace controlled_phase_shift_detail

        // [[deprecated]]
        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename Complex, typename StateInteger, typename BitInteger>
        inline RandomAccessRange& controlled_phase_shift_coeff_cp(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          Complex const& phase_coefficient,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_target_qubit,
          ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const permutated_control_qubit)
        {
          assert(not ::ket::mpi::page::is_on_page(permutated_target_qubit, local_state));
          auto const permutated_target_qubit_mask
            = ::ket::utility::integer_exp2<StateInteger>(permutated_target_qubit);
          auto const nonpage_lower_bits_mask = permutated_target_qubit_mask - StateInteger{1u};
          auto const nonpage_upper_bits_mask = compl nonpage_lower_bits_mask;

# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<1u>(
            parallel_policy, local_state, permutated_control_qubit,
            [phase_coefficient, permutated_target_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask](
              auto const, auto const one_first, StateInteger const index_wo_nonpage_qubit, int const)
            {
              auto const zero_index
                = ((index_wo_nonpage_qubit bitand nonpage_upper_bits_mask) << 1u)
                  bitor (index_wo_nonpage_qubit bitand nonpage_lower_bits_mask);
              auto const one_index = zero_index bitor permutated_target_qubit_mask;
              *(one_first + one_index) *= phase_coefficient;
            });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<1u>(
            parallel_policy, local_state, permutated_control_qubit,
            ::ket::mpi::gate::page::controlled_phase_shift_detail::make_controlled_phase_shift_coeff_cp(
              phase_coefficient,
              permutated_target_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask));
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        }
      } // namespace page
    } // namespace gate
  } // namespace mpi
} // namespace ket


#endif // KET_MPI_GATE_PAGE_CONTROLLED_PHASE_SHIFT_STANDARD_HPP
