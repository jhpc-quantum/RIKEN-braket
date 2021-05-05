#ifndef KET_MPI_GATE_PAGE_DETAIL_CONTROLLED_PHASE_SHIFT_COEFF_TP_DIAGONAL_HPP
# define KET_MPI_GATE_PAGE_DETAIL_CONTROLLED_PHASE_SHIFT_COEFF_TP_DIAGONAL_HPP

# include <boost/config.hpp>

# include <cassert>

# include <boost/range/size.hpp>

# include <ket/qubit.hpp>
# include <ket/control.hpp>
# include <ket/utility/integer_exp2.hpp>
# include <ket/utility/integer_log2.hpp>
# include <ket/mpi/qubit_permutation.hpp>
# include <ket/mpi/state.hpp>
# include <ket/mpi/gate/page/unsupported_page_gate_operation.hpp>
# include <ket/mpi/gate/page/detail/one_page_qubit_gate.hpp>


namespace ket
{
  namespace mpi
  {
    namespace gate
    {
      namespace page
      {
        namespace detail
        {
          namespace controlled_phase_shift_coeff_tp_detail
          {
            // tp_cl: target qubit is on page and control qubit is local
# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
            template <typename Complex, typename StateInteger>
            struct do_controlled_phase_shift_coeff_tp_cl
            {
              Complex phase_coefficient_;
              StateInteger control_qubit_mask_;
              StateInteger nonpage_lower_bits_mask_;
              StateInteger nonpage_upper_bits_mask_;

              do_controlled_phase_shift_coeff_tp_cl(
                Complex const& phase_coefficient,
                StateInteger const control_qubit_mask,
                StateInteger const nonpage_lower_bits_mask,
                StateInteger const nonpage_upper_bits_mask) noexcept
                : phase_coefficient_{phase_coefficient},
                  control_qubit_mask_{control_qubit_mask},
                  nonpage_lower_bits_mask_{nonpage_lower_bits_mask},
                  nonpage_upper_bits_mask_{nonpage_upper_bits_mask}
              { }

              template <typename Iterator>
              void operator()(
                Iterator const, Iterator const one_first,
                StateInteger const index_wo_nonpage_qubit) const
              {
                auto const zero_index
                  = ((index_wo_nonpage_qubit bitand nonpage_upper_bits_mask_) << 1u)
                    bitor (index_wo_nonpage_qubit bitand nonpage_lower_bits_mask_);
                auto const one_index = zero_index bitor control_qubit_mask_;
                *(one_first + one_index) *= phase_coefficient_;
              }
            }; // struct do_controlled_phase_shift_coeff_tp_cl<Complex, StateInteger>

            template <typename Complex, typename StateInteger>
            inline ::ket::mpi::gate::page::detail::controlled_phase_shift_coeff_tp_detail::do_controlled_phase_shift_coeff_tp_cl<Complex, StateInteger>
            make_do_controlled_phase_shift_coeff_tp_cl(
              Complex const& phase_coefficient,
              StateInteger const control_qubit_mask,
              StateInteger const nonpage_lower_bits_mask,
              StateInteger const nonpage_upper_bits_mask)
            { return {phase_coefficient, control_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask}; }
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS

            template <
              typename ParallelPolicy,
              typename Complex, int num_page_qubits_, typename StateAllocator,
              typename StateInteger, typename BitInteger, typename PermutationAllocator>
            inline ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator>&
            controlled_phase_shift_coeff_tp_cl(
              ::ket::mpi::utility::policy::general_mpi const mpi_policy,
              ParallelPolicy const parallel_policy,
              ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator>& local_state,
              Complex const& phase_coefficient,
              ::ket::qubit<StateInteger, BitInteger> const target_qubit,
              ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
              ::ket::mpi::qubit_permutation<
                StateInteger, BitInteger, PermutationAllocator> const& permutation)
            {
              assert(not local_state.is_page_qubit(permutation[control_qubit.qubit()]));
              auto const control_qubit_mask
                = ::ket::utility::integer_exp2<StateInteger>(permutation[control_qubit.qubit()]);
              auto const nonpage_lower_bits_mask = control_qubit_mask - StateInteger{1u};
              auto const nonpage_upper_bits_mask = compl nonpage_lower_bits_mask;

# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
              return ::ket::mpi::gate::page::detail::one_page_qubit_gate<1u>(
                mpi_policy, parallel_policy, local_state, target_qubit, permutation,
                [phase_coefficient, control_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask](
                  auto const, auto const one_first, StateInteger const index_wo_nonpage_qubit)
                {
                  auto const zero_index
                    = ((index_wo_nonpage_qubit bitand nonpage_upper_bits_mask) << 1u)
                      bitor (index_wo_nonpage_qubit bitand nonpage_lower_bits_mask);
                  auto const one_index = zero_index bitor control_qubit_mask;
                  *(one_first + one_index) *= phase_coefficient;
                });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
              return ::ket::mpi::gate::page::detail::one_page_qubit_gate<1u>(
                mpi_policy, parallel_policy, local_state, target_qubit, permutation,
                ::ket::mpi::gate::page::detail::controlled_phase_shift_coeff_tp_detail::make_do_controlled_phase_shift_coeff_tp_cl(
                  phase_coefficient,
                  control_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask));
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
            }

            // tp_cg: target qubit is on page and control qubit is global
# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
            template <typename Complex>
            struct do_controlled_phase_shift_coeff_tp_cg
            {
              Complex phase_coefficient_;

              explicit do_controlled_phase_shift_coeff_tp_cg(Complex const& phase_coefficient) noexcept
                : phase_coefficient_{phase_coefficient}
              { }

              template <typename Iterator, typename StateInteger>
              void operator()(
                Iterator const, Iterator const one_first, StateInteger const index) const
              { *(one_first + index) *= phase_coefficient_; }
            }; // struct do_controlled_phase_shift_coeff_tp_cg<Complex>

            template <typename Complex>
            inline ::ket::mpi::gate::page::detail::controlled_phase_shift_coeff_tp_detail::do_controlled_phase_shift_coeff_tp_cg<Complex>
            make_do_controlled_phase_shift_coeff_tp_cg(Complex const& phase_coefficient)
            { return ::ket::mpi::gate::page::detail::controlled_phase_shift_coeff_tp_detail::do_controlled_phase_shift_coeff_tp_cg<Complex>{phase_coefficient}; }
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS

            template <
              typename ParallelPolicy,
              typename Complex, int num_page_qubits_, typename StateAllocator,
              typename StateInteger, typename BitInteger, typename PermutationAllocator>
            inline ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator>&
            controlled_phase_shift_coeff_tp_cg(
              ::ket::mpi::utility::policy::general_mpi const mpi_policy,
              ParallelPolicy const parallel_policy,
              ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator>& local_state,
              Complex const& phase_coefficient,
              ::ket::qubit<StateInteger, BitInteger> const target_qubit,
              ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
              ::ket::mpi::qubit_permutation<
                StateInteger, BitInteger, PermutationAllocator> const& permutation,
              yampi::rank const rank)
            {
              assert(not local_state.is_page_qubit(permutation[control_qubit.qubit()]));

              using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
              auto const least_global_permutated_qubit
                = qubit_type{::ket::utility::integer_log2<BitInteger>(boost::size(local_state))};

              auto const control_qubit_mask
                = StateInteger{1u}
                  << (permutation[control_qubit.qubit()] - least_global_permutated_qubit);

              if ((static_cast<StateInteger>(rank.mpi_rank()) bitand control_qubit_mask) == StateInteger{0u})
                return local_state;

# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
              return ::ket::mpi::gate::page::detail::one_page_qubit_gate<0u>(
                mpi_policy, parallel_policy, local_state, target_qubit, permutation,
                [phase_coefficient](
                  auto const, auto const one_first, StateInteger const index)
                { *(one_first + index) *= phase_coefficient; });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
              return ::ket::mpi::gate::page::detail::one_page_qubit_gate<0u>(
                mpi_policy, parallel_policy, local_state, target_qubit, permutation,
                ::ket::mpi::gate::page::detail::controlled_phase_shift_coeff_tp_detail::make_do_controlled_phase_shift_coeff_tp_cg(phase_coefficient));
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
            }
          } // namespace controlled_phase_shift_coeff_tp_detail

          template <
            typename MpiPolicy, typename ParallelPolicy,
            typename RandomAccessRange, typename Complex,
            typename StateInteger, typename BitInteger, typename PermutationAllocator>
          [[noreturn]] inline RandomAccessRange& controlled_phase_shift_coeff_tp(
            MpiPolicy const, ParallelPolicy const,
            RandomAccessRange& local_state,
            Complex const&,
            ::ket::qubit<StateInteger, BitInteger> const,
            ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const,
            ::ket::mpi::qubit_permutation<
              StateInteger, BitInteger, PermutationAllocator> const&,
            yampi::rank const)
          { throw ::ket::mpi::gate::page::unsupported_page_gate_operation<0, false>{"controlled_phase_shift_coeff_tp"}; }

          template <
            typename ParallelPolicy,
            typename Complex, typename StateAllocator,
            typename StateInteger, typename BitInteger, typename PermutationAllocator>
          [[noreturn]] inline ::ket::mpi::state<Complex, 0, StateAllocator>&
          controlled_phaes_shift_coeff_tp(
            ::ket::mpi::utility::policy::general_mpi const, ParallelPolicy const,
            ::ket::mpi::state<Complex, 0, StateAllocator>& local_state,
            Complex const&,
            ::ket::qubit<StateInteger, BitInteger> const,
            ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const,
            ::ket::mpi::qubit_permutation<
              StateInteger, BitInteger, PermutationAllocator> const&,
            yampi::rank const)
          { throw ::ket::mpi::gate::page::unsupported_page_gate_operation<0>{"controlled_phase_shift_coeff_tp"}; }

          template <
            typename ParallelPolicy,
            typename Complex, int num_page_qubits_, typename StateAllocator,
            typename StateInteger, typename BitInteger, typename PermutationAllocator>
          inline ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator>&
          controlled_phase_shift_coeff_tp(
            ::ket::mpi::utility::policy::general_mpi const mpi_policy,
            ParallelPolicy const parallel_policy,
            ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator>& local_state,
            Complex const& phase_coefficient,
            ::ket::qubit<StateInteger, BitInteger> const target_qubit,
            ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
            ::ket::mpi::qubit_permutation<
              StateInteger, BitInteger, PermutationAllocator> const& permutation,
            yampi::rank const rank)
          {
            assert(not local_state.is_page_qubit(permutation[control_qubit.qubit()]));

            using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
            auto const least_permutated_global_qubit
              = qubit_type{::ket::utility::integer_log2<BitInteger>(boost::size(local_state))};

            if (permutation[control_qubit.qubit()] < least_permutated_global_qubit)
              return ::ket::mpi::gate::page::detail::controlled_phase_shift_coeff_tp_detail::controlled_phase_shift_coeff_tp_cl(
                mpi_policy, parallel_policy,
                local_state, phase_coefficient, target_qubit, control_qubit, permutation);

            return ::ket::mpi::gate::page::detail::controlled_phase_shift_coeff_tp_detail::controlled_phase_shift_coeff_tp_cg(
              mpi_policy, parallel_policy,
              local_state, phase_coefficient, target_qubit, control_qubit, permutation, rank);
          }
        } // namespace detail
      } // namespace page
    } // namespace gate
  } // namespace mpi
} // namespace ket


#endif // KET_MPI_GATE_PAGE_DETAIL_CONTROLLED_PHASE_SHIFT_COEFF_TP_DIAGONAL_HPP
