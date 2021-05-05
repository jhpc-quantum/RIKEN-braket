#ifndef KET_MPI_GATE_PAGE_CONTROLLED_PHASE_SHIFT_DIAGONAL_HPP
# define KET_MPI_GATE_PAGE_CONTROLLED_PHASE_SHIFT_DIAGONAL_HPP

# include <boost/config.hpp>

# include <ket/qubit.hpp>
# include <ket/control.hpp>
# include <ket/mpi/qubit_permutation.hpp>
# include <ket/mpi/state.hpp>
# include <ket/mpi/gate/page/detail/two_page_qubits_gate.hpp>
# include <ket/mpi/gate/page/detail/controlled_phase_shift_coeff_tp_diagonal.hpp>
# include <ket/mpi/gate/page/detail/controlled_phase_shift_coeff_cp_diagonal.hpp>


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

          template <typename Complex>
          inline ::ket::mpi::gate::page::controlled_phase_shift_detail::controlled_phase_shift_coeff_tcp<Complex>
          make_controlled_phase_shift_coeff_tcp(Complex const& phase_coefficient)
          { return ::ket::mpi::gate::page::controlled_phase_shift_detail::controlled_phase_shift_coeff_tcp<Complex>{phase_coefficient}; }
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        } // namespace controlled_phase_shift_detail

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename Complex,
          typename StateInteger, typename BitInteger, typename PermutationAllocator>
        inline RandomAccessRange& controlled_phase_shift_coeff_tcp(
          MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          Complex const& phase_coefficient,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, PermutationAllocator> const& permutation)
        {
# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::two_page_qubits_gate<0u>(
            mpi_policy, parallel_policy, local_state,
            target_qubit, control_qubit, permutation,
            [phase_coefficient](
              auto const, auto const, auto const, auto const first_11,
              StateInteger const index, int const)
            { *(first_11 + index) *= phase_coefficient; });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::two_page_qubits_gate<0u>(
            mpi_policy, parallel_policy, local_state,
            target_qubit, control_qubit, permutation,
            ::ket::mpi::gate::page::controlled_phase_shift_detail::make_controlled_phase_shift_coeff_tcp(phase_coefficient));
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        }

        // tp: only target qubit is on page
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename Complex,
          typename StateInteger, typename BitInteger, typename PermutationAllocator>
        inline RandomAccessRange& controlled_phase_shift_coeff_tp(
          MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          Complex const& phase_coefficient,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, PermutationAllocator> const& permutation,
          yampi::rank const rank)
        {
          return ::ket::mpi::gate::page::detail::controlled_phase_shift_coeff_tp(
            mpi_policy, parallel_policy,
            local_state, phase_coefficient, target_qubit, control_qubit,
            permutation, rank);
        }

        // cp: only control qubit is on page
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename Complex,
          typename StateInteger, typename BitInteger, typename PermutationAllocator>
        inline RandomAccessRange& controlled_phase_shift_coeff_cp(
          MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          Complex const& phase_coefficient,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, PermutationAllocator> const& permutation,
          yampi::rank const rank)
        {
          return ::ket::mpi::gate::page::detail::controlled_phase_shift_coeff_cp(
            mpi_policy, parallel_policy,
            local_state, phase_coefficient, target_qubit, control_qubit,
            permutation, rank);
        }
      } // namespace page
    } // namespage gate
  } // namespace mpi
} // namespace ket


#endif // KET_MPI_GATE_PAGE_CONTROLLED_PHASE_SHIFT_DIAGONAL_HPP
