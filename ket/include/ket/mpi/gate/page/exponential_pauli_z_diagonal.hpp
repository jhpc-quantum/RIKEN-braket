#ifndef KET_MPI_GATE_PAGE_EXPONENTIAL_PAULI_Z_DIAGONAL_HPP
# define KET_MPI_GATE_PAGE_EXPONENTIAL_PAULI_Z_DIAGONAL_HPP

# include <cassert>
# include <cmath>
# include <type_traits>
# include <memory>

# include <yampi/rank.hpp>

# include <ket/qubit.hpp>
# include <ket/control.hpp>
# include <ket/utility/integer_exp2.hpp>
# include <ket/utility/imaginary_unit.hpp>
# include <ket/mpi/permutated.hpp>
# include <ket/mpi/page/is_on_page.hpp>
# include <ket/mpi/gate/page/detail/two_page_qubits_gate.hpp>
# include <ket/mpi/gate/page/detail/one_page_qubit_gate.hpp>
# include <ket/mpi/gate/page/detail/exponential_pauli_z2_p_diagonal.hpp>
# include <ket/mpi/gate/page/detail/exponential_pauli_cz_tp_diagonal.hpp>
# include <ket/mpi/gate/page/detail/exponential_pauli_cz_cp_diagonal.hpp>


namespace ket
{
  namespace mpi
  {
    namespace gate
    {
      namespace page
      {
        // 1_p: the qubit of eY is on page
        // eZ_i(thta) = exp(i theta Z_i) = I cos(theta) + i Z_i sin(theta)
        // eZ_1(thta) (a_0 |0> + a_1 |1>) = e^{i theta} a_0 |0> + e^{-i theta} a_1 |1>
        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename Complex, typename StateInteger, typename BitInteger>
        inline auto exponential_pauli_z_coeff1(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_qubit)
        -> RandomAccessRange&
        {
          assert(::ket::mpi::page::is_on_page(permutated_qubit, local_state));

          using std::conj;
          auto const conj_phase_coefficient = conj(phase_coefficient);

          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<0u>(
            parallel_policy, local_state, permutated_qubit,
            [&phase_coefficient, &conj_phase_coefficient](
              auto const zero_first, auto const one_first, StateInteger const index, int const)
            {
              *(zero_first + index) *= phase_coefficient;
              *(one_first + index) *= conj_phase_coefficient;
            });
        }

        // 2_2p: both of qubits of eZZ are on page
        // eZZ_{ij}(theta) = exp(i theta Z_i Z_j) = I cos(theta) + i Z_i Z_j sin(theta)
        // eZZ_{1,2}(theta) (a_{00} |00> + a_{01} |01> + a_{10} |10> + a{11} |11>)
        //   = e^{i theta} |00> + e^{-i theta} |01> + e^{-i theta} |10> + e^{i theta} |11>
        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename Complex, typename StateInteger, typename BitInteger>
        inline auto exponential_pauli_z_coeff2_2p(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const page_permutated_qubit1,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const page_permutated_qubit2)
        -> RandomAccessRange&
        {
          using std::conj;
          auto const conj_phase_coefficient = conj(phase_coefficient);

          return ::ket::mpi::gate::page::detail::two_page_qubits_gate<0u>(
            parallel_policy, local_state, page_permutated_qubit1, page_permutated_qubit2,
            [&phase_coefficient, &conj_phase_coefficient](
              auto const first_00, auto const first_01, auto const first_10, auto const first_11,
              StateInteger const index, int const)
            {
              *(first_00 + index) *= phase_coefficient;
              *(first_01 + index) *= conj_phase_coefficient;
              *(first_10 + index) *= conj_phase_coefficient;
              *(first_11 + index) *= phase_coefficient;
            });
        }

        // 2_p: only one qubit of eZZ is on page
        // eZZ_{ij}(theta) = exp(i theta Z_i Z_j) = I cos(theta) + i Z_i Z_j sin(theta)
        // eZZ_{1,2}(theta) (a_{00} |00> + a_{01} |01> + a_{10} |10> + a{11} |11>)
        //   = e^{i theta} |00> + e^{-i theta} |01> + e^{-i theta} |10> + e^{i theta} |11>
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename Complex, typename StateInteger, typename BitInteger>
        inline auto exponential_pauli_z_coeff2_p(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const page_permutated_qubit,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const nonpage_permutated_qubit,
          yampi::rank const rank)
        -> RandomAccessRange&
        {
          return ::ket::mpi::gate::page::detail::exponential_pauli_z_coeff2_p(
            mpi_policy, parallel_policy, local_state,
            phase_coefficient, page_permutated_qubit, nonpage_permutated_qubit, rank);
        }

        // cz_coeff_tcp: both of target and control qubits of CeZ are on page
        // CeZ_{tc}(theta) = C[exp(i theta Z_t)]_c = C[I cos(theta) + i Z_t sin(theta)]_c, C1eZ_{tc}(theta), CeZ1_{tc}(theta), or C1eZ1_{tc}(theta)
        // CeZ_{1,2}(theta) (a_{00} |00> + a_{01} |01> + a_{10} |10> + a{11} |11>)
        //   = a_{00} |00> + a_{01} |01> + e^{i theta} a_{10} |10> + e^{-i theta} |11>
        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename Complex, typename StateInteger, typename BitInteger>
        inline auto exponential_pauli_cz_coeff_tcp(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state, Complex const& phase_coefficient,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_target_qubit,
          ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const permutated_control_qubit)
        -> RandomAccessRange&
        {
          assert(::ket::mpi::page::is_on_page(permutated_target_qubit, local_state));
          assert(::ket::mpi::page::is_on_page(permutated_control_qubit, local_state));

          using std::conj;
          auto const conj_phase_coefficient = conj(phase_coefficient);

          return ::ket::mpi::gate::page::detail::two_page_qubits_gate<0u>(
            parallel_policy, local_state, permutated_target_qubit, permutated_control_qubit,
            [&phase_coefficient, &conj_phase_coefficient](
              auto const, auto const, auto const first_10, auto const first_11, StateInteger const index, int const)
            {
              *(first_10 + index) *= phase_coefficient;
              *(first_11 + index) *= conj_phase_coefficient;
            });
        }

        // cz_coeff_tp: only target qubit is on page
        // CeZ_{tc}(theta) = C[exp(i theta Z_t)]_c = C[I cos(theta) + i Z_t sin(theta)]_c, C1eZ_{tc}(theta), CeZ1_{tc}(theta), or C1eZ1_{tc}(theta)
        // CeZ_{1,2}(theta) (a_{00} |00> + a_{01} |01> + a_{10} |10> + a{11} |11>)
        //   = a_{00} |00> + a_{01} |01> + e^{i theta} a_{10} |10> + e^{-i theta} |11>
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename Complex, typename StateInteger, typename BitInteger>
        inline auto exponential_pauli_cz_coeff_tp(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state, Complex const& phase_coefficient,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_target_qubit,
          ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const permutated_control_qubit,
          yampi::rank const rank)
        -> RandomAccessRange&
        {
          return ::ket::mpi::gate::page::detail::exponential_pauli_cz_coeff_tp(
            mpi_policy, parallel_policy, local_state,
            phase_coefficient, permutated_target_qubit, permutated_control_qubit, rank);
        }

        // cz_coeff_cp: only control qubit is on page
        // CeZ_{tc}(theta) = C[exp(i theta Z_t)]_c = C[I cos(theta) + i Z_t sin(theta)]_c, C1eZ_{tc}(theta), CeZ1_{tc}(theta), or C1eZ1_{tc}(theta)
        // CeZ_{1,2}(theta) (a_{00} |00> + a_{01} |01> + a_{10} |10> + a{11} |11>)
        //   = a_{00} |00> + a_{01} |01> + e^{i theta} a_{10} |10> + e^{-i theta} |11>
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename Complex, typename StateInteger, typename BitInteger>
        inline auto exponential_pauli_cz_coeff_cp(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state, Complex const& phase_coefficient,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_target_qubit,
          ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const permutated_control_qubit,
          yampi::rank const rank)
        -> RandomAccessRange&
        {
          return ::ket::mpi::gate::page::detail::exponential_pauli_cz_coeff_cp(
            mpi_policy, parallel_policy, local_state,
            phase_coefficient, permutated_target_qubit, permutated_control_qubit, rank);
        }
      } // namespace page
    } // namespace gate
  } // namespace mpi
} // namespace ket


#endif // KET_MPI_GATE_PAGE_EXPONENTIAL_PAULI_Z_DIAGONAL_HPP
