#ifndef KET_MPI_GATE_PAGE_SQRT_PAULI_Z_DIAGONAL_HPP
# define KET_MPI_GATE_PAGE_SQRT_PAULI_Z_DIAGONAL_HPP

# include <cassert>
# include <cmath>
# include <memory>

# include <boost/math/constants/constants.hpp>

# include <ket/qubit.hpp>
# include <ket/control.hpp>
# include <ket/utility/imaginary_unit.hpp>
# include <ket/utility/meta/real_of.hpp>
# include <ket/utility/meta/ranges.hpp>
# include <ket/mpi/permutated.hpp>
# include <ket/mpi/page/is_on_page.hpp>
# include <ket/mpi/gate/page/detail/two_page_qubits_gate.hpp>
# include <ket/mpi/gate/page/detail/one_page_qubit_gate.hpp>
# include <ket/mpi/gate/page/detail/sqrt_pauli_z2_p_diagonal.hpp>
# include <ket/mpi/gate/page/detail/sqrt_pauli_cz_tp_diagonal.hpp>
# include <ket/mpi/gate/page/detail/sqrt_pauli_cz_cp_diagonal.hpp>
# include <ket/mpi/gate/page/detail/adj_sqrt_pauli_z2_p_diagonal.hpp>
# include <ket/mpi/gate/page/detail/adj_sqrt_pauli_cz_tp_diagonal.hpp>
# include <ket/mpi/gate/page/detail/adj_sqrt_pauli_cz_cp_diagonal.hpp>


namespace ket
{
  namespace mpi
  {
    namespace gate
    {
      namespace page
      {
        // 1_p: the qubit of sZ is on page
        // sZ_i
        // sZ_1 (a_0 |0> + a_1 |1>) = a_0 |0> + i a_1 |1>
        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger>
        inline auto sqrt_pauli_z1(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_qubit)
        -> RandomAccessRange&
        {
          assert(::ket::mpi::page::is_on_page(permutated_qubit, local_state));

          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<0u>(
            parallel_policy, local_state, permutated_qubit,
            [](auto const, auto const one_first, StateInteger const index, int const)
            {
              using complex_type = ::ket::utility::meta::range_value_t<RandomAccessRange>;
              *(one_first + index) *= ::ket::utility::imaginary_unit<complex_type>();
            });
        }

        // 2_2p: both of qubits of sZZ are on page
        // sZZ_{ij} = sZ_i sZ_j or sZ2_{ij}
        // sZZ_{1,2} (a_{00} |00> + a_{01} |01> + a_{10} |10> + a{11} |11>)
        //   = a_{00} |00> + i a_{01} |01> + i a_{10} |10> - a{11} |11>
        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger>
        inline auto sqrt_pauli_z2_2p(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const page_permutated_qubit1,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const page_permutated_qubit2)
        -> RandomAccessRange&
        {
          return ::ket::mpi::gate::page::detail::two_page_qubits_gate<0u>(
            parallel_policy, local_state, page_permutated_qubit1, page_permutated_qubit2,
            [](auto const, auto const first_01, auto const first_10, auto const first_11,
              StateInteger const index, int const)
            {
              using complex_type = ::ket::utility::meta::range_value_t<RandomAccessRange>;
              using real_type = ::ket::utility::meta::real_t<complex_type>;
              *(first_01 + index) *= ::ket::utility::imaginary_unit<complex_type>();
              *(first_10 + index) *= ::ket::utility::imaginary_unit<complex_type>();
              *(first_11 + index) *= real_type{-1};
            });
        }

        // 2_p: only one qubit of eZZ is on page
        // sZZ_{ij} = sZ_i sZ_j or sZ2_{ij}
        // sZZ_{1,2} (a_{00} |00> + a_{01} |01> + a_{10} |10> + a{11} |11>)
        //   = a_{00} |00> + i a_{01} |01> + i a_{10} |10> - a{11} |11>
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger>
        inline auto sqrt_pauli_z2_p(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const page_permutated_qubit,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const nonpage_permutated_qubit,
          yampi::rank const rank)
        -> RandomAccessRange&
        {
          return ::ket::mpi::gate::page::detail::sqrt_pauli_z2_p(
            mpi_policy, parallel_policy, local_state,
            page_permutated_qubit, nonpage_permutated_qubit, rank);
        }

        // cz_tcp: both of target and control qubits of CsZ are on page
        // CsZ_{tc} or C1sZ_{tc}
        // CsZ_{1,2} (a_{00} |00> + a_{01} |01> + a_{10} |10> + a_{11} |11>)
        //   = a_{00} |00> + a_{01} |01> + a_{10} |10> + i a_{11} |11>
        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger>
        inline auto sqrt_pauli_cz_tcp(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_target_qubit,
          ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const permutated_control_qubit)
        -> RandomAccessRange&
        {
          assert(::ket::mpi::page::is_on_page(permutated_target_qubit, local_state));
          assert(::ket::mpi::page::is_on_page(permutated_control_qubit, local_state));

          return ::ket::mpi::gate::page::detail::two_page_qubits_gate<0u>(
            parallel_policy, local_state, permutated_target_qubit, permutated_control_qubit,
            [](auto const, auto const, auto const, auto const first_11, StateInteger const index, int const)
            {
              using complex_type = ::ket::utility::meta::range_value_t<RandomAccessRange>;
              *(first_11 + index) *= ::ket::utility::imaginary_unit<complex_type>();
            });
        }

        // cz_tp: only target qubit is on page
        // CsZ_{tc} or C1sZ_{tc}
        // CsZ_{1,2} (a_{00} |00> + a_{01} |01> + a_{10} |10> + a_{11} |11>)
        //   = a_{00} |00> + a_{01} |01> + a_{10} |10> + i a_{11} |11>
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger>
        inline auto sqrt_pauli_cz_tp(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_target_qubit,
          ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const permutated_control_qubit,
          yampi::rank const rank)
        -> RandomAccessRange&
        {
          return ::ket::mpi::gate::page::detail::sqrt_pauli_cz_tp(
            mpi_policy, parallel_policy, local_state, permutated_target_qubit, permutated_control_qubit, rank);
        }

        // cz_cp: only control qubit is on page
        // CsZ_{tc} or C1sZ_{tc}
        // CsZ_{1,2} (a_{00} |00> + a_{01} |01> + a_{10} |10> + a_{11} |11>)
        //   = a_{00} |00> + a_{01} |01> + a_{10} |10> + i a_{11} |11>
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger>
        inline auto sqrt_pauli_cz_cp(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_target_qubit,
          ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const permutated_control_qubit,
          yampi::rank const rank)
        -> RandomAccessRange&
        {
          return ::ket::mpi::gate::page::detail::sqrt_pauli_cz_cp(
            mpi_policy, parallel_policy, local_state, permutated_target_qubit, permutated_control_qubit, rank);
        }

        // 1_p: the qubit of sZ+ is on page
        // sZ+_i
        // sZ+_1 (a_0 |0> + a_1 |1>) = a_0 |0> - i a_1 |1>
        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger>
        inline auto adj_sqrt_pauli_z1(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_qubit)
        -> RandomAccessRange&
        {
          assert(::ket::mpi::page::is_on_page(permutated_qubit, local_state));

          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<0u>(
            parallel_policy, local_state, permutated_qubit,
            [](auto const zero_first, auto const one_first, StateInteger const index, int const)
            {
              using complex_type = ::ket::utility::meta::range_value_t<RandomAccessRange>;
              *(one_first + index) *= ::ket::utility::minus_imaginary_unit<complex_type>();
            });
        }

        // 2_2p: both of qubits of sZZ+ are on page
        // sZZ+_{ij} = sZ+_i sZ+_j or sZ2+_{ij}
        // sZZ+_{1,2} (a_{00} |00> + a_{01} |01> + a_{10} |10> + a{11} |11>)
        //   = a_{00} |00> - i a_{01} |01> - i a_{10} |10> - a{11} |11>
        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger>
        inline auto adj_sqrt_pauli_z2_2p(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const page_permutated_qubit1,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const page_permutated_qubit2)
        -> RandomAccessRange&
        {
          return ::ket::mpi::gate::page::detail::two_page_qubits_gate<0u>(
            parallel_policy, local_state, page_permutated_qubit1, page_permutated_qubit2,
            [](auto const first_00, auto const first_01, auto const first_10, auto const first_11,
              StateInteger const index, int const)
            {
              using complex_type = ::ket::utility::meta::range_value_t<RandomAccessRange>;
              using real_type = ::ket::utility::meta::real_t<complex_type>;
              *(first_01 + index) *= ::ket::utility::minus_imaginary_unit<complex_type>();
              *(first_10 + index) *= ::ket::utility::minus_imaginary_unit<complex_type>();
              *(first_11 + index) *= real_type{-1};
            });
        }

        // 2_p: only one qubit of sZZ+ is on page
        // sZZ+_{ij} = sZ+_i sZ+_j or sZ2+_{ij}
        // sZZ+_{1,2} (a_{00} |00> + a_{01} |01> + a_{10} |10> + a{11} |11>)
        //   = a_{00} |00> - i a_{01} |01> - i a_{10} |10> - a{11} |11>
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger>
        inline auto adj_sqrt_pauli_z2_p(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const page_permutated_qubit,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const nonpage_permutated_qubit,
          yampi::rank const rank)
        -> RandomAccessRange&
        {
          return ::ket::mpi::gate::page::detail::adj_sqrt_pauli_z2_p(
            mpi_policy, parallel_policy, local_state,
            page_permutated_qubit, nonpage_permutated_qubit, rank);
        }

        // cz_tcp: both of target and control qubits of CsZ+ are on page
        // CsZ+_{tc} or C1sZ+_{tc}
        // CsZ+_{1,2} (a_{00} |00> + a_{01} |01> + a_{10} |10> + a_{11} |11>)
        //   = a_{00} |00> + a_{01} |01> + a_{10} |10> - i a_{11} |11>
        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger>
        inline auto adj_sqrt_pauli_cz_tcp(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_target_qubit,
          ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const permutated_control_qubit)
        -> RandomAccessRange&
        {
          assert(::ket::mpi::page::is_on_page(permutated_target_qubit, local_state));
          assert(::ket::mpi::page::is_on_page(permutated_control_qubit, local_state));

          return ::ket::mpi::gate::page::detail::two_page_qubits_gate<0u>(
            parallel_policy, local_state, permutated_target_qubit, permutated_control_qubit,
            [](auto const, auto const, auto const, auto const first_11, StateInteger const index, int const)
            {
              using complex_type = ::ket::utility::meta::range_value_t<RandomAccessRange>;
              *(first_11 + index) *= ::ket::utility::minus_imaginary_unit<complex_type>();
            });
        }

        // cz_tp: only target qubit is on page
        // CsZ+_{tc} or C1sZ+_{tc}
        // CsZ+_{1,2} (a_{00} |00> + a_{01} |01> + a_{10} |10> + a_{11} |11>)
        //   = a_{00} |00> + a_{01} |01> + a_{10} |10> - i a_{11} |11>
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger>
        inline auto adj_sqrt_pauli_cz_tp(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_target_qubit,
          ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const permutated_control_qubit,
          yampi::rank const rank)
        -> RandomAccessRange&
        {
          return ::ket::mpi::gate::page::detail::adj_sqrt_pauli_cz_tp(
            mpi_policy, parallel_policy, local_state, permutated_target_qubit, permutated_control_qubit, rank);
        }

        // cz_cp: only control qubit is on page
        // CsZ+_{tc} or C1sZ+_{tc}
        // CsZ+_{1,2} (a_{00} |00> + a_{01} |01> + a_{10} |10> + a_{11} |11>)
        //   = a_{00} |00> + a_{01} |01> + a_{10} |10> - i a_{11} |11>
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger>
        inline auto adj_sqrt_pauli_cz_cp(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_target_qubit,
          ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const permutated_control_qubit,
          yampi::rank const rank)
        -> RandomAccessRange&
        {
          return ::ket::mpi::gate::page::detail::adj_sqrt_pauli_cz_cp(
            mpi_policy, parallel_policy, local_state, permutated_target_qubit, permutated_control_qubit, rank);
        }
      } // namespace page
    } // namespace gate
  } // namespace mpi
} // namespace ket


#endif // KET_MPI_GATE_PAGE_SQRT_PAULI_Z_DIAGONAL_HPP
