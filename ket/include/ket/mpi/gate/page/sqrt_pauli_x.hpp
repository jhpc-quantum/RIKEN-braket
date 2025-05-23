#ifndef KET_MPI_GATE_PAGE_SQRT_PAULI_X_HPP
# define KET_MPI_GATE_PAGE_SQRT_PAULI_X_HPP

# include <cassert>
# include <cmath>
# include <memory>

# include <boost/math/constants/constants.hpp>

# include <ket/qubit.hpp>
# include <ket/control.hpp>
# include <ket/utility/integer_exp2.hpp>
# include <ket/utility/meta/real_of.hpp>
# include <ket/utility/meta/ranges.hpp>
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
        // 1_p: the qubit of sX is on page
        // sX_i
        // sX_1 (a_0 |0> + a_1 |1>) = [(1+i) a_0 + (1-i) a_1]/2 |0> + [(1-i) a_0 + (1+i) a_1]/2 |1>
        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger>
        inline auto sqrt_pauli_x(
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
              auto const zero_iter = zero_first + index;
              auto const one_iter = one_first + index;
              auto const zero_iter_value = *zero_iter;

              using complex_type = ::ket::utility::meta::range_value_t<RandomAccessRange>;
              using real_type = ::ket::utility::meta::real_t<complex_type>;
              using boost::math::constants::half;
              constexpr auto half_one_plus_i = complex_type{half<real_type>(), half<real_type>()};
              constexpr auto half_one_minus_i = complex_type{half<real_type>(), -half<real_type>()};
              *zero_iter *= half_one_plus_i;
              *zero_iter += half_one_minus_i * *one_iter;
              *one_iter *= half_one_plus_i;
              *one_iter += half_one_minus_i * zero_iter_value;
            });
        }

        // cx_tcp: both of target and control qubits of CsX are on page
        // CsX_{tc} or C1sX_{tc}
        // CsX_{1,2} (a_{00} |00> + a_{01} |01> + a_{10} |10> + a_{11} |11>)
        //   = a_{00} |00> + a_{01} |01> + [(1+i) a_{10} + (1-i) a_{11}]/2 |10> + [(1-i) a_{10} + (1+i) a_{11}]/2 |11>
        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger>
        inline auto sqrt_pauli_cx_tcp(
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
            [](auto const, auto const, auto const first_10, auto const first_11, StateInteger const index, int const)
            {
              auto const control_on_iter = first_10 + index;
              auto const target_control_on_iter = first_11 + index;
              auto const control_on_iter_value = *control_on_iter;

              using complex_type = ::ket::utility::meta::range_value_t<RandomAccessRange>;
              using real_type = ::ket::utility::meta::real_t<complex_type>;
              using boost::math::constants::half;
              constexpr auto half_one_plus_i = complex_type{half<real_type>(), half<real_type>()};
              constexpr auto half_one_minus_i = complex_type{half<real_type>(), -half<real_type>()};
              *control_on_iter *= half_one_plus_i;
              *control_on_iter += half_one_minus_i * *target_control_on_iter;
              *target_control_on_iter *= half_one_plus_i;
              *target_control_on_iter += half_one_minus_i * control_on_iter_value;
            });
        }

        // cx_tp: only target qubit is on page
        // CsX_{tc} or C1sX_{tc}
        // CsX_{1,2} (a_{00} |00> + a_{01} |01> + a_{10} |10> + a_{11} |11>)
        //   = a_{00} |00> + a_{01} |01> + [(1+i) a_{10} + (1-i) a_{11}]/2 |10> + [(1-i) a_{10} + (1+i) a_{11}]/2 |11>
        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger>
        inline auto sqrt_pauli_cx_tp(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_target_qubit,
          ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const permutated_control_qubit)
        -> RandomAccessRange&
        {
          assert(not ::ket::mpi::page::is_on_page(permutated_control_qubit, local_state));
          auto const control_qubit_mask
            = ::ket::utility::integer_exp2<StateInteger>(permutated_control_qubit);
          auto const nonpage_lower_bits_mask = control_qubit_mask - StateInteger{1u};
          auto const nonpage_upper_bits_mask = compl nonpage_lower_bits_mask;

          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<1u>(
            parallel_policy, local_state, permutated_target_qubit,
            [control_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask](
              auto const zero_first, auto const one_first, StateInteger const index_wo_nonpage_qubit, int const)
            {
              auto const zero_index
                = ((index_wo_nonpage_qubit bitand nonpage_upper_bits_mask) << 1u)
                  bitor (index_wo_nonpage_qubit bitand nonpage_lower_bits_mask);
              auto const one_index = zero_index bitor control_qubit_mask;
              auto const control_on_iter = zero_first + one_index;
              auto const target_control_on_iter = one_first + one_index;
              auto const control_on_iter_value = *control_on_iter;

              using complex_type = ::ket::utility::meta::range_value_t<RandomAccessRange>;
              using real_type = ::ket::utility::meta::real_t<complex_type>;
              using boost::math::constants::half;
              constexpr auto half_one_plus_i = complex_type{half<real_type>(), half<real_type>()};
              constexpr auto half_one_minus_i = complex_type{half<real_type>(), -half<real_type>()};
              *control_on_iter *= half_one_plus_i;
              *control_on_iter += half_one_minus_i * *target_control_on_iter;
              *target_control_on_iter *= half_one_plus_i;
              *target_control_on_iter += half_one_minus_i * control_on_iter_value;
            });
        }

        // cx_cp: only control qubit is on page
        // CsX_{tc} or C1sX_{tc}
        // CsX_{1,2} (a_{00} |00> + a_{01} |01> + a_{10} |10> + a_{11} |11>)
        //   = a_{00} |00> + a_{01} |01> + [(1+i) a_{10} + (1-i) a_{11}]/2 |10> + [(1-i) a_{10} + (1+i) a_{11}]/2 |11>
        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger>
        inline auto sqrt_pauli_cx_cp(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_target_qubit,
          ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const permutated_control_qubit)
        -> RandomAccessRange&
        {
          assert(not ::ket::mpi::page::is_on_page(permutated_target_qubit, local_state));

          auto const target_qubit_mask
            = ::ket::utility::integer_exp2<StateInteger>(permutated_target_qubit);
          auto const nonpage_lower_bits_mask = target_qubit_mask - StateInteger{1u};
          auto const nonpage_upper_bits_mask = compl nonpage_lower_bits_mask;

          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<1u>(
            parallel_policy, local_state, permutated_control_qubit,
            [target_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask](
              auto const, auto const one_first, StateInteger const index_wo_nonpage_qubit, int const)
            {
              auto const zero_index
                = ((index_wo_nonpage_qubit bitand nonpage_upper_bits_mask) << 1u)
                  bitor (index_wo_nonpage_qubit bitand nonpage_lower_bits_mask);
              auto const one_index = zero_index bitor target_qubit_mask;
              auto const control_on_iter = one_first + zero_index;
              auto const target_control_on_iter = one_first + one_index;
              auto const control_on_iter_value = *control_on_iter;

              using complex_type = ::ket::utility::meta::range_value_t<RandomAccessRange>;
              using real_type = ::ket::utility::meta::real_t<complex_type>;
              using boost::math::constants::half;
              constexpr auto half_one_plus_i = complex_type{half<real_type>(), half<real_type>()};
              constexpr auto half_one_minus_i = complex_type{half<real_type>(), -half<real_type>()};
              *control_on_iter *= half_one_plus_i;
              *control_on_iter += half_one_minus_i * *target_control_on_iter;
              *target_control_on_iter *= half_one_plus_i;
              *target_control_on_iter += half_one_minus_i * control_on_iter_value;
            });
        }

        // 1_p: the qubit of sX is on page
        // sX+_i
        // sX+_1 (a_0 |0> + a_1 |1>) = [(1-i) a_0 + (1+i) a_1]/2 |0> + [(1+i) a_0 + (1-i) a_1]/2 |1>
        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger>
        inline auto adj_sqrt_pauli_x(
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
              auto const zero_iter = zero_first + index;
              auto const one_iter = one_first + index;
              auto const zero_iter_value = *zero_iter;

              using complex_type = ::ket::utility::meta::range_value_t<RandomAccessRange>;
              using real_type = ::ket::utility::meta::real_t<complex_type>;
              using boost::math::constants::half;
              constexpr auto half_one_plus_i = complex_type{half<real_type>(), half<real_type>()};
              constexpr auto half_one_minus_i = complex_type{half<real_type>(), -half<real_type>()};
              *zero_iter *= half_one_minus_i;
              *zero_iter += half_one_plus_i * *one_iter;
              *one_iter *= half_one_minus_i;
              *one_iter += half_one_plus_i * zero_iter_value;
            });
        }

        // cx_tcp: both of target and control qubits of CsX are on page
        // CsX+_{tc} or C1sX+_{tc}
        // CsX+_{1,2} (a_{00} |00> + a_{01} |01> + a_{10} |10> + a_{11} |11>)
        //   = a_{00} |00> + a_{01} |01> + [(1-i) a_{10} + (1+i) a_{11}]/2 |10> + [(1+i) a_{10} + (1-i) a_{11}]/2 |11>
        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger>
        inline auto adj_sqrt_pauli_cx_tcp(
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
            [](auto const, auto const, auto const first_10, auto const first_11, StateInteger const index, int const)
            {
              auto const control_on_iter = first_10 + index;
              auto const target_control_on_iter = first_11 + index;
              auto const control_on_iter_value = *control_on_iter;

              using complex_type = ::ket::utility::meta::range_value_t<RandomAccessRange>;
              using real_type = ::ket::utility::meta::real_t<complex_type>;
              using boost::math::constants::half;
              constexpr auto half_one_plus_i = complex_type{half<real_type>(), half<real_type>()};
              constexpr auto half_one_minus_i = complex_type{half<real_type>(), -half<real_type>()};
              *control_on_iter *= half_one_minus_i;
              *control_on_iter += half_one_plus_i * *target_control_on_iter;
              *target_control_on_iter *= half_one_minus_i;
              *target_control_on_iter += half_one_plus_i * control_on_iter_value;
            });
        }

        // cx_tp: only target qubit is on page
        // CsX+_{tc} or C1sX+_{tc}
        // CsX+_{1,2} (a_{00} |00> + a_{01} |01> + a_{10} |10> + a_{11} |11>)
        //   = a_{00} |00> + a_{01} |01> + [(1-i) a_{10} + (1+i) a_{11}]/2 |10> + [(1+i) a_{10} + (1-i) a_{11}]/2 |11>
        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger>
        inline auto adj_sqrt_pauli_cx_tp(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_target_qubit,
          ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const permutated_control_qubit)
        -> RandomAccessRange&
        {
          assert(not ::ket::mpi::page::is_on_page(permutated_control_qubit, local_state));
          auto const control_qubit_mask
            = ::ket::utility::integer_exp2<StateInteger>(permutated_control_qubit);
          auto const nonpage_lower_bits_mask = control_qubit_mask - StateInteger{1u};
          auto const nonpage_upper_bits_mask = compl nonpage_lower_bits_mask;

          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<1u>(
            parallel_policy, local_state, permutated_target_qubit,
            [control_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask](
              auto const zero_first, auto const one_first, StateInteger const index_wo_nonpage_qubit, int const)
            {
              auto const zero_index
                = ((index_wo_nonpage_qubit bitand nonpage_upper_bits_mask) << 1u)
                  bitor (index_wo_nonpage_qubit bitand nonpage_lower_bits_mask);
              auto const one_index = zero_index bitor control_qubit_mask;
              auto const control_on_iter = zero_first + one_index;
              auto const target_control_on_iter = one_first + one_index;
              auto const control_on_iter_value = *control_on_iter;

              using complex_type = ::ket::utility::meta::range_value_t<RandomAccessRange>;
              using real_type = ::ket::utility::meta::real_t<complex_type>;
              using boost::math::constants::half;
              constexpr auto half_one_plus_i = complex_type{half<real_type>(), half<real_type>()};
              constexpr auto half_one_minus_i = complex_type{half<real_type>(), -half<real_type>()};
              *control_on_iter *= half_one_minus_i;
              *control_on_iter += half_one_plus_i * *target_control_on_iter;
              *target_control_on_iter *= half_one_minus_i;
              *target_control_on_iter += half_one_plus_i * control_on_iter_value;
            });
        }

        // cx_cp: only control qubit is on page
        // CsX+_{tc} or C1sX+_{tc}
        // CsX+_{1,2} (a_{00} |00> + a_{01} |01> + a_{10} |10> + a_{11} |11>)
        //   = a_{00} |00> + a_{01} |01> + [(1-i) a_{10} + (1+i) a_{11}]/2 |10> + [(1+i) a_{10} + (1-i) a_{11}]/2 |11>
        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger>
        inline auto adj_sqrt_pauli_cx_cp(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_target_qubit,
          ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const permutated_control_qubit)
        -> RandomAccessRange&
        {
          assert(not ::ket::mpi::page::is_on_page(permutated_target_qubit, local_state));

          auto const target_qubit_mask
            = ::ket::utility::integer_exp2<StateInteger>(permutated_target_qubit);
          auto const nonpage_lower_bits_mask = target_qubit_mask - StateInteger{1u};
          auto const nonpage_upper_bits_mask = compl nonpage_lower_bits_mask;

          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<1u>(
            parallel_policy, local_state, permutated_control_qubit,
            [target_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask](
              auto const, auto const one_first, StateInteger const index_wo_nonpage_qubit, int const)
            {
              auto const zero_index
                = ((index_wo_nonpage_qubit bitand nonpage_upper_bits_mask) << 1u)
                  bitor (index_wo_nonpage_qubit bitand nonpage_lower_bits_mask);
              auto const one_index = zero_index bitor target_qubit_mask;
              auto const control_on_iter = one_first + zero_index;
              auto const target_control_on_iter = one_first + one_index;
              auto const control_on_iter_value = *control_on_iter;

              using complex_type = ::ket::utility::meta::range_value_t<RandomAccessRange>;
              using real_type = ::ket::utility::meta::real_t<complex_type>;
              using boost::math::constants::half;
              constexpr auto half_one_plus_i = complex_type{half<real_type>(), half<real_type>()};
              constexpr auto half_one_minus_i = complex_type{half<real_type>(), -half<real_type>()};
              *control_on_iter *= half_one_minus_i;
              *control_on_iter += half_one_plus_i * *target_control_on_iter;
              *target_control_on_iter *= half_one_minus_i;
              *target_control_on_iter += half_one_plus_i * control_on_iter_value;
            });
        }
      } // namespace page
    } // namespace gate
  } // namespace mpi
} // namespace ket


#endif // KET_MPI_GATE_PAGE_SQRT_PAULI_X_HPP
