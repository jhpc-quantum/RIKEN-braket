#ifndef KET_MPI_GATE_PAGE_CONTROLLED_V_HPP
# define KET_MPI_GATE_PAGE_CONTROLLED_V_HPP

# include <cassert>
# include <algorithm>

# include <boost/range/size.hpp>

# include <ket/qubit.hpp>
# include <ket/utility/loop_n.hpp>
# include <ket/utility/integer_exp2.hpp>
# include <ket/utility/exp_i.hpp>
# include <ket/utility/begin.hpp>
# include <ket/mpi/qubit_permutation.hpp>
# include <ket/mpi/state.hpp>


namespace ket
{
  namespace mpi
  {
    namespace gate
    {
      namespace page
      {
        // tcp: both of target qubit and control qubit are on page
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename Complex,
          typename StateInteger, typename BitInteger, typename Allocator>
        inline RandomAccessRange& controlled_v_coeff_tcp(
          MpiPolicy const, ParallelPolicy const,
          RandomAccessRange& local_state,
          Complex const&,
          ::ket::qubit<StateInteger, BitInteger> const,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, Allocator> const&)
        { return local_state; }

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename Complex, typename StateAllocator,
          typename StateInteger, typename BitInteger, typename Allocator>
        inline ::ket::mpi::state<Complex, 0, StateAllocator>&
        controlled_v_coeff_tcp(
          MpiPolicy const, ParallelPolicy const,
          ::ket::mpi::state<Complex, 0, StateAllocator>& local_state,
          Complex const&,
          ::ket::qubit<StateInteger, BitInteger> const,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, Allocator> const&)
        { return local_state; }

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename Complex, typename StateAllocator,
          typename StateInteger, typename BitInteger, typename Allocator>
        inline ::ket::mpi::state<Complex, 1, StateAllocator>&
        controlled_v_coeff_tcp(
          MpiPolicy const, ParallelPolicy const,
          ::ket::mpi::state<Complex, 1, StateAllocator>& local_state,
          Complex const&,
          ::ket::qubit<StateInteger, BitInteger> const,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, Allocator> const&)
        { return local_state; }

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename Complex, int num_page_qubits_, typename StateAllocator,
          typename StateInteger, typename BitInteger, typename PermutationAllocator>
        inline ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator>&
        controlled_v_coeff_tcp(
          MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
          ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator>& local_state,
          Complex const& phase_coefficient,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, PermutationAllocator> const& permutation)
        {
          static_assert(
            num_page_qubits_ >= 2,
            "num_page_qubits_ should be greater than or equal to 2");

          auto const permutated_target_qubit = permutation[target_qubit];
          auto const permutated_cqubit = permutation[control_qubit.qubit()];
          assert(local_state.is_page_qubit(permutated_target_qubit));
          assert(local_state.is_page_qubit(permutated_cqubit));

          auto const num_nonpage_qubits
            = static_cast<BitInteger>(local_state.num_local_qubits() - num_page_qubits_);

          using real_type = typename ::ket::utility::meta::real_of<Complex>::type;
          auto const one_plus_phase_coefficient = real_type{1} + phase_coefficient;
          auto const one_minus_phase_coefficient = real_type{1} - phase_coefficient;

          auto const minmax_qubits = std::minmax(permutated_target_qubit, permutated_cqubit);
          auto const target_qubit_mask
            = ::ket::utility::integer_exp2<StateInteger>(
                permutated_target_qubit - static_cast<BitInteger>(num_nonpage_qubits));
          auto const control_qubit_mask
            = ::ket::utility::integer_exp2<StateInteger>(
                permutated_cqubit - static_cast<BitInteger>(num_nonpage_qubits));
          auto const lower_bits_mask
            = ::ket::utility::integer_exp2<StateInteger>(
                minmax_qubits.first - static_cast<BitInteger>(num_nonpage_qubits))
              - StateInteger{1u};
          auto const middle_bits_mask
            = (::ket::utility::integer_exp2<StateInteger>(
                 minmax_qubits.second - (BitInteger{1u} + num_nonpage_qubits)) - StateInteger{1u})
              xor lower_bits_mask;
          auto const upper_bits_mask = compl (lower_bits_mask bitor middle_bits_mask);

          static constexpr auto num_pages
            = ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator>::num_pages;
          for (auto page_id_wo_qubits = std::size_t{0u};
               page_id_wo_qubits < num_pages / 4u; ++page_id_wo_qubits)
          {
            // x0_tx0_cx
            auto const base_page_id
              = ((page_id_wo_qubits bitand upper_bits_mask) << 2u)
                bitor ((page_id_wo_qubits bitand middle_bits_mask) << 1u)
                bitor (page_id_wo_qubits bitand lower_bits_mask);
            // x0_tx1_cx
            auto const page_id_01 = base_page_id bitor control_qubit_mask;
            // x1_tx1_cx
            auto const page_id_11 = page_id_01 bitor target_qubit_mask;

            auto page_range_01 = local_state.page_range(page_id_01);
            auto page_range_11 = local_state.page_range(page_id_11);

            auto const page_first_01 = ::ket::utility::begin(page_range_01);
            auto const page_first_11 = ::ket::utility::begin(page_range_11);

            using ::ket::utility::loop_n;
            loop_n(
              parallel_policy,
              boost::size(page_range_11),
              [page_first_01, page_first_11,
               &one_plus_phase_coefficient, &one_minus_phase_coefficient](
                StateInteger const index, int const)
              {
                auto const page_iter_01 = page_first_01 + index;
                auto const page_iter_11 = page_first_11 + index;
                auto const value_01 = *page_iter_01;

                using boost::math::constants::half;
                *page_iter_01
                  = half<real_type>()
                    * (one_plus_phase_coefficient * value_01
                       + one_minus_phase_coefficient * *page_iter_11);
                *page_iter_11
                  = half<real_type>()
                    * (one_minus_phase_coefficient * value_01
                       + one_plus_phase_coefficient * *page_iter_11);
              });
          }

          return local_state;
        }

        // tp: only target qubit is on page
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename Complex,
          typename StateInteger, typename BitInteger, typename Allocator>
        inline RandomAccessRange& controlled_v_coeff_tp(
          MpiPolicy const, ParallelPolicy const,
          RandomAccessRange& local_state,
          Complex const&,
          ::ket::qubit<StateInteger, BitInteger> const,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, Allocator> const&)
        { return local_state; }

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename Complex, typename StateAllocator,
          typename StateInteger, typename BitInteger, typename Allocator>
        inline ::ket::mpi::state<Complex, 0, StateAllocator>&
        controlled_v_coeff_tp(
          MpiPolicy const, ParallelPolicy const,
          ::ket::mpi::state<Complex, 0, StateAllocator>& local_state,
          Complex const&,
          ::ket::qubit<StateInteger, BitInteger> const,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, Allocator> const&)
        { return local_state; }

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename Complex, int num_page_qubits_, typename StateAllocator,
          typename StateInteger, typename BitInteger, typename PermutationAllocator>
        inline ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator>&
        controlled_v_coeff_tp(
          MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
          ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator>& local_state,
          Complex const& phase_coefficient,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, PermutationAllocator> const& permutation)
        {
          static_assert(
            num_page_qubits_ >= 1,
            "num_page_qubits_ should be greater than or equal to 1");
          assert(local_state.is_page_qubit(permutation[target_qubit]));
          assert(not local_state.is_page_qubit(permutation[control_qubit.qubit()]));

          auto const num_nonpage_qubits
            = static_cast<BitInteger>(local_state.num_local_qubits() - num_page_qubits_);

          using real_type = typename ::ket::utility::meta::real_of<Complex>::type;
          auto const one_plus_phase_coefficient = real_type{1} + phase_coefficient;
          auto const one_minus_phase_coefficient = real_type{1} - phase_coefficient;

          auto const target_qubit_mask
            = ::ket::utility::integer_exp2<StateInteger>(
                permutation[target_qubit] - static_cast<BitInteger>(num_nonpage_qubits));
          auto const control_qubit_mask
            = ::ket::utility::integer_exp2<StateInteger>(permutation[control_qubit.qubit()]);
          auto const page_lower_bits_mask = target_qubit_mask - StateInteger{1u};
          auto const nonpage_lower_bits_mask = control_qubit_mask - StateInteger{1u};
          auto const page_upper_bits_mask = compl page_lower_bits_mask;
          auto const nonpage_upper_bits_mask = compl nonpage_lower_bits_mask;

          static constexpr auto num_pages
            = ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator>::num_pages;
          for (auto page_id_wo_qubit = std::size_t{0u};
               page_id_wo_qubit < num_pages / 2u; ++page_id_wo_qubit)
          {
            // x0x
            auto const zero_page_id
              = ((page_id_wo_qubit bitand page_upper_bits_mask) << 1u)
                bitor (page_id_wo_qubit bitand page_lower_bits_mask);
            // x1x
            auto const one_page_id = zero_page_id bitor target_qubit_mask;

            auto zero_page_range = local_state.page_range(zero_page_id);
            auto one_page_range = local_state.page_range(one_page_id);

            auto const zero_first = ::ket::utility::begin(zero_page_range);
            auto const one_first = ::ket::utility::begin(one_page_range);

            using ::ket::utility::loop_n;
            loop_n(
              parallel_policy,
              boost::size(one_page_range) / 2u,
              [zero_first, one_first,
               &one_plus_phase_coefficient, &one_minus_phase_coefficient,
               control_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask](
                StateInteger const index_wo_qubit, int const)
              {
                auto const zero_index
                  = ((index_wo_qubit bitand nonpage_upper_bits_mask) << 1u)
                    bitor (index_wo_qubit bitand nonpage_lower_bits_mask);
                auto const one_index = zero_index bitor control_qubit_mask;

                auto const control_on_iter = zero_first + one_index;
                auto const target_control_on_iter = one_first + one_index;
                auto const control_on_value = *control_on_iter;

                using boost::math::constants::half;
                *control_on_iter
                  = half<real_type>()
                    * (one_plus_phase_coefficient * control_on_value
                       + one_minus_phase_coefficient * *target_control_on_iter);
                *target_control_on_iter
                  = half<real_type>()
                    * (one_minus_phase_coefficient * control_on_value
                       + one_plus_phase_coefficient * *target_control_on_iter);
              });
          }

          return local_state;
        }

        // cp: only control qubit is on page
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename Complex,
          typename StateInteger, typename BitInteger, typename Allocator>
        inline RandomAccessRange& controlled_v_coeff_cp(
          MpiPolicy const, ParallelPolicy const,
          RandomAccessRange& local_state,
          Complex const&,
          ::ket::qubit<StateInteger, BitInteger> const,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, Allocator> const&)
        { return local_state; }

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename Complex, typename StateAllocator,
          typename StateInteger, typename BitInteger, typename Allocator>
        inline ::ket::mpi::state<Complex, 0, StateAllocator>&
        controlled_v_coeff_cp(
          MpiPolicy const, ParallelPolicy const,
          ::ket::mpi::state<Complex, 0, StateAllocator>& local_state,
          Complex const&,
          ::ket::qubit<StateInteger, BitInteger> const,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, Allocator> const&)
        { return local_state; }

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename Complex, int num_page_qubits_, typename StateAllocator,
          typename StateInteger, typename BitInteger, typename PermutationAllocator>
        inline ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator>&
        controlled_v_coeff_cp(
          MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
          ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator>& local_state,
          Complex const& phase_coefficient,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, PermutationAllocator> const& permutation)
        {
          static_assert(
            num_page_qubits_ >= 1,
            "num_page_qubits_ should be greater than or equal to 1");
          assert(not local_state.is_page_qubit(permutation[target_qubit]));
          assert(local_state.is_page_qubit(permutation[control_qubit.qubit()]));

          auto const num_nonpage_qubits
            = static_cast<BitInteger>(local_state.num_local_qubits() - num_page_qubits_);

          using real_type = typename ::ket::utility::meta::real_of<Complex>::type;
          auto const one_plus_phase_coefficient = real_type{1} + phase_coefficient;
          auto const one_minus_phase_coefficient = real_type{1} - phase_coefficient;

          auto const target_qubit_mask
            = ::ket::utility::integer_exp2<StateInteger>(permutation[target_qubit]);
          auto const control_qubit_mask
            = ::ket::utility::integer_exp2<StateInteger>(
                permutation[control_qubit.qubit()] - static_cast<BitInteger>(num_nonpage_qubits));
          auto const page_lower_bits_mask = control_qubit_mask - StateInteger{1u};
          auto const nonpage_lower_bits_mask = target_qubit_mask - StateInteger{1u};
          auto const page_upper_bits_mask = compl page_lower_bits_mask;
          auto const nonpage_upper_bits_mask = compl nonpage_lower_bits_mask;

          static constexpr auto num_pages
            = ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator>::num_pages;
          for (auto page_id_wo_qubit = std::size_t{0u};
               page_id_wo_qubit < num_pages / 2u; ++page_id_wo_qubit)
          {
            // x0x
            auto const zero_page_id
              = ((page_id_wo_qubit bitand page_upper_bits_mask) << 1u)
                bitor (page_id_wo_qubit bitand page_lower_bits_mask);
            // x1x
            auto const one_page_id = zero_page_id bitor control_qubit_mask;

            auto one_page_range = local_state.page_range(one_page_id);
            auto const one_page_first = ::ket::utility::begin(one_page_range);

            using ::ket::utility::loop_n;
            loop_n(
              parallel_policy,
              boost::size(one_page_range) / 2u,
              [one_page_first, &one_plus_phase_coefficient, &one_minus_phase_coefficient,
               target_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask](
                StateInteger const index_wo_qubit, int const)
              {
                auto const zero_index
                  = ((index_wo_qubit bitand nonpage_upper_bits_mask) << 1u)
                    bitor (index_wo_qubit bitand nonpage_lower_bits_mask);
                auto const one_index = zero_index bitor target_qubit_mask;

                auto const control_on_iter = one_page_first + zero_index;
                auto const target_control_on_iter = one_page_first + one_index;
                auto const control_on_value = *control_on_iter;

                using boost::math::constants::half;
                *control_on_iter
                  = half<real_type>()
                    * (one_plus_phase_coefficient * control_on_value
                       + one_minus_phase_coefficient * *target_control_on_iter);
                *target_control_on_iter
                  = half<real_type>()
                    * (one_minus_phase_coefficient * control_on_value
                       + one_plus_phase_coefficient * *target_control_on_iter);
              });
          }

          return local_state;
        }
      } // namespace page
    } // namespace gate
  } // namespace mpi
} // namespace ket


#endif // KET_MPI_GATE_PAGE_CONTROLLED_V_HPP
