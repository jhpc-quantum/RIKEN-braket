#ifndef KET_USE_DIAGONAL_LOOP
# include <ket/mpi/gate/page/controlled_phase_shift.hpp.old>
#else // KET_USE_DIAGONAL_LOOP
//
#ifndef KET_MPI_GATE_PAGE_CONTROLLED_PHASE_SHIFT_HPP
# define KET_MPI_GATE_PAGE_CONTROLLED_PHASE_SHIFT_HPP

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
        inline RandomAccessRange& controlled_phase_shift_coeff_tcp(
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
        controlled_phase_shift_coeff_tcp(
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
        controlled_phase_shift_coeff_tcp(
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
        controlled_phase_shift_coeff_tcp(
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

          auto const minmax_qubits = std::minmax(permutated_target_qubit, permutated_cqubit);
          auto const target_qubit_mask
            = ::ket::utility::integer_exp2<StateInteger>(
                permutated_target_qubit - static_cast<BitInteger>(num_nonpage_qubits));
          auto const control_qubit_mask
            = ::ket::utility::integer_exp2<StateInteger>(
                permutated_cqubit - static_cast<BitInteger>(num_nonpage_qubits));
          auto const lower_bits_mask
            = ::ket::utility::integer_exp2<StateInteger>(
                minmax_qubits.first - static_cast<BitInteger>(num_nonpage_qubits)) - StateInteger{1u};
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
            // x1_tx1_cx
            auto const page_id_11
              = base_page_id bitor control_qubit_mask bitor target_qubit_mask;

            auto page_range_11 = local_state.page_range(page_id_11);
            auto const first_11 = ::ket::utility::begin(page_range_11);

            using ::ket::utility::loop_n;
            loop_n(
              parallel_policy,
              boost::size(page_range_11),
              [first_11, &phase_coefficient](StateInteger const index, int const)
              { *(first_11 + index) *= phase_coefficient; });
          }

          return local_state;
        }

        // tp: only target qubit is on page
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename Complex,
          typename StateInteger, typename BitInteger, typename Allocator>
        inline RandomAccessRange& controlled_phase_shift_coeff_tp(
          MpiPolicy const, ParallelPolicy const,
          RandomAccessRange& local_state,
          Complex const&,
          ::ket::qubit<StateInteger, BitInteger> const,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, Allocator> const&,
          yampi::rank const)
        { return local_state; }

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename Complex, typename StateAllocator,
          typename StateInteger, typename BitInteger, typename Allocator>
        inline ::ket::mpi::state<Complex, 0, StateAllocator>&
        controlled_phase_shift_coeff_tp(
          MpiPolicy const, ParallelPolicy const,
          ::ket::mpi::state<Complex, 0, StateAllocator>& local_state,
          Complex const&,
          ::ket::qubit<StateInteger, BitInteger> const,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, Allocator> const&,
          yampi::rank const)
        { return local_state; }

        namespace controlled_phase_shift_detail
        {
          // tp_cl: target qubit is on page and control qubit is local
          template <
            typename MpiPolicy, typename ParallelPolicy,
            typename Complex, int num_page_qubits_, typename StateAllocator,
            typename StateInteger, typename BitInteger, typename PermutationAllocator>
          inline ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator>&
          controlled_phase_shift_coeff_tp_cl(
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

              auto one_page_range = local_state.page_range(one_page_id);
              auto const one_first = ::ket::utility::begin(one_page_range);

              using ::ket::utility::loop_n;
              loop_n(
                parallel_policy,
                boost::size(one_page_range) / 2u,
                [one_first, &phase_coefficient,
                 control_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask](
                  StateInteger const index_wo_qubit, int const)
                {
                  auto const zero_index
                    = ((index_wo_qubit bitand nonpage_upper_bits_mask) << 1u)
                      bitor (index_wo_qubit bitand nonpage_lower_bits_mask);
                  auto const one_index = zero_index bitor control_qubit_mask;
                  *(one_first + one_index) *= phase_coefficient;
                });
            }

            return local_state;
          }

          // tp_cg: target qubit is on page and control qubit is global
          template <
            typename MpiPolicy, typename ParallelPolicy,
            typename Complex, int num_page_qubits_, typename StateAllocator,
            typename StateInteger, typename BitInteger, typename PermutationAllocator>
          inline ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator>&
          controlled_phase_shift_coeff_tp_cg(
            MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
            ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator>& local_state,
            Complex const& phase_coefficient,
            ::ket::qubit<StateInteger, BitInteger> const target_qubit,
            ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
            ::ket::mpi::qubit_permutation<
              StateInteger, BitInteger, PermutationAllocator> const& permutation,
            yampi::rank const rank)
          {
            static_assert(
              num_page_qubits_ >= 1,
              "num_page_qubits_ should be greater than or equal to 1");
            assert(local_state.is_page_qubit(permutation[target_qubit]));
            assert(not local_state.is_page_qubit(permutation[control_qubit.qubit()]));

            using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
            auto const least_global_permutated_qubit
              = qubit_type{::ket::utility::integer_log2<BitInteger>(boost::size(local_state))};
            static constexpr auto one_state_integer = StateInteger{1u};
            static constexpr auto zero_state_integer = StateInteger{0u};

            auto const mask
              = one_state_integer
                << (permutation[control_qubit.qubit()] - least_global_permutated_qubit);

            if ((static_cast<StateInteger>(rank.mpi_rank()) bitand mask) == zero_state_integer)
              return local_state;

            auto const num_nonpage_qubits
              = static_cast<BitInteger>(local_state.num_local_qubits() - num_page_qubits_);

            auto const target_qubit_mask
              = ::ket::utility::integer_exp2<StateInteger>(
                  permutation[target_qubit] - static_cast<BitInteger>(num_nonpage_qubits));
            auto const page_lower_bits_mask = target_qubit_mask - StateInteger{1u};
            auto const page_upper_bits_mask = compl page_lower_bits_mask;

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

              auto one_page_range = local_state.page_range(one_page_id);
              auto const one_first = ::ket::utility::begin(one_page_range);

              using ::ket::utility::loop_n;
              loop_n(
                parallel_policy,
                boost::size(one_page_range),
                [one_first, &phase_coefficient](StateInteger const index, int const)
                { *(one_first + index) *= phase_coefficient; });
            }

            return local_state;
          }
        } // namespace controlled_phase_shift_detail

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename Complex, int num_page_qubits_, typename StateAllocator,
          typename StateInteger, typename BitInteger, typename PermutationAllocator>
        inline ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator>&
        controlled_phase_shift_coeff_tp(
          MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
          ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator>& local_state,
          Complex const& phase_coefficient,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, PermutationAllocator> const& permutation,
          yampi::rank const rank)
        {
          static_assert(num_page_qubits_ >= 1, "num_page_qubits_ should be greater than or equal to 1");
          assert(local_state.is_page_qubit(permutation[target_qubit]));
          assert(not local_state.is_page_qubit(permutation[control_qubit.qubit()]));

          using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
          auto const least_permutated_global_qubit
            = qubit_type{::ket::utility::integer_log2<BitInteger>(boost::size(local_state))};

          if (permutation[control_qubit.qubit()] < least_permutated_global_qubit)
            return ::ket::mpi::gate::page::controlled_phase_shift_detail
              ::controlled_phase_shift_coeff_tp_cl(
                mpi_policy, parallel_policy,
                local_state, phase_coefficient, target_qubit, control_qubit, permutation);

          return ::ket::mpi::gate::page::controlled_phase_shift_detail
            ::controlled_phase_shift_coeff_tp_cg(
              mpi_policy, parallel_policy,
              local_state, phase_coefficient, target_qubit, control_qubit, permutation, rank);
        }

        // cp: only control qubit is on page
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename Complex,
          typename StateInteger, typename BitInteger, typename Allocator>
        inline RandomAccessRange& controlled_phase_shift_coeff_cp(
          MpiPolicy const, ParallelPolicy const,
          RandomAccessRange& local_state,
          Complex const&,
          ::ket::qubit<StateInteger, BitInteger> const,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, Allocator> const&,
          yampi::rank const)
        { return local_state; }

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename Complex, typename StateAllocator,
          typename StateInteger, typename BitInteger, typename Allocator>
        inline ::ket::mpi::state<Complex, 0, StateAllocator>&
        controlled_phase_shift_coeff_cp(
          MpiPolicy const, ParallelPolicy const,
          ::ket::mpi::state<Complex, 0, StateAllocator>& local_state,
          Complex const&,
          ::ket::qubit<StateInteger, BitInteger> const,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, Allocator> const&,
          yampi::rank const)
        { return local_state; }

        namespace controlled_phase_shift_detail
        {
          // cp_tl: control qubit is on page and target qubit is local
          template <
            typename MpiPolicy, typename ParallelPolicy,
            typename Complex, int num_page_qubits_, typename StateAllocator,
            typename StateInteger, typename BitInteger, typename PermutationAllocator>
          inline ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator>&
          controlled_phase_shift_coeff_cp_tl(
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
              auto const one_first = ::ket::utility::begin(one_page_range);

              using ::ket::utility::loop_n;
              loop_n(
                parallel_policy,
                boost::size(one_page_range) / 2u,
                [one_first, &phase_coefficient,
                 target_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask](
                  StateInteger const index_wo_qubit, int const)
                {
                  auto const zero_index
                    = ((index_wo_qubit bitand nonpage_upper_bits_mask) << 1u)
                      bitor (index_wo_qubit bitand nonpage_lower_bits_mask);
                  auto const one_index = zero_index bitor target_qubit_mask;
                  *(one_first + one_index) *= phase_coefficient;
                });
            }

            return local_state;
          }

          // cp_tg: control qubit is on page and target qubit is global
          template <
            typename MpiPolicy, typename ParallelPolicy,
            typename Complex, int num_page_qubits_, typename StateAllocator,
            typename StateInteger, typename BitInteger, typename PermutationAllocator>
          inline ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator>&
          controlled_phase_shift_coeff_cp_tg(
            MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
            ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator>& local_state,
            Complex const& phase_coefficient,
            ::ket::qubit<StateInteger, BitInteger> const target_qubit,
            ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
            ::ket::mpi::qubit_permutation<
              StateInteger, BitInteger, PermutationAllocator> const& permutation,
            yampi::rank const rank)
          {
            static_assert(
              num_page_qubits_ >= 1,
              "num_page_qubits_ should be greater than or equal to 1");
            assert(not local_state.is_page_qubit(permutation[target_qubit]));
            assert(local_state.is_page_qubit(permutation[control_qubit.qubit()]));

            using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
            auto const least_global_permutated_qubit
              = qubit_type{::ket::utility::integer_log2<BitInteger>(boost::size(local_state))};
            static constexpr auto one_state_integer = StateInteger{1u};
            static constexpr auto zero_state_integer = StateInteger{0u};

            auto const mask
              = one_state_integer << (permutation[target_qubit] - least_global_permutated_qubit);

            if ((static_cast<StateInteger>(rank.mpi_rank()) bitand mask) == zero_state_integer)
              return local_state;

            auto const num_nonpage_qubits
              = static_cast<BitInteger>(local_state.num_local_qubits() - num_page_qubits_);

            auto const control_qubit_mask
              = ::ket::utility::integer_exp2<StateInteger>(
                  permutation[control_qubit.qubit()] - static_cast<BitInteger>(num_nonpage_qubits));
            auto const page_lower_bits_mask = control_qubit_mask - StateInteger{1u};
            auto const page_upper_bits_mask = compl page_lower_bits_mask;

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
              auto const one_first = ::ket::utility::begin(one_page_range);

              using ::ket::utility::loop_n;
              loop_n(
                parallel_policy,
                boost::size(one_page_range),
                [one_first, &phase_coefficient](StateInteger const index, int const)
                { *(one_first + index) *= phase_coefficient; });
            }

            return local_state;
          }
        } // namespace controlled_phase_shift_detail

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename Complex, int num_page_qubits_, typename StateAllocator,
          typename StateInteger, typename BitInteger, typename PermutationAllocator>
        inline ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator>&
        controlled_phase_shift_coeff_cp(
          MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
          ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator>& local_state,
          Complex const& phase_coefficient,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, PermutationAllocator> const& permutation,
          yampi::rank const rank)
        {
          static_assert(
            num_page_qubits_ >= 1,
            "num_page_qubits_ should be greater than or equal to 1");
          assert(not local_state.is_page_qubit(permutation[target_qubit]));
          assert(local_state.is_page_qubit(permutation[control_qubit.qubit()]));

          using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
          auto const least_permutated_global_qubit
            = qubit_type{::ket::utility::integer_log2<BitInteger>(boost::size(local_state))};

          if (permutation[target_qubit] < least_permutated_global_qubit)
            return ::ket::mpi::gate::page::controlled_phase_shift_detail
              ::controlled_phase_shift_coeff_cp_tl(
                mpi_policy, parallel_policy,
                local_state, phase_coefficient, target_qubit, control_qubit, permutation);

          return ::ket::mpi::gate::page::controlled_phase_shift_detail
            ::controlled_phase_shift_coeff_cp_tg(
              mpi_policy, parallel_policy,
              local_state, phase_coefficient, target_qubit, control_qubit, permutation, rank);
        }
      } // namespace page
    } // namespage gate
  } // namespace mpi
} // namespace ket


#endif // KET_MPI_GATE_PAGE_CONTROLLED_PHASE_SHIFT_HPP
//
#endif // KET_USE_DIAGONAL_LOOP
