#ifndef KET_MPI_GATE_PAGE_DETAIL_PAULI_CZ_P_DIAGONAL_HPP
# define KET_MPI_GATE_PAGE_DETAIL_PAULI_CZ_P_DIAGONAL_HPP

# include <cassert>
# include <iterator>
# include <utility>

# include <ket/qubit.hpp>
# include <ket/control.hpp>
# include <ket/utility/integer_exp2.hpp>
# include <ket/utility/integer_log2.hpp>
# include <ket/utility/meta/real_of.hpp>
# include <ket/mpi/qubit_permutation.hpp>
# include <ket/mpi/state.hpp>
# include <ket/mpi/page/is_on_page.hpp>
# include <ket/mpi/gate/page/unsupported_page_gate_operation.hpp>
# include <ket/mpi/gate/page/detail/one_page_qubit_gate.hpp>
# include <ket/mpi/utility/simple_mpi.hpp>
# include <ket/mpi/utility/unit_mpi.hpp>


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
          namespace pauli_cz_p_detail
          {
            template <
              typename ParallelPolicy,
              typename Complex, typename Allocator, typename StateInteger, typename BitInteger>
            inline auto pauli_cz_p_l(
              ParallelPolicy const parallel_policy,
              ::ket::mpi::state<Complex, true, Allocator>& local_state,
              ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const page_permutated_control_qubit,
              ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const nonpage_permutated_control_qubit)
            -> ::ket::mpi::state<Complex, true, Allocator>&
            {
              assert(::ket::mpi::page::is_on_page(page_permutated_control_qubit, local_state));
              assert(not ::ket::mpi::page::is_on_page(nonpage_permutated_control_qubit, local_state));

              auto const nonpage_permutated_control_qubit_mask
                = ::ket::utility::integer_exp2<StateInteger>(nonpage_permutated_control_qubit);
              auto const nonpage_lower_bits_mask = nonpage_permutated_control_qubit_mask - StateInteger{1u};
              auto const nonpage_upper_bits_mask = compl nonpage_lower_bits_mask;

              return ::ket::mpi::gate::page::detail::one_page_qubit_gate<1u>(
                parallel_policy, local_state, page_permutated_control_qubit,
                [nonpage_permutated_control_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask](
                  auto const, auto const one_first, StateInteger const index_wo_nonpage_qubit, int const)
                {
                  auto const zero_index
                    = ((index_wo_nonpage_qubit bitand nonpage_upper_bits_mask) << 1u)
                      bitor (index_wo_nonpage_qubit bitand nonpage_lower_bits_mask);
                  auto const one_index = zero_index bitor nonpage_permutated_control_qubit_mask;

                  using real_type = ::ket::utility::meta::real_t<Complex>;
                  *(one_first + one_index) *= real_type{-1.0};
                });
            }

            template <
              typename StateInteger, typename BitInteger, typename NumProcesses,
              typename ParallelPolicy, typename Complex, typename Allocator>
            [[noreturn]] inline auto pauli_cz_p_u(
              ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> const& mpi_policy,
              ParallelPolicy const parallel_policy,
              ::ket::mpi::state<Complex, false, Allocator>& local_state,
              ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const page_permutated_control_qubit,
              ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const nonpage_permutated_control_qubit,
              ::yampi::rank const rank,
              ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const least_permutated_unit_qubit)
            -> ::ket::mpi::state<Complex, false, Allocator>&
            { throw ::ket::mpi::gate::page::unsupported_page_gate_operation{"pauli_cz_p_cu"}; }

            template <
              typename StateInteger, typename BitInteger, typename NumProcesses,
              typename ParallelPolicy, typename Complex, typename Allocator>
            inline auto pauli_cz_p_u(
              ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> const& mpi_policy,
              ParallelPolicy const parallel_policy,
              ::ket::mpi::state<Complex, true, Allocator>& local_state,
              ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const page_permutated_control_qubit,
              ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const nonpage_permutated_control_qubit,
              ::yampi::rank const rank,
              ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const least_permutated_unit_qubit)
            -> ::ket::mpi::state<Complex, true, Allocator>&
            {
              assert(not ::ket::mpi::page::is_on_page(nonpage_permutated_control_qubit, local_state));

              auto const nonpage_permutated_control_qubit_mask
                = StateInteger{1u} << (nonpage_permutated_control_qubit - least_permutated_unit_qubit);

              auto const num_nonpage_local_qubits
                = static_cast<BitInteger>(local_state.num_local_qubits() - local_state.num_page_qubits());
              auto const page_permutated_control_qubit_mask
                = ::ket::utility::integer_exp2<StateInteger>(
                    page_permutated_control_qubit - static_cast<BitInteger>(num_nonpage_local_qubits));
              auto const lower_bits_mask = page_permutated_control_qubit_mask - StateInteger{1u};
              auto const upper_bits_mask = compl lower_bits_mask;
              auto const num_pages = local_state.num_pages();

              auto const num_data_blocks = static_cast<StateInteger>(local_state.num_data_blocks());
              auto const rank_in_unit = ::ket::mpi::utility::policy::rank_in_unit(mpi_policy, rank);
              for (auto data_block_index = StateInteger{0u}; data_block_index < num_data_blocks; ++data_block_index)
              {
                auto const unit_qubit_value = ::ket::mpi::utility::policy::unit_qubit_value(mpi_policy, data_block_index, rank_in_unit);
                if ((unit_qubit_value bitand nonpage_permutated_control_qubit_mask) == StateInteger{0u})
                  continue;

                // nonpage_qubit is |1>
                for (auto page_index_wo_qubit = std::size_t{0u}; page_index_wo_qubit < num_pages / 2u; ++page_index_wo_qubit)
                {
                  // x0x
                  auto const zero_page_index
                    = ((page_index_wo_qubit bitand upper_bits_mask) << 1u)
                      bitor (page_index_wo_qubit bitand lower_bits_mask);
                  // x1x
                  auto const one_page_index = zero_page_index bitor page_permutated_control_qubit_mask;

                  auto const one_page_range = local_state.page_range(std::make_pair(data_block_index, one_page_index));
                  using std::begin;
                  auto const one_first = begin(one_page_range);

                  using std::end;
                  ::ket::utility::loop_n(
                    parallel_policy,
                    static_cast<StateInteger>(std::distance(begin(one_page_range), end(one_page_range))),
                    [one_first](StateInteger const index, int const)
                    {
                      using real_type = ::ket::utility::meta::real_t<Complex>;
                      *(one_first + index) *= static_cast<real_type>(-1);
                    });
                }
              }

              return local_state;
            }

            template <
              typename ParallelPolicy,
              typename Complex, typename Allocator, typename StateInteger, typename BitInteger>
            inline auto pauli_cz_p_g(
              ParallelPolicy const parallel_policy,
              ::ket::mpi::state<Complex, true, Allocator>& local_state,
              ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const page_permutated_control_qubit,
              ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const nonpage_permutated_control_qubit,
              ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const least_permutated_global_qubit,
              StateInteger const global_qubit_value)
            -> ::ket::mpi::state<Complex, true, Allocator>&
            {
              assert(not ::ket::mpi::page::is_on_page(nonpage_permutated_control_qubit, local_state));

              auto const nonpage_permutated_control_qubit_mask
                = StateInteger{1u} << (nonpage_permutated_control_qubit - least_permutated_global_qubit);

              if ((global_qubit_value bitand nonpage_permutated_control_qubit_mask) == StateInteger{0u})
                return local_state;

              // nonpage_qubit is |1>
              return ::ket::mpi::gate::page::detail::one_page_qubit_gate<0u>(
                parallel_policy, local_state, page_permutated_control_qubit,
                [](auto const, auto const one_first, StateInteger const index, int const)
                {
                  using real_type = ::ket::utility::meta::real_t<Complex>;
                  *(one_first + index) *= static_cast<real_type>(-1);
                });
            }
          } // namespace pauli_cz_p_detail

          template <
            typename MpiPolicy, typename ParallelPolicy,
            typename RandomAccessRange, typename StateInteger, typename BitInteger>
          [[noreturn]] inline auto pauli_cz_p(
            MpiPolicy const&, ParallelPolicy const,
            RandomAccessRange& local_state,
            ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const,
            ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const,
            yampi::rank const)
          -> RandomAccessRange&
          { throw ::ket::mpi::gate::page::unsupported_page_gate_operation{"pauli_cz_p"}; }

          template <
            typename ParallelPolicy,
            typename Complex, typename Allocator, typename StateInteger, typename BitInteger>
          [[noreturn]] inline auto pauli_cz_p(
            ::ket::mpi::utility::policy::simple_mpi const, ParallelPolicy const,
            ::ket::mpi::state<Complex, false, Allocator>& local_state,
            ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const,
            ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const,
            yampi::rank const)
          -> ::ket::mpi::state<Complex, false, Allocator>&
          { throw ::ket::mpi::gate::page::unsupported_page_gate_operation{"pauli_cz_p"}; }

          template <
            typename StateInteger, typename BitInteger, typename NumProcesses,
            typename ParallelPolicy, typename Complex, typename Allocator>
          [[noreturn]] inline auto pauli_cz_p(
            ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> const&,
            ParallelPolicy const,
            ::ket::mpi::state<Complex, false, Allocator>& local_state,
            ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const,
            ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const,
            yampi::rank const)
          -> ::ket::mpi::state<Complex, false, Allocator>&
          { throw ::ket::mpi::gate::page::unsupported_page_gate_operation{"pauli_cz_p"}; }

          template <
            typename ParallelPolicy,
            typename Complex, typename Allocator, typename StateInteger, typename BitInteger>
          inline auto pauli_cz_p(
            ::ket::mpi::utility::policy::simple_mpi const mpi_policy,
            ParallelPolicy const parallel_policy,
            ::ket::mpi::state<Complex, true, Allocator>& local_state,
            ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const page_permutated_control_qubit,
            ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const nonpage_permutated_control_qubit,
            yampi::rank const rank)
          -> ::ket::mpi::state<Complex, true, Allocator>&
          {
            assert(::ket::mpi::page::is_on_page(page_permutated_control_qubit, local_state));
            assert(not ::ket::mpi::page::is_on_page(nonpage_permutated_control_qubit, local_state));

            using permutated_qubit_type = ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> >;
            auto const least_permutated_global_qubit = permutated_qubit_type{static_cast<BitInteger>(local_state.num_local_qubits())};

            if (nonpage_permutated_control_qubit < least_permutated_global_qubit)
              return ::ket::mpi::gate::page::detail::pauli_cz_p_detail::pauli_cz_p_l(
                parallel_policy, local_state, page_permutated_control_qubit, nonpage_permutated_control_qubit);

            return ::ket::mpi::gate::page::detail::pauli_cz_p_detail::pauli_cz_p_g(
              parallel_policy,
              local_state, page_permutated_control_qubit, nonpage_permutated_control_qubit,
              least_permutated_global_qubit,
              static_cast<StateInteger>(::ket::mpi::utility::policy::global_qubit_value(mpi_policy, rank)));
          }

          template <
            typename StateInteger, typename BitInteger, typename NumProcesses,
            typename ParallelPolicy, typename Complex, typename Allocator>
          inline auto pauli_cz_p(
            ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> const& mpi_policy,
            ParallelPolicy const parallel_policy,
            ::ket::mpi::state<Complex, true, Allocator>& local_state,
            ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const page_permutated_control_qubit,
            ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const nonpage_permutated_control_qubit,
            yampi::rank const rank)
          -> ::ket::mpi::state<Complex, true, Allocator>&
          {
            assert(::ket::mpi::page::is_on_page(page_permutated_control_qubit, local_state));
            assert(not ::ket::mpi::page::is_on_page(nonpage_permutated_control_qubit, local_state));

            using permutated_qubit_type = ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> >;
            auto const least_permutated_unit_qubit = permutated_qubit_type{static_cast<BitInteger>(local_state.num_local_qubits())};
            auto const least_permutated_global_qubit = least_permutated_unit_qubit + static_cast<BitInteger>(mpi_policy.num_unit_qubits());

            if (nonpage_permutated_control_qubit < least_permutated_unit_qubit)
              return ::ket::mpi::gate::page::detail::pauli_cz_p_detail::pauli_cz_p_l(
                parallel_policy, local_state, page_permutated_control_qubit, nonpage_permutated_control_qubit);

            if (nonpage_permutated_control_qubit < least_permutated_global_qubit)
              return ::ket::mpi::gate::page::detail::pauli_cz_p_detail::pauli_cz_p_u(
                mpi_policy, parallel_policy,
                local_state, page_permutated_control_qubit, nonpage_permutated_control_qubit,
                rank, least_permutated_unit_qubit);

            return ::ket::mpi::gate::page::detail::pauli_cz_p_detail::pauli_cz_p_g(
              parallel_policy,
              local_state, page_permutated_control_qubit, nonpage_permutated_control_qubit,
              least_permutated_global_qubit,
              static_cast<StateInteger>(::ket::mpi::utility::policy::global_qubit_value(mpi_policy, rank)));
          }
        } // namespace detail
      } // namespace page
    } // namespace gate
  } // namespace mpi
} // namespace ket


#endif // KET_MPI_GATE_PAGE_DETAIL_PAULI_CZ_P_DIAGONAL_HPP
