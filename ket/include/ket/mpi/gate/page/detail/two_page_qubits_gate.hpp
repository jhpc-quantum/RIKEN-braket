#ifndef KET_MPI_GATE_PAGE_DETAIL_TWO_PAGE_QUBITS_GATE_HPP
# define KET_MPI_GATE_PAGE_DETAIL_TWO_PAGE_QUBITS_GATE_HPP

# include <boost/config.hpp>

# include <cstddef>
# include <cassert>
# include <algorithm>
# include <utility>

# include <boost/range/size.hpp>

# include <ket/qubit.hpp>
# include <ket/control.hpp>
# include <ket/utility/loop_n.hpp>
# include <ket/utility/integer_exp2.hpp>
# include <ket/utility/begin.hpp>
# include <ket/mpi/qubit_permutation.hpp>
# include <ket/mpi/state.hpp>
# include <ket/mpi/page/is_on_page.hpp>
# include <ket/mpi/gate/page/unsupported_page_gate_operation.hpp>


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
          template <
            std::size_t num_operated_nonpage_qubits,
            typename ParallelPolicy,
            typename RandomAccessRange,
            typename StateInteger, typename BitInteger, typename Allocator,
            typename Function>
          [[noreturn]] inline RandomAccessRange& two_page_qubits_gate(
            ParallelPolicy const,
            RandomAccessRange& local_state,
            ::ket::qubit<StateInteger, BitInteger> const,
            ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const,
            ::ket::mpi::qubit_permutation<
              StateInteger, BitInteger, Allocator> const&,
            Function&&)
          { throw ::ket::mpi::gate::page::unsupported_page_gate_operation<0, false>{"two_page_qubits_gate"}; }

          template <
            std::size_t num_operated_nonpage_qubits,
            typename ParallelPolicy,
            typename RandomAccessRange,
            typename StateInteger, typename BitInteger, typename Allocator,
            typename Function>
          [[noreturn]] inline RandomAccessRange& two_page_qubits_gate(
            ParallelPolicy const,
            RandomAccessRange& local_state,
            ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const,
            ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const,
            ::ket::mpi::qubit_permutation<
              StateInteger, BitInteger, Allocator> const&,
            Function&&)
          { throw ::ket::mpi::gate::page::unsupported_page_gate_operation<0, false>{"two_page_qubits_gate"}; }

          template <
            std::size_t num_operated_nonpage_qubits,
            typename ParallelPolicy,
            typename Complex, typename StateAllocator,
            typename StateInteger, typename BitInteger, typename PermutationAllocator,
            typename Function>
          [[noreturn]] inline ::ket::mpi::state<Complex, 0, StateAllocator>& two_page_qubits_gate(
            ParallelPolicy const,
            ::ket::mpi::state<Complex, 0, StateAllocator>& local_state,
            ::ket::qubit<StateInteger, BitInteger> const,
            ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const,
            ::ket::mpi::qubit_permutation<
              StateInteger, BitInteger, PermutationAllocator> const&,
            Function&&)
          { throw ::ket::mpi::gate::page::unsupported_page_gate_operation<0>{"two_page_qubits_gate"}; }

          template <
            std::size_t num_operated_nonpage_qubits,
            typename ParallelPolicy,
            typename Complex, typename StateAllocator,
            typename StateInteger, typename BitInteger, typename PermutationAllocator,
            typename Function>
          [[noreturn]] inline ::ket::mpi::state<Complex, 0, StateAllocator>& two_page_qubits_gate(
            ParallelPolicy const,
            ::ket::mpi::state<Complex, 0, StateAllocator>& local_state,
            ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const,
            ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const,
            ::ket::mpi::qubit_permutation<
              StateInteger, BitInteger, PermutationAllocator> const&,
            Function&&)
          { throw ::ket::mpi::gate::page::unsupported_page_gate_operation<0>{"two_page_qubits_gate"}; }

          template <
            std::size_t num_operated_nonpage_qubits,
            typename ParallelPolicy,
            typename Complex, typename StateAllocator,
            typename StateInteger, typename BitInteger, typename PermutationAllocator,
            typename Function>
          [[noreturn]] inline ::ket::mpi::state<Complex, 1, StateAllocator>& two_page_qubits_gate(
            ParallelPolicy const,
            ::ket::mpi::state<Complex, 1, StateAllocator>& local_state,
            ::ket::qubit<StateInteger, BitInteger> const,
            ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const,
            ::ket::mpi::qubit_permutation<
              StateInteger, BitInteger, PermutationAllocator> const&,
            Function&&)
          { throw ::ket::mpi::gate::page::unsupported_page_gate_operation<1>{"two_page_qubits_gate"}; }

          template <
            std::size_t num_operated_nonpage_qubits,
            typename ParallelPolicy,
            typename Complex, typename StateAllocator,
            typename StateInteger, typename BitInteger, typename PermutationAllocator,
            typename Function>
          [[noreturn]] inline ::ket::mpi::state<Complex, 1, StateAllocator>& two_page_qubits_gate(
            ParallelPolicy const,
            ::ket::mpi::state<Complex, 1, StateAllocator>& local_state,
            ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const,
            ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const,
            ::ket::mpi::qubit_permutation<
              StateInteger, BitInteger, PermutationAllocator> const&,
            Function&&)
          { throw ::ket::mpi::gate::page::unsupported_page_gate_operation<1>{"two_page_qubits_gate"}; }

          namespace two_page_qubits_gate_detail
          {
            template <
              std::size_t num_operated_nonpage_qubits,
              typename ParallelPolicy,
              typename Complex, int num_page_qubits_, typename StateAllocator,
              typename StateInteger, typename BitInteger, typename PermutationAllocator,
              typename Function>
            inline ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator>& two_page_qubits_gate(
              ParallelPolicy const parallel_policy,
              ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator>& local_state,
              ::ket::qubit<StateInteger, BitInteger> const qubit1,
              ::ket::qubit<StateInteger, BitInteger> const qubit2,
              ::ket::mpi::qubit_permutation<
                StateInteger, BitInteger, PermutationAllocator> const&
                permutation,
              Function&& function)
            {
              static_assert(
                num_page_qubits_ >= 2,
                "num_page_qubits_ should be greater than or equal to 2");
              assert(::ket::mpi::page::is_on_page(qubit1, local_state, permutation));
              assert(::ket::mpi::page::is_on_page(qubit2, local_state, permutation));

              auto const permutated_qubit1 = permutation[qubit1];
              auto const permutated_qubit2 = permutation[qubit2];

              auto const num_nonpage_local_qubits
                = static_cast<BitInteger>(local_state.num_local_qubits() - num_page_qubits_);

              auto const minmax_qubits = std::minmax(permutated_qubit1, permutated_qubit2);
              auto const qubit1_mask
                = ::ket::utility::integer_exp2<StateInteger>(
                    permutated_qubit1 - static_cast<BitInteger>(num_nonpage_local_qubits));
              auto const qubit2_mask
                = ::ket::utility::integer_exp2<StateInteger>(
                    permutated_qubit2 - static_cast<BitInteger>(num_nonpage_local_qubits));
              auto const lower_bits_mask
                = ::ket::utility::integer_exp2<StateInteger>(
                    minmax_qubits.first - static_cast<BitInteger>(num_nonpage_local_qubits)) - StateInteger{1u};
              auto const middle_bits_mask
                = (::ket::utility::integer_exp2<StateInteger>(
                     minmax_qubits.second - (BitInteger{1u} + num_nonpage_local_qubits)) - StateInteger{1u})
                  xor lower_bits_mask;
              auto const upper_bits_mask = compl (lower_bits_mask bitor middle_bits_mask);

              static constexpr auto num_pages
                = ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator>::num_pages;
              auto const num_data_blocks = local_state.num_data_blocks();
              for (auto data_block_index = std::size_t{0u}; data_block_index < num_data_blocks; ++data_block_index)
                for (auto page_index_wo_qubits = std::size_t{0u};
                     page_index_wo_qubits < num_pages / 4u; ++page_index_wo_qubits)
                {
                  // x0_2x0_1x
                  auto const page_index_00
                    = ((page_index_wo_qubits bitand upper_bits_mask) << 2u)
                      bitor ((page_index_wo_qubits bitand middle_bits_mask) << 1u)
                      bitor (page_index_wo_qubits bitand lower_bits_mask);
                  // x0_2x1_1x
                  auto const page_index_01 = page_index_00 bitor qubit1_mask;
                  // x1_2x0_1x
                  auto const page_index_10 = page_index_00 bitor qubit2_mask;
                  // x1_2x1_1x
                  auto const page_index_11 = page_index_10 bitor qubit1_mask;

                  auto page_range_00 = local_state.page_range(std::make_pair(data_block_index, page_index_00));
                  auto const first_00 = ::ket::utility::begin(page_range_00);
                  auto page_range_01 = local_state.page_range(std::make_pair(data_block_index, page_index_01));
                  auto const first_01 = ::ket::utility::begin(page_range_01);
                  auto page_range_10 = local_state.page_range(std::make_pair(data_block_index, page_index_10));
                  auto const first_10 = ::ket::utility::begin(page_range_10);
                  auto page_range_11 = local_state.page_range(std::make_pair(data_block_index, page_index_11));
                  auto const first_11 = ::ket::utility::begin(page_range_11);

                  using ::ket::utility::loop_n;
                  loop_n(
                    parallel_policy,
                    boost::size(page_range_11) >> num_operated_nonpage_qubits,
                    [first_00, first_01, first_10, first_11, &function](StateInteger const index_wo_nonpage_qubits, int const thread_index)
                    { function(first_00, first_01, first_10, first_11, index_wo_nonpage_qubits, thread_index); });
                }

              return local_state;
            }
          } // two_page_qubits_gate_detail

          template <
            std::size_t num_operated_nonpage_qubits,
            typename ParallelPolicy,
            typename Complex, int num_page_qubits_, typename StateAllocator,
            typename StateInteger, typename BitInteger, typename PermutationAllocator,
            typename Function>
          inline ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator>& two_page_qubits_gate(
            ParallelPolicy const parallel_policy,
            ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator>& local_state,
            ::ket::qubit<StateInteger, BitInteger> const target_qubit,
            ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
            ::ket::mpi::qubit_permutation<
              StateInteger, BitInteger, PermutationAllocator> const&
              permutation,
            Function&& function)
          {
            return ::ket::mpi::gate::page::detail::two_page_qubits_gate_detail::two_page_qubits_gate<num_operated_nonpage_qubits>(
              parallel_policy, local_state, target_qubit, control_qubit.qubit(),
              permutation, std::forward<Function>(function));
          }

          template <
            std::size_t num_operated_nonpage_qubits,
            typename ParallelPolicy,
            typename Complex, int num_page_qubits_, typename StateAllocator,
            typename StateInteger, typename BitInteger, typename PermutationAllocator,
            typename Function>
          inline ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator>& two_page_qubits_gate(
            ParallelPolicy const parallel_policy,
            ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator>& local_state,
            ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
            ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2,
            ::ket::mpi::qubit_permutation<
              StateInteger, BitInteger, PermutationAllocator> const&
              permutation,
            Function&& function)
          {
            return ::ket::mpi::gate::page::detail::two_page_qubits_gate_detail::two_page_qubits_gate<num_operated_nonpage_qubits>(
              parallel_policy, local_state, control_qubit1.qubit(), control_qubit2.qubit(),
              permutation, std::forward<Function>(function));
          }
        } // namespace detail
      } // namespace page
    } // namespace gate
  } // namespace mpi
} // namespace ket


#endif // KET_MPI_GATE_PAGE_DETAIL_TWO_PAGE_QUBIT_GATES_HPP
