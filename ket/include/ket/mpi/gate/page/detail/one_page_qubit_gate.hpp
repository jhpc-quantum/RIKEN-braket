#ifndef KET_MPI_GATE_PAGE_DETAIL_ONE_PAGE_QUBIT_GATE_HPP
# define KET_MPI_GATE_PAGE_DETAIL_ONE_PAGE_QUBIT_GATE_HPP

# include <boost/config.hpp>

# include <cstddef>
# include <cassert>
# include <iterator>

# include <boost/range/size.hpp>

# include <ket/qubit.hpp>
# include <ket/utility/loop_n.hpp>
# include <ket/utility/integer_exp2.hpp>
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
          [[noreturn]] inline RandomAccessRange& one_page_qubit_gate(
            ParallelPolicy const,
            RandomAccessRange& local_state,
            ::ket::qubit<StateInteger, BitInteger> const,
            ::ket::mpi::qubit_permutation<
              StateInteger, BitInteger, Allocator> const&,
            Function&&)
          { throw ::ket::mpi::gate::page::unsupported_page_gate_operation<0, false>{"one_page_qubit_gate"}; }

          template <
            std::size_t num_operated_nonpage_qubits,
            typename ParallelPolicy,
            typename Complex, typename StateAllocator,
            typename StateInteger, typename BitInteger, typename PermutationAllocator,
            typename Function>
          [[noreturn]] inline ::ket::mpi::state<Complex, 0, StateAllocator>& one_page_qubit_gate(
            ParallelPolicy const,
            ::ket::mpi::state<Complex, 0, StateAllocator>& local_state,
            ::ket::qubit<StateInteger, BitInteger> const,
            ::ket::mpi::qubit_permutation<
              StateInteger, BitInteger, PermutationAllocator> const&,
            Function&&)
          { throw ::ket::mpi::gate::page::unsupported_page_gate_operation<0>{"one_page_qubit_gate"}; }

          template <
            std::size_t num_operated_nonpage_qubits,
            typename ParallelPolicy,
            typename Complex, int num_page_qubits_, typename StateAllocator,
            typename StateInteger, typename BitInteger, typename PermutationAllocator,
            typename Function>
          inline ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator>& one_page_qubit_gate(
            ParallelPolicy const parallel_policy,
            ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator>& local_state,
            ::ket::qubit<StateInteger, BitInteger> const qubit,
            ::ket::mpi::qubit_permutation<
              StateInteger, BitInteger, PermutationAllocator> const&
              permutation,
            Function&& function)
          {
            static_assert(
              num_page_qubits_ >= 1,
              "num_page_qubits_ should be greater than or equal to 1");
            assert(::ket::mpi::page::is_on_page(qubit, local_state, permutation));

            auto const num_nonpage_local_qubits
              = static_cast<BitInteger>(local_state.num_local_qubits() - num_page_qubits_);
            auto const qubit_mask
              = ::ket::utility::integer_exp2<StateInteger>(
                  permutation[qubit] - static_cast<BitInteger>(num_nonpage_local_qubits));
            auto const lower_bits_mask = qubit_mask - StateInteger{1u};
            auto const upper_bits_mask = compl lower_bits_mask;

            static constexpr auto num_pages
              = ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator>::num_pages;
            auto const num_data_blocks = local_state.num_data_blocks();
            for (auto data_block_index = std::size_t{0u}; data_block_index < num_data_blocks; ++data_block_index)
              for (auto page_index_wo_qubit = std::size_t{0u}; page_index_wo_qubit < num_pages / 2u; ++page_index_wo_qubit)
              {
                // x0x
                auto const zero_page_index
                  = ((page_index_wo_qubit bitand upper_bits_mask) << 1u)
                    bitor (page_index_wo_qubit bitand lower_bits_mask);
                // x1x
                auto const one_page_index = zero_page_index bitor qubit_mask;

                auto zero_page_range = local_state.page_range(std::make_pair(data_block_index, zero_page_index));
                auto one_page_range = local_state.page_range(std::make_pair(data_block_index, one_page_index));
                assert(boost::size(zero_page_range) == boost::size(one_page_range));
                assert(::ket::utility::integer_exp2<std::size_t>(::ket::utility::integer_log2<std::size_t>(boost::size(zero_page_range))) == boost::size(zero_page_range));

                auto const zero_first = std::begin(zero_page_range);
                auto const one_first = std::begin(one_page_range);

                using ::ket::utility::loop_n;
                loop_n(
                  parallel_policy,
                  boost::size(zero_page_range) >> num_operated_nonpage_qubits,
                  [zero_first, one_first, &function](StateInteger const index_wo_nonpage_qubits, int const thread_index)
                  { function(zero_first, one_first, index_wo_nonpage_qubits, thread_index); });
              }

            return local_state;
          }
        } // namespace detail
      } // namespace page
    } // namespace gate
  } // namespace mpi
} // namespace ket


#endif // KET_MPI_GATE_PAGE_DETAIL_ONE_PAGE_QUBIT_GATE_HPP
