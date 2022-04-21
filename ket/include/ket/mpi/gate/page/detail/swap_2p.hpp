#ifndef KET_MPI_GATE_PAGE_DETAIL_SWAP_2P_HPP
# define KET_MPI_GATE_PAGE_DETAIL_SWAP_2P_HPP

# include <cstddef>
# include <cassert>
# include <algorithm>

# include <ket/qubit.hpp>
# include <ket/utility/loop_n.hpp>
# include <ket/utility/integer_exp2.hpp>
# include <ket/mpi/permutated.hpp>
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
            typename ParallelPolicy,
            typename RandomAccessRange, typename StateInteger, typename BitInteger>
          [[noreturn]] inline RandomAccessRange& swap_2p(
            ParallelPolicy const,
            RandomAccessRange& local_state,
            ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const,
            ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const)
          { throw ::ket::mpi::gate::page::unsupported_page_gate_operation{"swap_2p"}; }

          template <
            typename ParallelPolicy,
            typename Complex, typename Allocator, typename StateInteger, typename BitInteger>
          [[noreturn]] inline ::ket::mpi::state<Complex, false, Allocator>& swap_2p(
            ParallelPolicy const,
            ::ket::mpi::state<Complex, false, Allocator>& local_state,
            ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const,
            ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const)
          { throw ::ket::mpi::gate::page::unsupported_page_gate_operation{"swap_2p"}; }

          template <
            typename ParallelPolicy,
            typename Complex, typename Allocator, typename StateInteger, typename BitInteger>
          inline ::ket::mpi::state<Complex, true, Allocator>& swap_2p(
            ParallelPolicy const parallel_policy,
            ::ket::mpi::state<Complex, true, Allocator>& local_state,
            ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const page_permutated_qubit1,
            ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const page_permutated_qubit2)
          {
            assert(local_state.num_page_qubits() >= std::size_t{2u});

            assert(::ket::mpi::page::is_on_page(page_permutated_qubit1, local_state));
            assert(::ket::mpi::page::is_on_page(page_permutated_qubit2, local_state));

            auto const num_nonpage_qubits
              = static_cast<BitInteger>(local_state.num_local_qubits() - local_state.num_page_qubits());

            using permutated_qubit_type = ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> >;
            auto const minmax_permutated_qubits = std::minmax(page_permutated_qubit1, page_permutated_qubit2);
            auto const page_permutated_qubit1_mask
              = ::ket::utility::integer_exp2<StateInteger>(
                  page_permutated_qubit1 - static_cast<BitInteger>(num_nonpage_qubits));
            auto const page_permutated_qubit2_mask
              = ::ket::utility::integer_exp2<StateInteger>(
                  page_permutated_qubit2 - static_cast<BitInteger>(num_nonpage_qubits));
            auto const lower_bits_mask
              = ::ket::utility::integer_exp2<StateInteger>(
                  minmax_permutated_qubits.first - static_cast<BitInteger>(num_nonpage_qubits)) - StateInteger{1u};
            auto const middle_bits_mask
              = (::ket::utility::integer_exp2<StateInteger>(
                   minmax_permutated_qubits.second - (BitInteger{1u} + num_nonpage_qubits)) - StateInteger{1u})
                xor lower_bits_mask;
            auto const upper_bits_mask = compl (lower_bits_mask bitor middle_bits_mask);

            auto const num_pages = local_state.num_pages();
            auto const num_data_blocks = local_state.num_data_blocks();
            for (auto data_block_index = std::size_t{0u}; data_block_index < num_data_blocks; ++data_block_index)
              for (auto page_index_wo_qubits = std::size_t{0u};
                   page_index_wo_qubits < num_pages / 4u; ++page_index_wo_qubits)
              {
                // x0_2x0_1x
                auto const base_page_index
                  = ((page_index_wo_qubits bitand upper_bits_mask) << 2u)
                    bitor ((page_index_wo_qubits bitand middle_bits_mask) << 1u)
                    bitor (page_index_wo_qubits bitand lower_bits_mask);
                // x0_2x1_1x
                auto const page_index_01 = base_page_index bitor page_permutated_qubit1_mask;
                // x1_2x0_1x
                auto const page_index_10 = base_page_index bitor page_permutated_qubit2_mask;

                local_state.swap_pages(
                  std::make_pair(data_block_index, page_index_01), std::make_pair(data_block_index, page_index_10));
              }

            return local_state;
          }
        } // namespace detail
      } // namespace page
    } // namespace gate
  } // namespace mpi
} // namespace ket


#endif // KET_MPI_GATE_PAGE_DETAIL_SWAP_2P_HPP
