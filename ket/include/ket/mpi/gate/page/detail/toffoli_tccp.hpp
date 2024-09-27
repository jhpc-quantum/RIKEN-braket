#ifndef KET_MPI_GATE_PAGE_DETAIL_TOFFOLI_TCCP_HPP
# define KET_MPI_GATE_PAGE_DETAIL_TOFFOLI_TCCP_HPP

# include <cstddef>
# include <cassert>
# include <algorithm>
# include <iterator>
# include <utility>

# include <ket/qubit.hpp>
# include <ket/control.hpp>
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
          [[noreturn]] inline auto toffoli_tccp(
            ParallelPolicy const,
            RandomAccessRange& local_state,
            ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const,
            ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const,
            ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const)
          -> RandomAccessRange&
          { throw ::ket::mpi::gate::page::unsupported_page_gate_operation{"toffoli_tccp"}; }

          template <
            typename ParallelPolicy,
            typename Complex, typename Allocator, typename StateInteger, typename BitInteger>
          [[noreturn]] inline auto toffoli_tccp(
            ParallelPolicy const,
            ::ket::mpi::state<Complex, false, Allocator>& local_state,
            ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const,
            ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const,
            ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const)
          -> ::ket::mpi::state<Complex, true, Allocator>&
          { throw ::ket::mpi::gate::page::unsupported_page_gate_operation{"toffoli_tccp"}; }

          template <
            typename ParallelPolicy,
            typename Complex, typename Allocator, typename StateInteger, typename BitInteger>
          inline auto toffoli_tccp(
            ParallelPolicy const parallel_policy,
            ::ket::mpi::state<Complex, true, Allocator>& local_state,
            ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_target_qubit,
            ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const permutated_control_qubit1,
            ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const permutated_control_qubit2)
          -> ::ket::mpi::state<Complex, true, Allocator>&
          {
            assert(local_state.num_page_qubits() >= std::size_t{3u});
            assert(::ket::mpi::page::is_on_page(permutated_target_qubit, local_state));
            assert(::ket::mpi::page::is_on_page(permutated_control_qubit1, local_state));
            assert(::ket::mpi::page::is_on_page(permutated_control_qubit2, local_state));

            auto const num_nonpage_local_qubits
              = static_cast<BitInteger>(local_state.num_local_qubits() - local_state.num_page_qubits());

            auto const permutated_target_qubit_mask
              = ::ket::utility::integer_exp2<StateInteger>(
                  permutated_target_qubit - static_cast<BitInteger>(num_nonpage_local_qubits));
            auto const permutated_control_qubits_mask
              = ::ket::utility::integer_exp2<StateInteger>(
                  permutated_control_qubit1 - static_cast<BitInteger>(num_nonpage_local_qubits))
                bitor ::ket::utility::integer_exp2<StateInteger>(
                        permutated_control_qubit2 - static_cast<BitInteger>(num_nonpage_local_qubits));

            using permutated_qubit_type = ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> >;
            auto sorted_permutated_qubits
              = std::array<permutated_qubit_type, 3u>{
                  permutated_target_qubit,
                  ::ket::mpi::remove_control(permutated_control_qubit1),
                  ::ket::mpi::remove_control(permutated_control_qubit2)};
            using std::begin;
            using std::end;
            std::sort(begin(sorted_permutated_qubits), end(sorted_permutated_qubits));

            auto bits_mask = std::array<StateInteger, 4u>{};
            bits_mask[0u]
              = ::ket::utility::integer_exp2<StateInteger>(
                  sorted_permutated_qubits[0u] - static_cast<BitInteger>(num_nonpage_local_qubits))
                - StateInteger{1u};
            bits_mask[1u]
              = (::ket::utility::integer_exp2<StateInteger>(
                   sorted_permutated_qubits[1u] - (BitInteger{1u} + num_nonpage_local_qubits)) - StateInteger{1u})
                xor bits_mask[0u];
            bits_mask[2u]
              = (::ket::utility::integer_exp2<StateInteger>(
                   sorted_permutated_qubits[2u] - (BitInteger{2u} + num_nonpage_local_qubits)) - StateInteger{1u})
                xor (bits_mask[0u] bitor bits_mask[1u]);
            bits_mask[3u] = compl (bits_mask[0u] bitor bits_mask[1u] bitor bits_mask[2u]);

            auto const num_pages = local_state.num_pages();
            auto const num_data_blocks = local_state.num_data_blocks();
            for (auto data_block_index = std::size_t{0u}; data_block_index < num_data_blocks; ++data_block_index)
              for (auto page_index_wo_qubits = std::size_t{0u};
                   page_index_wo_qubits < num_pages / 8u; ++page_index_wo_qubits)
              {
                // x0_cx0_tx0_cx
                auto const base_page_index
                  = ((page_index_wo_qubits bitand bits_mask[3u]) << 3u)
                    bitor ((page_index_wo_qubits bitand bits_mask[2u]) << 2u)
                    bitor ((page_index_wo_qubits bitand bits_mask[1u]) << 1u)
                    bitor (page_index_wo_qubits bitand bits_mask[0u]);
                // x1_cx0_tx1_cx
                auto const control_on_page_index = base_page_index bitor permutated_control_qubits_mask;
                // x1_cx1_tx1_cx
                auto const target_control_on_page_index = control_on_page_index bitor permutated_target_qubit_mask;

                local_state.swap_pages(
                  std::make_pair(data_block_index, control_on_page_index),
                  std::make_pair(data_block_index, target_control_on_page_index));
              }

            return local_state;
          }
        } // namespace detail
      } // namespace page
    } // namespace gate
  } // namespace mpi
} // namespace ket


#endif // KET_MPI_GATE_PAGE_DETAIL_TOFFOLI_TCCP_HPP
