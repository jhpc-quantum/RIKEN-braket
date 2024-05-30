#ifndef KET_MPI_GATE_PAGE_DETAIL_PAULI_CX_TCP_HPP
# define KET_MPI_GATE_PAGE_DETAIL_PAULI_CX_TCP_HPP

# include <cstddef>
# include <cassert>
# include <algorithm>

# include <ket/qubit.hpp>
# include <ket/control.hpp>
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
          [[noreturn]] inline RandomAccessRange& pauli_cx_tcp(
            ParallelPolicy const,
            RandomAccessRange& local_state,
            ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const,
            ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const)
          { throw ::ket::mpi::gate::page::unsupported_page_gate_operation{"pauli_cx_tcp"}; }

          template <
            typename ParallelPolicy,
            typename Complex, typename Allocator, typename StateInteger, typename BitInteger>
          [[noreturn]] inline ::ket::mpi::state<Complex, false, Allocator>& pauli_cx_tcp(
            ParallelPolicy const,
            ::ket::mpi::state<Complex, false, Allocator>& local_state,
            ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const,
            ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const)
          { throw ::ket::mpi::gate::page::unsupported_page_gate_operation{"pauli_cx_tcp"}; }

          template <
            typename ParallelPolicy,
            typename Complex, typename Allocator, typename StateInteger, typename BitInteger>
          inline ::ket::mpi::state<Complex, true, Allocator>& pauli_cx_tcp(
            ParallelPolicy const parallel_policy,
            ::ket::mpi::state<Complex, true, Allocator>& local_state,
            ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_target_qubit,
            ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const permutated_control_qubit)
          {
            assert(local_state.num_page_qubits() >= std::size_t{2u});

            assert(::ket::mpi::page::is_on_page(permutated_target_qubit, local_state));
            assert(::ket::mpi::page::is_on_page(permutated_control_qubit, local_state));

            auto const num_nonpage_qubits
              = static_cast<BitInteger>(local_state.num_local_qubits() - local_state.num_page_qubits());

            using permutated_qubit_type = ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> >;
            auto const minmax_permutated_qubits
              = static_cast<std::pair<permutated_qubit_type, permutated_qubit_type>>(
                  std::minmax(permutated_target_qubit, ::ket::mpi::remove_control(permutated_control_qubit)));
            auto const permutated_target_qubit_mask
              = ::ket::utility::integer_exp2<StateInteger>(
                  permutated_target_qubit - static_cast<BitInteger>(num_nonpage_qubits));
            auto const permutated_control_qubit_mask
              = ::ket::utility::integer_exp2<StateInteger>(
                  permutated_control_qubit - static_cast<BitInteger>(num_nonpage_qubits));
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
                // x0_tx0_cx
                auto const base_page_index
                  = ((page_index_wo_qubits bitand upper_bits_mask) << 2u)
                    bitor ((page_index_wo_qubits bitand middle_bits_mask) << 1u)
                    bitor (page_index_wo_qubits bitand lower_bits_mask);
                // x0_tx1_cx
                auto const control_on_page_index = base_page_index bitor permutated_control_qubit_mask;
                // x1_tx1_cx
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


#endif // KET_MPI_GATE_PAGE_DETAIL_PAULI_CX_TCP_HPP
