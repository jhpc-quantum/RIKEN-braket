#ifndef KET_MPI_GATE_PAGE_DETAIL_CONTROLLED_NOT_TCP_HPP
# define KET_MPI_GATE_PAGE_DETAIL_CONTROLLED_NOT_TCP_HPP

# include <cstddef>
# include <cassert>
# include <algorithm>

# include <ket/qubit.hpp>
# include <ket/control.hpp>
# include <ket/utility/loop_n.hpp>
# include <ket/utility/integer_exp2.hpp>
# include <ket/mpi/qubit_permutation.hpp>
# include <ket/mpi/state.hpp>
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
            typename MpiPolicy, typename ParallelPolicy,
            typename RandomAccessRange,
            typename StateInteger, typename BitInteger, typename Allocator>
          [[noreturn]] inline RandomAccessRange& controlled_not_tcp(
            MpiPolicy const, ParallelPolicy const,
            RandomAccessRange& local_state,
            ::ket::qubit<StateInteger, BitInteger> const,
            ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const,
            ::ket::mpi::qubit_permutation<
              StateInteger, BitInteger, Allocator> const&)
          { throw ::ket::mpi::gate::page::unsupported_page_gate_operation<0, false>{"controlled_not_tcp"}; }

          template <
            typename ParallelPolicy,
            typename Complex, typename StateAllocator,
            typename StateInteger, typename BitInteger, typename PermutationAllocator>
          [[noreturn]] inline ::ket::mpi::state<Complex, 0, StateAllocator>& controlled_not_tcp(
            ::ket::mpi::utility::policy::general_mpi const, ParallelPolicy const,
            ::ket::mpi::state<Complex, 0, StateAllocator>& local_state,
            ::ket::qubit<StateInteger, BitInteger> const,
            ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const,
            ::ket::mpi::qubit_permutation<
              StateInteger, BitInteger, PermutationAllocator> const&)
          { throw ::ket::mpi::gate::page::unsupported_page_gate_operation<0>{"controlled_not_tcp"}; }

          template <
            typename ParallelPolicy,
            typename Complex, typename StateAllocator,
            typename StateInteger, typename BitInteger, typename PermutationAllocator>
          [[noreturn]] inline ::ket::mpi::state<Complex, 1, StateAllocator>& controlled_not_tcp(
            ::ket::mpi::utility::policy::general_mpi const, ParallelPolicy const,
            ::ket::mpi::state<Complex, 1, StateAllocator>& local_state,
            ::ket::qubit<StateInteger, BitInteger> const,
            ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const,
            ::ket::mpi::qubit_permutation<
              StateInteger, BitInteger, PermutationAllocator> const&)
          { throw ::ket::mpi::gate::page::unsupported_page_gate_operation<1>{"controlled_not_tcp"}; }

          template <
            typename ParallelPolicy,
            typename Complex, int num_page_qubits_, typename StateAllocator,
            typename StateInteger, typename BitInteger, typename PermutationAllocator>
          inline ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator>& controlled_not_tcp(
            ::ket::mpi::utility::policy::general_mpi const mpi_policy,
            ParallelPolicy const parallel_policy,
            ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator>& local_state,
            ::ket::qubit<StateInteger, BitInteger> const target_qubit,
            ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
            ::ket::mpi::qubit_permutation<
              StateInteger, BitInteger, PermutationAllocator> const&
              permutation)
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
              // x0_tx1_cx
              auto const control_on_page_id = base_page_id bitor control_qubit_mask;
              // x1_tx1_cx
              auto const target_control_on_page_id = control_on_page_id bitor target_qubit_mask;

              local_state.swap_pages(control_on_page_id, target_control_on_page_id);
            }

            return local_state;
          }
        } // namespace detail
      } // namespace page
    } // namespace gate
  } // namespace mpi
} // namespace ket


#endif // KET_MPI_GATE_PAGE_DETAIL_CONTROLLED_NOT_TCP_HPP