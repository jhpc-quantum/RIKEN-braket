#ifndef KET_MPI_GATE_PAGE_DETAIL_TOFFOLI_TCCP_HPP
# define KET_MPI_GATE_PAGE_DETAIL_TOFFOLI_TCCP_HPP

# include <cstddef>
# include <cassert>
# include <algorithm>
# include <utility>

# include <ket/qubit.hpp>
# include <ket/control.hpp>
# include <ket/utility/integer_exp2.hpp>
# include <ket/utility/begin.hpp>
# include <ket/utility/end.hpp>
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
          [[noreturn]] inline RandomAccessRange& toffoli_tccp(
            MpiPolicy const, ParallelPolicy const,
            RandomAccessRange& local_state,
            ::ket::qubit<StateInteger, BitInteger> const,
            ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const,
            ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const,
            ::ket::mpi::qubit_permutation<
              StateInteger, BitInteger, Allocator> const&)
          { throw ::ket::mpi::gate::page::unsupported_page_gate_operation<0, false>{"toffoli_tccp"}; }

          template <
            typename ParallelPolicy,
            typename Complex, typename StateAllocator,
            typename StateInteger, typename BitInteger, typename PermutationAllocator>
          [[noreturn]] inline ::ket::mpi::state<Complex, 0, StateAllocator>& toffoli_tccp(
            ::ket::mpi::utility::policy::general_mpi const, ParallelPolicy const,
            ::ket::mpi::state<Complex, 0, StateAllocator>& local_state,
            ::ket::qubit<StateInteger, BitInteger> const,
            ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const,
            ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const,
            ::ket::mpi::qubit_permutation<
              StateInteger, BitInteger, PermutationAllocator> const&)
          { throw ::ket::mpi::gate::page::unsupported_page_gate_operation<0>{"toffoli_tccp"}; }

          template <
            typename ParallelPolicy,
            typename Complex, typename StateAllocator,
            typename StateInteger, typename BitInteger, typename PermutationAllocator>
          [[noreturn]] inline ::ket::mpi::state<Complex, 1, StateAllocator>& toffoli_tccp(
            ::ket::mpi::utility::policy::general_mpi const, ParallelPolicy const,
            ::ket::mpi::state<Complex, 1, StateAllocator>& local_state,
            ::ket::qubit<StateInteger, BitInteger> const,
            ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const,
            ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const,
            ::ket::mpi::qubit_permutation<
              StateInteger, BitInteger, PermutationAllocator> const&)
          { throw ::ket::mpi::gate::page::unsupported_page_gate_operation<1>{"toffoli_tccp"}; }

          template <
            typename ParallelPolicy,
            typename Complex, typename StateAllocator,
            typename StateInteger, typename BitInteger, typename PermutationAllocator>
          [[noreturn]] inline ::ket::mpi::state<Complex, 2, StateAllocator>& toffoli_tccp(
            ::ket::mpi::utility::policy::general_mpi const, ParallelPolicy const,
            ::ket::mpi::state<Complex, 2, StateAllocator>& local_state,
            ::ket::qubit<StateInteger, BitInteger> const,
            ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const,
            ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const,
            ::ket::mpi::qubit_permutation<
              StateInteger, BitInteger, PermutationAllocator> const&)
          { throw ::ket::mpi::gate::page::unsupported_page_gate_operation<2>{"toffoli_tccp"}; }

          template <
            typename ParallelPolicy,
            typename Complex, int num_page_qubits_, typename StateAllocator,
            typename StateInteger, typename BitInteger, typename PermutationAllocator>
          inline ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator>&
          toffoli_tccp(
            ::ket::mpi::utility::policy::general_mpi const,
            ParallelPolicy const parallel_policy,
            ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator>& local_state,
            ::ket::qubit<StateInteger, BitInteger> const target_qubit,
            ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
            ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2,
            ::ket::mpi::qubit_permutation<
              StateInteger, BitInteger, PermutationAllocator> const&
              permutation)
          {
            static_assert(
              num_page_qubits_ >= 3,
              "num_page_qubits_ should be greater than or equal to 3");

            auto const permutated_target_qubit = permutation[target_qubit];
            auto const permutated_control_qubit1 = permutation[control_qubit1.qubit()];
            auto const permutated_control_qubit2 = permutation[control_qubit2.qubit()];
            assert(local_state.is_page_qubit(permutated_target_qubit));
            assert(local_state.is_page_qubit(permutated_control_qubit1));
            assert(local_state.is_page_qubit(permutated_control_qubit2));

            auto const num_nonpage_local_qubits
              = static_cast<BitInteger>(local_state.num_local_qubits() - num_page_qubits_);

            auto const target_qubit_mask
              = ::ket::utility::integer_exp2<StateInteger>(
                  permutated_target_qubit - static_cast<BitInteger>(num_nonpage_local_qubits));
            auto const control_qubits_mask
              = ::ket::utility::integer_exp2<StateInteger>(
                  permutated_control_qubit1 - static_cast<BitInteger>(num_nonpage_local_qubits))
                bitor ::ket::utility::integer_exp2<StateInteger>(
                        permutated_control_qubit2
                        - static_cast<BitInteger>(num_nonpage_local_qubits));

            using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
            auto sorted_permutated_qubits
              = std::array<qubit_type, 3u>{
                  permutated_target_qubit,
                  permutated_control_qubit1, permutated_control_qubit2};
            std::sort(
              ::ket::utility::begin(sorted_permutated_qubits),
              ::ket::utility::end(sorted_permutated_qubits));

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

            static constexpr auto num_pages
              = ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator>::num_pages;
            for (auto page_id_wo_qubits = std::size_t{0u};
                 page_id_wo_qubits < num_pages / 8u; ++page_id_wo_qubits)
            {
              // x0_cx0_tx0_cx
              auto const base_page_id
                = ((page_id_wo_qubits bitand bits_mask[3u]) << 3u)
                  bitor ((page_id_wo_qubits bitand bits_mask[2u]) << 2u)
                  bitor ((page_id_wo_qubits bitand bits_mask[1u]) << 1u)
                  bitor (page_id_wo_qubits bitand bits_mask[0u]);
              // x1_cx0_tx1_cx
              auto const control_on_page_id = base_page_id bitor control_qubits_mask;
              // x1_cx1_tx1_cx
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


#endif // KET_MPI_GATE_PAGE_DETAIL_TOFFOLI_TCCP_HPP
