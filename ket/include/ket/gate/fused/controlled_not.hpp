#ifndef KET_GATE_FUSED_CONTROLLED_NOT_HPP
# define KET_GATE_FUSED_CONTROLLED_NOT_HPP

# include <cstddef>
# include <array>

# include <ket/qubit.hpp>
# include <ket/control.hpp>
# include <ket/gate/fused/not_.hpp>


namespace ket
{
  namespace gate
  {
    namespace fused
    {
# ifndef KET_USE_BIT_MASKS_EXPLICITLY
      // CNOT_{tc} or C1NOT_{tc}
      // CNOT_{1,2} (a_{00} |00> + a_{01} |01> + a_{10} |10> + a_{11} |11>)
      //   = a_{00} |00> + a_{01} |01> + a_{11} |10> + a_{10} |11>
      // C...CNOT_{tc...c'} or CnNOT_{tc...c'}
      template <typename RandomAccessIterator, typename StateInteger, typename BitInteger, std::size_t num_fused_qubits, typename... ControlQubits>
      inline auto controlled_not(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_fused_qubits > const& unsorted_fused_qubits,
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_fused_qubits + 1u> const& sorted_fused_qubits_with_sentinel,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit, ControlQubits const... control_qubits)
      -> void
      { ::ket::gate::fused::not_(first, fused_index_wo_qubits, unsorted_fused_qubits, sorted_fused_qubits_with_sentinel, target_qubit, control_qubit, control_qubits...); }

      template <typename RandomAccessIterator, typename StateInteger, typename BitInteger, std::size_t num_fused_qubits, typename... ControlQubits>
      inline auto adj_controlled_not(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_fused_qubits > const& unsorted_fused_qubits,
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_fused_qubits + 1u> const& sorted_fused_qubits_with_sentinel,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit, ControlQubits const... control_qubits)
      -> void
      { ::ket::gate::fused::adj_not_(first, fused_index_wo_qubits, unsorted_fused_qubits, sorted_fused_qubits_with_sentinel, target_qubit, control_qubit, control_qubits...); }


      namespace runtime
      {
        template <typename RandomAccessIterator, typename StateInteger, typename QubitIterator1, typename QubitIterator2, typename BitInteger, typename ControlQubitIterator>
        inline auto controlled_not(
          RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
          QubitIterator1 const unsorted_fused_qubit_first, QubitIterator1 const unsorted_fused_qubit_last,
          QubitIterator2 const sorted_fused_qubit_with_sentinel_first, QubitIterator2 const sorted_fused_qubit_with_sentinel_last,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ControlQubitIterator const control_qubit_first, ControlQubitIterator const control_qubit_last)
        -> void
        {
          ::ket::gate::fused::runtime::not_(
            first, fused_index_wo_qubits,
            unsorted_fused_qubit_first, unsorted_fused_qubit_last,
            sorted_fused_qubit_with_sentinel_first, sorted_fused_qubit_with_sentinel_last,
            target_qubit, control_qubit_first, control_qubit_last);
        }

        template <typename RandomAccessIterator, typename StateInteger, typename QubitIterator1, typename QubitIterator2, typename BitInteger, typename ControlQubitIterator>
        inline auto adj_controlled_not(
          RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
          QubitIterator1 const unsorted_fused_qubit_first, QubitIterator1 const unsorted_fused_qubit_last,
          QubitIterator2 const sorted_fused_qubit_with_sentinel_first, QubitIterator2 const sorted_fused_qubit_with_sentinel_last,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ControlQubitIterator const control_qubit_first, ControlQubitIterator const control_qubit_last)
        -> void
        {
          ::ket::gate::fused::runtime::adj_not_(
            first, fused_index_wo_qubits,
            unsorted_fused_qubit_first, unsorted_fused_qubit_last,
            sorted_fused_qubit_with_sentinel_first, sorted_fused_qubit_with_sentinel_last,
            target_qubit, control_qubit_first, control_qubit_last);
        }

        namespace ranges
        {
          template <typename RandomAccessIterator, typename StateInteger, typename QubitsRange1, typename QubitsRange2, typename BitInteger, typename ControlQubitsRange>
          inline auto controlled_not(
            RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
            QubitsRange1 const& unsorted_fused_qubits, QubitsRange2 const& sorted_fused_qubits_with_sentinel,
            ::ket::qubit<StateInteger, BitInteger> const target_qubit,
            ControlQubitsRange const& control_qubits)
          -> void
          {
            ::ket::gate::fused::runtime::ranges::not_(
              first, fused_index_wo_qubits,
              unsorted_fused_qubits, sorted_fused_qubits_with_sentinel,
              target_qubit, control_qubits);
          }

          template <typename RandomAccessIterator, typename StateInteger, typename QubitsRange1, typename QubitsRange2, typename BitInteger, typename ControlQubitsRange>
          inline auto adj_controlled_not(
            RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
            QubitsRange1 const& unsorted_fused_qubits, QubitsRange2 const& sorted_fused_qubits_with_sentinel,
            ::ket::qubit<StateInteger, BitInteger> const target_qubit,
            ControlQubitsRange const& control_qubits)
          -> void
          {
            ::ket::gate::fused::runtime::ranges::adj_not_(
              first, fused_index_wo_qubits,
              unsorted_fused_qubits, sorted_fused_qubits_with_sentinel,
              target_qubit, control_qubits);
          }
        } // namespace ranges
      } // namespace runtime
# else // KET_USE_BIT_MASKS_EXPLICITLY
      // CNOT_{tc} or C1NOT_{tc}
      // CNOT_{1,2} (a_{00} |00> + a_{01} |01> + a_{10} |10> + a_{11} |11>)
      //   = a_{00} |00> + a_{01} |01> + a_{11} |10> + a_{10} |11>
      // C...CNOT_{tc...c'} or CnNOT_{tc...c'}
      template <typename RandomAccessIterator, typename StateInteger, std::size_t num_fused_qubits, typename BitInteger, typename... ControlQubits>
      inline auto controlled_not(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array<StateInteger, num_fused_qubits> const& fused_qubit_masks, std::array<StateInteger, num_fused_qubits + 1u> const& fused_index_masks,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit, ControlQubits const... control_qubits)
      -> void
      { ::ket::gate::fused::not_(first, fused_index_wo_qubits, fused_qubit_masks, fused_index_masks, target_qubit, control_qubit, control_qubits...); }

      template <typename RandomAccessIterator, typename StateInteger, std::size_t num_fused_qubits, typename BitInteger, typename... ControlQubits>
      inline auto adj_controlled_not(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array<StateInteger, num_fused_qubits> const& fused_qubit_masks, std::array<StateInteger, num_fused_qubits + 1u> const& fused_index_masks,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit, ControlQubits const... control_qubits)
      -> void
      { ::ket::gate::fused::adj_not_(first, fused_index_wo_qubits, fused_qubit_masks, fused_index_masks, target_qubit, control_qubit, control_qubits...); }


      namespace runtime
      {
        template <typename RandomAccessIterator, typename StateInteger, typename StateIntegerIterator1, typename StateIntegerIterator2, typename BitInteger, typename ControlQubitIterator>
        inline auto controlled_not(
          RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
          StateIntegerIterator1 const fused_qubit_mask_first, StateIntegerIterator1 const fused_qubit_mask_last,
          StateIntegerIterator2 const fused_index_mask_first, StateIntegerIterator2 const fused_index_mask_last,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ControlQubitIterator const control_qubit_first, ControlQubitIterator const control_qubit_last)
        -> void
        {
          ::ket::gate::fused::runtime::not_(
            first, fused_index_wo_qubits,
            fused_qubit_mask_first, fused_qubit_mask_last,
            fused_index_mask_first, fused_index_mask_last,
            target_qubit, control_qubit_first, control_qubit_last);
        }

        template <typename RandomAccessIterator, typename StateInteger, typename StateIntegerIterator1, typename StateIntegerIterator2, typename BitInteger, typename ControlQubitIterator>
        inline auto adj_controlled_not(
          RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
          StateIntegerIterator1 const fused_qubit_mask_first, StateIntegerIterator1 const fused_qubit_mask_last,
          StateIntegerIterator2 const fused_index_mask_first, StateIntegerIterator2 const fused_index_mask_last,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ControlQubitIterator const control_qubit_first, ControlQubitIterator const control_qubit_last)
        -> void
        {
          ::ket::gate::fused::runtime::adj_not_(
            first, fused_index_wo_qubits,
            fused_qubit_mask_first, fused_qubit_mask_last,
            fused_index_mask_first, fused_index_mask_last,
            target_qubit, control_qubit_first, control_qubit_last);
        }

        namespace ranges
        {
          template <typename RandomAccessIterator, typename StateInteger, typename StateIntegersRange1, typename StateIntegersRange2, typename BitInteger, typename ControlQubitsRange>
          inline auto controlled_not(
            RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
            StateIntegersRange1 const& fused_qubit_masks, StateIntegersRange2 const& fused_index_masks,
            ::ket::qubit<StateInteger, BitInteger> const target_qubit,
            ControlQubitsRange const& control_qubits)
          -> void
          {
            ::ket::gate::fused::runtime::ranges::not_(
              first, fused_index_wo_qubits,
              fused_qubit_masks, fused_index_masks,
              target_qubit, control_qubits);
          }

          template <typename RandomAccessIterator, typename StateInteger, typename StateIntegersRange1, typename StateIntegersRange2, typename BitInteger, typename ControlQubitsRange>
          inline auto adj_controlled_not(
            RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
            StateIntegersRange1 const& fused_qubit_masks, StateIntegersRange2 const& fused_index_masks,
            ::ket::qubit<StateInteger, BitInteger> const target_qubit,
            ControlQubitsRange const& control_qubits)
          -> void
          {
            ::ket::gate::fused::runtime::ranges::adj_not_(
              first, fused_index_wo_qubits,
              fused_qubit_masks, fused_index_masks,
              target_qubit, control_qubits);
          }
        } // namespace ranges
      } // namespace runtime
# endif // KET_USE_BIT_MASKS_EXPLICITLY
    } // namespace fused
  } // namespace gate
} // namespace ket


#endif // KET_GATE_FUSED_CONTROLLED_NOT_HPP
