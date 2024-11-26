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
    } // namespace fused
  } // namespace gate
} // namespace ket


#endif // KET_GATE_FUSED_CONTROLLED_NOT_HPP
