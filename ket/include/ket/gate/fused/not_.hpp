#ifndef KET_GATE_FUSED_NOT_HPP
# define KET_GATE_FUSED_NOT_HPP

# include <cstddef>
# include <algorithm>
# include <array>

# include <ket/qubit.hpp>
# include <ket/control.hpp>
# include <ket/gate/fused/pauli_x.hpp>
# include <ket/gate/fused/gate.hpp>
# include <ket/gate/utility/index_with_qubits.hpp>
# include <ket/utility/integer_exp2.hpp>


namespace ket
{
  namespace gate
  {
    namespace fused
    {
      // NOT_i
      // NOT_1 (a_0 |0> + a_1 |1>) = a_1 |0> + a_0 |1>
      template <typename RandomAccessIterator, typename StateInteger, std::size_t num_fused_qubits, typename BitInteger>
      inline auto not_(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array<StateInteger, num_fused_qubits> const& fused_qubit_masks, std::array<StateInteger, num_fused_qubits + 1u> const& fused_index_masks,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit)
      -> void
      { ::ket::gate::fused::pauli_x(first, fused_index_wo_qubits, fused_qubit_masks, fused_index_masks, target_qubit); }

      // CNOT_{tc}, or C1NOT_{tc}
      // CNOT_{1,2} (a_{00} |00> + a_{01} |01> + a_{10} |10> + a_{11} |11>)
      //   = a_{00} |00> + a_{01} |01> + a_{11} |10> + a_{10} |11>
      template <typename RandomAccessIterator, typename StateInteger, std::size_t num_fused_qubits, typename BitInteger>
      inline auto not_(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array<StateInteger, num_fused_qubits> const& fused_qubit_masks, std::array<StateInteger, num_fused_qubits + 1u> const& fused_index_masks,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit)
      -> void
      { ::ket::gate::fused::pauli_x(first, fused_index_wo_qubits, fused_qubit_masks, fused_index_masks, target_qubit, control_qubit); }

      // C...CNOT_{tc...c'}, or CnNOT_{tc...c'}
      template <typename RandomAccessIterator, typename StateInteger, std::size_t num_fused_qubits, typename BitInteger, typename... ControlQubits>
      inline auto not_(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array<StateInteger, num_fused_qubits> const& fused_qubit_masks, std::array<StateInteger, num_fused_qubits + 1u> const& fused_index_masks,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2, ControlQubits const... control_qubits)
      -> void
      {
        constexpr auto num_control_qubits = static_cast<BitInteger>(sizeof...(ControlQubits) + 2u);
        constexpr auto num_operated_qubits = num_control_qubits + BitInteger{1u};
        ::ket::gate::fused::gate<num_fused_qubits>(
          first,
          [fused_index_wo_qubits, &fused_qubit_masks, &fused_index_masks](
            auto const first, StateInteger const operated_index_wo_qubits,
            std::array<StateInteger, num_operated_qubits> const& operated_qubit_masks,
            std::array<StateInteger, num_operated_qubits + 1u> const& operated_index_masks)
          {
            // 0b11...10u
            constexpr auto index0 = ((std::size_t{1u} << num_control_qubits) - std::size_t{1u}) << BitInteger{1u};
            auto const iter0
              = first
                + ::ket::gate::utility::index_with_qubits(
                    fused_index_wo_qubits,
                    ::ket::gate::utility::index_with_qubits(operated_index_wo_qubits, index0, operated_qubit_masks, operated_index_masks),
                    fused_qubit_masks, fused_index_masks);
            // 0b11...11u
            auto const iter1
              = first
                + ::ket::gate::utility::index_with_qubits(
                    fused_index_wo_qubits,
                    ::ket::gate::utility::index_with_qubits(operated_index_wo_qubits, index0 bitor std::size_t{1u}, operated_qubit_masks, operated_index_masks),
                    fused_qubit_masks, fused_index_masks);

            std::iter_swap(iter0, iter1);
          },
          target_qubit, control_qubit1, control_qubit2, control_qubits...);
      }

      template <typename RandomAccessIterator, typename StateInteger, std::size_t num_fused_qubits, typename BitInteger, typename... ControlQubits>
      inline auto adj_not_(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array<StateInteger, num_fused_qubits> const& fused_qubit_masks, std::array<StateInteger, num_fused_qubits + 1u> const& fused_index_masks,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
      -> void
      { ::ket::gate::fused::not_(first, fused_index_wo_qubits, fused_qubit_masks, fused_index_masks, target_qubit, control_qubits...); }
    } // namespace fused
  } // namespace gate
} // namespace ket


#endif // KET_GATE_FUSED_NOT_HPP
