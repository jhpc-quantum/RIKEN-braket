#ifndef BRA_FUSED_GATE_FUSED_GATE_HPP
# define BRA_FUSED_GATE_FUSED_GATE_HPP

# include <cassert>
# include <vector>

# include <boost/optional.hpp>

# include <bra/types.hpp>


namespace bra
{
  namespace fused_gate
  {
    enum class cez_qubit_state : int { not_global, global_zero, global_one };

    template <typename Iterator>
    class fused_gate
    {
     public:
      fused_gate() = default;
      virtual ~fused_gate() = default;

      fused_gate(fused_gate const&) = delete;
      fused_gate& operator=(fused_gate const&) = delete;
      fused_gate(fused_gate&&) = delete;
      fused_gate& operator=(fused_gate&&) = delete;

# ifndef KET_USE_BIT_MASKS_EXPLICITLY
      auto call(
        Iterator const first, ::bra::state_integer_type const fused_index_wo_qubits,
        std::vector< ::bra::qubit_type > const& unsorted_fused_qubits,
        std::vector< ::bra::qubit_type > const& sorted_fused_qubits_with_sentinel,
        std::vector< ::bra::bit_integer_type > const& to_qubit_index_in_fused_gates) const -> void
      {
        assert(sorted_fused_qubits_with_sentinel.size() == unsorted_fused_qubits.size() + std::size_t{1u});
        do_call(first, fused_index_wo_qubits, unsorted_fused_qubits, sorted_fused_qubits_with_sentinel, to_qubit_index_in_fused_gates);
      }
# else // KET_USE_BIT_MASKS_EXPLICITLY
      auto call(
        Iterator const first, ::bra::state_integer_type const fused_index_wo_qubits,
        std::vector< ::bra::state_integer_type > const& qubit_masks,
        std::vector< ::bra::state_integer_type > const& index_masks,
        std::vector< ::bra::bit_integer_type > const& to_qubit_index_in_fused_gates) const -> void
      {
        assert(index_masks.size() == qubit_masks.size() + std::size_t{1u});
        do_call(first, fused_index_wo_qubits, qubit_masks, index_masks, to_qubit_index_in_fused_gates);
      }
# endif // KET_USE_BIT_MASKS_EXPLICITLY

      auto disable_control_qubits(
        typename std::vector< ::bra::qubit_type >::const_iterator const first,
        typename std::vector< ::bra::qubit_type >::const_iterator const last)
      -> void
      { do_disable_control_qubits(first, last); }

      auto disable_control_qubits(
        typename std::vector< ::bra::control_qubit_type >::const_iterator const first,
        typename std::vector< ::bra::control_qubit_type >::const_iterator const last)
      -> void
      { do_disable_control_qubits(first, last); }

      auto disable_cez_global_qubits(
        typename std::vector< ::bra::qubit_type >::const_iterator const first,
        typename std::vector< ::bra::qubit_type >::const_iterator const last)
      -> void
      { do_disable_control_qubits(first, last); }

      auto modify_cez(
        typename std::vector< ::bra::qubit_type >::const_iterator const first,
        typename std::vector< ::bra::qubit_type >::const_iterator const last,
        typename std::vector< ::bra::fused_gate::cez_qubit_state >::const_iterator const cez_qubit_state_first)
      -> void
      { do_modify_cez(first, last, cez_qubit_state_first); }

      auto maybe_phase_shiftize_ez(
        typename std::vector< ::bra::qubit_type >::const_iterator const first,
        typename std::vector< ::bra::qubit_type >::const_iterator const last)
      -> boost::optional<std::pair< ::bra::control_qubit_type, ::bra::real_type >>
      { return do_maybe_phase_shiftize_ez(first, last); }

     private:
# ifndef KET_USE_BIT_MASKS_EXPLICITLY
      virtual auto do_call(
        Iterator const first, ::bra::state_integer_type const fused_index_wo_qubits,
        std::vector< ::bra::qubit_type > const& unsorted_fused_qubits,
        std::vector< ::bra::qubit_type > const& sorted_fused_qubits_with_sentinel,
        std::vector< ::bra::bit_integer_type > const& to_qubit_index_in_fused_gates) const -> void = 0;
# else // KET_USE_BIT_MASKS_EXPLICITLY
      virtual auto do_call(
        Iterator const first, ::bra::state_integer_type const fused_index_wo_qubits,
        std::vector< ::bra::state_integer_type > const& qubit_masks,
        std::vector< ::bra::state_integer_type > const& index_masks,
        std::vector< ::bra::bit_integer_type > const& to_qubit_index_in_fused_gates) const -> void = 0;
# endif // KET_USE_BIT_MASKS_EXPLICITLY

      virtual auto do_disable_control_qubits(
        typename std::vector< ::bra::qubit_type >::const_iterator const first,
        typename std::vector< ::bra::qubit_type >::const_iterator const last) -> void;

      virtual auto do_disable_control_qubits(
        typename std::vector< ::bra::control_qubit_type >::const_iterator const first,
        typename std::vector< ::bra::control_qubit_type >::const_iterator const last) -> void;

      virtual auto do_modify_cez(
        typename std::vector< ::bra::qubit_type >::const_iterator const first,
        typename std::vector< ::bra::qubit_type >::const_iterator const last,
        typename std::vector< ::bra::fused_gate::cez_qubit_state >::const_iterator const cez_qubit_state_first) -> void;

      virtual auto do_maybe_phase_shiftize_ez(
        typename std::vector< ::bra::qubit_type >::const_iterator const first,
        typename std::vector< ::bra::qubit_type >::const_iterator const last)
      -> boost::optional<std::pair< ::bra::control_qubit_type, ::bra::real_type >>;
    }; // class fused_gate<Iterator>
  } // namespace fused_gate
} // namespace bra


#endif // BRA_FUSED_GATE_FUSED_GATE_HPP
