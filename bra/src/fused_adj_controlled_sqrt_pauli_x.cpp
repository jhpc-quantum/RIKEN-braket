#include <array>
#include <vector>
#include <utility>
#include <stdexcept>
#include <algorithm>

#include <ket/gate/fused/sqrt_pauli_x.hpp>
#include <ket/utility/exp_i.hpp>
#if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
# include <ket/gate/utility/cache_aware_iterator.hpp>
#endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)

#include <bra/types.hpp>
#include <bra/fused_gate/fused_gate.hpp>
#include <bra/fused_gate/fused_adj_controlled_sqrt_pauli_x.hpp>


namespace bra
{
  namespace fused_gate
  {
    template <typename Iterator>
    fused_adj_controlled_sqrt_pauli_x<Iterator>::fused_adj_controlled_sqrt_pauli_x(::bra::qubit_type const target_qubit, ::bra::control_qubit_type const control_qubit)
      : ::bra::fused_gate::fused_gate<Iterator>{}, target_qubit_{target_qubit}, control_qubit_{control_qubit}, is_control_qubit_enabled_{true}
    { }

#ifndef KET_USE_BIT_MASKS_EXPLICITLY
    template <typename Iterator>
    auto fused_adj_controlled_sqrt_pauli_x<Iterator>::do_call(
      Iterator const first, ::bra::state_integer_type const fused_index_wo_qubits,
      std::vector< ::bra::qubit_type > const& unsorted_fused_qubits,
      std::vector< ::bra::qubit_type > const& sorted_fused_qubits_with_sentinel,
      std::vector< ::bra::bit_integer_type > const& to_qubit_index_in_fused_gates) const -> void
    {
      if (unsorted_fused_qubits.size() < std::size_t{1u})
        throw std::runtime_error{"fused_adj_controlled_sqrt_pauli_x requires at least one fused qubit"};
      auto const target_qubit = static_cast< ::bra::qubit_type >(to_qubit_index_in_fused_gates[static_cast< ::bra::bit_integer_type >(target_qubit_)]);
      if (is_control_qubit_enabled_)
      {
        std::array< ::bra::control_qubit_type, 1u > const control_qubits{{
          static_cast< ::bra::control_qubit_type >(to_qubit_index_in_fused_gates[static_cast< ::bra::bit_integer_type >(control_qubit_.qubit())])}};
        ::ket::gate::fused::runtime::ranges::adj_sqrt_pauli_x(
          first, fused_index_wo_qubits, unsorted_fused_qubits, sorted_fused_qubits_with_sentinel,
          target_qubit, control_qubits);
      }
      else
        ::ket::gate::fused::runtime::ranges::adj_sqrt_pauli_x(
          first, fused_index_wo_qubits, unsorted_fused_qubits, sorted_fused_qubits_with_sentinel,
          target_qubit);
    }
#else // KET_USE_BIT_MASKS_EXPLICITLY
    template <typename Iterator>
    auto fused_adj_controlled_sqrt_pauli_x<Iterator>::do_call(
      Iterator const first, ::bra::state_integer_type const fused_index_wo_qubits,
      std::vector< ::bra::state_integer_type > const& qubit_masks,
      std::vector< ::bra::state_integer_type > const& index_masks,
      std::vector< ::bra::bit_integer_type > const& to_qubit_index_in_fused_gates) const -> void
    {
      if (qubit_masks.size() < std::size_t{1u})
        throw std::runtime_error{"fused_adj_controlled_sqrt_pauli_x requires at least one fused qubit"};
      auto const target_qubit = static_cast< ::bra::qubit_type >(to_qubit_index_in_fused_gates[static_cast< ::bra::bit_integer_type >(target_qubit_)]);
      if (is_control_qubit_enabled_)
      {
        std::array< ::bra::control_qubit_type, 1u > const control_qubits{{
          static_cast< ::bra::control_qubit_type >(to_qubit_index_in_fused_gates[static_cast< ::bra::bit_integer_type >(control_qubit_.qubit())])}};
        ::ket::gate::fused::runtime::ranges::adj_sqrt_pauli_x(
          first, fused_index_wo_qubits, qubit_masks, index_masks,
          target_qubit, control_qubits);
      }
      else
        ::ket::gate::fused::runtime::ranges::adj_sqrt_pauli_x(
          first, fused_index_wo_qubits, qubit_masks, index_masks,
          target_qubit);
    }
#endif // KET_USE_BIT_MASKS_EXPLICITLY

    template <typename Iterator>
    auto fused_adj_controlled_sqrt_pauli_x<Iterator>::do_disable_control_qubits(
      typename std::vector< ::bra::qubit_type >::const_iterator const first,
      typename std::vector< ::bra::qubit_type >::const_iterator const last)
    -> void
    {
      is_control_qubit_enabled_
        = is_control_qubit_enabled_
          and std::none_of(first, last, [this](::bra::qubit_type const found_qubit) { return found_qubit == this->control_qubit_; });
    }

    template <typename Iterator>
    auto fused_adj_controlled_sqrt_pauli_x<Iterator>::do_disable_control_qubits(
      typename std::vector< ::bra::control_qubit_type >::const_iterator const first,
      typename std::vector< ::bra::control_qubit_type >::const_iterator const last)
    -> void
    {
      is_control_qubit_enabled_
        = is_control_qubit_enabled_
          and std::none_of(first, last, [this](::bra::control_qubit_type const found_control_qubit) { return found_control_qubit == this->control_qubit_; });
    }

    template class fused_adj_controlled_sqrt_pauli_x< ::bra::data_type::iterator >;
#if !defined(BRA_NO_MPI) && (!defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)))
    template class fused_adj_controlled_sqrt_pauli_x< ::bra::paged_data_type::iterator >;
#endif // !defined(BRA_NO_MPI) && (!defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)))
#ifndef KET_USE_BIT_MASKS_EXPLICITLY
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    template class fused_adj_controlled_sqrt_pauli_x<ket::gate::utility::cache_aware_iterator< ::bra::data_type::iterator, ::bra::qubit_type >>;
#   ifndef BRA_NO_MPI
    template class fused_adj_controlled_sqrt_pauli_x<ket::gate::utility::cache_aware_iterator< ::bra::paged_data_type::iterator, ::bra::qubit_type >>;
#   endif // BRA_NO_MPI
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
#else // KET_USE_BIT_MASKS_EXPLICITLY
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    template class fused_adj_controlled_sqrt_pauli_x<ket::gate::utility::cache_aware_iterator< ::bra::data_type::iterator, ::bra::state_integer_type >>;
#   ifndef BRA_NO_MPI
    template class fused_adj_controlled_sqrt_pauli_x<ket::gate::utility::cache_aware_iterator< ::bra::paged_data_type::iterator, ::bra::state_integer_type >>;
#   endif // BRA_NO_MPI
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
#endif // KET_USE_BIT_MASKS_EXPLICITLY
  } // namespace fused_gate
} // namespace bra
