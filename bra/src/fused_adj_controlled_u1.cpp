#include <array>
#include <vector>
#include <utility>
#include <stdexcept>
#include <algorithm>

#include <ket/gate/fused/phase_shift.hpp>
#if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
# include <ket/gate/utility/cache_aware_iterator.hpp>
#endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)

#include <bra/types.hpp>
#include <bra/fused_gate/fused_gate.hpp>
#include <bra/fused_gate/fused_adj_controlled_u1.hpp>


namespace bra
{
  namespace fused_gate
  {
    template <typename Iterator>
    fused_adj_controlled_u1<Iterator>::fused_adj_controlled_u1(::bra::real_type const phase, ::bra::control_qubit_type const control_qubit1, ::bra::control_qubit_type const control_qubit2)
      : ::bra::fused_gate::fused_gate<Iterator>{}, phase_{phase}, control_qubit1_{control_qubit1}, control_qubit2_{control_qubit2}, is_control_qubit1_enabled_{true}, is_control_qubit2_enabled_{true}
    { }

#ifndef KET_USE_BIT_MASKS_EXPLICITLY
    template <typename Iterator>
    auto fused_adj_controlled_u1<Iterator>::do_call(
      Iterator const first, ::bra::state_integer_type const fused_index_wo_qubits,
      std::vector< ::bra::qubit_type > const& unsorted_fused_qubits,
      std::vector< ::bra::qubit_type > const& sorted_fused_qubits_with_sentinel,
      std::vector< ::bra::bit_integer_type > const& to_qubit_index_in_fused_gates) const -> void
    {
      if (unsorted_fused_qubits.size() < std::size_t{1u})
        throw std::runtime_error{"fused_adj_controlled_u1 requires at least one fused qubit"};
      auto enabled_control_qubits = std::vector< ::bra::control_qubit_type >{};
      enabled_control_qubits.reserve(2u);
      if (is_control_qubit1_enabled_)
        enabled_control_qubits.push_back(static_cast< ::bra::control_qubit_type >(to_qubit_index_in_fused_gates[static_cast< ::bra::bit_integer_type >(control_qubit1_.qubit())]));
      if (is_control_qubit2_enabled_)
        enabled_control_qubits.push_back(static_cast< ::bra::control_qubit_type >(to_qubit_index_in_fused_gates[static_cast< ::bra::bit_integer_type >(control_qubit2_.qubit())]));
      ::ket::gate::fused::runtime::ranges::adj_phase_shift(
        first, fused_index_wo_qubits, unsorted_fused_qubits, sorted_fused_qubits_with_sentinel,
        phase_, enabled_control_qubits);
    }
#else // KET_USE_BIT_MASKS_EXPLICITLY
    template <typename Iterator>
    auto fused_adj_controlled_u1<Iterator>::do_call(
      Iterator const first, ::bra::state_integer_type const fused_index_wo_qubits,
      std::vector< ::bra::state_integer_type > const& qubit_masks,
      std::vector< ::bra::state_integer_type > const& index_masks,
      std::vector< ::bra::bit_integer_type > const& to_qubit_index_in_fused_gates) const -> void
    {
      if (qubit_masks.size() < std::size_t{1u})
        throw std::runtime_error{"fused_adj_controlled_u1 requires at least one fused qubit"};
      auto enabled_control_qubits = std::vector< ::bra::control_qubit_type >{};
      enabled_control_qubits.reserve(2u);
      if (is_control_qubit1_enabled_)
        enabled_control_qubits.push_back(static_cast< ::bra::control_qubit_type >(to_qubit_index_in_fused_gates[static_cast< ::bra::bit_integer_type >(control_qubit1_.qubit())]));
      if (is_control_qubit2_enabled_)
        enabled_control_qubits.push_back(static_cast< ::bra::control_qubit_type >(to_qubit_index_in_fused_gates[static_cast< ::bra::bit_integer_type >(control_qubit2_.qubit())]));
      ::ket::gate::fused::runtime::ranges::adj_phase_shift(
        first, fused_index_wo_qubits, qubit_masks, index_masks,
        phase_, enabled_control_qubits);
    }
#endif // KET_USE_BIT_MASKS_EXPLICITLY

    template <typename Iterator>
    auto fused_adj_controlled_u1<Iterator>::do_disable_control_qubits(
      typename std::vector< ::bra::qubit_type >::const_iterator const first,
      typename std::vector< ::bra::qubit_type >::const_iterator const last)
    -> void
    {
      is_control_qubit1_enabled_ = is_control_qubit1_enabled_ and std::none_of(first, last, [this](::bra::qubit_type const q) { return q == this->control_qubit1_; });
      is_control_qubit2_enabled_ = is_control_qubit2_enabled_ and std::none_of(first, last, [this](::bra::qubit_type const q) { return q == this->control_qubit2_; });
    }

    template <typename Iterator>
    auto fused_adj_controlled_u1<Iterator>::do_disable_control_qubits(
      typename std::vector< ::bra::control_qubit_type >::const_iterator const first,
      typename std::vector< ::bra::control_qubit_type >::const_iterator const last)
    -> void
    {
      is_control_qubit1_enabled_ = is_control_qubit1_enabled_ and std::none_of(first, last, [this](::bra::control_qubit_type const q) { return q == this->control_qubit1_; });
      is_control_qubit2_enabled_ = is_control_qubit2_enabled_ and std::none_of(first, last, [this](::bra::control_qubit_type const q) { return q == this->control_qubit2_; });
    }

    template class fused_adj_controlled_u1< ::bra::data_type::iterator >;
#if !defined(BRA_NO_MPI) && (!defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)))
    template class fused_adj_controlled_u1< ::bra::paged_data_type::iterator >;
#endif // !defined(BRA_NO_MPI) && (!defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)))
#ifndef KET_USE_BIT_MASKS_EXPLICITLY
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    template class fused_adj_controlled_u1<ket::gate::utility::cache_aware_iterator< ::bra::data_type::iterator, ::bra::qubit_type >>;
#   ifndef BRA_NO_MPI
    template class fused_adj_controlled_u1<ket::gate::utility::cache_aware_iterator< ::bra::paged_data_type::iterator, ::bra::qubit_type >>;
#   endif // BRA_NO_MPI
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
#else // KET_USE_BIT_MASKS_EXPLICITLY
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    template class fused_adj_controlled_u1<ket::gate::utility::cache_aware_iterator< ::bra::data_type::iterator, ::bra::state_integer_type >>;
#   ifndef BRA_NO_MPI
    template class fused_adj_controlled_u1<ket::gate::utility::cache_aware_iterator< ::bra::paged_data_type::iterator, ::bra::state_integer_type >>;
#   endif // BRA_NO_MPI
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
#endif // KET_USE_BIT_MASKS_EXPLICITLY
  } // namespace fused_gate
} // namespace bra
