#include <array>
#include <vector>
#include <utility>
#include <stdexcept>
#include <algorithm>

#include <ket/gate/fused/exponential_pauli_z.hpp>
#include <ket/gate/fused/phase_shift.hpp>
#include <ket/utility/exp_i.hpp>
#if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
# include <ket/gate/utility/cache_aware_iterator.hpp>
#endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)

#include <bra/types.hpp>
#include <bra/fused_gate/fused_gate.hpp>
#include <bra/fused_gate/fused_adj_multi_controlled_exponential_pauli_z.hpp>


namespace bra
{
  namespace fused_gate
  {
    template <typename Iterator>
    fused_adj_multi_controlled_exponential_pauli_z<Iterator>::fused_adj_multi_controlled_exponential_pauli_z(::bra::real_type const phase, ::bra::qubit_type const target_qubit, std::vector< ::bra::control_qubit_type > const& control_qubits)
      : ::bra::fused_gate::fused_gate<Iterator>{}, phase_{phase}, target_qubit_{target_qubit}, control_qubits_{control_qubits}, qubit_state_{::bra::fused_gate::cez_qubit_state::not_global}, is_control_qubit_enabled_vec_(control_qubits_.size(), static_cast<int>(true))
    { }

    template <typename Iterator>
    fused_adj_multi_controlled_exponential_pauli_z<Iterator>::fused_adj_multi_controlled_exponential_pauli_z(::bra::real_type const phase, ::bra::qubit_type const target_qubit, std::vector< ::bra::control_qubit_type >&& control_qubits)
      : ::bra::fused_gate::fused_gate<Iterator>{}, phase_{phase}, target_qubit_{target_qubit}, control_qubits_{std::move(control_qubits)}, qubit_state_{::bra::fused_gate::cez_qubit_state::not_global}, is_control_qubit_enabled_vec_(control_qubits_.size(), static_cast<int>(true))
    { }

#ifndef KET_USE_BIT_MASKS_EXPLICITLY
    template <typename Iterator>
    auto fused_adj_multi_controlled_exponential_pauli_z<Iterator>::do_call(
      Iterator const first, ::bra::state_integer_type const fused_index_wo_qubits,
      std::vector< ::bra::qubit_type > const& unsorted_fused_qubits,
      std::vector< ::bra::qubit_type > const& sorted_fused_qubits_with_sentinel,
      std::vector< ::bra::bit_integer_type > const& to_qubit_index_in_fused_gates) const -> void
    {
      if (unsorted_fused_qubits.size() < std::size_t{1u})
        throw std::runtime_error{"fused_adj_multi_controlled_exponential_pauli_z requires at least one fused qubit"};
      std::array< ::bra::qubit_type, 1u > const target_qubits{{
        static_cast< ::bra::qubit_type >(to_qubit_index_in_fused_gates[static_cast< ::bra::bit_integer_type >(target_qubit_)])}};
      auto enabled_control_qubits = std::vector< ::bra::control_qubit_type >{};
      using complex_type = typename std::iterator_traits<Iterator>::value_type;
      auto const num_control_qubits = control_qubits_.size();
      enabled_control_qubits.reserve(num_control_qubits);
      for (auto index = decltype(num_control_qubits){0}; index < num_control_qubits; ++index)
        if (static_cast<bool>(is_control_qubit_enabled_vec_[index]))
          enabled_control_qubits.push_back(static_cast< ::bra::control_qubit_type >(to_qubit_index_in_fused_gates[static_cast< ::bra::bit_integer_type >(control_qubits_[index].qubit())]));
      if (qubit_state_ == ::bra::fused_gate::cez_qubit_state::global_zero)
        ::ket::gate::fused::runtime::ranges::adj_phase_shift_coeff(
          first, fused_index_wo_qubits, unsorted_fused_qubits, sorted_fused_qubits_with_sentinel,
          ::ket::utility::exp_i<complex_type>(phase_), enabled_control_qubits);
      else if (qubit_state_ == ::bra::fused_gate::cez_qubit_state::global_one)
        ::ket::gate::fused::runtime::ranges::adj_phase_shift_coeff(
          first, fused_index_wo_qubits, unsorted_fused_qubits, sorted_fused_qubits_with_sentinel,
          ::ket::utility::exp_i<complex_type>(-phase_), enabled_control_qubits);
      else
        ::ket::gate::fused::runtime::ranges::adj_exponential_pauli_z(
          first, fused_index_wo_qubits, unsorted_fused_qubits, sorted_fused_qubits_with_sentinel, phase_,
          target_qubits,
          enabled_control_qubits);
    }
#else // KET_USE_BIT_MASKS_EXPLICITLY
    template <typename Iterator>
    auto fused_adj_multi_controlled_exponential_pauli_z<Iterator>::do_call(
      Iterator const first, ::bra::state_integer_type const fused_index_wo_qubits,
      std::vector< ::bra::state_integer_type > const& qubit_masks,
      std::vector< ::bra::state_integer_type > const& index_masks,
      std::vector< ::bra::bit_integer_type > const& to_qubit_index_in_fused_gates) const -> void
    {
      if (qubit_masks.size() < std::size_t{1u})
        throw std::runtime_error{"fused_adj_multi_controlled_exponential_pauli_z requires at least one fused qubit"};
      std::array< ::bra::qubit_type, 1u > const target_qubits{{
        static_cast< ::bra::qubit_type >(to_qubit_index_in_fused_gates[static_cast< ::bra::bit_integer_type >(target_qubit_)])}};
      auto enabled_control_qubits = std::vector< ::bra::control_qubit_type >{};
      using complex_type = typename std::iterator_traits<Iterator>::value_type;
      auto const num_control_qubits = control_qubits_.size();
      enabled_control_qubits.reserve(num_control_qubits);
      for (auto index = decltype(num_control_qubits){0}; index < num_control_qubits; ++index)
        if (static_cast<bool>(is_control_qubit_enabled_vec_[index]))
          enabled_control_qubits.push_back(static_cast< ::bra::control_qubit_type >(to_qubit_index_in_fused_gates[static_cast< ::bra::bit_integer_type >(control_qubits_[index].qubit())]));
      if (qubit_state_ == ::bra::fused_gate::cez_qubit_state::global_zero)
        ::ket::gate::fused::runtime::ranges::adj_phase_shift_coeff(
          first, fused_index_wo_qubits, qubit_masks, index_masks,
          ::ket::utility::exp_i<complex_type>(phase_), enabled_control_qubits);
      else if (qubit_state_ == ::bra::fused_gate::cez_qubit_state::global_one)
        ::ket::gate::fused::runtime::ranges::adj_phase_shift_coeff(
          first, fused_index_wo_qubits, qubit_masks, index_masks,
          ::ket::utility::exp_i<complex_type>(-phase_), enabled_control_qubits);
      else
        ::ket::gate::fused::runtime::ranges::adj_exponential_pauli_z(
          first, fused_index_wo_qubits, qubit_masks, index_masks, phase_,
          target_qubits,
          enabled_control_qubits);
    }
#endif // KET_USE_BIT_MASKS_EXPLICITLY

    template <typename Iterator>
    auto fused_adj_multi_controlled_exponential_pauli_z<Iterator>::do_disable_control_qubits(
      typename std::vector< ::bra::qubit_type >::const_iterator const first,
      typename std::vector< ::bra::qubit_type >::const_iterator const last)
    -> void
    {
      auto const num_control_qubits = control_qubits_.size();
      for (auto index = decltype(num_control_qubits){0}; index < num_control_qubits; ++index)
        is_control_qubit_enabled_vec_[index]
          = static_cast<int>(static_cast<bool>(is_control_qubit_enabled_vec_[index]) and std::none_of(first, last, [this, index](::bra::qubit_type const q) { return q == this->control_qubits_[index]; }));
    }

    template <typename Iterator>
    auto fused_adj_multi_controlled_exponential_pauli_z<Iterator>::do_disable_control_qubits(
      typename std::vector< ::bra::control_qubit_type >::const_iterator const first,
      typename std::vector< ::bra::control_qubit_type >::const_iterator const last)
    -> void
    {
      auto const num_control_qubits = control_qubits_.size();
      for (auto index = decltype(num_control_qubits){0}; index < num_control_qubits; ++index)
        is_control_qubit_enabled_vec_[index]
          = static_cast<int>(static_cast<bool>(is_control_qubit_enabled_vec_[index]) and std::none_of(first, last, [this, index](::bra::control_qubit_type const q) { return q == this->control_qubits_[index]; }));
    }

    template <typename Iterator>
    auto fused_adj_multi_controlled_exponential_pauli_z<Iterator>::do_modify_cez(
      typename std::vector< ::bra::qubit_type >::const_iterator const first,
      typename std::vector< ::bra::qubit_type >::const_iterator const last,
      typename std::vector< ::bra::fused_gate::cez_qubit_state >::const_iterator const cez_qubit_state_first)
    -> void
    {
      auto const found = std::find(first, last, target_qubit_);
      if (found == last)
        return;

      qubit_state_ = cez_qubit_state_first[found - first];
    }

    template class fused_adj_multi_controlled_exponential_pauli_z< ::bra::data_type::iterator >;
#if !defined(BRA_NO_MPI) && (!defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)))
    template class fused_adj_multi_controlled_exponential_pauli_z< ::bra::paged_data_type::iterator >;
#endif // !defined(BRA_NO_MPI) && (!defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)))
#ifndef KET_USE_BIT_MASKS_EXPLICITLY
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    template class fused_adj_multi_controlled_exponential_pauli_z<ket::gate::utility::cache_aware_iterator< ::bra::data_type::iterator, ::bra::qubit_type >>;
#   ifndef BRA_NO_MPI
    template class fused_adj_multi_controlled_exponential_pauli_z<ket::gate::utility::cache_aware_iterator< ::bra::paged_data_type::iterator, ::bra::qubit_type >>;
#   endif // BRA_NO_MPI
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
#else // KET_USE_BIT_MASKS_EXPLICITLY
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    template class fused_adj_multi_controlled_exponential_pauli_z<ket::gate::utility::cache_aware_iterator< ::bra::data_type::iterator, ::bra::state_integer_type >>;
#   ifndef BRA_NO_MPI
    template class fused_adj_multi_controlled_exponential_pauli_z<ket::gate::utility::cache_aware_iterator< ::bra::paged_data_type::iterator, ::bra::state_integer_type >>;
#   endif // BRA_NO_MPI
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
#endif // KET_USE_BIT_MASKS_EXPLICITLY
  } // namespace fused_gate
} // namespace bra
