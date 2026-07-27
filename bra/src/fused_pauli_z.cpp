#include <array>
#include <vector>
#include <stdexcept>
#include <algorithm>


#include <ket/gate/fused/pauli_z.hpp>
#if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
# include <ket/gate/utility/cache_aware_iterator.hpp>
#endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)

#include <bra/types.hpp>
#include <bra/fused_gate/fused_gate.hpp>
#include <bra/fused_gate/fused_pauli_z.hpp>


namespace bra
{
  namespace fused_gate
  {
    template <typename Iterator>
    fused_pauli_z<Iterator>::fused_pauli_z(::bra::control_qubit_type const control_qubit)
      : ::bra::fused_gate::fused_gate<Iterator>{}, control_qubit_{control_qubit}, is_control_qubit_enabled_{true}
    { }

#ifndef KET_USE_BIT_MASKS_EXPLICITLY
    template <typename Iterator>
    auto fused_pauli_z<Iterator>::do_call(
      Iterator const first, ::bra::state_integer_type const fused_index_wo_qubits,
      std::vector< ::bra::qubit_type > const& unsorted_fused_qubits,
      std::vector< ::bra::qubit_type > const& sorted_fused_qubits_with_sentinel,
      std::vector< ::bra::bit_integer_type > const& to_qubit_index_in_fused_gates) const -> void
    {
      if (unsorted_fused_qubits.size() < std::size_t{1u})
        throw std::runtime_error{"fused_pauli_z requires at least one fused qubit"};

      std::array< ::bra::control_qubit_type, 1u > const control_qubits{{
        static_cast< ::bra::control_qubit_type >(to_qubit_index_in_fused_gates[static_cast< ::bra::bit_integer_type >(control_qubit_.qubit())])}};
      if (is_control_qubit_enabled_)
        ::ket::gate::fused::runtime::ranges::pauli_z(
          first, fused_index_wo_qubits, unsorted_fused_qubits, sorted_fused_qubits_with_sentinel,
          control_qubits);
      else
        ::ket::gate::fused::runtime::ranges::pauli_z(
          first, fused_index_wo_qubits, unsorted_fused_qubits, sorted_fused_qubits_with_sentinel);
    }
#else // KET_USE_BIT_MASKS_EXPLICITLY
    template <typename Iterator>
    auto fused_pauli_z<Iterator>::do_call(
      Iterator const first, ::bra::state_integer_type const fused_index_wo_qubits,
      std::vector< ::bra::state_integer_type > const& qubit_masks,
      std::vector< ::bra::state_integer_type > const& index_masks,
      std::vector< ::bra::bit_integer_type > const& to_qubit_index_in_fused_gates) const -> void
    {
      if (qubit_masks.size() < std::size_t{1u})
        throw std::runtime_error{"fused_pauli_z requires at least one fused qubit"};

      std::array< ::bra::control_qubit_type, 1u > const control_qubits{{
        static_cast< ::bra::control_qubit_type >(to_qubit_index_in_fused_gates[static_cast< ::bra::bit_integer_type >(control_qubit_.qubit())])}};
      if (is_control_qubit_enabled_)
        ::ket::gate::fused::runtime::ranges::pauli_z(
          first, fused_index_wo_qubits, qubit_masks, index_masks,
          control_qubits);
      else
        ::ket::gate::fused::runtime::ranges::pauli_z(
          first, fused_index_wo_qubits, qubit_masks, index_masks);
    }
#endif // KET_USE_BIT_MASKS_EXPLICITLY

    template <typename Iterator>
    auto fused_pauli_z<Iterator>::do_disable_control_qubits(
      typename std::vector< ::bra::qubit_type >::const_iterator const first,
      typename std::vector< ::bra::qubit_type >::const_iterator const last)
    -> void
    {
      is_control_qubit_enabled_
        = is_control_qubit_enabled_
          and std::none_of(
                first, last,
                [this](::bra::qubit_type const found_qubit)
                { return found_qubit == this->control_qubit_; });
    }

    template <typename Iterator>
    auto fused_pauli_z<Iterator>::do_disable_control_qubits(
      typename std::vector< ::bra::control_qubit_type >::const_iterator const first,
      typename std::vector< ::bra::control_qubit_type >::const_iterator const last)
    -> void
    {
      is_control_qubit_enabled_
        = is_control_qubit_enabled_
          and std::none_of(
                first, last,
                [this](::bra::control_qubit_type const found_control_qubit)
                { return found_control_qubit == this->control_qubit_; });
    }

    template class fused_pauli_z< ::bra::data_type::iterator >;
#if !defined(BRA_NO_MPI) && (!defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)))
    template class fused_pauli_z< ::bra::paged_data_type::iterator >;
#endif // !defined(BRA_NO_MPI) && (!defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)))
#ifndef KET_USE_BIT_MASKS_EXPLICITLY
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    template class fused_pauli_z<ket::gate::utility::cache_aware_iterator< ::bra::data_type::iterator, ::bra::qubit_type >>;
#   ifndef BRA_NO_MPI
    template class fused_pauli_z<ket::gate::utility::cache_aware_iterator< ::bra::paged_data_type::iterator, ::bra::qubit_type >>;
#   endif // BRA_NO_MPI
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
#else // KET_USE_BIT_MASKS_EXPLICITLY
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    template class fused_pauli_z<ket::gate::utility::cache_aware_iterator< ::bra::data_type::iterator, ::bra::state_integer_type >>;
#   ifndef BRA_NO_MPI
    template class fused_pauli_z<ket::gate::utility::cache_aware_iterator< ::bra::paged_data_type::iterator, ::bra::state_integer_type >>;
#   endif // BRA_NO_MPI
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
#endif // KET_USE_BIT_MASKS_EXPLICITLY
  } // namespace fused_gate
} // namespace bra
