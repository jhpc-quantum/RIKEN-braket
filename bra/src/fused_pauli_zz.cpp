#include <array>
#include <vector>
#include <stdexcept>


#include <ket/gate/fused/pauli_z.hpp>
#if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
# include <ket/gate/utility/cache_aware_iterator.hpp>
#endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)

#include <bra/types.hpp>
#include <bra/fused_gate/fused_gate.hpp>
#include <bra/fused_gate/fused_pauli_zz.hpp>


namespace bra
{
  namespace fused_gate
  {
    template <typename Iterator>
    fused_pauli_zz<Iterator>::fused_pauli_zz(::bra::qubit_type const qubit1, ::bra::qubit_type const qubit2)
      : ::bra::fused_gate::fused_gate<Iterator>{}, qubit1_{qubit1}, qubit2_{qubit2}
    { }

#ifndef KET_USE_BIT_MASKS_EXPLICITLY
    template <typename Iterator>
    auto fused_pauli_zz<Iterator>::do_call(
      Iterator const first, ::bra::state_integer_type const fused_index_wo_qubits,
      std::vector< ::bra::qubit_type > const& unsorted_fused_qubits,
      std::vector< ::bra::qubit_type > const& sorted_fused_qubits_with_sentinel,
      std::vector< ::bra::bit_integer_type > const& to_qubit_index_in_fused_gates) const -> void
    {
      if (unsorted_fused_qubits.size() < std::size_t{2u})
        throw std::runtime_error{"fused_pauli_zz requires at least two fused qubits"};

      std::array< ::bra::qubit_type, 2u > const target_qubits{{
        static_cast< ::bra::qubit_type >(to_qubit_index_in_fused_gates[static_cast< ::bra::bit_integer_type >(qubit1_)]),
        static_cast< ::bra::qubit_type >(to_qubit_index_in_fused_gates[static_cast< ::bra::bit_integer_type >(qubit2_)])}};
      ::ket::gate::fused::runtime::ranges::pauli_z(
        first, fused_index_wo_qubits, unsorted_fused_qubits, sorted_fused_qubits_with_sentinel,
        target_qubits);
    }
#else // KET_USE_BIT_MASKS_EXPLICITLY
    template <typename Iterator>
    auto fused_pauli_zz<Iterator>::do_call(
      Iterator const first, ::bra::state_integer_type const fused_index_wo_qubits,
      std::vector< ::bra::state_integer_type > const& qubit_masks,
      std::vector< ::bra::state_integer_type > const& index_masks,
      std::vector< ::bra::bit_integer_type > const& to_qubit_index_in_fused_gates) const -> void
    {
      if (qubit_masks.size() < std::size_t{2u})
        throw std::runtime_error{"fused_pauli_zz requires at least two fused qubits"};

      std::array< ::bra::qubit_type, 2u > const target_qubits{{
        static_cast< ::bra::qubit_type >(to_qubit_index_in_fused_gates[static_cast< ::bra::bit_integer_type >(qubit1_)]),
        static_cast< ::bra::qubit_type >(to_qubit_index_in_fused_gates[static_cast< ::bra::bit_integer_type >(qubit2_)])}};
      ::ket::gate::fused::runtime::ranges::pauli_z(
        first, fused_index_wo_qubits, qubit_masks, index_masks,
        target_qubits);
    }
#endif // KET_USE_BIT_MASKS_EXPLICITLY

    template class fused_pauli_zz< ::bra::data_type::iterator >;
#if !defined(BRA_NO_MPI) && (!defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)))
    template class fused_pauli_zz< ::bra::paged_data_type::iterator >;
#endif // !defined(BRA_NO_MPI) && (!defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)))
#ifndef KET_USE_BIT_MASKS_EXPLICITLY
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    template class fused_pauli_zz<ket::gate::utility::cache_aware_iterator< ::bra::data_type::iterator, ::bra::qubit_type >>;
#   ifndef BRA_NO_MPI
    template class fused_pauli_zz<ket::gate::utility::cache_aware_iterator< ::bra::paged_data_type::iterator, ::bra::qubit_type >>;
#   endif // BRA_NO_MPI
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
#else // KET_USE_BIT_MASKS_EXPLICITLY
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    template class fused_pauli_zz<ket::gate::utility::cache_aware_iterator< ::bra::data_type::iterator, ::bra::state_integer_type >>;
#   ifndef BRA_NO_MPI
    template class fused_pauli_zz<ket::gate::utility::cache_aware_iterator< ::bra::paged_data_type::iterator, ::bra::state_integer_type >>;
#   endif // BRA_NO_MPI
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
#endif // KET_USE_BIT_MASKS_EXPLICITLY
  } // namespace fused_gate
} // namespace bra
