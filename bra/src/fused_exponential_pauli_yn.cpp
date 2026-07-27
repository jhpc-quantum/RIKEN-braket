#include <vector>
#include <stdexcept>
#include <utility>


#include <ket/gate/fused/exponential_pauli_y.hpp>
#if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
# include <ket/gate/utility/cache_aware_iterator.hpp>
#endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)

#include <bra/types.hpp>
#include <bra/fused_gate/fused_gate.hpp>
#include <bra/fused_gate/fused_exponential_pauli_yn.hpp>


namespace bra
{
  namespace fused_gate
  {
    template <typename Iterator>
    fused_exponential_pauli_yn<Iterator>::fused_exponential_pauli_yn(::bra::real_type const phase, std::vector< ::bra::qubit_type > const& qubits)
      : ::bra::fused_gate::fused_gate<Iterator>{}, phase_{phase}, qubits_{qubits}
    { }

    template <typename Iterator>
    fused_exponential_pauli_yn<Iterator>::fused_exponential_pauli_yn(::bra::real_type const phase, std::vector< ::bra::qubit_type >&& qubits)
      : ::bra::fused_gate::fused_gate<Iterator>{}, phase_{phase}, qubits_{std::move(qubits)}
    { }

#ifndef KET_USE_BIT_MASKS_EXPLICITLY
    template <typename Iterator>
    auto fused_exponential_pauli_yn<Iterator>::do_call(
      Iterator const first, ::bra::state_integer_type const fused_index_wo_qubits,
      std::vector< ::bra::qubit_type > const& unsorted_fused_qubits,
      std::vector< ::bra::qubit_type > const& sorted_fused_qubits_with_sentinel,
      std::vector< ::bra::bit_integer_type > const& to_qubit_index_in_fused_gates) const -> void
    {
      if (unsorted_fused_qubits.size() < std::size_t{3u})
        throw std::runtime_error{"fused_exponential_pauli_yn requires at least three fused qubits"};

      std::vector< ::bra::qubit_type > target_qubits;
      target_qubits.reserve(qubits_.size());
      for (auto const qubit : qubits_)
        target_qubits.push_back(static_cast< ::bra::qubit_type >(to_qubit_index_in_fused_gates[static_cast< ::bra::bit_integer_type >(qubit)]));

      ::ket::gate::fused::runtime::ranges::exponential_pauli_y(
        first, fused_index_wo_qubits, unsorted_fused_qubits, sorted_fused_qubits_with_sentinel,
        phase_, target_qubits);
    }
#else // KET_USE_BIT_MASKS_EXPLICITLY
    template <typename Iterator>
    auto fused_exponential_pauli_yn<Iterator>::do_call(
      Iterator const first, ::bra::state_integer_type const fused_index_wo_qubits,
      std::vector< ::bra::state_integer_type > const& qubit_masks,
      std::vector< ::bra::state_integer_type > const& index_masks,
      std::vector< ::bra::bit_integer_type > const& to_qubit_index_in_fused_gates) const -> void
    {
      if (qubit_masks.size() < std::size_t{3u})
        throw std::runtime_error{"fused_exponential_pauli_yn requires at least three fused qubits"};

      std::vector< ::bra::qubit_type > target_qubits;
      target_qubits.reserve(qubits_.size());
      for (auto const qubit : qubits_)
        target_qubits.push_back(static_cast< ::bra::qubit_type >(to_qubit_index_in_fused_gates[static_cast< ::bra::bit_integer_type >(qubit)]));

      ::ket::gate::fused::runtime::ranges::exponential_pauli_y(
        first, fused_index_wo_qubits, qubit_masks, index_masks,
        phase_, target_qubits);
    }
#endif // KET_USE_BIT_MASKS_EXPLICITLY

    template class fused_exponential_pauli_yn< ::bra::data_type::iterator >;
#if !defined(BRA_NO_MPI) && (!defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)))
    template class fused_exponential_pauli_yn< ::bra::paged_data_type::iterator >;
#endif // !defined(BRA_NO_MPI) && (!defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)))
#ifndef KET_USE_BIT_MASKS_EXPLICITLY
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    template class fused_exponential_pauli_yn<ket::gate::utility::cache_aware_iterator< ::bra::data_type::iterator, ::bra::qubit_type >>;
#   ifndef BRA_NO_MPI
    template class fused_exponential_pauli_yn<ket::gate::utility::cache_aware_iterator< ::bra::paged_data_type::iterator, ::bra::qubit_type >>;
#   endif // BRA_NO_MPI
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
#else // KET_USE_BIT_MASKS_EXPLICITLY
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    template class fused_exponential_pauli_yn<ket::gate::utility::cache_aware_iterator< ::bra::data_type::iterator, ::bra::state_integer_type >>;
#   ifndef BRA_NO_MPI
    template class fused_exponential_pauli_yn<ket::gate::utility::cache_aware_iterator< ::bra::paged_data_type::iterator, ::bra::state_integer_type >>;
#   endif // BRA_NO_MPI
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
#endif // KET_USE_BIT_MASKS_EXPLICITLY
  } // namespace fused_gate
} // namespace bra
