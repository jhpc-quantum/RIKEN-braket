#include <array>

#include <boost/preprocessor/arithmetic/dec.hpp>
#include <boost/preprocessor/arithmetic/inc.hpp>
#include <boost/preprocessor/repetition/repeat_from_to.hpp>

#include <ket/gate/fused/exponential_pauli_x.hpp>

#include <bra/types.hpp>
#include <bra/fused_gate/fused_gate.hpp>
#include <bra/fused_gate/fused_controlled_exponential_pauli_x.hpp>


namespace bra
{
  namespace fused_gate
  {
    template <typename Iterator>
    fused_controlled_exponential_pauli_x<Iterator>::fused_controlled_exponential_pauli_x(::bra::real_type const phase, ::bra::qubit_type const target_qubit, ::bra::control_qubit_type const control_qubit)
      : ::bra::fused_gate::fused_gate<Iterator>{}, phase_{phase}, target_qubit_{target_qubit}, control_qubit_{control_qubit}
    { }

#ifndef KET_USE_BIT_MASKS_EXPLICITLY
    template <typename Iterator>
    [[noreturn]] auto fused_controlled_exponential_pauli_x<Iterator>::do_call(
      Iterator const first, ::bra::state_integer_type const fused_index_wo_qubits,
      std::array< ::bra::qubit_type, 1u > const& unsorted_fused_qubits,
      std::array< ::bra::qubit_type, 2u > const& sorted_fused_qubits_with_sentinel) const -> void
    { throw 1; }
#else // KET_USE_BIT_MASKS_EXPLICITLY
    template <typename Iterator>
    [[noreturn]] auto fused_controlled_exponential_pauli_x<Iterator>::do_call(
      Iterator const first, ::bra::state_integer_type const fused_index_wo_qubits,
      std::array< ::bra::state_integer_type, 1u > const& qubit_masks,
      std::array< ::bra::state_integer_type, 2u > const& index_masks) const -> void
    { throw 1; }
#endif // KET_USE_BIT_MASKS_EXPLICITLY

#ifndef BRA_MAX_NUM_FUSED_QUBITS
# ifdef KET_DEFAULT_NUM_ON_CACHE_QUBITS
#   define BRA_MAX_NUM_FUSED_QUBITS BOOST_PP_DEC(KET_DEFAULT_NUM_ON_CACHE_QUBITS)
# else // KET_DEFAULT_NUM_ON_CACHE_QUBITS
#   define BRA_MAX_NUM_FUSED_QUBITS 10
# endif // KET_DEFAULT_NUM_ON_CACHE_QUBITS
#endif // BRA_MAX_NUM_FUSED_QUBITS
#ifndef KET_USE_BIT_MASKS_EXPLICITLY
# define DO_CALL(z, num_fused_qubits, _) \
    template <typename Iterator>\
    auto fused_controlled_exponential_pauli_x<Iterator>::do_call(\
      Iterator const first, ::bra::state_integer_type const fused_index_wo_qubits,\
      std::array< ::bra::qubit_type, num_fused_qubits > const& unsorted_fused_qubits,\
      std::array< ::bra::qubit_type, num_fused_qubits + 1u > const& sorted_fused_qubits_with_sentinel) const -> void\
    { ::ket::gate::fused::exponential_pauli_x(first, fused_index_wo_qubits, unsorted_fused_qubits, sorted_fused_qubits_with_sentinel, phase_, target_qubit_, control_qubit_); }
#else // KET_USE_BIT_MASKS_EXPLICITLY
# define DO_CALL(z, num_fused_qubits, _) \
    template <typename Iterator>\
    auto fused_controlled_exponential_pauli_x<Iterator>::do_call(\
      Iterator const first, ::bra::state_integer_type const fused_index_wo_qubits,\
      std::array< ::bra::state_integer_type, num_fused_qubits > const& qubit_masks,\
      std::array< ::bra::state_integer_type, num_fused_qubits + 1u > const& index_masks) const -> void\
    { ::ket::gate::fused::exponential_pauli_x(first, fused_index_wo_qubits, qubit_masks, index_masks, phase_, target_qubit_, control_qubit_); }
#endif // KET_USE_BIT_MASKS_EXPLICITLY
BOOST_PP_REPEAT_FROM_TO(2, BOOST_PP_INC(BRA_MAX_NUM_FUSED_QUBITS), DO_CALL, nil)
#undef DO_CALL

  template class fused_controlled_exponential_pauli_x< ::bra::data_type::iterator >;
#if !defined(BRA_NO_MPI) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
  template class fused_controlled_exponential_pauli_x< ::bra::paged_data_type::iterator >;
#endif // !defined(BRA_NO_MPI) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
  } // namespace fused_gate
} // namespace bra
