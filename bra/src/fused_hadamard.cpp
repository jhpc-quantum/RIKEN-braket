#include <array>
#include <vector>

#include <boost/preprocessor/arithmetic/dec.hpp>
#include <boost/preprocessor/arithmetic/inc.hpp>
#include <boost/preprocessor/repetition/repeat_from_to.hpp>

#include <ket/gate/fused/hadamard.hpp>
#if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
# include <ket/gate/utility/cache_aware_iterator.hpp>
#endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)

#include <bra/types.hpp>
#include <bra/fused_gate/fused_gate.hpp>
#include <bra/fused_gate/fused_hadamard.hpp>


namespace bra
{
  namespace fused_gate
  {
    template <typename Iterator>
    fused_hadamard<Iterator>::fused_hadamard(::bra::qubit_type const qubit)
      : ::bra::fused_gate::fused_gate<Iterator>{}, qubit_{qubit}
    { }

#ifndef KET_USE_BIT_MASKS_EXPLICITLY
    template <typename Iterator>
    [[noreturn]] auto fused_hadamard<Iterator>::do_call(
      Iterator const first, ::bra::state_integer_type const fused_index_wo_qubits,
      std::array< ::bra::qubit_type, 0u > const& unsorted_fused_qubits,
      std::array< ::bra::qubit_type, 1u > const& sorted_fused_qubits_with_sentinel,
      std::vector< ::bra::bit_integer_type > const& to_qubit_index_in_fused_gates) const -> void
    { throw 1; }
#else // KET_USE_BIT_MASKS_EXPLICITLY
    template <typename Iterator>
    [[noreturn]] auto fused_hadamard<Iterator>::do_call(
      Iterator const first, ::bra::state_integer_type const fused_index_wo_qubits,
      std::array< ::bra::state_integer_type, 0u > const& qubit_masks,
      std::array< ::bra::state_integer_type, 1u > const& index_masks,
      std::vector< ::bra::bit_integer_type > const& to_qubit_index_in_fused_gates) const -> void
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
    auto fused_hadamard<Iterator>::do_call(\
      Iterator const first, ::bra::state_integer_type const fused_index_wo_qubits,\
      std::array< ::bra::qubit_type, num_fused_qubits > const& unsorted_fused_qubits,\
      std::array< ::bra::qubit_type, num_fused_qubits + 1u > const& sorted_fused_qubits_with_sentinel,\
      std::vector< ::bra::bit_integer_type > const& to_qubit_index_in_fused_gates) const -> void\
    {\
      ::ket::gate::fused::hadamard(\
        first, fused_index_wo_qubits, unsorted_fused_qubits, sorted_fused_qubits_with_sentinel,\
        static_cast< ::bra::qubit_type >(to_qubit_index_in_fused_gates[static_cast< ::bra::bit_integer_type >(qubit_)]));\
    }
#else // KET_USE_BIT_MASKS_EXPLICITLY
# define DO_CALL(z, num_fused_qubits, _) \
    template <typename Iterator>\
    auto fused_hadamard<Iterator>::do_call(\
      Iterator const first, ::bra::state_integer_type const fused_index_wo_qubits,\
      std::array< ::bra::state_integer_type, num_fused_qubits > const& qubit_masks,\
      std::array< ::bra::state_integer_type, num_fused_qubits + 1u > const& index_masks,\
      std::vector< ::bra::bit_integer_type > const& to_qubit_index_in_fused_gates) const -> void\
    {\
      ::ket::gate::fused::hadamard(\
        first, fused_index_wo_qubits, qubit_masks, index_masks,\
        static_cast< ::bra::qubit_type >(to_qubit_index_in_fused_gates[static_cast< ::bra::bit_integer_type >(qubit_)]));\
    }
#endif // KET_USE_BIT_MASKS_EXPLICITLY
BOOST_PP_REPEAT_FROM_TO(1, BOOST_PP_INC(BRA_MAX_NUM_FUSED_QUBITS), DO_CALL, nil)
#undef DO_CALL

    template class fused_hadamard< ::bra::data_type::iterator >;
#if !defined(BRA_NO_MPI) && (!defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)))
    template class fused_hadamard< ::bra::paged_data_type::iterator >;
#endif // !defined(BRA_NO_MPI) && (!defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)))
#ifndef KET_USE_BIT_MASKS_EXPLICITLY
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    template class fused_hadamard<ket::gate::utility::cache_aware_iterator< ::bra::data_type::iterator, ::bra::qubit_type >>;
#   ifndef BRA_NO_MPI
    template class fused_hadamard<ket::gate::utility::cache_aware_iterator< ::bra::paged_data_type::iterator, ::bra::qubit_type >>;
#   endif // BRA_NO_MPI
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
#else // KET_USE_BIT_MASKS_EXPLICITLY
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    template class fused_hadamard<ket::gate::utility::cache_aware_iterator< ::bra::data_type::iterator, ::bra::state_integer_type >>;
#   ifndef BRA_NO_MPI
    template class fused_hadamard<ket::gate::utility::cache_aware_iterator< ::bra::paged_data_type::iterator, ::bra::state_integer_type >>;
#   endif // BRA_NO_MPI
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
#endif // KET_USE_BIT_MASKS_EXPLICITLY
  } // namespace fused_gate
} // namespace bra
