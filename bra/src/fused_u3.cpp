#include <array>

#include <boost/preprocessor/arithmetic/dec.hpp>
#include <boost/preprocessor/arithmetic/inc.hpp>
#include <boost/preprocessor/repetition/repeat_from_to.hpp>

#include <ket/gate/fused/phase_shift.hpp>

#include <bra/types.hpp>
#include <bra/fused_gate/fused_gate.hpp>
#include <bra/fused_gate/fused_u3.hpp>


namespace bra
{
  namespace fused_gate
  {
    fused_u3::fused_u3(::bra::real_type const phase1, ::bra::real_type const phase2, ::bra::real_type const phase3, ::bra::qubit_type const qubit)
      : ::bra::fused_gate::fused_gate{}, phase1_{phase1}, phase2_{phase2}, phase3_{phase3}, qubit_{qubit}
    { }

#ifndef BRA_MAX_NUM_FUSED_QUBITS
# ifdef KET_DEFAULT_NUM_ON_CACHE_QUBITS
#   define BRA_MAX_NUM_FUSED_QUBITS BOOST_PP_DEC(KET_DEFAULT_NUM_ON_CACHE_QUBITS)
# else // KET_DEFAULT_NUM_ON_CACHE_QUBITS
#   define BRA_MAX_NUM_FUSED_QUBITS 10
# endif // KET_DEFAULT_NUM_ON_CACHE_QUBITS
#endif // BRA_MAX_NUM_FUSED_QUBITS
#ifndef KET_USE_BIT_MASKS_EXPLICITLY
# define DO_CALL(z, num_fused_qubits, _) \
    auto fused_u3::do_call(\
      ::bra::complex_type* const first, ::bra::state_integer_type const fused_index_wo_qubits,\
      std::array< ::bra::qubit_type, num_fused_qubits > const& unsorted_fused_qubits,\
      std::array< ::bra::qubit_type, num_fused_qubits + 1u > const& sorted_fused_qubits_with_sentinel) const -> void\
    { ::ket::gate::fused::phase_shift3(first, fused_index_wo_qubits, unsorted_fused_qubits, sorted_fused_qubits_with_sentinel, phase1_, phase2_, phase3_, qubit_); }
#else // KET_USE_BIT_MASKS_EXPLICITLY
# define DO_CALL(z, num_fused_qubits, _) \
    auto fused_u3::do_call(\
      ::bra::complex_type* const first, ::bra::state_integer_type const fused_index_wo_qubits,\
      std::array< ::bra::state_integer_type, num_fused_qubits > const& qubit_masks,\
      std::array< ::bra::state_integer_type, num_fused_qubits + 1u > const& index_masks) const -> void\
    { ::ket::gate::fused::phase_shift3(first, fused_index_wo_qubits, qubit_masks, index_masks, phase1_, phase2_, phase3_, qubit_); }
#endif // KET_USE_BIT_MASKS_EXPLICITLY
BOOST_PP_REPEAT_FROM_TO(1, BOOST_PP_INC(BRA_MAX_NUM_FUSED_QUBITS), DO_CALL, nil)
#undef DO_CALL
  } // namespace fused_gate
} // namespace bra
