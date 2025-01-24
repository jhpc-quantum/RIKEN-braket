#include <array>
#include <utility>

#include <boost/preprocessor/arithmetic/dec.hpp>
#include <boost/preprocessor/arithmetic/inc.hpp>
# include <boost/preprocessor/punctuation/comma_if.hpp>
#include <boost/preprocessor/repetition/repeat.hpp>
#include <boost/preprocessor/repetition/repeat_from_to.hpp>

#include <ket/gate/fused/phase_shift.hpp>

#include <bra/types.hpp>
#include <bra/fused_gate/fused_gate.hpp>
#include <bra/fused_gate/fused_adj_multi_controlled_u3.hpp>


namespace bra
{
  namespace fused_gate
  {
    fused_adj_multi_controlled_u3::fused_adj_multi_controlled_u3(::bra::real_type const phase1, ::bra::real_type const phase2, ::bra::real_type const phase3, ::bra::qubit_type const target_qubit, std::vector< ::bra::control_qubit_type > const& control_qubits)
      : ::bra::fused_gate::fused_gate{}, phase1_{phase1}, phase2_{phase2}, phase3_{phase3}, target_qubit_{target_qubit}, control_qubits_{control_qubits}
    { }

    fused_adj_multi_controlled_u3::fused_adj_multi_controlled_u3(::bra::real_type const phase1, ::bra::real_type const phase2, ::bra::real_type const phase3, ::bra::qubit_type const target_qubit, std::vector< ::bra::control_qubit_type >&& control_qubits)
      : ::bra::fused_gate::fused_gate{}, phase1_{phase1}, phase2_{phase2}, phase3_{phase3}, target_qubit_{target_qubit}, control_qubits_{std::move(control_qubits)}
    { }

#ifndef KET_USE_BIT_MASKS_EXPLICITLY
    [[noreturn]] auto fused_adj_multi_controlled_u3::do_call(
      ::bra::complex_type* const first, ::bra::state_integer_type const fused_index_wo_qubits,
      std::array< ::bra::qubit_type, 1u > const& unsorted_fused_qubits,
      std::array< ::bra::qubit_type, 2u > const& sorted_fused_qubits_with_sentinel) const -> void
    { throw 1; }

    [[noreturn]] auto fused_adj_multi_controlled_u3::do_call(
      ::bra::complex_type* const first, ::bra::state_integer_type const fused_index_wo_qubits,
      std::array< ::bra::qubit_type, 2u > const& unsorted_fused_qubits,
      std::array< ::bra::qubit_type, 3u > const& sorted_fused_qubits_with_sentinel) const -> void
    { throw 1; }
#else // KET_USE_BIT_MASKS_EXPLICITLY
    [[noreturn]] auto fused_adj_multi_controlled_u3::do_call(\
      ::bra::complex_type* const first, ::bra::state_integer_type const fused_index_wo_qubits,
      std::array< ::bra::state_integer_type, 1u > const& qubit_masks,
      std::array< ::bra::state_integer_type, 2u > const& index_masks) const -> void
    { throw 1; }

    [[noreturn]] auto fused_adj_multi_controlled_u3::do_call(\
      ::bra::complex_type* const first, ::bra::state_integer_type const fused_index_wo_qubits,
      std::array< ::bra::state_integer_type, 2u > const& qubit_masks,
      std::array< ::bra::state_integer_type, 3u > const& index_masks) const -> void
    { throw 1; }
#endif // KET_USE_BIT_MASKS_EXPLICITLY

#ifndef BRA_MAX_NUM_FUSED_QUBITS
# ifdef KET_DEFAULT_NUM_ON_CACHE_QUBITS
#   define BRA_MAX_NUM_FUSED_QUBITS BOOST_PP_DEC(KET_DEFAULT_NUM_ON_CACHE_QUBITS)
# else // KET_DEFAULT_NUM_ON_CACHE_QUBITS
#   define BRA_MAX_NUM_FUSED_QUBITS 10
# endif // KET_DEFAULT_NUM_ON_CACHE_QUBITS
#endif // BRA_MAX_NUM_FUSED_QUBITS
#define CONTROL_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) control_qubits_[n]
#ifndef KET_USE_BIT_MASKS_EXPLICITLY
# define DO_CALL(z, num_fused_qubits, _) \
    auto fused_adj_multi_controlled_u3::do_call(\
      ::bra::complex_type* const first, ::bra::state_integer_type const fused_index_wo_qubits,\
      std::array< ::bra::qubit_type, num_fused_qubits > const& unsorted_fused_qubits,\
      std::array< ::bra::qubit_type, num_fused_qubits + 1u > const& sorted_fused_qubits_with_sentinel) const -> void\
    {\
      ::ket::gate::fused::adj_phase_shift3(\
        first, fused_index_wo_qubits, unsorted_fused_qubits, sorted_fused_qubits_with_sentinel,\
        phase1_, phase2_, phase3_, target_qubit_, BOOST_PP_REPEAT_ ## z(BOOST_PP_DEC(num_fused_qubits), CONTROL_QUBITS, nil));\
    }
#else // KET_USE_BIT_MASKS_EXPLICITLY
# define DO_CALL(z, num_fused_qubits, _) \
    auto fused_adj_multi_controlled_u3::do_call(\
        ::bra::complex_type* const first, ::bra::state_integer_type const fused_index_wo_qubits,\
        std::array< ::bra::state_integer_type, num_fused_qubits > const& qubit_masks,\
        std::array< ::bra::state_integer_type, num_fused_qubits + 1u > const& index_masks) const -> void\
    {\
      ::ket::gate::fused::adj_phase_shift3(\
        first, fused_index_wo_qubits, qubit_masks, index_masks,\
        phase1_, phase2_, phase3_, target_qubit_, BOOST_PP_REPEAT_ ## z(BOOST_PP_DEC(num_fused_qubits), CONTROL_QUBITS, nil));\
    }
#endif // KET_USE_BIT_MASKS_EXPLICITLY
BOOST_PP_REPEAT_FROM_TO(3, BOOST_PP_INC(BRA_MAX_NUM_FUSED_QUBITS), DO_CALL, nil)
#undef DO_CALL
#undef CONTROL_QUBITS
  } // namespace fused_gate
} // namespace bra
