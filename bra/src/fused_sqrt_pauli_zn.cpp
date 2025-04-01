#include <array>
#include <utility>

#include <boost/preprocessor/arithmetic/dec.hpp>
#include <boost/preprocessor/arithmetic/inc.hpp>
# include <boost/preprocessor/punctuation/comma_if.hpp>
#include <boost/preprocessor/repetition/repeat.hpp>
#include <boost/preprocessor/repetition/repeat_from_to.hpp>

#include <ket/gate/fused/sqrt_pauli_z.hpp>
#if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
# include <ket/gate/utility/cache_aware_iterator.hpp>
#endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)

#include <bra/types.hpp>
#include <bra/fused_gate/fused_gate.hpp>
#include <bra/fused_gate/fused_sqrt_pauli_zn.hpp>


namespace bra
{
  namespace fused_gate
  {
    template <typename Iterator>
    fused_sqrt_pauli_zn<Iterator>::fused_sqrt_pauli_zn(std::vector< ::bra::qubit_type > const& qubits)
      : ::bra::fused_gate::fused_gate<Iterator>{}, qubits_{qubits}
    { }

    template <typename Iterator>
    fused_sqrt_pauli_zn<Iterator>::fused_sqrt_pauli_zn(std::vector< ::bra::qubit_type >&& qubits)
      : ::bra::fused_gate::fused_gate<Iterator>{}, qubits_{std::move(qubits)}
    { }

#ifndef KET_USE_BIT_MASKS_EXPLICITLY
    template <typename Iterator>
    [[noreturn]] auto fused_sqrt_pauli_zn<Iterator>::do_call(
      Iterator const first, ::bra::state_integer_type const fused_index_wo_qubits,
      std::array< ::bra::qubit_type, 1u > const& unsorted_fused_qubits,
      std::array< ::bra::qubit_type, 2u > const& sorted_fused_qubits_with_sentinel) const -> void
    { throw 1; }

    template <typename Iterator>
    [[noreturn]] auto fused_sqrt_pauli_zn<Iterator>::do_call(
      Iterator const first, ::bra::state_integer_type const fused_index_wo_qubits,
      std::array< ::bra::qubit_type, 2u > const& unsorted_fused_qubits,
      std::array< ::bra::qubit_type, 3u > const& sorted_fused_qubits_with_sentinel) const -> void
    { throw 1; }
#else // KET_USE_BIT_MASKS_EXPLICITLY
    template <typename Iterator>
    [[noreturn]] auto fused_sqrt_pauli_zn<Iterator>::do_call(
      Iterator const first, ::bra::state_integer_type const fused_index_wo_qubits,
      std::array< ::bra::state_integer_type, 1u > const& qubit_masks,
      std::array< ::bra::state_integer_type, 2u > const& index_masks) const -> void
    { throw 1; }

    template <typename Iterator>
    [[noreturn]] auto fused_sqrt_pauli_zn<Iterator>::do_call(
      Iterator const first, ::bra::state_integer_type const fused_index_wo_qubits,
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
#define QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) qubits_[n]
#ifndef KET_USE_BIT_MASKS_EXPLICITLY
# define DO_CALL(z, num_fused_qubits, _) \
    template <typename Iterator>\
    auto fused_sqrt_pauli_zn<Iterator>::do_call(\
      Iterator const first, ::bra::state_integer_type const fused_index_wo_qubits,\
      std::array< ::bra::qubit_type, num_fused_qubits > const& unsorted_fused_qubits,\
      std::array< ::bra::qubit_type, num_fused_qubits + 1u > const& sorted_fused_qubits_with_sentinel) const -> void\
    {\
      ::ket::gate::fused::sqrt_pauli_z(\
        first, fused_index_wo_qubits, unsorted_fused_qubits, sorted_fused_qubits_with_sentinel,\
        BOOST_PP_REPEAT_ ## z(num_fused_qubits, QUBITS, nil));\
    }
#else // KET_USE_BIT_MASKS_EXPLICITLY
# define DO_CALL(z, num_fused_qubits, _) \
    template <typename Iterator>\
    auto fused_sqrt_pauli_zn<Iterator>::do_call(\
        Iterator const first, ::bra::state_integer_type const fused_index_wo_qubits,\
        std::array< ::bra::state_integer_type, num_fused_qubits > const& qubit_masks,\
        std::array< ::bra::state_integer_type, num_fused_qubits + 1u > const& index_masks) const -> void\
    {\
      ::ket::gate::fused::sqrt_pauli_z(\
        first, fused_index_wo_qubits, qubit_masks, index_masks,\
        BOOST_PP_REPEAT_ ## z(num_fused_qubits, QUBITS, nil));\
    }
#endif // KET_USE_BIT_MASKS_EXPLICITLY
BOOST_PP_REPEAT_FROM_TO(3, BOOST_PP_INC(BRA_MAX_NUM_FUSED_QUBITS), DO_CALL, nil)
#undef DO_CALL
#undef QUBITS

  template class fused_sqrt_pauli_zn< ::bra::data_type::iterator >;
#if !defined(BRA_NO_MPI) && (!defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)))
  template class fused_sqrt_pauli_zn< ::bra::paged_data_type::iterator >;
#endif // !defined(BRA_NO_MPI) && (!defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)))
#if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
  template class fused_sqrt_pauli_zn<ket::gate::utility::cache_aware_iterator< ::bra::data_type::iterator >>;
# ifndef BRA_NO_MPI
  template class fused_sqrt_pauli_zn<ket::gate::utility::cache_aware_iterator< ::bra::paged_data_type::iterator >>;
# endif // BRA_NO_MPI
#endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
  } // namespace fused_gate
} // namespace bra
