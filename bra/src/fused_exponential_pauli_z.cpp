#include <array>
#include <vector>

#include <boost/preprocessor/arithmetic/dec.hpp>
#include <boost/preprocessor/arithmetic/inc.hpp>
#include <boost/preprocessor/repetition/repeat_from_to.hpp>

#include <ket/gate/fused/exponential_pauli_z.hpp>
#include <ket/gate/fused/phase_shift.hpp>
#if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
# include <ket/gate/utility/cache_aware_iterator.hpp>
#endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)

#include <bra/types.hpp>
#include <bra/fused_gate/fused_gate.hpp>
#include <bra/fused_gate/fused_exponential_pauli_z.hpp>


namespace bra
{
  namespace fused_gate
  {
    template <typename Iterator>
    fused_exponential_pauli_z<Iterator>::fused_exponential_pauli_z(::bra::real_type const phase, ::bra::qubit_type const qubit)
      : ::bra::fused_gate::fused_gate<Iterator>{}, phase_{phase}, qubit_{qubit}, qubit_state_{::bra::fused_gate::cez_qubit_state::not_global}, is_qubit_unit_{false}
    { }

#ifndef KET_USE_BIT_MASKS_EXPLICITLY
    template <typename Iterator>
    [[noreturn]] auto fused_exponential_pauli_z<Iterator>::do_call(
      Iterator const first, ::bra::state_integer_type const fused_index_wo_qubits,
      std::array< ::bra::qubit_type, 0u > const& unsorted_fused_qubits,
      std::array< ::bra::qubit_type, 1u > const& sorted_fused_qubits_with_sentinel,
      std::vector< ::bra::bit_integer_type > const& to_qubit_index_in_fused_gates) const -> void
    { throw 1; }
#else // KET_USE_BIT_MASKS_EXPLICITLY
    template <typename Iterator>
    [[noreturn]] auto fused_exponential_pauli_z<Iterator>::do_call(
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
    auto fused_exponential_pauli_z<Iterator>::do_call(\
      Iterator const first, ::bra::state_integer_type const fused_index_wo_qubits,\
      std::array< ::bra::qubit_type, num_fused_qubits > const& unsorted_fused_qubits,\
      std::array< ::bra::qubit_type, num_fused_qubits + 1u > const& sorted_fused_qubits_with_sentinel,\
      std::vector< ::bra::bit_integer_type > const& to_qubit_index_in_fused_gates) const -> void\
    {\
      assert(not (qubit_state_ != ::bra::fused_gate::cez_qubit_state::not_global and is_qubit_unit_));\
      if (is_qubit_unit_)\
        ket::gate::fused::phase_shift(\
          first, fused_index_wo_qubits, unsorted_fused_qubits, sorted_fused_qubits_with_sentinel,\
          ::bra::real_type{-2} * phase_);\
      else if (qubit_state_ == ::bra::fused_gate::cez_qubit_state::global_zero)\
        ket::gate::fused::phase_shift(\
          first, fused_index_wo_qubits, unsorted_fused_qubits, sorted_fused_qubits_with_sentinel,\
          phase_);\
      else if (qubit_state_ == ::bra::fused_gate::cez_qubit_state::global_one)\
        ket::gate::fused::phase_shift(\
          first, fused_index_wo_qubits, unsorted_fused_qubits, sorted_fused_qubits_with_sentinel,\
          -phase_);\
      else\
        ket::gate::fused::exponential_pauli_z(\
          first, fused_index_wo_qubits, unsorted_fused_qubits, sorted_fused_qubits_with_sentinel,\
          phase_,\
          static_cast< ::bra::qubit_type >(to_qubit_index_in_fused_gates[static_cast< ::bra::bit_integer_type >(qubit_)]));\
    }
#else // KET_USE_BIT_MASKS_EXPLICITLY
# define DO_CALL(z, num_fused_qubits, _) \
    template <typename Iterator>\
    auto fused_exponential_pauli_z<Iterator>::do_call(\
      Iterator const first, ::bra::state_integer_type const fused_index_wo_qubits,\
      std::array< ::bra::state_integer_type, num_fused_qubits > const& qubit_masks,\
      std::array< ::bra::state_integer_type, num_fused_qubits + 1u > const& index_masks,\
      std::vector< ::bra::bit_integer_type > const& to_qubit_index_in_fused_gates) const -> void\
    {\
      assert(not (qubit_state_ != ::bra::fused_gate::cez_qubit_state::not_global and is_qubit_unit_));\
      if (is_qubit_unit_)\
        ket::gate::fused::phase_shift(\
          first, fused_index_wo_qubits, qubit_masks, index_masks,\
          ::bra::real_type{-2} * phase_);\
      else if (qubit_state_ == ::bra::fused_gate::cez_qubit_state::global_zero)\
        ket::gate::fused::phase_shift(\
          first, fused_index_wo_qubits, qubit_masks, index_masks,\
          phase_);\
      else if (qubit_state_ == ::bra::fused_gate::cez_qubit_state::global_one)\
        ket::gate::fused::phase_shift(\
          first, fused_index_wo_qubits, qubit_masks, index_masks,\
          -phase_);\
      else\
        ket::gate::fused::exponential_pauli_z(\
          first, fused_index_wo_qubits, qubit_masks, index_masks,\
          phase_,\
          static_cast< ::bra::qubit_type >(to_qubit_index_in_fused_gates[static_cast< ::bra::bit_integer_type >(qubit_)]));\
    }
#endif // KET_USE_BIT_MASKS_EXPLICITLY
BOOST_PP_REPEAT_FROM_TO(1, BOOST_PP_INC(BRA_MAX_NUM_FUSED_QUBITS), DO_CALL, nil)
#undef DO_CALL

    template <typename Iterator>
    auto fused_exponential_pauli_z<Iterator>::do_modify_cez(
      typename std::vector< ::bra::qubit_type >::const_iterator const first,
      typename std::vector< ::bra::qubit_type >::const_iterator const last,
      typename std::vector< ::bra::fused_gate::cez_qubit_state >::const_iterator const cez_qubit_state_first)
    -> void
    {
      auto const found = std::find(first, last, qubit_);
      if (found == last)
        return;

      qubit_state_ = cez_qubit_state_first[found - first];
    }

    template <typename Iterator>
    auto fused_exponential_pauli_z<Iterator>::do_maybe_phase_shiftize_ez(
      typename std::vector< ::bra::qubit_type >::const_iterator const first,
      typename std::vector< ::bra::qubit_type >::const_iterator const last)
    -> boost::optional<std::pair< ::bra::control_qubit_type, ::bra::real_type >>
    {
      if (std::none_of(first, last, [this](::bra::qubit_type const found_qubit) { return found_qubit == this->qubit_; }))
        return boost::none;

      is_qubit_unit_ = true;
      return std::make_pair(ket::make_control(qubit_), phase_);
    }

    template class fused_exponential_pauli_z< ::bra::data_type::iterator >;
#if !defined(BRA_NO_MPI) && (!defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)))
    template class fused_exponential_pauli_z< ::bra::paged_data_type::iterator >;
#endif // !defined(BRA_NO_MPI) && (!defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)))
#ifndef KET_USE_BIT_MASKS_EXPLICITLY
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    template class fused_exponential_pauli_z<ket::gate::utility::cache_aware_iterator< ::bra::data_type::iterator, ::bra::qubit_type >>;
#   ifndef BRA_NO_MPI
    template class fused_exponential_pauli_z<ket::gate::utility::cache_aware_iterator< ::bra::paged_data_type::iterator, ::bra::qubit_type >>;
#   endif // BRA_NO_MPI
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
#else // KET_USE_BIT_MASKS_EXPLICITLY
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    template class fused_exponential_pauli_z<ket::gate::utility::cache_aware_iterator< ::bra::data_type::iterator, ::bra::state_integer_type >>;
#   ifndef BRA_NO_MPI
    template class fused_exponential_pauli_z<ket::gate::utility::cache_aware_iterator< ::bra::paged_data_type::iterator, ::bra::state_integer_type >>;
#   endif // BRA_NO_MPI
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
#endif // KET_USE_BIT_MASKS_EXPLICITLY
  } // namespace fused_gate
} // namespace bra
