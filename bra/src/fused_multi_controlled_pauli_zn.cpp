#include <array>
#include <vector>
#include <utility>

#include <boost/preprocessor/arithmetic/dec.hpp>
#include <boost/preprocessor/arithmetic/inc.hpp>
#include <boost/preprocessor/arithmetic/sub.hpp>
#include <boost/preprocessor/repetition/repeat.hpp>
#include <boost/preprocessor/repetition/repeat_from_to.hpp>

#include <ket/gate/fused/pauli_z.hpp>
#if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
# include <ket/gate/utility/cache_aware_iterator.hpp>
#endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)

#include <bra/types.hpp>
#include <bra/fused_gate/fused_gate.hpp>
#include <bra/fused_gate/fused_multi_controlled_pauli_zn.hpp>


namespace bra
{
  namespace fused_gate
  {
    template <typename Iterator>
    fused_multi_controlled_pauli_zn<Iterator>::fused_multi_controlled_pauli_zn(std::vector< ::bra::qubit_type > const& target_qubits, std::vector< ::bra::control_qubit_type > const& control_qubits)
      : ::bra::fused_gate::fused_gate<Iterator>{}, target_qubits_{target_qubits}, control_qubits_{control_qubits}, is_control_qubit_enabled_vec_(control_qubits_.size(), static_cast<int>(true))
    { }

    template <typename Iterator>
    fused_multi_controlled_pauli_zn<Iterator>::fused_multi_controlled_pauli_zn(std::vector< ::bra::qubit_type >&& target_qubits, std::vector< ::bra::control_qubit_type > const& control_qubits)
      : ::bra::fused_gate::fused_gate<Iterator>{}, target_qubits_{std::move(target_qubits)}, control_qubits_{control_qubits}, is_control_qubit_enabled_vec_(control_qubits_.size(), static_cast<int>(true))
    { }

    template <typename Iterator>
    fused_multi_controlled_pauli_zn<Iterator>::fused_multi_controlled_pauli_zn(std::vector< ::bra::qubit_type > const& target_qubits, std::vector< ::bra::control_qubit_type >&& control_qubits)
      : ::bra::fused_gate::fused_gate<Iterator>{}, target_qubits_{target_qubits}, control_qubits_{std::move(control_qubits)}, is_control_qubit_enabled_vec_(control_qubits_.size(), static_cast<int>(true))
    { }

    template <typename Iterator>
    fused_multi_controlled_pauli_zn<Iterator>::fused_multi_controlled_pauli_zn(std::vector< ::bra::qubit_type >&& target_qubits, std::vector< ::bra::control_qubit_type >&& control_qubits)
      : ::bra::fused_gate::fused_gate<Iterator>{}, target_qubits_{std::move(target_qubits)}, control_qubits_{std::move(control_qubits)}, is_control_qubit_enabled_vec_(control_qubits_.size(), static_cast<int>(true))
    { }

#ifndef KET_USE_BIT_MASKS_EXPLICITLY
    template <typename Iterator>
    [[noreturn]] auto fused_multi_controlled_pauli_zn<Iterator>::do_call(
      Iterator const first, ::bra::state_integer_type const fused_index_wo_qubits,
      std::array< ::bra::qubit_type, 0u > const& unsorted_fused_qubits,
      std::array< ::bra::qubit_type, 1u > const& sorted_fused_qubits_with_sentinel,
      std::vector< ::bra::bit_integer_type > const& to_qubit_index_in_fused_gates) const -> void
    { throw 1; }

    template <typename Iterator>
    [[noreturn]] auto fused_multi_controlled_pauli_zn<Iterator>::do_call(
      Iterator const first, ::bra::state_integer_type const fused_index_wo_qubits,
      std::array< ::bra::qubit_type, 1u > const& unsorted_fused_qubits,
      std::array< ::bra::qubit_type, 2u > const& sorted_fused_qubits_with_sentinel,
      std::vector< ::bra::bit_integer_type > const& to_qubit_index_in_fused_gates) const -> void
    { throw 1; }

    template <typename Iterator>
    [[noreturn]] auto fused_multi_controlled_pauli_zn<Iterator>::do_call(
      Iterator const first, ::bra::state_integer_type const fused_index_wo_qubits,
      std::array< ::bra::qubit_type, 2u > const& unsorted_fused_qubits,
      std::array< ::bra::qubit_type, 3u > const& sorted_fused_qubits_with_sentinel,
      std::vector< ::bra::bit_integer_type > const& to_qubit_index_in_fused_gates) const -> void
    { throw 1; }
#else // KET_USE_BIT_MASKS_EXPLICITLY
    template <typename Iterator>
    [[noreturn]] auto fused_multi_controlled_pauli_zn<Iterator>::do_call(
      Iterator const first, ::bra::state_integer_type const fused_index_wo_qubits,
      std::array< ::bra::state_integer_type, 0u > const& qubit_masks,
      std::array< ::bra::state_integer_type, 1u > const& index_masks,
      std::vector< ::bra::bit_integer_type > const& to_qubit_index_in_fused_gates) const -> void
    { throw 1; }

    template <typename Iterator>
    [[noreturn]] auto fused_multi_controlled_pauli_zn<Iterator>::do_call(
      Iterator const first, ::bra::state_integer_type const fused_index_wo_qubits,
      std::array< ::bra::state_integer_type, 1u > const& qubit_masks,
      std::array< ::bra::state_integer_type, 2u > const& index_masks,
      std::vector< ::bra::bit_integer_type > const& to_qubit_index_in_fused_gates) const -> void
    { throw 1; }

    template <typename Iterator>
    [[noreturn]] auto fused_multi_controlled_pauli_zn<Iterator>::do_call(
      Iterator const first, ::bra::state_integer_type const fused_index_wo_qubits,
      std::array< ::bra::state_integer_type, 2u > const& qubit_masks,
      std::array< ::bra::state_integer_type, 3u > const& index_masks,
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
#define TARGET_QUBITS(z, n, _) , static_cast< ::bra::qubit_type >(to_qubit_index_in_fused_gates[static_cast< ::bra::bit_integer_type >(target_qubits_[n])])
#define CONTROL_QUBITS(z, n, _) , static_cast< ::bra::control_qubit_type >(to_qubit_index_in_fused_gates[static_cast< ::bra::bit_integer_type >(control_qubits[n].qubit())])
#ifndef KET_USE_BIT_MASKS_EXPLICITLY
# define CALL_PAULI_Z(z, num_control_qubits, num_target_qubits) \
    template <typename Iterator>\
    template <std::size_t num_fused_qubits>\
    auto fused_multi_controlled_pauli_zn<Iterator>::call_pauli_z_ ## num_target_qubits ## _ ## num_control_qubits(\
      Iterator const first, ::bra::state_integer_type const fused_index_wo_qubits,\
      std::array< ::bra::qubit_type, num_fused_qubits > const& unsorted_fused_qubits,\
      std::array< ::bra::qubit_type, num_fused_qubits + 1u > const& sorted_fused_qubits_with_sentinel,\
      std::vector< ::bra::bit_integer_type > const& to_qubit_index_in_fused_gates,\
      std::vector< ::bra::control_qubit_type > const& control_qubits) const -> void\
    {\
      ::ket::gate::fused::pauli_z(\
        first, fused_index_wo_qubits, unsorted_fused_qubits, sorted_fused_qubits_with_sentinel\
        BOOST_PP_REPEAT_ ## z(num_target_qubits, TARGET_QUBITS, nil)\
        BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
    }\

#else // KET_USE_BIT_MASKS_EXPLICITLY
# define CALL_PAULI_Z(z, num_control_qubits, num_target_qubits) \
    template <typename Iterator>\
    template <std::size_t num_fused_qubits>\
    auto fused_multi_controlled_pauli_zn<Iterator>::call_pauli_z_ ## num_target_qubits ## _ ## num_control_qubits(\
      Iterator const first, ::bra::state_integer_type const fused_index_wo_qubits,\
      std::array< ::bra::state_integer_type, num_fused_qubits > const& qubit_masks,\
      std::array< ::bra::state_integer_type, num_fused_qubits + 1u > const& index_masks,\
      std::vector< ::bra::bit_integer_type > const& to_qubit_index_in_fused_gates,\
      std::vector< ::bra::control_qubit_type > const& control_qubits) const -> void\
    {\
      ::ket::gate::fused::pauli_z(\
        first, fused_index_wo_qubits, qubit_masks, index_masks\
        BOOST_PP_REPEAT_ ## z(num_target_qubits, TARGET_QUBITS, nil)\
        BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
    }\

#endif // KET_USE_BIT_MASKS_EXPLICITLY
#define DO_CALL_PAULI_Z(z, num_target_qubits, _) \
BOOST_PP_REPEAT_FROM_TO(0, BOOST_PP_INC(BOOST_PP_SUB(BRA_MAX_NUM_FUSED_QUBITS, num_target_qubits)), CALL_PAULI_Z, num_target_qubits)
BOOST_PP_REPEAT_FROM_TO(1, BOOST_PP_INC(BRA_MAX_NUM_FUSED_QUBITS), DO_CALL_PAULI_Z, nil)
#undef DO_CALL_PAULI_Z
#undef CALL_PAULI_Z
#undef CONTROL_QUBITS
#undef TARGET_QUBITS

#ifndef KET_USE_BIT_MASKS_EXPLICITLY
# define CASE_CN(z, num_control_qubits, num_target_qubits) \
         case num_control_qubits:\
          call_pauli_z_ ## num_target_qubits ## _ ## num_control_qubits(\
            first, fused_index_wo_qubits, unsorted_fused_qubits, sorted_fused_qubits_with_sentinel, to_qubit_index_in_fused_gates, control_qubits);\
          break;\

# define CASE_N(z, num_target_qubits, num_fused_qubits) \
       case num_target_qubits:\
        switch (control_qubits.size())\
        {\
BOOST_PP_REPEAT_FROM_TO_ ## z(0, BOOST_PP_SUB(num_fused_qubits, num_target_qubits), CASE_CN, num_target_qubits)\
        }\
        break;\

# define DO_CALL(z, num_fused_qubits, _) \
    template <typename Iterator>\
    auto fused_multi_controlled_pauli_zn<Iterator>::do_call(\
      Iterator const first, ::bra::state_integer_type const fused_index_wo_qubits,\
      std::array< ::bra::qubit_type, num_fused_qubits > const& unsorted_fused_qubits,\
      std::array< ::bra::qubit_type, num_fused_qubits + 1u > const& sorted_fused_qubits_with_sentinel,\
      std::vector< ::bra::bit_integer_type > const& to_qubit_index_in_fused_gates) const -> void\
    {\
      auto control_qubits = std::vector< ::bra::control_qubit_type >{};\
      auto const num_control_qubits = control_qubits_.size();\
      control_qubits.reserve(num_control_qubits);\
      for (auto index = decltype(num_control_qubits){0}; index < num_control_qubits; ++index)\
        if (static_cast<bool>(is_control_qubit_enabled_vec_[index]))\
          control_qubits.push_back(control_qubits_[index]);\
\
      switch (target_qubits_.size())\
      {\
BOOST_PP_REPEAT_FROM_TO_ ## z(1, BOOST_PP_INC(num_fused_qubits), CASE_N, num_fused_qubits)\
      }\
    }\

#else // KET_USE_BIT_MASKS_EXPLICITLY
# define CASE_CN(z, num_control_qubits, num_target_qubits) \
         case num_control_qubits:\
          call_pauli_z_ ## num_target_qubits ## _ ## num_control_qubits(\
            first, fused_index_wo_qubits, qubit_masks, index_masks, to_qubit_index_in_fused_gates, control_qubits);\
          break;\

# define CASE_N(z, num_target_qubits, num_fused_qubits) \
       case num_target_qubits:\
        switch (control_qubits.size())\
        {\
BOOST_PP_REPEAT_FROM_TO_ ## z(0, BOOST_PP_SUB(num_fused_qubits, num_target_qubits), CASE_CN, num_target_qubits)\
        }\
        break;\

# define DO_CALL(z, num_fused_qubits, _) \
    template <typename Iterator>\
    auto fused_multi_controlled_pauli_zn<Iterator>::do_call(\
      Iterator const first, ::bra::state_integer_type const fused_index_wo_qubits,\
      std::array< ::bra::state_integer_type, num_fused_qubits > const& qubit_masks,\
      std::array< ::bra::state_integer_type, num_fused_qubits + 1u > const& index_masks,\
      std::vector< ::bra::bit_integer_type > const& to_qubit_index_in_fused_gates) const -> void\
    {\
      auto control_qubits = std::vector< ::bra::control_qubit_type >{};\
      auto const num_control_qubits = control_qubits_.size();\
      control_qubits.reserve(num_control_qubits);\
      for (auto index = decltype(num_control_qubits){0}; index < num_control_qubits; ++index)\
        if (static_cast<bool>(is_control_qubit_enabled_vec_[index]))\
          control_qubits.push_back(control_qubits_[index]);\
\
      switch (target_qubits_.size())\
      {\
BOOST_PP_REPEAT_FROM_TO_ ## z(1, BOOST_PP_INC(num_fused_qubits), CASE_N, num_fused_qubits)\
      }\
    }\

#endif // KET_USE_BIT_MASKS_EXPLICITLY
BOOST_PP_REPEAT_FROM_TO(3, BOOST_PP_INC(BRA_MAX_NUM_FUSED_QUBITS), DO_CALL, nil)
#undef DO_CALL
#undef CASE_N
#undef CASE_CN

    template <typename Iterator>
    auto fused_multi_controlled_pauli_zn<Iterator>::do_disable_control_qubits(
      typename std::vector< ::bra::qubit_type >::const_iterator const first,
      typename std::vector< ::bra::qubit_type >::const_iterator const last)
    -> void
    {
      auto const num_control_qubits = control_qubits_.size();
      for (auto index = decltype(num_control_qubits){0}; index < num_control_qubits; ++index)
        is_control_qubit_enabled_vec_[index]
          = static_cast<int>(
              static_cast<bool>(is_control_qubit_enabled_vec_[index])
              and std::none_of(
                    first, last,
                    [this, index](::bra::qubit_type const found_qubit)
                    { return found_qubit == this->control_qubits_[index]; }));
    }

    template <typename Iterator>
    auto fused_multi_controlled_pauli_zn<Iterator>::do_disable_control_qubits(
      typename std::vector< ::bra::control_qubit_type >::const_iterator const first,
      typename std::vector< ::bra::control_qubit_type >::const_iterator const last)
    -> void
    {
      auto const num_control_qubits = control_qubits_.size();
      for (auto index = decltype(num_control_qubits){0}; index < num_control_qubits; ++index)
        is_control_qubit_enabled_vec_[index]
          = static_cast<int>(
              static_cast<bool>(is_control_qubit_enabled_vec_[index])
              and std::none_of(
                    first, last,
                    [this, index](::bra::control_qubit_type const found_control_qubit)
                    { return found_control_qubit == this->control_qubits_[index]; }));
    }

    template class fused_multi_controlled_pauli_zn< ::bra::data_type::iterator >;
#if !defined(BRA_NO_MPI) && (!defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)))
    template class fused_multi_controlled_pauli_zn< ::bra::paged_data_type::iterator >;
#endif // !defined(BRA_NO_MPI) && (!defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)))
#ifndef KET_USE_BIT_MASKS_EXPLICITLY
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    template class fused_multi_controlled_pauli_zn<ket::gate::utility::cache_aware_iterator< ::bra::data_type::iterator, ::bra::qubit_type >>;
#   ifndef BRA_NO_MPI
    template class fused_multi_controlled_pauli_zn<ket::gate::utility::cache_aware_iterator< ::bra::paged_data_type::iterator, ::bra::qubit_type >>;
#   endif // BRA_NO_MPI
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
#else // KET_USE_BIT_MASKS_EXPLICITLY
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    template class fused_multi_controlled_pauli_zn<ket::gate::utility::cache_aware_iterator< ::bra::data_type::iterator, ::bra::state_integer_type >>;
#   ifndef BRA_NO_MPI
    template class fused_multi_controlled_pauli_zn<ket::gate::utility::cache_aware_iterator< ::bra::paged_data_type::iterator, ::bra::state_integer_type >>;
#   endif // BRA_NO_MPI
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
#endif // KET_USE_BIT_MASKS_EXPLICITLY
  } // namespace fused_gate
} // namespace bra
