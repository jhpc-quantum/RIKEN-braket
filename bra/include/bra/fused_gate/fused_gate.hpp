#ifndef BRA_FUSED_GATE_FUSED_GATE_HPP
# define BRA_FUSED_GATE_FUSED_GATE_HPP

# include <array>

# include <boost/preprocessor/arithmetic/dec.hpp>
# include <boost/preprocessor/arithmetic/inc.hpp>
# include <boost/preprocessor/repetition/repeat_from_to.hpp>

# include <bra/types.hpp>


namespace bra
{
  namespace fused_gate
  {
    class fused_gate
    {
     public:
      fused_gate() = default;
      virtual ~fused_gate() = default;

      fused_gate(fused_gate const&) = delete;
      fused_gate& operator=(fused_gate const&) = delete;
      fused_gate(fused_gate&&) = delete;
      fused_gate& operator=(fused_gate&&) = delete;

# ifndef BRA_MAX_NUM_FUSED_QUBITS
#   ifdef KET_DEFAULT_NUM_ON_CACHE_QUBITS
#     define BRA_MAX_NUM_FUSED_QUBITS BOOST_PP_DEC(KET_DEFAULT_NUM_ON_CACHE_QUBITS)
#   else // KET_DEFAULT_NUM_ON_CACHE_QUBITS
#     define BRA_MAX_NUM_FUSED_QUBITS 10
#   endif // KET_DEFAULT_NUM_ON_CACHE_QUBITS
# endif // BRA_MAX_NUM_FUSED_QUBITS
# ifndef KET_USE_BIT_MASKS_EXPLICITLY
#   define CALL(z, num_fused_qubits, _) \
      auto call(\
        ::bra::complex_type* const first, ::bra::state_integer_type const fused_index_wo_qubits,\
        std::array< ::bra::qubit_type, num_fused_qubits > const& unsorted_fused_qubits,\
        std::array< ::bra::qubit_type, num_fused_qubits + 1u > const& sorted_fused_qubits_with_sentinel) const -> void\
      { do_call(first, fused_index_wo_qubits, unsorted_fused_qubits, sorted_fused_qubits_with_sentinel); }
# else // KET_USE_BIT_MASKS_EXPLICITLY
#   define CALL(z, num_fused_qubits, _) \
      auto call(\
        ::bra::complex_type* const first, ::bra::state_integer_type const fused_index_wo_qubits,\
        std::array< ::bra::state_integer_type, num_fused_qubits > const& qubit_masks,\
        std::array< ::bra::state_integer_type, num_fused_qubits + 1u > const& index_masks) const -> void\
      { do_call(first, fused_index_wo_qubits, qubit_masks, index_masks); }
# endif // KET_USE_BIT_MASKS_EXPLICITLY
BOOST_PP_REPEAT_FROM_TO(1, BOOST_PP_INC(BRA_MAX_NUM_FUSED_QUBITS), CALL, nil)
# undef CALL

     protected:
# ifndef KET_USE_BIT_MASKS_EXPLICITLY
#   define DO_CALL(z, num_fused_qubits, _) \
      virtual auto do_call(\
        ::bra::complex_type* const first, ::bra::state_integer_type const fused_index_wo_qubits,\
        std::array< ::bra::qubit_type, num_fused_qubits > const& unsorted_fused_qubits,\
        std::array< ::bra::qubit_type, num_fused_qubits + 1u > const& sorted_fused_qubits_with_sentinel) const -> void = 0;
# else // KET_USE_BIT_MASKS_EXPLICITLY
#   define DO_CALL(z, num_fused_qubits, _) \
      virtual auto do_call(\
        ::bra::complex_type* const first, ::bra::state_integer_type const fused_index_wo_qubits,\
        std::array< ::bra::state_integer_type, num_fused_qubits > const& qubit_masks,\
        std::array< ::bra::state_integer_type, num_fused_qubits + 1u > const& index_masks) const -> void = 0;
# endif // KET_USE_BIT_MASKS_EXPLICITLY
BOOST_PP_REPEAT_FROM_TO(1, BOOST_PP_INC(BRA_MAX_NUM_FUSED_QUBITS), DO_CALL, nil)
# undef DO_CALL
    }; // class fused_gate
  } // namespace fused_gate
} // namespace bra


#endif // BRA_FUSED_GATE_FUSED_GATE_HPP
