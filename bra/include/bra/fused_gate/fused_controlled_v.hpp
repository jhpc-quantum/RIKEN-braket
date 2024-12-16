#ifndef BRA_FUSED_GATE_FUSED_CONTROLLED_V_HPP
# define BRA_FUSED_GATE_FUSED_CONTROLLED_V_HPP

# include <array>

# include <boost/preprocessor/arithmetic/dec.hpp>
# include <boost/preprocessor/arithmetic/inc.hpp>
# include <boost/preprocessor/repetition/repeat_from_to.hpp>

# include <bra/types.hpp>
# include <bra/fused_gate/fused_gate.hpp>


namespace bra
{
  namespace fused_gate
  {
    class fused_controlled_v final
      : public ::bra::fused_gate::fused_gate
    {
     private:
      ::bra::complex_type phase_coefficient_;
      ::bra::qubit_type target_qubit_;
      ::bra::control_qubit_type control_qubit_;

     public:
      explicit fused_controlled_v(::bra::complex_type const& phase_coefficient, ::bra::qubit_type const target_qubit, ::bra::control_qubit_type const control_qubit);

      ~fused_controlled_v() = default;
      fused_controlled_v(fused_controlled_v const&) = delete;
      fused_controlled_v& operator=(fused_controlled_v const&) = delete;
      fused_controlled_v(fused_controlled_v&&) = delete;
      fused_controlled_v& operator=(fused_controlled_v&&) = delete;

     private:
# ifndef BRA_MAX_NUM_FUSED_QUBITS
#   ifdef KET_DEFAULT_NUM_ON_CACHE_QUBITS
#     define BRA_MAX_NUM_FUSED_QUBITS BOOST_PP_DEC(KET_DEFAULT_NUM_ON_CACHE_QUBITS)
#   else // KET_DEFAULT_NUM_ON_CACHE_QUBITS
#     define BRA_MAX_NUM_FUSED_QUBITS 10
#   endif // KET_DEFAULT_NUM_ON_CACHE_QUBITS
# endif // BRA_MAX_NUM_FUSED_QUBITS
# ifndef KET_USE_BIT_MASKS_EXPLICITLY
#   define DO_CALL(z, num_fused_qubits, _) \
      auto do_call(\
        ::bra::complex_type* const first, ::bra::state_integer_type const fused_index_wo_qubits,\
        std::array< ::bra::qubit_type, num_fused_qubits > const& unsorted_fused_qubits,\
        std::array< ::bra::qubit_type, num_fused_qubits + 1u > const& sorted_fused_qubits_with_sentinel) const -> void override;
# else // KET_USE_BIT_MASKS_EXPLICITLY
#   define DO_CALL(z, num_fused_qubits, _) \
      auto do_call(\
        ::bra::complex_type* const first, ::bra::state_integer_type const fused_index_wo_qubits,\
        std::array< ::bra::state_integer_type, num_fused_qubits > const& qubit_masks,\
        std::array< ::bra::state_integer_type, num_fused_qubits + 1u > const& index_masks) const -> void override;
# endif // KET_USE_BIT_MASKS_EXPLICITLY
BOOST_PP_REPEAT_FROM_TO(1, BOOST_PP_INC(BRA_MAX_NUM_FUSED_QUBITS), DO_CALL, nil)
# undef DO_CALL
    }; // class fused_controlled_v
  } // namespace fused_gate
} // namespace bra


#endif // BRA_FUSED_GATE_FUSED_CONTROLLED_V_HPP
