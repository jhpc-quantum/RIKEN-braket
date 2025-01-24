#ifndef BRA_FUSED_GATE_FUSED_ADJ_U3_HPP
# define BRA_FUSED_GATE_FUSED_ADJ_U3_HPP

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
    class fused_adj_u3 final
      : public ::bra::fused_gate::fused_gate
    {
     private:
      ::bra::real_type phase1_;
      ::bra::real_type phase2_;
      ::bra::real_type phase3_;
      ::bra::qubit_type qubit_;

     public:
      explicit fused_adj_u3(::bra::real_type const phase1, ::bra::real_type const phase2, ::bra::real_type const phase3, ::bra::qubit_type const qubit);

      ~fused_adj_u3() = default;
      fused_adj_u3(fused_adj_u3 const&) = delete;
      fused_adj_u3& operator=(fused_adj_u3 const&) = delete;
      fused_adj_u3(fused_adj_u3&&) = delete;
      fused_adj_u3& operator=(fused_adj_u3&&) = delete;

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
    }; // class fused_adj_u3
  } // namespace fused_gate
} // namespace bra


#endif // BRA_FUSED_GATE_FUSED_ADJ_U3_HPP
