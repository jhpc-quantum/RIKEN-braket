#ifndef BRA_FUSED_GATE_FUSED_EXPONENTIAL_SWAP_HPP
# define BRA_FUSED_GATE_FUSED_EXPONENTIAL_SWAP_HPP

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
    class fused_exponential_swap final
      : public ::bra::fused_gate::fused_gate
    {
     private:
      ::bra::real_type phase_;
      ::bra::qubit_type qubit1_;
      ::bra::qubit_type qubit2_;

     public:
      explicit fused_exponential_swap(::bra::real_type const phase, ::bra::qubit_type const qubit1, ::bra::qubit_type const qubit2);

      ~fused_exponential_swap() = default;
      fused_exponential_swap(fused_exponential_swap const&) = delete;
      fused_exponential_swap& operator=(fused_exponential_swap const&) = delete;
      fused_exponential_swap(fused_exponential_swap&&) = delete;
      fused_exponential_swap& operator=(fused_exponential_swap&&) = delete;

     private:
# ifndef KET_USE_BIT_MASKS_EXPLICITLY
      [[noreturn]] auto do_call(
        ::bra::complex_type* const first, ::bra::state_integer_type const fused_index_wo_qubits,
        std::array< ::bra::qubit_type, 1u > const& unsorted_fused_qubits,
        std::array< ::bra::qubit_type, 2u > const& sorted_fused_qubits_with_sentinel) const -> void override;
# else // KET_USE_BIT_MASKS_EXPLICITLY
      [[noreturn]] auto do_call(
        ::bra::complex_type* const first, ::bra::state_integer_type const fused_index_wo_qubits,
        std::array< ::bra::state_integer_type, 1u > const& qubit_masks,
        std::array< ::bra::state_integer_type, 2u > const& index_masks) const -> void override;
# endif // KET_USE_BIT_MASKS_EXPLICITLY

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
BOOST_PP_REPEAT_FROM_TO(2, BOOST_PP_INC(BRA_MAX_NUM_FUSED_QUBITS), DO_CALL, nil)
# undef DO_CALL
    }; // class fused_exponential_swap
  } // namespace fused_gate
} // namespace bra


#endif // BRA_FUSED_GATE_FUSED_EXPONENTIAL_SWAP_HPP
