#ifndef BRA_FUSED_GATE_FUSED_MULTI_CONTROLLED_SWAP_HPP
# define BRA_FUSED_GATE_FUSED_MULTI_CONTROLLED_SWAP_HPP

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
    template <typename Iterator>
    class fused_multi_controlled_swap final
      : public ::bra::fused_gate::fused_gate<Iterator>
    {
     private:
      ::bra::qubit_type target_qubit1_;
      ::bra::qubit_type target_qubit2_;
      std::vector< ::bra::control_qubit_type > control_qubits_;

     public:
      explicit fused_multi_controlled_swap(::bra::qubit_type const target_qubit1, ::bra::qubit_type const target_qubit2, std::vector< ::bra::control_qubit_type > const& control_qubits);
      explicit fused_multi_controlled_swap(::bra::qubit_type const target_qubit1, ::bra::qubit_type const target_qubit2, std::vector< ::bra::control_qubit_type >&& control_qubits);

      ~fused_multi_controlled_swap() = default;
      fused_multi_controlled_swap(fused_multi_controlled_swap const&) = delete;
      fused_multi_controlled_swap& operator=(fused_multi_controlled_swap const&) = delete;
      fused_multi_controlled_swap(fused_multi_controlled_swap&&) = delete;
      fused_multi_controlled_swap& operator=(fused_multi_controlled_swap&&) = delete;

     private:
# ifndef KET_USE_BIT_MASKS_EXPLICITLY
      [[noreturn]] auto do_call(
        Iterator const first, ::bra::state_integer_type const fused_index_wo_qubits,
        std::array< ::bra::qubit_type, 1u > const& unsorted_fused_qubits,
        std::array< ::bra::qubit_type, 2u > const& sorted_fused_qubits_with_sentinel) const -> void override;

      [[noreturn]] auto do_call(
        Iterator const first, ::bra::state_integer_type const fused_index_wo_qubits,
        std::array< ::bra::qubit_type, 2u > const& unsorted_fused_qubits,
        std::array< ::bra::qubit_type, 3u > const& sorted_fused_qubits_with_sentinel) const -> void override;

      [[noreturn]] auto do_call(
        Iterator const first, ::bra::state_integer_type const fused_index_wo_qubits,
        std::array< ::bra::qubit_type, 3u > const& unsorted_fused_qubits,
        std::array< ::bra::qubit_type, 4u > const& sorted_fused_qubits_with_sentinel) const -> void override;
# else // KET_USE_BIT_MASKS_EXPLICITLY
      [[noreturn]] auto do_call(
        Iterator const first, ::bra::state_integer_type const fused_index_wo_qubits,
        std::array< ::bra::state_integer_type, 1u > const& qubit_masks,
        std::array< ::bra::state_integer_type, 2u > const& index_masks) const -> void override;

      [[noreturn]] auto do_call(
        Iterator const first, ::bra::state_integer_type const fused_index_wo_qubits,
        std::array< ::bra::state_integer_type, 2u > const& qubit_masks,
        std::array< ::bra::state_integer_type, 3u > const& index_masks) const -> void override;

      [[noreturn]] auto do_call(
        Iterator const first, ::bra::state_integer_type const fused_index_wo_qubits,
        std::array< ::bra::state_integer_type, 3u > const& qubit_masks,
        std::array< ::bra::state_integer_type, 4u > const& index_masks) const -> void override;
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
        Iterator const first, ::bra::state_integer_type const fused_index_wo_qubits,\
        std::array< ::bra::qubit_type, num_fused_qubits > const& unsorted_fused_qubits,\
        std::array< ::bra::qubit_type, num_fused_qubits + 1u > const& sorted_fused_qubits_with_sentinel) const -> void override;
# else // KET_USE_BIT_MASKS_EXPLICITLY
#   define DO_CALL(z, num_fused_qubits, _) \
      auto do_call(\
        Iterator const first, ::bra::state_integer_type const fused_index_wo_qubits,\
        std::array< ::bra::state_integer_type, num_fused_qubits > const& qubit_masks,\
        std::array< ::bra::state_integer_type, num_fused_qubits + 1u > const& index_masks) const -> void override;
# endif // KET_USE_BIT_MASKS_EXPLICITLY
BOOST_PP_REPEAT_FROM_TO(4, BOOST_PP_INC(BRA_MAX_NUM_FUSED_QUBITS), DO_CALL, nil)
# undef DO_CALL
    }; // class fused_multi_controlled_swap<Iterator>
  } // namespace fused_gate
} // namespace bra


#endif // BRA_FUSED_GATE_FUSED_MULTI_CONTROLLED_SWAP_HPP
