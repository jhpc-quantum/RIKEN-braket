#ifndef BRA_FUSED_GATE_FUSED_ADJ_MULTI_CONTROLLED_EXPONENTIAL_PAULI_XN_HPP
# define BRA_FUSED_GATE_FUSED_ADJ_MULTI_CONTROLLED_EXPONENTIAL_PAULI_XN_HPP

# include <array>
# include <vector>

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
    class fused_adj_multi_controlled_exponential_pauli_xn final
      : public ::bra::fused_gate::fused_gate<Iterator>
    {
     private:
      ::bra::real_type phase_;
      std::vector< ::bra::qubit_type > target_qubits_;
      std::vector< ::bra::control_qubit_type > control_qubits_;

      std::vector<int> is_control_qubit_enabled_vec_;

     public:
      explicit fused_adj_multi_controlled_exponential_pauli_xn(::bra::real_type const phase, std::vector< ::bra::qubit_type > const& target_qubits, std::vector< ::bra::control_qubit_type > const& control_qubits);
      explicit fused_adj_multi_controlled_exponential_pauli_xn(::bra::real_type const phase, std::vector< ::bra::qubit_type >&& target_qubits, std::vector< ::bra::control_qubit_type > const& control_qubits);
      explicit fused_adj_multi_controlled_exponential_pauli_xn(::bra::real_type const phase, std::vector< ::bra::qubit_type > const& target_qubits, std::vector< ::bra::control_qubit_type >&& control_qubits);
      explicit fused_adj_multi_controlled_exponential_pauli_xn(::bra::real_type const phase, std::vector< ::bra::qubit_type >&& target_qubits, std::vector< ::bra::control_qubit_type >&& control_qubits);

      ~fused_adj_multi_controlled_exponential_pauli_xn() = default;
      fused_adj_multi_controlled_exponential_pauli_xn(fused_adj_multi_controlled_exponential_pauli_xn const&) = delete;
      fused_adj_multi_controlled_exponential_pauli_xn& operator=(fused_adj_multi_controlled_exponential_pauli_xn const&) = delete;
      fused_adj_multi_controlled_exponential_pauli_xn(fused_adj_multi_controlled_exponential_pauli_xn&&) = delete;
      fused_adj_multi_controlled_exponential_pauli_xn& operator=(fused_adj_multi_controlled_exponential_pauli_xn&&) = delete;

     private:
# ifndef KET_USE_BIT_MASKS_EXPLICITLY
      [[noreturn]] auto do_call(
        Iterator const first, ::bra::state_integer_type const fused_index_wo_qubits,
        std::array< ::bra::qubit_type, 0u > const& unsorted_fused_qubits,
        std::array< ::bra::qubit_type, 1u > const& sorted_fused_qubits_with_sentinel,
        std::vector< ::bra::bit_integer_type > const& to_qubit_index_in_fused_gates) const -> void override;

      [[noreturn]] auto do_call(
        Iterator const first, ::bra::state_integer_type const fused_index_wo_qubits,
        std::array< ::bra::qubit_type, 1u > const& unsorted_fused_qubits,
        std::array< ::bra::qubit_type, 2u > const& sorted_fused_qubits_with_sentinel,
        std::vector< ::bra::bit_integer_type > const& to_qubit_index_in_fused_gates) const -> void override;

      [[noreturn]] auto do_call(
        Iterator const first, ::bra::state_integer_type const fused_index_wo_qubits,
        std::array< ::bra::qubit_type, 2u > const& unsorted_fused_qubits,
        std::array< ::bra::qubit_type, 3u > const& sorted_fused_qubits_with_sentinel,
        std::vector< ::bra::bit_integer_type > const& to_qubit_index_in_fused_gates) const -> void override;
# else // KET_USE_BIT_MASKS_EXPLICITLY
      [[noreturn]] auto do_call(
        Iterator const first, ::bra::state_integer_type const fused_index_wo_qubits,
        std::array< ::bra::state_integer_type, 0u > const& qubit_masks,
        std::array< ::bra::state_integer_type, 1u > const& index_masks,
        std::vector< ::bra::bit_integer_type > const& to_qubit_index_in_fused_gates) const -> void override;

      [[noreturn]] auto do_call(
        Iterator const first, ::bra::state_integer_type const fused_index_wo_qubits,
        std::array< ::bra::state_integer_type, 1u > const& qubit_masks,
        std::array< ::bra::state_integer_type, 2u > const& index_masks,
        std::vector< ::bra::bit_integer_type > const& to_qubit_index_in_fused_gates) const -> void override;

      [[noreturn]] auto do_call(
        Iterator const first, ::bra::state_integer_type const fused_index_wo_qubits,
        std::array< ::bra::state_integer_type, 2u > const& qubit_masks,
        std::array< ::bra::state_integer_type, 3u > const& index_masks,
        std::vector< ::bra::bit_integer_type > const& to_qubit_index_in_fused_gates) const -> void override;
# endif // KET_USE_BIT_MASKS_EXPLICITLY

# ifndef BRA_MAX_NUM_FUSED_QUBITS
#   ifdef KET_DEFAULT_NUM_ON_CACHE_QUBITS
#     define BRA_MAX_NUM_FUSED_QUBITS BOOST_PP_DEC(KET_DEFAULT_NUM_ON_CACHE_QUBITS)
#   else // KET_DEFAULT_NUM_ON_CACHE_QUBITS
#     define BRA_MAX_NUM_FUSED_QUBITS 10
#   endif // KET_DEFAULT_NUM_ON_CACHE_QUBITS
# endif // BRA_MAX_NUM_FUSED_QUBITS
# ifndef KET_USE_BIT_MASKS_EXPLICITLY
#   define CALL_ADJ_EXPONENTIAL_PAULI_X(z, num_control_qubits, num_target_qubits) \
    template <std::size_t num_fused_qubits>\
    auto call_adj_exponential_pauli_x_ ## num_target_qubits ## _ ## num_control_qubits(\
      Iterator const first, ::bra::state_integer_type const fused_index_wo_qubits,\
      std::array< ::bra::qubit_type, num_fused_qubits > const& unsorted_fused_qubits,\
      std::array< ::bra::qubit_type, num_fused_qubits + 1u > const& sorted_fused_qubits_with_sentinel,\
      std::vector< ::bra::bit_integer_type > const& to_qubit_index_in_fused_gates,\
      std::vector< ::bra::control_qubit_type > const& control_qubits) const -> void;
# else // KET_USE_BIT_MASKS_EXPLICITLY
#   define CALL_ADJ_EXPONENTIAL_PAULI_X(z, num_control_qubits, num_target_qubits) \
    template <std::size_t num_fused_qubits>\
    auto call_adj_exponential_pauli_x_ ## num_target_qubits ## _ ## num_control_qubits(\
      Iterator const first, ::bra::state_integer_type const fused_index_wo_qubits,\
      std::array< ::bra::state_integer_type, num_fused_qubits > const& qubit_masks,\
      std::array< ::bra::state_integer_type, num_fused_qubits + 1u > const& index_masks,\
      std::vector< ::bra::bit_integer_type > const& to_qubit_index_in_fused_gates,\
      std::vector< ::bra::control_qubit_type > const& control_qubits) const -> void;
# endif // KET_USE_BIT_MASKS_EXPLICITLY
# define DO_CALL_ADJ_EXPONENTIAL_PAULI_X(z, num_target_qubits, _) \
BOOST_PP_REPEAT_FROM_TO(0, BOOST_PP_INC(BOOST_PP_SUB(BRA_MAX_NUM_FUSED_QUBITS, num_target_qubits)), CALL_ADJ_EXPONENTIAL_PAULI_X, num_target_qubits)
BOOST_PP_REPEAT_FROM_TO(1, BOOST_PP_INC(BRA_MAX_NUM_FUSED_QUBITS), DO_CALL_ADJ_EXPONENTIAL_PAULI_X, nil)
# undef DO_CALL_ADJ_EXPONENTIAL_PAULI_X
# undef CALL_ADJ_EXPONENTIAL_PAULI_X

# ifndef KET_USE_BIT_MASKS_EXPLICITLY
#   define DO_CALL(z, num_fused_qubits, _) \
      auto do_call(\
        Iterator const first, ::bra::state_integer_type const fused_index_wo_qubits,\
        std::array< ::bra::qubit_type, num_fused_qubits > const& unsorted_fused_qubits,\
        std::array< ::bra::qubit_type, num_fused_qubits + 1u > const& sorted_fused_qubits_with_sentinel,\
        std::vector< ::bra::bit_integer_type > const& to_qubit_index_in_fused_gates) const -> void override;
# else // KET_USE_BIT_MASKS_EXPLICITLY
#   define DO_CALL(z, num_fused_qubits, _) \
      auto do_call(\
        Iterator const first, ::bra::state_integer_type const fused_index_wo_qubits,\
        std::array< ::bra::state_integer_type, num_fused_qubits > const& qubit_masks,\
        std::array< ::bra::state_integer_type, num_fused_qubits + 1u > const& index_masks,\
        std::vector< ::bra::bit_integer_type > const& to_qubit_index_in_fused_gates) const -> void override;
# endif // KET_USE_BIT_MASKS_EXPLICITLY
BOOST_PP_REPEAT_FROM_TO(3, BOOST_PP_INC(BRA_MAX_NUM_FUSED_QUBITS), DO_CALL, nil)
# undef DO_CALL

      auto do_disable_control_qubits(
        typename std::vector< ::bra::qubit_type >::const_iterator const first,
        typename std::vector< ::bra::qubit_type >::const_iterator const last) -> void override;

      auto do_disable_control_qubits(
        typename std::vector< ::bra::control_qubit_type >::const_iterator const first,
        typename std::vector< ::bra::control_qubit_type >::const_iterator const last) -> void override;
    }; // class fused_adj_multi_controlled_exponential_pauli_xn<Iterator>
  } // namespace fused_gate
} // namespace bra


#endif // BRA_FUSED_GATE_FUSED_ADJ_MULTI_CONTROLLED_EXPONENTIAL_PAULI_XN_HPP
