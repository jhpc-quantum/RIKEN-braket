#ifndef BRA_FUSED_GATE_FUSED_ADJ_CONTROLLED_EXPONENTIAL_PAULI_Z_HPP
# define BRA_FUSED_GATE_FUSED_ADJ_CONTROLLED_EXPONENTIAL_PAULI_Z_HPP

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
    class fused_adj_controlled_exponential_pauli_z final
      : public ::bra::fused_gate::fused_gate<Iterator>
    {
     private:
      ::bra::real_type phase_;
      ::bra::qubit_type target_qubit_;
      ::bra::control_qubit_type control_qubit_;

      ::bra::fused_gate::cez_qubit_state qubit_state_;
      bool is_control_qubit_enabled_;

     public:
      explicit fused_adj_controlled_exponential_pauli_z(::bra::real_type const phase, ::bra::qubit_type const target_qubit, ::bra::control_qubit_type const control_qubit);

      ~fused_adj_controlled_exponential_pauli_z() = default;
      fused_adj_controlled_exponential_pauli_z(fused_adj_controlled_exponential_pauli_z const&) = delete;
      fused_adj_controlled_exponential_pauli_z& operator=(fused_adj_controlled_exponential_pauli_z const&) = delete;
      fused_adj_controlled_exponential_pauli_z(fused_adj_controlled_exponential_pauli_z&&) = delete;
      fused_adj_controlled_exponential_pauli_z& operator=(fused_adj_controlled_exponential_pauli_z&&) = delete;

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
BOOST_PP_REPEAT_FROM_TO(2, BOOST_PP_INC(BRA_MAX_NUM_FUSED_QUBITS), DO_CALL, nil)
# undef DO_CALL

      auto do_disable_control_qubits(
        typename std::vector< ::bra::qubit_type >::const_iterator const first,
        typename std::vector< ::bra::qubit_type >::const_iterator const last) -> void override;

      auto do_disable_control_qubits(
        typename std::vector< ::bra::control_qubit_type >::const_iterator const first,
        typename std::vector< ::bra::control_qubit_type >::const_iterator const last) -> void override;

      auto do_modify_cez(
        typename std::vector< ::bra::qubit_type >::const_iterator const first,
        typename std::vector< ::bra::qubit_type >::const_iterator const last,
        typename std::vector< ::bra::fused_gate::cez_qubit_state >::const_iterator const cez_qubit_state_first) -> void override;
    }; // class fused_adj_controlled_exponential_pauli_z<Iterator>
  } // namespace fused_gate
} // namespace bra


#endif // BRA_FUSED_GATE_FUSED_ADJ_CONTROLLED_EXPONENTIAL_PAULI_Z_HPP
