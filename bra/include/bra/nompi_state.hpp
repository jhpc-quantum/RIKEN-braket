#ifndef BRA_NOMPI_STATE_HPP
# define BRA_NOMPI_STATE_HPP

# ifdef BRA_NO_MPI
#   include <vector>
#   include <string>
#   include <memory>

#   include <ket/gate/projective_measurement.hpp>
#   if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
#     include <ket/gate/utility/cache_aware_iterator.hpp>
#   endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
#   include <ket/utility/integer_exp2.hpp>
#   include <ket/utility/parallel/loop_n.hpp>

#   include <bra/types.hpp>
#   include <bra/state.hpp>
#   include <bra/fused_gate/fused_gate.hpp>


namespace bra
{
  class nompi_state final
    : public ::bra::state
  {
    friend void inner_product(nompi_state& state1, nompi_state& state2);

    friend void inner_product_all(std::vector<nompi_state>& states);

    friend void inner_product_op(
      nompi_state& state1, nompi_state& state2,
      std::string const& operator_literal_or_variable_name,
      std::vector< ::bra::qubit_type > const& operated_qubits);

    friend void inner_product_all_op(
      std::vector<nompi_state>& states,
      std::string const& operator_literal_or_variable_name,
      std::vector< ::bra::qubit_type > const& operated_qubits);

    friend void fidelity(nompi_state& state1, nompi_state& state2);

    friend void fidelity_all(std::vector<nompi_state>& states);

    friend void fidelity_op(
      nompi_state& state1, nompi_state& state2,
      std::string const& operator_literal_or_variable_name,
      std::vector< ::bra::qubit_type > const& operated_qubits);

    friend void fidelity_all_op(
      std::vector<nompi_state>& states,
      std::string const& operator_literal_or_variable_name,
      std::vector< ::bra::qubit_type > const& operated_qubits);

    ket::utility::policy::parallel<unsigned int> parallel_policy_;

    using data_type = ::bra::data_type;
    data_type data_;
#   if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && defined(KET_USE_ON_CACHE_STATE_VECTOR)
    data_type on_cache_data_;
#   endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && defined(KET_USE_ON_CACHE_STATE_VECTOR)

    using fused_gate_iterator = data_type::iterator;
    std::vector<std::unique_ptr< ::bra::fused_gate::fused_gate<fused_gate_iterator> >> fused_gates_; // related to begin_fusion/end_fusion
#   if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
#     ifndef KET_USE_BIT_MASKS_EXPLICITLY
    using cache_aware_fused_gate_iterator = ket::gate::utility::cache_aware_iterator<fused_gate_iterator, ::bra::qubit_type>;
#     else // KET_USE_BIT_MASKS_EXPLICITLY
    using cache_aware_fused_gate_iterator = ket::gate::utility::cache_aware_iterator<fused_gate_iterator, ::bra::state_integer_type>;
#     endif // KET_USE_BIT_MASKS_EXPLICITLY
    std::vector<std::unique_ptr< ::bra::fused_gate::fused_gate<cache_aware_fused_gate_iterator> >> cache_aware_fused_gates_; // related to begin_fusion/end_fusion
#   endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)

    bool is_waiting_;

   public:
    nompi_state(
      ::bra::state::state_integer_type const initial_integer,
      unsigned int const total_num_qubits,
      unsigned int const num_threads, ::bra::state::seed_type const seed, int const circuit_index);

   private:
    data_type make_initial_data(
      ::bra::state::state_integer_type const initial_integer,
      unsigned int const total_num_qubits)
    {
      auto result
        = data_type(
            ket::utility::integer_exp2<state_integer_type>(total_num_qubits),
            complex_type{real_type{0}});
      result[initial_integer] = complex_type{real_type{1}};
      return result;
    }

   public:
    ~nompi_state() = default;
    nompi_state(nompi_state const&) = default;
    nompi_state& operator=(nompi_state const&) = default;
    nompi_state(nompi_state&&) = default;
    nompi_state& operator=(nompi_state&&) = default;

   private:
    auto do_is_waiting() const -> bool override;
    auto do_cancel_waiting() -> void override;

    void do_i_gate(qubit_type const qubit) override;
    void do_ic_gate(control_qubit_type const control_qubit) override;
    void do_ii_gate(qubit_type const qubit1, qubit_type const qubit2) override;
    void do_in_gate(std::vector<qubit_type> const& qubits) override;
    void do_hadamard(qubit_type const qubit) override;
    void do_not_(qubit_type const qubit) override;
    void do_pauli_x(qubit_type const qubit) override;
    void do_pauli_xx(qubit_type const qubit1, qubit_type const qubit2) override;
    void do_pauli_xn(std::vector<qubit_type> const& qubits) override;
    void do_pauli_y(qubit_type const qubit) override;
    void do_pauli_yy(qubit_type const qubit1, qubit_type const qubit2) override;
    void do_pauli_yn(std::vector<qubit_type> const& qubits) override;
    void do_pauli_z(control_qubit_type const control_qubit) override;
    void do_pauli_zz(qubit_type const qubit1, qubit_type const qubit2) override;
    void do_pauli_zn(std::vector<qubit_type> const& qubits) override;
    void do_swap(qubit_type const qubit1, qubit_type const qubit2) override;
    void do_sqrt_pauli_x(qubit_type const qubit) override;
    void do_adj_sqrt_pauli_x(qubit_type const qubit) override;
    void do_sqrt_pauli_y(qubit_type const qubit) override;
    void do_adj_sqrt_pauli_y(qubit_type const qubit) override;
    void do_sqrt_pauli_z(control_qubit_type const control_qubit) override;
    void do_adj_sqrt_pauli_z(control_qubit_type const control_qubit) override;
    void do_sqrt_pauli_zz(qubit_type const qubit1, qubit_type const qubit2) override;
    void do_adj_sqrt_pauli_zz(qubit_type const qubit1, qubit_type const qubit2) override;
    void do_sqrt_pauli_zn(std::vector<qubit_type> const& qubits) override;
    void do_adj_sqrt_pauli_zn(std::vector<qubit_type> const& qubits) override;
    void do_u1(real_type const phase, control_qubit_type const control_qubit) override;
    void do_adj_u1(real_type const phase, control_qubit_type const control_qubit) override;
    void do_u2(
      real_type const phase1, real_type const phase2,
      qubit_type const qubit) override;
    void do_adj_u2(
      real_type const phase1, real_type const phase2,
      qubit_type const qubit) override;
    void do_u3(
      real_type const phase1, real_type const phase2, real_type const phase3,
      qubit_type const qubit) override;
    void do_adj_u3(
      real_type const phase1, real_type const phase2, real_type const phase3,
      qubit_type const qubit) override;
    void do_phase_shift(
      complex_type const& phase_coefficient, control_qubit_type const control_qubit) override;
    void do_adj_phase_shift(
      complex_type const& phase_coefficient, control_qubit_type const control_qubit) override;
    void do_x_rotation_half_pi(qubit_type const qubit) override;
    void do_adj_x_rotation_half_pi(qubit_type const qubit) override;
    void do_y_rotation_half_pi(qubit_type const qubit) override;
    void do_adj_y_rotation_half_pi(qubit_type const qubit) override;
    void do_exponential_pauli_x(real_type const phase, qubit_type const qubit) override;
    void do_adj_exponential_pauli_x(real_type const phase, qubit_type const qubit) override;
    void do_exponential_pauli_xx(
      real_type const phase, qubit_type const qubit1, qubit_type const qubit2) override;
    void do_adj_exponential_pauli_xx(
      real_type const phase, qubit_type const qubit1, qubit_type const qubit2) override;
    void do_exponential_pauli_xn(
      real_type const phase, std::vector<qubit_type> const& qubits) override;
    void do_adj_exponential_pauli_xn(
      real_type const phase, std::vector<qubit_type> const& qubits) override;
    void do_exponential_pauli_y(real_type const phase, qubit_type const qubit) override;
    void do_adj_exponential_pauli_y(real_type const phase, qubit_type const qubit) override;
    void do_exponential_pauli_yy(
      real_type const phase, qubit_type const qubit1, qubit_type const qubit2) override;
    void do_adj_exponential_pauli_yy(
      real_type const phase, qubit_type const qubit1, qubit_type const qubit2) override;
    void do_exponential_pauli_yn(
      real_type const phase, std::vector<qubit_type> const& qubits) override;
    void do_adj_exponential_pauli_yn(
      real_type const phase, std::vector<qubit_type> const& qubits) override;
    void do_exponential_pauli_z(real_type const phase, qubit_type const qubit) override;
    void do_adj_exponential_pauli_z(real_type const phase, qubit_type const qubit) override;
    void do_exponential_pauli_zz(
      real_type const phase, qubit_type const qubit1, qubit_type const qubit2) override;
    void do_adj_exponential_pauli_zz(
      real_type const phase, qubit_type const qubit1, qubit_type const qubit2) override;
    void do_exponential_pauli_zn(
      real_type const phase, std::vector<qubit_type> const& qubits) override;
    void do_adj_exponential_pauli_zn(
      real_type const phase, std::vector<qubit_type> const& qubits) override;
    void do_exponential_swap(
      real_type const phase, qubit_type const qubit1, qubit_type const qubit2) override;
    void do_adj_exponential_swap(
      real_type const phase, qubit_type const qubit1, qubit_type const qubit2) override;
    void do_toffoli(
      qubit_type const target_qubit,
      control_qubit_type const control_qubit1,
      control_qubit_type const control_qubit2) override;

    ket::gate::outcome do_projective_measurement(qubit_type const qubit) override;
    void do_expectation_values() override;
    void do_amplitudes() override;
    void do_measure() override;
    void do_generate_events(int const num_events, int const seed) override;
    void do_expectation_value(std::string const& operator_literal_or_variable_name, std::vector<qubit_type> const& operated_qubits) override;
    void do_inner_product(std::string const& circuit_index_or_all) override;
    void do_inner_product(std::string const& remote_circuit_index_or_all, std::string const& operator_literal_or_variable_name, std::vector<qubit_type> const& operated_qubits) override;
    void do_fidelity(std::string const& remote_circuit_index_or_all) override;
    void do_fidelity(std::string const& remote_circuit_index_or_all, std::string const& operator_literal_or_variable_name, std::vector<qubit_type> const& operated_qubits) override;
    void do_shor_box(
      state_integer_type const divisor, state_integer_type const base,
      std::vector<qubit_type> const& exponent_qubits,
      std::vector<qubit_type> const& modular_exponentiation_qubits) override;
    void do_begin_fusion() override;
    void do_end_fusion() override;
    void do_clear(qubit_type const qubit) override;
    void do_set(qubit_type const qubit) override;

    void do_controlled_i_gate(
      qubit_type const target_qubit, control_qubit_type const control_qubit) override;
    void do_controlled_ic_gate(
      control_qubit_type const control_qubit1, control_qubit_type const control_qubit2) override;
    void do_multi_controlled_in_gate(
      std::vector<qubit_type> const& target_qubits,
      std::vector<control_qubit_type> const& control_qubits) override;
    void do_multi_controlled_ic_gate(
      std::vector<control_qubit_type> const& control_qubits) override;
    void do_controlled_hadamard(
      qubit_type const target_qubit, control_qubit_type const control_qubit) override;
    void do_multi_controlled_hadamard(
      qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits) override;
    void do_controlled_not(
      qubit_type const target_qubit, control_qubit_type const control_qubit) override;
    void do_multi_controlled_not(
      qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits) override;
    void do_controlled_pauli_x(
      qubit_type const target_qubit, control_qubit_type const control_qubit) override;
    void do_multi_controlled_pauli_xn(
      std::vector<qubit_type> const& target_qubits,
      std::vector<control_qubit_type> const& control_qubits) override;
    void do_controlled_pauli_y(
      qubit_type const target_qubit, control_qubit_type const control_qubit) override;
    void do_multi_controlled_pauli_yn(
      std::vector<qubit_type> const& target_qubits,
      std::vector<control_qubit_type> const& control_qubits) override;
    void do_controlled_pauli_z(
      control_qubit_type const control_qubit1, control_qubit_type const control_qubit2) override;
    void do_multi_controlled_pauli_z(std::vector<control_qubit_type> const& control_qubits) override;
    void do_multi_controlled_pauli_zn(
      std::vector<qubit_type> const& target_qubits,
      std::vector<control_qubit_type> const& control_qubits) override;
    void do_multi_controlled_swap(
      qubit_type const target_qubit1, qubit_type const target_qubit2,
      std::vector<control_qubit_type> const& control_qubits) override;
    void do_controlled_sqrt_pauli_x(
      qubit_type const target_qubit, control_qubit_type const control_qubit) override;
    void do_adj_controlled_sqrt_pauli_x(
      qubit_type const target_qubit, control_qubit_type const control_qubit) override;
    void do_multi_controlled_sqrt_pauli_x(
      qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits) override;
    void do_adj_multi_controlled_sqrt_pauli_x(
      qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits) override;
    void do_controlled_sqrt_pauli_y(
      qubit_type const target_qubit, control_qubit_type const control_qubit) override;
    void do_adj_controlled_sqrt_pauli_y(
      qubit_type const target_qubit, control_qubit_type const control_qubit) override;
    void do_multi_controlled_sqrt_pauli_y(
      qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits) override;
    void do_adj_multi_controlled_sqrt_pauli_y(
      qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits) override;
    void do_controlled_sqrt_pauli_z(
      control_qubit_type const control_qubit1, control_qubit_type const control_qubit2) override;
    void do_adj_controlled_sqrt_pauli_z(
      control_qubit_type const control_qubit1, control_qubit_type const control_qubit2) override;
    void do_multi_controlled_sqrt_pauli_z(std::vector<control_qubit_type> const& control_qubits) override;
    void do_adj_multi_controlled_sqrt_pauli_z(std::vector<control_qubit_type> const& control_qubits) override;
    void do_multi_controlled_sqrt_pauli_zn(
      std::vector<qubit_type> const& target_qubits,
      std::vector<control_qubit_type> const& control_qubits) override;
    void do_adj_multi_controlled_sqrt_pauli_zn(
      std::vector<qubit_type> const& target_qubits,
      std::vector<control_qubit_type> const& control_qubits) override;
    void do_controlled_phase_shift(
      complex_type const& phase_coefficient,
      control_qubit_type const control_qubit1, control_qubit_type const control_qubit2) override;
    void do_adj_controlled_phase_shift(
      complex_type const& phase_coefficient,
      control_qubit_type const control_qubit1, control_qubit_type const control_qubit2) override;
    void do_multi_controlled_phase_shift(
      complex_type const& phase_coefficient,
      std::vector<control_qubit_type> const& control_qubit) override;
    void do_adj_multi_controlled_phase_shift(
      complex_type const& phase_coefficient,
      std::vector<control_qubit_type> const& control_qubit) override;
    void do_controlled_u1(
      real_type const phase, control_qubit_type const control_qubit1, control_qubit_type const control_qubit2) override;
    void do_adj_controlled_u1(
      real_type const phase, control_qubit_type const control_qubit1, control_qubit_type const control_qubit2) override;
    void do_multi_controlled_u1(
      real_type const phase, std::vector<control_qubit_type> const& control_qubits) override;
    void do_adj_multi_controlled_u1(
      real_type const phase, std::vector<control_qubit_type> const& control_qubits) override;
    void do_controlled_u2(
      real_type const phase1, real_type const phase2, qubit_type const target_qubit, control_qubit_type const control_qubit) override;
    void do_adj_controlled_u2(
      real_type const phase1, real_type const phase2, qubit_type const target_qubit, control_qubit_type const control_qubit) override;
    void do_multi_controlled_u2(
      real_type const phase1, real_type const phase2, qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits) override;
    void do_adj_multi_controlled_u2(
      real_type const phase1, real_type const phase2, qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits) override;
    void do_controlled_u3(
      real_type const phase1, real_type const phase2, real_type const phase3,
      qubit_type const target_qubit, control_qubit_type const control_qubit) override;
    void do_adj_controlled_u3(
      real_type const phase1, real_type const phase2, real_type const phase3,
      qubit_type const target_qubit, control_qubit_type const control_qubit) override;
    void do_multi_controlled_u3(
      real_type const phase1, real_type const phase2, real_type const phase3,
      qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits) override;
    void do_adj_multi_controlled_u3(
      real_type const phase1, real_type const phase2, real_type const phase3,
      qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits) override;
    void do_controlled_x_rotation_half_pi(
      qubit_type const target_qubit, control_qubit_type const control_qubit) override;
    void do_adj_controlled_x_rotation_half_pi(
      qubit_type const target_qubit, control_qubit_type const control_qubit) override;
    void do_multi_controlled_x_rotation_half_pi(
      qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits) override;
    void do_adj_multi_controlled_x_rotation_half_pi(
      qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits) override;
    void do_controlled_y_rotation_half_pi(
      qubit_type const target_qubit, control_qubit_type const control_qubit) override;
    void do_adj_controlled_y_rotation_half_pi(
      qubit_type const target_qubit, control_qubit_type const control_qubit) override;
    void do_multi_controlled_y_rotation_half_pi(
      qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits) override;
    void do_adj_multi_controlled_y_rotation_half_pi(
      qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits) override;
    void do_controlled_exponential_pauli_x(
      real_type const phase, qubit_type const target_qubit, control_qubit_type const control_qubit) override;
    void do_adj_controlled_exponential_pauli_x(
      real_type const phase, qubit_type const target_qubit, control_qubit_type const control_qubit) override;
    void do_multi_controlled_exponential_pauli_xn(
      real_type const phase, std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits) override;
    void do_adj_multi_controlled_exponential_pauli_xn(
      real_type const phase, std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits) override;
    void do_controlled_exponential_pauli_y(
      real_type const phase, qubit_type const target_qubit, control_qubit_type const control_qubit) override;
    void do_adj_controlled_exponential_pauli_y(
      real_type const phase, qubit_type const target_qubit, control_qubit_type const control_qubit) override;
    void do_multi_controlled_exponential_pauli_yn(
      real_type const phase, std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits) override;
    void do_adj_multi_controlled_exponential_pauli_yn(
      real_type const phase, std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits) override;
    void do_controlled_exponential_pauli_z(
      real_type const phase, qubit_type const target_qubit, control_qubit_type const control_qubit) override;
    void do_adj_controlled_exponential_pauli_z(
      real_type const phase, qubit_type const target_qubit, control_qubit_type const control_qubit) override;
    void do_multi_controlled_exponential_pauli_z(
      real_type const phase, qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits) override;
    void do_adj_multi_controlled_exponential_pauli_z(
      real_type const phase, qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits) override;
    void do_multi_controlled_exponential_pauli_zn(
      real_type const phase, std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits) override;
    void do_adj_multi_controlled_exponential_pauli_zn(
      real_type const phase, std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits) override;
    void do_multi_controlled_exponential_swap(
      real_type const phase, qubit_type const target_qubit1, qubit_type const target_qubit2,
      std::vector<control_qubit_type> const& control_qubits) override;
    void do_adj_multi_controlled_exponential_swap(
      real_type const phase, qubit_type const target_qubit1, qubit_type const target_qubit2,
      std::vector<control_qubit_type> const& control_qubits) override;
  }; // class nompi_state

  void inner_product(nompi_state& state1, nompi_state& state2);

  void inner_product_all(std::vector< ::bra::nompi_state >& states);

  void inner_product_op(
    nompi_state& state1, nompi_state& state2,
    std::string const& operator_literal_or_variable_name,
    std::vector< ::bra::qubit_type > const& operated_qubits);

  void inner_product_all_op(
    std::vector< ::bra::nompi_state >& states,
    std::string const& operator_literal_or_variable_name,
    std::vector< ::bra::qubit_type > const& operated_qubits);

  void fidelity(nompi_state& state1, nompi_state& state2);

  void fidelity_all(std::vector< ::bra::nompi_state >& states);

  void fidelity_op(
    nompi_state& state1, nompi_state& state2,
    std::string const& operator_literal_or_variable_name,
    std::vector< ::bra::qubit_type > const& operated_qubits);

  void fidelity_all_op(
    std::vector< ::bra::nompi_state >& states,
    std::string const& operator_literal_or_variable_name,
    std::vector< ::bra::qubit_type > const& operated_qubits);
} // namespace bra


# endif // BRA_NO_MPI

#endif // BRA_NOMPI_STATE_HPP
