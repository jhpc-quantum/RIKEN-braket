#ifndef BRA_GATE_ADJ_MULTI_CONTROLLED_SQRT_PAULI_X_HPP
# define BRA_GATE_ADJ_MULTI_CONTROLLED_SQRT_PAULI_X_HPP

# include <string>
# include <vector>
# include <iosfwd>

# include <bra/gate/gate.hpp>
# include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    class adj_multi_controlled_sqrt_pauli_x final
      : public ::bra::gate::gate
    {
     public:
      using qubit_type = ::bra::state::qubit_type;
      using control_qubit_type = ::bra::state::control_qubit_type;

     private:
      qubit_type target_qubit_;
      std::vector<control_qubit_type> control_qubits_;

      std::string name_;

     public:
      adj_multi_controlled_sqrt_pauli_x(
        qubit_type const target_qubit,
        std::vector<control_qubit_type> const& control_qubits);

      adj_multi_controlled_sqrt_pauli_x(
        qubit_type const target_qubit,
        std::vector<control_qubit_type>&& control_qubits);

      ~adj_multi_controlled_sqrt_pauli_x() = default;
      adj_multi_controlled_sqrt_pauli_x(adj_multi_controlled_sqrt_pauli_x const&) = delete;
      adj_multi_controlled_sqrt_pauli_x& operator=(adj_multi_controlled_sqrt_pauli_x const&) = delete;
      adj_multi_controlled_sqrt_pauli_x(adj_multi_controlled_sqrt_pauli_x&&) = delete;
      adj_multi_controlled_sqrt_pauli_x& operator=(adj_multi_controlled_sqrt_pauli_x&&) = delete;

     private:
      ::bra::state& do_apply(::bra::state& state) const override;
      std::string const& do_name() const override;
      std::string do_representation(
        std::ostringstream& repr_stream, int const parameter_width) const override;
    }; // class adj_multi_controlled_sqrt_pauli_x
  } // namespace gate
} // namespace bra


#endif // BRA_GATE_ADJ_MULTI_CONTROLLED_SQRT_PAULI_X_HPP
