#ifndef BRA_GATE_ADJ_MULTI_CONTROLLED_SQRT_PAULI_Z_HPP
# define BRA_GATE_ADJ_MULTI_CONTROLLED_SQRT_PAULI_Z_HPP

# include <string>
# include <vector>
# include <iosfwd>

# include <bra/gate/gate.hpp>
# include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    class adj_multi_controlled_sqrt_pauli_z final
      : public ::bra::gate::gate
    {
     public:
      using control_qubit_type = ::bra::state::control_qubit_type;

     private:
      std::vector<control_qubit_type> control_qubits_;

      std::string name_;

     public:
      explicit adj_multi_controlled_sqrt_pauli_z(std::vector<control_qubit_type> const& control_qubits);

      explicit adj_multi_controlled_sqrt_pauli_z(std::vector<control_qubit_type>&& control_qubits);

      ~adj_multi_controlled_sqrt_pauli_z() = default;
      adj_multi_controlled_sqrt_pauli_z(adj_multi_controlled_sqrt_pauli_z const&) = delete;
      adj_multi_controlled_sqrt_pauli_z& operator=(adj_multi_controlled_sqrt_pauli_z const&) = delete;
      adj_multi_controlled_sqrt_pauli_z(adj_multi_controlled_sqrt_pauli_z&&) = delete;
      adj_multi_controlled_sqrt_pauli_z& operator=(adj_multi_controlled_sqrt_pauli_z&&) = delete;

     private:
      ::bra::state& do_apply(::bra::state& state) const override;
      std::string const& do_name() const override;
      std::string do_representation(
        std::ostringstream& repr_stream, int const parameter_width) const override;
    }; // class adj_multi_controlled_sqrt_pauli_z
  } // namespace gate
} // namespace bra


#endif // BRA_GATE_ADJ_MULTI_CONTROLLED_SQRT_PAULI_Z_HPP
