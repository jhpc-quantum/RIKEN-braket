#ifndef BRA_GATE_MULTI_CONTROLLED_SQRT_PAULI_Y_HPP
# define BRA_GATE_MULTI_CONTROLLED_SQRT_PAULI_Y_HPP

# include <string>
# include <vector>
# include <iosfwd>

# include <bra/gate/gate.hpp>
# include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    class multi_controlled_sqrt_pauli_y final
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
      multi_controlled_sqrt_pauli_y(
        qubit_type const target_qubit,
        std::vector<control_qubit_type> const& control_qubits);

      multi_controlled_sqrt_pauli_y(
        qubit_type const target_qubit,
        std::vector<control_qubit_type>&& control_qubits);

      ~multi_controlled_sqrt_pauli_y() = default;
      multi_controlled_sqrt_pauli_y(multi_controlled_sqrt_pauli_y const&) = delete;
      multi_controlled_sqrt_pauli_y& operator=(multi_controlled_sqrt_pauli_y const&) = delete;
      multi_controlled_sqrt_pauli_y(multi_controlled_sqrt_pauli_y&&) = delete;
      multi_controlled_sqrt_pauli_y& operator=(multi_controlled_sqrt_pauli_y&&) = delete;

     private:
      ::bra::state& do_apply(::bra::state& state) const override;
      std::string const& do_name() const override;
      std::string do_representation(
        std::ostringstream& repr_stream, int const parameter_width) const override;
    }; // class multi_controlled_sqrt_pauli_y
  } // namespace gate
} // namespace bra


#endif // BRA_GATE_MULTI_CONTROLLED_SQRT_PAULI_Y_HPP
