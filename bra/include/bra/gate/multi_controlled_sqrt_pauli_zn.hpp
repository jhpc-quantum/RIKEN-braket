#ifndef BRA_GATE_MULTI_CONTROLLED_SQRT_PAULI_ZN_HPP
# define BRA_GATE_MULTI_CONTROLLED_SQRT_PAULI_ZN_HPP

# include <string>
# include <vector>
# include <iosfwd>

# include <bra/gate/gate.hpp>
# include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    class multi_controlled_sqrt_pauli_zn final
      : public ::bra::gate::gate
    {
     public:
      using qubit_type = ::bra::state::qubit_type;
      using control_qubit_type = ::bra::state::control_qubit_type;

     private:
      std::vector<qubit_type> target_qubits_;
      std::vector<control_qubit_type> control_qubits_;

      std::string name_;

     public:
      multi_controlled_sqrt_pauli_zn(
        std::vector<qubit_type> const& target_qubits,
        std::vector<control_qubit_type> const& control_qubits);

      multi_controlled_sqrt_pauli_zn(
        std::vector<qubit_type>&& target_qubits,
        std::vector<control_qubit_type>&& control_qubits);

      ~multi_controlled_sqrt_pauli_zn() = default;
      multi_controlled_sqrt_pauli_zn(multi_controlled_sqrt_pauli_zn const&) = delete;
      multi_controlled_sqrt_pauli_zn& operator=(multi_controlled_sqrt_pauli_zn const&) = delete;
      multi_controlled_sqrt_pauli_zn(multi_controlled_sqrt_pauli_zn&&) = delete;
      multi_controlled_sqrt_pauli_zn& operator=(multi_controlled_sqrt_pauli_zn&&) = delete;

     private:
      ::bra::state& do_apply(::bra::state& state) const override;
      std::string const& do_name() const override;
      std::string do_representation(
        std::ostringstream& repr_stream, int const parameter_width) const override;
    }; // class multi_controlled_sqrt_pauli_zn
  } // namespace gate
} // namespace bra


#endif // BRA_GATE_MULTI_CONTROLLED_SQRT_PAULI_ZN_HPP
