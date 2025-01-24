#ifndef BRA_GATE_MULTI_CONTROLLED_PAULI_XN_HPP
# define BRA_GATE_MULTI_CONTROLLED_PAULI_XN_HPP

# include <string>
# include <vector>
# include <iosfwd>

# include <bra/gate/gate.hpp>
# include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    class multi_controlled_pauli_xn final
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
      multi_controlled_pauli_xn(
        std::vector<qubit_type> const& target_qubits,
        std::vector<control_qubit_type> const& control_qubits);

      multi_controlled_pauli_xn(
        std::vector<qubit_type>&& target_qubits,
        std::vector<control_qubit_type>&& control_qubits);

      ~multi_controlled_pauli_xn() = default;
      multi_controlled_pauli_xn(multi_controlled_pauli_xn const&) = delete;
      multi_controlled_pauli_xn& operator=(multi_controlled_pauli_xn const&) = delete;
      multi_controlled_pauli_xn(multi_controlled_pauli_xn&&) = delete;
      multi_controlled_pauli_xn& operator=(multi_controlled_pauli_xn&&) = delete;

     private:
      ::bra::state& do_apply(::bra::state& state) const override;
      std::string const& do_name() const override;
      std::string do_representation(
        std::ostringstream& repr_stream, int const parameter_width) const override;
    }; // class multi_controlled_pauli_xn
  } // namespace gate
} // namespace bra


#endif // BRA_GATE_MULTI_CONTROLLED_PAULI_XN_HPP
