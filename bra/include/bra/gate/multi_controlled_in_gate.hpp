#ifndef BRA_GATE_MULTI_CONTROLLED_IN_GATE_HPP
# define BRA_GATE_MULTI_CONTROLLED_IN_GATE_HPP

# include <string>
# include <vector>
# include <iosfwd>

# include <bra/gate/gate.hpp>
# include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    class multi_controlled_in_gate final
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
      multi_controlled_in_gate(
        std::vector<qubit_type> const& target_qubits,
        std::vector<control_qubit_type> const& control_qubits);

      multi_controlled_in_gate(
        std::vector<qubit_type>&& target_qubits,
        std::vector<control_qubit_type>&& control_qubits);

      ~multi_controlled_in_gate() = default;
      multi_controlled_in_gate(multi_controlled_in_gate const&) = delete;
      multi_controlled_in_gate& operator=(multi_controlled_in_gate const&) = delete;
      multi_controlled_in_gate(multi_controlled_in_gate&&) = delete;
      multi_controlled_in_gate& operator=(multi_controlled_in_gate&&) = delete;

     private:
      ::bra::state& do_apply(::bra::state& state) const override;
      std::string const& do_name() const override;
      std::string do_representation(
        std::ostringstream& repr_stream, int const parameter_width) const override;
    }; // class multi_controlled_in_gate
  } // namespace gate
} // namespace bra


#endif // BRA_GATE_MULTI_CONTROLLED_IN_GATE_HPP
