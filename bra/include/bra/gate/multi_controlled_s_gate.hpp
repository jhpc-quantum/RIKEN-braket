#ifndef BRA_GATE_MULTI_CONTROLLED_S_GATE_HPP
# define BRA_GATE_MULTI_CONTROLLED_S_GATE_HPP

# include <string>
# include <vector>
# include <iosfwd>

# include <bra/gate/gate.hpp>
# include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    class multi_controlled_s_gate final
      : public ::bra::gate::gate
    {
     public:
      using control_qubit_type = ::bra::state::control_qubit_type;

     private:
      std::vector<control_qubit_type> control_qubits_;

      std::string name_;

     public:
      explicit multi_controlled_s_gate(std::vector<control_qubit_type> const& control_qubits);

      explicit multi_controlled_s_gate(std::vector<control_qubit_type>&& control_qubits);

      ~multi_controlled_s_gate() = default;
      multi_controlled_s_gate(multi_controlled_s_gate const&) = delete;
      multi_controlled_s_gate& operator=(multi_controlled_s_gate const&) = delete;
      multi_controlled_s_gate(multi_controlled_s_gate&&) = delete;
      multi_controlled_s_gate& operator=(multi_controlled_s_gate&&) = delete;

     private:
      ::bra::state& do_apply(::bra::state& state) const override;
      std::string const& do_name() const override;
      std::string do_representation(
        std::ostringstream& repr_stream, int const parameter_width) const override;
    }; // class multi_controlled_s_gate
  } // namespace gate
} // namespace bra


#endif // BRA_GATE_MULTI_CONTROLLED_S_GATE_HPP
