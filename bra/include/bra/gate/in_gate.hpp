#ifndef BRA_GATE_IN_GATE_HPP
# define BRA_GATE_IN_GATE_HPP

# include <vector>
# include <string>
# include <iosfwd>

# include <bra/gate/gate.hpp>
# include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    class in_gate final
      : public ::bra::gate::gate
    {
     public:
      using qubit_type = ::bra::state::qubit_type;

     private:
      std::vector<qubit_type> qubits_;

      std::string name_;

     public:
      explicit in_gate(std::vector<qubit_type> const& qubits);
      explicit in_gate(std::vector<qubit_type>&& qubits);

      ~in_gate() = default;
      in_gate(in_gate const&) = delete;
      in_gate& operator=(in_gate const&) = delete;
      in_gate(in_gate&&) = delete;
      in_gate& operator=(in_gate&&) = delete;

     private:
      ::bra::state& do_apply(::bra::state& state) const override;
      std::string const& do_name() const override;
      std::string do_representation(
        std::ostringstream& repr_stream, int const parameter_width) const override;
    }; // class in_gate
  } // namespace gate
} // namespace bra


#endif // BRA_GATE_IN_GATE_HPP
