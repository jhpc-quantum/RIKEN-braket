#ifndef BRA_GATE_CONTROLLED_S_GATE_HPP
# define BRA_GATE_CONTROLLED_S_GATE_HPP

# include <string>
# include <iosfwd>

# include <bra/gate/gate.hpp>
# include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    class controlled_s_gate final
      : public ::bra::gate::gate
    {
     public:
      using control_qubit_type = ::bra::state::control_qubit_type;

     private:
      control_qubit_type control_qubit1_;
      control_qubit_type control_qubit2_;

      static std::string const name_;

     public:
      controlled_s_gate(
        control_qubit_type const control_qubit1, control_qubit_type const control_qubit2);

      ~controlled_s_gate() = default;
      controlled_s_gate(controlled_s_gate const&) = delete;
      controlled_s_gate& operator=(controlled_s_gate const&) = delete;
      controlled_s_gate(controlled_s_gate&&) = delete;
      controlled_s_gate& operator=(controlled_s_gate&&) = delete;

     private:
      ::bra::state& do_apply(::bra::state& state) const override;
      std::string const& do_name() const override;
      std::string do_representation(
        std::ostringstream& repr_stream, int const parameter_width) const override;
    }; // class controlled_s_gate
  } // namespace gate
} // namespace bra


#endif // BRA_GATE_CONTROLLED_S_GATE_HPP
