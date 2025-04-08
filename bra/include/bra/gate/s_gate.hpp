#ifndef BRA_GATE_S_GATE_HPP
# define BRA_GATE_S_GATE_HPP

# include <string>
# include <iosfwd>

# include <bra/gate/gate.hpp>
# include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    class s_gate final
      : public ::bra::gate::gate
    {
     public:
      using control_qubit_type = ::bra::state::control_qubit_type;

     private:
      control_qubit_type control_qubit_;

      static std::string const name_;

     public:
      explicit s_gate(control_qubit_type const control_qubit);

      ~s_gate() = default;
      s_gate(s_gate const&) = delete;
      s_gate& operator=(s_gate const&) = delete;
      s_gate(s_gate&&) = delete;
      s_gate& operator=(s_gate&&) = delete;

     private:
      ::bra::state& do_apply(::bra::state& state) const override;
      std::string const& do_name() const override;
      std::string do_representation(
        std::ostringstream& repr_stream, int const parameter_width) const override;
    }; // class s_gate
  } // namespace gate
} // namespace bra


#endif // BRA_GATE_S_GATE_HPP
