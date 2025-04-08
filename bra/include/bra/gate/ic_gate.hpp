#ifndef BRA_GATE_IC_GATE_HPP
# define BRA_GATE_IC_GATE_HPP

# include <string>
# include <iosfwd>

# include <bra/gate/gate.hpp>
# include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    class ic_gate final
      : public ::bra::gate::gate
    {
     public:
      using control_qubit_type = ::bra::state::control_qubit_type;

     private:
      control_qubit_type control_qubit_;

      static std::string const name_;

     public:
      explicit ic_gate(control_qubit_type const control_qubit);

      ~ic_gate() = default;
      ic_gate(ic_gate const&) = delete;
      ic_gate& operator=(ic_gate const&) = delete;
      ic_gate(ic_gate&&) = delete;
      ic_gate& operator=(ic_gate&&) = delete;

     private:
      ::bra::state& do_apply(::bra::state& state) const override;
      std::string const& do_name() const override;
      std::string do_representation(
        std::ostringstream& repr_stream, int const parameter_width) const override;
    }; // class ic_gate
  } // namespace gate
} // namespace bra


#endif // BRA_GATE_IC_GATE_HPP
