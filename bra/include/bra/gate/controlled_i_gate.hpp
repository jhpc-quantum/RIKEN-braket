#ifndef BRA_GATE_CONTROLLED_I_GATE_HPP
# define BRA_GATE_CONTROLLED_I_GATE_HPP

# include <string>
# include <iosfwd>

# include <bra/gate/gate.hpp>
# include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    class controlled_i_gate final
      : public ::bra::gate::gate
    {
     public:
      using qubit_type = ::bra::state::qubit_type;
      using control_qubit_type = ::bra::state::control_qubit_type;

     private:
      qubit_type target_qubit_;
      control_qubit_type control_qubit_;

      static std::string const name_;

     public:
      controlled_i_gate(
        qubit_type const target_qubit,
        control_qubit_type const control_qubit);

      ~controlled_i_gate() = default;
      controlled_i_gate(controlled_i_gate const&) = delete;
      controlled_i_gate& operator=(controlled_i_gate const&) = delete;
      controlled_i_gate(controlled_i_gate&&) = delete;
      controlled_i_gate& operator=(controlled_i_gate&&) = delete;

     private:
      ::bra::state& do_apply(::bra::state& state) const override;
      std::string const& do_name() const override;
      std::string do_representation(
        std::ostringstream& repr_stream, int const parameter_width) const override;
    }; // class controlled_i_gate
  } // namespace gate
} // namespace bra


#endif // BRA_GATE_CONTROLLED_I_GATE_HPP