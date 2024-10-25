#ifndef BRA_GATE_I_GATE_HPP
# define BRA_GATE_I_GATE_HPP

# include <string>
# include <iosfwd>

# include <bra/gate/gate.hpp>
# include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    class i_gate final
      : public ::bra::gate::gate
    {
     public:
      using qubit_type = ::bra::state::qubit_type;

     private:
      qubit_type qubit_;

      static std::string const name_;

     public:
      explicit i_gate(qubit_type const qubit);

      ~i_gate() = default;
      i_gate(i_gate const&) = delete;
      i_gate& operator=(i_gate const&) = delete;
      i_gate(i_gate&&) = delete;
      i_gate& operator=(i_gate&&) = delete;

     private:
      ::bra::state& do_apply(::bra::state& state) const override;
      std::string const& do_name() const override;
      std::string do_representation(
        std::ostringstream& repr_stream, int const parameter_width) const override;
    }; // class i_gate
  } // namespace gate
} // namespace bra


#endif // BRA_GATE_I_GATE_HPP
