#ifndef BRA_GATE_T_GATE_HPP
# define BRA_GATE_T_GATE_HPP

# include <string>
# include <iosfwd>

# include <bra/gate/gate.hpp>
# include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    class t_gate final
      : public ::bra::gate::gate
    {
     public:
      using qubit_type = ::bra::state::qubit_type;
      using complex_type = ::bra::state::complex_type;

     private:
      complex_type phase_coefficient_;
      qubit_type qubit_;

      static std::string const name_;

     public:
      t_gate(
        complex_type const& phase_coefficient,
        qubit_type const qubit);

      ~t_gate() = default;
      t_gate(t_gate const&) = delete;
      t_gate& operator=(t_gate const&) = delete;
      t_gate(t_gate&&) = delete;
      t_gate& operator=(t_gate&&) = delete;

     private:
      ::bra::state& do_apply(::bra::state& state) const override;
      std::string const& do_name() const override;
      std::string do_representation(
        std::ostringstream& repr_stream, int const parameter_width) const override;
    }; // class t_gate
  } // namespace gate
} // namespace bra


#endif // BRA_GATE_T_GATE_HPP
