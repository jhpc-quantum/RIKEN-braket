#ifndef BRA_GATE_TOFFOLI_HPP
# define BRA_GATE_TOFFOLI_HPP

# include <string>
# include <iosfwd>

# include <bra/gate/gate.hpp>
# include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    class toffoli final
      : public ::bra::gate::gate
    {
     public:
      using qubit_type = ::bra::state::qubit_type;
      using control_qubit_type = ::bra::state::control_qubit_type;

     private:
      qubit_type target_qubit_;
      control_qubit_type control_qubit1_;
      control_qubit_type control_qubit2_;

      static std::string const name_;

     public:
      toffoli(
        qubit_type const target_qubit,
        control_qubit_type const control_qubit1, control_qubit_type const control_qubit2);

      ~toffoli() = default;
      toffoli(toffoli const&) = delete;
      toffoli& operator=(toffoli const&) = delete;
      toffoli(toffoli&&) = delete;
      toffoli& operator=(toffoli&&) = delete;

     private:
      ::bra::state& do_apply(::bra::state& state) const override;
      std::string const& do_name() const override;
      std::string do_representation(
        std::ostringstream& repr_stream, int const parameter_width) const override;
    }; // class toffoli
  } // namespace gate
} // namespace bra


#endif // BRA_GATE_TOFFOLI_HPP
