#ifndef BRA_GATE_CONTROLLED_PHASE_SHIFT2_HPP
# define BRA_GATE_CONTROLLED_PHASE_SHIFT2_HPP

# include <string>
# include <iosfwd>

# include <boost/variant/variant.hpp>

# include <bra/gate/gate.hpp>
# include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    class controlled_phase_shift_ final
      : public ::bra::gate::gate
    {
     public:
      using control_qubit_type = ::bra::state::control_qubit_type;
      using complex_type = ::bra::state::complex_type;
      using int_type = ::bra::state::int_type;

     private:
      boost::variant<int_type, std::string> phase_exponent_;
      control_qubit_type control_qubit1_;
      control_qubit_type control_qubit2_;

      static std::string const name_;

     public:
      controlled_phase_shift_(
        boost::variant<int_type, std::string> const& phase_exponent,
        control_qubit_type const control_qubit1,
        control_qubit_type const control_qubit2);

      ~controlled_phase_shift_() = default;
      controlled_phase_shift_(controlled_phase_shift_ const&) = delete;
      controlled_phase_shift_& operator=(controlled_phase_shift_ const&) = delete;
      controlled_phase_shift_(controlled_phase_shift_&&) = delete;
      controlled_phase_shift_& operator=(controlled_phase_shift_&&) = delete;

     private:
      ::bra::state& do_apply(::bra::state& state) const override;
      std::string const& do_name() const override;
      std::string do_representation(
        std::ostringstream& repr_stream, int const parameter_width) const override;
    }; // class controlled_phase_shift_
  } // namespace gate
} // namespace bra


#endif // BRA_GATE_CONTROLLED_PHASE_SHIFT2_HPP
