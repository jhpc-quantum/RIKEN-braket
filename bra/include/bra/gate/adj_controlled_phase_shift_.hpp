#ifndef BRA_GATE_ADJ_CONTROLLED_PHASE_SHIFT2_HPP
# define BRA_GATE_ADJ_CONTROLLED_PHASE_SHIFT2_HPP

# include <string>
# include <iosfwd>

# include <bra/gate/gate.hpp>
# include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    class adj_controlled_phase_shift_ final
      : public ::bra::gate::gate
    {
     public:
      using control_qubit_type = ::bra::state::control_qubit_type;
      using complex_type = ::bra::state::complex_type;

     private:
      int phase_exponent_;
      complex_type phase_coefficient_;
      control_qubit_type control_qubit1_;
      control_qubit_type control_qubit2_;

      static std::string const name_;

     public:
      adj_controlled_phase_shift_(
        int const phase_exponent,
        complex_type const& phase_coefficient,
        control_qubit_type const control_qubit1,
        control_qubit_type const control_qubit2);

      ~adj_controlled_phase_shift_() = default;
      adj_controlled_phase_shift_(adj_controlled_phase_shift_ const&) = delete;
      adj_controlled_phase_shift_& operator=(adj_controlled_phase_shift_ const&) = delete;
      adj_controlled_phase_shift_(adj_controlled_phase_shift_&&) = delete;
      adj_controlled_phase_shift_& operator=(adj_controlled_phase_shift_&&) = delete;

     private:
      ::bra::state& do_apply(::bra::state& state) const override;
      std::string const& do_name() const override;
      std::string do_representation(
        std::ostringstream& repr_stream, int const parameter_width) const override;
    }; // class adj_controlled_phase_shift_
  } // namespace gate
} // namespace bra


#endif // BRA_GATE_ADJ_CONTROLLED_PHASE_SHIFT2_HPP
