#ifndef BRA_GATE_ADJ_PHASE_SHIFT_HPP
# define BRA_GATE_ADJ_PHASE_SHIFT_HPP

# include <string>
# include <iosfwd>

# include <bra/gate/gate.hpp>
# include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    class adj_phase_shift final
      : public ::bra::gate::gate
    {
     public:
      using control_qubit_type = ::bra::state::control_qubit_type;
      using complex_type = ::bra::state::complex_type;

     private:
      int phase_exponent_;
      complex_type phase_coefficient_;
      control_qubit_type control_qubit_;

      static std::string const name_;

     public:
      adj_phase_shift(
        int const phase_exponent,
        complex_type const& phase_coefficient,
        control_qubit_type const control_qubit);

      ~adj_phase_shift() = default;
      adj_phase_shift(adj_phase_shift const&) = delete;
      adj_phase_shift& operator=(adj_phase_shift const&) = delete;
      adj_phase_shift(adj_phase_shift&&) = delete;
      adj_phase_shift& operator=(adj_phase_shift&&) = delete;

     private:
      ::bra::state& do_apply(::bra::state& state) const override;
      std::string const& do_name() const override;
      std::string do_representation(
        std::ostringstream& repr_stream, int const parameter_width) const override;
    }; // class adj_phase_shift
  } // namespace gate
} // namespace bra


#endif // BRA_GATE_ADJ_PHASE_SHIFT_HPP
