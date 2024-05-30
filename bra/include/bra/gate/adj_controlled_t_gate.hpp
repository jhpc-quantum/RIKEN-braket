#ifndef BRA_GATE_ADJ_CONTROLLED_T_GATE_HPP
# define BRA_GATE_ADJ_CONTROLLED_T_GATE_HPP

# include <string>
# include <iosfwd>

# include <bra/gate/gate.hpp>
# include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    class adj_controlled_t_gate final
      : public ::bra::gate::gate
    {
     public:
      using qubit_type = ::bra::state::qubit_type;
      using control_qubit_type = ::bra::state::control_qubit_type;
      using complex_type = ::bra::state::complex_type;

     private:
      complex_type phase_coefficient_;
      qubit_type target_qubit_;
      control_qubit_type control_qubit_;

      static std::string const name_;

     public:
      adj_controlled_t_gate(
        complex_type const& phase_coefficient,
        qubit_type const target_qubit, control_qubit_type const control_qubit);

      ~adj_controlled_t_gate() = default;
      adj_controlled_t_gate(adj_controlled_t_gate const&) = delete;
      adj_controlled_t_gate& operator=(adj_controlled_t_gate const&) = delete;
      adj_controlled_t_gate(adj_controlled_t_gate&&) = delete;
      adj_controlled_t_gate& operator=(adj_controlled_t_gate&&) = delete;

     private:
      ::bra::state& do_apply(::bra::state& state) const override;
      std::string const& do_name() const override;
      std::string do_representation(
        std::ostringstream& repr_stream, int const parameter_width) const override;
    }; // class adj_controlled_t_gate
  } // namespace gate
} // namespace bra


#endif // BRA_GATE_ADJ_CONTROLLED_T_GATE_HPP
