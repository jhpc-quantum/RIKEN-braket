#ifndef BRA_GATE_ADJ_TOFFOLI_HPP
# define BRA_GATE_ADJ_TOFFOLI_HPP

# include <string>
# include <iosfwd>

# include <bra/gate/gate.hpp>
# include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    class adj_toffoli final
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
      adj_toffoli(
        qubit_type const target_qubit,
        control_qubit_type const control_qubit1, control_qubit_type const control_qubit2);

      ~adj_toffoli() = default;
      adj_toffoli(adj_toffoli const&) = delete;
      adj_toffoli& operator=(adj_toffoli const&) = delete;
      adj_toffoli(adj_toffoli&&) = delete;
      adj_toffoli& operator=(adj_toffoli&&) = delete;

     private:
      ::bra::state& do_apply(::bra::state& state) const override;
      std::string const& do_name() const override;
      std::string do_representation(
        std::ostringstream& repr_stream, int const parameter_width) const override;
    }; // class adj_toffoli
  } // namespace gate
} // namespace bra


#endif // BRA_GATE_ADJ_TOFFOLI_HPP
