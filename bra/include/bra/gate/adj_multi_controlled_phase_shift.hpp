#ifndef BRA_GATE_ADJ_MULTI_CONTROLLED_PHASE_SHIFT_HPP
# define BRA_GATE_ADJ_MULTI_CONTROLLED_PHASE_SHIFT_HPP

# include <string>
# include <vector>
# include <iosfwd>

# include <boost/variant/variant.hpp>

# include <bra/gate/gate.hpp>
# include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    class adj_multi_controlled_phase_shift final
      : public ::bra::gate::gate
    {
     public:
      using control_qubit_type = ::bra::state::control_qubit_type;
      using complex_type = ::bra::state::complex_type;
      using int_type = ::bra::state::int_type;

     private:
      boost::variant<int_type, std::string> phase_exponent_;
      std::vector<control_qubit_type> control_qubits_;

      std::string name_;

     public:
      adj_multi_controlled_phase_shift(
        boost::variant<int_type, std::string> const& phase_exponent,
        std::vector<control_qubit_type> const& control_qubits);

      adj_multi_controlled_phase_shift(
        boost::variant<int_type, std::string> const& phase_exponent,
        std::vector<control_qubit_type>&& control_qubits);

      ~adj_multi_controlled_phase_shift() = default;
      adj_multi_controlled_phase_shift(adj_multi_controlled_phase_shift const&) = delete;
      adj_multi_controlled_phase_shift& operator=(adj_multi_controlled_phase_shift const&) = delete;
      adj_multi_controlled_phase_shift(adj_multi_controlled_phase_shift&&) = delete;
      adj_multi_controlled_phase_shift& operator=(adj_multi_controlled_phase_shift&&) = delete;

     private:
      ::bra::state& do_apply(::bra::state& state) const override;
      std::string const& do_name() const override;
      std::string do_representation(
        std::ostringstream& repr_stream, int const parameter_width) const override;
    }; // class adj_multi_controlled_phase_shift
  } // namespace gate
} // namespace bra


#endif // BRA_GATE_ADJ_MULTI_CONTROLLED_PHASE_SHIFT_HPP
