#ifndef BRA_GATE_ADJ_MULTI_CONTROLLED_HADAMARD_HPP
# define BRA_GATE_ADJ_MULTI_CONTROLLED_HADAMARD_HPP

# include <string>
# include <vector>
# include <iosfwd>

# include <bra/gate/gate.hpp>
# include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    class adj_multi_controlled_hadamard final
      : public ::bra::gate::gate
    {
     public:
      using qubit_type = ::bra::state::qubit_type;
      using control_qubit_type = ::bra::state::control_qubit_type;

     private:
      qubit_type target_qubit_;
      std::vector<control_qubit_type> control_qubits_;

      std::string name_;

     public:
      adj_multi_controlled_hadamard(
        qubit_type const target_qubit,
        std::vector<control_qubit_type> const& control_qubits);

      adj_multi_controlled_hadamard(
        qubit_type const target_qubit,
        std::vector<control_qubit_type>&& control_qubits);

      ~adj_multi_controlled_hadamard() = default;
      adj_multi_controlled_hadamard(adj_multi_controlled_hadamard const&) = delete;
      adj_multi_controlled_hadamard& operator=(adj_multi_controlled_hadamard const&) = delete;
      adj_multi_controlled_hadamard(adj_multi_controlled_hadamard&&) = delete;
      adj_multi_controlled_hadamard& operator=(adj_multi_controlled_hadamard&&) = delete;

     private:
      ::bra::state& do_apply(::bra::state& state) const override;
      std::string const& do_name() const override;
      std::string do_representation(
        std::ostringstream& repr_stream, int const parameter_width) const override;
    }; // class adj_multi_controlled_hadamard
  } // namespace gate
} // namespace bra


#endif // BRA_GATE_ADJ_MULTI_CONTROLLED_HADAMARD_HPP
