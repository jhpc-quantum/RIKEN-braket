#ifndef BRA_GATE_ADJ_MULTI_CONTROLLED_S_GATE_HPP
# define BRA_GATE_ADJ_MULTI_CONTROLLED_S_GATE_HPP

# include <string>
# include <vector>
# include <iosfwd>

# include <bra/gate/gate.hpp>
# include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    class adj_multi_controlled_s_gate final
      : public ::bra::gate::gate
    {
     public:
      using control_qubit_type = ::bra::state::control_qubit_type;

     private:
      std::vector<control_qubit_type> control_qubits_;

      std::string name_;

     public:
      explicit adj_multi_controlled_s_gate(std::vector<control_qubit_type> const& control_qubit);

      explicit adj_multi_controlled_s_gate(std::vector<control_qubit_type>&& control_qubit);

      ~adj_multi_controlled_s_gate() = default;
      adj_multi_controlled_s_gate(adj_multi_controlled_s_gate const&) = delete;
      adj_multi_controlled_s_gate& operator=(adj_multi_controlled_s_gate const&) = delete;
      adj_multi_controlled_s_gate(adj_multi_controlled_s_gate&&) = delete;
      adj_multi_controlled_s_gate& operator=(adj_multi_controlled_s_gate&&) = delete;

     private:
      ::bra::state& do_apply(::bra::state& state) const override;
      std::string const& do_name() const override;
      std::string do_representation(
        std::ostringstream& repr_stream, int const parameter_width) const override;
    }; // class adj_multi_controlled_s_gate
  } // namespace gate
} // namespace bra


#endif // BRA_GATE_ADJ_MULTI_CONTROLLED_S_GATE_HPP
