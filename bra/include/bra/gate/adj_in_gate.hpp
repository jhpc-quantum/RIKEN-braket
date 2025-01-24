#ifndef BRA_GATE_ADJ_IN_GATE_HPP
# define BRA_GATE_ADJ_IN_GATE_HPP

# include <vector>
# include <string>
# include <iosfwd>

# include <bra/gate/gate.hpp>
# include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    class adj_in_gate final
      : public ::bra::gate::gate
    {
     public:
      using qubit_type = ::bra::state::qubit_type;

     private:
      std::vector<qubit_type> qubits_;

      std::string name_;

     public:
      explicit adj_in_gate(std::vector<qubit_type> const& qubits);
      explicit adj_in_gate(std::vector<qubit_type>&& qubits);

      ~adj_in_gate() = default;
      adj_in_gate(adj_in_gate const&) = delete;
      adj_in_gate& operator=(adj_in_gate const&) = delete;
      adj_in_gate(adj_in_gate&&) = delete;
      adj_in_gate& operator=(adj_in_gate&&) = delete;

     private:
      ::bra::state& do_apply(::bra::state& state) const override;
      std::string const& do_name() const override;
      std::string do_representation(
        std::ostringstream& repr_stream, int const parameter_width) const override;
    }; // class adj_in_gate
  } // namespace gate
} // namespace bra


#endif // BRA_GATE_ADJ_IN_GATE_HPP
