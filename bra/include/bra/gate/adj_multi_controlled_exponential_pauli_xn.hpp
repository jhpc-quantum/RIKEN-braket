#ifndef BRA_GATE_ADJ_MULTI_CONTROLLED_EXPONENTIAL_PAULI_XN_HPP
# define BRA_GATE_ADJ_MULTI_CONTROLLED_EXPONENTIAL_PAULI_XN_HPP

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
    class adj_multi_controlled_exponential_pauli_xn final
      : public ::bra::gate::gate
    {
     public:
      using qubit_type = ::bra::state::qubit_type;
      using control_qubit_type = ::bra::state::control_qubit_type;
      using real_type = ::bra::state::real_type;

     private:
      boost::variant<real_type, std::string> phase_;
      std::vector<qubit_type> target_qubits_;
      std::vector<control_qubit_type> control_qubits_;

      std::string name_;

     public:
      adj_multi_controlled_exponential_pauli_xn(
        boost::variant<real_type, std::string> const& phase, std::vector<qubit_type> const& target_qubits,
        std::vector<control_qubit_type> const& control_qubits);

      adj_multi_controlled_exponential_pauli_xn(
        boost::variant<real_type, std::string> const& phase, std::vector<qubit_type>&& target_qubits,
        std::vector<control_qubit_type>&& control_qubits);

      ~adj_multi_controlled_exponential_pauli_xn() = default;
      adj_multi_controlled_exponential_pauli_xn(adj_multi_controlled_exponential_pauli_xn const&) = delete;
      adj_multi_controlled_exponential_pauli_xn& operator=(adj_multi_controlled_exponential_pauli_xn const&) = delete;
      adj_multi_controlled_exponential_pauli_xn(adj_multi_controlled_exponential_pauli_xn&&) = delete;
      adj_multi_controlled_exponential_pauli_xn& operator=(adj_multi_controlled_exponential_pauli_xn&&) = delete;

     private:
      ::bra::state& do_apply(::bra::state& state) const override;
      std::string const& do_name() const override;
      std::string do_representation(
        std::ostringstream& repr_stream, int const parameter_width) const override;
    }; // class adj_multi_controlled_exponential_pauli_xn
  } // namespace gate
} // namespace bra


#endif // BRA_GATE_ADJ_MULTI_CONTROLLED_EXPONENTIAL_PAULI_XN_HPP
