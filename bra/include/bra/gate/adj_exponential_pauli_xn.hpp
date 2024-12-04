#ifndef BRA_GATE_ADJ_EXPONENTIAL_PAULI_XN_HPP
# define BRA_GATE_ADJ_EXPONENTIAL_PAULI_XN_HPP

# include <vector>
# include <string>
# include <iosfwd>

# include <bra/gate/gate.hpp>
# include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    class adj_exponential_pauli_xn final
      : public ::bra::gate::gate
    {
     public:
      using qubit_type = ::bra::state::qubit_type;
      using real_type = ::bra::state::real_type;

     private:
      real_type phase_;
      std::vector<qubit_type> qubits_;

      std::string name_;

     public:
      adj_exponential_pauli_xn(real_type const phase, std::vector<qubit_type> const& qubits);
      adj_exponential_pauli_xn(real_type const phase, std::vector<qubit_type>&& qubits);

      ~adj_exponential_pauli_xn() = default;
      adj_exponential_pauli_xn(adj_exponential_pauli_xn const&) = delete;
      adj_exponential_pauli_xn& operator=(adj_exponential_pauli_xn const&) = delete;
      adj_exponential_pauli_xn(adj_exponential_pauli_xn&&) = delete;
      adj_exponential_pauli_xn& operator=(adj_exponential_pauli_xn&&) = delete;

     private:
      ::bra::state& do_apply(::bra::state& state) const override;
      std::string const& do_name() const override;
      std::string do_representation(
        std::ostringstream& repr_stream, int const parameter_width) const override;
    }; // class adj_exponential_pauli_xn
  } // namespace gate
} // namespace bra


#endif // BRA_GATE_ADJ_EXPONENTIAL_PAULI_XN_HPP
