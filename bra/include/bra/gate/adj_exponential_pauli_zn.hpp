#ifndef BRA_GATE_ADJ_EXPONENTIAL_PAULI_ZN_HPP
# define BRA_GATE_ADJ_EXPONENTIAL_PAULI_ZN_HPP

# include <vector>
# include <string>
# include <iosfwd>

# include <bra/gate/gate.hpp>
# include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    class adj_exponential_pauli_zn final
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
      adj_exponential_pauli_zn(real_type const phase, std::vector<qubit_type>&& qubits);

      ~adj_exponential_pauli_zn() = default;
      adj_exponential_pauli_zn(adj_exponential_pauli_zn const&) = delete;
      adj_exponential_pauli_zn& operator=(adj_exponential_pauli_zn const&) = delete;
      adj_exponential_pauli_zn(adj_exponential_pauli_zn&&) = delete;
      adj_exponential_pauli_zn& operator=(adj_exponential_pauli_zn&&) = delete;

     private:
      ::bra::state& do_apply(::bra::state& state) const override;
      std::string const& do_name() const override;
      std::string do_representation(
        std::ostringstream& repr_stream, int const parameter_width) const override;
    }; // class adj_exponential_pauli_zn
  } // namespace gate
} // namespace bra


#endif // BRA_GATE_ADJ_EXPONENTIAL_PAULI_ZN_HPP
