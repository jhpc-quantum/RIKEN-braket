#ifndef BRA_GATE_ADJ_EXPONENTIAL_PAULI_YN_HPP
# define BRA_GATE_ADJ_EXPONENTIAL_PAULI_YN_HPP

# include <vector>
# include <string>
# include <iosfwd>

# include <bra/gate/gate.hpp>
# include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    class adj_exponential_pauli_yn final
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
      explicit adj_exponential_pauli_yn(real_type const phase, std::vector<qubit_type>&& qubits);

      ~adj_exponential_pauli_yn() = default;
      adj_exponential_pauli_yn(adj_exponential_pauli_yn const&) = delete;
      adj_exponential_pauli_yn& operator=(adj_exponential_pauli_yn const&) = delete;
      adj_exponential_pauli_yn(adj_exponential_pauli_yn&&) = delete;
      adj_exponential_pauli_yn& operator=(adj_exponential_pauli_yn&&) = delete;

     private:
      ::bra::state& do_apply(::bra::state& state) const override;
      std::string const& do_name() const override;
      std::string do_representation(
        std::ostringstream& repr_stream, int const parameter_width) const override;
    }; // class adj_exponential_pauli_yn
  } // namespace gate
} // namespace bra


#endif // BRA_GATE_ADJ_EXPONENTIAL_PAULI_YN_HPP
