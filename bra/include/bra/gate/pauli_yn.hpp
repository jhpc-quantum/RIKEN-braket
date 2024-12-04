#ifndef BRA_GATE_PAULI_YN_HPP
# define BRA_GATE_PAULI_YN_HPP

# include <vector>
# include <string>
# include <iosfwd>

# include <bra/gate/gate.hpp>
# include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    class pauli_yn final
      : public ::bra::gate::gate
    {
     public:
      using qubit_type = ::bra::state::qubit_type;

     private:
      std::vector<qubit_type> qubits_;

      std::string name_;

     public:
      explicit pauli_yn(std::vector<qubit_type> const& qubits);
      explicit pauli_yn(std::vector<qubit_type>&& qubits);

      ~pauli_yn() = default;
      pauli_yn(pauli_yn const&) = delete;
      pauli_yn& operator=(pauli_yn const&) = delete;
      pauli_yn(pauli_yn&&) = delete;
      pauli_yn& operator=(pauli_yn&&) = delete;

     private:
      ::bra::state& do_apply(::bra::state& state) const override;
      std::string const& do_name() const override;
      std::string do_representation(
        std::ostringstream& repr_stream, int const parameter_width) const override;
    }; // class pauli_yn
  } // namespace gate
} // namespace bra


#endif // BRA_GATE_PAULI_YN_HPP
