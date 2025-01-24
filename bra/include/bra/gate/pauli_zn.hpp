#ifndef BRA_GATE_PAULI_ZN_HPP
# define BRA_GATE_PAULI_ZN_HPP

# include <vector>
# include <string>
# include <iosfwd>

# include <bra/gate/gate.hpp>
# include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    class pauli_zn final
      : public ::bra::gate::gate
    {
     public:
      using qubit_type = ::bra::state::qubit_type;

     private:
      std::vector<qubit_type> qubits_;

      std::string name_;

     public:
      explicit pauli_zn(std::vector<qubit_type> const& qubits);
      explicit pauli_zn(std::vector<qubit_type>&& qubits);

      ~pauli_zn() = default;
      pauli_zn(pauli_zn const&) = delete;
      pauli_zn& operator=(pauli_zn const&) = delete;
      pauli_zn(pauli_zn&&) = delete;
      pauli_zn& operator=(pauli_zn&&) = delete;

     private:
      ::bra::state& do_apply(::bra::state& state) const override;
      std::string const& do_name() const override;
      std::string do_representation(
        std::ostringstream& repr_stream, int const parameter_width) const override;
    }; // class pauli_zn
  } // namespace gate
} // namespace bra


#endif // BRA_GATE_PAULI_ZN_HPP
