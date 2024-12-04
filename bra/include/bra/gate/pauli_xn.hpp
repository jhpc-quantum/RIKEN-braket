#ifndef BRA_GATE_PAULI_XN_HPP
# define BRA_GATE_PAULI_XN_HPP

# include <vector>
# include <string>
# include <iosfwd>

# include <bra/gate/gate.hpp>
# include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    class pauli_xn final
      : public ::bra::gate::gate
    {
     public:
      using qubit_type = ::bra::state::qubit_type;

     private:
      std::vector<qubit_type> qubits_;

      std::string name_;

     public:
      explicit pauli_xn(std::vector<qubit_type> const& qubits);
      explicit pauli_xn(std::vector<qubit_type>&& qubits);

      ~pauli_xn() = default;
      pauli_xn(pauli_xn const&) = delete;
      pauli_xn& operator=(pauli_xn const&) = delete;
      pauli_xn(pauli_xn&&) = delete;
      pauli_xn& operator=(pauli_xn&&) = delete;

     private:
      ::bra::state& do_apply(::bra::state& state) const override;
      std::string const& do_name() const override;
      std::string do_representation(
        std::ostringstream& repr_stream, int const parameter_width) const override;
    }; // class pauli_xn
  } // namespace gate
} // namespace bra


#endif // BRA_GATE_PAULI_XN_HPP
