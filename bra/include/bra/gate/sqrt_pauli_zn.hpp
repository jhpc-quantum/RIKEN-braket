#ifndef BRA_GATE_SQRT_PAULI_ZN_HPP
# define BRA_GATE_SQRT_PAULI_ZN_HPP

# include <vector>
# include <string>
# include <iosfwd>

# include <bra/gate/gate.hpp>
# include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    class sqrt_pauli_zn final
      : public ::bra::gate::gate
    {
     public:
      using qubit_type = ::bra::state::qubit_type;

     private:
      std::vector<qubit_type> qubits_;

      std::string name_;

     public:
      explicit sqrt_pauli_zn(std::vector<qubit_type> const& qubits);
      explicit sqrt_pauli_zn(std::vector<qubit_type>&& qubits);

      ~sqrt_pauli_zn() = default;
      sqrt_pauli_zn(sqrt_pauli_zn const&) = delete;
      sqrt_pauli_zn& operator=(sqrt_pauli_zn const&) = delete;
      sqrt_pauli_zn(sqrt_pauli_zn&&) = delete;
      sqrt_pauli_zn& operator=(sqrt_pauli_zn&&) = delete;

     private:
      ::bra::state& do_apply(::bra::state& state) const override;
      std::string const& do_name() const override;
      std::string do_representation(
        std::ostringstream& repr_stream, int const parameter_width) const override;
    }; // class sqrt_pauli_zn
  } // namespace gate
} // namespace bra


#endif // BRA_GATE_SQRT_PAULI_ZN_HPP
