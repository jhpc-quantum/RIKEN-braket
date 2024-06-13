#ifndef BRA_GATE_PAULI_ZZ_HPP
# define BRA_GATE_PAULI_ZZ_HPP

# include <string>
# include <iosfwd>

# include <bra/gate/gate.hpp>
# include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    class pauli_zz final
      : public ::bra::gate::gate
    {
     public:
      using qubit_type = ::bra::state::qubit_type;

     private:
      qubit_type qubit1_;
      qubit_type qubit2_;

      static std::string const name_;

     public:
      pauli_zz(qubit_type const qubit1, qubit_type const qubit2);

      ~pauli_zz() = default;
      pauli_zz(pauli_zz const&) = delete;
      pauli_zz& operator=(pauli_zz const&) = delete;
      pauli_zz(pauli_zz&&) = delete;
      pauli_zz& operator=(pauli_zz&&) = delete;

     private:
      ::bra::state& do_apply(::bra::state& state) const override;
      std::string const& do_name() const override;
      std::string do_representation(
        std::ostringstream& repr_stream, int const parameter_width) const override;
    }; // class pauli_zz
  } // namespace gate
} // namespace bra


#endif // BRA_GATE_PAULI_ZZ_HPP
