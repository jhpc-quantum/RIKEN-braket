#ifndef BRA_GATE_PAULI_Z_HPP
# define BRA_GATE_PAULI_Z_HPP

# include <string>
# include <iosfwd>

# include <bra/gate/gate.hpp>
# include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    class pauli_z final
      : public ::bra::gate::gate
    {
     public:
      using qubit_type = ::bra::state::qubit_type;

     private:
      qubit_type qubit_;

      static std::string const name_;

     public:
      explicit pauli_z(qubit_type const qubit);

      ~pauli_z() = default;
      pauli_z(pauli_z const&) = delete;
      pauli_z& operator=(pauli_z const&) = delete;
      pauli_z(pauli_z&&) = delete;
      pauli_z& operator=(pauli_z&&) = delete;

     private:
      ::bra::state& do_apply(::bra::state& state) const override;
      std::string const& do_name() const override;
      std::string do_representation(
        std::ostringstream& repr_stream, int const parameter_width) const override;
    }; // class pauli_z
  } // namespace gate
} // namespace bra


#endif // BRA_GATE_PAULI_Z_HPP
