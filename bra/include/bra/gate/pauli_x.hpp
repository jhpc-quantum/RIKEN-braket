#ifndef BRA_GATE_PAULI_X_HPP
# define BRA_GATE_PAULI_X_HPP

# include <string>
# include <iosfwd>

# include <bra/gate/gate.hpp>
# include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    class pauli_x final
      : public ::bra::gate::gate
    {
     public:
      using qubit_type = ::bra::state::qubit_type;

     private:
      qubit_type qubit_;

      static std::string const name_;

     public:
      explicit pauli_x(qubit_type const qubit);

      ~pauli_x() = default;
      pauli_x(pauli_x const&) = delete;
      pauli_x& operator=(pauli_x const&) = delete;
      pauli_x(pauli_x&&) = delete;
      pauli_x& operator=(pauli_x&&) = delete;

     private:
      ::bra::state& do_apply(::bra::state& state) const override;
      std::string const& do_name() const override;
      std::string do_representation(
        std::ostringstream& repr_stream, int const parameter_width) const override;
    }; // class pauli_x
  } // namespace gate
} // namespace bra


#endif // BRA_GATE_PAULI_X_HPP
