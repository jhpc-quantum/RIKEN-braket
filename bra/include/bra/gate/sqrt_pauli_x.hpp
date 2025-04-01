#ifndef BRA_GATE_SQRT_PAULI_X_HPP
# define BRA_GATE_SQRT_PAULI_X_HPP

# include <string>
# include <iosfwd>

# include <bra/gate/gate.hpp>
# include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    class sqrt_pauli_x final
      : public ::bra::gate::gate
    {
     public:
      using qubit_type = ::bra::state::qubit_type;

     private:
      qubit_type qubit_;

      static std::string const name_;

     public:
      explicit sqrt_pauli_x(qubit_type const qubit);

      ~sqrt_pauli_x() = default;
      sqrt_pauli_x(sqrt_pauli_x const&) = delete;
      sqrt_pauli_x& operator=(sqrt_pauli_x const&) = delete;
      sqrt_pauli_x(sqrt_pauli_x&&) = delete;
      sqrt_pauli_x& operator=(sqrt_pauli_x&&) = delete;

     private:
      ::bra::state& do_apply(::bra::state& state) const override;
      std::string const& do_name() const override;
      std::string do_representation(
        std::ostringstream& repr_stream, int const parameter_width) const override;
    }; // class sqrt_pauli_x
  } // namespace gate
} // namespace bra


#endif // BRA_GATE_SQRT_PAULI_X_HPP
