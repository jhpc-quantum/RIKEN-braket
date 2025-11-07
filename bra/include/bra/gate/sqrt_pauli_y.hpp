#ifndef BRA_GATE_SQRT_PAULI_Y_HPP
# define BRA_GATE_SQRT_PAULI_Y_HPP

# include <string>
# include <iosfwd>

# include <bra/gate/gate.hpp>
# include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    class sqrt_pauli_y final
      : public ::bra::gate::gate
    {
     public:
      using qubit_type = ::bra::state::qubit_type;

     private:
      qubit_type qubit_;

      static std::string const name_;

     public:
      explicit sqrt_pauli_y(qubit_type const qubit);

      ~sqrt_pauli_y() = default;
      sqrt_pauli_y(sqrt_pauli_y const&) = delete;
      sqrt_pauli_y& operator=(sqrt_pauli_y const&) = delete;
      sqrt_pauli_y(sqrt_pauli_y&&) = delete;
      sqrt_pauli_y& operator=(sqrt_pauli_y&&) = delete;

     private:
      ::bra::state& do_apply(::bra::state& state) const override;
      std::string const& do_name() const override;
      std::string do_representation(
        std::ostringstream& repr_stream, int const parameter_width) const override;
    }; // class sqrt_pauli_y
  } // namespace gate
} // namespace bra


#endif // BRA_GATE_SQRT_PAULI_Y_HPP
