#ifndef BRA_GATE_EXPONENTIAL_PAULI_Z_HPP
# define BRA_GATE_EXPONENTIAL_PAULI_Z_HPP

# include <string>
# include <iosfwd>

# include <bra/gate/gate.hpp>
# include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    class exponential_pauli_z final
      : public ::bra::gate::gate
    {
     public:
      using qubit_type = ::bra::state::qubit_type;
      using real_type = ::bra::state::real_type;

     private:
      real_type phase_;
      qubit_type qubit_;

      static std::string const name_;

     public:
      explicit exponential_pauli_z(real_type const phase, qubit_type const qubit);

      ~exponential_pauli_z() = default;
      exponential_pauli_z(exponential_pauli_z const&) = delete;
      exponential_pauli_z& operator=(exponential_pauli_z const&) = delete;
      exponential_pauli_z(exponential_pauli_z&&) = delete;
      exponential_pauli_z& operator=(exponential_pauli_z&&) = delete;

     private:
      ::bra::state& do_apply(::bra::state& state) const override;
      std::string const& do_name() const override;
      std::string do_representation(
        std::ostringstream& repr_stream, int const parameter_width) const override;
    }; // class exponential_pauli_z
  } // namespace gate
} // namespace bra


#endif // BRA_GATE_EXPONENTIAL_PAULI_Z_HPP
