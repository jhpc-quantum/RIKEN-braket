#ifndef BRA_GATE_EXPONENTIAL_PAULI_YY_HPP
# define BRA_GATE_EXPONENTIAL_PAULI_YY_HPP

# include <string>
# include <iosfwd>

# include <bra/gate/gate.hpp>
# include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    class exponential_pauli_yy final
      : public ::bra::gate::gate
    {
     public:
      using qubit_type = ::bra::state::qubit_type;
      using real_type = ::bra::state::real_type;

     private:
      real_type phase_;
      qubit_type qubit1_;
      qubit_type qubit2_;

      static std::string const name_;

     public:
      exponential_pauli_yy(real_type const phase, qubit_type const qubit1, qubit_type const qubit2);

      ~exponential_pauli_yy() = default;
      exponential_pauli_yy(exponential_pauli_yy const&) = delete;
      exponential_pauli_yy& operator=(exponential_pauli_yy const&) = delete;
      exponential_pauli_yy(exponential_pauli_yy&&) = delete;
      exponential_pauli_yy& operator=(exponential_pauli_yy&&) = delete;

     private:
      ::bra::state& do_apply(::bra::state& state) const override;
      std::string const& do_name() const override;
      std::string do_representation(
        std::ostringstream& repr_stream, int const parameter_width) const override;
    }; // class exponential_pauli_yy
  } // namespace gate
} // namespace bra


#endif // BRA_GATE_EXPONENTIAL_PAULI_YY_HPP
