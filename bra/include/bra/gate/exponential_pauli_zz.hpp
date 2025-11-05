#ifndef BRA_GATE_EXPONENTIAL_PAULI_ZZ_HPP
# define BRA_GATE_EXPONENTIAL_PAULI_ZZ_HPP

# include <string>
# include <iosfwd>

# include <boost/variant/variant.hpp>

# include <bra/gate/gate.hpp>
# include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    class exponential_pauli_zz final
      : public ::bra::gate::gate
    {
     public:
      using qubit_type = ::bra::state::qubit_type;
      using real_type = ::bra::state::real_type;

     private:
      boost::variant<real_type, std::string> phase_;
      qubit_type qubit1_;
      qubit_type qubit2_;

      static std::string const name_;

     public:
      exponential_pauli_zz(boost::variant<real_type, std::string> const& phase, qubit_type const qubit1, qubit_type const qubit2);

      ~exponential_pauli_zz() = default;
      exponential_pauli_zz(exponential_pauli_zz const&) = delete;
      exponential_pauli_zz& operator=(exponential_pauli_zz const&) = delete;
      exponential_pauli_zz(exponential_pauli_zz&&) = delete;
      exponential_pauli_zz& operator=(exponential_pauli_zz&&) = delete;

     private:
      ::bra::state& do_apply(::bra::state& state) const override;
      std::string const& do_name() const override;
      std::string do_representation(
        std::ostringstream& repr_stream, int const parameter_width) const override;
    }; // class exponential_pauli_zz
  } // namespace gate
} // namespace bra


#endif // BRA_GATE_EXPONENTIAL_PAULI_ZZ_HPP
