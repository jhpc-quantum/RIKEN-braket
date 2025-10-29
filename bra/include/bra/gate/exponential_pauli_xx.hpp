#ifndef BRA_GATE_EXPONENTIAL_PAULI_XX_HPP
# define BRA_GATE_EXPONENTIAL_PAULI_XX_HPP

# include <string>
# include <iosfwd>

# include <boost/variant/variant.hpp>

# include <bra/gate/gate.hpp>
# include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    class exponential_pauli_xx final
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
      exponential_pauli_xx(boost::variant<real_type, std::string> const& phase, qubit_type const qubit1, qubit_type const qubit2);

      ~exponential_pauli_xx() = default;
      exponential_pauli_xx(exponential_pauli_xx const&) = delete;
      exponential_pauli_xx& operator=(exponential_pauli_xx const&) = delete;
      exponential_pauli_xx(exponential_pauli_xx&&) = delete;
      exponential_pauli_xx& operator=(exponential_pauli_xx&&) = delete;

     private:
      ::bra::state& do_apply(::bra::state& state) const override;
      std::string const& do_name() const override;
      std::string do_representation(
        std::ostringstream& repr_stream, int const parameter_width) const override;
    }; // class exponential_pauli_xx
  } // namespace gate
} // namespace bra


#endif // BRA_GATE_EXPONENTIAL_PAULI_XX_HPP
