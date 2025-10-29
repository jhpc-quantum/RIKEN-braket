#ifndef BRA_GATE_ADJ_EXPONENTIAL_PAULI_XX_HPP
# define BRA_GATE_ADJ_EXPONENTIAL_PAULI_XX_HPP

# include <string>
# include <iosfwd>

# include <boost/variant/variant.hpp>

# include <bra/gate/gate.hpp>
# include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    class adj_exponential_pauli_xx final
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
      adj_exponential_pauli_xx(boost::variant<real_type, std::string> const& phase, qubit_type const qubit1, qubit_type const qubit2);

      ~adj_exponential_pauli_xx() = default;
      adj_exponential_pauli_xx(adj_exponential_pauli_xx const&) = delete;
      adj_exponential_pauli_xx& operator=(adj_exponential_pauli_xx const&) = delete;
      adj_exponential_pauli_xx(adj_exponential_pauli_xx&&) = delete;
      adj_exponential_pauli_xx& operator=(adj_exponential_pauli_xx&&) = delete;

     private:
      ::bra::state& do_apply(::bra::state& state) const override;
      std::string const& do_name() const override;
      std::string do_representation(
        std::ostringstream& repr_stream, int const parameter_width) const override;
    }; // class adj_exponential_pauli_xx
  } // namespace gate
} // namespace bra


#endif // BRA_GATE_ADJ_EXPONENTIAL_PAULI_XX_HPP
