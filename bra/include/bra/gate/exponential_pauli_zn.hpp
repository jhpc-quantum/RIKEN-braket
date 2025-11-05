#ifndef BRA_GATE_EXPONENTIAL_PAULI_ZN_HPP
# define BRA_GATE_EXPONENTIAL_PAULI_ZN_HPP

# include <vector>
# include <string>
# include <iosfwd>

# include <boost/variant/variant.hpp>

# include <bra/gate/gate.hpp>
# include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    class exponential_pauli_zn final
      : public ::bra::gate::gate
    {
     public:
      using qubit_type = ::bra::state::qubit_type;
      using real_type = ::bra::state::real_type;

     private:
      boost::variant<real_type, std::string> phase_;
      std::vector<qubit_type> qubits_;

      std::string name_;

     public:
      exponential_pauli_zn(boost::variant<real_type, std::string> const& phase, std::vector<qubit_type> const& qubits);
      exponential_pauli_zn(boost::variant<real_type, std::string> const& phase, std::vector<qubit_type>&& qubits);

      ~exponential_pauli_zn() = default;
      exponential_pauli_zn(exponential_pauli_zn const&) = delete;
      exponential_pauli_zn& operator=(exponential_pauli_zn const&) = delete;
      exponential_pauli_zn(exponential_pauli_zn&&) = delete;
      exponential_pauli_zn& operator=(exponential_pauli_zn&&) = delete;

     private:
      ::bra::state& do_apply(::bra::state& state) const override;
      std::string const& do_name() const override;
      std::string do_representation(
        std::ostringstream& repr_stream, int const parameter_width) const override;
    }; // class exponential_pauli_zn
  } // namespace gate
} // namespace bra


#endif // BRA_GATE_EXPONENTIAL_PAULI_ZN_HPP
