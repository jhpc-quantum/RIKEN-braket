#ifndef BRA_GATE_EXPONENTIAL_PAULI_YN_HPP
# define BRA_GATE_EXPONENTIAL_PAULI_YN_HPP

# include <vector>
# include <string>
# include <iosfwd>

# include <bra/gate/gate.hpp>
# include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    class exponential_pauli_yn final
      : public ::bra::gate::gate
    {
     public:
      using qubit_type = ::bra::state::qubit_type;
      using real_type = ::bra::state::real_type;

     private:
      real_type phase_;
      std::vector<qubit_type> qubits_;

      std::string name_;

     public:
      explicit exponential_pauli_yn(real_type const phase, std::vector<qubit_type>&& qubits);

      ~exponential_pauli_yn() = default;
      exponential_pauli_yn(exponential_pauli_yn const&) = delete;
      exponential_pauli_yn& operator=(exponential_pauli_yn const&) = delete;
      exponential_pauli_yn(exponential_pauli_yn&&) = delete;
      exponential_pauli_yn& operator=(exponential_pauli_yn&&) = delete;

     private:
      ::bra::state& do_apply(::bra::state& state) const override;
      std::string const& do_name() const override;
      std::string do_representation(
        std::ostringstream& repr_stream, int const parameter_width) const override;
    }; // class exponential_pauli_yn
  } // namespace gate
} // namespace bra


#endif // BRA_GATE_EXPONENTIAL_PAULI_YN_HPP
