#ifndef BRA_GATE_ADJ_EXPONENTIAL_PAULI_Z_HPP
# define BRA_GATE_ADJ_EXPONENTIAL_PAULI_Z_HPP

# include <string>
# include <iosfwd>

# include <bra/gate/gate.hpp>
# include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    class adj_exponential_pauli_z final
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
      explicit adj_exponential_pauli_z(real_type const phase, qubit_type const qubit);

      ~adj_exponential_pauli_z() = default;
      adj_exponential_pauli_z(adj_exponential_pauli_z const&) = delete;
      adj_exponential_pauli_z& operator=(adj_exponential_pauli_z const&) = delete;
      adj_exponential_pauli_z(adj_exponential_pauli_z&&) = delete;
      adj_exponential_pauli_z& operator=(adj_exponential_pauli_z&&) = delete;

     private:
      ::bra::state& do_apply(::bra::state& state) const override;
      std::string const& do_name() const override;
      std::string do_representation(
        std::ostringstream& repr_stream, int const parameter_width) const override;
    }; // class adj_exponential_pauli_z
  } // namespace gate
} // namespace bra


#endif // BRA_GATE_ADJ_EXPONENTIAL_PAULI_Z_HPP
