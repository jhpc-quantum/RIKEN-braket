#ifndef BRA_GATE_ADJ_PAULI_YY_HPP
# define BRA_GATE_ADJ_PAULI_YY_HPP

# include <string>
# include <iosfwd>

# include <bra/gate/gate.hpp>
# include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    class adj_pauli_yy final
      : public ::bra::gate::gate
    {
     public:
      using qubit_type = ::bra::state::qubit_type;

     private:
      qubit_type qubit1_;
      qubit_type qubit2_;

      static std::string const name_;

     public:
      adj_pauli_yy(qubit_type const qubit1, qubit_type const qubit2);

      ~adj_pauli_yy() = default;
      adj_pauli_yy(adj_pauli_yy const&) = delete;
      adj_pauli_yy& operator=(adj_pauli_yy const&) = delete;
      adj_pauli_yy(adj_pauli_yy&&) = delete;
      adj_pauli_yy& operator=(adj_pauli_yy&&) = delete;

     private:
      ::bra::state& do_apply(::bra::state& state) const override;
      std::string const& do_name() const override;
      std::string do_representation(
        std::ostringstream& repr_stream, int const parameter_width) const override;
    }; // class adj_pauli_yy
  } // namespace gate
} // namespace bra


#endif // BRA_GATE_ADJ_PAULI_YY_HPP
