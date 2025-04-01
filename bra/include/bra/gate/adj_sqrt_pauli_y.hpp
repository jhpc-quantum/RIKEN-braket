#ifndef BRA_GATE_ADJ_SQRT_PAULI_Y_HPP
# define BRA_GATE_ADJ_SQRT_PAULI_Y_HPP

# include <string>
# include <iosfwd>

# include <bra/gate/gate.hpp>
# include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    class adj_sqrt_pauli_y final
      : public ::bra::gate::gate
    {
     public:
      using qubit_type = ::bra::state::qubit_type;

     private:
      qubit_type qubit_;

      static std::string const name_;

     public:
      explicit adj_sqrt_pauli_y(qubit_type const qubit);

      ~adj_sqrt_pauli_y() = default;
      adj_sqrt_pauli_y(adj_sqrt_pauli_y const&) = delete;
      adj_sqrt_pauli_y& operator=(adj_sqrt_pauli_y const&) = delete;
      adj_sqrt_pauli_y(adj_sqrt_pauli_y&&) = delete;
      adj_sqrt_pauli_y& operator=(adj_sqrt_pauli_y&&) = delete;

     private:
      ::bra::state& do_apply(::bra::state& state) const override;
      std::string const& do_name() const override;
      std::string do_representation(
        std::ostringstream& repr_stream, int const parameter_width) const override;
    }; // class adj_sqrt_pauli_y
  } // namespace gate
} // namespace bra


#endif // BRA_GATE_ADJ_SQRT_PAULI_Y_HPP
