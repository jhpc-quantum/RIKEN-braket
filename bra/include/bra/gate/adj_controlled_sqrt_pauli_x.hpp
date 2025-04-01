#ifndef BRA_GATE_ADJ_CONTROLLED_SQRT_PAULI_X_HPP
# define BRA_GATE_ADJ_CONTROLLED_SQRT_PAULI_X_HPP

# include <string>
# include <iosfwd>

# include <bra/gate/gate.hpp>
# include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    class adj_controlled_sqrt_pauli_x final
      : public ::bra::gate::gate
    {
     public:
      using qubit_type = ::bra::state::qubit_type;
      using control_qubit_type = ::bra::state::control_qubit_type;

     private:
      qubit_type target_qubit_;
      control_qubit_type control_qubit_;

      static std::string const name_;

     public:
      adj_controlled_sqrt_pauli_x(
        qubit_type const target_qubit,
        control_qubit_type const control_qubit);

      ~adj_controlled_sqrt_pauli_x() = default;
      adj_controlled_sqrt_pauli_x(adj_controlled_sqrt_pauli_x const&) = delete;
      adj_controlled_sqrt_pauli_x& operator=(adj_controlled_sqrt_pauli_x const&) = delete;
      adj_controlled_sqrt_pauli_x(adj_controlled_sqrt_pauli_x&&) = delete;
      adj_controlled_sqrt_pauli_x& operator=(adj_controlled_sqrt_pauli_x&&) = delete;

     private:
      ::bra::state& do_apply(::bra::state& state) const override;
      std::string const& do_name() const override;
      std::string do_representation(
        std::ostringstream& repr_stream, int const parameter_width) const override;
    }; // class adj_controlled_sqrt_pauli_x
  } // namespace gate
} // namespace bra


#endif // BRA_GATE_ADJ_CONTROLLED_SQRT_PAULI_X_HPP
