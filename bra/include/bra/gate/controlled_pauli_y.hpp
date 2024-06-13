#ifndef BRA_GATE_CONTROLLED_PAULI_Y_HPP
# define BRA_GATE_CONTROLLED_PAULI_Y_HPP

# include <string>
# include <iosfwd>

# include <bra/gate/gate.hpp>
# include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    class controlled_pauli_y final
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
      controlled_pauli_y(
        qubit_type const target_qubit,
        control_qubit_type const control_qubit);

      ~controlled_pauli_y() = default;
      controlled_pauli_y(controlled_pauli_y const&) = delete;
      controlled_pauli_y& operator=(controlled_pauli_y const&) = delete;
      controlled_pauli_y(controlled_pauli_y&&) = delete;
      controlled_pauli_y& operator=(controlled_pauli_y&&) = delete;

     private:
      ::bra::state& do_apply(::bra::state& state) const override;
      std::string const& do_name() const override;
      std::string do_representation(
        std::ostringstream& repr_stream, int const parameter_width) const override;
    }; // class controlled_pauli_y
  } // namespace gate
} // namespace bra


#endif // BRA_GATE_CONTROLLED_PAULI_Y_HPP
