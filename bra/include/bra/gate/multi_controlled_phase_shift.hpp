#ifndef BRA_GATE_MULTI_CONTROLLED_PHASE_SHIFT_HPP
# define BRA_GATE_MULTI_CONTROLLED_PHASE_SHIFT_HPP

# include <string>
# include <vector>
# include <iosfwd>

# include <bra/gate/gate.hpp>
# include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    class multi_controlled_phase_shift final
      : public ::bra::gate::gate
    {
     public:
      using qubit_type = ::bra::state::qubit_type;
      using control_qubit_type = ::bra::state::control_qubit_type;
      using complex_type = ::bra::state::complex_type;

     private:
      int phase_exponent_;
      complex_type phase_coefficient_;
      qubit_type target_qubit_;
      std::vector<control_qubit_type> control_qubits_;

      std::string name_;

     public:
      multi_controlled_phase_shift(
        int const phase_exponent,
        complex_type const& phase_coefficient,
        qubit_type const target_qubit,
        std::vector<control_qubit_type>&& control_qubits);

      ~multi_controlled_phase_shift() = default;
      multi_controlled_phase_shift(multi_controlled_phase_shift const&) = delete;
      multi_controlled_phase_shift& operator=(multi_controlled_phase_shift const&) = delete;
      multi_controlled_phase_shift(multi_controlled_phase_shift&&) = delete;
      multi_controlled_phase_shift& operator=(multi_controlled_phase_shift&&) = delete;

     private:
      ::bra::state& do_apply(::bra::state& state) const override;
      std::string const& do_name() const override;
      std::string do_representation(
        std::ostringstream& repr_stream, int const parameter_width) const override;
    }; // class multi_controlled_phase_shift
  } // namespace gate
} // namespace bra


#endif // BRA_GATE_MULTI_CONTROLLED_PHASE_SHIFT_HPP
