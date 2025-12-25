#ifndef BRA_GATE_FIDELITY_WITH_OP_HPP
# define BRA_GATE_FIDELITY_WITH_OP_HPP

# include <string>
# include <vector>
# include <iosfwd>

# include <bra/gate/gate.hpp>
# include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    class fidelity_with_op final
      : public ::bra::gate::gate
    {
     public:
      using qubit_type = ::bra::state::qubit_type;

     private:
      std::string remote_circuit_index_or_all_;
      std::string operator_literal_or_variable_name_;
      std::vector<qubit_type> operated_qubits_;

      static std::string const name_;

     public:
      fidelity_with_op(std::string const& remote_circuit_index_or_all, std::string const& operator_literal_or_variable_name, std::vector<qubit_type> const& operated_qubits);

      ~fidelity_with_op() = default;
      fidelity_with_op(fidelity_with_op const&) = delete;
      fidelity_with_op& operator=(fidelity_with_op const&) = delete;
      fidelity_with_op(fidelity_with_op&&) = delete;
      fidelity_with_op& operator=(fidelity_with_op&&) = delete;

     private:
      ::bra::state& do_apply(::bra::state& state) const override;
      std::string const& do_name() const override;
      std::string do_representation(
        std::ostringstream& repr_stream, int const parameter_width) const override;
    }; // class fidelity_with_op
  } // namespace gate
} // namespace bra


#endif // BRA_GATE_FIDELITY_WITH_OP_HPP
