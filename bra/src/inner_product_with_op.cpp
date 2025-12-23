#include <string>
#include <ios>
#include <iomanip>
#include <sstream>

#include <ket/qubit_io.hpp>

#include <bra/gate/gate.hpp>
#include <bra/gate/inner_product_with_op.hpp>
#include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    std::string const inner_product_with_op::name_ = "INNER PRODUCT";

    inner_product_with_op::inner_product_with_op(std::string const& remote_circuit_index_or_all, std::string const& operator_literal_or_variable_name, std::vector<qubit_type> const& operated_qubits)
      : ::bra::gate::gate{}, remote_circuit_index_or_all_{remote_circuit_index_or_all}, operator_literal_or_variable_name_{operator_literal_or_variable_name}, operated_qubits_{operated_qubits}
    { }

    ::bra::state& inner_product_with_op::do_apply(::bra::state& state) const
    {
      state.inner_product(remote_circuit_index_or_all_, operator_literal_or_variable_name_, operated_qubits_);
      return state;
    }

    std::string const& inner_product_with_op::do_name() const { return name_; }
    std::string inner_product_with_op::do_representation(
      std::ostringstream& repr_stream, int const parameter_width) const
    { return ""; }
  } // namespace gate
} // namespace bra
