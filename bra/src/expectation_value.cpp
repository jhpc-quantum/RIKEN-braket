#include <string>
#include <ios>
#include <iomanip>
#include <sstream>

#include <ket/qubit_io.hpp>

#include <bra/gate/gate.hpp>
#include <bra/gate/expectation_value.hpp>
#include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    std::string const expectation_value::name_ = "EXPECTATION VALUE";

    expectation_value::expectation_value(std::string const& operator_literal_or_variable_name, std::vector<qubit_type> const& operated_qubits)
      : ::bra::gate::gate{}, operator_literal_or_variable_name_{operator_literal_or_variable_name}, operated_qubits_{operated_qubits}
    { }

    ::bra::state& expectation_value::do_apply(::bra::state& state) const
    {
      state.expectation_value(operator_literal_or_variable_name_, operated_qubits_);
      return state;
    }

    std::string const& expectation_value::do_name() const { return name_; }
    std::string expectation_value::do_representation(
      std::ostringstream& repr_stream, int const parameter_width) const
    { return ""; }
  } // namespace gate
} // namespace bra
