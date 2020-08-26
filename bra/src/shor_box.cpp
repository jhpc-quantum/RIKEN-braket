#include <string>
#include <ios>
#include <iomanip>
#include <sstream>

#include <bra/gate/gate.hpp>
#include <bra/gate/shor_box.hpp>
#include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    std::string const shor_box::name_ = "SHORBOX";

    shor_box::shor_box(bit_integer_type const num_exponent_qubits, state_integer_type const divisor, state_integer_type const base)
      : ::bra::gate::gate{}, num_exponent_qubits_{num_exponent_qubits}, divisor_{divisor}, base_{base}
    { }

    ::bra::state& shor_box::do_apply(::bra::state& state) const
    { return state.shor_box(num_exponent_qubits_, divisor_, base_); }

    std::string const& shor_box::do_name() const { return name_; }
    std::string shor_box::do_representation(
      std::ostringstream& repr_stream, int const) const
    { return repr_stream.str(); }
  } // namespace gate
} // namespace bra
