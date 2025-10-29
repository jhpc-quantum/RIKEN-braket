#include <string>
#include <ios>
#include <iomanip>
#include <sstream>
#include <utility>

#include <boost/variant/variant.hpp>
#include <boost/variant/apply_visitor.hpp>

#include <ket/qubit_io.hpp>
#include <ket/control_io.hpp>

#include <bra/gate/gate.hpp>
#include <bra/gate/controlled_exponential_pauli_y.hpp>
#include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    std::string const controlled_exponential_pauli_y::name_ = "CeY";

    controlled_exponential_pauli_y::controlled_exponential_pauli_y(
      boost::variant<real_type, std::string> const& phase,
      qubit_type const target_qubit, control_qubit_type const control_qubit)
      : ::bra::gate::gate{}, phase_{phase}, target_qubit_{target_qubit}, control_qubit_{control_qubit}
    { }

    ::bra::state& controlled_exponential_pauli_y::do_apply(::bra::state& state) const
    { return state.controlled_exponential_pauli_y(phase_, target_qubit_, control_qubit_); }

    std::string const& controlled_exponential_pauli_y::do_name() const { return name_; }
    std::string controlled_exponential_pauli_y::do_representation(
      std::ostringstream& repr_stream, int const parameter_width) const
    {
      repr_stream
        << std::right
        << std::setw(parameter_width) << control_qubit_
        << std::setw(parameter_width) << target_qubit_
        << std::setw(parameter_width) << boost::apply_visitor(::bra::gate::gate_detail::output_visitor<real_type>{}, phase_);
      return repr_stream.str();
    }
  } // namespace gate
} // namespace bra
