#include <string>
#include <ios>
#include <iomanip>
#include <sstream>
#include <utility>

#include <boost/variant/variant.hpp>
#include <boost/variant/apply_visitor.hpp>

#include <ket/qubit_io.hpp>

#include <bra/gate/gate.hpp>
#include <bra/gate/exponential_pauli_y.hpp>
#include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    std::string const exponential_pauli_y::name_ = "eY";

    exponential_pauli_y::exponential_pauli_y(
      boost::variant<real_type, std::string> const& phase,
      qubit_type const qubit)
      : ::bra::gate::gate{}, phase_{phase}, qubit_{qubit}
    { }

    ::bra::state& exponential_pauli_y::do_apply(::bra::state& state) const
    { return state.exponential_pauli_y(phase_, qubit_); }

    std::string const& exponential_pauli_y::do_name() const { return name_; }
    std::string exponential_pauli_y::do_representation(
      std::ostringstream& repr_stream, int const parameter_width) const
    {
      repr_stream
        << std::right
        << std::setw(parameter_width) << qubit_
        << std::setw(parameter_width) << boost::apply_visitor(::bra::gate::gate_detail::output_visitor<real_type>{}, phase_);
      return repr_stream.str();
    }
  } // namespace gate
} // namespace bra
