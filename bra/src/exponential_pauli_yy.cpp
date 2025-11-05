#include <string>
#include <ios>
#include <iomanip>
#include <sstream>
#include <utility>

#include <boost/variant/variant.hpp>
#include <boost/variant/apply_visitor.hpp>

#include <ket/qubit_io.hpp>

#include <bra/gate/gate.hpp>
#include <bra/gate/exponential_pauli_yy.hpp>
#include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    std::string const exponential_pauli_yy::name_ = "eYY";

    exponential_pauli_yy::exponential_pauli_yy(
      boost::variant<real_type, std::string> const& phase,
      qubit_type const qubit1, qubit_type const qubit2)
      : ::bra::gate::gate{}, phase_{phase}, qubit1_{qubit1}, qubit2_{qubit2}
    { }

    ::bra::state& exponential_pauli_yy::do_apply(::bra::state& state) const
    { return state.exponential_pauli_yy(phase_, qubit1_, qubit2_); }

    std::string const& exponential_pauli_yy::do_name() const { return name_; }
    std::string exponential_pauli_yy::do_representation(
      std::ostringstream& repr_stream, int const parameter_width) const
    {
      repr_stream
        << std::right
        << std::setw(parameter_width) << qubit1_
        << std::setw(parameter_width) << qubit2_
        << std::setw(parameter_width) << boost::apply_visitor(::bra::gate::gate_detail::output_visitor<real_type>{}, phase_);
      return repr_stream.str();
    }
  } // namespace gate
} // namespace bra
