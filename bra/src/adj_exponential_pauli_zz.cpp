#include <string>
#include <ios>
#include <iomanip>
#include <sstream>
#include <utility>

#include <boost/variant/variant.hpp>
#include <boost/variant/apply_visitor.hpp>

#include <ket/qubit_io.hpp>

#include <bra/gate/gate.hpp>
#include <bra/gate/adj_exponential_pauli_zz.hpp>
#include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    std::string const adj_exponential_pauli_zz::name_ = "eZZ+";

    adj_exponential_pauli_zz::adj_exponential_pauli_zz(
      boost::variant<real_type, std::string> const& phase,
      qubit_type const qubit1, qubit_type const qubit2)
      : ::bra::gate::gate{}, phase_{phase}, qubit1_{qubit1}, qubit2_{qubit2}
    { }

    ::bra::state& adj_exponential_pauli_zz::do_apply(::bra::state& state) const
    { return state.adj_exponential_pauli_zz(phase_, qubit1_, qubit2_); }

    std::string const& adj_exponential_pauli_zz::do_name() const { return name_; }
    std::string adj_exponential_pauli_zz::do_representation(
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
