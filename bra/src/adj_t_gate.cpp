#include <string>
#include <ios>
#include <iomanip>
#include <sstream>

#include <ket/qubit_io.hpp>
#include <ket/control_io.hpp>

#include <bra/gate/gate.hpp>
#include <bra/gate/adj_t_gate.hpp>
#include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    std::string const adj_t_gate::name_ = "T+";

    adj_t_gate::adj_t_gate(control_qubit_type const control_qubit)
      : ::bra::gate::gate{},
        control_qubit_{control_qubit}
    { }

    ::bra::state& adj_t_gate::do_apply(::bra::state& state) const
    { return state.adj_phase_shift(3, control_qubit_); }

    std::string const& adj_t_gate::do_name() const { return name_; }
    std::string adj_t_gate::do_representation(
      std::ostringstream& repr_stream, int const parameter_width) const
    {
      repr_stream
        << std::right
        << std::setw(parameter_width) << control_qubit_;
      return repr_stream.str();
    }
  } // namespace gate
} // namespace bra
