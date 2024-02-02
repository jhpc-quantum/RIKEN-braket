#include <string>
#include <ios>
#include <iomanip>
#include <sstream>

#include <ket/qubit_io.hpp>

#include <bra/gate/gate.hpp>
#include <bra/gate/exponential_pauli_zz.hpp>
#include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    std::string const exponential_pauli_zz::name_ = "eZZ";

    exponential_pauli_zz::exponential_pauli_zz(real_type const phase, qubit_type const qubit1, qubit_type const qubit2)
      : ::bra::gate::gate{}, phase_{phase}, qubit1_{qubit1}, qubit2_{qubit2}
    { }

    ::bra::state& exponential_pauli_zz::do_apply(::bra::state& state) const
    { return state.exponential_pauli_zz(phase_, qubit1_, qubit2_); }

    std::string const& exponential_pauli_zz::do_name() const { return name_; }
    std::string exponential_pauli_zz::do_representation(
      std::ostringstream& repr_stream, int const parameter_width) const
    {
      repr_stream
        << std::right
        << std::setw(parameter_width) << qubit1_
        << std::setw(parameter_width) << qubit2_
        << std::setw(parameter_width) << phase_;
      return repr_stream.str();
    }
  } // namespace gate
} // namespace bra
