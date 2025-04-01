#include <string>
#include <ios>
#include <iomanip>
#include <sstream>

#include <ket/qubit_io.hpp>

#include <bra/gate/gate.hpp>
#include <bra/gate/sqrt_pauli_zz.hpp>
#include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    std::string const sqrt_pauli_zz::name_ = "sZZ";

    sqrt_pauli_zz::sqrt_pauli_zz(qubit_type const qubit1, qubit_type const qubit2)
      : ::bra::gate::gate{}, qubit1_{qubit1}, qubit2_{qubit2}
    { }

    ::bra::state& sqrt_pauli_zz::do_apply(::bra::state& state) const
    { return state.sqrt_pauli_zz(qubit1_, qubit2_); }

    std::string const& sqrt_pauli_zz::do_name() const { return name_; }
    std::string sqrt_pauli_zz::do_representation(
      std::ostringstream& repr_stream, int const parameter_width) const
    {
      repr_stream
        << std::right
        << std::setw(parameter_width) << qubit1_
        << std::setw(parameter_width) << qubit2_;
      return repr_stream.str();
    }
  } // namespace gate
} // namespace bra
