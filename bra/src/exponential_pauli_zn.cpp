#include <ios>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>
#include <utility>

#include <ket/qubit_io.hpp>

#include <bra/gate/gate.hpp>
#include <bra/gate/exponential_pauli_zn.hpp>
#include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    exponential_pauli_zn::exponential_pauli_zn(real_type const phase, std::vector<qubit_type>&& qubits)
      : ::bra::gate::gate{}, phase_{phase}, qubits_{std::move(qubits)}, name_{std::string{"e"}.append(qubits_.size(), 'Z')}
    { }

    ::bra::state& exponential_pauli_zn::do_apply(::bra::state& state) const
    { return state.exponential_pauli_zn(phase_, qubits_); }

    std::string const& exponential_pauli_zn::do_name() const { return name_; }
    std::string exponential_pauli_zn::do_representation(
      std::ostringstream& repr_stream, int const parameter_width) const
    {
      for (auto&& qubit: qubits_)
        repr_stream << std::right << std::setw(parameter_width) << qubit;
      repr_stream << std::right << std::setw(parameter_width) << phase_;
      return repr_stream.str();
    }
  } // namespace gate
} // namespace bra
