#include <vector>
#include <string>
#include <ios>
#include <iomanip>
#include <sstream>

#include <ket/qubit_io.hpp>

#include <bra/gate/gate.hpp>
#include <bra/gate/begin_fusion.hpp>
#include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    std::string const begin_fusion::name_ = "BEGIN FUSION";

    begin_fusion::begin_fusion()
      : ::bra::gate::gate{}
    { }

    ::bra::state& begin_fusion::do_apply(::bra::state& state) const
    { return state.begin_fusion(); }

    std::string const& begin_fusion::do_name() const { return name_; }
    std::string begin_fusion::do_representation(
      std::ostringstream& repr_stream, int const parameter_width) const
    { return repr_stream.str(); }
  } // namespace gate
} // namespace bra
