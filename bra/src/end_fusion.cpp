#include <string>
#include <ios>
#include <iomanip>
#include <sstream>

#include <ket/qubit_io.hpp>

#include <bra/gate/gate.hpp>
#include <bra/gate/end_fusion.hpp>
#include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    std::string const end_fusion::name_ = "END FUSION";

    end_fusion::end_fusion()
      : ::bra::gate::gate{}
    { }

    ::bra::state& end_fusion::do_apply(::bra::state& state) const
    { return state.end_fusion(); }

    std::string const& end_fusion::do_name() const { return name_; }
    std::string end_fusion::do_representation(
      std::ostringstream& repr_stream, int const parameter_width) const
    { return repr_stream.str(); }
  } // namespace gate
} // namespace bra
