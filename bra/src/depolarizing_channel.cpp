#include <string>
#include <ios>
#include <iomanip>
#include <sstream>

#include <bra/gate/gate.hpp>
#include <bra/gate/depolarizing_channel.hpp>
#include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    std::string const depolarizing_channel::name_ = "DEPOLARIZING CHANNEL";

    depolarizing_channel::depolarizing_channel(real_type const px, real_type const py, real_type const pz, int seed)
      : ::bra::gate::gate{}, px_{px}, py_{py}, pz_{pz}, seed_{seed}
    { }

    ::bra::state& depolarizing_channel::do_apply(::bra::state& state) const
    { return state.depolarizing_channel(px_, py_, pz_, seed_); }

    std::string const& depolarizing_channel::do_name() const { return name_; }
    std::string depolarizing_channel::do_representation(
      std::ostringstream& repr_stream, int const) const
    { return repr_stream.str(); }
  } // namespace gate
} // namespace bra
