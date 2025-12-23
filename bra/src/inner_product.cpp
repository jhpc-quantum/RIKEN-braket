#include <string>
#include <ios>
#include <iomanip>
#include <sstream>

#include <ket/qubit_io.hpp>

#include <bra/gate/gate.hpp>
#include <bra/gate/inner_product.hpp>
#include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    std::string const inner_product::name_ = "INNER PRODUCT";

    inner_product::inner_product(std::string const& remote_circuit_index_or_all)
      : ::bra::gate::gate{}, remote_circuit_index_or_all_{remote_circuit_index_or_all}
    { }

    ::bra::state& inner_product::do_apply(::bra::state& state) const
    {
      state.inner_product(remote_circuit_index_or_all_);
      return state;
    }

    std::string const& inner_product::do_name() const { return name_; }
    std::string inner_product::do_representation(
      std::ostringstream& repr_stream, int const parameter_width) const
    { return ""; }
  } // namespace gate
} // namespace bra
