#ifndef BRA_ANGLEBITS2PHASE_HPP
# define BRA_ANGLEBITS2PHASE_HPP

# include <memory>
# include <string>
# include <vector>
# include <bra/types.hpp>

namespace bra
{
  // Immutable instruction metadata, not a classical variable/operand type.
  struct anglebits2phase_chain;
  using anglebits2phase_chain_ptr = std::shared_ptr< ::bra::anglebits2phase_chain const >;

  auto parse_anglebits2phase_chain(std::vector<std::string> const& descriptors)
    -> ::bra::anglebits2phase_chain_ptr;
  auto evaluate_anglebits2phase(::bra::uint_type bits, ::bra::int_type width,
                               ::bra::anglebits2phase_chain const& chain) -> ::bra::real_type;
}

#endif // BRA_ANGLEBITS2PHASE_HPP
