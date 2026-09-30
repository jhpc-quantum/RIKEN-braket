#ifndef BRA_UTILITY_MAKE_STATE_HPP
# define BRA_UTILITY_MAKE_STATE_HPP

# include <memory>
# include <utility>

# include <bra/state.hpp>

namespace bra
{
  namespace utility
  {
    template <typename State, typename... Args>
    auto make_state(Args&&... args) -> std::unique_ptr< ::bra::state >
    { return std::unique_ptr< ::bra::state >{new State{std::forward<Args>(args)...}}; }
  } // namespace utility
} // namespace bra

#endif // BRA_UTILITY_MAKE_STATE_HPP
