#ifndef KET_META_BASE_HPP
# define KET_META_BASE_HPP

# include <type_traits>

# include <ket/qubit.hpp>
# include <ket/meta/state_integer_of.hpp>

namespace ket
{
  namespace meta
  {
    // Specialize this trait for future quantum-digit types.
    template <typename Qubit>
    struct base;

    template <typename StateInteger, typename BitInteger>
    struct base< ::ket::qubit<StateInteger, BitInteger> >
      : std::integral_constant<StateInteger, StateInteger{2u}>
    { };

    template <typename Qubit>
    constexpr auto base_v = ::ket::meta::base<Qubit>::value;
  }
}

#endif // KET_META_BASE_HPP
