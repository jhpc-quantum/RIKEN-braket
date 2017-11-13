#ifndef KET_META_STATE_INTEGER_OF_HPP
# define KET_META_STATE_INTEGER_OF_HPP

# include <boost/config.hpp>


namespace ket
{
  namespace meta
  {
    template <typename Qubit>
    struct state_integer_of
    { typedef typename Qubit::state_integer_type type; };
  } // namespace meta
} // namespace ket


#endif

