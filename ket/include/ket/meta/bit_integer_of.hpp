#ifndef KET_META_BIT_INTEGER_OF_HPP
# define KET_META_BIT_INTEGER_OF_HPP

# include <boost/config.hpp>


namespace ket
{
  namespace meta
  {
    template <typename Qubit>
    struct bit_integer_of
    { typedef typename Qubit::bit_integer_type type; };
  } // namespace meta
} // namespace ket


#endif

