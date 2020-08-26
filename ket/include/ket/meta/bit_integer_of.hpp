#ifndef KET_META_BIT_INTEGER_OF_HPP
# define KET_META_BIT_INTEGER_OF_HPP


namespace ket
{
  namespace meta
  {
    template <typename Qubit>
    struct bit_integer_of
    { using type = typename Qubit::bit_integer_type; };
  } // namespace meta
} // namespace ket


#endif // KET_META_BIT_INTEGER_OF_HPP
