#ifndef KET_META_STATE_INTEGER_OF_HPP
# define KET_META_STATE_INTEGER_OF_HPP


namespace ket
{
  namespace meta
  {
    template <typename Qubit>
    struct state_integer_of
    { using type = typename Qubit::state_integer_type; };

    template <typename Qubit>
    using state_integer_t = typename ::ket::meta::state_integer_of<Qubit>::type;
  } // namespace meta
} // namespace ket


#endif // KET_META_STATE_INTEGER_OF_HPP
