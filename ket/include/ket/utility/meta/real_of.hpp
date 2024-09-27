#ifndef KET_UTILTIY_META_REAL_OF_HPP
# define KET_UTILTIY_META_REAL_OF_HPP

# include <complex>


namespace ket
{
  namespace utility
  {
    namespace meta
    {
      template <typename Complex>
      struct real_of;

      template <typename T>
      struct real_of<std::complex<T> >
      { typedef typename std::complex<T>::value_type type; };

      template <typename T>
      using real_t = typename ::ket::utility::meta::real_of<T>::type;
    } // namespace meta
  } // namespace utility
} // namespace ket


#endif // KET_UTILTIY_META_REAL_OF_HPP
