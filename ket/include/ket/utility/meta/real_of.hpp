#ifndef KET_UTILTIY_META_REAL_OF_HPP
# define KET_UTILTIY_META_REAL_OF_HPP

# include <boost/config.hpp>

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
    }
  }
}


#endif

