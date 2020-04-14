#ifndef KET_UTILITY_IMAGINARY_UNIT_HPP
# define KET_UTILITY_IMAGINARY_UNIT_HPP

# include <boost/config.hpp>

# include <complex>


namespace ket
{
  namespace utility
  {
# if !defined(BOOST_NO_CXX11_CONSTEXPR)
    namespace imaginary_unit_detail
    {
      template <typename Complex>
      struct imaginary_unit;

      template <typename T>
      struct imaginary_unit<std::complex<T> >
      {
        static constexpr std::complex<T> value
          = std::complex<T>(static_cast<T>(0), static_cast<T>(1));
      };

      template <typename T>
      constexpr std::complex<T> imaginary_unit<std::complex<T> >::value;
    }

    template <typename Complex>
    inline constexpr Complex imaginary_unit() BOOST_NOEXCEPT_OR_NOTHROW
    { return imaginary_unit_detail::imaginary_unit<Complex>::value; }
# else
    namespace imaginary_unit_detail
    {
      template <typename Complex>
      struct imaginary_unit
      {
        static Complex call();
      };

      template <typename T>
      struct imaginary_unit<std::complex<T> >
      {
        static std::complex<T> call()
        { return std::complex<T>(static_cast<T>(0), static_cast<T>(1)); }
      };
    }

    template <typename Complex>
    inline Complex imaginary_unit() BOOST_NOEXCEPT_OR_NOTHROW
    { return imaginary_unit_detail::imaginary_unit<Complex>::call(); }
# endif
  }
}


#endif

