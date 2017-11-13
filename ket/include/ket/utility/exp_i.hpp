#ifndef KET_UTILITY_EXP_I_HPP
# define KET_UTILITY_EXP_I_HPP

# include <boost/config.hpp>

# include <cmath>
# include <complex>


namespace ket
{
  namespace utility
  {
    namespace exp_i_detail
    {
      template <typename Complex>
      struct exp_i
      {
        template <typename Real>
        static BOOST_CONSTEXPR Complex call(Real const real);
      };

      template <typename Real>
      struct exp_i<std::complex<Real> >
      {
        static BOOST_CONSTEXPR std::complex<Real> call(Real const real)
          BOOST_NOEXCEPT_IF(( BOOST_NOEXCEPT_EXPR(( std::exp(std::complex<Real>(static_cast<Real>(0), real)) )) ))
        {
          using std::exp;
          return exp(std::complex<Real>(static_cast<Real>(0), real));
        }
      };
    }

    template <typename Complex, typename Real>
    inline BOOST_CONSTEXPR Complex exp_i(Real const phase)
      BOOST_NOEXCEPT_IF(( BOOST_NOEXCEPT_EXPR(( exp_i_detail::exp_i<Complex>::call(phase) )) ))
    { return exp_i_detail::exp_i<Complex>::call(phase); }
  }
}


#endif

