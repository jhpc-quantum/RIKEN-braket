#ifndef KET_UTILITY_EXP_I_HPP
# define KET_UTILITY_EXP_I_HPP

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
        static constexpr auto call(Real const real) -> Complex;
      };

      template <typename Real>
      struct exp_i<std::complex<Real>>
      {
        static constexpr auto call(Real const real)
        noexcept(noexcept(std::exp(std::complex<Real>{Real{0}, real}))) -> std::complex<Real>
        {
          using std::exp;
          return exp(std::complex<Real>{Real{0}, real});
        }
      };
    }

    template <typename Complex, typename Real>
    inline constexpr auto exp_i(Real const phase)
    noexcept(noexcept(::ket::utility::exp_i_detail::exp_i<Complex>::call(phase))) -> Complex
    { return ::ket::utility::exp_i_detail::exp_i<Complex>::call(phase); }
  } // namespace utility
} // namespace ket


#endif // KET_UTILITY_EXP_I_HPP
