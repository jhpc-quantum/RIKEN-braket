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
          // Keep the library boundary: under fast-math GCC can otherwise
          // lower exp(i*real) to x87 FSINCOS, whose argument range and range
          // reduction accuracy are insufficient for large finite phases.
          // Direct sin/cos calls can undergo the same transformation. An
          // indirect call retains the library's full-range reduction without
          // changing compiler options or reducing modulo an approximate 2*pi.
          using function_type = std::complex<Real> (*)(std::complex<Real> const&);
          function_type volatile function = &std::exp<Real>;
          return function(std::complex<Real>{Real{0}, real});
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
