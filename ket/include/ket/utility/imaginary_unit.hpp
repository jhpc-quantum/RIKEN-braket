#ifndef KET_UTILITY_IMAGINARY_UNIT_HPP
# define KET_UTILITY_IMAGINARY_UNIT_HPP

# include <complex>


namespace ket
{
  namespace utility
  {
    namespace imaginary_unit_detail
    {
      template <typename Complex>
      struct imaginary_unit;

      template <typename T>
      struct imaginary_unit<std::complex<T>>
      { static constexpr auto value = std::complex<T>{T{0}, T{1}}; };
    } // namespace imaginary_unit_detail

    template <typename Complex>
    inline constexpr Complex imaginary_unit() noexcept
    { return ::ket::utility::imaginary_unit_detail::imaginary_unit<Complex>::value; }
  } // namespace utility
} // namespace ket


#endif // KET_UTILITY_IMAGINARY_UNIT_HPP
