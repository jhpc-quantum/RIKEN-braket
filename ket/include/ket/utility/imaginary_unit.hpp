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

# if __cplusplus >= 201703L
      template <typename T>
      struct imaginary_unit<std::complex<T>>
      { inline static constexpr auto value = std::complex<T>{T{0}, T{1}}; };
# else
      template <typename T>
      struct imaginary_unit<std::complex<T>>
      { static constexpr auto value() -> std::complex<T> { return {T{0}, T{1}}; } };
# endif

      template <typename Complex>
      struct minus_imaginary_unit;

# if __cplusplus >= 201703L
      template <typename T>
      struct minus_imaginary_unit<std::complex<T>>
      { inline static constexpr auto value = std::complex<T>{T{0}, T{-1}}; };
# else
      template <typename T>
      struct minus_imaginary_unit<std::complex<T>>
      { static constexpr auto value() -> std::complex<T> { return {T{0}, T{-1}}; } };
# endif
    } // namespace imaginary_unit_detail

# if __cplusplus >= 201703L
    template <typename Complex>
    inline constexpr auto imaginary_unit() noexcept -> Complex
    { return ::ket::utility::imaginary_unit_detail::imaginary_unit<Complex>::value; }

    template <typename Complex>
    inline constexpr auto minus_imaginary_unit() noexcept -> Complex
    { return ::ket::utility::imaginary_unit_detail::minus_imaginary_unit<Complex>::value; }
# else
    template <typename Complex>
    inline constexpr auto imaginary_unit() noexcept -> Complex
    { return ::ket::utility::imaginary_unit_detail::imaginary_unit<Complex>::value(); }

    template <typename Complex>
    inline constexpr auto minus_imaginary_unit() noexcept -> Complex
    { return ::ket::utility::imaginary_unit_detail::minus_imaginary_unit<Complex>::value(); }
# endif
  } // namespace utility
} // namespace ket


#endif // KET_UTILITY_IMAGINARY_UNIT_HPP
