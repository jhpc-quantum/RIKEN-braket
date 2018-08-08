#ifndef BRA_UTILITY_CLOSEST_FLOATING_POINT_OF_HPP
# define BRA_UTILITY_CLOSEST_FLOATING_POINT_OF_HPP


namespace bra
{
  namespace utility
  {
    template <typename Real>
    struct closest_floating_point_of;

    template <>
    struct closest_floating_point_of<float>
    { typedef float type; };

    template <>
    struct closest_floating_point_of<double>
    { typedef double type; };

    template <>
    struct closest_floating_point_of<long double>
    { typedef long double type; };
  } // namespace utility
} // namespace bra


#endif // BRA_UTILITY_CLOSEST_FLOATING_POINT_OF_HPP
