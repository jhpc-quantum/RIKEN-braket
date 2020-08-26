#ifndef KET_UTILITY_POSITIVE_RANDOM_VALUE_UPTO_HPP
# define KET_UTILITY_POSITIVE_RANDOM_VALUE_UPTO_HPP

# include <random>
# include <type_traits>


namespace ket
{
  namespace utility
  {
    template <typename Real, typename RandomNumberGenerator>
    inline typename std::enable_if<std::is_floating_point<Real>::value, Real>::type
    positive_random_value_upto(
      Real const maximum_value, RandomNumberGenerator& random_number_generator)
    {
      auto distribution = std::uniform_real_distribution<Real>{Real{0}, maximum_value};
      return distribution(random_number_generator);
    }

    template <typename Integer, typename RandomNumberGenerator>
    inline typename std::enable_if<not std::is_floating_point<Integer>::value, Integer>::type
    positive_random_value_upto(
      Integer const maximum_value, RandomNumberGenerator& random_number_generator)
    {
      return static_cast<Integer>(
        ::ket::utility::positive_random_value_upto(
          static_cast<double>(maximum_value), random_number_generator));
    }
  } // namespace utility
} // namespace ket


#endif // KET_UTILITY_POSITIVE_RANDOM_VALUE_UPTO_HPP
