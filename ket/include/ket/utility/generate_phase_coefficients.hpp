#ifndef KET_UTILITY_GENERATE_PHASE_COEFFICIENTS_HPP
# define KET_UTILITY_GENERATE_PHASE_COEFFICIENTS_HPP

# include <cstddef>

# include <vector>
# include <type_traits>

# include <boost/math/constants/constants.hpp>

# include <ket/utility/exp_i.hpp>
# include <ket/utility/integer_exp2.hpp>
# include <ket/utility/meta/real_of.hpp>


namespace ket
{
  namespace utility
  {
    template <typename Complex, typename Allocator, typename BitInteger>
    inline auto generate_phase_coefficients(
      std::vector<Complex, Allocator>& phase_coefficients, BitInteger const num_qubits)
    -> void
    {
      static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");

      phase_coefficients.reserve(num_qubits + BitInteger{1});
      for (auto phase_exponent = phase_coefficients.size();
           phase_exponent <= static_cast<decltype(phase_exponent)>(num_qubits); ++phase_exponent)
      {
        using real_type = typename ::ket::utility::meta::real_t<Complex>;
        using boost::math::constants::two_pi;
        phase_coefficients.push_back(
          ket::utility::exp_i<Complex>(
            two_pi<real_type>()
            / static_cast<real_type>(::ket::utility::integer_exp2<std::size_t>(phase_exponent))));
      }
      phase_coefficients.resize(num_qubits + BitInteger{1});
    }

    template <typename Complex, typename Allocator, typename BitInteger>
    inline auto generate_phase_coefficients(BitInteger const num_qubits) -> std::vector<Complex, Allocator>
    {
      auto result = std::vector<Complex, Allocator>{};
      ::ket::utility::generate_phase_coefficients(result, num_qubits);
      return result;
    }

    template <typename Complex, typename BitInteger>
    inline auto generate_phase_coefficients(BitInteger const num_qubits) -> std::vector<Complex>
    {
      auto result = std::vector<Complex>{};
      ::ket::utility::generate_phase_coefficients(result, num_qubits);
      return result;
    }
  } // namespace utility
} // namespace ket


#endif // KET_UTILITY_GENERATE_PHASE_COEFFICIENTS_HPP
