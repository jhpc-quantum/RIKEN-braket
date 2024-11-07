#ifndef KET_UTILITY_ALL_IN_STATE_VECTOR_HPP
# define KET_UTILITY_ALL_IN_STATE_VECTOR_HPP

# include <utility>
# include <type_traits>

# include <ket/utility/variadic/all_of.hpp>


namespace ket
{
  namespace utility
  {
# if __cpp_constexpr < 201603
    namespace all_in_state_vector_detail
    {
      template <typename BitInteger>
      struct is_in_state_vector
      {
        BitInteger num_qubits_;

        template <typename Qubit>
        constexpr auto operator()(Qubit&& qubit) const noexcept
        { return std::forward<Qubit>(qubit) < std::remove_cv_t<std::remove_reference_t<decltype(qubit)>>{num_qubits_}; }
      }; // struct is_in_state_vector<BitInteger>
    } // namespace all_in_state_vector_detail

# endif // __cpp_constexpr >= 201603
    template <typename BitInteger, typename Qubit, typename... Qubits>
    inline constexpr auto all_in_state_vector(BitInteger const num_qubits, Qubit&& qubit, Qubits&&... qubits) -> bool
    {
# if __cpp_constexpr >= 201603
      return ::ket::utility::variadic::all_of(
        [num_qubits](auto&& qubit) { return std::forward<Qubit>(qubit) < std::remove_cv_t<std::remove_reference_t<decltype(qubit)>>{num_qubits}; },
        std::forward<Qubit>(qubit), std::forward<Qubits>(qubits)...);
# else // __cpp_constexpr >= 201603
      return ::ket::utility::variadic::all_of(
        ::ket::utility::all_in_state_vector_detail::is_in_state_vector<BitInteger>{},
        std::forward<Qubit>(qubit), std::forward<Qubits>(qubits)...);
# endif // __cpp_constexpr >= 201603
    }
  } // namespace utility
} // namespace ket


#endif // KET_UTILITY_ALL_IN_STATE_VECTOR_HPP

