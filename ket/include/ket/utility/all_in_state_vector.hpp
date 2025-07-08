#ifndef KET_UTILITY_ALL_IN_STATE_VECTOR_HPP
# define KET_UTILITY_ALL_IN_STATE_VECTOR_HPP

# include <utility>
# include <type_traits>

# include <ket/utility/variadic/all_of.hpp>


namespace ket
{
  namespace utility
  {
# if __cpp_constexpr < 201603L
    namespace all_in_state_vector_detail
    {
      template <typename BitInteger>
      struct is_in_state_vector
      {
        BitInteger num_qubits_;

        is_in_state_vector(BitInteger const num_qubits) : num_qubits_{num_qubits} { }

        template <typename Qubit>
        constexpr auto operator()(Qubit&& qubit) const noexcept
        { return std::forward<Qubit>(qubit) < std::remove_cv_t<std::remove_reference_t<decltype(qubit)>>{num_qubits_}; }
      }; // struct is_in_state_vector<BitInteger>
    } // namespace all_in_state_vector_detail

# endif // __cpp_constexpr >= 201603L
    template <typename BitInteger, typename... Qubits>
    inline constexpr auto all_in_state_vector(BitInteger const num_qubits, Qubits&&... qubits) -> bool
    {
# if __cpp_constexpr >= 201603L
#   if __cpp_generic_lambdas >= 201707L
      return ::ket::utility::variadic::all_of(
        [num_qubits]<typename Qubit_>(Qubit_&& qubit) { return std::forward<Qubit_>(qubit) < std::remove_cv_t<std::remove_reference_t<Qubit_>>{num_qubits}; },
        std::forward<Qubits>(qubits)...);
#   else // __cpp_generic_lambdas >= 201707L
      return ::ket::utility::variadic::all_of(
        [num_qubits](auto&& qubit) { return std::forward<decltype(qubit)>(qubit) < std::remove_cv_t<std::remove_reference_t<decltype(qubit)>>{num_qubits}; },
        std::forward<Qubits>(qubits)...);
#   endif // __cpp_generic_lambdas >= 201707L
# else // __cpp_constexpr >= 201603L
      return ::ket::utility::variadic::all_of(
        ::ket::utility::all_in_state_vector_detail::is_in_state_vector<BitInteger>{num_qubits},
        std::forward<Qubits>(qubits)...);
# endif // __cpp_constexpr >= 201603L
    }
  } // namespace utility
} // namespace ket


#endif // KET_UTILITY_ALL_IN_STATE_VECTOR_HPP

