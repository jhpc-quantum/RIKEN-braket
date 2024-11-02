#ifndef KET_UTILITY_ALL_IN_STATE_VECTOR_HPP
# define KET_UTILITY_ALL_IN_STATE_VECTOR_HPP

# include <utility>
# include <type_traits>

# include <ket/utility/variadic/all_of.hpp>


namespace ket
{
  namespace utility
  {
    template <typename BitInteger, typename Qubit, typename... Qubits>
    inline auto all_in_state_vector(BitInteger const num_qubits, Qubit&& qubit, Qubits&&... qubits) -> bool
    {
      return ::ket::utility::variadic::all_of(
        [num_qubits](auto&& qubit) { return qubit < std::remove_cv_t<std::remove_reference_t<decltype(qubit)>>{num_qubits}; },
        std::forward<Qubit>(qubit), std::forward<Qubits>(qubits)...);
    }
  } // namespace utility
} // namespace ket


#endif // KET_UTILITY_ALL_IN_STATE_VECTOR_HPP

