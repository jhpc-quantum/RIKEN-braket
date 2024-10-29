#ifndef KET_UTILITY_ALL_IN_STATE_VECTOR_HPP
# define KET_UTILITY_ALL_IN_STATE_VECTOR_HPP

# include <iterator>
# include <utility>
# include <type_traits>

# include <ket/utility/integer_log2.hpp>
# ifndef NDEBUG
#   include <ket/utility/integer_exp2.hpp>
# endif // NDEBUG
# include <ket/utility/variadic/all_of.hpp>
# include <ket/meta/state_integer_of.hpp>
# include <ket/meta/bit_integer_of.hpp>


namespace ket
{
  namespace utility
  {
    template <typename RandomAccessIterator, typename Qubit, typename... Qubits>
    inline auto all_in_state_vector(RandomAccessIterator const first, RandomAccessIterator const last, Qubit&& qubit, Qubits&&... qubits) -> bool
    {
      using state_integer_type = ::ket::meta::state_integer_t<std::remove_cv_t<std::remove_reference_t<Qubit>>>;
      using bit_integer_type = ::ket::meta::bit_integer_t<std::remove_cv_t<std::remove_reference_t<Qubit>>>;
      static_assert(std::is_unsigned<state_integer_type>::value, "StateInteger should be unsigned");
      static_assert(std::is_unsigned<bit_integer_type>::value, "BitInteger should be unsigned");

      auto const count = static_cast<state_integer_type>(last - first);
      auto const num_qubits = ::ket::utility::integer_log2<bit_integer_type>(count);
      assert(::ket::utility::integer_exp2<state_integer_type>(num_qubits) == count);

      return ::ket::utility::variadic::all_of(
        [num_qubits](auto&& qubit) { return qubit < std::remove_cv_t<std::remove_reference_t<decltype(qubit)>>{num_qubits}; },
        std::forward<Qubit>(qubit), std::forward<Qubits>(qubits)...);
    }

    namespace range
    {
      template <typename RandomAccessRange, typename Qubit, typename... Qubits>
      inline auto all_in_state_vector(RandomAccessRange const& state, Qubit&& qubit, Qubits&&... qubits) -> bool
      {
        using std::begin;
        using std::end;
        return ::ket::utility::all_in_state_vector(begin(state), end(state), std::forward<Qubit>(qubit), std::forward<Qubits>(qubits)...);
      }
    } // namespace range
  } // namespace utility
} // namespace ket


#endif // KET_UTILITY_ALL_IN_STATE_VECTOR_HPP

