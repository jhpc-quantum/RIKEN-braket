#ifndef KET_UTILITY_ALL_IN_STATE_VECTOR_HPP
# define KET_UTILITY_ALL_IN_STATE_VECTOR_HPP

# include <algorithm>
# include <iterator>
# include <utility>
# include <type_traits>

# include <ket/qubit.hpp>
# include <ket/utility/variadic/all_of.hpp>


namespace ket
{
  namespace utility
  {
    namespace all_in_state_vector_detail
    {
      template <typename Qubit>
      constexpr auto wire_of(Qubit const qubit, int) -> decltype(remove_control(qubit))
      { return remove_control(qubit); }

      // Preserve support for raw positions and other existing wire types.
      template <typename Qubit>
      constexpr auto wire_of(Qubit const qubit, long) -> Qubit
      { return qubit; }

      template <typename BitInteger, typename Qubit>
      constexpr auto is_wire_in_state_vector(BitInteger const num_qubits, Qubit const qubit) -> bool
      {
        // Bounds depend on the wire, not its control value. ADL also supports
        // MPI's permutated operands without coupling this utility to MPI.
        auto const wire = ::ket::utility::all_in_state_vector_detail::wire_of(qubit, 0);
        using wire_type = std::decay_t<decltype(wire)>;
        return wire < wire_type{num_qubits};
      }
    }

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
        { return ::ket::utility::all_in_state_vector_detail::is_wire_in_state_vector(num_qubits_, std::forward<Qubit>(qubit)); }
      }; // struct is_in_state_vector<BitInteger>
    } // namespace all_in_state_vector_detail

# endif // __cpp_constexpr >= 201603L
    template <typename BitInteger, typename... Qubits>
    inline constexpr auto all_in_state_vector(BitInteger const num_qubits, Qubits&&... qubits) -> bool
    {
# if __cpp_constexpr >= 201603L
#   if __cpp_generic_lambdas >= 201707L
      return ::ket::utility::variadic::all_of(
        [num_qubits]<typename Qubit_>(Qubit_&& qubit) { return ::ket::utility::all_in_state_vector_detail::is_wire_in_state_vector(num_qubits, std::forward<Qubit_>(qubit)); },
        std::forward<Qubits>(qubits)...);
#   else // __cpp_generic_lambdas >= 201707L
      return ::ket::utility::variadic::all_of(
        [num_qubits](auto&& qubit) { return ::ket::utility::all_in_state_vector_detail::is_wire_in_state_vector(num_qubits, std::forward<decltype(qubit)>(qubit)); },
        std::forward<Qubits>(qubits)...);
#   endif // __cpp_generic_lambdas >= 201707L
# else // __cpp_constexpr >= 201603L
      return ::ket::utility::variadic::all_of(
        ::ket::utility::all_in_state_vector_detail::is_in_state_vector<BitInteger>{num_qubits},
        std::forward<Qubits>(qubits)...);
# endif // __cpp_constexpr >= 201603L
    }

    namespace runtime
    {
      template <typename BitInteger, typename QubitIterator>
      inline constexpr auto all_in_state_vector(BitInteger const num_qubits, QubitIterator const qubit_first, QubitIterator const qubit_last) -> bool
      {
        using qubit_type = typename std::iterator_traits<QubitIterator>::value_type;
        return std::all_of(qubit_first, qubit_last, [num_qubits](qubit_type const qubit) {
          return ::ket::utility::all_in_state_vector_detail::is_wire_in_state_vector(num_qubits, qubit);
        });
      }

      template <typename BitInteger, typename StateInteger, typename ControlQubitIterator>
      inline constexpr auto all_in_state_vector(
        BitInteger const num_qubits,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ControlQubitIterator const control_qubit_first, ControlQubitIterator const control_qubit_last) -> bool
      {
        using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
        if (target_qubit >= qubit_type{num_qubits})
          return false;

        return ::ket::utility::runtime::all_in_state_vector(num_qubits, control_qubit_first, control_qubit_last);
      }

      template <typename BitInteger, typename QubitIterator, typename ControlQubitIterator>
      inline constexpr auto all_in_state_vector(
        BitInteger const num_qubits,
        QubitIterator const target_qubit_first, QubitIterator const target_qubit_last,
        ControlQubitIterator const control_qubit_first, ControlQubitIterator const control_qubit_last) -> bool
      {
        return ::ket::utility::runtime::all_in_state_vector(num_qubits, target_qubit_first, target_qubit_last)
          and ::ket::utility::runtime::all_in_state_vector(num_qubits, control_qubit_first, control_qubit_last);
      }

      namespace ranges
      {
        template <typename BitInteger, typename QubitsRange>
        inline constexpr auto all_in_state_vector(BitInteger const num_qubits, QubitsRange const& qubits) -> bool
        {
          using std::begin;
          using std::end;
          return ::ket::utility::runtime::all_in_state_vector(num_qubits, begin(qubits), end(qubits));
        }

        template <typename BitInteger, typename StateInteger, typename ControlQubitsRange>
        inline constexpr auto all_in_state_vector(
          BitInteger const num_qubits,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ControlQubitsRange const& control_qubits) -> bool
        {
          using std::begin;
          using std::end;
          return ::ket::utility::runtime::all_in_state_vector(num_qubits, target_qubit, begin(control_qubits), end(control_qubits));
        }

        template <typename BitInteger, typename QubitsRange, typename ControlQubitsRange>
        inline constexpr auto all_in_state_vector(
          BitInteger const num_qubits, QubitsRange const& target_qubits, ControlQubitsRange const& control_qubits) -> bool
        {
          using std::begin;
          using std::end;
          return ::ket::utility::runtime::all_in_state_vector(
            num_qubits, begin(target_qubits), end(target_qubits), begin(control_qubits), end(control_qubits));
        }
      } // namespace ranges
    } // namespace runtime
  } // namespace utility
} // namespace ket


#endif // KET_UTILITY_ALL_IN_STATE_VECTOR_HPP

