#ifndef KET_GATE_FUSED_GATE_HPP
# define KET_GATE_FUSED_GATE_HPP

# include <cassert>
# include <cstddef>
# include <array>
# include <utility>
# include <type_traits>

# include <ket/gate/gate.hpp>
# include <ket/meta/state_integer_of.hpp>
# include <ket/meta/bit_integer_of.hpp>
# include <ket/utility/integer_exp2.hpp>
# include <ket/utility/integer_log2.hpp>
# ifndef NDEBUG
#   include <ket/utility/all_in_state_vector.hpp>
# endif
# include <ket/utility/variadic/all_of.hpp>


namespace ket
{
  namespace gate
  {
    namespace fused
    {
      template <std::size_t num_fused_qubits, typename RandomAccessIterator, typename Function, typename Qubit, typename... Qubits>
      inline auto gate(RandomAccessIterator const first, Function&& function, Qubit&& qubit, Qubits&&... qubits)
      -> void
      {
        using state_integer_type = ::ket::meta::state_integer_t<std::remove_cv_t<std::remove_reference_t<Qubit>>>;
        static_assert(std::is_unsigned<state_integer_type>::value, "state_integer_type of Qubit should be unsigned");
# if __cpp_constexpr >= 201603L
        static_assert(
          ::ket::utility::variadic::proj::all_of(
            [](auto integer) { return std::is_same<decltype(integer), state_integer_type>::value; },
            [](auto qubit) { return ::ket::meta::state_integer_t<decltype(qubit)>{}; },
            std::remove_cv_t<std::remove_reference_t<Qubits>>{}...),
          "state_integer_type's of Qubit and Qubits should be the same");
# else // __cpp_constexpr >= 201603L
        static_assert(
          ::ket::utility::variadic::proj::all_of(
            ::ket::gate::gate_detail::is_same_to<state_integer_type>{}, ::ket::gate::gate_detail::state_integer_of{},
            std::remove_cv_t<std::remove_reference_t<Qubits>>{}...),
          "state_integer_type's of Qubit and Qubits should be the same");
# endif // __cpp_constexpr >= 201603L

        using bit_integer_type = ::ket::meta::bit_integer_t<std::remove_cv_t<std::remove_reference_t<Qubit>>>;
        static_assert(std::is_unsigned<bit_integer_type>::value, "bit_integer_type of Qubit should be unsigned");
# if __cpp_constexpr >= 201603L
        static_assert(
          ::ket::utility::variadic::proj::all_of(
            [](auto integer) { return std::is_same<decltype(integer), bit_integer_type>::value; },
            [](auto qubit) { return ::ket::meta::bit_integer_t<decltype(qubit)>{}; },
            std::remove_cv_t<std::remove_reference_t<Qubits>>{}...),
          "bit_integer_type's of Qubit and Qubits should be the same");
# else // __cpp_constexpr >= 201603L
        static_assert(
          ::ket::utility::variadic::proj::all_of(
            ::ket::gate::gate_detail::is_same_to<bit_integer_type>{}, ::ket::gate::gate_detail::bit_integer_of{},
            std::remove_cv_t<std::remove_reference_t<Qubits>>{}...),
          "bit_integer_type's of Qubit and Qubits should be the same");
# endif // __cpp_constexpr >= 201603L

        assert(::ket::utility::all_in_state_vector(static_cast<bit_integer_type>(num_fused_qubits), qubit, qubits...));

        constexpr auto num_operated_qubits = static_cast<bit_integer_type>(sizeof...(Qubits) + 1u);

        std::array<state_integer_type, num_operated_qubits> operated_qubit_masks{};
        ::ket::gate::gate_detail::make_qubit_masks(operated_qubit_masks, qubit, qubits...);
        std::array<state_integer_type, num_operated_qubits + 1u> operated_index_masks{};
        ::ket::gate::gate_detail::make_index_masks(operated_index_masks, std::forward<Qubit>(qubit), std::forward<Qubits>(qubits)...);

        constexpr auto count = ::ket::utility::integer_exp2<state_integer_type>(num_fused_qubits - num_operated_qubits);
        for (auto operated_index_wo_qubits = state_integer_type{0u}; operated_index_wo_qubits < count; ++operated_index_wo_qubits)
          function(first, operated_index_wo_qubits, operated_qubit_masks, operated_index_masks);
      }
    } // namespace fused
  } // namespace gate
} // namespace ket


#endif // KET_GATE_FUSED_GATE_HPP
