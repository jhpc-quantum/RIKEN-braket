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
      template <
        typename RandomAccessIterator, typename StateInteger, std::size_t num_indices,
        typename Function, typename Qubit, typename... Qubits>
      inline auto gate(
        RandomAccessIterator const first, std::array<StateInteger, num_indices> const& indices,
        Function&& function, Qubit&& qubit, Qubits&&... qubits)
      -> void
      {
        static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
# if __cpp_constexpr >= 201603L
        static_assert(
          ::ket::utility::variadic::proj::all_of(
            [](auto integer) { return std::is_same<decltype(integer), StateInteger>::value; },
            [](auto qubit) { return ::ket::meta::state_integer_t<decltype(qubit)>{}; },
            std::remove_cv_t<std::remove_reference_t<Qubit>>{}, std::remove_cv_t<std::remove_reference_t<Qubits>>{}...),
          "state_integer_type's of Qubit and Qubits should be the same to StateInteger");
# else // __cpp_constexpr >= 201603L
        static_assert(
          ::ket::utility::variadic::proj::all_of(
            ::ket::gate::gate_detail::is_same_to<StateInteger>{}, ::ket::gate::gate_detail::state_integer_of{},
            std::remove_cv_t<std::remove_reference_t<Qubit>>{}, std::remove_cv_t<std::remove_reference_t<Qubits>>{}...),
          "state_integer_type's of Qubit and Qubits should be the same to StateInteger");
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

        constexpr auto num_operated_qubits = static_cast<bit_integer_type>(sizeof...(Qubits) + 1u);
        constexpr auto num_fused_qubits = ::ket::utility::integer_log2<bit_integer_type>(num_indices);
        static_assert(::ket::utility::integer_exp2<StateInteger>(num_fused_qubits) == num_indices, "num_indices should be the power of num_fused_qubits");
        assert(::ket::utility::all_in_state_vector(num_fused_qubits, qubit, qubits...));

        std::array<StateInteger, num_operated_qubits> qubit_masks{};
        ::ket::gate::gate_detail::make_qubit_masks(qubit_masks, qubit, qubits...);
        std::array<StateInteger, num_operated_qubits + 1u> index_masks{};
        ::ket::gate::gate_detail::make_index_masks(index_masks, std::forward<Qubit>(qubit), std::forward<Qubits>(qubits)...);

        std::array<StateInteger, ::ket::utility::integer_exp2<std::size_t>(num_operated_qubits)> index_map;
        constexpr auto count = ::ket::utility::integer_exp2<StateInteger>(num_fused_qubits - num_operated_qubits);
        for (auto index_wo_qubits = StateInteger{0u}; index_wo_qubits < count; ++index_wo_qubits)
        {
          // ex. qubit_masks[0]=00100; qubit_masks[1]=10000; qubit_masks[2]=00001;
          //  index_map[0b000]=0x0x0; index_map[0b001]=0x1x0; index_map[0b010]=1x0x0; index_map[0b011]=1x1x0;
          //  index_map[0b100]=0x0x1; index_map[0b101]=0x1x1; index_map[0b110]=1x0x1; index_map[0b111]=1x1x1;
          // Usage: auto iter010 = first + indices[index_map[0b010]];
          ::ket::gate::gate_detail::make_indices(index_map, index_wo_qubits, qubit_masks, index_masks);
          function(first, indices, index_map);
        }
      }
    } // namespace fused
  } // namespace gate
} // namespace ket


#endif // KET_GATE_FUSED_GATE_HPP
