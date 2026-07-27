#ifndef KET_GATE_FUSED_GATE_HPP
# define KET_GATE_FUSED_GATE_HPP

# include <cassert>
# include <cstddef>
# include <array>
# include <vector>
# ifndef KET_USE_BIT_MASKS_EXPLICITLY
#   include <algorithm>
# endif // KET_USE_BIT_MASKS_EXPLICITLY
# include <iterator>
# include <utility>
# include <type_traits>

# include <boost/range/iterator_range.hpp>

# ifndef KET_USE_BIT_MASKS_EXPLICITLY
#   include <ket/qubit.hpp>
# endif // KET_USE_BIT_MASKS_EXPLICITLY
# include <ket/gate/gate.hpp>
# include <ket/meta/state_integer_of.hpp>
# include <ket/meta/bit_integer_of.hpp>
# include <ket/utility/integer_exp2.hpp>
# include <ket/utility/meta/ranges.hpp>
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
# ifndef KET_USE_BIT_MASKS_EXPLICITLY
      template <std::size_t num_fused_qubits, typename RandomAccessIterator, typename Function, typename Qubit>
      inline auto gate(RandomAccessIterator const first, Function&& function, Qubit&& qubit)
      -> void
      {
        using state_integer_type = ::ket::meta::state_integer_t<std::remove_cv_t<std::remove_reference_t<Qubit>>>;
        static_assert(std::is_unsigned<state_integer_type>::value, "state_integer_type of Qubit should be unsigned");

        using bit_integer_type = ::ket::meta::bit_integer_t<std::remove_cv_t<std::remove_reference_t<Qubit>>>;
        static_assert(std::is_unsigned<bit_integer_type>::value, "bit_integer_type of Qubit should be unsigned");

        auto const sentinel_qubit = ::ket::make_qubit<state_integer_type>(static_cast<bit_integer_type>(num_fused_qubits));
        assert(qubit < sentinel_qubit);

        using qubit_type = ::ket::qubit<state_integer_type, bit_integer_type>;
        constexpr auto num_operated_qubits = bit_integer_type{1u};
        static_assert(num_fused_qubits >= num_operated_qubits, "num_fused_qubits should be greater than or equal to the number of operated qubits");
        std::array<qubit_type, num_operated_qubits + bit_integer_type{1u}> sorted_operated_qubits_with_sentinel{::ket::remove_control(qubit), sentinel_qubit};

        std::array<qubit_type, num_operated_qubits> unsorted_operated_qubits{::ket::remove_control(std::forward<Qubit>(qubit))};

        constexpr auto count = ::ket::utility::integer_exp2<state_integer_type>(num_fused_qubits - num_operated_qubits);
        for (auto operated_index_wo_qubits = state_integer_type{0u}; operated_index_wo_qubits < count; ++operated_index_wo_qubits)
          function(first, operated_index_wo_qubits, unsorted_operated_qubits, sorted_operated_qubits_with_sentinel);
      }

      template <std::size_t num_fused_qubits, typename RandomAccessIterator, typename Function, typename Qubit1, typename Qubit2>
      inline auto gate(RandomAccessIterator const first, Function&& function, Qubit1&& qubit1, Qubit2&& qubit2)
      -> void
      {
        using state_integer_type = ::ket::meta::state_integer_t<std::remove_cv_t<std::remove_reference_t<Qubit1>>>;
        static_assert(std::is_unsigned<state_integer_type>::value, "state_integer_type of Qubit1 should be unsigned");
        static_assert(
          std::is_same< ::ket::meta::state_integer_t<std::remove_cv_t<std::remove_reference_t<Qubit2>>>, state_integer_type >::value,
          "state_integer_type's of Qubit1 and Qubit2 should be the same");

        using bit_integer_type = ::ket::meta::bit_integer_t<std::remove_cv_t<std::remove_reference_t<Qubit1>>>;
        static_assert(std::is_unsigned<bit_integer_type>::value, "bit_integer_type of Qubit1 should be unsigned");
        static_assert(
          std::is_same< ::ket::meta::bit_integer_t<std::remove_cv_t<std::remove_reference_t<Qubit2>>>, bit_integer_type >::value,
          "bit_integer_type's of Qubit1 and Qubit2 should be the same");

        auto const sentinel_qubit = ::ket::make_qubit<state_integer_type>(static_cast<bit_integer_type>(num_fused_qubits));
        assert(qubit1 < sentinel_qubit);
        assert(qubit2 < sentinel_qubit);
        assert(qubit1 != qubit2);

        using qubit_type = ::ket::qubit<state_integer_type, bit_integer_type>;
        constexpr auto num_operated_qubits = bit_integer_type{2u};
        static_assert(num_fused_qubits >= num_operated_qubits, "num_fused_qubits should be greater than or equal to the number of operated qubits");
        auto const minmax_qubits = std::minmax(qubit1, qubit2);
        std::array<qubit_type, num_operated_qubits + bit_integer_type{1u}> sorted_operated_qubits_with_sentinel{
          ::ket::remove_control(minmax_qubits.first), ::ket::remove_control(minmax_qubits.second), sentinel_qubit};

        std::array<qubit_type, num_operated_qubits> unsorted_operated_qubits{
          ::ket::remove_control(std::forward<Qubit1>(qubit1)), ::ket::remove_control(std::forward<Qubit2>(qubit2))};

        constexpr auto count = ::ket::utility::integer_exp2<state_integer_type>(num_fused_qubits - num_operated_qubits);
        for (auto operated_index_wo_qubits = state_integer_type{0u}; operated_index_wo_qubits < count; ++operated_index_wo_qubits)
          function(first, operated_index_wo_qubits, unsorted_operated_qubits, sorted_operated_qubits_with_sentinel);
      }

      template <std::size_t num_fused_qubits, typename RandomAccessIterator, typename Function, typename Qubit1, typename Qubit2, typename Qubit3, typename... Qubits>
      inline auto gate(RandomAccessIterator const first, Function&& function, Qubit1&& qubit1, Qubit2&& qubit2, Qubit3&& qubit3, Qubits&&... qubits)
      -> void
      {
        using state_integer_type = ::ket::meta::state_integer_t<std::remove_cv_t<std::remove_reference_t<Qubit1>>>;
        static_assert(std::is_unsigned<state_integer_type>::value, "state_integer_type of Qubit1 should be unsigned");
#   if __cpp_constexpr >= 201603L
        static_assert(
          ::ket::utility::variadic::proj::all_of(
            [](auto integer) { return std::is_same<decltype(integer), state_integer_type>::value; },
            [](auto qubit) { return ::ket::meta::state_integer_t<decltype(qubit)>{}; },
            std::remove_cv_t<std::remove_reference_t<Qubit2>>{}, std::remove_cv_t<std::remove_reference_t<Qubit3>>{},
            std::remove_cv_t<std::remove_reference_t<Qubits>>{}...),
          "state_integer_type's of Qubit1, Qubit2, Qubit3 and Qubits should be the same");
#   else // __cpp_constexpr >= 201603L
        static_assert(
          ::ket::utility::variadic::proj::all_of(
            ::ket::gate::gate_detail::is_same_to<state_integer_type>{}, ::ket::gate::gate_detail::state_integer_of{},
            std::remove_cv_t<std::remove_reference_t<Qubit2>>{}, std::remove_cv_t<std::remove_reference_t<Qubit3>>{},
            std::remove_cv_t<std::remove_reference_t<Qubits>>{}...),
          "state_integer_type's of Qubit1, Qubit2, Qubit3 and Qubits should be the same");
#   endif // __cpp_constexpr >= 201603L

        using bit_integer_type = ::ket::meta::bit_integer_t<std::remove_cv_t<std::remove_reference_t<Qubit1>>>;
        static_assert(std::is_unsigned<bit_integer_type>::value, "bit_integer_type of Qubit1 should be unsigned");
#   if __cpp_constexpr >= 201603L
        static_assert(
          ::ket::utility::variadic::proj::all_of(
            [](auto integer) { return std::is_same<decltype(integer), bit_integer_type>::value; },
            [](auto qubit) { return ::ket::meta::bit_integer_t<decltype(qubit)>{}; },
            std::remove_cv_t<std::remove_reference_t<Qubit2>>{}, std::remove_cv_t<std::remove_reference_t<Qubit3>>{},
            std::remove_cv_t<std::remove_reference_t<Qubits>>{}...),
          "bit_integer_type's of Qubit1, Qubit2, Qubit3 and Qubits should be the same");
#   else // __cpp_constexpr >= 201603L
        static_assert(
          ::ket::utility::variadic::proj::all_of(
            ::ket::gate::gate_detail::is_same_to<bit_integer_type>{}, ::ket::gate::gate_detail::bit_integer_of{},
            std::remove_cv_t<std::remove_reference_t<Qubit2>>{}, std::remove_cv_t<std::remove_reference_t<Qubit3>>{},
            std::remove_cv_t<std::remove_reference_t<Qubits>>{}...),
          "bit_integer_type's of Qubit1, Qubit2, Qubit3 and Qubits should be the same");
#   endif // __cpp_constexpr >= 201603L

        assert(::ket::utility::all_in_state_vector(static_cast<bit_integer_type>(num_fused_qubits), qubit1, qubit2, qubit3, qubits...));

        using qubit_type = ::ket::qubit<state_integer_type, bit_integer_type>;
        constexpr auto num_operated_qubits = static_cast<bit_integer_type>(sizeof...(Qubits) + 3u);
        static_assert(num_fused_qubits >= num_operated_qubits, "num_fused_qubits should be greater than or equal to the number of operated qubits");
        std::array<qubit_type, num_operated_qubits + bit_integer_type{1u}> sorted_operated_qubits_with_sentinel{
          ::ket::remove_control(qubit1), ::ket::remove_control(qubit2),
          ::ket::remove_control(qubit3), ::ket::remove_control(qubits)...,
          ::ket::make_qubit<state_integer_type>(static_cast<bit_integer_type>(num_fused_qubits))};
        using std::begin;
        using std::end;
        std::sort(begin(sorted_operated_qubits_with_sentinel), std::prev(end(sorted_operated_qubits_with_sentinel)));

        std::array<qubit_type, num_operated_qubits> unsorted_operated_qubits{
          ::ket::remove_control(std::forward<Qubit1>(qubit1)), ::ket::remove_control(std::forward<Qubit2>(qubit2)),
          ::ket::remove_control(std::forward<Qubit3>(qubit3)), ::ket::remove_control(std::forward<Qubits>(qubits))...};

        constexpr auto count = ::ket::utility::integer_exp2<state_integer_type>(num_fused_qubits - num_operated_qubits);
        for (auto operated_index_wo_qubits = state_integer_type{0u}; operated_index_wo_qubits < count; ++operated_index_wo_qubits)
          function(first, operated_index_wo_qubits, unsorted_operated_qubits, sorted_operated_qubits_with_sentinel);
      }


      namespace runtime
      {
        namespace ranges
        {
          template <typename RandomAccessIterator, typename BitInteger, typename Function, typename QubitsRange>
          inline auto gate(RandomAccessIterator const first, BitInteger const num_fused_qubits, Function&& function, QubitsRange const& qubits)
          -> void
          {
            using qubit_type = ::ket::utility::meta::range_value_t<QubitsRange>;

            using state_integer_type = ::ket::meta::state_integer_t<qubit_type>;
            static_assert(std::is_unsigned<state_integer_type>::value, "The state_integer_type of value_type of QubitsRange should be unsigned");

            using bit_integer_type = ::ket::meta::bit_integer_t<qubit_type>;
            static_assert(std::is_unsigned<bit_integer_type>::value, "The bit_integer_type of value_type of QubitsRange should be unsigned");

            static_assert(std::is_same<qubit_type, ::ket::qubit<state_integer_type, bit_integer_type>>::value, "The value_type of QubitsRange should be the same as ::ket::qubit<S,B>");

            using std::begin;
            using std::end;
            auto const num_operated_qubits = static_cast<bit_integer_type>(end(qubits) - begin(qubits));
            assert(num_operated_qubits <= num_fused_qubits);

            assert(::ket::utility::runtime::ranges::all_in_state_vector(num_fused_qubits, qubits));

            auto unsorted_operated_qubits = std::vector<qubit_type>(begin(qubits), end(qubits));
            auto sorted_operated_qubits_with_sentinel = std::vector<qubit_type>{};
            sorted_operated_qubits_with_sentinel.reserve(num_operated_qubits + bit_integer_type{1u});
            sorted_operated_qubits_with_sentinel.assign(begin(qubits), end(qubits));
            std::sort(begin(sorted_operated_qubits_with_sentinel), end(sorted_operated_qubits_with_sentinel));
            sorted_operated_qubits_with_sentinel.push_back(qubit_type{num_fused_qubits});

            auto const count = ::ket::utility::integer_exp2<state_integer_type>(num_fused_qubits - num_operated_qubits);
            for (auto operated_index_wo_qubits = state_integer_type{0u}; operated_index_wo_qubits < count; ++operated_index_wo_qubits)
              function(first, operated_index_wo_qubits, unsorted_operated_qubits, sorted_operated_qubits_with_sentinel);
          }
        } // namespace ranges

        template <typename RandomAccessIterator, typename BitInteger, typename Function, typename QubitIterator>
        inline auto gate(RandomAccessIterator const first, BitInteger const num_fused_qubits, Function&& function, QubitIterator const qubit_first, QubitIterator const qubit_last)
        -> void
        { ::ket::gate::fused::runtime::ranges::gate(first, num_fused_qubits, std::forward<Function>(function), boost::make_iterator_range(qubit_first, qubit_last)); }
      } // namespace runtime
# else // KET_USE_BIT_MASKS_EXPLICITLY
      template <std::size_t num_fused_qubits, typename RandomAccessIterator, typename Function, typename Qubit, typename... Qubits>
      inline auto gate(RandomAccessIterator const first, Function&& function, Qubit&& qubit, Qubits&&... qubits)
      -> void
      {
        using state_integer_type = ::ket::meta::state_integer_t<std::remove_cv_t<std::remove_reference_t<Qubit>>>;
        static_assert(std::is_unsigned<state_integer_type>::value, "state_integer_type of Qubit should be unsigned");
#   if __cpp_constexpr >= 201603L
        static_assert(
          ::ket::utility::variadic::proj::all_of(
            [](auto integer) { return std::is_same<decltype(integer), state_integer_type>::value; },
            [](auto qubit) { return ::ket::meta::state_integer_t<decltype(qubit)>{}; },
            std::remove_cv_t<std::remove_reference_t<Qubits>>{}...),
          "state_integer_type's of Qubit and Qubits should be the same");
#   else // __cpp_constexpr >= 201603L
        static_assert(
          ::ket::utility::variadic::proj::all_of(
            ::ket::gate::gate_detail::is_same_to<state_integer_type>{}, ::ket::gate::gate_detail::state_integer_of{},
            std::remove_cv_t<std::remove_reference_t<Qubits>>{}...),
          "state_integer_type's of Qubit and Qubits should be the same");
#   endif // __cpp_constexpr >= 201603L

        using bit_integer_type = ::ket::meta::bit_integer_t<std::remove_cv_t<std::remove_reference_t<Qubit>>>;
        static_assert(std::is_unsigned<bit_integer_type>::value, "bit_integer_type of Qubit should be unsigned");
#   if __cpp_constexpr >= 201603L
        static_assert(
          ::ket::utility::variadic::proj::all_of(
            [](auto integer) { return std::is_same<decltype(integer), bit_integer_type>::value; },
            [](auto qubit) { return ::ket::meta::bit_integer_t<decltype(qubit)>{}; },
            std::remove_cv_t<std::remove_reference_t<Qubits>>{}...),
          "bit_integer_type's of Qubit and Qubits should be the same");
#   else // __cpp_constexpr >= 201603L
        static_assert(
          ::ket::utility::variadic::proj::all_of(
            ::ket::gate::gate_detail::is_same_to<bit_integer_type>{}, ::ket::gate::gate_detail::bit_integer_of{},
            std::remove_cv_t<std::remove_reference_t<Qubits>>{}...),
          "bit_integer_type's of Qubit and Qubits should be the same");
#   endif // __cpp_constexpr >= 201603L

        assert(::ket::utility::all_in_state_vector(static_cast<bit_integer_type>(num_fused_qubits), qubit, qubits...));

        constexpr auto num_operated_qubits = static_cast<bit_integer_type>(sizeof...(Qubits) + 1u);
        static_assert(num_fused_qubits >= num_operated_qubits, "num_fused_qubits should be greater than or equal to the number of operated qubits");

        std::array<state_integer_type, num_operated_qubits> operated_qubit_masks{};
        ::ket::gate::gate_detail::make_qubit_masks(operated_qubit_masks, qubit, qubits...);
        std::array<state_integer_type, num_operated_qubits + 1u> operated_index_masks{};
        ::ket::gate::gate_detail::make_index_masks(operated_index_masks, std::forward<Qubit>(qubit), std::forward<Qubits>(qubits)...);

        constexpr auto count = ::ket::utility::integer_exp2<state_integer_type>(num_fused_qubits - num_operated_qubits);
        for (auto operated_index_wo_qubits = state_integer_type{0u}; operated_index_wo_qubits < count; ++operated_index_wo_qubits)
          function(first, operated_index_wo_qubits, operated_qubit_masks, operated_index_masks);
      }


      namespace runtime
      {
        namespace ranges
        {
          template <typename RandomAccessIterator, typename BitInteger, typename Function, typename QubitsRange>
          inline auto gate(RandomAccessIterator const first, BitInteger const num_fused_qubits, Function&& function, QubitsRange const& qubits)
          -> void
          {
            using qubit_type = ::ket::utility::meta::range_value_t<QubitsRange>;

            using state_integer_type = ::ket::meta::state_integer_t<qubit_type>;
            static_assert(std::is_unsigned<state_integer_type>::value, "The state_integer_type of value_type of QubitsRange should be unsigned");

            using bit_integer_type = ::ket::meta::bit_integer_t<qubit_type>;
            static_assert(std::is_unsigned<bit_integer_type>::value, "The bit_integer_type of value_type of QubitsRange should be unsigned");

            static_assert(std::is_same<qubit_type, ::ket::qubit<state_integer_type, bit_integer_type>>::value, "The value_type of QubitsRange should be the same as ::ket::qubit<S,B>");

            using std::begin;
            using std::end;
            auto const num_operated_qubits = static_cast<bit_integer_type>(end(qubits) - begin(qubits));
            assert(num_operated_qubits <= num_fused_qubits);

            assert(::ket::utility::runtime::ranges::all_in_state_vector(num_fused_qubits, qubits));

            auto operated_qubit_masks = std::vector<state_integer_type>{};
            operated_qubit_masks.reserve(num_operated_qubits);
            ::ket::gate::gate_detail::runtime::ranges::make_qubit_masks(qubits, std::back_inserter(operated_qubit_masks));
            auto operated_index_masks = std::vector<state_integer_type>{};
            operated_index_masks.reserve(num_operated_qubits + bit_integer_type{1u});
            ::ket::gate::gate_detail::runtime::ranges::make_index_masks(qubits, std::back_inserter(operated_index_masks));

            auto const count = ::ket::utility::integer_exp2<state_integer_type>(num_fused_qubits - num_operated_qubits);
            for (auto operated_index_wo_qubits = state_integer_type{0u}; operated_index_wo_qubits < count; ++operated_index_wo_qubits)
              function(first, operated_index_wo_qubits, operated_qubit_masks, operated_index_masks);
          }
        } // namespace ranges

        template <typename RandomAccessIterator, typename BitInteger, typename Function, typename QubitIterator>
        inline auto gate(RandomAccessIterator const first, BitInteger const num_fused_qubits, Function&& function, QubitIterator const qubit_first, QubitIterator const qubit_last)
        -> void
        { ::ket::gate::fused::runtime::ranges::gate(first, num_fused_qubits, std::forward<Function>(function), boost::make_iterator_range(qubit_first, qubit_last)); }
      } // namespace runtime
# endif // KET_USE_BIT_MASKS_EXPLICITLY
    } // namespace fused
  } // namespace gate
} // namespace ket


#endif // KET_GATE_FUSED_GATE_HPP
