#ifndef KET_GATE_GATE_HPP
# define KET_GATE_GATE_HPP

# include <cassert>
# include <cstddef>
# include <tuple>
# include <array>
# include <vector>
# include <iterator>
# include <algorithm>
# include <numeric>
# include <utility>
# include <type_traits>

# include <ket/qubit.hpp>
# include <ket/control.hpp>
# include <ket/meta/state_integer_of.hpp>
# include <ket/meta/bit_integer_of.hpp>
# include <ket/utility/loop_n.hpp>
# include <ket/utility/contains.hpp>
# include <ket/utility/integer_exp2.hpp>
# include <ket/utility/integer_log2.hpp>
# if !defined(NDEBUG) || defined(KET_USE_ON_CACHE_STATE_VECTOR)
#   include <ket/utility/all_in_state_vector.hpp>
# endif
# include <ket/utility/variadic/transform.hpp>
# include <ket/utility/variadic/all_of.hpp>
# include <ket/utility/tuple/transform.hpp>
# include <ket/utility/tuple/all_of.hpp>
# include <ket/utility/tuple/to_array.hpp>


namespace ket
{
  namespace gate
  {
    namespace gate_detail
    {
      namespace runtime
      {
        template <typename InputIterator, typename OutputIterator>
        inline auto make_qubit_masks(InputIterator const qubits_first, InputIterator const qubits_last, OutputIterator const d_first) -> OutputIterator
        {
          using qubit_type = typename std::iterator_traits<InputIterator>::value_type;
          using state_integer_type = ::ket::meta::state_integer_t<qubit_type>;
          static_assert(std::is_unsigned<state_integer_type>::value, "state_integer_type of value_type of InputIterator should be unsigned");

          return std::transform(qubits_first, qubits_last, d_first, [](qubit_type const qubit) { return state_integer_type{1u} << qubit; });
        }

        namespace unsafe
        {
          template <typename Qubit, typename Allocator, typename OutputIterator>
          inline auto make_index_masks(std::vector<Qubit, Allocator> const& sorted_qubits, OutputIterator d_first) -> OutputIterator
          {
            using state_integer_type = ::ket::meta::state_integer_t<Qubit>;
            using bit_integer_type = ::ket::meta::bit_integer_t<Qubit>;
            static_assert(std::is_unsigned<state_integer_type>::value, "state_integer_type of Qubit should be unsigned");
            static_assert(std::is_unsigned<bit_integer_type>::value, "bit_integer_type of Qubit should be unsigned");
# ifndef NDEBUG
            using std::begin;
            using std::end;
# endif // NDEBUG
            assert(std::is_sorted(begin(sorted_qubits), end(sorted_qubits)));
            assert(not sorted_qubits.empty());

            auto previous_partial_sums = (state_integer_type{1u} << sorted_qubits.front()) - state_integer_type{1u};
            *d_first++ = previous_partial_sums;

            auto const num_operated_qubits = static_cast<bit_integer_type>(sorted_qubits.size());
            for (auto index = bit_integer_type{1u}; index < num_operated_qubits; ++index)
            {
              auto const partial_sums = (state_integer_type{1u} << (sorted_qubits[index] - index)) - state_integer_type{1u};
              *d_first++ = partial_sums xor previous_partial_sums;
              previous_partial_sums = partial_sums;
            }

            *d_first++ = (compl state_integer_type{0u}) xor previous_partial_sums;
            return d_first;
          }
        } // namespace unsafe

        template <typename InputIterator, typename OutputIterator>
        inline auto make_index_masks(InputIterator const qubits_first, InputIterator const qubits_last, OutputIterator d_first) -> OutputIterator
        {
          using qubit_type = typename std::iterator_traits<InputIterator>::value_type;
          if (qubits_first == qubits_last)
          {
            using state_integer_type = ::ket::meta::state_integer_t<qubit_type>;
            *d_first++ = compl state_integer_type{0u};
            return d_first;
          }

          auto sorted_qubits = std::vector<qubit_type>(qubits_first, qubits_last);
          using std::begin;
          using std::end;
          std::sort(begin(sorted_qubits), end(sorted_qubits));
          return ::ket::gate::gate_detail::runtime::unsafe::make_index_masks(sorted_qubits, d_first);
        }

        template <typename RandomAccessIterator, typename InputIterator, typename StateInteger, typename OutputIterator>
        inline auto make_indices(
          RandomAccessIterator const qubit_masks_first, RandomAccessIterator const qubit_masks_last,
          InputIterator index_masks_first, InputIterator const index_masks_last,
          StateInteger const index_wo_qubits, OutputIterator d_first)
        -> OutputIterator
        {
          static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
          static_assert(std::is_same<typename std::iterator_traits<RandomAccessIterator>::value_type, StateInteger>::value, "value_type of RandomAccessIterator and StateInteger should be the same");
          static_assert(std::is_same<typename std::iterator_traits<InputIterator>::value_type, StateInteger>::value, "value_type of InputIterator and StateInteger should be the same");
          assert(std::distance(qubit_masks_first, qubit_masks_last) + 1 == std::distance(index_masks_first, index_masks_last));

          if (qubit_masks_first == qubit_masks_last)
          {
            *d_first++ = index_wo_qubits;
            return d_first;
          }

          // xx0xx0xx0xx
          auto base_index = StateInteger{0u};
          auto const num_index_masks = std::distance(index_masks_first, index_masks_last);
          for (auto index_masks_index = decltype(num_index_masks){0}; index_masks_index < num_index_masks; ++index_masks_index)
            base_index |= (index_wo_qubits bitand *index_masks_first++) << index_masks_index;
          *d_first++ = base_index;

          auto const num_operated_qubits = std::distance(qubit_masks_first, qubit_masks_last);
          auto const num_indices = ::ket::utility::integer_exp2<std::size_t>(num_operated_qubits);
          for (auto n = std::size_t{1u}; n < num_indices; ++n)
          {
            auto index = base_index;
            for (auto qubit_masks_index = decltype(num_operated_qubits){0}; qubit_masks_index < num_operated_qubits; ++qubit_masks_index)
              if (((StateInteger{1u} << qubit_masks_index) bitand n) != StateInteger{0u})
                index |= qubit_masks_first[qubit_masks_index];
            *d_first++ = index;
          }

          return d_first;
        }

        namespace ranges
        {
          template <typename Range, typename OutputIterator>
          inline auto make_qubit_masks(Range&& qubits, OutputIterator const d_first) -> OutputIterator
          {
            using std::begin;
            using std::end;
            auto const qubits_last = end(qubits);
            return ::ket::gate::gate_detail::runtime::make_qubit_masks(begin(std::forward<Range>(qubits)), qubits_last, d_first);
          }

          template <typename Range, typename OutputIterator>
          inline auto make_index_masks(Range&& qubits, OutputIterator d_first) -> OutputIterator
          {
            using std::begin;
            using std::end;
            auto const qubits_last = end(qubits);
            return ::ket::gate::gate_detail::runtime::make_index_masks(begin(std::forward<Range>(qubits)), qubits_last, d_first);
          }

          template <typename RandomAccessRange, typename Range, typename StateInteger, typename ForwardIterator>
          inline auto make_indices(
            RandomAccessRange&& qubit_masks, Range&& index_masks,
            StateInteger const index_wo_qubits, ForwardIterator result)
          -> ForwardIterator
          {
            using std::begin;
            using std::end;
            auto const qubit_masks_last = end(qubit_masks);
            auto const index_masks_last = end(index_masks);
            return ::ket::gate::gate_detail::runtime::make_indices(
              begin(std::forward<RandomAccessRange>(qubit_masks)), qubit_masks_last,
              begin(std::forward<Range>(index_masks)), index_masks_last,
              index_wo_qubits, result);
          }
        } // namespace ranges
      } // namespace runtime

# if __cpp_constexpr < 201603
      template <typename U>
      struct is_same_to
      {
        template <typename T>
        constexpr auto operator()(T) const noexcept -> bool { return std::is_same<T, U>::value; }
      }; // struct is_same_to<U>

      struct is_unsigned
      {
        template <typename T>
        inline constexpr auto operator()(T) const noexcept -> bool { return std::is_unsigned<T>::value; }
      }; // struct is_unsigned

      struct state_integer_of
      {
        template <typename Qubit>
        constexpr auto operator()(Qubit) const noexcept { return ::ket::meta::state_integer_t<Qubit>{}; }
      }; // struct state_integer_of

      struct bit_integer_of
      {
        template <typename Qubit>
        constexpr auto operator()(Qubit) const noexcept { return ::ket::meta::bit_integer_t<Qubit>{}; }
      }; // struct bit_integer_of
# endif // __cpp_constexpr < 201603

      template <typename StateInteger, typename... Qubits>
      inline auto make_qubit_masks(std::array<StateInteger, sizeof...(Qubits)>& result, Qubits&&... qubits) -> void
      {
        static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
# if __cpp_constexpr >= 201603
        static_assert(
          ::ket::utility::variadic::proj::all_of(
            [](auto integer) { return std::is_same<decltype(integer), StateInteger>::value; },
            [](auto qubit) { return ::ket::meta::state_integer_t<decltype(qubit)>{}; },
            std::remove_cv_t<std::remove_reference_t<Qubits>>{}...),
          "state_integer_type's of Qubits should be the same to StateInteger");
# else // __cpp_constexpr >= 201603
        static_assert(
          ::ket::utility::variadic::proj::all_of(
            ::ket::gate::gate_detail::is_same_to<StateInteger>{}, ::ket::gate::gate_detail::state_integer_of{},
            std::remove_cv_t<std::remove_reference_t<Qubits>>{}...),
          "state_integer_type's of Qubits should be the same to StateInteger");
# endif // __cpp_constexpr >= 201603

        ::ket::utility::tuple::to_array(
          ::ket::utility::variadic::transform([](auto&& qubit) { return StateInteger{1u} << qubit; }, std::forward<Qubits>(qubits)...),
          result);
      }

      template <typename Qubits, typename StateInteger, std::size_t num_operated_qubits>
      inline auto make_qubit_masks(
        Qubits const& qubits,
        std::array<StateInteger, num_operated_qubits>& result)
      -> void
      {
        static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
# if __cpp_constexpr >= 201603
        static_assert(
          ::ket::utility::tuple::proj::all_of(
            Qubits{},
            [](auto integer) { return std::is_same<decltype(integer), StateInteger>::value; },
            [](auto qubit) { return ::ket::meta::state_integer_t<decltype(qubit)>{}; }),
          "state_integer_type's of all elements of Qubits should be the same to StateInteger");
# else // __cpp_constexpr >= 201603
        static_assert(
          ::ket::utility::tuple::proj::all_of(
            Qubits{}, ::ket::gate::gate_detail::is_same_to<StateInteger>{}, ::ket::gate::gate_detail::state_integer_of{}),
          "state_integer_type's of all elements of Qubits should be the same to StateInteger");
# endif // __cpp_constexpr >= 201603

        ::ket::utility::tuple::to_array(
          ::ket::utility::tuple::transform(qubits, [](auto const qubit) { return StateInteger{1u} << qubit; }),
          result);
      }

      template <typename Qubit, std::size_t num_operated_qubits, typename StateInteger>
      inline auto make_index_masks_impl(
        std::array<Qubit, num_operated_qubits> qubits_array, std::array<StateInteger, num_operated_qubits + 1u>& result)
      -> void
      {
        using std::begin;
        using std::end;
        std::sort(begin(qubits_array), end(qubits_array));

        for (auto index = 0u; index < num_operated_qubits; ++index)
          result[index] = (StateInteger{1u} << (qubits_array[index] - index)) - StateInteger{1u};
        result.back() = compl StateInteger{0u};

        std::adjacent_difference(begin(result), end(result), begin(result));
      }

      template <typename StateInteger, typename Qubit1, typename Qubit2, typename Qubit3, typename... Qubits>
      inline auto make_index_masks(
        std::array<StateInteger, sizeof...(Qubits) + 4u>& result,
        Qubit1&& qubit1, Qubit2&& qubit2, Qubit3&& qubit3, Qubits&&... qubits)
      -> void
      {
        static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
# if __cpp_constexpr >= 201603
        static_assert(
          ::ket::utility::variadic::proj::all_of(
            [](auto integer) { return std::is_same<decltype(integer), StateInteger>::value; },
            [](auto qubit) { return ::ket::meta::state_integer_t<decltype(qubit)>{}; },
            std::remove_cv_t<std::remove_reference_t<Qubit1>>{}, std::remove_cv_t<std::remove_reference_t<Qubit2>>{},
            std::remove_cv_t<std::remove_reference_t<Qubit3>>{}, std::remove_cv_t<std::remove_reference_t<Qubits>>{}...),
          "state_integer_type's of Qubit1, Qubit2, Qubit3, and Qubits should be same to StateInteger");
# else // __cpp_constexpr >= 201603
        static_assert(
          ::ket::utility::variadic::proj::all_of(
            ::ket::gate::gate_detail::is_same_to<StateInteger>{}, ::ket::gate::gate_detail::state_integer_of{},
            std::remove_cv_t<std::remove_reference_t<Qubit1>>{}, std::remove_cv_t<std::remove_reference_t<Qubit2>>{},
            std::remove_cv_t<std::remove_reference_t<Qubit3>>{}, std::remove_cv_t<std::remove_reference_t<Qubits>>{}...),
          "state_integer_type's of Qubit1, Qubit2, Qubit3, and Qubits should be same to StateInteger");
# endif // __cpp_constexpr >= 201603

        using bit_integer_type = ::ket::meta::bit_integer_t<std::remove_cv_t<std::remove_reference_t<Qubit1>>>;
        static_assert(std::is_unsigned<bit_integer_type>::value, "bit_integer_type of Qubit1 should be unsigned");
# if __cpp_constexpr >= 201603
        static_assert(
          ::ket::utility::variadic::proj::all_of(
            [](auto integer) { return std::is_same<decltype(integer), bit_integer_type>::value; },
            [](auto qubit) { return ::ket::meta::bit_integer_t<decltype(qubit)>{}; },
            std::remove_cv_t<std::remove_reference_t<Qubit2>>{}, std::remove_cv_t<std::remove_reference_t<Qubit3>>{},
            std::remove_cv_t<std::remove_reference_t<Qubits>>{}...),
          "bit_integer_type's of Qubit1, Qubit2, Qubit3, and Qubits should be the same");
# else // __cpp_constexpr >= 201603
        static_assert(
          ::ket::utility::variadic::proj::all_of(
            ::ket::gate::gate_detail::is_same_to<bit_integer_type>{}, ::ket::gate::gate_detail::bit_integer_of{},
            std::remove_cv_t<std::remove_reference_t<Qubit2>>{}, std::remove_cv_t<std::remove_reference_t<Qubit3>>{},
            std::remove_cv_t<std::remove_reference_t<Qubits>>{}...),
          "bit_integer_type's of Qubit1, Qubit2, Qubit3, and Qubits should be the same");
# endif // __cpp_constexpr >= 201603

        static constexpr auto num_operated_qubits = sizeof...(Qubits) + 3u;
        using qubit_type = ::ket::qubit<StateInteger, bit_integer_type>;
        std::array<qubit_type, num_operated_qubits> qubits_array{
          ::ket::remove_control(std::forward<Qubit1>(qubit1)), ::ket::remove_control(std::forward<Qubit2>(qubit2)),
          ::ket::remove_control(std::forward<Qubit3>(qubit3)), ::ket::remove_control(std::forward<Qubits>(qubits))...};
        ::ket::gate::gate_detail::make_index_masks_impl(qubits_array, result);
      }

      template <typename StateInteger, typename Qubit1, typename Qubit2>
      inline auto make_index_masks(std::array<StateInteger, 3u>& result, Qubit1&& qubit1, Qubit2&& qubit2) -> void
      {
        static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
# if __cpp_constexpr >= 201603
        static_assert(
          ::ket::utility::variadic::proj::all_of(
            [](auto integer) { return std::is_same<decltype(integer), StateInteger>::value; },
            [](auto qubit) { return ::ket::meta::state_integer_t<decltype(qubit)>{}; },
            std::remove_cv_t<std::remove_reference_t<Qubit1>>{}, std::remove_cv_t<std::remove_reference_t<Qubit2>>{}),
          "state_integer_type's of Qubit1, Qubit2, Qubit3, and Qubits should be same to StateInteger");
# else // __cpp_constexpr >= 201603
        static_assert(
          ::ket::utility::variadic::proj::all_of(
            ::ket::gate::gate_detail::is_same_to<StateInteger>{}, ::ket::gate::gate_detail::state_integer_of{},
            std::remove_cv_t<std::remove_reference_t<Qubit1>>{}, std::remove_cv_t<std::remove_reference_t<Qubit2>>{}),
          "state_integer_type's of Qubit1, Qubit2, Qubit3, and Qubits should be same to StateInteger");
# endif // __cpp_constexpr >= 201603

        auto const raw_qubit1 = ::ket::remove_control(std::forward<Qubit1>(qubit1));
        auto const raw_qubit2 = ::ket::remove_control(std::forward<Qubit2>(qubit2));
        auto const minmax_qubits = std::minmax(raw_qubit1, raw_qubit2);
        result[0u] = (StateInteger{1u} << minmax_qubits.first) - StateInteger{1u};
        result[1u] = ((StateInteger{1u} << (minmax_qubits.second - 1u)) - StateInteger{1u}) xor result[0u];
        result[2u] = compl (result[0u] bitor result[1u]);
      }

      template <typename StateInteger, typename Qubit>
      inline auto make_index_masks(std::array<StateInteger, 2u>& result, Qubit&& qubit) -> void
      {
        static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
        static_assert(
          std::is_same<StateInteger, ::ket::meta::state_integer_t<std::remove_cv_t<std::remove_reference_t<Qubit>>>>::value,
          "state_integer_type of Qubit should be same to StateInteger");

        result[0u] = (StateInteger{1u} << std::forward<Qubit>(qubit)) - StateInteger{1u};
        result[1u] = compl result[0u];
      }

      template <typename Qubits, typename StateInteger>
      inline auto make_index_masks(
        Qubits const& qubits,
        std::array<StateInteger, std::tuple_size<Qubits>::value + 1u>& result)
      -> void
      {
        static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
# if __cpp_constexpr >= 201603
        static_assert(
          ::ket::utility::tuple::proj::all_of(
            Qubits{},
            [](auto integer) { return std::is_same<decltype(integer), StateInteger>::value; },
            [](auto qubit) { return ::ket::meta::state_integer_t<decltype(qubit)>{}; }),
          "state_integer_type's of all elements of Qubits should be the same to StateInteger");
# else // __cpp_constexpr >= 201603
        static_assert(
          ::ket::utility::tuple::proj::all_of(
            Qubits{}, ::ket::gate::gate_detail::is_same_to<StateInteger>{}, ::ket::gate::gate_detail::state_integer_of{}),
          "state_integer_type's of all elements of Qubits should be the same to StateInteger");
# endif // __cpp_constexpr >= 201603

        using bit_integer_type = ::ket::meta::bit_integer_t<std::remove_cv_t<std::remove_reference_t<std::tuple_element_t<0u, Qubits>>>>;
        static_assert(std::is_unsigned<bit_integer_type>::value, "bit_integer_type of the first element of Qubits should be unsigned");
# if __cpp_constexpr >= 201603
        static_assert(
          ::ket::utility::tuple::proj::all_of(
            Qubits{},
            [](auto integer) { return std::is_same<decltype(integer), bit_integer_type>::value; },
            [](auto qubit) { return ::ket::meta::bit_integer_t<decltype(qubit)>{}; }),
          "bit_integer_type's of all elements of Qubits should be the same");
# else // __cpp_constexpr >= 201603
        static_assert(
          ::ket::utility::tuple::proj::all_of(
            Qubits{}, ::ket::gate::gate_detail::is_same_to<bit_integer_type>{}, ::ket::gate::gate_detail::bit_integer_of{}),
          "bit_integer_type's of all elements of Qubits should be the same");
# endif // __cpp_constexpr >= 201603

        using qubit_type = ::ket::qubit<StateInteger, bit_integer_type>;
        std::array<qubit_type, std::tuple_size<Qubits>::value> qubits_array;
        ::ket::utility::tuple::to_array(
          ::ket::utility::tuple::transform(qubits, [](auto const qubit) { return ::ket::remove_control(qubit); }),
          qubits_array);
        ::ket::gate::gate_detail::make_index_masks_impl(qubits_array, result);
      }

      template <typename Qubit1, typename Qubit2, typename StateInteger>
      inline auto make_index_masks(std::tuple<Qubit1&&, Qubit2&&> const qubits, std::array<StateInteger, 3u>& result) -> void
      {
        static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
# if __cpp_constexpr >= 201603
        static_assert(
          ::ket::utility::variadic::proj::all_of(
            [](auto integer) { return std::is_same<decltype(integer), StateInteger>::value; },
            [](auto qubit) { return ::ket::meta::state_integer_t<decltype(qubit)>{}; },
            std::remove_cv_t<std::remove_reference_t<Qubit1>>{}, std::remove_cv_t<std::remove_reference_t<Qubit2>>{}),
          "state_integer_type's of Qubit1 and Qubit2 should be same to StateInteger");
# else // __cpp_constexpr >= 201603
        static_assert(
          ::ket::utility::variadic::all_of(
            ::ket::gate::gate_detail::is_same_to<StateInteger>{}, ::ket::gate::gate_detail::state_integer_of{},
            std::remove_cv_t<std::remove_reference_t<Qubit1>>{}, std::remove_cv_t<std::remove_reference_t<Qubit2>>{}),
          "state_integer_type's of Qubit1 and Qubit2 should be same to StateInteger");
# endif // __cpp_constexpr >= 201603

        auto const raw_qubit1 = ::ket::remove_control(std::forward<Qubit1&&>(std::get<0u>(qubits)));
        auto const raw_qubit2 = ::ket::remove_control(std::forward<Qubit2&&>(std::get<1u>(qubits)));
        auto const minmax_qubits = std::minmax(raw_qubit1, raw_qubit2);
        result[0u] = (StateInteger{1u} << minmax_qubits.first) - StateInteger{1u};
        result[1u] = ((StateInteger{1u} << (minmax_qubits.second - 1u)) - StateInteger{1u}) xor result[0u];
        result[2u] = compl (result[0u] bitor result[1u]);
      }

      template <typename Qubit, typename StateInteger>
      inline auto make_index_masks(std::tuple<Qubit&&> const qubit, std::array<StateInteger, 2u>& result) -> void
      {
        static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
        static_assert(
          std::is_same<StateInteger, ::ket::meta::state_integer_t<std::remove_cv_t<std::remove_reference_t<Qubit>>>>::value,
          "state_integer_type of Qubit should be same to StateInteger");

        result[0u] = (StateInteger{1u} << std::forward<Qubit>(qubit)) - StateInteger{1u};
        result[1u] = compl result[0u];
      }

      template <typename StateInteger, std::size_t num_operated_qubits>
      inline constexpr auto make_indices(
        std::array<StateInteger, ::ket::utility::integer_exp2<std::size_t>(num_operated_qubits)>& result,
        StateInteger const index_wo_qubits,
        std::array<StateInteger, num_operated_qubits> const& qubit_masks,
        std::array<StateInteger, num_operated_qubits + 1u> const& index_masks)
      -> void
      {
        // xx0xx0xx0xx
        result[0u] = StateInteger{0u};
        for (auto index_mask_index = std::size_t{0u}; index_mask_index < num_operated_qubits + std::size_t{1u}; ++index_mask_index)
          result[0u] |= (index_wo_qubits bitand index_masks[index_mask_index]) << index_mask_index;

        constexpr auto num_indices = ::ket::utility::integer_exp2<std::size_t>(num_operated_qubits);
        for (auto n = std::size_t{1u}; n < num_indices; ++n)
        {
          result[n] = result[0u];
          for (auto qubit_index = std::size_t{0u}; qubit_index < num_operated_qubits; ++qubit_index)
            if (((StateInteger{1u} << qubit_index) bitand n) != StateInteger{0u})
              result[n] |= qubit_masks[qubit_index];
        }
      }

      template <typename StateInteger>
      inline constexpr auto make_indices(
        std::array<StateInteger, 2u>& result, StateInteger const index_wo_qubits,
        std::array<StateInteger, 1u> const& qubit_masks, std::array<StateInteger, 2u> const& index_masks)
      -> void
      {
        // xx0xx
        result[0u] = (index_wo_qubits bitand index_masks[0u]) bitor ((index_wo_qubits bitand index_masks[1u]) << 1u);
        // xx1xx
        result[1u] = result[0u] bitor qubit_masks[0u];
      }

      template <typename ParallelPolicy, typename RandomAccessIterator, typename StateInteger, std::size_t num_operated_qubits, typename Function>
      inline auto gate(
        ParallelPolicy const parallel_policy,
        RandomAccessIterator const first, RandomAccessIterator const last,
        std::array<StateInteger, num_operated_qubits> const& qubit_masks,
        std::array<StateInteger, num_operated_qubits + 1u> const& index_masks,
        Function&& function)
      -> void
      {
        using indices_type = std::array<StateInteger, ::ket::utility::integer_exp2<std::size_t>(num_operated_qubits)>;
        auto indices_vector = std::vector<indices_type>(::ket::utility::num_threads(parallel_policy));
        ::ket::utility::loop_n(
          parallel_policy, static_cast<StateInteger>(last - first) >> num_operated_qubits,
          [first, &function, qubit_masks, index_masks, &indices_vector](StateInteger const index_wo_qubits, int const thread_index)
          {
            // ex. qubit_masks[0]=00000100000; qubit_masks[1]=00100000000; qubit_masks[2]=00000000100;
            // indices[0b000]=xx0xx0xx0xx; indices[0b001]=xx0xx1xx0xx; indices[0b010]=xx1xx0xx0xx; indices[0b011]=xx1xx1xx0xx;
            // indices[0b100]=xx0xx0xx1xx; indices[0b101]=xx0xx1xx1xx; indices[0b110]=xx1xx0xx1xx; indices[0b111]=xx1xx1xx1xx;
            ::ket::gate::gate_detail::make_indices(indices_vector[thread_index], index_wo_qubits, qubit_masks, index_masks);
            function(first, indices_vector[thread_index], thread_index);
          });
      }
    } // namespace gate_detail

    // USAGE:
    // - for Hadamard gate
    //   ::ket::gate::gate(parallel_policy, first, last,
    //     [](auto const first, auto const& indices, int const)
    //     {
    //       auto const zero_iter = first + indices[0b0u];
    //       auto const one_iter = first + indices[0b1u];
    //       auto const zero_iter_value = *zero_iter;
    //
    //       *zero_iter += *one_iter;
    //       *zero_iter *= one_div_root_two;
    //       *one_iter = zero_iter_value - *one_iter;
    //       *one_iter *= one_div_root_two;
    //     },
    //     qubit);
    // - for CNOT gate
    //   ::ket::gate::gate(parallel_policy, first, last,
    //     [](auto const first, auto const& indices, int const)
    //     { std::iter_swap(first + indices[0b10u], first + indices[0b11u]); },
    //     target_qubit, control_qubit);
# ifdef KET_USE_ON_CACHE_STATE_VECTOR
    namespace cache
    {
      // Assumption: Case 2) Some of the operated qubits are off-cache qubits
      namespace unsafe
      {
        template <
          typename ParallelPolicy, typename RandomAccessIterator1, typename RandomAccessIterator2,
          typename Function, typename Qubit, typename... Qubits>
        inline auto gate(
          ParallelPolicy const parallel_policy,
          RandomAccessIterator1 const state_first, RandomAccessIterator1 const state_last,
          RandomAccessIterator2 const on_cache_state_first, RandomAccessIterator2 const on_cache_state_last,
          Function&& function, Qubit&& qubit, Qubits&&... qubits)
        -> void
        {
          using state_integer_type = ::ket::meta::state_integer_t<std::remove_reference_t<Qubit>>;
          using bit_integer_type = ::ket::meta::bit_integer_t<std::remove_reference_t<Qubit>>;
          static_assert(std::is_unsigned<state_integer_type>::value, "state_integer_type of Qubit should be unsigned");
          static_assert(std::is_unsigned<bit_integer_type>::value, "bit_integer_type of Qubit should be unsigned");

          auto const state_size = static_cast<state_integer_type>(state_last - state_first);
          auto const num_qubits = ::ket::utility::integer_log2<bit_integer_type>(state_size);
          assert(::ket::utility::integer_exp2<state_integer_type>(num_qubits) == state_size);
          assert(::ket::utility::all_in_state_vector(num_qubits, qubit, qubits...));
          auto const on_cache_state_size = static_cast<state_integer_type>(on_cache_state_last - on_cache_state_first);
          auto const num_on_cache_qubits = ::ket::utility::integer_log2<bit_integer_type>(on_cache_state_size);
          assert(::ket::utility::integer_exp2<state_integer_type>(num_on_cache_qubits) == on_cache_state_size);
          assert(num_on_cache_qubits < num_qubits);
          auto const num_off_cache_qubits = num_qubits - num_on_cache_qubits;
          // It is required to be confirmed not to satisfy Case 1)
          assert(not ::ket::utility::all_in_state_vector(num_on_cache_qubits, qubit, qubits...));

          constexpr auto num_operated_qubits = bit_integer_type{sizeof...(Qubits) + 1u};

          // xxxx|yyyy|zzzzzz: (local) qubits
          // * xxxx: off-cache qubits
          // * yyyy|zzzzzz: on-cache qubits
          //   - yyyy: chunk qubits
          // * xxxx|yyyy: tag qubits
          // * zzzzzz: nontag qubits

          using qubit_type = ::ket::qubit<state_integer_type, bit_integer_type>;
          auto const least_significant_off_cache_qubit = qubit_type{num_on_cache_qubits};

          // operated_on_cache_qubits_first, operated_on_cache_qubits_last
          std::array<qubit_type, num_operated_qubits> sorted_qubits{::ket::remove_control(qubit), ::ket::remove_control(qubits)...};
          using std::begin;
          using std::end;
          std::sort(begin(sorted_qubits), end(sorted_qubits));
          auto const operated_on_cache_qubits_last
            = std::lower_bound(begin(sorted_qubits), end(sorted_qubits), least_significant_off_cache_qubit);
          auto const operated_on_cache_qubits_first = begin(sorted_qubits);
          auto const num_operated_on_cache_qubits = static_cast<bit_integer_type>(operated_on_cache_qubits_last - operated_on_cache_qubits_first);
          auto const num_operated_off_cache_qubits = num_operated_qubits - num_operated_on_cache_qubits;
          assert(num_operated_off_cache_qubits > bit_integer_type{0u});

          // Case 2-1) There is no operated on-cache qubit
          if (num_operated_on_cache_qubits == bit_integer_type{0u})
          {
            // num_chunk_qubits, chunk_size, least_significant_chunk_qubit, num_tag_qubits, num_nontag_qubits
            constexpr auto num_chunk_qubits = num_operated_qubits; // num_operated_qubits == num_operated_off_cache_qubits;
            constexpr auto num_chunks_in_on_cache_state = ::ket::utility::integer_exp2<state_integer_type>(num_chunk_qubits);
            auto const chunk_size = on_cache_state_size / num_chunks_in_on_cache_state;
            auto const least_significant_chunk_qubit = least_significant_off_cache_qubit - num_chunk_qubits;
            auto const num_tag_qubits = num_off_cache_qubits + num_chunk_qubits;
            auto const num_nontag_qubits = num_on_cache_qubits - num_chunk_qubits;

            // on_cache_qubit_masks, on_cache_index_masks
            std::array<state_integer_type, num_operated_qubits> on_cache_qubit_masks;
            for (auto index = bit_integer_type{0u}; index < num_operated_qubits; ++index)
              on_cache_qubit_masks[index] = state_integer_type{1u} << (least_significant_chunk_qubit + index);
            std::array<state_integer_type, num_operated_qubits + bit_integer_type{1u}> on_cache_index_masks;
            on_cache_index_masks.front() = (state_integer_type{1u} << least_significant_chunk_qubit) - state_integer_type{1u};
            // on_cache_index_masks.size() >= 2 => std::prev(end(on_cache_index_masks)) >= std::next(begin(on_cache_index_masks))
            std::fill(std::next(begin(on_cache_index_masks)), std::prev(end(on_cache_index_masks)), state_integer_type{0u});
            on_cache_index_masks.back() = compl on_cache_index_masks.front();

            // tag_qubit_masks, tag_index_masks
            std::array<state_integer_type, num_operated_qubits> tag_qubit_masks;
            ::ket::gate::gate_detail::make_qubit_masks(tag_qubit_masks, ::ket::remove_control(qubit) - num_nontag_qubits, (::ket::remove_control(qubits) - num_nontag_qubits)...);
            std::array<state_integer_type, num_operated_qubits + bit_integer_type{1u}> tag_index_masks;
            ::ket::gate::gate_detail::make_index_masks(tag_index_masks, ::ket::remove_control(qubit) - num_nontag_qubits, (::ket::remove_control(qubits) - num_nontag_qubits)...);

            // tag_indices
            constexpr auto num_tag_indices = num_chunks_in_on_cache_state;// ::ket::utility::integer_exp2<state_integer_type>(num_operated_qubits);
            std::array<state_integer_type, num_tag_indices> tag_indices;

            auto const tag_loop_size = ::ket::utility::integer_exp2<state_integer_type>(num_tag_qubits - num_operated_qubits);
            for (auto tag_index_wo_qubits = state_integer_type{0u}; tag_index_wo_qubits < tag_loop_size; ++tag_index_wo_qubits)
            {
              ::ket::gate::gate_detail::make_indices(tag_indices, tag_index_wo_qubits, tag_qubit_masks, tag_index_masks);

              for (auto index = state_integer_type{0u}; index < num_tag_indices; ++index)
                ::ket::utility::copy_n(
                  parallel_policy,
                  state_first + tag_indices[index] * chunk_size, chunk_size,
                  on_cache_state_first + index * chunk_size);

              ::ket::gate::gate_detail::gate(parallel_policy, on_cache_state_first, on_cache_state_last, on_cache_qubit_masks, on_cache_index_masks, std::forward<Function>(function));

              for (auto index = state_integer_type{0u}; index < num_tag_indices; ++index)
                ::ket::utility::copy_n(
                  parallel_policy,
                  on_cache_state_first + index * chunk_size, chunk_size,
                  state_first + tag_indices[index] * chunk_size);
            }

            return;
          }

          // Case 2-2) There are some operated on-cache qubits (num_operated_on_cache_qubits > 0)
          // least_significant_chunk_qubit, num_chunk_qubits, chunk_size, num_tag_qubits, num_nontag_qubits
          auto operated_on_cache_qubits_iter = std::prev(operated_on_cache_qubits_last);
          auto free_most_significant_on_cache_qubit = least_significant_off_cache_qubit - bit_integer_type{1u};
          for (auto num_found_off_cache_chunk_qubits = bit_integer_type{0u};
               num_found_off_cache_chunk_qubits < num_operated_off_cache_qubits; ++num_found_off_cache_chunk_qubits)
            while (free_most_significant_on_cache_qubit-- == *operated_on_cache_qubits_iter)
              if (operated_on_cache_qubits_iter != operated_on_cache_qubits_first)
                --operated_on_cache_qubits_iter;
          auto const least_significant_chunk_qubit = free_most_significant_on_cache_qubit + bit_integer_type{1u};
          auto const num_chunk_qubits = static_cast<bit_integer_type>(least_significant_off_cache_qubit - least_significant_chunk_qubit);
          assert(num_chunk_qubits <= num_operated_qubits);
          auto const num_chunks_in_on_cache_state = ::ket::utility::integer_exp2<state_integer_type>(num_chunk_qubits);
          auto const chunk_size = on_cache_state_size / num_chunks_in_on_cache_state;
          auto const num_tag_qubits = num_off_cache_qubits + num_chunk_qubits;
          auto const num_nontag_qubits = num_on_cache_qubits - num_chunk_qubits;

          // operated_tag_qubits, on_cache_qubit_masks, on_cache_index_masks
          auto operated_tag_qubits = std::vector<qubit_type>{};
          operated_tag_qubits.reserve(num_chunk_qubits);
          auto present_chunk_qubit = least_significant_chunk_qubit;
          auto const modified_operated_qubits
            = ::ket::utility::variadic::transform(
                [least_significant_chunk_qubit, num_nontag_qubits, &operated_tag_qubits, &present_chunk_qubit](auto qubit)
                {
                  if (qubit < least_significant_chunk_qubit)
                    return qubit;

                  operated_tag_qubits.push_back(::ket::remove_control(qubit) - num_nontag_qubits);
                  return static_cast<decltype(qubit)>(present_chunk_qubit++);
                },
                qubit, qubits...);
          assert(present_chunk_qubit == least_significant_off_cache_qubit);
          assert(static_cast<bit_integer_type>(operated_tag_qubits.size()) == num_chunk_qubits);
          std::array<state_integer_type, num_operated_qubits> on_cache_qubit_masks{};
          ::ket::gate::gate_detail::make_qubit_masks(modified_operated_qubits, on_cache_qubit_masks);
          std::array<state_integer_type, num_operated_qubits + 1u> on_cache_index_masks{};
          ::ket::gate::gate_detail::make_index_masks(modified_operated_qubits, on_cache_index_masks);

          // tag_qubit_masks, tag_index_masks
          auto tag_qubit_masks = std::vector<state_integer_type>{};
          tag_qubit_masks.reserve(operated_tag_qubits.size());
          ::ket::gate::gate_detail::runtime::ranges::make_qubit_masks(operated_tag_qubits, std::back_inserter(tag_qubit_masks));
          assert(tag_qubit_masks.size() == operated_tag_qubits.size());
          auto tag_index_masks = std::vector<state_integer_type>{};
          tag_index_masks.reserve(operated_tag_qubits.size() + 1u);
          ::ket::gate::gate_detail::runtime::ranges::make_index_masks(operated_tag_qubits, std::back_inserter(tag_index_masks));
          assert(tag_index_masks.size() == operated_tag_qubits.size() + 1u);

          // tag_indices
          auto const num_tag_indices = num_chunks_in_on_cache_state;// ::ket::utility::integer_exp2<state_integer_type>(operated_tag_qubits.size());
          auto tag_indices = std::vector<state_integer_type>{};
          tag_indices.reserve(num_tag_indices);

          auto const tag_loop_size = ::ket::utility::integer_exp2<state_integer_type>(num_tag_qubits - num_chunk_qubits); // num_chunk_qubits == operated_tag_qubits.size()
          for (auto tag_index_wo_qubits = state_integer_type{0u}; tag_index_wo_qubits < tag_loop_size; ++tag_index_wo_qubits)
          {
            tag_indices.clear();
            ::ket::gate::gate_detail::runtime::ranges::make_indices(
              tag_qubit_masks, tag_index_masks, tag_index_wo_qubits, std::back_inserter(tag_indices));
            assert(tag_indices.size() == num_tag_indices);

            for (auto index = state_integer_type{0u}; index < num_tag_indices; ++index)
              ::ket::utility::copy_n(
                parallel_policy,
                state_first + tag_indices[index] * chunk_size, chunk_size,
                on_cache_state_first + index * chunk_size);

            ::ket::gate::gate_detail::gate(parallel_policy, on_cache_state_first, on_cache_state_last, on_cache_qubit_masks, on_cache_index_masks, std::forward<Function>(function));

            for (auto index = state_integer_type{0u}; index < num_tag_indices; ++index)
              ::ket::utility::copy_n(
                parallel_policy,
                on_cache_state_first + index * chunk_size, chunk_size,
                state_first + tag_indices[index] * chunk_size);
          }
        }

        template <
          typename RandomAccessIterator1, typename RandomAccessIterator2,
          typename Function, typename Qubit, typename... Qubits>
        inline auto gate(
          RandomAccessIterator1 const state_first, RandomAccessIterator1 const state_last,
          RandomAccessIterator2 const on_cache_state_first, RandomAccessIterator2 const on_cache_state_last,
          Function&& function, Qubit&& qubit, Qubits&&... qubits)
        -> void
        {
          ::ket::gate::cache::unsafe::gate(
            ::ket::utility::policy::make_sequential(),
            state_first, state_last, on_cache_state_first, on_cache_state_last,
            std::forward<Function>(function), std::forward<Qubit>(qubit), std::forward<Qubits>(qubits)...);
        }

        namespace ranges
        {
          template <
            typename ParallelPolicy, typename RandomAccessRange1, typename RandomAccessRange2,
            typename Function, typename Qubit, typename... Qubits>
          inline std::enable_if_t< ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value, RandomAccessRange1& > gate(
            ParallelPolicy const parallel_policy,
            RandomAccessRange1& state, RandomAccessRange2& on_cache_state,
            Function&& function, Qubit&& qubit, Qubits&&... qubits)
          {
            using std::begin;
            using std::end;
            ::ket::gate::cache::unsafe::gate(
              parallel_policy,
              begin(state), end(state), begin(on_cache_state), end(on_cache_state),
              std::forward<Function>(function), std::forward<Qubit>(qubit), std::forward<Qubits>(qubits)...);
            return state;
          }

          template <typename RandomAccessRange1, typename RandomAccessRange2, typename Function, typename Qubit, typename... Qubits>
          inline std::enable_if_t<not ::ket::utility::policy::meta::is_loop_n_policy<RandomAccessRange1&>::value, RandomAccessRange1&> gate(
            RandomAccessRange1& state, RandomAccessRange2& on_cache_state,
            Function&& function, Qubit&& qubit, Qubits&&... qubits)
          {
            return ::ket::gate::cache::unsafe::ranges::gate(
              ::ket::utility::policy::make_sequential(),
              state, on_cache_state, std::forward<Function>(function), std::forward<Qubit>(qubit), std::forward<Qubits>(qubits)...);
          }
        } // namespace ranges
      } // namespace unsafe

      template <
        typename ParallelPolicy, typename RandomAccessIterator1, typename RandomAccessIterator2,
        typename Function, typename Qubit, typename... Qubits>
      inline auto gate(
        ParallelPolicy const parallel_policy,
        RandomAccessIterator1 const state_first, RandomAccessIterator1 const state_last,
        RandomAccessIterator2 const on_cache_state_first, RandomAccessIterator2 const on_cache_state_last,
        Function&& function, Qubit&& qubit, Qubits&&... qubits)
      -> void
      {
        using state_integer_type = ::ket::meta::state_integer_t<std::remove_reference_t<Qubit>>;
        using bit_integer_type = ::ket::meta::bit_integer_t<std::remove_reference_t<Qubit>>;
        static_assert(std::is_unsigned<state_integer_type>::value, "state_integer_type of Qubit should be unsigned");
        static_assert(std::is_unsigned<bit_integer_type>::value, "bit_integer_type of Qubit should be unsigned");

# ifndef NDEBUG
        auto const state_size = static_cast<state_integer_type>(state_last - state_first);
        auto const num_qubits = ::ket::utility::integer_log2<bit_integer_type>(state_size);
# endif // NDEBUG
        assert(::ket::utility::integer_exp2<state_integer_type>(num_qubits) == state_size);
        assert(::ket::utility::all_in_state_vector(num_qubits, qubit, qubits...));
        auto const on_cache_state_size = static_cast<state_integer_type>(on_cache_state_last - on_cache_state_first);
        auto const num_on_cache_qubits = ::ket::utility::integer_log2<bit_integer_type>(on_cache_state_size);
        assert(::ket::utility::integer_exp2<state_integer_type>(num_on_cache_qubits) == on_cache_state_size);
        assert(num_on_cache_qubits < num_qubits);

        constexpr auto num_operated_qubits = sizeof...(Qubits) + 1u;

        // xxxx|yyyy|zzzzzz: (local) qubits
        // * xxxx: off-cache qubits
        // * yyyy|zzzzzz: on-cache qubits
        //   - yyyy: chunk qubits

        // Case 1) All operated qubits are on-cache qubits
        if (::ket::utility::all_in_state_vector(num_on_cache_qubits, qubit, qubits...))
        {
          std::array<state_integer_type, num_operated_qubits> qubit_masks{};
          ::ket::gate::gate_detail::make_qubit_masks(qubit_masks, qubit, qubits...);
          std::array<state_integer_type, num_operated_qubits + 1u> index_masks{};
          ::ket::gate::gate_detail::make_index_masks(index_masks, std::forward<Qubit>(qubit), std::forward<Qubits>(qubits)...);

          for (auto iter = state_first; iter < state_last; iter += on_cache_state_size)
          {
            ::ket::utility::copy_n(parallel_policy, iter, on_cache_state_size, on_cache_state_first);
            ::ket::gate::gate_detail::gate(parallel_policy, on_cache_state_first, on_cache_state_last, qubit_masks, index_masks, std::forward<Function>(function));
            ::ket::utility::copy(parallel_policy, on_cache_state_first, on_cache_state_last, iter);
          }

          return;
        }

        // Case 2) Some of the operated qubits are off-cache qubits
        ::ket::gate::cache::unsafe::gate(
          parallel_policy,
          state_first, state_last, on_cache_state_first, on_cache_state_last,
          std::forward<Function>(function), std::forward<Qubit>(qubit), std::forward<Qubits>(qubits)...);
      }

      template <
        typename RandomAccessIterator1, typename RandomAccessIterator2,
        typename Function, typename Qubit, typename... Qubits>
      inline auto gate(
        RandomAccessIterator1 const state_first, RandomAccessIterator1 const state_last,
        RandomAccessIterator2 const on_cache_state_first, RandomAccessIterator2 const on_cache_state_last,
        Function&& function, Qubit&& qubit, Qubits&&... qubits)
      -> void
      {
        ::ket::gate::cache::gate(
          ::ket::utility::policy::make_sequential(),
          state_first, state_last, on_cache_state_first, on_cache_state_last,
          std::forward<Function>(function), std::forward<Qubit>(qubit), std::forward<Qubits>(qubits)...);
      }

      namespace ranges
      {
        template <
          typename ParallelPolicy, typename RandomAccessRange1, typename RandomAccessRange2,
          typename Function, typename Qubit, typename... Qubits>
        inline std::enable_if_t< ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value, RandomAccessRange1& > gate(
          ParallelPolicy const parallel_policy,
          RandomAccessRange1& state, RandomAccessRange2& on_cache_state,
          Function&& function, Qubit&& qubit, Qubits&&... qubits)
        {
          using std::begin;
          using std::end;
          ::ket::gate::cache::gate(
            parallel_policy,
            begin(state), end(state), begin(on_cache_state), end(on_cache_state),
            std::forward<Function>(function), std::forward<Qubit>(qubit), std::forward<Qubits>(qubits)...);
          return state;
        }

        template <typename RandomAccessRange1, typename RandomAccessRange2, typename Function, typename Qubit, typename... Qubits>
        inline std::enable_if_t<not ::ket::utility::policy::meta::is_loop_n_policy<RandomAccessRange1&>::value, RandomAccessRange1&> gate(
          RandomAccessRange1& state, RandomAccessRange2& on_cache_state,
          Function&& function, Qubit&& qubit, Qubits&&... qubits)
        {
          return ::ket::gate::cache::ranges::gate(
            ::ket::utility::policy::make_sequential(),
            state, on_cache_state, std::forward<Function>(function), std::forward<Qubit>(qubit), std::forward<Qubits>(qubits)...);
        }
      } // namespace ranges
    } // namespace cache
# endif // KET_USE_ON_CACHE_STATE_VECTOR

    namespace nocache
    {
      template <typename ParallelPolicy, typename RandomAccessIterator, typename Function, typename Qubit, typename... Qubits>
      inline auto gate(
        ParallelPolicy const parallel_policy,
        RandomAccessIterator const first, RandomAccessIterator const last,
        Function&& function, Qubit&& qubit, Qubits&&... qubits)
      -> void
      {
        using state_integer_type = ::ket::meta::state_integer_t<std::remove_reference_t<Qubit>>;
        using bit_integer_type = ::ket::meta::bit_integer_t<std::remove_reference_t<Qubit>>;
        static_assert(std::is_unsigned<state_integer_type>::value, "state_integer_type of Qubit should be unsigned");
        static_assert(std::is_unsigned<bit_integer_type>::value, "bit_integer_type of Qubit should be unsigned");

        static constexpr auto num_operated_qubits = sizeof...(Qubits) + 1u;

#   ifndef NDEBUG
        auto const state_size = static_cast<state_integer_type>(last - first);
        auto const num_qubits = ::ket::utility::integer_log2<bit_integer_type>(state_size);
#   endif // NDEBUG
        assert(::ket::utility::integer_exp2<state_integer_type>(num_qubits) == state_size);
        assert(::ket::utility::all_in_state_vector(state_size, qubit, qubits...));

        std::array<state_integer_type, num_operated_qubits> qubit_masks{};
        ::ket::gate::gate_detail::make_qubit_masks(qubit_masks, qubit, qubits...);
        std::array<state_integer_type, num_operated_qubits + 1u> index_masks{};
        ::ket::gate::gate_detail::make_index_masks(index_masks, std::forward<Qubit>(qubit), std::forward<Qubits>(qubits)...);

        ::ket::gate::gate_detail::gate(parallel_policy, first, last, qubit_masks, index_masks, std::forward<Function>(function));
      }

      template <typename RandomAccessIterator, typename Function, typename Qubit, typename... Qubits>
      inline auto gate(
        RandomAccessIterator const first, RandomAccessIterator const last,
        Function&& function, Qubit&& qubit, Qubits&&... qubits)
      -> void
      {
        ::ket::gate::nocache::gate(
          ::ket::utility::policy::make_sequential(),
          first, last, std::forward<Function>(function),
          std::forward<Qubit>(qubit), std::forward<Qubits>(qubits)...);
      }

      namespace ranges
      {
        template <typename ParallelPolicy, typename RandomAccessRange, typename Function, typename Qubit, typename... Qubits>
        inline std::enable_if_t< ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value, RandomAccessRange& > gate(
          ParallelPolicy const parallel_policy, RandomAccessRange& state,
          Function&& function, Qubit&& qubit, Qubits&&... qubits)
        {
          using std::begin;
          using std::end;
          ::ket::gate::nocache::gate(
            parallel_policy, begin(state), end(state), std::forward<Function>(function),
            std::forward<Qubit>(qubit), std::forward<Qubits>(qubits)...);
          return state;
        }

        template <typename RandomAccessRange, typename Function, typename Qubit, typename... Qubits>
        inline std::enable_if_t<not ::ket::utility::policy::meta::is_loop_n_policy<RandomAccessRange&>::value, RandomAccessRange&> gate(
          RandomAccessRange& state, Function&& function, Qubit&& qubit, Qubits&&... qubits)
        {
          return ::ket::gate::nocache::ranges::gate(
            ::ket::utility::policy::make_sequential(),
            state, std::forward<Function>(function), std::forward<Qubit>(qubit), std::forward<Qubits>(qubits)...);
        }
      } // namespace ranges
    } // namespace nocache

    template <typename ParallelPolicy, typename RandomAccessIterator, typename Function, typename Qubit, typename... Qubits>
    inline auto gate(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      Function&& function, Qubit&& qubit, Qubits&&... qubits)
    -> void
    {
      using state_integer_type = ::ket::meta::state_integer_t<std::remove_reference_t<Qubit>>;
      using bit_integer_type = ::ket::meta::bit_integer_t<std::remove_reference_t<Qubit>>;
      static_assert(std::is_unsigned<state_integer_type>::value, "state_integer_type of Qubit should be unsigned");
      static_assert(std::is_unsigned<bit_integer_type>::value, "bit_integer_type of Qubit should be unsigned");

      static constexpr auto num_operated_qubits = sizeof...(Qubits) + 1u;

# ifndef KET_USE_ON_CACHE_STATE_VECTOR
      ::ket::gate::nocache::gate(
        parallel_policy, first, last, std::forward<Function>(function), std::forward<Qubit>(qubit), std::forward<Qubits>(qubits)...);
      return;
# else // KET_USE_ON_CACHE_STATE_VECTOR
      auto const state_size = static_cast<state_integer_type>(last - first);
      auto const num_qubits = ::ket::utility::integer_log2<bit_integer_type>(state_size);
      assert(::ket::utility::integer_exp2<state_integer_type>(num_qubits) == state_size);
      assert(::ket::utility::all_in_state_vector(state_size, qubit, qubits...));

#   ifndef KET_DEFAULT_NUM_ON_CACHE_QUBITS
#     define KET_DEFAULT_NUM_ON_CACHE_QUBITS 16
#   endif // KET_DEFAULT_NUM_ON_CACHE_QUBITS
      constexpr auto num_on_cache_qubits = bit_integer_type{KET_DEFAULT_NUM_ON_CACHE_QUBITS};
      if (num_qubits <= num_on_cache_qubits)
      {
        ::ket::gate::nocache::gate(
          parallel_policy, first, last, std::forward<Function>(function), std::forward<Qubit>(qubit), std::forward<Qubits>(qubits)...);
        return;
      }

      constexpr auto on_cache_state_size = ::ket::utility::integer_exp2<state_integer_type>(num_on_cache_qubits);

      // xxxx|yyyy|zzzzzz: (local) qubits
      // * xxxx: off-cache qubits
      // * yyyy|zzzzzz: on-cache qubits
      //   - yyyy: chunk qubits

      // Case 1) All operated qubits are on-cache qubits
      if (::ket::utility::all_in_state_vector(num_on_cache_qubits, qubit, qubits...))
      {
        std::array<state_integer_type, num_operated_qubits> qubit_masks{};
        ::ket::gate::gate_detail::make_qubit_masks(qubit_masks, qubit, qubits...);
        std::array<state_integer_type, num_operated_qubits + 1u> index_masks{};
        ::ket::gate::gate_detail::make_index_masks(index_masks, std::forward<Qubit>(qubit), std::forward<Qubits>(qubits)...);

        for (auto iter = first; iter < last; iter += on_cache_state_size)
          ::ket::gate::gate_detail::gate(parallel_policy, iter, iter + on_cache_state_size, qubit_masks, index_masks, std::forward<Function>(function));

        return;
      }

      // Case 2) Some of the operated qubits are off-cache qubits
      auto on_cache_state = std::vector<typename std::iterator_traits<RandomAccessIterator>::value_type>(on_cache_state_size);

      using std::begin;
      using std::end;
      ::ket::gate::cache::unsafe::gate(
        parallel_policy,
        first, last, begin(on_cache_state), end(on_cache_state),
        std::forward<Function>(function), std::forward<Qubit>(qubit), std::forward<Qubits>(qubits)...);
# endif // KET_USE_ON_CACHE_STATE_VECTOR
    }

    template <typename RandomAccessIterator, typename Function, typename Qubit, typename... Qubits>
    inline auto gate(
      RandomAccessIterator const first, RandomAccessIterator const last,
      Function&& function, Qubit&& qubit, Qubits&&... qubits)
    -> void
    {
      ::ket::gate::gate(
        ::ket::utility::policy::make_sequential(),
        first, last, std::forward<Function>(function),
        std::forward<Qubit>(qubit), std::forward<Qubits>(qubits)...);
    }

    namespace ranges
    {
      template <typename ParallelPolicy, typename RandomAccessRange, typename Function, typename Qubit, typename... Qubits>
      inline std::enable_if_t< ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value, RandomAccessRange& > gate(
        ParallelPolicy const parallel_policy, RandomAccessRange& state,
        Function&& function, Qubit&& qubit, Qubits&&... qubits)
      {
        using std::begin;
        using std::end;
        ::ket::gate::gate(
          parallel_policy, begin(state), end(state), std::forward<Function>(function),
          std::forward<Qubit>(qubit), std::forward<Qubits>(qubits)...);
        return state;
      }

      template <typename RandomAccessRange, typename Function, typename Qubit, typename... Qubits>
      inline std::enable_if_t<not ::ket::utility::policy::meta::is_loop_n_policy<RandomAccessRange&>::value, RandomAccessRange&> gate(
        RandomAccessRange& state, Function&& function, Qubit&& qubit, Qubits&&... qubits)
      {
        return ::ket::gate::ranges::gate(
          ::ket::utility::policy::make_sequential(),
          state, std::forward<Function>(function), std::forward<Qubit>(qubit), std::forward<Qubits>(qubits)...);
      }
    } // namespace ranges
  } // namespace gate
} // namespace ket


#endif // KET_GATE_GATE_HPP
