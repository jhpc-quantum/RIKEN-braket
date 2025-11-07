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
# include <ket/gate/utility/index_with_qubits.hpp>
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) and !defined(KET_USE_ON_CACHE_STATE_VECTOR)
#   include <ket/gate/utility/cache_aware_iterator.hpp>
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) and !defined(KET_USE_ON_CACHE_STATE_VECTOR)
# include <ket/meta/state_integer_of.hpp>
# include <ket/meta/bit_integer_of.hpp>
# include <ket/utility/loop_n.hpp>
# include <ket/utility/integer_exp2.hpp>
# include <ket/utility/integer_log2.hpp>
# if !defined(NDEBUG) || defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION)
#   include <ket/utility/all_in_state_vector.hpp>
# endif // !defined(NDEBUG) || defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION)
# ifdef KET_ENABLE_CACHE_AWARE_GATE_FUNCTION
#   include <ket/utility/none_in_state_vector.hpp>
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
# if __cpp_constexpr < 201603L
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
# endif // __cpp_constexpr < 201603L

      template <typename ParallelPolicy, typename RandomAccessIterator, typename StateInteger, typename BitInteger, std::size_t num_operated_qubits, typename Function>
      inline auto gate_n(
        ParallelPolicy const parallel_policy,
        RandomAccessIterator const first, StateInteger const size,
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_operated_qubits > const& unsorted_qubits,
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_operated_qubits + 1u > const& sorted_qubits_with_sentinel,
        Function&& function)
      -> void
      {
        ::ket::utility::loop_n(
          parallel_policy, size >> num_operated_qubits,
          [first, &function, &unsorted_qubits, &sorted_qubits_with_sentinel](StateInteger const index_wo_qubits, int const thread_index)
          { function(first, index_wo_qubits, unsorted_qubits, sorted_qubits_with_sentinel, thread_index); });
      }

      template <typename ParallelPolicy, typename RandomAccessIterator, typename StateInteger, typename BitInteger, std::size_t num_operated_qubits, typename Function>
      inline auto gate(
        ParallelPolicy const parallel_policy,
        RandomAccessIterator const first, RandomAccessIterator const last,
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_operated_qubits > const& unsorted_qubits,
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_operated_qubits + 1u > const& sorted_qubits_with_sentinel,
        Function&& function)
      -> void
      { ::ket::gate::gate_detail::gate_n(parallel_policy, first, static_cast<StateInteger>(last - first), unsorted_qubits, sorted_qubits_with_sentinel, std::forward<Function>(function)); }
    } // namespace gate_detail

    // USAGE:
    // - for Hadamard gate
    //   ::ket::gate::gate(parallel_policy, first, last,
    //     [](auto const first, auto const index_wo_qubits, auto const& unsorted_qubits, auto const& sorted_qubits_with_sentinel, int const)
    //     {
    //       using std::begin;
    //       using std::end;
    //       auto const zero_iter = first + ket::gate::utility::index_with_qubits(index_wo_qubits, 0b0u, begin(unsorted_qubits), end(unsorted_qubits), begin(sorted_qubits_with_sentinel), end(sorted_qubits_with_sentinel));
    //       auto const one_iter = first + ket::gate::utility::index_with_qubits(index_wo_qubits, 0b1u, begin(unsorted_qubits), end(unsorted_qubits), begin(sorted_qubits_with_sentinel), end(sorted_qubits_with_sentinel));
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
    //     [](auto const first, auto const index_wo_qubits, auto const& unsorted_qubits, auto const& sorted_qubits_with_sentinel, int const)
    //     {
    //       using std::begin;
    //       using std::end;
    //       std::iter_swap(
    //         first + ket::gate::utility::index_with_qubits(index_wo_qubits, 0b10u, begin(unsorted_qubits), end(unsorted_qubits), begin(sorted_qubits_with_sentinel), end(sorted_qubits_with_sentinel)),
    //         first + ket::gate::utility::index_with_qubits(index_wo_qubits, 0b11u, begin(unsorted_qubits), end(unsorted_qubits), begin(sorted_qubits_with_sentinel), end(sorted_qubits_with_sentinel)));
    //     },
    //     target_qubit, control_qubit);
    namespace nocache
    {
      template <typename ParallelPolicy, typename RandomAccessIterator, typename Function>
      inline auto gate(
        ParallelPolicy const parallel_policy,
        RandomAccessIterator const first, RandomAccessIterator const last,
        Function&& function)
      -> void
      {
        using qubit_type = ::ket::qubit<>;
        using state_integer_type = ::ket::meta::state_integer_t<qubit_type>;
        using bit_integer_type = ::ket::meta::bit_integer_t<qubit_type>;

        auto const state_size = static_cast<state_integer_type>(last - first);
        auto const num_qubits = ::ket::utility::integer_log2<bit_integer_type>(state_size);
        assert(::ket::utility::integer_exp2<state_integer_type>(num_qubits) == state_size);
        auto const sentinel_qubit = ::ket::make_qubit<state_integer_type>(num_qubits);

        std::array<qubit_type, 0u> unsorted_qubits{};
        std::array<qubit_type, 1u> sorted_qubits_with_sentinel{sentinel_qubit};

        ::ket::gate::gate_detail::gate_n(parallel_policy, first, state_size, unsorted_qubits, sorted_qubits_with_sentinel, std::forward<Function>(function));
      }

      template <typename ParallelPolicy, typename RandomAccessIterator, typename Function, typename Qubit>
      inline auto gate(
        ParallelPolicy const parallel_policy,
        RandomAccessIterator const first, RandomAccessIterator const last,
        Function&& function, Qubit&& qubit)
      -> void
      {
        using state_integer_type = ::ket::meta::state_integer_t<std::remove_cv_t<std::remove_reference_t<Qubit>>>;
        static_assert(std::is_unsigned<state_integer_type>::value, "state_integer_type of Qubit should be unsigned");

        using bit_integer_type = ::ket::meta::bit_integer_t<std::remove_cv_t<std::remove_reference_t<Qubit>>>;
        static_assert(std::is_unsigned<bit_integer_type>::value, "bit_integer_type of Qubit should be unsigned");

        auto const state_size = static_cast<state_integer_type>(last - first);
        auto const num_qubits = ::ket::utility::integer_log2<bit_integer_type>(state_size);
        assert(::ket::utility::integer_exp2<state_integer_type>(num_qubits) == state_size);
        auto const sentinel_qubit = ::ket::make_qubit<state_integer_type>(num_qubits);
        assert(qubit < sentinel_qubit);

        using qubit_type = ::ket::qubit<state_integer_type, bit_integer_type>;
        constexpr auto num_operated_qubits = bit_integer_type{1u};
        std::array<qubit_type, num_operated_qubits> unsorted_qubits{::ket::remove_control(std::forward<Qubit>(qubit))};

        std::array<qubit_type, num_operated_qubits + bit_integer_type{1u}> sorted_qubits_with_sentinel{::ket::remove_control(qubit), sentinel_qubit};

        ::ket::gate::gate_detail::gate_n(parallel_policy, first, state_size, unsorted_qubits, sorted_qubits_with_sentinel, std::forward<Function>(function));
      }

      template <typename ParallelPolicy, typename RandomAccessIterator, typename Function, typename Qubit1, typename Qubit2>
      inline auto gate(
        ParallelPolicy const parallel_policy,
        RandomAccessIterator const first, RandomAccessIterator const last,
        Function&& function, Qubit1&& qubit1, Qubit2&& qubit2)
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

        auto const state_size = static_cast<state_integer_type>(last - first);
        auto const num_qubits = ::ket::utility::integer_log2<bit_integer_type>(state_size);
        assert(::ket::utility::integer_exp2<state_integer_type>(num_qubits) == state_size);
        auto const sentinel_qubit = ::ket::make_qubit<state_integer_type>(num_qubits);
        assert(qubit1 < sentinel_qubit);
        assert(qubit2 < sentinel_qubit);
        assert(qubit1 != qubit2);

        auto const minmax_qubits = std::minmax(ket::remove_control(qubit1), ket::remove_control(qubit2));
        using qubit_type = ::ket::qubit<state_integer_type, bit_integer_type>;
        constexpr auto num_operated_qubits = bit_integer_type{2u};
        std::array<qubit_type, num_operated_qubits + bit_integer_type{1u}> sorted_qubits_with_sentinel{
          ::ket::remove_control(minmax_qubits.first), ::ket::remove_control(minmax_qubits.second), sentinel_qubit};
        std::array<qubit_type, num_operated_qubits> unsorted_qubits{
          ::ket::remove_control(std::forward<Qubit1>(qubit1)), ::ket::remove_control(std::forward<Qubit2>(qubit2))};

        ::ket::gate::gate_detail::gate_n(parallel_policy, first, state_size, unsorted_qubits, sorted_qubits_with_sentinel, std::forward<Function>(function));
      }

      template <typename ParallelPolicy, typename RandomAccessIterator, typename Function, typename Qubit1, typename Qubit2, typename Qubit3, typename... Qubits>
      inline auto gate(
        ParallelPolicy const parallel_policy,
        RandomAccessIterator const first, RandomAccessIterator const last,
        Function&& function, Qubit1&& qubit1, Qubit2&& qubit2, Qubit3&& qubit3, Qubits&&... qubits)
      -> void
      {
        using state_integer_type = ::ket::meta::state_integer_t<std::remove_cv_t<std::remove_reference_t<Qubit1>>>;
        static_assert(std::is_unsigned<state_integer_type>::value, "state_integer_type of Qubit1 should be unsigned");
# if __cpp_constexpr >= 201603L
        static_assert(
          ::ket::utility::variadic::proj::all_of(
            [](auto integer) { return std::is_same<decltype(integer), state_integer_type>::value; },
            [](auto qubit) { return ::ket::meta::state_integer_t<decltype(qubit)>{}; },
            std::remove_cv_t<std::remove_reference_t<Qubit2>>{}, std::remove_cv_t<std::remove_reference_t<Qubit3>>{},
            std::remove_cv_t<std::remove_reference_t<Qubits>>{}...),
          "state_integer_type's of Qubit1, Qubit2, Qubit3 and Qubits should be the same");
# else // __cpp_constexpr >= 201603L
        static_assert(
          ::ket::utility::variadic::proj::all_of(
            ::ket::gate::gate_detail::is_same_to<state_integer_type>{}, ::ket::gate::gate_detail::state_integer_of{},
            std::remove_cv_t<std::remove_reference_t<Qubit2>>{}, std::remove_cv_t<std::remove_reference_t<Qubit3>>{},
            std::remove_cv_t<std::remove_reference_t<Qubits>>{}...),
          "state_integer_type's of Qubit1, Qubit2, Qubit3 and Qubits should be the same");
# endif // __cpp_constexpr >= 201603L

        using bit_integer_type = ::ket::meta::bit_integer_t<std::remove_cv_t<std::remove_reference_t<Qubit1>>>;
        static_assert(std::is_unsigned<bit_integer_type>::value, "bit_integer_type of Qubit1 should be unsigned");
# if __cpp_constexpr >= 201603L
        static_assert(
          ::ket::utility::variadic::proj::all_of(
            [](auto integer) { return std::is_same<decltype(integer), bit_integer_type>::value; },
            [](auto qubit) { return ::ket::meta::bit_integer_t<decltype(qubit)>{}; },
            std::remove_cv_t<std::remove_reference_t<Qubit2>>{}, std::remove_cv_t<std::remove_reference_t<Qubit3>>{},
            std::remove_cv_t<std::remove_reference_t<Qubits>>{}...),
          "bit_integer_type's of Qubit1, Qubit2, Qubit3 and Qubits should be the same");
# else // __cpp_constexpr >= 201603L
        static_assert(
          ::ket::utility::variadic::proj::all_of(
            ::ket::gate::gate_detail::is_same_to<bit_integer_type>{}, ::ket::gate::gate_detail::bit_integer_of{},
            std::remove_cv_t<std::remove_reference_t<Qubit2>>{}, std::remove_cv_t<std::remove_reference_t<Qubit3>>{},
            std::remove_cv_t<std::remove_reference_t<Qubits>>{}...),
          "bit_integer_type's of Qubit1, Qubit2, Qubit3 and Qubits should be the same");
# endif // __cpp_constexpr >= 201603L

        auto const state_size = static_cast<state_integer_type>(last - first);
        auto const num_qubits = ::ket::utility::integer_log2<bit_integer_type>(state_size);
        assert(::ket::utility::integer_exp2<state_integer_type>(num_qubits) == state_size);
        assert(::ket::utility::all_in_state_vector(num_qubits, qubit1, qubit2, qubit3, qubits...));

        using qubit_type = ::ket::qubit<state_integer_type, bit_integer_type>;
        constexpr auto num_operated_qubits = static_cast<bit_integer_type>(sizeof...(Qubits) + 3u);
        std::array<qubit_type, num_operated_qubits + bit_integer_type{1u}> sorted_qubits_with_sentinel{
          ::ket::remove_control(qubit1), ::ket::remove_control(qubit2),
          ::ket::remove_control(qubit3), ::ket::remove_control(qubits)...,
          ::ket::make_qubit<state_integer_type>(num_qubits)};
        using std::begin;
        using std::end;
        std::sort(begin(sorted_qubits_with_sentinel), std::prev(end(sorted_qubits_with_sentinel)));
        std::array<qubit_type, num_operated_qubits> unsorted_qubits{
          ::ket::remove_control(std::forward<Qubit1>(qubit1)), ::ket::remove_control(std::forward<Qubit2>(qubit2)),
          ::ket::remove_control(std::forward<Qubit3>(qubit3)), ::ket::remove_control(std::forward<Qubits>(qubits))...};

        ::ket::gate::gate_detail::gate_n(parallel_policy, first, state_size, unsorted_qubits, sorted_qubits_with_sentinel, std::forward<Function>(function));
      }

      template <typename RandomAccessIterator, typename Function, typename... Qubits>
      inline auto gate(
        RandomAccessIterator const first, RandomAccessIterator const last,
        Function&& function, Qubits&&... qubits)
      -> void
      {
        ::ket::gate::nocache::gate(
          ::ket::utility::policy::make_sequential(),
          first, last, std::forward<Function>(function), std::forward<Qubits>(qubits)...);
      }

      namespace ranges
      {
        template <typename ParallelPolicy, typename RandomAccessRange, typename Function, typename... Qubits>
        inline std::enable_if_t< ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value, RandomAccessRange& > gate(
          ParallelPolicy const parallel_policy, RandomAccessRange& state,
          Function&& function, Qubits&&... qubits)
        {
          using std::begin;
          using std::end;
          ::ket::gate::nocache::gate(
            parallel_policy, begin(state), end(state), std::forward<Function>(function), std::forward<Qubits>(qubits)...);
          return state;
        }

        template <typename RandomAccessRange, typename Function, typename... Qubits>
        inline std::enable_if_t<not ::ket::utility::policy::meta::is_loop_n_policy<RandomAccessRange&>::value, RandomAccessRange&> gate(
          RandomAccessRange& state, Function&& function, Qubits&&... qubits)
        {
          return ::ket::gate::nocache::ranges::gate(
            ::ket::utility::policy::make_sequential(),
            state, std::forward<Function>(function), std::forward<Qubits>(qubits)...);
        }
      } // namespace ranges
    } // namespace nocache
# ifdef KET_ENABLE_CACHE_AWARE_GATE_FUNCTION

    namespace cache
    {
      // Case 1) All operated qubits are on-cache qubits
      //   ex: xxxx|zzzzzzzzzz
      //             ^  ^   ^  <- operated qubits
      namespace all_on_cache
      {
        // First argument of Function: RandomAccessIterator
        template <typename ParallelPolicy, typename RandomAccessIterator, typename Function>
        inline auto gate(
          ParallelPolicy const parallel_policy,
          RandomAccessIterator const first, RandomAccessIterator const last,
          Function&& function)
        -> void
        {
          using qubit_type = ::ket::qubit<>;
          using state_integer_type = ::ket::meta::state_integer_t<qubit_type>;
          using bit_integer_type = ::ket::meta::bit_integer_t<qubit_type>;

          auto const state_size = static_cast<state_integer_type>(last - first);
          auto const num_qubits = ::ket::utility::integer_log2<bit_integer_type>(state_size);
          assert(::ket::utility::integer_exp2<state_integer_type>(num_qubits) == state_size);

#   ifndef KET_DEFAULT_NUM_ON_CACHE_QUBITS
#     define KET_DEFAULT_NUM_ON_CACHE_QUBITS 16
#   endif // KET_DEFAULT_NUM_ON_CACHE_QUBITS
          constexpr auto num_on_cache_qubits = bit_integer_type{KET_DEFAULT_NUM_ON_CACHE_QUBITS};
          constexpr auto cache_size = ::ket::utility::integer_exp2<state_integer_type>(num_on_cache_qubits);
          assert(num_on_cache_qubits < num_qubits);

          // xxxx|yyyy|zzzzzz: (local) qubits
          // * xxxx: off-cache qubits
          // * yyyy|zzzzzz: on-cache qubits
          //   - yyyy: chunk qubits (chunk qubits are determined dynamically, and sometimes there is no chunk qubit)

          std::array<qubit_type, 0u> unsorted_qubits{};
          std::array<qubit_type, 1u> sorted_qubits_with_sentinel{::ket::make_qubit<state_integer_type>(num_qubits)};

          for (auto iter = first; iter < last; iter += cache_size)
            ::ket::gate::gate_detail::gate_n(parallel_policy, iter, cache_size, unsorted_qubits, sorted_qubits_with_sentinel, std::forward<Function>(function));
        }

        template <typename ParallelPolicy, typename RandomAccessIterator, typename Function, typename Qubit, typename... Qubits>
        inline auto gate(
          ParallelPolicy const parallel_policy,
          RandomAccessIterator const first, RandomAccessIterator const last,
          Function&& function, Qubit&& qubit, Qubits&&... qubits)
        -> void
        {
          using state_integer_type = ::ket::meta::state_integer_t<std::remove_reference_t<Qubit>>;
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

          using bit_integer_type = ::ket::meta::bit_integer_t<std::remove_reference_t<Qubit>>;
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

          auto const state_size = static_cast<state_integer_type>(last - first);
          auto const num_qubits = ::ket::utility::integer_log2<bit_integer_type>(state_size);
          assert(::ket::utility::integer_exp2<state_integer_type>(num_qubits) == state_size);
          assert(::ket::utility::all_in_state_vector(num_qubits, qubit, qubits...));

#   ifndef KET_DEFAULT_NUM_ON_CACHE_QUBITS
#     define KET_DEFAULT_NUM_ON_CACHE_QUBITS 16
#   endif // KET_DEFAULT_NUM_ON_CACHE_QUBITS
          constexpr auto num_on_cache_qubits = bit_integer_type{KET_DEFAULT_NUM_ON_CACHE_QUBITS};
          constexpr auto cache_size = ::ket::utility::integer_exp2<state_integer_type>(num_on_cache_qubits);
          assert(num_on_cache_qubits < num_qubits);
          // It is required to be confirmed to satisfy Case 1)
          assert(::ket::utility::all_in_state_vector(num_on_cache_qubits, qubit, qubits...));

          constexpr auto num_operated_qubits = bit_integer_type{sizeof...(Qubits) + 1u};
          assert(num_operated_qubits < num_on_cache_qubits);

          // xxxx|yyyy|zzzzzz: (local) qubits
          // * xxxx: off-cache qubits
          // * yyyy|zzzzzz: on-cache qubits
          //   - yyyy: chunk qubits (chunk qubits are determined dynamically, and sometimes there is no chunk qubit)

          using qubit_type = ::ket::qubit<state_integer_type, bit_integer_type>;
          std::array<qubit_type, num_operated_qubits> unsorted_qubits{
            ::ket::remove_control(std::forward<Qubit>(qubit)), ::ket::remove_control(std::forward<Qubits>(qubits))...};

          std::array<qubit_type, num_operated_qubits + bit_integer_type{1u}> sorted_qubits_with_sentinel{
            ::ket::remove_control(qubit), ::ket::remove_control(qubits)..., ::ket::make_qubit<state_integer_type>(num_qubits)};
          using std::begin;
          using std::end;
          std::sort(begin(sorted_qubits_with_sentinel), std::prev(end(sorted_qubits_with_sentinel)));

          for (auto iter = first; iter < last; iter += cache_size)
            ::ket::gate::gate_detail::gate_n(parallel_policy, iter, cache_size, unsorted_qubits, sorted_qubits_with_sentinel, std::forward<Function>(function));
        }
      } // namespace all_on_cache

#   ifndef KET_USE_ON_CACHE_STATE_VECTOR
      // Case 2) There is no operated on-cache qubit
      //   ex: xxxx|yyy|zzzzzzz
      //       ^^ ^             <- operated qubits
      namespace none_on_cache
      {
        // First argument of Function: ::ket::gate::utility::cache_aware_iterator<RandomAccessIterator>
        template <typename ParallelPolicy, typename RandomAccessIterator, typename Function>
        inline auto gate(
          ParallelPolicy const parallel_policy,
          RandomAccessIterator const first, RandomAccessIterator const last,
          Function&& function)
        -> void
        {
          using qubit_type = ::ket::qubit<>;
          using state_integer_type = ::ket::meta::state_integer_t<qubit_type>;
          using bit_integer_type = ::ket::meta::bit_integer_t<qubit_type>;

          auto const state_size = static_cast<state_integer_type>(last - first);
          auto const num_qubits = ::ket::utility::integer_log2<bit_integer_type>(state_size);
          assert(::ket::utility::integer_exp2<state_integer_type>(num_qubits) == state_size);

#     ifndef KET_DEFAULT_NUM_ON_CACHE_QUBITS
#       define KET_DEFAULT_NUM_ON_CACHE_QUBITS 16
#     endif // KET_DEFAULT_NUM_ON_CACHE_QUBITS
          constexpr auto num_on_cache_qubits = bit_integer_type{KET_DEFAULT_NUM_ON_CACHE_QUBITS};
          constexpr auto cache_size = ::ket::utility::integer_exp2<state_integer_type>(num_on_cache_qubits);
          assert(num_on_cache_qubits < num_qubits);
          auto const num_off_cache_qubits = num_qubits - num_on_cache_qubits;

          // xxxx|yyyy|zzzzzz: (local) qubits
          // * xxxx: off-cache qubits
          // * yyyy|zzzzzz: on-cache qubits
          //   - yyyy: chunk qubits (chunk qubits are determined dynamically, and sometimes there is no chunk qubit)
          // * xxxx|yyyy: tag qubits
          // * zzzzzz: nontag qubits

          // chunk_size, num_tag_qubits
          auto const chunk_size = cache_size;
          auto const num_tag_qubits = num_off_cache_qubits;

          // unsorted_tag_qubits, sorted_tag_qubits_with_sentinel
          std::array<qubit_type, 0u> unsorted_tag_qubits{};
          std::array<qubit_type, 1u> sorted_tag_qubits_with_sentinel{qubit_type{num_tag_qubits}};

          // unsorted_on_cache_qubits, sorted_on_cache_qubits_with_sentinel
          std::array<qubit_type, 0u> unsorted_on_cache_qubits{};
          std::array<qubit_type, 1u> sorted_on_cache_qubits_with_sentinel{::ket::make_qubit<state_integer_type>(num_on_cache_qubits)};

          auto const tag_loop_size = ::ket::utility::integer_exp2<state_integer_type>(num_tag_qubits);
          for (auto tag_index_wo_qubits = state_integer_type{0u}; tag_index_wo_qubits < tag_loop_size; ++tag_index_wo_qubits)
            ::ket::gate::gate_detail::gate_n(
              parallel_policy,
              ::ket::gate::utility::make_cache_aware_iterator(
                first, tag_index_wo_qubits, chunk_size,
                unsorted_tag_qubits.data(), unsorted_tag_qubits.data(),
                sorted_tag_qubits_with_sentinel.data(), sorted_tag_qubits_with_sentinel.data() + 1u),
              cache_size, unsorted_on_cache_qubits, sorted_on_cache_qubits_with_sentinel, std::forward<Function>(function));
        }

        namespace impl
        {
          template <typename ParallelPolicy, typename RandomAccessIterator, typename Function, typename Qubit, typename... Qubits, std::size_t... indices_for_qubits>
          inline auto gate(
            std::index_sequence<indices_for_qubits...> const,
            ParallelPolicy const parallel_policy,
            RandomAccessIterator const first, RandomAccessIterator const last,
            Function&& function, Qubit&& qubit, Qubits&&... qubits)
          -> void
          {
            using state_integer_type = ::ket::meta::state_integer_t<std::remove_reference_t<Qubit>>;
            static_assert(std::is_unsigned<state_integer_type>::value, "state_integer_type of Qubit should be unsigned");
#     if __cpp_constexpr >= 201603L
            static_assert(
              ::ket::utility::variadic::proj::all_of(
                [](auto integer) { return std::is_same<decltype(integer), state_integer_type>::value; },
                [](auto qubit) { return ::ket::meta::state_integer_t<decltype(qubit)>{}; },
                std::remove_cv_t<std::remove_reference_t<Qubits>>{}...),
              "state_integer_type's of Qubit and Qubits should be the same");
#     else // __cpp_constexpr >= 201603L
            static_assert(
              ::ket::utility::variadic::proj::all_of(
                ::ket::gate::gate_detail::is_same_to<state_integer_type>{}, ::ket::gate::gate_detail::state_integer_of{},
                std::remove_cv_t<std::remove_reference_t<Qubits>>{}...),
              "state_integer_type's of Qubit and Qubits should be the same");
#     endif // __cpp_constexpr >= 201603L

            using bit_integer_type = ::ket::meta::bit_integer_t<std::remove_reference_t<Qubit>>;
            static_assert(std::is_unsigned<bit_integer_type>::value, "bit_integer_type of Qubit should be unsigned");
#     if __cpp_constexpr >= 201603L
            static_assert(
              ::ket::utility::variadic::proj::all_of(
                [](auto integer) { return std::is_same<decltype(integer), bit_integer_type>::value; },
                [](auto qubit) { return ::ket::meta::bit_integer_t<decltype(qubit)>{}; },
                std::remove_cv_t<std::remove_reference_t<Qubits>>{}...),
              "bit_integer_type's of Qubit and Qubits should be the same");
#     else // __cpp_constexpr >= 201603L
            static_assert(
              ::ket::utility::variadic::proj::all_of(
                ::ket::gate::gate_detail::is_same_to<bit_integer_type>{}, ::ket::gate::gate_detail::bit_integer_of{},
                std::remove_cv_t<std::remove_reference_t<Qubits>>{}...),
              "bit_integer_type's of Qubit and Qubits should be the same");
#     endif // __cpp_constexpr >= 201603L

            auto const state_size = static_cast<state_integer_type>(last - first);
            auto const num_qubits = ::ket::utility::integer_log2<bit_integer_type>(state_size);
            assert(::ket::utility::integer_exp2<state_integer_type>(num_qubits) == state_size);
            assert(::ket::utility::all_in_state_vector(num_qubits, qubit, qubits...));

#     ifndef KET_DEFAULT_NUM_ON_CACHE_QUBITS
#       define KET_DEFAULT_NUM_ON_CACHE_QUBITS 16
#     endif // KET_DEFAULT_NUM_ON_CACHE_QUBITS
            constexpr auto num_on_cache_qubits = bit_integer_type{KET_DEFAULT_NUM_ON_CACHE_QUBITS};
            constexpr auto cache_size = ::ket::utility::integer_exp2<state_integer_type>(num_on_cache_qubits);
            assert(num_on_cache_qubits < num_qubits);
            auto const num_off_cache_qubits = num_qubits - num_on_cache_qubits;
            // It is required to be confirmed not to satisfy Case 1)
            assert(not ::ket::utility::all_in_state_vector(num_on_cache_qubits, qubit, qubits...));

            constexpr auto num_operated_qubits = bit_integer_type{sizeof...(Qubits) + 1u};
            assert(num_operated_qubits < num_on_cache_qubits);

            // xxxx|yyyy|zzzzzz: (local) qubits
            // * xxxx: off-cache qubits
            // * yyyy|zzzzzz: on-cache qubits
            //   - yyyy: chunk qubits (chunk qubits are determined dynamically, and sometimes there is no chunk qubit)
            // * xxxx|yyyy: tag qubits
            // * zzzzzz: nontag qubits

            using qubit_type = ::ket::qubit<state_integer_type, bit_integer_type>;
            auto const least_significant_off_cache_qubit = qubit_type{num_on_cache_qubits};

            // num_chunk_qubits, chunk_size, least_significant_chunk_qubit, num_tag_qubits, num_nontag_qubits
            constexpr auto num_chunk_qubits = num_operated_qubits;
            constexpr auto num_chunks_in_on_cache_state = ::ket::utility::integer_exp2<state_integer_type>(num_chunk_qubits);
            auto const chunk_size = cache_size / num_chunks_in_on_cache_state;
            auto const least_significant_chunk_qubit = least_significant_off_cache_qubit - num_chunk_qubits;
            auto const num_tag_qubits = num_off_cache_qubits + num_chunk_qubits;
            auto const num_nontag_qubits = num_on_cache_qubits - num_chunk_qubits;

            // unsorted_tag_qubits, sorted_tag_qubits_with_sentinel
            std::array<qubit_type, num_operated_qubits> unsorted_tag_qubits{::ket::remove_control(qubit) - num_nontag_qubits, (::ket::remove_control(qubits) - num_nontag_qubits)...};
            std::array<qubit_type, num_operated_qubits + bit_integer_type{1u}> sorted_tag_qubits_with_sentinel{
              ::ket::remove_control(qubit) - num_nontag_qubits, (::ket::remove_control(qubits) - num_nontag_qubits)..., qubit_type{num_tag_qubits}};
            using std::begin;
            using std::end;
            std::sort(begin(sorted_tag_qubits_with_sentinel), std::prev(end(sorted_tag_qubits_with_sentinel)));

            // unsorted_on_cache_qubits, sorted_on_cache_qubits_with_sentinel
            std::array<qubit_type, num_operated_qubits> unsorted_on_cache_qubits{least_significant_chunk_qubit, (least_significant_chunk_qubit + bit_integer_type{1u} + bit_integer_type{indices_for_qubits})...};
            std::array<qubit_type, num_operated_qubits + bit_integer_type{1u}> sorted_on_cache_qubits_with_sentinel{least_significant_chunk_qubit, (least_significant_chunk_qubit + bit_integer_type{1u} + bit_integer_type{indices_for_qubits})..., ::ket::make_qubit<state_integer_type>(num_on_cache_qubits)};

            auto const tag_loop_size = ::ket::utility::integer_exp2<state_integer_type>(num_tag_qubits - num_operated_qubits);
            for (auto tag_index_wo_qubits = state_integer_type{0u}; tag_index_wo_qubits < tag_loop_size; ++tag_index_wo_qubits)
              ::ket::gate::gate_detail::gate_n(
                parallel_policy,
                ::ket::gate::utility::make_cache_aware_iterator(
                  first, tag_index_wo_qubits, chunk_size,
                  unsorted_tag_qubits.data(), unsorted_tag_qubits.data() + unsorted_tag_qubits.size(),
                  sorted_tag_qubits_with_sentinel.data(), sorted_tag_qubits_with_sentinel.data() + sorted_tag_qubits_with_sentinel.size()),
                cache_size, unsorted_on_cache_qubits, sorted_on_cache_qubits_with_sentinel, std::forward<Function>(function));
          }
        } // namespace impl

        template <
          typename ParallelPolicy, typename RandomAccessIterator, typename Function, typename Qubit, typename... Qubits,
          typename IndicesForQubits = std::make_index_sequence<sizeof...(Qubits)>>
        inline auto gate(
          ParallelPolicy const parallel_policy,
          RandomAccessIterator const first, RandomAccessIterator const last,
          Function&& function, Qubit&& qubit, Qubits&&... qubits)
        -> void
        {
          ::ket::gate::cache::none_on_cache::impl::gate(
            IndicesForQubits{},
            parallel_policy, first, last, std::forward<Function>(function), std::forward<Qubit>(qubit), std::forward<Qubits>(qubits)...);
        }
      } // namespace none_on_cache

      // Case 3) There are some operated on-cache qubits
      //   ex: xxxx|yyy|zzzzzzz
      //        ^^   ^    ^     <- operated qubits
      namespace some_on_cache
      {
        // First argument of Function: ::ket::gate::utility::runtime::cache_aware_iterator<RandomAccessIterator>
        template <typename ParallelPolicy, typename RandomAccessIterator, typename Function>
        inline auto gate(
          ParallelPolicy const parallel_policy,
          RandomAccessIterator const first, RandomAccessIterator const last,
          Function&& function)
        -> void
        {
          using qubit_type = ::ket::qubit<>;
          using state_integer_type = ::ket::meta::state_integer_t<qubit_type>;
          using bit_integer_type = ::ket::meta::bit_integer_t<qubit_type>;

          auto const state_size = static_cast<state_integer_type>(last - first);
          auto const num_qubits = ::ket::utility::integer_log2<bit_integer_type>(state_size);
          assert(::ket::utility::integer_exp2<state_integer_type>(num_qubits) == state_size);

#     ifndef KET_DEFAULT_NUM_ON_CACHE_QUBITS
#       define KET_DEFAULT_NUM_ON_CACHE_QUBITS 16
#     endif // KET_DEFAULT_NUM_ON_CACHE_QUBITS
          constexpr auto num_on_cache_qubits = bit_integer_type{KET_DEFAULT_NUM_ON_CACHE_QUBITS};
          constexpr auto cache_size = ::ket::utility::integer_exp2<state_integer_type>(num_on_cache_qubits);
          assert(num_on_cache_qubits < num_qubits);
          auto const num_off_cache_qubits = num_qubits - num_on_cache_qubits;

          // xxxx|yyyy|zzzzzz: (local) qubits
          // * xxxx: off-cache qubits
          // * yyyy|zzzzzz: on-cache qubits
          //   - yyyy: chunk qubits (chunk qubits are determined dynamically, and sometimes there is no chunk qubit)
          // * xxxx|yyyy: tag qubits
          // * zzzzzz: nontag qubits

          // chunk_size, num_tag_qubits
          auto const chunk_size = cache_size;
          auto const num_tag_qubits = num_off_cache_qubits;

          // unsorted_tag_qubits, sorted_tag_qubits_with_sentinel
          std::array<qubit_type, 0u> unsorted_tag_qubits{};
          std::array<qubit_type, 1u> sorted_tag_qubits_with_sentinel{qubit_type{num_tag_qubits}};

          // unsorted_on_cache_qubits, sorted_on_cache_qubits_with_sentinel
          std::array<qubit_type, 0u> unsorted_on_cache_qubits{};
          std::array<qubit_type, 1u> sorted_on_cache_qubits_with_sentinel{::ket::make_qubit<state_integer_type>(num_on_cache_qubits)};

          auto const tag_loop_size = ::ket::utility::integer_exp2<state_integer_type>(num_tag_qubits);
          for (auto tag_index_wo_qubits = state_integer_type{0u}; tag_index_wo_qubits < tag_loop_size; ++tag_index_wo_qubits)
            ::ket::gate::gate_detail::gate_n(
              parallel_policy,
              ::ket::gate::utility::make_cache_aware_iterator(
                first, tag_index_wo_qubits, chunk_size,
                unsorted_tag_qubits.data(), unsorted_tag_qubits.data(),
                sorted_tag_qubits_with_sentinel.data(), sorted_tag_qubits_with_sentinel.data() + 1u),
              cache_size, unsorted_on_cache_qubits, sorted_on_cache_qubits_with_sentinel, std::forward<Function>(function));
        }

        namespace impl
        {
          template <typename ParallelPolicy, typename RandomAccessIterator, typename Function, typename Qubit, typename... Qubits, std::size_t... indices_for_qubits>
          inline auto gate(
            std::index_sequence<indices_for_qubits...> const,
            ParallelPolicy const parallel_policy,
            RandomAccessIterator const first, RandomAccessIterator const last,
            Function&& function, Qubit&& qubit, Qubits&&... qubits)
          -> void
          {
            using state_integer_type = ::ket::meta::state_integer_t<std::remove_reference_t<Qubit>>;
            static_assert(std::is_unsigned<state_integer_type>::value, "state_integer_type of Qubit should be unsigned");
#     if __cpp_constexpr >= 201603L
            static_assert(
              ::ket::utility::variadic::proj::all_of(
                [](auto integer) { return std::is_same<decltype(integer), state_integer_type>::value; },
                [](auto qubit) { return ::ket::meta::state_integer_t<decltype(qubit)>{}; },
                std::remove_cv_t<std::remove_reference_t<Qubits>>{}...),
              "state_integer_type's of Qubit and Qubits should be the same");
#     else // __cpp_constexpr >= 201603L
            static_assert(
              ::ket::utility::variadic::proj::all_of(
                ::ket::gate::gate_detail::is_same_to<state_integer_type>{}, ::ket::gate::gate_detail::state_integer_of{},
                std::remove_cv_t<std::remove_reference_t<Qubits>>{}...),
              "state_integer_type's of Qubit and Qubits should be the same");
#     endif // __cpp_constexpr >= 201603L

            using bit_integer_type = ::ket::meta::bit_integer_t<std::remove_reference_t<Qubit>>;
            static_assert(std::is_unsigned<bit_integer_type>::value, "bit_integer_type of Qubit should be unsigned");
#     if __cpp_constexpr >= 201603L
            static_assert(
              ::ket::utility::variadic::proj::all_of(
                [](auto integer) { return std::is_same<decltype(integer), bit_integer_type>::value; },
                [](auto qubit) { return ::ket::meta::bit_integer_t<decltype(qubit)>{}; },
                std::remove_cv_t<std::remove_reference_t<Qubits>>{}...),
              "bit_integer_type's of Qubit and Qubits should be the same");
#     else // __cpp_constexpr >= 201603L
            static_assert(
              ::ket::utility::variadic::proj::all_of(
                ::ket::gate::gate_detail::is_same_to<bit_integer_type>{}, ::ket::gate::gate_detail::bit_integer_of{},
                std::remove_cv_t<std::remove_reference_t<Qubits>>{}...),
              "bit_integer_type's of Qubit and Qubits should be the same");
#     endif // __cpp_constexpr >= 201603L

            auto const state_size = static_cast<state_integer_type>(last - first);
            auto const num_qubits = ::ket::utility::integer_log2<bit_integer_type>(state_size);
            assert(::ket::utility::integer_exp2<state_integer_type>(num_qubits) == state_size);
            assert(::ket::utility::all_in_state_vector(num_qubits, qubit, qubits...));

#     ifndef KET_DEFAULT_NUM_ON_CACHE_QUBITS
#       define KET_DEFAULT_NUM_ON_CACHE_QUBITS 16
#     endif // KET_DEFAULT_NUM_ON_CACHE_QUBITS
            constexpr auto num_on_cache_qubits = bit_integer_type{KET_DEFAULT_NUM_ON_CACHE_QUBITS};
            constexpr auto cache_size = ::ket::utility::integer_exp2<state_integer_type>(num_on_cache_qubits);
            assert(num_on_cache_qubits < num_qubits);
            auto const num_off_cache_qubits = num_qubits - num_on_cache_qubits;
            // It is required to be confirmed not to satisfy Case 1)
            assert(not ::ket::utility::all_in_state_vector(num_on_cache_qubits, qubit, qubits...));
            // It is required to be confirmed not to satisfy Case 2)
            assert(not ::ket::utility::none_in_state_vector(num_on_cache_qubits, qubit, qubits...));

            constexpr auto num_operated_qubits = bit_integer_type{sizeof...(Qubits) + 1u};
            assert(num_operated_qubits < num_on_cache_qubits);

            // xxxx|yyyy|zzzzzz: (local) qubits
            // * xxxx: off-cache qubits
            // * yyyy|zzzzzz: on-cache qubits
            //   - yyyy: chunk qubits (chunk qubits are determined dynamically, and sometimes there is no chunk qubit)
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
            auto const operated_off_cache_qubits_first = operated_on_cache_qubits_last;
            auto const operated_off_cache_qubits_last = end(sorted_qubits);
            // from Assumption: Case 3) Some of the operated qubits are off-cache qubits
            assert(operated_on_cache_qubits_first != operated_on_cache_qubits_last);
            assert(operated_off_cache_qubits_first != operated_off_cache_qubits_last);

            // least_significant_chunk_qubit, num_chunk_qubits, chunk_size, num_tag_qubits, num_nontag_qubits
            auto operated_on_cache_qubits_iter = std::prev(operated_on_cache_qubits_last);
            auto free_most_significant_on_cache_qubit = least_significant_off_cache_qubit - bit_integer_type{1u};
            auto const num_operated_off_cache_qubits
              = static_cast<bit_integer_type>(operated_off_cache_qubits_last - operated_off_cache_qubits_first);
            for (auto num_found_operated_off_cache_qubits = bit_integer_type{0u};
                 num_found_operated_off_cache_qubits < num_operated_off_cache_qubits; ++num_found_operated_off_cache_qubits)
              while (free_most_significant_on_cache_qubit-- == *operated_on_cache_qubits_iter)
                if (operated_on_cache_qubits_iter != operated_on_cache_qubits_first)
                  --operated_on_cache_qubits_iter;
            auto const least_significant_chunk_qubit = free_most_significant_on_cache_qubit + bit_integer_type{1u};
            auto const num_chunk_qubits = static_cast<bit_integer_type>(least_significant_off_cache_qubit - least_significant_chunk_qubit);
            assert(num_chunk_qubits <= num_operated_qubits);
            auto const num_chunks_in_on_cache_state = ::ket::utility::integer_exp2<state_integer_type>(num_chunk_qubits);
            auto const chunk_size = cache_size / num_chunks_in_on_cache_state;
            auto const num_tag_qubits = num_off_cache_qubits + num_chunk_qubits;
            auto const num_nontag_qubits = num_on_cache_qubits - num_chunk_qubits;

            // unsorted_tag_qubits, modified_operated_qubits
            auto unsorted_tag_qubits = std::vector<qubit_type>{};
            unsorted_tag_qubits.reserve(num_chunk_qubits);
            auto present_chunk_qubit = least_significant_chunk_qubit;
            auto const modified_operated_qubits
              = ::ket::utility::variadic::transform(
                  [least_significant_chunk_qubit, num_nontag_qubits, &unsorted_tag_qubits, &present_chunk_qubit](auto qubit)
                  {
                    if (qubit < least_significant_chunk_qubit)
                      return qubit;

                    unsorted_tag_qubits.push_back(::ket::remove_control(qubit) - num_nontag_qubits);
                    return static_cast<decltype(qubit)>(present_chunk_qubit++);
                  },
                  qubit, qubits...);
            assert(present_chunk_qubit == least_significant_off_cache_qubit);
            assert(static_cast<bit_integer_type>(unsorted_tag_qubits.size()) == num_chunk_qubits);

            // sorted_tag_qubits_with_sentinel
            auto sorted_tag_qubits_with_sentinel = std::vector<qubit_type>{};
            sorted_tag_qubits_with_sentinel.reserve(unsorted_tag_qubits.size() + 1u);
            std::copy(begin(unsorted_tag_qubits), end(unsorted_tag_qubits), std::back_inserter(sorted_tag_qubits_with_sentinel));
            sorted_tag_qubits_with_sentinel.push_back(qubit_type{num_tag_qubits});
            std::sort(begin(sorted_tag_qubits_with_sentinel), std::prev(end(sorted_tag_qubits_with_sentinel)));
            assert(sorted_tag_qubits_with_sentinel.size() == unsorted_tag_qubits.size() + 1u);

            // unsorted_on_cache_qubits, sorted_on_cache_qubits_with_sentinel
            std::array<qubit_type, num_operated_qubits> unsorted_on_cache_qubits{ket::remove_control(std::get<0u>(modified_operated_qubits)), ket::remove_control(std::get<1u + indices_for_qubits>(modified_operated_qubits))...};
            std::array<qubit_type, num_operated_qubits + bit_integer_type{1u}> sorted_on_cache_qubits_with_sentinel{ket::remove_control(std::get<0u>(modified_operated_qubits)), ket::remove_control(std::get<1u + indices_for_qubits>(modified_operated_qubits))..., ::ket::make_qubit<state_integer_type>(num_on_cache_qubits)};
            std::sort(begin(sorted_on_cache_qubits_with_sentinel), std::prev(end(sorted_on_cache_qubits_with_sentinel)));

            auto const tag_loop_size = ::ket::utility::integer_exp2<state_integer_type>(num_tag_qubits - num_chunk_qubits); // num_chunk_qubits == operated_tag_qubits.size()
            for (auto tag_index_wo_qubits = state_integer_type{0u}; tag_index_wo_qubits < tag_loop_size; ++tag_index_wo_qubits)
              ::ket::gate::gate_detail::gate_n(
                parallel_policy,
                ::ket::gate::utility::make_cache_aware_iterator(
                  first, tag_index_wo_qubits, chunk_size,
                  unsorted_tag_qubits.data(), unsorted_tag_qubits.data() + unsorted_tag_qubits.size(),
                  sorted_tag_qubits_with_sentinel.data(), sorted_tag_qubits_with_sentinel.data() + sorted_tag_qubits_with_sentinel.size()),
                cache_size, unsorted_on_cache_qubits, sorted_on_cache_qubits_with_sentinel, std::forward<Function>(function));
          }
        } // namespace impl

        template <
          typename ParallelPolicy, typename RandomAccessIterator, typename Function, typename Qubit, typename... Qubits,
          typename IndicesForQubits = std::make_index_sequence<sizeof...(Qubits)>>
        inline auto gate(
          ParallelPolicy const parallel_policy,
          RandomAccessIterator const first, RandomAccessIterator const last,
          Function&& function, Qubit&& qubit, Qubits&&... qubits)
        -> void
        {
          ::ket::gate::cache::some_on_cache::impl::gate(
            IndicesForQubits{},
            parallel_policy, first, last, std::forward<Function>(function), std::forward<Qubit>(qubit), std::forward<Qubits>(qubits)...);
        }
      } // namespace some_on_cache
#   else // KET_USE_ON_CACHE_STATE_VECTOR
      // Case 2) There is no operated on-cache qubit
      //   ex: xxxx|yyy|zzzzzzz
      //       ^^ ^             <- operated qubits
      namespace none_on_cache
      {
        // First argument of Function: RandomAccessIterator2 (not RandomAccessIterator1)
        namespace gate_detail
        {
          template <bool is_state_iterator_mutable>
          struct gate
          {
            template <typename ParallelPolicy, typename RandomAccessIterator1, typename RandomAccessIterator2, typename Function>
            static auto call(
              ParallelPolicy const parallel_policy,
              RandomAccessIterator1 const state_first, RandomAccessIterator1 const state_last,
              RandomAccessIterator2 const on_cache_state_first, RandomAccessIterator2 const on_cache_state_last,
              Function&& function)
            -> void
            {
              using qubit_type = ::ket::qubit<>;
              using state_integer_type = ::ket::meta::state_integer_t<qubit_type>;
              using bit_integer_type = ::ket::meta::bit_integer_t<qubit_type>;

              auto const state_size = static_cast<state_integer_type>(state_last - state_first);
              auto const num_qubits = ::ket::utility::integer_log2<bit_integer_type>(state_size);
              assert(::ket::utility::integer_exp2<state_integer_type>(num_qubits) == state_size);

              auto const cache_size = static_cast<state_integer_type>(on_cache_state_last - on_cache_state_first);
              auto const num_on_cache_qubits = ::ket::utility::integer_log2<bit_integer_type>(cache_size);
              assert(::ket::utility::integer_exp2<state_integer_type>(num_on_cache_qubits) == cache_size);
              assert(num_on_cache_qubits < num_qubits);
              auto const num_off_cache_qubits = num_qubits - num_on_cache_qubits;

              // xxxx|yyyy|zzzzzz: (local) qubits
              // * xxxx: off-cache qubits
              // * yyyy|zzzzzz: on-cache qubits
              //   - yyyy: chunk qubits (chunk qubits are determined dynamically, and sometimes there is no chunk qubit)
              // * xxxx|yyyy: tag qubits
              // * zzzzzz: nontag qubits

              // chunk_size, num_tag_qubits
              auto const chunk_size = cache_size;
              auto const num_tag_qubits = num_off_cache_qubits;

              // unsorted_tag_qubits, sorted_tag_qubits_with_sentinel
              std::array<qubit_type, 0u> unsorted_tag_qubits{};
              std::array<qubit_type, 1u> sorted_tag_qubits_with_sentinel{qubit_type{num_tag_qubits}};

              // unsorted_on_cache_qubits, sorted_on_cache_qubits_with_sentinel
              std::array<qubit_type, 0u> unsorted_on_cache_qubits{};
              std::array<qubit_type, 1u> sorted_on_cache_qubits_with_sentinel{::ket::make_qubit<state_integer_type>(num_on_cache_qubits)};

              auto const tag_loop_size = ::ket::utility::integer_exp2<state_integer_type>(num_tag_qubits);
              for (auto tag_index_wo_qubits = state_integer_type{0u}; tag_index_wo_qubits < tag_loop_size; ++tag_index_wo_qubits)
              {
                using std::begin;
                using std::end;
                ::ket::utility::copy_n(
                  parallel_policy,
                  state_first
                  + ::ket::gate::utility::index_with_qubits(
                      tag_index_wo_qubits, state_integer_type{0u},
                      begin(unsorted_tag_qubits), end(unsorted_tag_qubits),
                      begin(sorted_tag_qubits_with_sentinel), end(sorted_tag_qubits_with_sentinel)) * chunk_size,
                  chunk_size, on_cache_state_first);

                ::ket::gate::gate_detail::gate_n(parallel_policy, on_cache_state_first, cache_size, unsorted_on_cache_qubits, sorted_on_cache_qubits_with_sentinel, std::forward<Function>(function));

                ::ket::utility::copy_n(
                  parallel_policy,
                  on_cache_state_first, chunk_size,
                  state_first
                  + ::ket::gate::utility::index_with_qubits(
                      tag_index_wo_qubits, state_integer_type{0u},
                      begin(unsorted_tag_qubits), end(unsorted_tag_qubits),
                      begin(sorted_tag_qubits_with_sentinel), end(sorted_tag_qubits_with_sentinel)) * chunk_size);
              }
            }
          }; // struct gate<is_state_iterator_mutable>

          template <>
          struct gate<false>
          {
            template <typename ParallelPolicy, typename RandomAccessIterator1, typename RandomAccessIterator2, typename Function>
            static auto call(
              ParallelPolicy const parallel_policy,
              RandomAccessIterator1 const state_first, RandomAccessIterator1 const state_last,
              RandomAccessIterator2 const on_cache_state_first, RandomAccessIterator2 const on_cache_state_last,
              Function&& function)
            -> void
            {
              using qubit_type = ::ket::qubit<>;
              using state_integer_type = ::ket::meta::state_integer_t<qubit_type>;
              using bit_integer_type = ::ket::meta::bit_integer_t<qubit_type>;

              auto const state_size = static_cast<state_integer_type>(state_last - state_first);
              auto const num_qubits = ::ket::utility::integer_log2<bit_integer_type>(state_size);
              assert(::ket::utility::integer_exp2<state_integer_type>(num_qubits) == state_size);

              auto const cache_size = static_cast<state_integer_type>(on_cache_state_last - on_cache_state_first);
              auto const num_on_cache_qubits = ::ket::utility::integer_log2<bit_integer_type>(cache_size);
              assert(::ket::utility::integer_exp2<state_integer_type>(num_on_cache_qubits) == cache_size);
              assert(num_on_cache_qubits < num_qubits);
              auto const num_off_cache_qubits = num_qubits - num_on_cache_qubits;

              // xxxx|yyyy|zzzzzz: (local) qubits
              // * xxxx: off-cache qubits
              // * yyyy|zzzzzz: on-cache qubits
              //   - yyyy: chunk qubits (chunk qubits are determined dynamically, and sometimes there is no chunk qubit)
              // * xxxx|yyyy: tag qubits
              // * zzzzzz: nontag qubits

              // chunk_size, num_tag_qubits
              auto const chunk_size = cache_size;
              auto const num_tag_qubits = num_off_cache_qubits;

              // unsorted_tag_qubits, sorted_tag_qubits_with_sentinel
              std::array<qubit_type, 0u> unsorted_tag_qubits{};
              std::array<qubit_type, 1u> sorted_tag_qubits_with_sentinel{qubit_type{num_tag_qubits}};

              // unsorted_on_cache_qubits, sorted_on_cache_qubits_with_sentinel
              std::array<qubit_type, 0u> unsorted_on_cache_qubits{};
              std::array<qubit_type, 1u> sorted_on_cache_qubits_with_sentinel{::ket::make_qubit<state_integer_type>(num_on_cache_qubits)};

              auto const tag_loop_size = ::ket::utility::integer_exp2<state_integer_type>(num_tag_qubits);
              for (auto tag_index_wo_qubits = state_integer_type{0u}; tag_index_wo_qubits < tag_loop_size; ++tag_index_wo_qubits)
              {
                using std::begin;
                using std::end;
                ::ket::utility::copy_n(
                  parallel_policy,
                  state_first
                  + ::ket::gate::utility::index_with_qubits(
                      tag_index_wo_qubits, state_integer_type{0u},
                      begin(unsorted_tag_qubits), end(unsorted_tag_qubits),
                      begin(sorted_tag_qubits_with_sentinel), end(sorted_tag_qubits_with_sentinel)) * chunk_size,
                  chunk_size, on_cache_state_first);

                ::ket::gate::gate_detail::gate_n(parallel_policy, on_cache_state_first, cache_size, unsorted_on_cache_qubits, sorted_on_cache_qubits_with_sentinel, std::forward<Function>(function));
              }
            }
          }; // struct gate<false>
        } // namespace gate_detail

        template <typename ParallelPolicy, typename RandomAccessIterator1, typename RandomAccessIterator2, typename Function>
        inline auto gate(
          ParallelPolicy const parallel_policy,
          RandomAccessIterator1 const state_first, RandomAccessIterator1 const state_last,
          RandomAccessIterator2 const on_cache_state_first, RandomAccessIterator2 const on_cache_state_last,
          Function&& function)
        -> void
        {
          ::ket::gate::cache::none_on_cache::gate_detail::gate<std::is_assignable<decltype(*state_first), typename std::iterator_traits<RandomAccessIterator1>::value_type>::value>::call(
            parallel_policy, state_first, state_last, on_cache_state_first, on_cache_state_last,
            std::forward<Function>(function));
        }

        namespace impl
        {
          template <bool is_state_iterator_mutable>
          struct gate_impl
          {
            template <
              typename ParallelPolicy, typename RandomAccessIterator1, typename RandomAccessIterator2,
              typename Qubit, std::size_t num_operated_qubits, typename Function>
            static auto call(
              ParallelPolicy const parallel_policy,
              RandomAccessIterator1 const state_first, RandomAccessIterator1 const state_last,
              RandomAccessIterator2 const on_cache_state_first, RandomAccessIterator2 const on_cache_state_last,
              std::array<Qubit, num_operated_qubits> const& unsorted_tag_qubits,
              std::array<Qubit, num_operated_qubits + 1u> const& sorted_tag_qubits_with_sentinel,
              std::array<Qubit, num_operated_qubits> const& unsorted_on_cache_qubits,
              std::array<Qubit, num_operated_qubits + 1u> const& sorted_on_cache_qubits_with_sentinel,
              Function&& function)
            -> void
            {
              using state_integer_type = ::ket::meta::state_integer_t<Qubit>;
              using bit_integer_type = ::ket::meta::bit_integer_t<Qubit>;

              auto const state_size = static_cast<state_integer_type>(state_last - state_first);
              auto const num_qubits = ::ket::utility::integer_log2<bit_integer_type>(state_size);

              auto const cache_size = static_cast<state_integer_type>(on_cache_state_last - on_cache_state_first);
              auto const num_on_cache_qubits = ::ket::utility::integer_log2<bit_integer_type>(cache_size);
              auto const num_off_cache_qubits = num_qubits - num_on_cache_qubits;

              // num_chunk_qubits, chunk_size, num_tag_qubits
              constexpr auto num_chunk_qubits = num_operated_qubits;
              constexpr auto num_chunks_in_on_cache_state = ::ket::utility::integer_exp2<state_integer_type>(num_chunk_qubits);
              auto const chunk_size = cache_size / num_chunks_in_on_cache_state;
              auto const num_tag_qubits = num_off_cache_qubits + num_chunk_qubits;

              auto const tag_loop_size = ::ket::utility::integer_exp2<state_integer_type>(num_tag_qubits - num_operated_qubits);
              for (auto tag_index_wo_qubits = state_integer_type{0u}; tag_index_wo_qubits < tag_loop_size; ++tag_index_wo_qubits)
              {
                using std::begin;
                using std::end;
                for (auto chunk_index = state_integer_type{0u}; chunk_index < num_chunks_in_on_cache_state; ++chunk_index)
                  ::ket::utility::copy_n(
                    parallel_policy,
                    state_first
                    + ::ket::gate::utility::index_with_qubits(
                        tag_index_wo_qubits, chunk_index,
                        begin(unsorted_tag_qubits), end(unsorted_tag_qubits),
                        begin(sorted_tag_qubits_with_sentinel), end(sorted_tag_qubits_with_sentinel)) * chunk_size,
                    chunk_size, on_cache_state_first + chunk_index * chunk_size);

                ::ket::gate::gate_detail::gate_n(parallel_policy, on_cache_state_first, cache_size, unsorted_on_cache_qubits, sorted_on_cache_qubits_with_sentinel, std::forward<Function>(function));

                for (auto chunk_index = state_integer_type{0u}; chunk_index < num_chunks_in_on_cache_state; ++chunk_index)
                  ::ket::utility::copy_n(
                    parallel_policy,
                    on_cache_state_first + chunk_index * chunk_size, chunk_size,
                    state_first
                    + ::ket::gate::utility::index_with_qubits(
                        tag_index_wo_qubits, chunk_index,
                        begin(unsorted_tag_qubits), end(unsorted_tag_qubits),
                        begin(sorted_tag_qubits_with_sentinel), end(sorted_tag_qubits_with_sentinel)) * chunk_size);
              }
            }
          }; // struct gate_impl<is_state_iterator_mutable>

          template <>
          struct gate_impl<false>
          {
            template <
              typename ParallelPolicy, typename RandomAccessIterator1, typename RandomAccessIterator2,
              typename Qubit, std::size_t num_operated_qubits, typename Function>
            static auto call(
              ParallelPolicy const parallel_policy,
              RandomAccessIterator1 const state_first, RandomAccessIterator1 const state_last,
              RandomAccessIterator2 const on_cache_state_first, RandomAccessIterator2 const on_cache_state_last,
              std::array<Qubit, num_operated_qubits> const& unsorted_tag_qubits,
              std::array<Qubit, num_operated_qubits + 1u> const& sorted_tag_qubits_with_sentinel,
              std::array<Qubit, num_operated_qubits> const& unsorted_on_cache_qubits,
              std::array<Qubit, num_operated_qubits + 1u> const& sorted_on_cache_qubits_with_sentinel,
              Function&& function)
            -> void
            {
              using state_integer_type = ::ket::meta::state_integer_t<Qubit>;
              using bit_integer_type = ::ket::meta::bit_integer_t<Qubit>;

              auto const state_size = static_cast<state_integer_type>(state_last - state_first);
              auto const num_qubits = ::ket::utility::integer_log2<bit_integer_type>(state_size);

              auto const cache_size = static_cast<state_integer_type>(on_cache_state_last - on_cache_state_first);
              auto const num_on_cache_qubits = ::ket::utility::integer_log2<bit_integer_type>(cache_size);
              auto const num_off_cache_qubits = num_qubits - num_on_cache_qubits;

              // num_chunk_qubits, chunk_size, num_tag_qubits
              constexpr auto num_chunk_qubits = num_operated_qubits;
              constexpr auto num_chunks_in_on_cache_state = ::ket::utility::integer_exp2<state_integer_type>(num_chunk_qubits);
              auto const chunk_size = cache_size / num_chunks_in_on_cache_state;
              auto const num_tag_qubits = num_off_cache_qubits + num_chunk_qubits;

              auto const tag_loop_size = ::ket::utility::integer_exp2<state_integer_type>(num_tag_qubits - num_operated_qubits);
              for (auto tag_index_wo_qubits = state_integer_type{0u}; tag_index_wo_qubits < tag_loop_size; ++tag_index_wo_qubits)
              {
                using std::begin;
                using std::end;
                for (auto chunk_index = state_integer_type{0u}; chunk_index < num_chunks_in_on_cache_state; ++chunk_index)
                  ::ket::utility::copy_n(
                    parallel_policy,
                    state_first
                    + ::ket::gate::utility::index_with_qubits(
                        tag_index_wo_qubits, chunk_index,
                        begin(unsorted_tag_qubits), end(unsorted_tag_qubits),
                        begin(sorted_tag_qubits_with_sentinel), end(sorted_tag_qubits_with_sentinel)) * chunk_size,
                    chunk_size, on_cache_state_first + chunk_index * chunk_size);

                ::ket::gate::gate_detail::gate_n(parallel_policy, on_cache_state_first, cache_size, unsorted_on_cache_qubits, sorted_on_cache_qubits_with_sentinel, std::forward<Function>(function));
              }
            }
          }; // struct gate_impl<false>

          template <
            typename ParallelPolicy, typename RandomAccessIterator1, typename RandomAccessIterator2,
            typename Function, typename Qubit, typename... Qubits, std::size_t... indices_for_qubits>
          inline auto gate(
            std::index_sequence<indices_for_qubits...> const,
            ParallelPolicy const parallel_policy,
            RandomAccessIterator1 const state_first, RandomAccessIterator1 const state_last,
            RandomAccessIterator2 const on_cache_state_first, RandomAccessIterator2 const on_cache_state_last,
            Function&& function, Qubit&& qubit, Qubits&&... qubits)
          -> void
          {
            using state_integer_type = ::ket::meta::state_integer_t<std::remove_reference_t<Qubit>>;
            static_assert(std::is_unsigned<state_integer_type>::value, "state_integer_type of Qubit should be unsigned");
#     if __cpp_constexpr >= 201603L
            static_assert(
              ::ket::utility::variadic::proj::all_of(
                [](auto integer) { return std::is_same<decltype(integer), state_integer_type>::value; },
                [](auto qubit) { return ::ket::meta::state_integer_t<decltype(qubit)>{}; },
                std::remove_cv_t<std::remove_reference_t<Qubits>>{}...),
              "state_integer_type's of Qubit and Qubits should be the same");
#     else // __cpp_constexpr >= 201603L
            static_assert(
              ::ket::utility::variadic::proj::all_of(
                ::ket::gate::gate_detail::is_same_to<state_integer_type>{}, ::ket::gate::gate_detail::state_integer_of{},
                std::remove_cv_t<std::remove_reference_t<Qubits>>{}...),
              "state_integer_type's of Qubit and Qubits should be the same");
#     endif // __cpp_constexpr >= 201603L

            using bit_integer_type = ::ket::meta::bit_integer_t<std::remove_reference_t<Qubit>>;
            static_assert(std::is_unsigned<bit_integer_type>::value, "bit_integer_type of Qubit should be unsigned");
#     if __cpp_constexpr >= 201603L
            static_assert(
              ::ket::utility::variadic::proj::all_of(
                [](auto integer) { return std::is_same<decltype(integer), bit_integer_type>::value; },
                [](auto qubit) { return ::ket::meta::bit_integer_t<decltype(qubit)>{}; },
                std::remove_cv_t<std::remove_reference_t<Qubits>>{}...),
              "bit_integer_type's of Qubit and Qubits should be the same");
#     else // __cpp_constexpr >= 201603L
            static_assert(
              ::ket::utility::variadic::proj::all_of(
                ::ket::gate::gate_detail::is_same_to<bit_integer_type>{}, ::ket::gate::gate_detail::bit_integer_of{},
                std::remove_cv_t<std::remove_reference_t<Qubits>>{}...),
              "bit_integer_type's of Qubit and Qubits should be the same");
#     endif // __cpp_constexpr >= 201603L

            auto const state_size = static_cast<state_integer_type>(state_last - state_first);
            auto const num_qubits = ::ket::utility::integer_log2<bit_integer_type>(state_size);
            assert(::ket::utility::integer_exp2<state_integer_type>(num_qubits) == state_size);
            assert(::ket::utility::all_in_state_vector(num_qubits, qubit, qubits...));

            auto const cache_size = static_cast<state_integer_type>(on_cache_state_last - on_cache_state_first);
            auto const num_on_cache_qubits = ::ket::utility::integer_log2<bit_integer_type>(cache_size);
            assert(::ket::utility::integer_exp2<state_integer_type>(num_on_cache_qubits) == cache_size);
            assert(num_on_cache_qubits < num_qubits);
            auto const num_off_cache_qubits = num_qubits - num_on_cache_qubits;
            // It is required to be confirmed not to satisfy Case 1)
            assert(not ::ket::utility::all_in_state_vector(num_on_cache_qubits, qubit, qubits...));

            constexpr auto num_operated_qubits = bit_integer_type{sizeof...(Qubits) + 1u};
            assert(num_operated_qubits < num_on_cache_qubits);

            // xxxx|yyyy|zzzzzz: (local) qubits
            // * xxxx: off-cache qubits
            // * yyyy|zzzzzz: on-cache qubits
            //   - yyyy: chunk qubits (chunk qubits are determined dynamically, and sometimes there is no chunk qubit)
            // * xxxx|yyyy: tag qubits
            // * zzzzzz: nontag qubits

            using qubit_type = ::ket::qubit<state_integer_type, bit_integer_type>;
            auto const least_significant_off_cache_qubit = qubit_type{num_on_cache_qubits};

            // num_chunk_qubits, least_significant_chunk_qubit, num_tag_qubits, num_nontag_qubits
            constexpr auto num_chunk_qubits = num_operated_qubits;
            auto const least_significant_chunk_qubit = least_significant_off_cache_qubit - num_chunk_qubits;
            auto const num_tag_qubits = num_off_cache_qubits + num_chunk_qubits;
            auto const num_nontag_qubits = num_on_cache_qubits - num_chunk_qubits;

            // unsorted_tag_qubits, sorted_tag_qubits_with_sentinel
            std::array<qubit_type, num_operated_qubits> unsorted_tag_qubits{::ket::remove_control(qubit) - num_nontag_qubits, (::ket::remove_control(qubits) - num_nontag_qubits)...};
            std::array<qubit_type, num_operated_qubits + bit_integer_type{1u}> sorted_tag_qubits_with_sentinel{
              ::ket::remove_control(qubit) - num_nontag_qubits, (::ket::remove_control(qubits) - num_nontag_qubits)..., qubit_type{num_tag_qubits}};
            std::sort(begin(sorted_tag_qubits_with_sentinel), std::prev(end(sorted_tag_qubits_with_sentinel)));

            // unsorted_on_cache_qubits, sorted_on_cache_qubits_with_sentinel
            std::array<qubit_type, num_operated_qubits> unsorted_on_cache_qubits{least_significant_chunk_qubit, (least_significant_chunk_qubit + bit_integer_type{1u} + bit_integer_type{indices_for_qubits})...};
            std::array<qubit_type, num_operated_qubits + bit_integer_type{1u}> sorted_on_cache_qubits_with_sentinel{least_significant_chunk_qubit, (least_significant_chunk_qubit + bit_integer_type{1u} + bit_integer_type{indices_for_qubits})..., ::ket::make_qubit<state_integer_type>(num_on_cache_qubits)};

            ::ket::gate::cache::none_on_cache::impl::gate_impl<std::is_assignable<decltype(*state_first), typename std::iterator_traits<RandomAccessIterator1>::value_type>::value>::call(
              parallel_policy, state_first, state_last, on_cache_state_first, on_cache_state_last,
              unsorted_tag_qubits, sorted_tag_qubits_with_sentinel, unsorted_on_cache_qubits, sorted_on_cache_qubits_with_sentinel,
              std::forward<Function>(function));
          }
        } // namespace impl

        template <
          typename ParallelPolicy, typename RandomAccessIterator1, typename RandomAccessIterator2,
          typename Function, typename Qubit, typename... Qubits,
          typename IndicesForQubits = std::make_index_sequence<sizeof...(Qubits)>>
        inline auto gate(
          ParallelPolicy const parallel_policy,
          RandomAccessIterator1 const state_first, RandomAccessIterator1 const state_last,
          RandomAccessIterator2 const on_cache_state_first, RandomAccessIterator2 const on_cache_state_last,
          Function&& function, Qubit&& qubit, Qubits&&... qubits)
        -> void
        {
          ::ket::gate::cache::none_on_cache::impl::gate(
            IndicesForQubits{},
            parallel_policy, state_first, state_last, on_cache_state_first, on_cache_state_last,
            std::forward<Function>(function), std::forward<Qubit>(qubit), std::forward<Qubits>(qubits)...);
        }
      } // namespace none_on_cache

      // Case 3) There are some operated on-cache qubits
      //   ex: xxxx|yyy|zzzzzzz
      //        ^^   ^    ^     <- operated qubits
      namespace some_on_cache
      {
        // First argument of Function: RandomAccessIterator2 (not RandomAccessIterator1)
        namespace gate_detail
        {
          template <bool is_state_iterator_mutable>
          struct gate
          {
            template <typename ParallelPolicy, typename RandomAccessIterator1, typename RandomAccessIterator2, typename Function>
            static auto call(
              ParallelPolicy const parallel_policy,
              RandomAccessIterator1 const state_first, RandomAccessIterator1 const state_last,
              RandomAccessIterator2 const on_cache_state_first, RandomAccessIterator2 const on_cache_state_last,
              Function&& function)
            -> void
            {
              using qubit_type = ::ket::qubit<>;
              using state_integer_type = ::ket::meta::state_integer_t<qubit_type>;
              using bit_integer_type = ::ket::meta::bit_integer_t<qubit_type>;

              auto const state_size = static_cast<state_integer_type>(state_last - state_first);
              auto const num_qubits = ::ket::utility::integer_log2<bit_integer_type>(state_size);
              assert(::ket::utility::integer_exp2<state_integer_type>(num_qubits) == state_size);

              auto const cache_size = static_cast<state_integer_type>(on_cache_state_last - on_cache_state_first);
              auto const num_on_cache_qubits = ::ket::utility::integer_log2<bit_integer_type>(cache_size);
              assert(::ket::utility::integer_exp2<state_integer_type>(num_on_cache_qubits) == cache_size);
              assert(num_on_cache_qubits < num_qubits);
              auto const num_off_cache_qubits = num_qubits - num_on_cache_qubits;

              // xxxx|yyyy|zzzzzz: (local) qubits
              // * xxxx: off-cache qubits
              // * yyyy|zzzzzz: on-cache qubits
              //   - yyyy: chunk qubits (chunk qubits are determined dynamically, and sometimes there is no chunk qubit)
              // * xxxx|yyyy: tag qubits
              // * zzzzzz: nontag qubits

              // chunk_size, num_tag_qubits
              auto const chunk_size = cache_size;
              auto const num_tag_qubits = num_off_cache_qubits;

              // unsorted_tag_qubits, sorted_tag_qubits_with_sentinel
              auto unsorted_tag_qubits = std::vector<qubit_type>{};
              auto sorted_tag_qubits_with_sentinel = std::vector<qubit_type>{qubit_type{num_tag_qubits}};

              // unsorted_on_cache_qubits, sorted_on_cache_qubits_with_sentinel
              std::array<qubit_type, 0u> unsorted_on_cache_qubits{};
              std::array<qubit_type, 1u> sorted_on_cache_qubits_with_sentinel{::ket::make_qubit<state_integer_type>(num_on_cache_qubits)};

              auto const tag_loop_size = ::ket::utility::integer_exp2<state_integer_type>(num_tag_qubits);
              for (auto tag_index_wo_qubits = state_integer_type{0u}; tag_index_wo_qubits < tag_loop_size; ++tag_index_wo_qubits)
              {
                ::ket::utility::copy_n(
                  parallel_policy,
                  state_first
                  + ::ket::gate::utility::index_with_qubits(
                      tag_index_wo_qubits, state_integer_type{0u},
                      begin(unsorted_tag_qubits), end(unsorted_tag_qubits),
                      begin(sorted_tag_qubits_with_sentinel), end(sorted_tag_qubits_with_sentinel)) * chunk_size,
                  chunk_size, on_cache_state_first);

                ::ket::gate::gate_detail::gate_n(parallel_policy, on_cache_state_first, cache_size, unsorted_on_cache_qubits, sorted_on_cache_qubits_with_sentinel, std::forward<Function>(function));

                ::ket::utility::copy_n(
                  parallel_policy,
                  on_cache_state_first, chunk_size,
                  state_first
                  + ::ket::gate::utility::index_with_qubits(
                      tag_index_wo_qubits, state_integer_type{0u},
                      begin(unsorted_tag_qubits), end(unsorted_tag_qubits),
                      begin(sorted_tag_qubits_with_sentinel), end(sorted_tag_qubits_with_sentinel)) * chunk_size);
              }
            }
          }; // struct gate<is_state_iterator_mutable>

          template <>
          struct gate<false>
          {
            template <typename ParallelPolicy, typename RandomAccessIterator1, typename RandomAccessIterator2, typename Function>
            static auto call(
              ParallelPolicy const parallel_policy,
              RandomAccessIterator1 const state_first, RandomAccessIterator1 const state_last,
              RandomAccessIterator2 const on_cache_state_first, RandomAccessIterator2 const on_cache_state_last,
              Function&& function)
            -> void
            {
              using qubit_type = ::ket::qubit<>;
              using state_integer_type = ::ket::meta::state_integer_t<qubit_type>;
              using bit_integer_type = ::ket::meta::bit_integer_t<qubit_type>;

              auto const state_size = static_cast<state_integer_type>(state_last - state_first);
              auto const num_qubits = ::ket::utility::integer_log2<bit_integer_type>(state_size);
              assert(::ket::utility::integer_exp2<state_integer_type>(num_qubits) == state_size);

              auto const cache_size = static_cast<state_integer_type>(on_cache_state_last - on_cache_state_first);
              auto const num_on_cache_qubits = ::ket::utility::integer_log2<bit_integer_type>(cache_size);
              assert(::ket::utility::integer_exp2<state_integer_type>(num_on_cache_qubits) == cache_size);
              assert(num_on_cache_qubits < num_qubits);
              auto const num_off_cache_qubits = num_qubits - num_on_cache_qubits;

              // xxxx|yyyy|zzzzzz: (local) qubits
              // * xxxx: off-cache qubits
              // * yyyy|zzzzzz: on-cache qubits
              //   - yyyy: chunk qubits (chunk qubits are determined dynamically, and sometimes there is no chunk qubit)
              // * xxxx|yyyy: tag qubits
              // * zzzzzz: nontag qubits

              // chunk_size, num_tag_qubits
              auto const chunk_size = cache_size;
              auto const num_tag_qubits = num_off_cache_qubits;

              // unsorted_tag_qubits, sorted_tag_qubits_with_sentinel
              auto unsorted_tag_qubits = std::vector<qubit_type>{};
              auto sorted_tag_qubits_with_sentinel = std::vector<qubit_type>{qubit_type{num_tag_qubits}};

              // unsorted_on_cache_qubits, sorted_on_cache_qubits_with_sentinel
              std::array<qubit_type, 0u> unsorted_on_cache_qubits{};
              std::array<qubit_type, 1u> sorted_on_cache_qubits_with_sentinel{::ket::make_qubit<state_integer_type>(num_on_cache_qubits)};

              auto const tag_loop_size = ::ket::utility::integer_exp2<state_integer_type>(num_tag_qubits);
              for (auto tag_index_wo_qubits = state_integer_type{0u}; tag_index_wo_qubits < tag_loop_size; ++tag_index_wo_qubits)
              {
                ::ket::utility::copy_n(
                  parallel_policy,
                  state_first
                  + ::ket::gate::utility::index_with_qubits(
                      tag_index_wo_qubits, state_integer_type{0u},
                      begin(unsorted_tag_qubits), end(unsorted_tag_qubits),
                      begin(sorted_tag_qubits_with_sentinel), end(sorted_tag_qubits_with_sentinel)) * chunk_size,
                  chunk_size, on_cache_state_first);

                ::ket::gate::gate_detail::gate_n(parallel_policy, on_cache_state_first, cache_size, unsorted_on_cache_qubits, sorted_on_cache_qubits_with_sentinel, std::forward<Function>(function));
              }
            }
          }; // struct gate<false>
        } // namespace gate_detail

        template <typename ParallelPolicy, typename RandomAccessIterator1, typename RandomAccessIterator2, typename Function>
        inline auto gate(
          ParallelPolicy const parallel_policy,
          RandomAccessIterator1 const state_first, RandomAccessIterator1 const state_last,
          RandomAccessIterator2 const on_cache_state_first, RandomAccessIterator2 const on_cache_state_last,
          Function&& function)
        -> void
        {
          ::ket::gate::cache::some_on_cache::gate_detail::gate<std::is_assignable<decltype(*state_first), typename std::iterator_traits<RandomAccessIterator1>::value_type>::value>::call(
            parallel_policy, state_first, state_last, on_cache_state_first, on_cache_state_last,
            std::forward<Function>(function));
        }

        namespace impl
        {
          template <bool is_state_iterator_mutable>
          struct gate_impl
          {
            template <
              typename ParallelPolicy, typename RandomAccessIterator1, typename RandomAccessIterator2,
              typename Qubit, std::size_t num_operated_qubits, typename Function>
            static auto call(
              ParallelPolicy const parallel_policy,
              RandomAccessIterator1 const state_first, RandomAccessIterator1 const state_last,
              RandomAccessIterator2 const on_cache_state_first, RandomAccessIterator2 const on_cache_state_last,
              std::vector<Qubit> const& unsorted_tag_qubits,
              std::vector<Qubit> const& sorted_tag_qubits_with_sentinel,
              std::array<Qubit, num_operated_qubits> const& unsorted_on_cache_qubits,
              std::array<Qubit, num_operated_qubits + 1u> const& sorted_on_cache_qubits_with_sentinel,
              Function&& function)
            -> void
            {
              using state_integer_type = ::ket::meta::state_integer_t<Qubit>;
              using bit_integer_type = ::ket::meta::bit_integer_t<Qubit>;

              auto const state_size = static_cast<state_integer_type>(state_last - state_first);
              auto const num_qubits = ::ket::utility::integer_log2<bit_integer_type>(state_size);

              auto const cache_size = static_cast<state_integer_type>(on_cache_state_last - on_cache_state_first);
              auto const num_on_cache_qubits = ::ket::utility::integer_log2<bit_integer_type>(cache_size);
              auto const num_off_cache_qubits = num_qubits - num_on_cache_qubits;

              // num_chunk_qubits, chunk_size, num_tag_qubits
              auto const num_chunk_qubits = unsorted_tag_qubits.size();
              auto const num_chunks_in_on_cache_state = ::ket::utility::integer_exp2<state_integer_type>(num_chunk_qubits);
              auto const chunk_size = cache_size / num_chunks_in_on_cache_state;
              auto const num_tag_qubits = num_off_cache_qubits + num_chunk_qubits;

              auto const tag_loop_size = ::ket::utility::integer_exp2<state_integer_type>(num_tag_qubits - num_chunk_qubits); // num_chunk_qubits == operated_tag_qubits.size()
              for (auto tag_index_wo_qubits = state_integer_type{0u}; tag_index_wo_qubits < tag_loop_size; ++tag_index_wo_qubits)
              {
                using std::begin;
                using std::end;
                for (auto chunk_index = state_integer_type{0u}; chunk_index < num_chunks_in_on_cache_state; ++chunk_index)
                  ::ket::utility::copy_n(
                    parallel_policy,
                    state_first
                    + ::ket::gate::utility::index_with_qubits(
                        tag_index_wo_qubits, chunk_index,
                        begin(unsorted_tag_qubits), end(unsorted_tag_qubits),
                        begin(sorted_tag_qubits_with_sentinel), end(sorted_tag_qubits_with_sentinel)) * chunk_size,
                    chunk_size, on_cache_state_first + chunk_index * chunk_size);

                ::ket::gate::gate_detail::gate_n(parallel_policy, on_cache_state_first, cache_size, unsorted_on_cache_qubits, sorted_on_cache_qubits_with_sentinel, std::forward<Function>(function));

                for (auto chunk_index = state_integer_type{0u}; chunk_index < num_chunks_in_on_cache_state; ++chunk_index)
                  ::ket::utility::copy_n(
                    parallel_policy,
                    on_cache_state_first + chunk_index * chunk_size, chunk_size,
                    state_first
                    + ::ket::gate::utility::index_with_qubits(
                        tag_index_wo_qubits, chunk_index,
                        begin(unsorted_tag_qubits), end(unsorted_tag_qubits),
                        begin(sorted_tag_qubits_with_sentinel), end(sorted_tag_qubits_with_sentinel)) * chunk_size);
              }
            }
          }; // struct gate_impl<is_state_iterator_mutable>

          template <>
          struct gate_impl<false>
          {
            template <
              typename ParallelPolicy, typename RandomAccessIterator1, typename RandomAccessIterator2,
              typename Qubit, std::size_t num_operated_qubits, typename Function>
            static auto call(
              ParallelPolicy const parallel_policy,
              RandomAccessIterator1 const state_first, RandomAccessIterator1 const state_last,
              RandomAccessIterator2 const on_cache_state_first, RandomAccessIterator2 const on_cache_state_last,
              std::vector<Qubit> const& unsorted_tag_qubits,
              std::vector<Qubit> const& sorted_tag_qubits_with_sentinel,
              std::array<Qubit, num_operated_qubits> const& unsorted_on_cache_qubits,
              std::array<Qubit, num_operated_qubits + 1u> const& sorted_on_cache_qubits_with_sentinel,
              Function&& function)
            -> void
            {
              using state_integer_type = ::ket::meta::state_integer_t<Qubit>;
              using bit_integer_type = ::ket::meta::bit_integer_t<Qubit>;

              auto const state_size = static_cast<state_integer_type>(state_last - state_first);
              auto const num_qubits = ::ket::utility::integer_log2<bit_integer_type>(state_size);

              auto const cache_size = static_cast<state_integer_type>(on_cache_state_last - on_cache_state_first);
              auto const num_on_cache_qubits = ::ket::utility::integer_log2<bit_integer_type>(cache_size);
              auto const num_off_cache_qubits = num_qubits - num_on_cache_qubits;

              // num_chunk_qubits, chunk_size, num_tag_qubits
              auto const num_chunk_qubits = unsorted_tag_qubits.size();
              auto const num_chunks_in_on_cache_state = ::ket::utility::integer_exp2<state_integer_type>(num_chunk_qubits);
              auto const chunk_size = cache_size / num_chunks_in_on_cache_state;
              auto const num_tag_qubits = num_off_cache_qubits + num_chunk_qubits;

              auto const tag_loop_size = ::ket::utility::integer_exp2<state_integer_type>(num_tag_qubits - num_chunk_qubits); // num_chunk_qubits == operated_tag_qubits.size()
              for (auto tag_index_wo_qubits = state_integer_type{0u}; tag_index_wo_qubits < tag_loop_size; ++tag_index_wo_qubits)
              {
                using std::begin;
                using std::end;
                for (auto chunk_index = state_integer_type{0u}; chunk_index < num_chunks_in_on_cache_state; ++chunk_index)
                  ::ket::utility::copy_n(
                    parallel_policy,
                    state_first
                    + ::ket::gate::utility::index_with_qubits(
                        tag_index_wo_qubits, chunk_index,
                        begin(unsorted_tag_qubits), end(unsorted_tag_qubits),
                        begin(sorted_tag_qubits_with_sentinel), end(sorted_tag_qubits_with_sentinel)) * chunk_size,
                    chunk_size, on_cache_state_first + chunk_index * chunk_size);

                ::ket::gate::gate_detail::gate_n(parallel_policy, on_cache_state_first, cache_size, unsorted_on_cache_qubits, sorted_on_cache_qubits_with_sentinel, std::forward<Function>(function));
              }
            }
          }; // struct gate_impl<false>

          template <
            typename ParallelPolicy, typename RandomAccessIterator1, typename RandomAccessIterator2,
            typename Function, typename Qubit, typename... Qubits, std::size_t... indices_for_qubits>
          inline auto gate(
            std::index_sequence<indices_for_qubits...> const,
            ParallelPolicy const parallel_policy,
            RandomAccessIterator1 const state_first, RandomAccessIterator1 const state_last,
            RandomAccessIterator2 const on_cache_state_first, RandomAccessIterator2 const on_cache_state_last,
            Function&& function, Qubit&& qubit, Qubits&&... qubits)
          -> void
          {
            using state_integer_type = ::ket::meta::state_integer_t<std::remove_reference_t<Qubit>>;
            static_assert(std::is_unsigned<state_integer_type>::value, "state_integer_type of Qubit should be unsigned");
#     if __cpp_constexpr >= 201603L
            static_assert(
              ::ket::utility::variadic::proj::all_of(
                [](auto integer) { return std::is_same<decltype(integer), state_integer_type>::value; },
                [](auto qubit) { return ::ket::meta::state_integer_t<decltype(qubit)>{}; },
                std::remove_cv_t<std::remove_reference_t<Qubits>>{}...),
              "state_integer_type's of Qubit and Qubits should be the same");
#     else // __cpp_constexpr >= 201603L
            static_assert(
              ::ket::utility::variadic::proj::all_of(
                ::ket::gate::gate_detail::is_same_to<state_integer_type>{}, ::ket::gate::gate_detail::state_integer_of{},
                std::remove_cv_t<std::remove_reference_t<Qubits>>{}...),
              "state_integer_type's of Qubit and Qubits should be the same");
#     endif // __cpp_constexpr >= 201603L

            using bit_integer_type = ::ket::meta::bit_integer_t<std::remove_reference_t<Qubit>>;
            static_assert(std::is_unsigned<bit_integer_type>::value, "bit_integer_type of Qubit should be unsigned");
#     if __cpp_constexpr >= 201603L
            static_assert(
              ::ket::utility::variadic::proj::all_of(
                [](auto integer) { return std::is_same<decltype(integer), bit_integer_type>::value; },
                [](auto qubit) { return ::ket::meta::bit_integer_t<decltype(qubit)>{}; },
                std::remove_cv_t<std::remove_reference_t<Qubits>>{}...),
              "bit_integer_type's of Qubit and Qubits should be the same");
#     else // __cpp_constexpr >= 201603L
            static_assert(
              ::ket::utility::variadic::proj::all_of(
                ::ket::gate::gate_detail::is_same_to<bit_integer_type>{}, ::ket::gate::gate_detail::bit_integer_of{},
                std::remove_cv_t<std::remove_reference_t<Qubits>>{}...),
              "bit_integer_type's of Qubit and Qubits should be the same");
#     endif // __cpp_constexpr >= 201603L

            auto const state_size = static_cast<state_integer_type>(state_last - state_first);
            auto const num_qubits = ::ket::utility::integer_log2<bit_integer_type>(state_size);
            assert(::ket::utility::integer_exp2<state_integer_type>(num_qubits) == state_size);
            assert(::ket::utility::all_in_state_vector(num_qubits, qubit, qubits...));

            auto const cache_size = static_cast<state_integer_type>(on_cache_state_last - on_cache_state_first);
            auto const num_on_cache_qubits = ::ket::utility::integer_log2<bit_integer_type>(cache_size);
            assert(::ket::utility::integer_exp2<state_integer_type>(num_on_cache_qubits) == cache_size);
            assert(num_on_cache_qubits < num_qubits);
            auto const num_off_cache_qubits = num_qubits - num_on_cache_qubits;
            // It is required to be confirmed not to satisfy Case 1)
            assert(not ::ket::utility::all_in_state_vector(num_on_cache_qubits, qubit, qubits...));
            // It is required to be confirmed not to satisfy Case 2)
            assert(not ::ket::utility::none_in_state_vector(num_on_cache_qubits, qubit, qubits...));

            constexpr auto num_operated_qubits = bit_integer_type{sizeof...(Qubits) + 1u};
            assert(num_operated_qubits < num_on_cache_qubits);

            // xxxx|yyyy|zzzzzz: (local) qubits
            // * xxxx: off-cache qubits
            // * yyyy|zzzzzz: on-cache qubits
            //   - yyyy: chunk qubits (chunk qubits are determined dynamically, and sometimes there is no chunk qubit)
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
            auto const operated_off_cache_qubits_first = operated_on_cache_qubits_last;
            auto const operated_off_cache_qubits_last = end(sorted_qubits);
            // from Assumption: Case 3) Some of the operated qubits are off-cache qubits
            assert(operated_on_cache_qubits_first != operated_on_cache_qubits_last);
            assert(operated_off_cache_qubits_first != operated_off_cache_qubits_last);

            // least_significant_chunk_qubit, num_chunk_qubits, num_tag_qubits, num_nontag_qubits
            auto operated_on_cache_qubits_iter = std::prev(operated_on_cache_qubits_last);
            auto free_most_significant_on_cache_qubit = least_significant_off_cache_qubit - bit_integer_type{1u};
            auto const num_operated_off_cache_qubits
              = static_cast<bit_integer_type>(operated_off_cache_qubits_last - operated_off_cache_qubits_first);
            for (auto num_found_operated_off_cache_qubits = bit_integer_type{0u};
                 num_found_operated_off_cache_qubits < num_operated_off_cache_qubits; ++num_found_operated_off_cache_qubits)
              while (free_most_significant_on_cache_qubit-- == *operated_on_cache_qubits_iter)
                if (operated_on_cache_qubits_iter != operated_on_cache_qubits_first)
                  --operated_on_cache_qubits_iter;
            auto const least_significant_chunk_qubit = free_most_significant_on_cache_qubit + bit_integer_type{1u};
            auto const num_chunk_qubits = static_cast<bit_integer_type>(least_significant_off_cache_qubit - least_significant_chunk_qubit);
            assert(num_chunk_qubits <= num_operated_qubits);
            auto const num_tag_qubits = num_off_cache_qubits + num_chunk_qubits;
            auto const num_nontag_qubits = num_on_cache_qubits - num_chunk_qubits;

            // unsorted_tag_qubits, modified_operated_qubits
            auto unsorted_tag_qubits = std::vector<qubit_type>{};
            unsorted_tag_qubits.reserve(num_chunk_qubits);
            auto present_chunk_qubit = least_significant_chunk_qubit;
            auto const modified_operated_qubits
              = ::ket::utility::variadic::transform(
                  [least_significant_chunk_qubit, num_nontag_qubits, &unsorted_tag_qubits, &present_chunk_qubit](auto qubit)
                  {
                    if (qubit < least_significant_chunk_qubit)
                      return qubit;

                    unsorted_tag_qubits.push_back(::ket::remove_control(qubit) - num_nontag_qubits);
                    return static_cast<decltype(qubit)>(present_chunk_qubit++);
                  },
                  qubit, qubits...);
            assert(present_chunk_qubit == least_significant_off_cache_qubit);
            assert(static_cast<bit_integer_type>(unsorted_tag_qubits.size()) == num_chunk_qubits);

            // sorted_tag_qubits_with_sentinel
            auto sorted_tag_qubits_with_sentinel = std::vector<qubit_type>{};
            sorted_tag_qubits_with_sentinel.reserve(unsorted_tag_qubits.size() + 1u);
            std::copy(begin(unsorted_tag_qubits), end(unsorted_tag_qubits), std::back_inserter(sorted_tag_qubits_with_sentinel));
            sorted_tag_qubits_with_sentinel.push_back(qubit_type{num_tag_qubits});
            std::sort(begin(sorted_tag_qubits_with_sentinel), std::prev(end(sorted_tag_qubits_with_sentinel)));
            assert(sorted_tag_qubits_with_sentinel.size() == unsorted_tag_qubits.size() + 1u);

            // unsorted_on_cache_qubits, sorted_on_cache_qubits_with_sentinel
            std::array<qubit_type, num_operated_qubits> unsorted_on_cache_qubits{ket::remove_control(std::get<0u>(modified_operated_qubits)), ket::remove_control(std::get<1u + indices_for_qubits>(modified_operated_qubits))...};
            std::array<qubit_type, num_operated_qubits + bit_integer_type{1u}> sorted_on_cache_qubits_with_sentinel{ket::remove_control(std::get<0u>(modified_operated_qubits)), ket::remove_control(std::get<1u + indices_for_qubits>(modified_operated_qubits))..., ::ket::make_qubit<state_integer_type>(num_on_cache_qubits)};
            std::sort(begin(sorted_on_cache_qubits_with_sentinel), std::prev(end(sorted_on_cache_qubits_with_sentinel)));

            ::ket::gate::cache::some_on_cache::impl::gate_impl<std::is_assignable<decltype(*state_first), typename std::iterator_traits<RandomAccessIterator1>::value_type>::value>::call(
              parallel_policy, state_first, state_last, on_cache_state_first, on_cache_state_last,
              unsorted_tag_qubits, sorted_tag_qubits_with_sentinel, unsorted_on_cache_qubits, sorted_on_cache_qubits_with_sentinel,
              std::forward<Function>(function));
          }
        } // namespace impl

        template <
          typename ParallelPolicy, typename RandomAccessIterator1, typename RandomAccessIterator2,
          typename Function, typename Qubit, typename... Qubits,
          typename IndicesForQubits = std::make_index_sequence<sizeof...(Qubits)>>
        inline auto gate(
          ParallelPolicy const parallel_policy,
          RandomAccessIterator1 const state_first, RandomAccessIterator1 const state_last,
          RandomAccessIterator2 const on_cache_state_first, RandomAccessIterator2 const on_cache_state_last,
          Function&& function, Qubit&& qubit, Qubits&&... qubits)
        -> void
        {
          ::ket::gate::cache::some_on_cache::impl::gate(
            IndicesForQubits{},
            parallel_policy, state_first, state_last, on_cache_state_first, on_cache_state_last,
            std::forward<Function>(function), std::forward<Qubit>(qubit), std::forward<Qubits>(qubits)...);
        }
      } // namespace some_on_cache
#   endif // KET_USE_ON_CACHE_STATE_VECTOR
    } // namespace cache
# endif // KET_ENABLE_CACHE_AWARE_GATE_FUNCTION

# ifndef KET_ENABLE_CACHE_AWARE_GATE_FUNCTION
    template <typename ParallelPolicy, typename RandomAccessIterator, typename Function, typename... Qubits>
    inline auto gate(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      Function&& function, Qubits&&... qubits)
    -> void
    { ::ket::gate::nocache::gate(parallel_policy, first, last, std::forward<Function>(function), std::forward<Qubits>(qubits)...); }
# else // KET_ENABLE_CACHE_AWARE_GATE_FUNCTION
    template <typename ParallelPolicy, typename RandomAccessIterator, typename Function>
    inline auto gate(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      Function&& function)
    -> void
    {
      using qubit_type = ::ket::qubit<>;
      using state_integer_type = ::ket::meta::state_integer_t<qubit_type>;
      using bit_integer_type = ::ket::meta::bit_integer_t<qubit_type>;

      auto const state_size = static_cast<state_integer_type>(last - first);
#   ifndef NDEBUG
      auto const num_qubits = ::ket::utility::integer_log2<bit_integer_type>(state_size);
#   endif // NDEBUG
      assert(::ket::utility::integer_exp2<state_integer_type>(num_qubits) == state_size);

#   ifndef KET_DEFAULT_NUM_ON_CACHE_QUBITS
#     define KET_DEFAULT_NUM_ON_CACHE_QUBITS 16
#   endif // KET_DEFAULT_NUM_ON_CACHE_QUBITS
      constexpr auto num_on_cache_qubits = bit_integer_type{KET_DEFAULT_NUM_ON_CACHE_QUBITS};
      constexpr auto cache_size = ::ket::utility::integer_exp2<state_integer_type>(num_on_cache_qubits);
      if (state_size <= cache_size)
      {
        ::ket::gate::nocache::gate(parallel_policy, first, last, std::forward<Function>(function));
        return;
      }

      // xxxx|yyyy|zzzzzz: (local) qubits
      // * xxxx: off-cache qubits
      // * yyyy|zzzzzz: on-cache qubits
      //   - yyyy: chunk qubits (chunk qubits are determined dynamically, and sometimes there is no chunk qubit)
      // * xxxx|yyyy: tag qubits
      // * zzzzzz: nontag qubits

      // Case 1) All operated qubits are on-cache qubits
      //   ex: xxxx|zzzzzzzzzz
      //             ^  ^   ^  <- operated qubits
      ::ket::gate::cache::all_on_cache::gate(parallel_policy, first, last, std::forward<Function>(function));
    }

    template <typename ParallelPolicy, typename RandomAccessIterator, typename Function, typename Qubit, typename... Qubits>
    inline auto gate(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      Function&& function, Qubit&& qubit, Qubits&&... qubits)
    -> void
    {
      using state_integer_type = ::ket::meta::state_integer_t<std::remove_reference_t<Qubit>>;
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

      using bit_integer_type = ::ket::meta::bit_integer_t<std::remove_reference_t<Qubit>>;
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

      auto const state_size = static_cast<state_integer_type>(last - first);
#   ifndef NDEBUG
      auto const num_qubits = ::ket::utility::integer_log2<bit_integer_type>(state_size);
#   endif // NDEBUG
      assert(::ket::utility::integer_exp2<state_integer_type>(num_qubits) == state_size);
      assert(::ket::utility::all_in_state_vector(num_qubits, qubit, qubits...));

#   ifndef KET_DEFAULT_NUM_ON_CACHE_QUBITS
#     define KET_DEFAULT_NUM_ON_CACHE_QUBITS 16
#   endif // KET_DEFAULT_NUM_ON_CACHE_QUBITS
      constexpr auto num_on_cache_qubits = bit_integer_type{KET_DEFAULT_NUM_ON_CACHE_QUBITS};
      constexpr auto cache_size = ::ket::utility::integer_exp2<state_integer_type>(num_on_cache_qubits);
      if (state_size <= cache_size)
      {
        ::ket::gate::nocache::gate(parallel_policy, first, last, std::forward<Function>(function), std::forward<Qubit>(qubit), std::forward<Qubits>(qubits)...);
        return;
      }

      // xxxx|yyyy|zzzzzz: (local) qubits
      // * xxxx: off-cache qubits
      // * yyyy|zzzzzz: on-cache qubits
      //   - yyyy: chunk qubits (chunk qubits are determined dynamically, and sometimes there is no chunk qubit)
      // * xxxx|yyyy: tag qubits
      // * zzzzzz: nontag qubits

      if (::ket::utility::all_in_state_vector(num_on_cache_qubits, qubit, qubits...))
      {
        // Case 1) All operated qubits are on-cache qubits
        //   ex: xxxx|zzzzzzzzzz
        //             ^  ^   ^  <- operated qubits
        ::ket::gate::cache::all_on_cache::gate(parallel_policy, first, last, std::forward<Function>(function), std::forward<Qubit>(qubit), std::forward<Qubits>(qubits)...);
      }
      else
      {
#   ifndef KET_USE_ON_CACHE_STATE_VECTOR
        if (::ket::utility::none_in_state_vector(num_on_cache_qubits, qubit, qubits...))
        {
          // Case 2) There is no operated on-cache qubit
          //   ex: xxxx|yyy|zzzzzzz
          //       ^^ ^             <- operated qubits
          ::ket::gate::cache::none_on_cache::gate(parallel_policy, first, last, std::forward<Function>(function), std::forward<Qubit>(qubit), std::forward<Qubits>(qubits)...);
        }
        else
        {
          // Case 3) There are some operated on-cache qubits
          //   ex: xxxx|yyy|zzzzzzz
          //        ^^   ^    ^     <- operated qubits
          ::ket::gate::cache::some_on_cache::gate(parallel_policy, first, last, std::forward<Function>(function), std::forward<Qubit>(qubit), std::forward<Qubits>(qubits)...);
        }
#   else // KET_USE_ON_CACHE_STATE_VECTOR
        auto on_cache_state = std::vector<typename std::iterator_traits<RandomAccessIterator>::value_type>(cache_size);

        using std::begin;
        using std::end;
        if (::ket::utility::none_in_state_vector(num_on_cache_qubits, qubit, qubits...))
        {
          // Case 2) There is no operated on-cache qubit
          //   ex: xxxx|yyy|zzzzzzz
          //       ^^ ^             <- operated qubits
          ::ket::gate::cache::none_on_cache::gate(
            parallel_policy,
            first, last, begin(on_cache_state), end(on_cache_state),
            std::forward<Function>(function), std::forward<Qubit>(qubit), std::forward<Qubits>(qubits)...);
        }
        else
        {
          // Case 3) There are some operated on-cache qubits
          //   ex: xxxx|yyy|zzzzzzz
          //        ^^   ^    ^     <- operated qubits
          ::ket::gate::cache::some_on_cache::gate(
            parallel_policy,
            first, last, begin(on_cache_state), end(on_cache_state),
            std::forward<Function>(function), std::forward<Qubit>(qubit), std::forward<Qubits>(qubits)...);
        }
#   endif // KET_USE_ON_CACHE_STATE_VECTOR
      }
    }
# endif // KET_ENABLE_CACHE_AWARE_GATE_FUNCTION

    template <typename RandomAccessIterator, typename Function, typename... Qubits>
    inline auto gate(
      RandomAccessIterator const first, RandomAccessIterator const last,
      Function&& function, Qubits&&... qubits)
    -> void
    {
      ::ket::gate::gate(
        ::ket::utility::policy::make_sequential(),
        first, last, std::forward<Function>(function), std::forward<Qubits>(qubits)...);
    }

    namespace ranges
    {
      template <typename ParallelPolicy, typename RandomAccessRange, typename Function, typename... Qubits>
      inline std::enable_if_t< ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value, RandomAccessRange& > gate(
        ParallelPolicy const parallel_policy, RandomAccessRange& state,
        Function&& function, Qubits&&... qubits)
      {
        using std::begin;
        using std::end;
        ::ket::gate::gate(parallel_policy, begin(state), end(state), std::forward<Function>(function), std::forward<Qubits>(qubits)...);
        return state;
      }

      template <typename RandomAccessRange, typename Function, typename... Qubits>
      inline std::enable_if_t<not ::ket::utility::policy::meta::is_loop_n_policy<RandomAccessRange&>::value, RandomAccessRange&> gate(
        RandomAccessRange& state, Function&& function, Qubits&&... qubits)
      {
        return ::ket::gate::ranges::gate(
          ::ket::utility::policy::make_sequential(),
          state, std::forward<Function>(function), std::forward<Qubits>(qubits)...);
      }
    } // namespace ranges
  } // namespace gate
} // namespace ket


#endif // KET_GATE_GATE_HPP
