#ifndef KET_EXPECTATION_VALUE_HPP
# define KET_EXPECTATION_VALUE_HPP

# include <array>
# include <vector>
# include <iterator>
# include <numeric>
# include <utility>
# include <type_traits>

# include <ket/qubit.hpp>
# include <ket/gate/gate.hpp>
# include <ket/meta/state_integer_of.hpp>
# include <ket/meta/bit_integer_of.hpp>
# include <ket/utility/loop_n.hpp>
# if !defined(NDEBUG) || defined(KET_USE_ON_CACHE_STATE_VECTOR)
#   include <ket/utility/all_in_state_vector.hpp>
# endif
# include <ket/utility/variadic/all_of.hpp>
# include <ket/utility/meta/real_of.hpp>
# include <ket/utility/meta/ranges.hpp>


namespace ket
{
  namespace expectation_value_detail
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
  } // namespace expectation_value_detail

  template <typename ParallelPolicy, typename RandomAccessIterator, typename Observable, typename StateInteger, typename BitInteger, typename... Qubits>
  inline auto expectation_value(
    ParallelPolicy const parallel_policy,
    RandomAccessIterator const first, RandomAccessIterator const last,
    Observable&& observable, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
  -> ::ket::utility::meta::real_t<typename std::iterator_traits<RandomAccessIterator>::value_type>
  {
    static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
# if __cpp_constexpr >= 201603L
    static_assert(
      ::ket::utility::variadic::proj::all_of(
        [](auto integer) { return std::is_same<decltype(integer), StateInteger>::value; },
        [](auto qubit) { return ::ket::meta::state_integer_t<decltype(qubit)>{}; },
        std::remove_cv_t<std::remove_reference_t<Qubits>>{}...),
      "state_integer_type's of Qubits should be same to StateInteger");
# else // __cpp_constexpr >= 201603L
    static_assert(
      ::ket::utility::variadic::proj::all_of(
        ::ket::gate::gate_detail::is_same_to<StateInteger>{}, ::ket::gate::gate_detail::state_integer_of{},
        std::remove_cv_t<std::remove_reference_t<Qubits>>{}...),
      "state_integer_type's of Qubits should be same to StateInteger");
# endif // __cpp_constexpr >= 201603L

    static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");
# if __cpp_constexpr >= 201603L
    static_assert(
      ::ket::utility::variadic::proj::all_of(
        [](auto integer) { return std::is_same<decltype(integer), BitInteger>::value; },
        [](auto qubit) { return ::ket::meta::bit_integer_t<decltype(qubit)>{}; },
        std::remove_cv_t<std::remove_reference_t<Qubits>>{}...),
      "bit_integer_type's of Qubits should be same to BitInteger");
# else // __cpp_constexpr >= 201603L
    static_assert(
      ::ket::utility::variadic::proj::all_of(
        ::ket::gate::gate_detail::is_same_to<BitInteger>{}, ::ket::gate::gate_detail::bit_integer_of{},
        std::remove_cv_t<std::remove_reference_t<Qubits>>{}...),
      "bit_integer_type's of Qubits should be same to BitInteger");
# endif // __cpp_constexpr >= 201603L

    constexpr auto num_operated_qubits = static_cast<BitInteger>(sizeof...(Qubits) + 1u);

    using real_type = ::ket::utility::meta::real_t<typename std::iterator_traits<RandomAccessIterator>::value_type>;
    auto partial_sums = std::vector<real_type>(::ket::utility::num_threads(parallel_policy));

    struct { Observable call; } wrapped_observable{std::forward<Observable>(observable)};
# ifndef KET_USE_BIT_MASKS_EXPLICITLY
    using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
    ::ket::gate::gate(
      parallel_policy, first, last,
      [wrapped_observable = std::move(wrapped_observable), &partial_sums](
        auto const first, StateInteger const index_wo_qubits,
        std::array<qubit_type, num_operated_qubits> const& unsorted_qubits,
        std::array<qubit_type, num_operated_qubits + 1u> const& sorted_qubits_with_sentinel,
        int const thread_index)
      { partial_sums[thread_index] += wrapped_observable.call(first, index_wo_qubits, unsorted_qubits, sorted_qubits_with_sentinel); },
      qubit, qubits...);
# else // KET_USE_BIT_MASKS_EXPLICITLY
    ::ket::gate::gate(
      parallel_policy, first, last,
      [wrapped_observable = std::move(wrapped_observable), &partial_sums](
        auto const first, StateInteger const index_wo_qubits,
        std::array<StateInteger, num_operated_qubits> const& qubit_masks,
        std::array<StateInteger, num_operated_qubits + 1u> const& index_masks,
        int const thread_index)
      { partial_sums[thread_index] += wrapped_observable.call(first, index_wo_qubits, qubit_masks, index_masks); },
      qubit, qubits...);
# endif // KET_USE_BIT_MASKS_EXPLICITLY

    using std::begin;
    using std::end;
    return std::accumulate(begin(partial_sums), end(partial_sums), real_type{0});
  }

  template <typename RandomAccessIterator, typename Observable, typename StateInteger, typename BitInteger, typename... Qubits>
  inline auto expectation_value(
    RandomAccessIterator const first, RandomAccessIterator const last,
    Observable&& observable, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
  -> ::ket::utility::meta::real_t<typename std::iterator_traits<RandomAccessIterator>::value_type>
  { return ::ket::expectation_value(::ket::utility::policy::make_sequential(), first, last, std::forward<Observable>(observable), qubit, qubits...); }

  namespace ranges
  {
    template <typename ParallelPolicy, typename RandomAccessRange, typename Observable, typename StateInteger, typename BitInteger, typename... Qubits>
    inline auto expectation_value(
      ParallelPolicy const parallel_policy,
      RandomAccessRange const& state, Observable&& observable, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
    -> ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<RandomAccessRange> >
    { using std::begin; using std::end; return ::ket::expectation_value(parallel_policy, begin(state), end(state), std::forward<Observable>(observable), qubit, qubits...); }

    template <typename RandomAccessRange, typename Observable, typename StateInteger, typename BitInteger, typename... Qubits>
    inline auto expectation_value(
      RandomAccessRange const& state, Observable&& observable, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
    -> ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<RandomAccessRange> >
    { using std::begin; using std::end; return ::ket::expectation_value(begin(state), end(state), std::forward<Observable>(observable), qubit, qubits...); }
  } // namespace ranges
} // namespace ket


#endif // KET_EXPECTATION_VALUE_HPP
