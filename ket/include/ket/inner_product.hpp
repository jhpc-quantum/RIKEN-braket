#ifndef KET_INNER_PRODUCT_HPP
# define KET_INNER_PRODUCT_HPP

# include <complex>
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
# ifndef NDEBUG
#   include <ket/utility/integer_exp2.hpp>
#   include <ket/utility/integer_log2.hpp>
#   include <ket/utility/all_in_state_vector.hpp>
# endif
# include <ket/utility/variadic/all_of.hpp>
# include <ket/utility/meta/ranges.hpp>


namespace ket
{
  // <Psi_2|Psi_1>
  template <typename ParallelPolicy, typename RandomAccessIterator1, typename RandomAccessIterator2>
  inline auto inner_product(
    ParallelPolicy const parallel_policy,
    RandomAccessIterator1 const first1, RandomAccessIterator1 const last1, RandomAccessIterator2 const first2)
  -> typename std::iterator_traits<RandomAccessIterator1>::value_type
  {
    using complex_type = typename std::iterator_traits<RandomAccessIterator1>::value_type;
    static_assert(
      std::is_same<complex_type, typename std::iterator_traits<RandomAccessIterator2>::value_type>::value,
      "value_type's of RandomAccessIterator1 and RandomAccessIterator2 should be the same");
    auto partial_sums = std::vector<complex_type>(::ket::utility::num_threads(parallel_policy));

    using difference_type = typename std::iterator_traits<RandomAccessIterator1>::difference_type;
    static_assert(
      std::is_convertible<difference_type, typename std::iterator_traits<RandomAccessIterator2>::difference_type>::value,
      "difference_type of RandomAccessIterator1 should be convertible to that of RandomAccessIterator2");
    ::ket::utility::loop_n(
      parallel_policy, last1 - first1,
      [&partial_sums, first1, first2](difference_type const index, int const thread_index)
      //{ using std::conj; partial_sums[thread_index] += conj(first2[index]) * first1[index]; });
      { using std::conj; partial_sums[thread_index] += conj(*(first2 + index)) * *(first1 + index); });

    using std::begin;
    using std::end;
    return std::accumulate(begin(partial_sums), end(partial_sums), complex_type{});
  }

  template <typename RandomAccessIterator1, typename RandomAccessIterator2>
  inline auto inner_product(
    RandomAccessIterator1 const first1, RandomAccessIterator1 const last1, RandomAccessIterator2 const first2)
  -> typename std::iterator_traits<RandomAccessIterator1>::value_type
  { return ::ket::inner_product(::ket::utility::policy::make_sequential(), first1, last1, first2); }

  namespace ranges
  {
    template <typename ParallelPolicy, typename RandomAccessRange1, typename RandomAccessRange2>
    inline auto inner_product(
      ParallelPolicy const parallel_policy,
      RandomAccessRange1 const& state1, RandomAccessRange2 const& state2)
    -> ::ket::utility::meta::range_value_t<RandomAccessRange1>
    { using std::begin; using std::end; return ::ket::inner_product(parallel_policy, begin(state1), end(state1), begin(state2)); }

    template <typename RandomAccessRange1, typename RandomAccessRange2>
    inline auto inner_product(RandomAccessRange1 const& state1, RandomAccessRange2 const& state2)
    -> ::ket::utility::meta::range_value_t<RandomAccessRange1>
    { using std::begin; using std::end; return ::ket::inner_product(begin(state1), end(state1), begin(state2)); }
  } // namespace ranges

  // <Psi_2| A_{ij} |Psi_1>
  namespace inner_product_detail
  {
# if __cpp_constexpr < 201603L
    template <typename U>
    struct is_same_to
    {
      template <typename T>
      constexpr auto operator()(T) const noexcept -> bool { return std::is_same<T, U>::value; }
    }; // struct is_same_to<U>

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
  } // namespace inner_product_detail

  template <
    typename ParallelPolicy, typename RandomAccessIterator1, typename RandomAccessIterator2,
    typename Observable, typename StateInteger, typename BitInteger, typename... Qubits>
  inline auto inner_product(
    ParallelPolicy const parallel_policy,
    RandomAccessIterator1 const first1, RandomAccessIterator1 const last1, RandomAccessIterator2 const first2,
    Observable&& observable, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
  -> typename std::iterator_traits<RandomAccessIterator1>::value_type
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
        ::ket::inner_product_detail::is_same_to<StateInteger>{}, ::ket::inner_product_detail::state_integer_of{},
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
        ::ket::inner_product_detail::is_same_to<BitInteger>{}, ::ket::inner_product_detail::bit_integer_of{},
        std::remove_cv_t<std::remove_reference_t<Qubits>>{}...),
      "bit_integer_type's of Qubits should be same to BitInteger");
# endif // __cpp_constexpr >= 201603L

    constexpr auto num_operated_qubits = static_cast<BitInteger>(sizeof...(Qubits) + 1u);

# ifndef NDEBUG
    auto const state_size = static_cast<StateInteger>(last1 - first1);
    auto const num_qubits = ::ket::utility::integer_log2<BitInteger>(state_size);
# endif // NDEBUG
    assert(::ket::utility::integer_exp2<StateInteger>(num_qubits) == state_size);
    assert(::ket::utility::all_in_state_vector(num_qubits, qubit, qubits...));

    using complex_type = typename std::iterator_traits<RandomAccessIterator1>::value_type;
    static_assert(
      std::is_same<complex_type, typename std::iterator_traits<RandomAccessIterator2>::value_type>::value,
      "value_type's of RandomAccessIterator1 and RandomAccessIterator2 should be the same");
    auto partial_sums = std::vector<complex_type>(::ket::utility::num_threads(parallel_policy));

# ifndef KET_USE_BIT_MASKS_EXPLICITLY
    using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
    ::ket::gate::nocache::gate(
      parallel_policy, first1, last1,
      [first2, &observable, &partial_sums](
        auto const first1, StateInteger const index_wo_qubits,
        std::array<qubit_type, num_operated_qubits> const& unsorted_qubits,
        std::array<qubit_type, num_operated_qubits + 1u> const& sorted_qubits_with_sentinel,
        int const thread_index)
      { partial_sums[thread_index] += observable(first1, first2, index_wo_qubits, unsorted_qubits, sorted_qubits_with_sentinel); },
      qubit, qubits...);
# else // KET_USE_BIT_MASKS_EXPLICITLY
    ::ket::gate::nocache::gate(
      parallel_policy, first1, last1,
      [first2, &observable, &partial_sums](
        auto const first1, StateInteger const index_wo_qubits,
        std::array<StateInteger, num_operated_qubits> const& qubit_masks,
        std::array<StateInteger, num_operated_qubits + 1u> const& index_masks,
        int const thread_index)
      { partial_sums[thread_index] += observable(first1, first2, index_wo_qubits, qubit_masks, index_masks); },
      qubit, qubits...);
# endif // KET_USE_BIT_MASKS_EXPLICITLY

    using std::begin;
    using std::end;
    return std::accumulate(begin(partial_sums), end(partial_sums), complex_type{});
  }

  template <
    typename RandomAccessIterator1, typename RandomAccessIterator2,
    typename Observable, typename StateInteger, typename BitInteger, typename... Qubits>
  inline auto inner_product(
    RandomAccessIterator1 const first1, RandomAccessIterator1 const last1, RandomAccessIterator2 const first2,
    Observable&& observable, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
  -> typename std::iterator_traits<RandomAccessIterator1>::value_type
  { return ::ket::inner_product(::ket::utility::policy::make_sequential(), first1, last1, first2, std::forward<Observable>(observable), qubit, qubits...); }

  namespace ranges
  {
    template <
      typename ParallelPolicy, typename RandomAccessRange1, typename RandomAccessRange2,
      typename Observable, typename StateInteger, typename BitInteger, typename... Qubits>
    inline auto inner_product(
      ParallelPolicy const parallel_policy,
      RandomAccessRange1 const& state1, RandomAccessRange2 const& state2,
      Observable&& observable, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
    -> ::ket::utility::meta::range_value_t<RandomAccessRange1>
    { using std::begin; using std::end; return ::ket::inner_product(parallel_policy, begin(state1), end(state1), begin(state2), std::forward<Observable>(observable), qubit, qubits...); }

    template <
      typename RandomAccessRange1, typename RandomAccessRange2,
      typename Observable, typename StateInteger, typename BitInteger, typename... Qubits>
    inline auto inner_product(
      RandomAccessRange1 const& state1, RandomAccessRange2 const& state2,
      Observable&& observable, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
    -> ::ket::utility::meta::range_value_t<RandomAccessRange1>
    { using std::begin; using std::end; return ::ket::inner_product(begin(state1), end(state1), begin(state2), std::forward<Observable>(observable), qubit, qubits...); }
  } // namespace ranges
} // namespace ket


#endif // KET_INNER_PRODUCT_HPP
