#ifndef KET_SPIN_EXPECTATION_VALUE_HPP
# define KET_SPIN_EXPECTATION_VALUE_HPP

# include <cassert>
# include <vector>
# include <iterator>
# include <numeric>
# include <utility>
# ifndef NDEBUG
#   include <type_traits>
# endif
# include <array>

# include <boost/range/value_type.hpp>
# include <boost/math/constants/constants.hpp>

# include <ket/qubit.hpp>
# include <ket/utility/loop_n.hpp>
# include <ket/utility/integer_exp2.hpp>
# ifndef NDEBUG
#   include <ket/utility/integer_log2.hpp>
# endif
# include <ket/utility/begin.hpp>
# include <ket/utility/end.hpp>
# include <ket/utility/meta/real_of.hpp>


namespace ket
{
  template <
    typename ParallelPolicy,
    typename RandomAccessIterator, typename StateInteger, typename BitInteger>
  inline
  std::array<
    typename ::ket::utility::meta::real_of<
      typename std::iterator_traits<RandomAccessIterator>::value_type>::type, 3u>
  spin_expectation_value(
    ParallelPolicy const parallel_policy,
    RandomAccessIterator const first, RandomAccessIterator const last,
    ::ket::qubit<StateInteger, BitInteger> const qubit)
  {
    static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
    static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");
    assert(
      ::ket::utility::integer_exp2<StateInteger>(qubit)
      < static_cast<StateInteger>(last-first));
    assert(
      ::ket::utility::integer_exp2<StateInteger>(
        ::ket::utility::integer_log2<BitInteger>(last-first))
      == static_cast<StateInteger>(last-first));

    auto const qubit_mask = ::ket::utility::integer_exp2<StateInteger>(qubit);
    auto const lower_bits_mask = qubit_mask - StateInteger{1u};
    auto const upper_bits_mask = compl lower_bits_mask;

    using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
    using hd_spin_type = std::array<long double, 3u>;
    auto constexpr zero_spin = hd_spin_type{ };
    auto spins_in_threads
      = std::vector<hd_spin_type>(::ket::utility::num_threads(parallel_policy), zero_spin);

    using ::ket::utility::loop_n;
    loop_n(
      parallel_policy,
      static_cast<StateInteger>(last - first)/2u,
      [first, qubit_mask, lower_bits_mask, upper_bits_mask,
       &spins_in_threads](
        StateInteger const value_wo_qubit, int const thread_index)
      {
        // xxxxx0xxxxxx
        auto const zero_index
          = ((value_wo_qubit bitand upper_bits_mask) << 1u)
            bitor (value_wo_qubit bitand lower_bits_mask);
        // xxxxx1xxxxxx
        auto const one_index = zero_index bitor qubit_mask;

        using std::conj;
        auto const conj_zero_value = conj(*(first+zero_index));
        auto const one_value = *(first+one_index);
        auto const conj_zero_times_one = conj_zero_value * one_value;

        using std::real;
        spins_in_threads[thread_index][0u] += static_cast<long double>(real(conj_zero_times_one));
        using std::imag;
        spins_in_threads[thread_index][1u] += static_cast<long double>(imag(conj_zero_times_one));
        using std::norm;
        spins_in_threads[thread_index][2u]
          += static_cast<long double>(norm(conj_zero_value)) - static_cast<long double>(norm(one_value));
      });

    hd_spin_type hd_spin
      = std::accumulate(
          ::ket::utility::begin(spins_in_threads), ::ket::utility::end(spins_in_threads), zero_spin,
          [](hd_spin_type accumulated_spin, hd_spin_type const& spin)
          {
            accumulated_spin[0u] += spin[0u];
            accumulated_spin[1u] += spin[1u];
            accumulated_spin[2u] += spin[2u];
            return accumulated_spin;
          });

    using real_type = typename ::ket::utility::meta::real_of<complex_type>::type;
    using spin_type = std::array<real_type, 3u>;
    auto spin = spin_type{};
    spin[0u] = static_cast<real_type>(hd_spin[0u]);
    spin[1u] = static_cast<real_type>(hd_spin[1u]);
    spin[2u] = static_cast<real_type>(hd_spin[2u]);

    using boost::math::constants::half;
    spin[2u] *= half<real_type>();

    return spin;
  }

  template <typename RandomAccessIterator, typename StateInteger, typename BitInteger>
  inline
  typename std::enable_if<
    not ::ket::utility::policy::meta::is_loop_n_policy<RandomAccessIterator>::value,
    std::array<
      typename ::ket::utility::meta::real_of<
        typename std::iterator_traits<RandomAccessIterator>::value_type>::type, 3u> >::type
  spin_expectation_value(
    RandomAccessIterator const first, RandomAccessIterator const last,
    ::ket::qubit<StateInteger, BitInteger> const qubit)
  { return ::ket::spin_expectation_value(::ket::utility::policy::make_sequential(), first, last, qubit); }


  namespace ranges
  {
    template <
      typename ParallelPolicy,
      typename RandomAccessRange, typename StateInteger, typename BitInteger>
    inline
    typename std::enable_if<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      std::array<
        typename ::ket::utility::meta::real_of<
          typename boost::range_value<RandomAccessRange const>::type>::type, 3u> >::type
    spin_expectation_value(
      ParallelPolicy const parallel_policy,
      RandomAccessRange const& state,
      ::ket::qubit<StateInteger, BitInteger> const qubit)
    { return ::ket::spin_expectation_value(parallel_policy, ::ket::utility::begin(state), ::ket::utility::end(state), qubit); }

    template <typename RandomAccessRange, typename StateInteger, typename BitInteger>
    inline
    std::array<
      typename ::ket::utility::meta::real_of<
        typename boost::range_value<RandomAccessRange const>::type>::type, 3u>
    spin_expectation_value(
      RandomAccessRange const& state,
      ::ket::qubit<StateInteger, BitInteger> const qubit)
    { return ::ket::spin_expectation_value(::ket::utility::begin(state), ::ket::utility::end(state), qubit); }
  } // namespace ranges
} // namespace ket


#endif // KET_SPIN_EXPECTATION_VALUE_HPP
