#ifndef KET_SPIN_EXPECTATION_VALUE_HPP
# define KET_SPIN_EXPECTATION_VALUE_HPP

# include <array>
# include <vector>
# include <iterator>
# include <numeric>

# include <boost/math/constants/constants.hpp>

# include <ket/qubit.hpp>
# include <ket/gate/gate.hpp>
# include <ket/utility/loop_n.hpp>
# include <ket/utility/meta/real_of.hpp>
# include <ket/utility/meta/ranges.hpp>


namespace ket
{
  template <typename ParallelPolicy, typename RandomAccessIterator, typename StateInteger, typename BitInteger>
  inline auto spin_expectation_value(
    ParallelPolicy const parallel_policy,
    RandomAccessIterator const first, RandomAccessIterator const last, ::ket::qubit<StateInteger, BitInteger> const qubit)
  -> std::array< ::ket::utility::meta::real_t<typename std::iterator_traits<RandomAccessIterator>::value_type>, 3u >
  {
    using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
    using hd_spin_type = std::array<long double, 3u>;
    constexpr hd_spin_type zero_spin{ };
    auto spins_in_threads
      = std::vector<hd_spin_type>(::ket::utility::num_threads(parallel_policy), zero_spin);

    auto const qubit_mask = ::ket::utility::integer_exp2<StateInteger>(qubit);
    auto const lower_bits_mask = qubit_mask - StateInteger{1u};
    auto const upper_bits_mask = compl lower_bits_mask;

    ::ket::utility::loop_n(
      parallel_policy, static_cast<StateInteger>(last - first) >> 1u,
      [first, &spins_in_threads, qubit_mask, lower_bits_mask, upper_bits_mask](StateInteger const value_wo_qubit, int const thread_index)
      {
        // xxxxx0xxxxxx
        auto const zero_index = ((value_wo_qubit bitand upper_bits_mask) << 1u) bitor (value_wo_qubit bitand lower_bits_mask);
        // xxxxx1xxxxxx
        auto const one_index = zero_index bitor qubit_mask;
        using std::conj;
        auto const conj_zero_value = conj(*(first + zero_index));
        auto const one_value = *(first + one_index);
        auto const conj_zero_times_one = conj_zero_value * one_value;

        using std::real;
        spins_in_threads[thread_index][0u] += static_cast<long double>(real(conj_zero_times_one));
        using std::imag;
        spins_in_threads[thread_index][1u] += static_cast<long double>(imag(conj_zero_times_one));
        using std::norm;
        spins_in_threads[thread_index][2u]
          += static_cast<long double>(norm(conj_zero_value)) - static_cast<long double>(norm(one_value));
      });

    using std::begin;
    using std::end;
    auto const hd_spin
      = std::accumulate(
          begin(spins_in_threads), end(spins_in_threads), zero_spin,
          [](hd_spin_type accumulated_spin, hd_spin_type const& spin)
          {
            accumulated_spin[0u] += spin[0u];
            accumulated_spin[1u] += spin[1u];
            accumulated_spin[2u] += spin[2u];
            return accumulated_spin;
          });

    using real_type = ::ket::utility::meta::real_t<complex_type>;
    using spin_type = std::array<real_type, 3u>;
    spin_type spin{static_cast<real_type>(hd_spin[0u]), static_cast<real_type>(hd_spin[1u]), static_cast<real_type>(hd_spin[2u])};

    using boost::math::constants::half;
    spin[2u] *= half<real_type>();

    return spin;
  }

  template <typename RandomAccessIterator, typename StateInteger, typename BitInteger>
  inline auto spin_expectation_value(
    RandomAccessIterator const first, RandomAccessIterator const last, ::ket::qubit<StateInteger, BitInteger> const qubit)
  -> std::array< ::ket::utility::meta::real_t<typename std::iterator_traits<RandomAccessIterator>::value_type>, 3u >
  { return ::ket::spin_expectation_value(::ket::utility::policy::make_sequential(), first, last, qubit); }

  namespace ranges
  {
    template <typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger>
    inline auto spin_expectation_value(
      ParallelPolicy const parallel_policy, RandomAccessRange const& state, ::ket::qubit<StateInteger, BitInteger> const qubit)
    -> std::array< ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<RandomAccessRange> >, 3u >
    { using std::begin; using std::end; return ::ket::spin_expectation_value(parallel_policy, begin(state), end(state), qubit); }

    template <typename RandomAccessRange, typename StateInteger, typename BitInteger>
    inline auto spin_expectation_value(RandomAccessRange const& state, ::ket::qubit<StateInteger, BitInteger> const qubit)
    -> std::array< ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<RandomAccessRange> >, 3u >
    { using std::begin; using std::end; return ::ket::spin_expectation_value(begin(state), end(state), qubit); }
  } // namespace ranges
} // namespace ket


#endif // KET_SPIN_EXPECTATION_VALUE_HPP
