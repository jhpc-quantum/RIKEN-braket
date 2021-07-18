#ifndef KET_SPIN_EXPECTATION_VALUE_HPP
# define KET_SPIN_EXPECTATION_VALUE_HPP

# include <boost/config.hpp>

# include <array>
# include <vector>
# include <iterator>
# include <numeric>

# include <boost/range/value_type.hpp>
# include <boost/math/constants/constants.hpp>

# include <ket/qubit.hpp>
# include <ket/gate/gate.hpp>
# include <ket/utility/loop_n.hpp>
# include <ket/utility/meta/real_of.hpp>


namespace ket
{
  template <typename ParallelPolicy, typename RandomAccessIterator, typename StateInteger, typename BitInteger>
  inline
  std::array<
    typename ::ket::utility::meta::real_of<
      typename std::iterator_traits<RandomAccessIterator>::value_type>::type, 3u>
  spin_expectation_value(
    ParallelPolicy const parallel_policy,
    RandomAccessIterator const first, RandomAccessIterator const last, ::ket::qubit<StateInteger, BitInteger> const qubit)
  {
    using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
    using hd_spin_type = std::array<long double, 3u>;
    constexpr auto zero_spin = hd_spin_type{ };
    auto spins_in_threads
      = std::vector<hd_spin_type>(::ket::utility::num_threads(parallel_policy), zero_spin);

    ::ket::gate::gate(
      parallel_policy, first, last,
      [&spins_in_threads](RandomAccessIterator const first, std::array<StateInteger, 2u> const& indices, int const thread_index)
      {
        using std::conj;
# ifndef BOOST_NO_CXX14_BINARY_LITERALS
        auto const conj_zero_value = conj(*(first + indices[0b0u]));
        auto const one_value = *(first + indices[0b1u]);
# else // BOOST_NO_CXX14_BINARY_LITERALS
        auto const conj_zero_value = conj(*(first + indices[0u]));
        auto const one_value = *(first + indices[1u]);
# endif // BOOST_NO_CXX14_BINARY_LITERALS
        auto const conj_zero_times_one = conj_zero_value * one_value;

        using std::real;
        spins_in_threads[thread_index][0u] += static_cast<long double>(real(conj_zero_times_one));
        using std::imag;
        spins_in_threads[thread_index][1u] += static_cast<long double>(imag(conj_zero_times_one));
        using std::norm;
        spins_in_threads[thread_index][2u]
          += static_cast<long double>(norm(conj_zero_value)) - static_cast<long double>(norm(one_value));
      },
      qubit);

    auto const hd_spin
      = std::accumulate(
          std::begin(spins_in_threads), std::end(spins_in_threads), zero_spin,
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
  std::array<
    typename ::ket::utility::meta::real_of<
      typename std::iterator_traits<RandomAccessIterator>::value_type>::type, 3u>
  spin_expectation_value(
    RandomAccessIterator const first, RandomAccessIterator const last, ::ket::qubit<StateInteger, BitInteger> const qubit)
  { return ::ket::spin_expectation_value(::ket::utility::policy::make_sequential(), first, last, qubit); }

  namespace ranges
  {
    template <typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger>
    inline
    std::array<
      typename ::ket::utility::meta::real_of<
        typename boost::range_value<RandomAccessRange const>::type>::type, 3u>
    spin_expectation_value(
      ParallelPolicy const parallel_policy,
      RandomAccessRange const& state, ::ket::qubit<StateInteger, BitInteger> const qubit)
    { return ::ket::spin_expectation_value(parallel_policy, std::begin(state), std::end(state), qubit); }

    template <typename RandomAccessRange, typename StateInteger, typename BitInteger>
    inline
    std::array<
      typename ::ket::utility::meta::real_of<
        typename boost::range_value<RandomAccessRange const>::type>::type, 3u>
    spin_expectation_value(
      RandomAccessRange const& state,
      ::ket::qubit<StateInteger, BitInteger> const qubit)
    { return ::ket::spin_expectation_value(std::begin(state), std::end(state), qubit); }
  } // namespace ranges
} // namespace ket


#endif // KET_SPIN_EXPECTATION_VALUE_HPP
