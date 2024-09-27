#ifndef KET_SHOR_BOX_HPP
# define KET_SHOR_BOX_HPP

# include <cmath>
//# include <vector>
# include <iterator>
# include <type_traits>

# include <ket/qubit.hpp>
# include <ket/meta/state_integer_of.hpp>
# include <ket/meta/bit_integer_of.hpp>
# include <ket/utility/integer_exp2.hpp>
# include <ket/utility/meta/real_of.hpp>
# include <ket/utility/meta/ranges.hpp>


namespace ket
{
  namespace shor_box_detail
  {
    template <typename StateInteger, typename BitInteger>
    inline constexpr auto reverse_bits(StateInteger const integer, BitInteger const num_bits) -> StateInteger
    {
      auto result = StateInteger{0u};

      for (auto next_from_bit = num_bits, to_bit = BitInteger{0u};
           next_from_bit > BitInteger{0u}; --next_from_bit, ++to_bit)
        result |= ((integer >> (next_from_bit - BitInteger{1u})) bitand StateInteger{1u}) << to_bit;

      return result;
    }

    template <typename UnsignedInteger, typename Qubits>
    inline auto make_filtered_integer(UnsignedInteger const unsigned_integer, Qubits const& qubits) -> UnsignedInteger
    {
      static_assert(std::is_unsigned<UnsignedInteger>::value, "UnsignedInteger should be unsigned.");

      using bit_integer_type = ::ket::meta::bit_integer_t< ::ket::utility::meta::range_value_t<Qubits> >;
      static_assert(std::is_unsigned< ::ket::meta::state_integer_t< ::ket::utility::meta::range_value_t<Qubits> > >::value, "StateInteger should be unsigned");
      static_assert(std::is_unsigned<bit_integer_type>::value, "BitInteger should be unsigned");

      using std::begin;
      using std::end;
      auto const qubits_first = begin(qubits);
      auto const num_qubits = static_cast<bit_integer_type>(std::distance(qubits_first, end(qubits)));

      auto result = UnsignedInteger{0u};
      for (auto index = bit_integer_type{0u}; index < num_qubits; ++index)
        result |= ((unsigned_integer >> index) bitand UnsignedInteger{1u}) << *(qubits_first + index);

      return result;
    }

    template <typename StateInteger, typename Qubits>
    inline auto calculate_index(
      StateInteger const exponent, Qubits const& exponent_qubits,
      StateInteger const modular_exponentiation_value, Qubits const& modular_exponentiation_qubits)
    -> StateInteger
    {
      return
        ::ket::shor_box_detail::make_filtered_integer(exponent, exponent_qubits)
        bitor ::ket::shor_box_detail::make_filtered_integer(modular_exponentiation_value, modular_exponentiation_qubits);
    }
  } // namespace shor_box_detail


  template <
    typename ParallelPolicy,
    typename RandomAccessIterator, typename StateInteger, typename Qubits>
  inline auto shor_box(
    ParallelPolicy const parallel_policy,
    RandomAccessIterator const first, RandomAccessIterator const last,
    StateInteger const base, StateInteger const divisor,
    Qubits const& exponent_qubits, Qubits const& modular_exponentiation_qubits)
  -> void
  {
    using qubit_type = ::ket::utility::meta::range_value_t<Qubits>;
    using bit_integer_type = ::ket::meta::bit_integer_t<qubit_type>;
    static_assert(
      (std::is_same<StateInteger, ::ket::meta::state_integer_t<qubit_type>>::value),
      "StateInteger should be state_integer_type of qubit_type");
    static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
    static_assert(std::is_unsigned<bit_integer_type>::value, "BitInteger should be unsigned");

    using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
    using real_type = ::ket::utility::meta::real_t<complex_type>;
    ::ket::utility::fill(parallel_policy, first, last, complex_type{real_type{0}});

    using std::begin;
    using std::end;
    auto const num_exponent_qubits = static_cast<bit_integer_type>(std::distance(begin(exponent_qubits), end(exponent_qubits)));
    auto const num_exponents = ::ket::utility::integer_exp2<StateInteger>(num_exponent_qubits);
    auto modular_exponentiation_value = StateInteger{1u};

    using std::pow;
    auto const constant_coefficient = complex_type{pow(static_cast<real_type>(num_exponents), real_type{-0.5})};

    for (auto exponent = StateInteger{0u}; exponent < num_exponents; ++exponent)
    {
      auto const index
        = ::ket::shor_box_detail::calculate_index(
            ::ket::shor_box_detail::reverse_bits(exponent, num_exponent_qubits), exponent_qubits,
            modular_exponentiation_value, modular_exponentiation_qubits);

      *(first + index) = constant_coefficient;

      modular_exponentiation_value *= base;
      modular_exponentiation_value %= divisor;
    }


    /*
    // preliminary implement of parallel shor box
    using std::begin;
    using std::end;
    auto const num_exponent_qubits = static_cast<bit_integer_type>(std::distance(begin(exponent_qubits), end(exponent_qubits)));
    auto const num_modular_exponentiation_qubits
      = static_cast<bit_integer_type>(std::distance(begin(modular_exponentiation_qubits), end(modular_exponentiation_qubits)));

    using qubit_type = ::ket::utility::meta::range_value_t<Qubits>;

    static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
    static_assert(
      std::is_unsigned< ::ket::meta::bit_integer_t<qubit_type> >::value,
      "BitInteger should be unsigned");

    auto modular_squares = std::vector<StateInteger>{};
    modular_squares.reserve(num_exponent_qubits);
    modular_squares.push_back(base);
    ::ket::utility::loop_n(
      ::ket::utility::policy::make_sequential(), num_exponent_qubits - decltype(num_exponent_qubits){1u},
      [&modular_squares, divisor](decltype(num_exponent_qubits) const, int const)
      { modular_squares.push_back((modular_squares.back() * modular_squares.back()) % divisor); });

    auto modular_exponentiation_values = std::vector<StateInteger>{};
    modular_exponentiation_values.reserve(::ket::utility::num_threads());

    ::ket::utility::loop_n(
      parallel_policy, static_cast<StateInteger>(std::distance(begin(state), end(state))),
      [first](StateInteger const index, int const)
      { *(first + index) = complex_type{real_type{0}}; });

    using std::pow;
    auto const constant_coefficient = complex_type{pow(num_exponents, -0.5)};
    auto const num_exponents = ::ket::utility::integer_exp2<StateInteger>(num_exponent_qubits);
    auto modular_exponentiation_value = StateInteger{1u};
    auto const num_threads = ::ket::utility::num_threads();
    ::ket::utility::loop_n(
      parallel_policy, num_exponents,
      [](StateInteger const exponent, int const thread_index)
      {
        a
      });


    using std::pow;
    auto const constant_coefficient = complex_type{pow(num_exponents, -0.5)};
    auto const num_exponents = ::ket::utility::integer_exp2<StateInteger>(num_exponent_qubits);
    auto modular_exponentiation_value = StateInteger{1u};
    for (auto exponent = StateInteger{0u}; exponent < num_exponents; ++exponent)
    {
      auto const index
        = ::ket::shor_box_detail::reverse_bits(exponent, num_exponent_qubits)
          bitor (modular_exponentiation_value << num_exponent_qubits);

      *(first + convert(index, exponent_qubits, modular_exponentiation_qubits))
        = constant_coefficient;

      modular_exponentiation_value *= base;
      modular_exponentiation_value %= divisor;
    }
    */
  }

  template <typename RandomAccessIterator, typename StateInteger, typename Qubits>
  inline auto shor_box(
    RandomAccessIterator const first, RandomAccessIterator const last,
    StateInteger const base, StateInteger const divisor,
    Qubits const& exponent_qubits, Qubits const& modular_exponentiation_qubits)
  -> void
  {
    ::ket::shor_box(
      ::ket::utility::policy::make_sequential(),
      first, last, base, divisor, exponent_qubits, modular_exponentiation_qubits);
  }

  namespace ranges
  {
    template <
      typename ParallelPolicy,
      typename RandomAccessRange, typename StateInteger, typename Qubits>
    inline auto shor_box(
      ParallelPolicy const parallel_policy, RandomAccessRange& state,
      StateInteger const base, StateInteger const divisor,
      Qubits const& exponent_qubits, Qubits const& modular_exponentiation_qubits)
    -> RandomAccessRange&
    {
      using std::begin;
      using std::end;
      ::ket::shor_box(
        parallel_policy, begin(state), end(state),
        base, divisor, exponent_qubits, modular_exponentiation_qubits);
      return state;
    }

    template <typename RandomAccessRange, typename StateInteger, typename Qubits>
    inline auto shor_box(
      RandomAccessRange& state,
      StateInteger const base, StateInteger const divisor,
      Qubits const& exponent_qubits, Qubits const& modular_exponentiation_qubits)
    -> RandomAccessRange&
    {
      using std::begin;
      using std::end;
      ::ket::shor_box(
        ::ket::utility::policy::make_sequential(),
        begin(state), end(state), base, divisor, exponent_qubits, modular_exponentiation_qubits);
      return state;
    }
  } // namespace ranges
} // namespace ket


#endif // KET_SHOR_BOX_HPP
