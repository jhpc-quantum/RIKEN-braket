#ifndef KET_GATE_SQRT_PAULI_Y_HPP
# define KET_GATE_SQRT_PAULI_Y_HPP

# include <cassert>
# include <array>
# include <iterator>
# include <algorithm>
# include <utility>
# include <type_traits>

# include <boost/math/constants/constants.hpp>
# include <boost/range/iterator_range.hpp>
# include <boost/range/join.hpp>
# include <boost/range/adaptor/transformed.hpp>

# include <ket/qubit.hpp>
# include <ket/control.hpp>
# include <ket/gate/gate.hpp>
# include <ket/gate/utility/index_with_qubits.hpp>
# include <ket/utility/loop_n.hpp>
# include <ket/utility/integer_exp2.hpp>
# ifndef NDEBUG
#   include <ket/utility/integer_log2.hpp>
# endif
# include <ket/utility/meta/real_of.hpp>


namespace ket
{
  namespace gate
  {
    // sY_i
    // sY_1 (a_0 |0> + a_1 |1>) = [(1+i) a_0 - (1+i) a_1]/2 |0> + [(1+i) a_0 + (1+i) a_1]/2 |1>
    template <typename ParallelPolicy, typename RandomAccessIterator, typename StateInteger, typename BitInteger>
    inline auto sqrt_pauli_y(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      ::ket::qubit<StateInteger, BitInteger> const qubit)
    -> void
    {
      static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
      static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");
      assert(::ket::utility::integer_exp2<StateInteger>(qubit) < static_cast<StateInteger>(last - first));
      assert(
        ::ket::utility::integer_exp2<StateInteger>(::ket::utility::integer_log2<BitInteger>(last - first))
        == static_cast<StateInteger>(last - first));

      auto const qubit_mask = ::ket::utility::integer_exp2<StateInteger>(qubit);
      auto const lower_bits_mask = qubit_mask - StateInteger{1u};
      auto const upper_bits_mask = compl lower_bits_mask;

      ::ket::utility::loop_n(
        parallel_policy, static_cast<StateInteger>(last - first) >> 1u,
        [first, qubit_mask, lower_bits_mask, upper_bits_mask](StateInteger const value_wo_qubit, int const)
        {
          // xxxxx0xxxxxx
          auto const zero_index = ((value_wo_qubit bitand upper_bits_mask) << 1u) bitor (value_wo_qubit bitand lower_bits_mask);
          // xxxxx1xxxxxx
          auto const one_index = zero_index bitor qubit_mask;
          auto const zero_iter = first + zero_index;
          auto const one_iter = first + one_index;
          auto const zero_iter_value = *zero_iter;

          using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
          using real_type = ::ket::utility::meta::real_t<complex_type>;
          using boost::math::constants::half;
          constexpr auto half_one_plus_i = complex_type{half<real_type>(), half<real_type>()};
          *zero_iter *= half_one_plus_i;
          *zero_iter -= half_one_plus_i * *one_iter;
          *one_iter *= half_one_plus_i;
          *one_iter += half_one_plus_i * zero_iter_value;
        });
    }

    // CsY_{tc} or C1sY_{tc}
    // CsY_{1,2} (a_{00} |00> + a_{01} |01> + a_{10} |10> + a_{11} |11>)
    //   = a_{00} |00> + a_{01} |01> + [(1+i) a_{10} - (1+i) a_{11}]/2 |10> + [(1+i) a_{10} + (1+i) a_{11}]/2 |11>
    template <typename ParallelPolicy, typename RandomAccessIterator, typename StateInteger, typename BitInteger>
    inline auto sqrt_pauli_y(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit)
    -> void
    {
      static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
      static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");

      assert(::ket::utility::integer_exp2<StateInteger>(target_qubit) < static_cast<StateInteger>(last - first));
      assert(::ket::utility::integer_exp2<StateInteger>(control_qubit) < static_cast<StateInteger>(last - first));
      assert(target_qubit != control_qubit);
      assert(
        ::ket::utility::integer_exp2<StateInteger>(::ket::utility::integer_log2<BitInteger>(last - first))
        == static_cast<StateInteger>(last - first));

      auto const minmax_qubits = std::minmax(target_qubit, control_qubit.qubit());
      auto const target_qubit_mask = ::ket::utility::integer_exp2<StateInteger>(target_qubit);
      auto const control_qubit_mask = ::ket::utility::integer_exp2<StateInteger>(control_qubit);
      auto const lower_bits_mask = ::ket::utility::integer_exp2<StateInteger>(minmax_qubits.first) - StateInteger{1u};
      auto const middle_bits_mask
        = (::ket::utility::integer_exp2<StateInteger>(minmax_qubits.second - BitInteger{1u}) - StateInteger{1u})
          xor lower_bits_mask;
      auto const upper_bits_mask = compl (lower_bits_mask bitor middle_bits_mask);

      ::ket::utility::loop_n(
        parallel_policy, static_cast<StateInteger>(last - first) >> 2u,
        [first, target_qubit_mask, control_qubit_mask, lower_bits_mask, middle_bits_mask, upper_bits_mask](
          StateInteger const value_wo_qubits, int const)
        {
          // xxx0_txxx0_cxxx
          auto const base_index
            = ((value_wo_qubits bitand upper_bits_mask) << 2u)
              bitor ((value_wo_qubits bitand middle_bits_mask) << 1u)
              bitor (value_wo_qubits bitand lower_bits_mask);
          // xxx0_txxx1_cxxx
          auto const control_on_index = base_index bitor control_qubit_mask;
          // xxx1_txxx1_cxxx
          auto const target_control_on_index = control_on_index bitor target_qubit_mask;
          auto const control_on_iter = first + control_on_index;
          auto const target_control_on_iter = first + target_control_on_index;
          auto const control_on_iter_value = *control_on_iter;

          using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
          using real_type = ::ket::utility::meta::real_t<complex_type>;
          using boost::math::constants::half;
          constexpr auto half_one_plus_i = complex_type{half<real_type>(), half<real_type>()};
          *control_on_iter *= half_one_plus_i;
          *control_on_iter -= half_one_plus_i * *target_control_on_iter;
          *target_control_on_iter *= half_one_plus_i;
          *target_control_on_iter += half_one_plus_i * control_on_iter_value;
        });
    }

    // C...CsY_{tc...c'} or CnsY_{tc...c'}
    template <typename ParallelPolicy, typename RandomAccessIterator, typename StateInteger, typename BitInteger, typename... ControlQubits>
    inline auto sqrt_pauli_y(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2, ControlQubits const... control_qubits)
    -> void
    {
      static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
      static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");
      assert(::ket::utility::integer_exp2<StateInteger>(target_qubit) < static_cast<StateInteger>(last - first));
      assert(
        ::ket::utility::integer_exp2<StateInteger>(::ket::utility::integer_log2<BitInteger>(last - first))
        == static_cast<StateInteger>(last - first));

# ifndef KET_USE_BIT_MASKS_EXPLICITLY
      using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
      constexpr auto num_control_qubits = static_cast<BitInteger>(sizeof...(ControlQubits) + 2u);
      constexpr auto num_operated_qubits = num_control_qubits + BitInteger{1u};

      ::ket::gate::nocache::gate(
        parallel_policy, first, last,
        [](
          auto const first, StateInteger const index_wo_qubits,
          std::array<qubit_type, num_operated_qubits> const& unsorted_qubits,
          std::array<qubit_type, num_operated_qubits + BitInteger{1u}> const& sorted_qubits_with_sentinel,
          int const)
        {
          // 0b11...10u
          constexpr auto index0 = ((std::size_t{1u} << num_control_qubits) - std::size_t{1u}) << BitInteger{1u};
          // 0b11...11u
          constexpr auto index1 = index0 bitor std::size_t{1u};

          using std::begin;
          using std::end;
          auto const iter0
            = first
              + ::ket::gate::utility::ranges::index_with_qubits(index_wo_qubits, index0, unsorted_qubits, sorted_qubits_with_sentinel);
          auto const iter1
            = first
              + ::ket::gate::utility::ranges::index_with_qubits(index_wo_qubits, index1, unsorted_qubits, sorted_qubits_with_sentinel);
          auto const value0 = *iter0;

          using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
          using real_type = ::ket::utility::meta::real_t<complex_type>;
          using boost::math::constants::half;
          constexpr auto half_one_plus_i = complex_type{half<real_type>(), half<real_type>()};
          *iter0 *= half_one_plus_i;
          *iter0 -= half_one_plus_i * *iter1;
          *iter1 *= half_one_plus_i;
          *iter1 += half_one_plus_i * value0;
        },
        target_qubit, control_qubit1, control_qubit2, control_qubits...);
# else // KET_USE_BIT_MASKS_EXPLICITLY
      constexpr auto num_control_qubits = static_cast<BitInteger>(sizeof...(ControlQubits) + 2u);
      constexpr auto num_operated_qubits = num_control_qubits + BitInteger{1u};

      ::ket::gate::nocache::gate(
        parallel_policy, first, last,
        [](
          auto const first, StateInteger const index_wo_qubits,
          std::array<StateInteger, num_operated_qubits> const& qubit_masks,
          std::array<StateInteger, num_operated_qubits + 1u> const& index_masks,
          int const)
        {
          // 0b11...10u
          constexpr auto index0 = ((std::size_t{1u} << num_control_qubits) - std::size_t{1u}) << BitInteger{1u};
          // 0b11...11u
          constexpr auto index1 = index0 bitor std::size_t{1u};

          using std::begin;
          using std::end;
          auto const iter0
            = first
              + ::ket::gate::utility::ranges::index_with_qubits(index_wo_qubits, index0, qubit_masks, index_masks);
          auto const iter1
            = first
              + ::ket::gate::utility::ranges::index_with_qubits(index_wo_qubits, index1, qubit_masks, index_masks);
          auto const value0 = *iter0;

          using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
          using real_type = ::ket::utility::meta::real_t<complex_type>;
          using boost::math::constants::half;
          constexpr auto half_one_plus_i = complex_type{half<real_type>(), half<real_type>()};
          *iter0 *= half_one_plus_i;
          *iter0 -= half_one_plus_i * *iter1;
          *iter1 *= half_one_plus_i;
          *iter1 += half_one_plus_i * value0;
        },
        target_qubit, control_qubit1, control_qubit2, control_qubits...);
# endif // KET_USE_BIT_MASKS_EXPLICITLY
    }

    template <typename RandomAccessIterator, typename StateInteger, typename BitInteger, typename... ControlQubits>
    inline auto sqrt_pauli_y(
      RandomAccessIterator const first, RandomAccessIterator const last,
      ::ket::qubit<StateInteger, BitInteger> const qubit, ControlQubits const... control_qubits)
    -> void
    { ::ket::gate::sqrt_pauli_y(::ket::utility::policy::make_sequential(), first, last, qubit, control_qubits...); }

    namespace ranges
    {
      template <typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename... ControlQubits>
      inline auto sqrt_pauli_y(
        ParallelPolicy const parallel_policy, RandomAccessRange& state,
        ::ket::qubit<StateInteger, BitInteger> const qubit, ControlQubits const... control_qubits)
      -> std::enable_if_t< ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value, RandomAccessRange& >
      {
        using std::begin;
        using std::end;
        ::ket::gate::sqrt_pauli_y(parallel_policy, begin(state), end(state), qubit, control_qubits...);
        return state;
      }

      template <typename RandomAccessRange, typename StateInteger, typename BitInteger, typename... ControlQubits>
      inline auto sqrt_pauli_y(RandomAccessRange& state, ::ket::qubit<StateInteger, BitInteger> const qubit, ControlQubits const... control_qubits) -> RandomAccessRange&
      { return ::ket::gate::ranges::sqrt_pauli_y(::ket::utility::policy::make_sequential(), state, qubit, control_qubits...); }
    } // namespace ranges

    // sY+_i
    // sY+_1 (a_0 |0> + a_1 |1>) = [(1-i) a_0 + (1-i) a_1]/2 |0> + [-(1-i) a_0 + (1-i) a_1]/2 |1>
    template <typename ParallelPolicy, typename RandomAccessIterator, typename StateInteger, typename BitInteger>
    inline auto adj_sqrt_pauli_y(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      ::ket::qubit<StateInteger, BitInteger> const qubit)
    -> void
    {
      static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
      static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");
      assert(::ket::utility::integer_exp2<StateInteger>(qubit) < static_cast<StateInteger>(last - first));
      assert(
        ::ket::utility::integer_exp2<StateInteger>(::ket::utility::integer_log2<BitInteger>(last - first))
        == static_cast<StateInteger>(last - first));

      auto const qubit_mask = ::ket::utility::integer_exp2<StateInteger>(qubit);
      auto const lower_bits_mask = qubit_mask - StateInteger{1u};
      auto const upper_bits_mask = compl lower_bits_mask;

      ::ket::utility::loop_n(
        parallel_policy, static_cast<StateInteger>(last - first) >> 1u,
        [first, qubit_mask, lower_bits_mask, upper_bits_mask](StateInteger const value_wo_qubit, int const)
        {
          // xxxxx0xxxxxx
          auto const zero_index = ((value_wo_qubit bitand upper_bits_mask) << 1u) bitor (value_wo_qubit bitand lower_bits_mask);
          // xxxxx1xxxxxx
          auto const one_index = zero_index bitor qubit_mask;
          auto const zero_iter = first + zero_index;
          auto const one_iter = first + one_index;
          auto const zero_iter_value = *zero_iter;

          using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
          using real_type = ::ket::utility::meta::real_t<complex_type>;
          using boost::math::constants::half;
          constexpr auto half_one_minus_i = complex_type{half<real_type>(), -half<real_type>()};
          *zero_iter *= half_one_minus_i;
          *zero_iter += half_one_minus_i * *one_iter;
          *one_iter *= half_one_minus_i;
          *one_iter -= half_one_minus_i * zero_iter_value;
        });
    }

    // CsY+_{tc} or C1sY+_{tc}
    // CsY+_{1,2} (a_{00} |00> + a_{01} |01> + a_{10} |10> + a_{11} |11>)
    //   = a_{00} |00> + a_{01} |01> + [(1-i) a_{10} + (1-i) a_{11}]/2 |10> + [-(1-i) a_{10} + (1-i) a_{11}]/2 |11>
    template <typename ParallelPolicy, typename RandomAccessIterator, typename StateInteger, typename BitInteger>
    inline auto adj_sqrt_pauli_y(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit)
    -> void
    {
      static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
      static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");

      assert(::ket::utility::integer_exp2<StateInteger>(target_qubit) < static_cast<StateInteger>(last - first));
      assert(::ket::utility::integer_exp2<StateInteger>(control_qubit) < static_cast<StateInteger>(last - first));
      assert(target_qubit != control_qubit);
      assert(
        ::ket::utility::integer_exp2<StateInteger>(::ket::utility::integer_log2<BitInteger>(last - first))
        == static_cast<StateInteger>(last - first));

      auto const minmax_qubits = std::minmax(target_qubit, control_qubit.qubit());
      auto const target_qubit_mask = ::ket::utility::integer_exp2<StateInteger>(target_qubit);
      auto const control_qubit_mask = ::ket::utility::integer_exp2<StateInteger>(control_qubit);
      auto const lower_bits_mask = ::ket::utility::integer_exp2<StateInteger>(minmax_qubits.first) - StateInteger{1u};
      auto const middle_bits_mask
        = (::ket::utility::integer_exp2<StateInteger>(minmax_qubits.second - BitInteger{1u}) - StateInteger{1u})
          xor lower_bits_mask;
      auto const upper_bits_mask = compl (lower_bits_mask bitor middle_bits_mask);

      ::ket::utility::loop_n(
        parallel_policy, static_cast<StateInteger>(last - first) >> 2u,
        [first, target_qubit_mask, control_qubit_mask, lower_bits_mask, middle_bits_mask, upper_bits_mask](
          StateInteger const value_wo_qubits, int const)
        {
          // xxx0_txxx0_cxxx
          auto const base_index
            = ((value_wo_qubits bitand upper_bits_mask) << 2u)
              bitor ((value_wo_qubits bitand middle_bits_mask) << 1u)
              bitor (value_wo_qubits bitand lower_bits_mask);
          // xxx0_txxx1_cxxx
          auto const control_on_index = base_index bitor control_qubit_mask;
          // xxx1_txxx1_cxxx
          auto const target_control_on_index = control_on_index bitor target_qubit_mask;
          auto const control_on_iter = first + control_on_index;
          auto const target_control_on_iter = first + target_control_on_index;
          auto const control_on_iter_value = *control_on_iter;

          using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
          using real_type = ::ket::utility::meta::real_t<complex_type>;
          using boost::math::constants::half;
          constexpr auto half_one_minus_i = complex_type{half<real_type>(), -half<real_type>()};
          *control_on_iter *= half_one_minus_i;
          *control_on_iter += half_one_minus_i * *target_control_on_iter;
          *target_control_on_iter *= half_one_minus_i;
          *target_control_on_iter -= half_one_minus_i * control_on_iter_value;
        });
    }

    // C...CsY+_{tc...c'} or CnsY+_{tc...c'}
    template <typename ParallelPolicy, typename RandomAccessIterator, typename StateInteger, typename BitInteger, typename... ControlQubits>
    inline auto adj_sqrt_pauli_y(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2, ControlQubits const... control_qubits)
    -> void
    {
      static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
      static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");
      assert(::ket::utility::integer_exp2<StateInteger>(target_qubit) < static_cast<StateInteger>(last - first));
      assert(
        ::ket::utility::integer_exp2<StateInteger>(::ket::utility::integer_log2<BitInteger>(last - first))
        == static_cast<StateInteger>(last - first));

# ifndef KET_USE_BIT_MASKS_EXPLICITLY
      using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
      constexpr auto num_control_qubits = static_cast<BitInteger>(sizeof...(ControlQubits) + 2u);
      constexpr auto num_operated_qubits = num_control_qubits + BitInteger{1u};

      ::ket::gate::nocache::gate(
        parallel_policy, first, last,
        [](
          auto const first, StateInteger const index_wo_qubits,
          std::array<qubit_type, num_operated_qubits> const& unsorted_qubits,
          std::array<qubit_type, num_operated_qubits + BitInteger{1u}> const& sorted_qubits_with_sentinel,
          int const)
        {
          // 0b11...10u
          constexpr auto index0 = ((std::size_t{1u} << num_control_qubits) - std::size_t{1u}) << BitInteger{1u};
          // 0b11...11u
          constexpr auto index1 = index0 bitor std::size_t{1u};

          using std::begin;
          using std::end;
          auto const iter0
            = first
              + ::ket::gate::utility::ranges::index_with_qubits(index_wo_qubits, index0, unsorted_qubits, sorted_qubits_with_sentinel);
          auto const iter1
            = first
              + ::ket::gate::utility::ranges::index_with_qubits(index_wo_qubits, index1, unsorted_qubits, sorted_qubits_with_sentinel);
          auto const value0 = *iter0;

          using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
          using real_type = ::ket::utility::meta::real_t<complex_type>;
          using boost::math::constants::half;
          constexpr auto half_one_minus_i = complex_type{half<real_type>(), -half<real_type>()};
          *iter0 *= half_one_minus_i;
          *iter0 += half_one_minus_i * *iter1;
          *iter1 *= half_one_minus_i;
          *iter1 -= half_one_minus_i * value0;
        },
        target_qubit, control_qubit1, control_qubit2, control_qubits...);
# else // KET_USE_BIT_MASKS_EXPLICITLY
      constexpr auto num_control_qubits = static_cast<BitInteger>(sizeof...(ControlQubits) + 2u);
      constexpr auto num_operated_qubits = num_control_qubits + BitInteger{1u};

      ::ket::gate::nocache::gate(
        parallel_policy, first, last,
        [](
          auto const first, StateInteger const index_wo_qubits,
          std::array<StateInteger, num_operated_qubits> const& qubit_masks,
          std::array<StateInteger, num_operated_qubits + 1u> const& index_masks,
          int const)
        {
          // 0b11...10u
          constexpr auto index0 = ((std::size_t{1u} << num_control_qubits) - std::size_t{1u}) << BitInteger{1u};
          // 0b11...11u
          constexpr auto index1 = index0 bitor std::size_t{1u};

          using std::begin;
          using std::end;
          auto const iter0
            = first
              + ::ket::gate::utility::ranges::index_with_qubits(index_wo_qubits, index0, qubit_masks, index_masks);
          auto const iter1
            = first
              + ::ket::gate::utility::ranges::index_with_qubits(index_wo_qubits, index1, qubit_masks, index_masks);
          auto const value0 = *iter0;

          using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
          using real_type = ::ket::utility::meta::real_t<complex_type>;
          using boost::math::constants::half;
          constexpr auto half_one_minus_i = complex_type{half<real_type>(), -half<real_type>()};
          *iter0 *= half_one_minus_i;
          *iter0 += half_one_minus_i * *iter1;
          *iter1 *= half_one_minus_i;
          *iter1 -= half_one_minus_i * value0;
        },
        target_qubit, control_qubit1, control_qubit2, control_qubits...);
# endif // KET_USE_BIT_MASKS_EXPLICITLY
    }

    template <typename RandomAccessIterator, typename StateInteger, typename BitInteger, typename... ControlQubits>
    inline auto adj_sqrt_pauli_y(
      RandomAccessIterator const first, RandomAccessIterator const last,
      ::ket::qubit<StateInteger, BitInteger> const qubit, ControlQubits const... control_qubits)
    -> void
    { ::ket::gate::adj_sqrt_pauli_y(::ket::utility::policy::make_sequential(), first, last, qubit, control_qubits...); }

    namespace ranges
    {
      template <typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename... ControlQubits>
      inline auto adj_sqrt_pauli_y(
        ParallelPolicy const parallel_policy, RandomAccessRange& state,
        ::ket::qubit<StateInteger, BitInteger> const qubit, ControlQubits const... control_qubits)
      -> std::enable_if_t< ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value, RandomAccessRange& >
      {
        using std::begin;
        using std::end;
        ::ket::gate::adj_sqrt_pauli_y(parallel_policy, begin(state), end(state), qubit, control_qubits...);
        return state;
      }

      template <typename RandomAccessRange, typename StateInteger, typename BitInteger, typename... ControlQubits>
      inline auto adj_sqrt_pauli_y(RandomAccessRange& state, ::ket::qubit<StateInteger, BitInteger> const qubit, ControlQubits const... control_qubits) -> RandomAccessRange&
      { return ::ket::gate::ranges::adj_sqrt_pauli_y(::ket::utility::policy::make_sequential(), state, qubit, control_qubits...); }
    } // namespace ranges


    namespace runtime
    {
      // C...CsY_{tc...c'} or CnsY_{tc...c'}
      namespace qubit_ranges
      {
        template <typename ParallelPolicy, typename RandomAccessIterator, typename StateInteger, typename BitInteger, typename ControlQubitsRange>
        inline auto sqrt_pauli_y(
          ParallelPolicy const parallel_policy,
          RandomAccessIterator const first, RandomAccessIterator const last,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ControlQubitsRange const& control_qubits)
        -> void
        {
          using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
          using control_qubit_type = ::ket::control<qubit_type>;
          static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
          static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");
          static_assert(std::is_same< ::ket::utility::meta::range_value_t<ControlQubitsRange>, control_qubit_type >::value, "The value_type of ControlQubitsRange should be the same as ket::control<ket::qubit<S, B> >");

          assert(::ket::utility::integer_exp2<StateInteger>(target_qubit) < static_cast<StateInteger>(last - first));
          assert(
            ::ket::utility::integer_exp2<StateInteger>(::ket::utility::integer_log2<BitInteger>(last - first))
            == static_cast<StateInteger>(last - first));
          using std::begin;
          using std::end;
          auto const num_control_qubits = static_cast<BitInteger>(end(control_qubits) - begin(control_qubits));

# ifndef KET_USE_BIT_MASKS_EXPLICITLY
          ::ket::gate::runtime::nocache::qubit_ranges::gate(
            parallel_policy, first, last,
            [num_control_qubits](
              auto const first, StateInteger const index_wo_qubits,
              auto const& unsorted_qubits, auto const& sorted_qubits_with_sentinel,
              int const)
            {
              // 0b11...10u
              auto const index0 = ((std::size_t{1u} << num_control_qubits) - std::size_t{1u}) << BitInteger{1u};
              // 0b11...11u
              auto const index1 = index0 bitor std::size_t{1u};

              auto const iter0
                = first
                  + ::ket::gate::utility::ranges::index_with_qubits(index_wo_qubits, index0, unsorted_qubits, sorted_qubits_with_sentinel);
              auto const iter1
                = first
                  + ::ket::gate::utility::ranges::index_with_qubits(index_wo_qubits, index1, unsorted_qubits, sorted_qubits_with_sentinel);
              auto const value0 = *iter0;

              using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
              using real_type = ::ket::utility::meta::real_t<complex_type>;
              using boost::math::constants::half;
              constexpr auto half_one_plus_i = complex_type{half<real_type>(), half<real_type>()};
              *iter0 *= half_one_plus_i;
              *iter0 -= half_one_plus_i * *iter1;
              *iter1 *= half_one_plus_i;
              *iter1 += half_one_plus_i * value0;
            },
            boost::join(
              boost::make_iterator_range(std::addressof(target_qubit), std::next(std::addressof(target_qubit))),
              control_qubits | boost::adaptors::transformed(
                [](control_qubit_type const control_qubit) { return control_qubit.qubit(); })));
# else // KET_USE_BIT_MASKS_EXPLICITLY
          ::ket::gate::runtime::nocache::qubit_ranges::gate(
            parallel_policy, first, last,
            [num_control_qubits](
              auto const first, StateInteger const index_wo_qubits,
              auto const& qubit_masks, auto const& index_masks,
              int const)
            {
              // 0b11...10u
              auto const index0 = ((std::size_t{1u} << num_control_qubits) - std::size_t{1u}) << BitInteger{1u};
              // 0b11...11u
              auto const index1 = index0 bitor std::size_t{1u};

              auto const iter0
                = first
                  + ::ket::gate::utility::ranges::index_with_qubits(index_wo_qubits, index0, qubit_masks, index_masks);
              auto const iter1
                = first
                  + ::ket::gate::utility::ranges::index_with_qubits(index_wo_qubits, index1, qubit_masks, index_masks);
              auto const value0 = *iter0;

              using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
              using real_type = ::ket::utility::meta::real_t<complex_type>;
              using boost::math::constants::half;
              constexpr auto half_one_plus_i = complex_type{half<real_type>(), half<real_type>()};
              *iter0 *= half_one_plus_i;
              *iter0 -= half_one_plus_i * *iter1;
              *iter1 *= half_one_plus_i;
              *iter1 += half_one_plus_i * value0;
            },
            boost::join(
              boost::make_iterator_range(std::addressof(target_qubit), std::next(std::addressof(target_qubit))),
              control_qubits | boost::adaptors::transformed(
                [](control_qubit_type const control_qubit) { return control_qubit.qubit(); })));
# endif // KET_USE_BIT_MASKS_EXPLICITLY
        }

        template <typename ParallelPolicy, typename RandomAccessIterator, typename StateInteger, typename BitInteger>
        inline auto sqrt_pauli_y(
          ParallelPolicy const parallel_policy,
          RandomAccessIterator const first, RandomAccessIterator const last,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit)
        -> void
        {
          using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
          using control_qubit_type = ::ket::control<qubit_type>;
          std::array<control_qubit_type, 0u> const control_qubits{};
          ::ket::gate::runtime::qubit_ranges::sqrt_pauli_y(parallel_policy, first, last, target_qubit, control_qubits);
        }

        template <typename RandomAccessIterator, typename StateInteger, typename BitInteger, typename ControlQubitsRange>
        inline auto sqrt_pauli_y(
          RandomAccessIterator const first, RandomAccessIterator const last,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ControlQubitsRange const& control_qubits)
        -> void
        { ::ket::gate::runtime::qubit_ranges::sqrt_pauli_y(::ket::utility::policy::make_sequential(), first, last, target_qubit, control_qubits); }

        template <typename RandomAccessIterator, typename StateInteger, typename BitInteger>
        inline auto sqrt_pauli_y(
          RandomAccessIterator const first, RandomAccessIterator const last,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit)
        -> void
        { ::ket::gate::runtime::qubit_ranges::sqrt_pauli_y(::ket::utility::policy::make_sequential(), first, last, target_qubit); }
      } // namespace qubit_ranges

      template <typename ParallelPolicy, typename RandomAccessIterator, typename StateInteger, typename BitInteger, typename ControlQubitIterator>
      inline auto sqrt_pauli_y(
        ParallelPolicy const parallel_policy,
        RandomAccessIterator const first, RandomAccessIterator const last,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ControlQubitIterator const control_qubit_first, ControlQubitIterator const control_qubit_last)
      -> void
      {
        ::ket::gate::runtime::qubit_ranges::sqrt_pauli_y(
          parallel_policy, first, last,
          target_qubit, boost::make_iterator_range(control_qubit_first, control_qubit_last));
      }

      template <typename ParallelPolicy, typename RandomAccessIterator, typename StateInteger, typename BitInteger>
      inline auto sqrt_pauli_y(
        ParallelPolicy const parallel_policy,
        RandomAccessIterator const first, RandomAccessIterator const last,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit)
      -> void
      { ::ket::gate::runtime::qubit_ranges::sqrt_pauli_y(parallel_policy, first, last, target_qubit); }

      template <typename RandomAccessIterator, typename StateInteger, typename BitInteger, typename ControlQubitIterator>
      inline auto sqrt_pauli_y(
        RandomAccessIterator const first, RandomAccessIterator const last,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ControlQubitIterator const control_qubit_first, ControlQubitIterator const control_qubit_last)
      -> void
      {
        ::ket::gate::runtime::qubit_ranges::sqrt_pauli_y(
          first, last,
          target_qubit, boost::make_iterator_range(control_qubit_first, control_qubit_last));
      }

      template <typename RandomAccessIterator, typename StateInteger, typename BitInteger>
      inline auto sqrt_pauli_y(
        RandomAccessIterator const first, RandomAccessIterator const last,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit)
      -> void
      { ::ket::gate::runtime::qubit_ranges::sqrt_pauli_y(first, last, target_qubit); }

      namespace ranges
      {
        template <typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename ControlQubitsRange>
        inline auto sqrt_pauli_y(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& state,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ControlQubitsRange const& control_qubits)
        -> RandomAccessRange&
        {
          using std::begin;
          using std::end;
          ::ket::gate::runtime::qubit_ranges::sqrt_pauli_y(parallel_policy, begin(state), end(state), target_qubit, control_qubits);
          return state;
        }

        template <typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger>
        inline auto sqrt_pauli_y(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& state,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit)
        -> RandomAccessRange&
        {
          using std::begin;
          using std::end;
          ::ket::gate::runtime::qubit_ranges::sqrt_pauli_y(parallel_policy, begin(state), end(state), target_qubit);
          return state;
        }

        template <typename RandomAccessRange, typename StateInteger, typename BitInteger, typename ControlQubitsRange>
        inline auto sqrt_pauli_y(
          RandomAccessRange& state,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ControlQubitsRange const& control_qubits)
        -> RandomAccessRange&
        {
          using std::begin;
          using std::end;
          ::ket::gate::runtime::qubit_ranges::sqrt_pauli_y(begin(state), end(state), target_qubit, control_qubits);
          return state;
        }

        template <typename RandomAccessRange, typename StateInteger, typename BitInteger>
        inline auto sqrt_pauli_y(
          RandomAccessRange& state,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit)
        -> RandomAccessRange&
        {
          using std::begin;
          using std::end;
          ::ket::gate::runtime::qubit_ranges::sqrt_pauli_y(begin(state), end(state), target_qubit);
          return state;
        }
      } // namespace ranges

      // C...CsY+_{tc...c'} or CnsY+_{tc...c'}
      namespace qubit_ranges
      {
        template <typename ParallelPolicy, typename RandomAccessIterator, typename StateInteger, typename BitInteger, typename ControlQubitsRange>
        inline auto adj_sqrt_pauli_y(
          ParallelPolicy const parallel_policy,
          RandomAccessIterator const first, RandomAccessIterator const last,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ControlQubitsRange const& control_qubits)
        -> void
        {
          using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
          using control_qubit_type = ::ket::control<qubit_type>;
          static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
          static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");
          static_assert(std::is_same< ::ket::utility::meta::range_value_t<ControlQubitsRange>, control_qubit_type >::value, "The value_type of ControlQubitsRange should be the same as ket::control<ket::qubit<S, B> >");

          assert(::ket::utility::integer_exp2<StateInteger>(target_qubit) < static_cast<StateInteger>(last - first));
          assert(
            ::ket::utility::integer_exp2<StateInteger>(::ket::utility::integer_log2<BitInteger>(last - first))
            == static_cast<StateInteger>(last - first));
          using std::begin;
          using std::end;
          auto const num_control_qubits = static_cast<BitInteger>(end(control_qubits) - begin(control_qubits));

# ifndef KET_USE_BIT_MASKS_EXPLICITLY
          ::ket::gate::runtime::nocache::qubit_ranges::gate(
            parallel_policy, first, last,
            [num_control_qubits](
              auto const first, StateInteger const index_wo_qubits,
              auto const& unsorted_qubits, auto const& sorted_qubits_with_sentinel,
              int const)
            {
              // 0b11...10u
              auto const index0 = ((std::size_t{1u} << num_control_qubits) - std::size_t{1u}) << BitInteger{1u};
              // 0b11...11u
              auto const index1 = index0 bitor std::size_t{1u};

              auto const iter0
                = first
                  + ::ket::gate::utility::ranges::index_with_qubits(index_wo_qubits, index0, unsorted_qubits, sorted_qubits_with_sentinel);
              auto const iter1
                = first
                  + ::ket::gate::utility::ranges::index_with_qubits(index_wo_qubits, index1, unsorted_qubits, sorted_qubits_with_sentinel);
              auto const value0 = *iter0;

              using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
              using real_type = ::ket::utility::meta::real_t<complex_type>;
              using boost::math::constants::half;
              constexpr auto half_one_minus_i = complex_type{half<real_type>(), -half<real_type>()};
              *iter0 *= half_one_minus_i;
              *iter0 += half_one_minus_i * *iter1;
              *iter1 *= half_one_minus_i;
              *iter1 -= half_one_minus_i * value0;
            },
            boost::join(
              boost::make_iterator_range(std::addressof(target_qubit), std::next(std::addressof(target_qubit))),
              control_qubits | boost::adaptors::transformed(
                [](control_qubit_type const control_qubit) { return control_qubit.qubit(); })));
# else // KET_USE_BIT_MASKS_EXPLICITLY
          ::ket::gate::runtime::nocache::qubit_ranges::gate(
            parallel_policy, first, last,
            [num_control_qubits](
              auto const first, StateInteger const index_wo_qubits,
              auto const& qubit_masks, auto const& index_masks,
              int const)
            {
              // 0b11...10u
              auto const index0 = ((std::size_t{1u} << num_control_qubits) - std::size_t{1u}) << BitInteger{1u};
              // 0b11...11u
              auto const index1 = index0 bitor std::size_t{1u};

              auto const iter0
                = first
                  + ::ket::gate::utility::ranges::index_with_qubits(index_wo_qubits, index0, qubit_masks, index_masks);
              auto const iter1
                = first
                  + ::ket::gate::utility::ranges::index_with_qubits(index_wo_qubits, index1, qubit_masks, index_masks);
              auto const value0 = *iter0;

              using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
              using real_type = ::ket::utility::meta::real_t<complex_type>;
              using boost::math::constants::half;
              constexpr auto half_one_minus_i = complex_type{half<real_type>(), -half<real_type>()};
              *iter0 *= half_one_minus_i;
              *iter0 += half_one_minus_i * *iter1;
              *iter1 *= half_one_minus_i;
              *iter1 -= half_one_minus_i * value0;
            },
            boost::join(
              boost::make_iterator_range(std::addressof(target_qubit), std::next(std::addressof(target_qubit))),
              control_qubits | boost::adaptors::transformed(
                [](control_qubit_type const control_qubit) { return control_qubit.qubit(); })));
# endif // KET_USE_BIT_MASKS_EXPLICITLY
        }

        template <typename ParallelPolicy, typename RandomAccessIterator, typename StateInteger, typename BitInteger>
        inline auto adj_sqrt_pauli_y(
          ParallelPolicy const parallel_policy,
          RandomAccessIterator const first, RandomAccessIterator const last,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit)
        -> void
        {
          using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
          using control_qubit_type = ::ket::control<qubit_type>;
          std::array<control_qubit_type, 0u> const control_qubits{};
          ::ket::gate::runtime::qubit_ranges::adj_sqrt_pauli_y(parallel_policy, first, last, target_qubit, control_qubits);
        }

        template <typename RandomAccessIterator, typename StateInteger, typename BitInteger, typename ControlQubitsRange>
        inline auto adj_sqrt_pauli_y(
          RandomAccessIterator const first, RandomAccessIterator const last,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ControlQubitsRange const& control_qubits)
        -> void
        { ::ket::gate::runtime::qubit_ranges::adj_sqrt_pauli_y(::ket::utility::policy::make_sequential(), first, last, target_qubit, control_qubits); }

        template <typename RandomAccessIterator, typename StateInteger, typename BitInteger>
        inline auto adj_sqrt_pauli_y(
          RandomAccessIterator const first, RandomAccessIterator const last,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit)
        -> void
        { ::ket::gate::runtime::qubit_ranges::adj_sqrt_pauli_y(::ket::utility::policy::make_sequential(), first, last, target_qubit); }
      } // namespace qubit_ranges

      template <typename ParallelPolicy, typename RandomAccessIterator, typename StateInteger, typename BitInteger, typename ControlQubitIterator>
      inline auto adj_sqrt_pauli_y(
        ParallelPolicy const parallel_policy,
        RandomAccessIterator const first, RandomAccessIterator const last,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ControlQubitIterator const control_qubit_first, ControlQubitIterator const control_qubit_last)
      -> void
      {
        ::ket::gate::runtime::qubit_ranges::adj_sqrt_pauli_y(
          parallel_policy, first, last,
          target_qubit, boost::make_iterator_range(control_qubit_first, control_qubit_last));
      }

      template <typename ParallelPolicy, typename RandomAccessIterator, typename StateInteger, typename BitInteger>
      inline auto adj_sqrt_pauli_y(
        ParallelPolicy const parallel_policy,
        RandomAccessIterator const first, RandomAccessIterator const last,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit)
      -> void
      { ::ket::gate::runtime::qubit_ranges::adj_sqrt_pauli_y(parallel_policy, first, last, target_qubit); }

      template <typename RandomAccessIterator, typename StateInteger, typename BitInteger, typename ControlQubitIterator>
      inline auto adj_sqrt_pauli_y(
        RandomAccessIterator const first, RandomAccessIterator const last,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ControlQubitIterator const control_qubit_first, ControlQubitIterator const control_qubit_last)
      -> void
      {
        ::ket::gate::runtime::qubit_ranges::adj_sqrt_pauli_y(
          first, last,
          target_qubit, boost::make_iterator_range(control_qubit_first, control_qubit_last));
      }

      template <typename RandomAccessIterator, typename StateInteger, typename BitInteger>
      inline auto adj_sqrt_pauli_y(
        RandomAccessIterator const first, RandomAccessIterator const last,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit)
      -> void
      { ::ket::gate::runtime::qubit_ranges::adj_sqrt_pauli_y(first, last, target_qubit); }

      namespace ranges
      {
        template <typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename ControlQubitsRange>
        inline auto adj_sqrt_pauli_y(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& state,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ControlQubitsRange const& control_qubits)
        -> RandomAccessRange&
        {
          using std::begin;
          using std::end;
          ::ket::gate::runtime::qubit_ranges::adj_sqrt_pauli_y(parallel_policy, begin(state), end(state), target_qubit, control_qubits);
          return state;
        }

        template <typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger>
        inline auto adj_sqrt_pauli_y(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& state,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit)
        -> RandomAccessRange&
        {
          using std::begin;
          using std::end;
          ::ket::gate::runtime::qubit_ranges::adj_sqrt_pauli_y(parallel_policy, begin(state), end(state), target_qubit);
          return state;
        }

        template <typename RandomAccessRange, typename StateInteger, typename BitInteger, typename ControlQubitsRange>
        inline auto adj_sqrt_pauli_y(
          RandomAccessRange& state,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ControlQubitsRange const& control_qubits)
        -> RandomAccessRange&
        {
          using std::begin;
          using std::end;
          ::ket::gate::runtime::qubit_ranges::adj_sqrt_pauli_y(begin(state), end(state), target_qubit, control_qubits);
          return state;
        }

        template <typename RandomAccessRange, typename StateInteger, typename BitInteger>
        inline auto adj_sqrt_pauli_y(RandomAccessRange& state, ::ket::qubit<StateInteger, BitInteger> const target_qubit) -> RandomAccessRange&
        {
          using std::begin;
          using std::end;
          ::ket::gate::runtime::qubit_ranges::adj_sqrt_pauli_y(begin(state), end(state), target_qubit);
          return state;
        }
      } // namespace ranges
    } // namespace runtime
  } // namespace gate
} // namespace ket


#endif // KET_GATE_SQRT_PAULI_Y_HPP
