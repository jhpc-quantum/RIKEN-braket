#ifndef KET_GATE_PHASE_SHIFT_HPP
# define KET_GATE_PHASE_SHIFT_HPP

# include <cassert>
# include <cstddef>
# include <cstdint>
# include <cmath>
# include <array>
# include <algorithm>
# include <iterator>
# include <utility>
# include <type_traits>

# include <boost/math/constants/constants.hpp>

# include <ket/qubit.hpp>
# include <ket/control.hpp>
# include <ket/gate/gate.hpp>
# include <ket/gate/utility/index_with_qubits.hpp>
# include <ket/utility/loop_n.hpp>
# include <ket/utility/integer_exp2.hpp>
# ifndef NDEBUG
#   include <ket/utility/integer_log2.hpp>
# endif
# include <ket/utility/exp_i.hpp>
# include <ket/utility/meta/real_of.hpp>
# include <ket/utility/meta/ranges.hpp>


namespace ket
{
  namespace gate
  {
    // phase_shift_coeff
    // Case 1: the first argument of qubits is ket::control<ket::qubit<S, B>>
    template <typename ParallelPolicy, typename RandomAccessIterator, typename Complex>
    inline auto phase_shift_coeff(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      Complex const& phase_coefficient) // exp(i theta) = cos(theta) + i sin(theta)
    -> void
    {
      static_assert(
        std::is_same<Complex, typename std::iterator_traits<RandomAccessIterator>::value_type>::value,
        "Complex must be the same to value_type of RandomAccessIterator");

      assert(
        ::ket::utility::integer_exp2<std::uint64_t>(::ket::utility::integer_log2<std::size_t>(last - first))
        == static_cast<std::uint64_t>(last - first));

      ::ket::utility::loop_n(
        parallel_policy, static_cast<std::uint64_t>(last - first),
        [first, &phase_coefficient](std::uint64_t const index, int const)
        { *(first + index) *= phase_coefficient; });
    }

    // U1_i(theta)
    // U1_1(theta) (a_0 |0> + a_1 |1>) = a_0 |0> + e^{i theta} a_1 |1>
    template <typename ParallelPolicy, typename RandomAccessIterator, typename Complex, typename StateInteger, typename BitInteger>
    inline auto phase_shift_coeff(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit)
    -> void
    {
      static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
      static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");
      static_assert(
        std::is_same<Complex, typename std::iterator_traits<RandomAccessIterator>::value_type>::value,
        "Complex must be the same to value_type of RandomAccessIterator");

      assert(::ket::utility::integer_exp2<StateInteger>(control_qubit) < static_cast<StateInteger>(last - first));
      assert(
        ::ket::utility::integer_exp2<StateInteger>(::ket::utility::integer_log2<BitInteger>(last - first))
        == static_cast<StateInteger>(last - first));

      auto const qubit_mask = ::ket::utility::integer_exp2<StateInteger>(control_qubit);
      auto const lower_bits_mask = qubit_mask - StateInteger{1u};
      auto const upper_bits_mask = compl lower_bits_mask;

      ::ket::utility::loop_n(
        parallel_policy, static_cast<StateInteger>(last - first) >> 1u,
        [first, &phase_coefficient, qubit_mask, lower_bits_mask, upper_bits_mask](StateInteger const value_wo_qubit, int const)
        {
          // xxxxx1xxxxxx
          auto const one_index = ((value_wo_qubit bitand upper_bits_mask) << 1u) bitor (value_wo_qubit bitand lower_bits_mask) bitor qubit_mask;
          *(first + one_index) *= phase_coefficient;
        });
    }

    // CU1_{cc'}(theta) or C1U1_{cc'}(theta)
    // CU1_{1,2}(theta) (a_{00} |00> + a_{01} |01> + a_{10} |10> + a{11} |11>)
    //   = a_{00} |00> + a_{01} |01> + a_{10} |10> + e^{i thta} a_{11} |11>
    template <typename ParallelPolicy, typename RandomAccessIterator, typename Complex, typename StateInteger, typename BitInteger>
    inline auto phase_shift_coeff(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2)
    -> void
    {
      static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
      static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");
      static_assert(
        std::is_same<Complex, typename std::iterator_traits<RandomAccessIterator>::value_type>::value,
        "Complex must be the same to value_type of RandomAccessIterator");

      assert(::ket::utility::integer_exp2<StateInteger>(control_qubit1) < static_cast<StateInteger>(last - first));
      assert(::ket::utility::integer_exp2<StateInteger>(control_qubit2) < static_cast<StateInteger>(last - first));
      assert(control_qubit1 != control_qubit2);
      assert(
        ::ket::utility::integer_exp2<StateInteger>(::ket::utility::integer_log2<BitInteger>(last - first))
        == static_cast<StateInteger>(last - first));

      auto const minmax_qubits = std::minmax(control_qubit1, control_qubit2);
      auto const control_qubit1_mask = ::ket::utility::integer_exp2<StateInteger>(control_qubit1);
      auto const control_qubit2_mask = ::ket::utility::integer_exp2<StateInteger>(control_qubit2);
      auto const lower_bits_mask = ::ket::utility::integer_exp2<StateInteger>(minmax_qubits.first) - StateInteger{1u};
      auto const middle_bits_mask
        = (::ket::utility::integer_exp2<StateInteger>(minmax_qubits.second - BitInteger{1u}) - StateInteger{1u})
          xor lower_bits_mask;
      auto const upper_bits_mask = compl (lower_bits_mask bitor middle_bits_mask);

      ::ket::utility::loop_n(
        parallel_policy, static_cast<StateInteger>(last - first) >> 2u,
        [first, &phase_coefficient, control_qubit1_mask, control_qubit2_mask, lower_bits_mask, middle_bits_mask, upper_bits_mask](
          StateInteger const value_wo_qubits, int const)
        {
          // xxx1_c1xxx1_c2xxx
          auto const index11
            = ((value_wo_qubits bitand upper_bits_mask) << 2u)
              bitor ((value_wo_qubits bitand middle_bits_mask) << 1u)
              bitor (value_wo_qubits bitand lower_bits_mask)
              bitor control_qubit1_mask bitor control_qubit2_mask;
          *(first + index11) *= phase_coefficient;
        });
    }

    // C...CU1_{c0,c...c'}(theta) or CnU1_{c0,c...c'}(theta)
    template <typename ParallelPolicy, typename RandomAccessIterator, typename Complex, typename StateInteger, typename BitInteger, typename... Qubits>
    inline auto phase_shift_coeff(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit3,
      ::ket::control<Qubits> const... control_qubits)
    -> void
    {
      static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
      static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");
      static_assert(
        (std::is_same<Complex, typename std::iterator_traits<RandomAccessIterator>::value_type>::value),
        "Complex must be the same to value_type of RandomAccessIterator");
      assert(
        ::ket::utility::integer_exp2<StateInteger>(::ket::utility::integer_log2<BitInteger>(last - first))
        == static_cast<StateInteger>(last - first));

# ifndef KET_USE_BIT_MASKS_EXPLICITLY
      using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
      constexpr auto num_control_qubits = BitInteger{sizeof...(Qubits) + 3u};
      constexpr auto num_operated_qubits = num_control_qubits;

      ::ket::gate::nocache::gate(
        parallel_policy, first, last,
        [&phase_coefficient](
          auto const first, StateInteger const index_wo_qubits,
          std::array<qubit_type, num_operated_qubits> const& unsorted_qubits,
          std::array<qubit_type, num_operated_qubits + BitInteger{1u}> const& sorted_qubits_with_sentinel,
          int const)
        {
          // 0b11...11u
          constexpr auto index = ((std::size_t{1u} << num_operated_qubits) - std::size_t{1u});

          using std::begin;
          using std::end;
          auto const iter
            = first
              + ::ket::gate::utility::index_with_qubits(
                  index_wo_qubits, index,
                  begin(unsorted_qubits), end(unsorted_qubits),
                  begin(sorted_qubits_with_sentinel), end(sorted_qubits_with_sentinel));
          *iter *= phase_coefficient;
        },
        control_qubit1, control_qubit2, control_qubit3, control_qubits...);
# else // KET_USE_BIT_MASKS_EXPLICITLY
      constexpr auto num_control_qubits = BitInteger{sizeof...(Qubits) + 3u};
      constexpr auto num_operated_qubits = num_control_qubits;

      ::ket::gate::nocache::gate(
        parallel_policy, first, last,
        [&phase_coefficient](
          auto const first, StateInteger const index_wo_qubits,
          std::array<StateInteger, num_operated_qubits> const& qubit_masks,
          std::array<StateInteger, num_operated_qubits + 1u> const& index_masks,
          int const)
        {
          // 0b11...11u
          constexpr auto index = ((std::size_t{1u} << num_operated_qubits) - std::size_t{1u});

          using std::begin;
          using std::end;
          auto const iter
            = first
              + ::ket::gate::utility::index_with_qubits(
                  index_wo_qubits, index,
                  begin(qubit_masks), end(qubit_masks), begin(index_masks), end(index_masks));
          *iter *= phase_coefficient;
        },
        control_qubit1, control_qubit2, control_qubit3, control_qubits...);
# endif // KET_USE_BIT_MASKS_EXPLICITLY
    }

    template <typename RandomAccessIterator, typename Complex, typename... Qubits>
    inline auto phase_shift_coeff(
      RandomAccessIterator const first, RandomAccessIterator const last,
      Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
      ::ket::control<Qubits> const... control_qubits)
    -> void
    { ::ket::gate::phase_shift_coeff(::ket::utility::policy::make_sequential(), first, last, phase_coefficient, control_qubits...); }

    namespace ranges
    {
      template <typename ParallelPolicy, typename RandomAccessRange, typename Complex, typename... Qubits>
      inline auto phase_shift_coeff(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& state,
        Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
        ::ket::control<Qubits> const... control_qubits)
      -> RandomAccessRange&
      {
        using std::begin;
        using std::end;
        ::ket::gate::phase_shift_coeff(parallel_policy, begin(state), end(state), phase_coefficient, control_qubits...);
        return state;
      }

      template <typename RandomAccessRange, typename Complex, typename... Qubits>
      inline auto phase_shift_coeff(
        RandomAccessRange& state,
        Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
        ::ket::control<Qubits> const... control_qubits)
      -> RandomAccessRange&
      { return ::ket::gate::ranges::phase_shift_coeff(::ket::utility::policy::make_sequential(), state, phase_coefficient, control_qubits...); }
    } // namespace ranges

    template <typename ParallelPolicy, typename RandomAccessIterator, typename Complex, typename... Qubits>
    inline auto adj_phase_shift_coeff(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
      ::ket::control<Qubits> const... control_qubits)
    -> void
    { using std::conj; ::ket::gate::phase_shift_coeff(parallel_policy, first, last, conj(phase_coefficient), control_qubits...); }

    template <typename RandomAccessIterator, typename Complex, typename... Qubits>
    inline auto adj_phase_shift_coeff(
      RandomAccessIterator const first, RandomAccessIterator const last,
      Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
      ::ket::control<Qubits> const... control_qubits)
    -> void
    { using std::conj; ::ket::gate::phase_shift_coeff(first, last, conj(phase_coefficient), control_qubits...); }

    namespace ranges
    {
      template <typename ParallelPolicy, typename RandomAccessRange, typename Complex, typename... Qubits>
      inline auto adj_phase_shift_coeff(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& state,
        Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
        ::ket::control<Qubits> const... control_qubits)
      -> RandomAccessRange&
      { using std::conj; return ::ket::gate::ranges::phase_shift_coeff(parallel_policy, state, conj(phase_coefficient), control_qubits...); }

      template <typename RandomAccessRange, typename Complex, typename... Qubits>
      inline auto adj_phase_shift_coeff(
        RandomAccessRange& state,
        Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
        ::ket::control<Qubits> const... control_qubits)
      -> RandomAccessRange&
      { using std::conj; return ::ket::gate::ranges::phase_shift_coeff(state, conj(phase_coefficient), control_qubits...); }
    } // namespace ranges

    // Case 2: the first argument of qubits is ket::qubit<S, B>
    // U1_i(theta)
    // U1_1(theta) (a_0 |0> + a_1 |1>) = a_0 |0> + e^{i theta} a_1 |1>
    template <typename ParallelPolicy, typename RandomAccessIterator, typename Complex, typename StateInteger, typename BitInteger>
    inline auto phase_shift_coeff(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
      ::ket::qubit<StateInteger, BitInteger> const qubit)
    -> void
    {
      static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
      static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");
      static_assert(
        std::is_same<Complex, typename std::iterator_traits<RandomAccessIterator>::value_type>::value,
        "Complex must be the same to value_type of RandomAccessIterator");

      assert(::ket::utility::integer_exp2<StateInteger>(qubit) < static_cast<StateInteger>(last - first));
      assert(
        ::ket::utility::integer_exp2<StateInteger>(::ket::utility::integer_log2<BitInteger>(last - first))
        == static_cast<StateInteger>(last - first));

      auto const qubit_mask = ::ket::utility::integer_exp2<StateInteger>(qubit);
      auto const lower_bits_mask = qubit_mask - StateInteger{1u};
      auto const upper_bits_mask = compl lower_bits_mask;

      ::ket::utility::loop_n(
        parallel_policy, static_cast<StateInteger>(last - first) >> 1u,
        [first, &phase_coefficient, qubit_mask, lower_bits_mask, upper_bits_mask](StateInteger const value_wo_qubit, int const)
        {
          // xxxxx1xxxxxx
          auto const one_index = ((value_wo_qubit bitand upper_bits_mask) << 1u) bitor (value_wo_qubit bitand lower_bits_mask) bitor qubit_mask;
          *(first + one_index) *= phase_coefficient;
        });
    }

    // CU1_{tc}(theta) or C1U1_{tc}(theta)
    // CU1_{1,2}(theta) (a_{00} |00> + a_{01} |01> + a_{10} |10> + a{11} |11>)
    //   = a_{00} |00> + a_{01} |01> + a_{10} |10> + e^{i thta} a_{11} |11>
    template <typename ParallelPolicy, typename RandomAccessIterator, typename Complex, typename StateInteger, typename BitInteger>
    inline auto phase_shift_coeff(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
      ::ket::qubit<StateInteger, BitInteger> const target_qubit,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit)
    -> void
    {
      static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
      static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");
      static_assert(
        std::is_same<Complex, typename std::iterator_traits<RandomAccessIterator>::value_type>::value,
        "Complex must be the same to value_type of RandomAccessIterator");

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
        [first, &phase_coefficient, target_qubit_mask, control_qubit_mask, lower_bits_mask, middle_bits_mask, upper_bits_mask](
          StateInteger const value_wo_qubits, int const)
        {
          // xxx1_txxx1_cxxx
          auto const target_control_on_index
            = ((value_wo_qubits bitand upper_bits_mask) << 2u)
              bitor ((value_wo_qubits bitand middle_bits_mask) << 1u)
              bitor (value_wo_qubits bitand lower_bits_mask)
              bitor control_qubit_mask bitor target_qubit_mask;
          *(first + target_control_on_index) *= phase_coefficient;
        });
    }

    // C...CU1_{tc...c'}(theta) or CnU1_{tc...c'}(theta)
    template <typename ParallelPolicy, typename RandomAccessIterator, typename Complex, typename StateInteger, typename BitInteger, typename... ControlQubits>
    inline auto phase_shift_coeff(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
      ::ket::qubit<StateInteger, BitInteger> const target_qubit,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2, ControlQubits const... control_qubits)
    -> void
    {
      static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
      static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");
      static_assert(
        (std::is_same<Complex, typename std::iterator_traits<RandomAccessIterator>::value_type>::value),
        "Complex must be the same to value_type of RandomAccessIterator");
      assert(
        ::ket::utility::integer_exp2<StateInteger>(::ket::utility::integer_log2<BitInteger>(last - first))
        == static_cast<StateInteger>(last - first));

# ifndef KET_USE_BIT_MASKS_EXPLICITLY
      using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
      constexpr auto num_control_qubits = static_cast<BitInteger>(sizeof...(ControlQubits) + 2u);
      constexpr auto num_operated_qubits = num_control_qubits + BitInteger{1u};

      ::ket::gate::nocache::gate(
        parallel_policy, first, last,
        [&phase_coefficient](
          auto const first, StateInteger const index_wo_qubits,
          std::array<qubit_type, num_operated_qubits> const& unsorted_qubits,
          std::array<qubit_type, num_operated_qubits + BitInteger{1u}> const& sorted_qubits_with_sentinel,
          int const)
        {
          // 0b11...11u
          constexpr auto index = ((std::size_t{1u} << num_operated_qubits) - std::size_t{1u});

          using std::begin;
          using std::end;
          auto const iter
            = first
              + ::ket::gate::utility::index_with_qubits(
                  index_wo_qubits, index,
                  begin(unsorted_qubits), end(unsorted_qubits),
                  begin(sorted_qubits_with_sentinel), end(sorted_qubits_with_sentinel));
          *iter *= phase_coefficient;
        },
        target_qubit, control_qubit1, control_qubit2, control_qubits...);
# else // KET_USE_BIT_MASKS_EXPLICITLY
      constexpr auto num_control_qubits = static_cast<BitInteger>(sizeof...(ControlQubits) + 2u);
      constexpr auto num_operated_qubits = num_control_qubits + BitInteger{1u};

      ::ket::gate::nocache::gate(
        parallel_policy, first, last,
        [&phase_coefficient](
          auto const first, StateInteger const index_wo_qubits,
          std::array<StateInteger, num_operated_qubits> const& qubit_masks,
          std::array<StateInteger, num_operated_qubits + 1u> const& index_masks,
          int const)
        {
          // 0b11...11u
          constexpr auto index = ((std::size_t{1u} << num_operated_qubits) - std::size_t{1u});

          using std::begin;
          using std::end;
          auto const iter
            = first
              + ::ket::gate::utility::index_with_qubits(
                  index_wo_qubits, index,
                  begin(qubit_masks), end(qubit_masks), begin(index_masks), end(index_masks));
          *iter *= phase_coefficient;
        },
        target_qubit, control_qubit1, control_qubit2, control_qubits...);
# endif // KET_USE_BIT_MASKS_EXPLICITLY
    }

    template <typename RandomAccessIterator, typename Complex, typename StateInteger, typename BitInteger, typename... ControlQubits>
    inline auto phase_shift_coeff(
      RandomAccessIterator const first, RandomAccessIterator const last,
      Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
      ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
    -> void
    { ::ket::gate::phase_shift_coeff(::ket::utility::policy::make_sequential(), first, last, phase_coefficient, target_qubit, control_qubits...); }

    namespace ranges
    {
      template <typename ParallelPolicy, typename RandomAccessRange, typename Complex, typename StateInteger, typename BitInteger, typename... ControlQubits>
      inline auto phase_shift_coeff(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& state,
        Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
        ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
      -> RandomAccessRange&
      {
        using std::begin;
        using std::end;
        ::ket::gate::phase_shift_coeff(parallel_policy, begin(state), end(state), phase_coefficient, target_qubit, control_qubits...);
        return state;
      }

      template <typename RandomAccessRange, typename Complex, typename StateInteger, typename BitInteger, typename... ControlQubits>
      inline auto phase_shift_coeff(
        RandomAccessRange& state,
        Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
        ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
      -> RandomAccessRange&
      { return ::ket::gate::ranges::phase_shift_coeff(::ket::utility::policy::make_sequential(), state, phase_coefficient, target_qubit, control_qubits...); }
    } // namespace ranges

    template <typename ParallelPolicy, typename RandomAccessIterator, typename Complex, typename StateInteger, typename BitInteger, typename... ControlQubits>
    inline auto adj_phase_shift_coeff(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
      ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
    -> void
    { using std::conj; ::ket::gate::phase_shift_coeff(parallel_policy, first, last, conj(phase_coefficient), target_qubit, control_qubits...); }

    template <typename RandomAccessIterator, typename Complex, typename StateInteger, typename BitInteger, typename... ControlQubits>
    inline auto adj_phase_shift_coeff(
      RandomAccessIterator const first, RandomAccessIterator const last,
      Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
      ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
    -> void
    { using std::conj; ::ket::gate::phase_shift_coeff(first, last, conj(phase_coefficient), target_qubit, control_qubits...); }

    namespace ranges
    {
      template <typename ParallelPolicy, typename RandomAccessRange, typename Complex, typename StateInteger, typename BitInteger, typename... ControlQubits>
      inline auto adj_phase_shift_coeff(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& state,
        Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
        ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
      -> RandomAccessRange&
      { using std::conj; return ::ket::gate::ranges::phase_shift_coeff(parallel_policy, state, conj(phase_coefficient), target_qubit, control_qubits...); }

      template <typename RandomAccessRange, typename Complex, typename StateInteger, typename BitInteger, typename... ControlQubits>
      inline auto adj_phase_shift_coeff(
        RandomAccessRange& state,
        Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
        ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
      -> RandomAccessRange&
      { using std::conj; return ::ket::gate::ranges::phase_shift_coeff(state, conj(phase_coefficient), target_qubit, control_qubits...); }
    } // namespace ranges

    // phase_shift
    // Case 1: the first argument of qubits is ket::control<ket::qubit<S, B>>
    template <typename ParallelPolicy, typename RandomAccessIterator, typename Real, typename... Qubits>
    inline auto phase_shift(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      Real const phase, ::ket::control<Qubits> const... control_qubits)
    -> void
    {
      using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
      ::ket::gate::phase_shift_coeff(parallel_policy, first, last, ::ket::utility::exp_i<complex_type>(phase), control_qubits...);
    }

    template <typename RandomAccessIterator, typename Real, typename... Qubits>
    inline auto phase_shift(
      RandomAccessIterator const first, RandomAccessIterator const last,
      Real const phase, ::ket::control<Qubits> const... control_qubits)
    -> void
    {
      using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
      ::ket::gate::phase_shift_coeff(first, last, ::ket::utility::exp_i<complex_type>(phase), control_qubits...);
    }

    namespace ranges
    {
      template <typename ParallelPolicy, typename RandomAccessRange, typename Real, typename... Qubits>
      inline auto phase_shift(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& state, Real const phase, ::ket::control<Qubits> const... control_qubits)
      -> RandomAccessRange&
      {
        using complex_type = ::ket::utility::meta::range_value_t<RandomAccessRange>;
        return ::ket::gate::ranges::phase_shift_coeff(parallel_policy, state, ::ket::utility::exp_i<complex_type>(phase), control_qubits...);
      }

      template <typename RandomAccessRange, typename Real, typename... Qubits>
      inline auto phase_shift(RandomAccessRange& state, Real const phase, ::ket::control<Qubits> const... control_qubits) -> RandomAccessRange&
      {
        using complex_type = ::ket::utility::meta::range_value_t<RandomAccessRange>;
        return ::ket::gate::ranges::phase_shift_coeff(state, ::ket::utility::exp_i<complex_type>(phase), control_qubits...);
      }
    } // namespace ranges

    template <typename ParallelPolicy, typename RandomAccessIterator, typename Real, typename... Qubits>
    inline auto adj_phase_shift(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      Real const phase, ::ket::control<Qubits> const... control_qubits)
    -> void
    { ::ket::gate::phase_shift(parallel_policy, first, last, -phase, control_qubits...); }

    template <typename RandomAccessIterator, typename Real, typename... Qubits>
    inline auto adj_phase_shift(
      RandomAccessIterator const first, RandomAccessIterator const last,
      Real const phase, ::ket::control<Qubits> const... control_qubits)
    -> void
    { ::ket::gate::phase_shift(first, last, -phase, control_qubits...); }

    namespace ranges
    {
      template <typename ParallelPolicy, typename RandomAccessRange, typename Real, typename... Qubits>
      inline auto adj_phase_shift(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& state, Real const phase, ::ket::control<Qubits> const... control_qubits)
      -> RandomAccessRange&
      { return ::ket::gate::ranges::phase_shift(parallel_policy, state, -phase, control_qubits...); }

      template <typename RandomAccessRange, typename Real, typename... Qubits>
      inline auto adj_phase_shift(RandomAccessRange& state, Real const phase, ::ket::control<Qubits> const... control_qubits) -> RandomAccessRange&
      { return ::ket::gate::ranges::phase_shift(state, -phase, control_qubits...); }
    } // namespace ranges

    // Case 2: the first argument of qubits is ket::qubit<S, B>
    template <typename ParallelPolicy, typename RandomAccessIterator, typename Real, typename StateInteger, typename BitInteger, typename... ControlQubits>
    inline auto phase_shift(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      Real const phase, ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
    -> void
    {
      using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
      ::ket::gate::phase_shift_coeff(parallel_policy, first, last, ::ket::utility::exp_i<complex_type>(phase), target_qubit, control_qubits...);
    }

    template <typename RandomAccessIterator, typename Real, typename StateInteger, typename BitInteger, typename... ControlQubits>
    inline auto phase_shift(
      RandomAccessIterator const first, RandomAccessIterator const last,
      Real const phase, ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
    -> void
    {
      using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
      ::ket::gate::phase_shift_coeff(first, last, ::ket::utility::exp_i<complex_type>(phase), target_qubit, control_qubits...);
    }

    namespace ranges
    {
      template <typename ParallelPolicy, typename RandomAccessRange, typename Real, typename StateInteger, typename BitInteger, typename... ControlQubits>
      inline auto phase_shift(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& state, Real const phase, ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
      -> RandomAccessRange&
      {
        using complex_type = ::ket::utility::meta::range_value_t<RandomAccessRange>;
        return ::ket::gate::ranges::phase_shift_coeff(parallel_policy, state, ::ket::utility::exp_i<complex_type>(phase), target_qubit, control_qubits...);
      }

      template <typename RandomAccessRange, typename Real, typename StateInteger, typename BitInteger, typename... ControlQubits>
      inline auto phase_shift(RandomAccessRange& state, Real const phase, ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits) -> RandomAccessRange&
      {
        using complex_type = ::ket::utility::meta::range_value_t<RandomAccessRange>;
        return ::ket::gate::ranges::phase_shift_coeff(state, ::ket::utility::exp_i<complex_type>(phase), target_qubit, control_qubits...);
      }
    } // namespace ranges

    template <typename ParallelPolicy, typename RandomAccessIterator, typename Real, typename StateInteger, typename BitInteger, typename... ControlQubits>
    inline auto adj_phase_shift(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      Real const phase, ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
    -> void
    { ::ket::gate::phase_shift(parallel_policy, first, last, -phase, target_qubit, control_qubits...); }

    template <typename RandomAccessIterator, typename Real, typename StateInteger, typename BitInteger, typename... ControlQubits>
    inline auto adj_phase_shift(
      RandomAccessIterator const first, RandomAccessIterator const last,
      Real const phase, ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
    -> void
    { ::ket::gate::phase_shift(first, last, -phase, target_qubit, control_qubits...); }

    namespace ranges
    {
      template <typename ParallelPolicy, typename RandomAccessRange, typename Real, typename StateInteger, typename BitInteger, typename... ControlQubits>
      inline auto adj_phase_shift(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& state, Real const phase, ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
      -> RandomAccessRange&
      { return ::ket::gate::ranges::phase_shift(parallel_policy, state, -phase, target_qubit, control_qubits...); }

      template <typename RandomAccessRange, typename Real, typename StateInteger, typename BitInteger, typename... ControlQubits>
      inline auto adj_phase_shift(RandomAccessRange& state, Real const phase, ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits) -> RandomAccessRange&
      { return ::ket::gate::ranges::phase_shift(state, -phase, target_qubit, control_qubits...); }
    } // namespace ranges

    // generalized phase_shift
    // U2_i(theta, theta')
    // U2_1(theta, theta') (a_0 |0> + a_1 |1>)
    //   = (a_0 - e^{i theta'} a_1)/sqrt(2) |0> + (e^{i theta} a_0 + e^{i(theta + theta')} a_1)/sqrt(2) |1>
    template <
      typename ParallelPolicy, typename RandomAccessIterator, typename Real,
      typename StateInteger, typename BitInteger>
    inline auto phase_shift2(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      Real const phase1, Real const phase2,
      ::ket::qubit<StateInteger, BitInteger> const qubit)
    -> void
    {
      static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
      static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");

      using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
      static_assert(
        std::is_same<Real, ::ket::utility::meta::real_t<complex_type>>::value,
        "Real must be the same to \"real part\" of value_type of RandomAccessIterator");

      assert(::ket::utility::integer_exp2<StateInteger>(qubit) < static_cast<StateInteger>(last - first));
      assert(
        ::ket::utility::integer_exp2<StateInteger>(::ket::utility::integer_log2<BitInteger>(last - first))
        == static_cast<StateInteger>(last - first));

      auto const phase_coefficient1 = ::ket::utility::exp_i<complex_type>(phase1);
      auto const phase_coefficient2 = ::ket::utility::exp_i<complex_type>(phase2);

      using boost::math::constants::one_div_root_two;
      auto const modified_phase_coefficient1 = one_div_root_two<Real>() * phase_coefficient1;

      auto const qubit_mask = ::ket::utility::integer_exp2<StateInteger>(qubit);
      auto const lower_bits_mask = qubit_mask - StateInteger{1u};
      auto const upper_bits_mask = compl lower_bits_mask;

      ::ket::utility::loop_n(
        parallel_policy, static_cast<StateInteger>(last - first) >> 1u,
        [first, &modified_phase_coefficient1, &phase_coefficient2, qubit_mask, lower_bits_mask, upper_bits_mask](
          StateInteger const value_wo_qubit, int const)
        {
          // xxxxx0xxxxxx
          auto const zero_index = ((value_wo_qubit bitand upper_bits_mask) << 1u) bitor (value_wo_qubit bitand lower_bits_mask);
          // xxxxx1xxxxxx
          auto const one_index = zero_index bitor qubit_mask;
          auto const zero_iter = first + zero_index;
          auto const one_iter = first + one_index;
          auto const zero_iter_value = *zero_iter;

          *zero_iter -= phase_coefficient2 * *one_iter;
          *zero_iter *= one_div_root_two<Real>();
          *one_iter *= phase_coefficient2;
          *one_iter += zero_iter_value;
          *one_iter *= modified_phase_coefficient1;
        });
    }

    // CU2_{tc}(theta, theta') or C1U2_{tc}(theta, theta')
    // CU2_{1,2}(theta, theta') (a_{00} |00> + a_{01} |01> + a_{10} |10> + a{11} |11>)
    //   = a_{00} |00> + a_{01} |01> + (a_{10} - e^{i theta'} a_{11})/sqrt(2) |10>
    //     + (e^{i theta} a_{10} + e^{i(theta + theta')} a_{11})/sqrt(2) |11>
    template <
      typename ParallelPolicy, typename RandomAccessIterator, typename Real,
      typename StateInteger, typename BitInteger>
    inline auto phase_shift2(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      Real const phase1, Real const phase2,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit)
    -> void
    {
      static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
      static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");

      using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
      static_assert(
        std::is_same<Real, ::ket::utility::meta::real_t<complex_type>>::value,
        "Real must be the same to \"real part\" of value_type of RandomAccessIterator");

      assert(::ket::utility::integer_exp2<StateInteger>(target_qubit) < static_cast<StateInteger>(last - first));
      assert(::ket::utility::integer_exp2<StateInteger>(control_qubit) < static_cast<StateInteger>(last - first));
      assert(target_qubit != control_qubit);
      assert(
        ::ket::utility::integer_exp2<StateInteger>(::ket::utility::integer_log2<BitInteger>(last - first))
        == static_cast<StateInteger>(last - first));

      auto const phase_coefficient1 = ::ket::utility::exp_i<complex_type>(phase1);
      auto const phase_coefficient2 = ::ket::utility::exp_i<complex_type>(phase2);

      using boost::math::constants::one_div_root_two;
      auto const modified_phase_coefficient1 = one_div_root_two<Real>() * phase_coefficient1;

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
        [first, &modified_phase_coefficient1, &phase_coefficient2, target_qubit_mask, control_qubit_mask, lower_bits_mask, middle_bits_mask, upper_bits_mask](
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

          *control_on_iter -= phase_coefficient2 * *target_control_on_iter;
          *control_on_iter *= one_div_root_two<Real>();
          *target_control_on_iter *= phase_coefficient2;
          *target_control_on_iter += control_on_iter_value;
          *target_control_on_iter *= modified_phase_coefficient1;
        });
    }

    // C...CU2_{tc...c'}(theta, theta') or CnU2_{tc...c'}(theta, theta')
    template <typename ParallelPolicy, typename RandomAccessIterator, typename Real, typename StateInteger, typename BitInteger, typename... ControlQubits>
    inline auto phase_shift2(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      Real const phase1, Real const phase2,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2, ControlQubits const... control_qubits)
    -> void
    {
      static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
      static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");

      using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
      static_assert(
        std::is_same<Real, ::ket::utility::meta::real_t<complex_type>>::value,
        "Real must be the same to \"real part\" of value_type of RandomAccessIterator");

      assert(::ket::utility::integer_exp2<StateInteger>(target_qubit) < static_cast<StateInteger>(last - first));
      assert(::ket::utility::integer_exp2<StateInteger>(control_qubit1) < static_cast<StateInteger>(last - first));
      assert(::ket::utility::integer_exp2<StateInteger>(control_qubit2) < static_cast<StateInteger>(last - first));
      assert(target_qubit != control_qubit1);
      assert(target_qubit != control_qubit2);
      assert(
        ::ket::utility::integer_exp2<StateInteger>(::ket::utility::integer_log2<BitInteger>(last - first))
        == static_cast<StateInteger>(last - first));

# ifndef KET_USE_BIT_MASKS_EXPLICITLY
      auto const phase_coefficient1 = ::ket::utility::exp_i<complex_type>(phase1);
      auto const phase_coefficient2 = ::ket::utility::exp_i<complex_type>(phase2);

      using boost::math::constants::one_div_root_two;
      auto const modified_phase_coefficient1 = one_div_root_two<Real>() * phase_coefficient1;

      using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
      constexpr auto num_control_qubits = static_cast<BitInteger>(sizeof...(ControlQubits) + 2u);
      constexpr auto num_operated_qubits = num_control_qubits + BitInteger{1u};

      ::ket::gate::nocache::gate(
        parallel_policy, first, last,
        [&modified_phase_coefficient1, &phase_coefficient2](
          auto const first, StateInteger const index_wo_qubits,
          std::array<qubit_type, num_operated_qubits> const& unsorted_qubits,
          std::array<qubit_type, num_operated_qubits + BitInteger{1u}> const& sorted_qubits_with_sentinel,
          int const)
        {
          // 0b11...10u
          constexpr auto index0 = ((std::size_t{1u} << num_control_qubits) - std::size_t{1u}) << std::size_t{1u};
          // 0b11...11u
          constexpr auto index1 = index0 bitor std::size_t{1u};

          using std::begin;
          using std::end;
          auto const iter0
            = first
              + ::ket::gate::utility::index_with_qubits(
                  index_wo_qubits, index0,
                  begin(unsorted_qubits), end(unsorted_qubits),
                  begin(sorted_qubits_with_sentinel), end(sorted_qubits_with_sentinel));
          auto const iter1
            = first
              + ::ket::gate::utility::index_with_qubits(
                  index_wo_qubits, index1,
                  begin(unsorted_qubits), end(unsorted_qubits),
                  begin(sorted_qubits_with_sentinel), end(sorted_qubits_with_sentinel));
          auto const value0 = *iter0;

          *iter0 -= phase_coefficient2 * *iter1;
          *iter0 *= one_div_root_two<Real>();
          *iter1 *= phase_coefficient2;
          *iter1 += value0;
          *iter1 *= modified_phase_coefficient1;
        },
        target_qubit, control_qubit1, control_qubit2, control_qubits...);
# else // KET_USE_BIT_MASKS_EXPLICITLY
      auto const phase_coefficient1 = ::ket::utility::exp_i<complex_type>(phase1);
      auto const phase_coefficient2 = ::ket::utility::exp_i<complex_type>(phase2);

      using boost::math::constants::one_div_root_two;
      auto const modified_phase_coefficient1 = one_div_root_two<Real>() * phase_coefficient1;

      constexpr auto num_control_qubits = static_cast<BitInteger>(sizeof...(ControlQubits) + 2u);
      constexpr auto num_operated_qubits = num_control_qubits + BitInteger{1u};

      ::ket::gate::nocache::gate(
        parallel_policy, first, last,
        [&modified_phase_coefficient1, &phase_coefficient2](
          auto const first, StateInteger const index_wo_qubits,
          std::array<StateInteger, num_operated_qubits> const& qubit_masks,
          std::array<StateInteger, num_operated_qubits + 1u> const& index_masks,
          int const)
        {
          // 0b11...10u
          constexpr auto index0 = ((std::size_t{1u} << num_control_qubits) - std::size_t{1u}) << std::size_t{1u};
          // 0b11...11u
          constexpr auto index1 = index0 bitor std::size_t{1u};

          using std::begin;
          using std::end;
          auto const iter0
            = first
              + ::ket::gate::utility::index_with_qubits(
                  index_wo_qubits, index0,
                  begin(qubit_masks), end(qubit_masks), begin(index_masks), end(index_masks));
          auto const iter1
            = first
              + ::ket::gate::utility::index_with_qubits(
                  index_wo_qubits, index1,
                  begin(qubit_masks), end(qubit_masks), begin(index_masks), end(index_masks));
          auto const value0 = *iter0;

          *iter0 -= phase_coefficient2 * *iter1;
          *iter0 *= one_div_root_two<Real>();
          *iter1 *= phase_coefficient2;
          *iter1 += value0;
          *iter1 *= modified_phase_coefficient1;
        },
        target_qubit, control_qubit1, control_qubit2, control_qubits...);
# endif // KET_USE_BIT_MASKS_EXPLICITLY
    }

    template <typename RandomAccessIterator, typename Real, typename StateInteger, typename BitInteger, typename... ControlQubits>
    inline auto phase_shift2(
      RandomAccessIterator const first, RandomAccessIterator const last,
      Real const phase1, Real const phase2, ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
    -> void
    { ::ket::gate::phase_shift2(::ket::utility::policy::make_sequential(), first, last, phase1, phase2, target_qubit, control_qubits...); }

    namespace ranges
    {
      template <typename ParallelPolicy, typename RandomAccessRange, typename Real, typename StateInteger, typename BitInteger, typename... ControlQubits>
      inline auto phase_shift2(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& state, Real const phase1, Real const phase2,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
      -> RandomAccessRange&
      {
        using std::begin;
        using std::end;
        ::ket::gate::phase_shift2(parallel_policy, begin(state), end(state), phase1, phase2, target_qubit, control_qubits...);
        return state;
      }

      template <typename RandomAccessRange, typename Real, typename StateInteger, typename BitInteger, typename... ControlQubits>
      inline auto phase_shift2(
        RandomAccessRange& state, Real const phase1, Real const phase2,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
      -> RandomAccessRange&
      { return ::ket::gate::ranges::phase_shift2(::ket::utility::policy::make_sequential(), state, phase1, phase2, target_qubit, control_qubits...); }
    } // namespace ranges

    // U2+_i(theta, theta')
    // U2+_1(theta, theta') (a_0 |0> + a_1 |1>)
    //   = (a_0 + e^{-i theta} a_1)/sqrt(2) |0>
    //     + (-e^{-i theta'} a_0 + e^{-i(theta + theta')} a_1)/sqrt(2) |1>
    template <
      typename ParallelPolicy, typename RandomAccessIterator, typename Real,
      typename StateInteger, typename BitInteger>
    inline auto adj_phase_shift2(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      Real const phase1, Real const phase2,
      ::ket::qubit<StateInteger, BitInteger> const qubit)
    -> void
    {
      static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
      static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");

      using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
      static_assert(
        std::is_same<Real, ::ket::utility::meta::real_t<complex_type>>::value,
        "Real must be the same to \"real part\" of value_type of RandomAccessIterator");

      assert(::ket::utility::integer_exp2<StateInteger>(qubit) < static_cast<StateInteger>(last - first));
      assert(
        ::ket::utility::integer_exp2<StateInteger>(::ket::utility::integer_log2<BitInteger>(last - first))
        == static_cast<StateInteger>(last - first));

      auto const phase_coefficient1 = ::ket::utility::exp_i<complex_type>(-phase1);
      auto const phase_coefficient2 = ::ket::utility::exp_i<complex_type>(-phase2);

      using boost::math::constants::one_div_root_two;
      auto const modified_phase_coefficient2 = one_div_root_two<Real>() * phase_coefficient2;

      auto const qubit_mask = ::ket::utility::integer_exp2<StateInteger>(qubit);
      auto const lower_bits_mask = qubit_mask - StateInteger{1u};
      auto const upper_bits_mask = compl lower_bits_mask;

      ::ket::utility::loop_n(
        parallel_policy, static_cast<StateInteger>(last - first) >> 1u,
        [first, &phase_coefficient1, &modified_phase_coefficient2, qubit_mask, lower_bits_mask, upper_bits_mask](
          StateInteger const value_wo_qubit, int const)
        {
          // xxxxx0xxxxxx
          auto const zero_index = ((value_wo_qubit bitand upper_bits_mask) << 1u) bitor (value_wo_qubit bitand lower_bits_mask);
          // xxxxx1xxxxxx
          auto const one_index = zero_index bitor qubit_mask;
          auto const zero_iter = first + zero_index;
          auto const one_iter = first + one_index;
          auto const zero_iter_value = *zero_iter;

          *zero_iter += phase_coefficient1 * *one_iter;
          *zero_iter *= one_div_root_two<Real>();
          *one_iter *= phase_coefficient1;
          *one_iter -= zero_iter_value;
          *one_iter *= modified_phase_coefficient2;
        });
    }

    // CU2+_{tc}(theta, theta') or C1U2+_{tc}(theta, theta')
    // CU2+_{1,2}(theta, theta') (a_{00} |00> + a_{01} |01> + a_{10} |10> + a{11} |11>)
    //   = a_{00} |00> + a_{01} |01> + (a_{10} + e^{-i theta} a_{11})/sqrt(2) |10> 
    //     + (-e^{-i theta'} a_{10} + e^{-i(theta + theta')} a_{11})/sqrt(2) |11>
    template <
      typename ParallelPolicy, typename RandomAccessIterator, typename Real,
      typename StateInteger, typename BitInteger>
    inline auto adj_phase_shift2(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      Real const phase1, Real const phase2,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit)
    -> void
    {
      static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
      static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");

      using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
      static_assert(
        std::is_same<Real, ::ket::utility::meta::real_t<complex_type>>::value,
        "Real must be the same to \"real part\" of value_type of RandomAccessIterator");

      assert(::ket::utility::integer_exp2<StateInteger>(target_qubit) < static_cast<StateInteger>(last - first));
      assert(::ket::utility::integer_exp2<StateInteger>(control_qubit) < static_cast<StateInteger>(last - first));
      assert(target_qubit != control_qubit);
      assert(
        ::ket::utility::integer_exp2<StateInteger>(::ket::utility::integer_log2<BitInteger>(last - first))
        == static_cast<StateInteger>(last - first));

      auto const phase_coefficient1 = ::ket::utility::exp_i<complex_type>(-phase1);
      auto const phase_coefficient2 = ::ket::utility::exp_i<complex_type>(-phase2);

      using boost::math::constants::one_div_root_two;
      auto const modified_phase_coefficient2 = one_div_root_two<Real>() * phase_coefficient2;

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
        [first, &phase_coefficient1, &modified_phase_coefficient2, target_qubit_mask, control_qubit_mask, lower_bits_mask, middle_bits_mask, upper_bits_mask](
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

          *control_on_iter += phase_coefficient1 * *target_control_on_iter;
          *control_on_iter *= one_div_root_two<Real>();
          *target_control_on_iter *= phase_coefficient1;
          *target_control_on_iter -= control_on_iter_value;
          *target_control_on_iter *= modified_phase_coefficient2;
        });
    }

    // C...CU2+_{tc...c'}(theta, theta'), or CnU2+_{tc...c'}(theta, theta')
    template <typename ParallelPolicy, typename RandomAccessIterator, typename Real, typename StateInteger, typename BitInteger, typename... ControlQubits>
    inline auto adj_phase_shift2(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      Real const phase1, Real const phase2,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2, ControlQubits const... control_qubits)
    -> void
    {
      static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
      static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");

      using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
      static_assert(
        std::is_same<Real, ::ket::utility::meta::real_t<complex_type>>::value,
        "Real must be the same to \"real part\" of value_type of RandomAccessIterator");

      assert(::ket::utility::integer_exp2<StateInteger>(target_qubit) < static_cast<StateInteger>(last - first));
      assert(::ket::utility::integer_exp2<StateInteger>(control_qubit1) < static_cast<StateInteger>(last - first));
      assert(::ket::utility::integer_exp2<StateInteger>(control_qubit2) < static_cast<StateInteger>(last - first));
      assert(target_qubit != control_qubit1);
      assert(target_qubit != control_qubit2);
      assert(
        ::ket::utility::integer_exp2<StateInteger>(::ket::utility::integer_log2<BitInteger>(last - first))
        == static_cast<StateInteger>(last - first));

# ifndef KET_USE_BIT_MASKS_EXPLICITLY
      auto const phase_coefficient1 = ::ket::utility::exp_i<complex_type>(-phase1);
      auto const phase_coefficient2 = ::ket::utility::exp_i<complex_type>(-phase2);

      using boost::math::constants::one_div_root_two;
      auto const modified_phase_coefficient2 = one_div_root_two<Real>() * phase_coefficient2;

      using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
      constexpr auto num_control_qubits = static_cast<BitInteger>(sizeof...(ControlQubits) + 2u);
      constexpr auto num_operated_qubits = num_control_qubits + BitInteger{1u};

      ::ket::gate::nocache::gate(
        parallel_policy, first, last,
        [&phase_coefficient1, &modified_phase_coefficient2](
          auto const first, StateInteger const index_wo_qubits,
          std::array<qubit_type, num_operated_qubits> const& unsorted_qubits,
          std::array<qubit_type, num_operated_qubits + BitInteger{1u}> const& sorted_qubits_with_sentinel,
          int const)
        {
          // 0b11...10u
          constexpr auto index0 = ((std::size_t{1u} << num_control_qubits) - std::size_t{1u}) << std::size_t{1u};
          // 0b11...11u
          constexpr auto index1 = index0 bitor std::size_t{1u};

          using std::begin;
          using std::end;
          auto const iter0
            = first
              + ::ket::gate::utility::index_with_qubits(
                  index_wo_qubits, index0,
                  begin(unsorted_qubits), end(unsorted_qubits),
                  begin(sorted_qubits_with_sentinel), end(sorted_qubits_with_sentinel));
          auto const iter1
            = first
              + ::ket::gate::utility::index_with_qubits(
                  index_wo_qubits, index1,
                  begin(unsorted_qubits), end(unsorted_qubits),
                  begin(sorted_qubits_with_sentinel), end(sorted_qubits_with_sentinel));
          auto const value0 = *iter0;

          *iter0 += phase_coefficient1 * *iter1;
          *iter0 *= one_div_root_two<Real>();
          *iter1 *= phase_coefficient1;
          *iter1 -= value0;
          *iter1 *= modified_phase_coefficient2;
        },
        target_qubit, control_qubit1, control_qubit2, control_qubits...);
# else // KET_USE_BIT_MASKS_EXPLICITLY
      auto const phase_coefficient1 = ::ket::utility::exp_i<complex_type>(-phase1);
      auto const phase_coefficient2 = ::ket::utility::exp_i<complex_type>(-phase2);

      using boost::math::constants::one_div_root_two;
      auto const modified_phase_coefficient2 = one_div_root_two<Real>() * phase_coefficient2;

      constexpr auto num_control_qubits = static_cast<BitInteger>(sizeof...(ControlQubits) + 2u);
      constexpr auto num_operated_qubits = num_control_qubits + BitInteger{1u};

      ::ket::gate::nocache::gate(
        parallel_policy, first, last,
        [&phase_coefficient1, &modified_phase_coefficient2](
          auto const first, StateInteger const index_wo_qubits,
          std::array<StateInteger, num_operated_qubits> const& qubit_masks,
          std::array<StateInteger, num_operated_qubits + 1u> const& index_masks,
          int const)
        {
          // 0b11...10u
          constexpr auto index0 = ((std::size_t{1u} << num_control_qubits) - std::size_t{1u}) << std::size_t{1u};
          // 0b11...11u
          constexpr auto index1 = index0 bitor std::size_t{1u};

          using std::begin;
          using std::end;
          auto const iter0
            = first
              + ::ket::gate::utility::index_with_qubits(
                  index_wo_qubits, index0,
                  begin(qubit_masks), end(qubit_masks), begin(index_masks), end(index_masks));
          auto const iter1
            = first
              + ::ket::gate::utility::index_with_qubits(
                  index_wo_qubits, index1,
                  begin(qubit_masks), end(qubit_masks), begin(index_masks), end(index_masks));
          auto const value0 = *iter0;

          *iter0 += phase_coefficient1 * *iter1;
          *iter0 *= one_div_root_two<Real>();
          *iter1 *= phase_coefficient1;
          *iter1 -= value0;
          *iter1 *= modified_phase_coefficient2;
        },
        target_qubit, control_qubit1, control_qubit2, control_qubits...);
# endif // KET_USE_BIT_MASKS_EXPLICITLY
    }

    template <typename RandomAccessIterator, typename Real, typename StateInteger, typename BitInteger, typename... ControlQubits>
    inline auto adj_phase_shift2(
      RandomAccessIterator const first, RandomAccessIterator const last,
      Real const phase1, Real const phase2, ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
    -> void
    { ::ket::gate::adj_phase_shift2(::ket::utility::policy::make_sequential(), first, last, phase1, phase2, target_qubit, control_qubits...); }

    namespace ranges
    {
      template <typename ParallelPolicy, typename RandomAccessRange, typename Real, typename StateInteger, typename BitInteger, typename... ControlQubits>
      inline auto adj_phase_shift2(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& state, Real const phase1, Real const phase2,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
      -> RandomAccessRange&
      {
        using std::begin;
        using std::end;
        ::ket::gate::adj_phase_shift2(parallel_policy, begin(state), end(state), phase1, phase2, target_qubit, control_qubits...);
        return state;
      }

      template <typename RandomAccessRange, typename Real, typename StateInteger, typename BitInteger, typename... ControlQubits>
      inline auto adj_phase_shift2(
        RandomAccessRange& state, Real const phase1, Real const phase2,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
      -> RandomAccessRange&
      { return ::ket::gate::ranges::adj_phase_shift2(::ket::utility::policy::make_sequential(), state, phase1, phase2, target_qubit, control_qubits...); }
    } // namespace ranges

    // U3_i(theta, theta', theta'')
    // U3_1(theta, theta', theta'') (a_0 |0> + a_1 |1>)
    //   = (cos(theta/2) a_0 - e^{i theta''} sin(theta/2) a_1) |0>
    //     + (e^{i theta'} sin(theta/2) a_0 + e^{i(theta' + theta'')} cos(theta/2) a_1) |1>
    template <typename ParallelPolicy, typename RandomAccessIterator, typename Real, typename StateInteger, typename BitInteger>
    inline auto phase_shift3(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      Real const phase1, Real const phase2, Real const phase3, ::ket::qubit<StateInteger, BitInteger> const qubit)
    -> void
    {
      static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
      static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");

      using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
      static_assert(
        std::is_same<Real, ::ket::utility::meta::real_t<complex_type>>::value,
        "Real must be the same to \"real part\" of value_type of RandomAccessIterator");

      assert(::ket::utility::integer_exp2<StateInteger>(qubit) < static_cast<StateInteger>(last - first));
      assert(
        ::ket::utility::integer_exp2<StateInteger>(::ket::utility::integer_log2<BitInteger>(last - first))
        == static_cast<StateInteger>(last - first));

      using std::cos;
      using std::sin;
      using boost::math::constants::half;
      auto const sine = sin(half<Real>() * phase1);
      auto const cosine = cos(half<Real>() * phase1);

      auto const phase_coefficient2 = ::ket::utility::exp_i<complex_type>(phase2);
      auto const phase_coefficient3 = ::ket::utility::exp_i<complex_type>(phase3);

      auto const sine_phase_coefficient3 = sine * phase_coefficient3;
      auto const cosine_phase_coefficient3 = cosine * phase_coefficient3;

      auto const qubit_mask = ::ket::utility::integer_exp2<StateInteger>(qubit);
      auto const lower_bits_mask = qubit_mask - StateInteger{1u};
      auto const upper_bits_mask = compl lower_bits_mask;

      ::ket::utility::loop_n(
        parallel_policy, static_cast<StateInteger>(last - first) >> 1u,
        [first, sine, cosine, &phase_coefficient2, &sine_phase_coefficient3, &cosine_phase_coefficient3, qubit_mask, lower_bits_mask, upper_bits_mask](
          StateInteger const value_wo_qubit, int const)
        {
          // xxxxx0xxxxxx
          auto const zero_index = ((value_wo_qubit bitand upper_bits_mask) << 1u) bitor (value_wo_qubit bitand lower_bits_mask);
          // xxxxx1xxxxxx
          auto const one_index = zero_index bitor qubit_mask;
          auto const zero_iter = first + zero_index;
          auto const one_iter = first + one_index;
          auto const zero_iter_value = *zero_iter;

          *zero_iter *= cosine;
          *zero_iter -= sine_phase_coefficient3 * *one_iter;
          *one_iter *= cosine_phase_coefficient3;
          *one_iter += sine * zero_iter_value;
          *one_iter *= phase_coefficient2;
        });
    }

    // CU3_{tc}(theta, theta', theta''), or C1U3_{tc}(theta, theta', theta'')
    // CU3_{1,2}(theta, theta', theta'') (a_{00} |00> + a_{01} |01> + a_{10} |10> + a{11} |11>)
    //   = a_{00} |00> + a_{01} |01> + (cos(theta/2) a_{10} - e^{i theta''} sin(theta/2) a_{11}) |10>
    //     + (e^{i theta'} sin(theta/2) a_{10} + e^{i(theta' + theta'')} cos(theta/2) a_{11}) |11>
    template <
      typename ParallelPolicy, typename RandomAccessIterator, typename Real,
      typename StateInteger, typename BitInteger>
    inline auto phase_shift3(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      Real const phase1, Real const phase2, Real const phase3,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit)
    -> void
    {
      static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
      static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");

      using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
      static_assert(
        std::is_same<Real, ::ket::utility::meta::real_t<complex_type>>::value,
        "Real must be the same to \"real part\" of value_type of RandomAccessIterator");

      assert(::ket::utility::integer_exp2<StateInteger>(target_qubit) < static_cast<StateInteger>(last - first));
      assert(::ket::utility::integer_exp2<StateInteger>(control_qubit) < static_cast<StateInteger>(last - first));
      assert(target_qubit != control_qubit);
      assert(
        ::ket::utility::integer_exp2<StateInteger>(::ket::utility::integer_log2<BitInteger>(last - first))
        == static_cast<StateInteger>(last - first));

      using std::cos;
      using std::sin;
      using boost::math::constants::half;
      auto const sine = sin(half<Real>() * phase1);
      auto const cosine = cos(half<Real>() * phase1);

      auto const phase_coefficient2 = ::ket::utility::exp_i<complex_type>(phase2);
      auto const phase_coefficient3 = ::ket::utility::exp_i<complex_type>(phase3);

      auto const sine_phase_coefficient3 = sine * phase_coefficient3;
      auto const cosine_phase_coefficient3 = cosine * phase_coefficient3;

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
        [first, sine, cosine, &phase_coefficient2, &sine_phase_coefficient3, &cosine_phase_coefficient3,
         target_qubit_mask, control_qubit_mask, lower_bits_mask, middle_bits_mask, upper_bits_mask](
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

          *control_on_iter *= cosine;
          *control_on_iter -= sine_phase_coefficient3 * *target_control_on_iter;
          *target_control_on_iter *= cosine_phase_coefficient3;
          *target_control_on_iter += sine * control_on_iter_value;
          *target_control_on_iter *= phase_coefficient2;
        });
    }

    // C...CU3_{tc...c'}(theta, theta', theta''), or CnU3_{tc...c'}(theta, theta', theta'')
    template <typename ParallelPolicy, typename RandomAccessIterator, typename Real, typename StateInteger, typename BitInteger, typename... ControlQubits>
    inline auto phase_shift3(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      Real const phase1, Real const phase2, Real const phase3,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2, ControlQubits const... control_qubits)
    -> void
    {
      static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
      static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");

      using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
      static_assert(
        std::is_same<Real, ::ket::utility::meta::real_t<complex_type>>::value,
        "Real must be the same to \"real part\" of value_type of RandomAccessIterator");

      assert(::ket::utility::integer_exp2<StateInteger>(target_qubit) < static_cast<StateInteger>(last - first));
      assert(::ket::utility::integer_exp2<StateInteger>(control_qubit1) < static_cast<StateInteger>(last - first));
      assert(::ket::utility::integer_exp2<StateInteger>(control_qubit2) < static_cast<StateInteger>(last - first));
      assert(target_qubit != control_qubit1);
      assert(target_qubit != control_qubit2);
      assert(
        ::ket::utility::integer_exp2<StateInteger>(::ket::utility::integer_log2<BitInteger>(last - first))
        == static_cast<StateInteger>(last - first));

# ifndef KET_USE_BIT_MASKS_EXPLICITLY
      using std::cos;
      using std::sin;
      using boost::math::constants::half;
      auto const sine = sin(half<Real>() * phase1);
      auto const cosine = cos(half<Real>() * phase1);

      auto const phase_coefficient2 = ::ket::utility::exp_i<complex_type>(phase2);
      auto const phase_coefficient3 = ::ket::utility::exp_i<complex_type>(phase3);

      auto const sine_phase_coefficient3 = sine * phase_coefficient3;
      auto const cosine_phase_coefficient3 = cosine * phase_coefficient3;

      using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
      constexpr auto num_control_qubits = static_cast<BitInteger>(sizeof...(ControlQubits) + 2u);
      constexpr auto num_operated_qubits = num_control_qubits + BitInteger{1u};

      ::ket::gate::nocache::gate(
        parallel_policy, first, last,
        [sine, cosine, &phase_coefficient2, &sine_phase_coefficient3, &cosine_phase_coefficient3](
          auto const first, StateInteger const index_wo_qubits,
          std::array<qubit_type, num_operated_qubits> const& unsorted_qubits,
          std::array<qubit_type, num_operated_qubits + BitInteger{1u}> const& sorted_qubits_with_sentinel,
          int const)
        {
          // 0b11...10u
          constexpr auto index0 = ((std::size_t{1u} << num_control_qubits) - std::size_t{1u}) << std::size_t{1u};
          // 0b11...11u
          constexpr auto index1 = index0 bitor std::size_t{1u};

          using std::begin;
          using std::end;
          auto const iter0
            = first
              + ::ket::gate::utility::index_with_qubits(
                  index_wo_qubits, index0,
                  begin(unsorted_qubits), end(unsorted_qubits),
                  begin(sorted_qubits_with_sentinel), end(sorted_qubits_with_sentinel));
          auto const iter1
            = first
              + ::ket::gate::utility::index_with_qubits(
                  index_wo_qubits, index1,
                  begin(unsorted_qubits), end(unsorted_qubits),
                  begin(sorted_qubits_with_sentinel), end(sorted_qubits_with_sentinel));
          auto const value0 = *iter0;

          *iter0 *= cosine;
          *iter0 -= sine_phase_coefficient3 * *iter1;
          *iter1 *= cosine_phase_coefficient3;
          *iter1 += sine * value0;
          *iter1 *= phase_coefficient2;
        },
        target_qubit, control_qubit1, control_qubit2, control_qubits...);
# else // KET_USE_BIT_MASKS_EXPLICITLY
      using std::cos;
      using std::sin;
      using boost::math::constants::half;
      auto const sine = sin(half<Real>() * phase1);
      auto const cosine = cos(half<Real>() * phase1);

      auto const phase_coefficient2 = ::ket::utility::exp_i<complex_type>(phase2);
      auto const phase_coefficient3 = ::ket::utility::exp_i<complex_type>(phase3);

      auto const sine_phase_coefficient3 = sine * phase_coefficient3;
      auto const cosine_phase_coefficient3 = cosine * phase_coefficient3;

      constexpr auto num_control_qubits = static_cast<BitInteger>(sizeof...(ControlQubits) + 2u);
      constexpr auto num_operated_qubits = num_control_qubits + BitInteger{1u};

      ::ket::gate::nocache::gate(
        parallel_policy, first, last,
        [sine, cosine, &phase_coefficient2, &sine_phase_coefficient3, &cosine_phase_coefficient3](
          auto const first, StateInteger const index_wo_qubits,
          std::array<StateInteger, num_operated_qubits> const& qubit_masks,
          std::array<StateInteger, num_operated_qubits + 1u> const& index_masks,
          int const)
        {
          // 0b11...10u
          constexpr auto index0 = ((std::size_t{1u} << num_control_qubits) - std::size_t{1u}) << std::size_t{1u};
          // 0b11...11u
          constexpr auto index1 = index0 bitor std::size_t{1u};

          using std::begin;
          using std::end;
          auto const iter0
            = first
              + ::ket::gate::utility::index_with_qubits(
                  index_wo_qubits, index0,
                  begin(qubit_masks), end(qubit_masks), begin(index_masks), end(index_masks));
          auto const iter1
            = first
              + ::ket::gate::utility::index_with_qubits(
                  index_wo_qubits, index1,
                  begin(qubit_masks), end(qubit_masks), begin(index_masks), end(index_masks));
          auto const value0 = *iter0;

          *iter0 *= cosine;
          *iter0 -= sine_phase_coefficient3 * *iter1;
          *iter1 *= cosine_phase_coefficient3;
          *iter1 += sine * value0;
          *iter1 *= phase_coefficient2;
        },
        target_qubit, control_qubit1, control_qubit2, control_qubits...);
# endif // KET_USE_BIT_MASKS_EXPLICITLY
    }

    template <typename RandomAccessIterator, typename Real, typename StateInteger, typename BitInteger, typename... ControlQubits>
    inline auto phase_shift3(
      RandomAccessIterator const first, RandomAccessIterator const last,
      Real const phase1, Real const phase2, Real const phase3,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
    -> void
    { ::ket::gate::phase_shift3(::ket::utility::policy::make_sequential(), first, last, phase1, phase2, phase3, target_qubit, control_qubits...); }

    namespace ranges
    {
      template <typename ParallelPolicy, typename RandomAccessRange, typename Real, typename StateInteger, typename BitInteger, typename... ControlQubits>
      inline auto phase_shift3(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& state, Real const phase1, Real const phase2, Real const phase3,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
      -> RandomAccessRange&
      {
        using std::begin;
        using std::end;
        ::ket::gate::phase_shift3(parallel_policy, begin(state), end(state), phase1, phase2, phase3, target_qubit, control_qubits...);
        return state;
      }

      template <typename RandomAccessRange, typename Real, typename StateInteger, typename BitInteger, typename... ControlQubits>
      inline auto phase_shift3(
        RandomAccessRange& state, Real const phase1, Real const phase2, Real const phase3,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
      -> RandomAccessRange&
      { return ::ket::gate::ranges::phase_shift3(::ket::utility::policy::make_sequential(), state, phase1, phase2, phase3, target_qubit, control_qubits...); }
    } // namespace ranges

    // U3+_i(theta, theta', theta'')
    // U3+_1(theta, theta', theta'') (a_0 |0> + a_1 |1>)
    //   = (cos(theta/2) a_0 + e^{-i theta'} sin(theta/2) a_1) |0>
    //     + (-e^{-i theta''} sin(theta/2) a_0 + e^{-i(theta' + theta'')} cos(theta/2) a_1) |1>
    template <typename ParallelPolicy, typename RandomAccessIterator, typename Real, typename StateInteger, typename BitInteger>
    inline auto adj_phase_shift3(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      Real const phase1, Real const phase2, Real const phase3, ::ket::qubit<StateInteger, BitInteger> const qubit)
    -> void
    {
      static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
      static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");

      using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
      static_assert(
        std::is_same<Real, ::ket::utility::meta::real_t<complex_type>>::value,
        "Real must be the same to \"real part\" of value_type of RandomAccessIterator");

      assert(::ket::utility::integer_exp2<StateInteger>(qubit) < static_cast<StateInteger>(last - first));
      assert(
        ::ket::utility::integer_exp2<StateInteger>(::ket::utility::integer_log2<BitInteger>(last - first))
        == static_cast<StateInteger>(last - first));

      using std::cos;
      using std::sin;
      using boost::math::constants::half;
      auto const sine = sin(half<Real>() * phase1);
      auto const cosine = cos(half<Real>() * phase1);

      auto const phase_coefficient2 = ::ket::utility::exp_i<complex_type>(-phase2);
      auto const phase_coefficient3 = ::ket::utility::exp_i<complex_type>(-phase3);

      auto const sine_phase_coefficient2 = sine * phase_coefficient2;
      auto const cosine_phase_coefficient2 = cosine * phase_coefficient2;

      auto const qubit_mask = ::ket::utility::integer_exp2<StateInteger>(qubit);
      auto const lower_bits_mask = qubit_mask - StateInteger{1u};
      auto const upper_bits_mask = compl lower_bits_mask;

      ::ket::utility::loop_n(
        parallel_policy, static_cast<StateInteger>(last - first) >> 1u,
        [first, sine, cosine, &sine_phase_coefficient2, &cosine_phase_coefficient2, &phase_coefficient3, qubit_mask, lower_bits_mask, upper_bits_mask](
          StateInteger const value_wo_qubit, int const)
        {
          // xxxxx0xxxxxx
          auto const zero_index = ((value_wo_qubit bitand upper_bits_mask) << 1u) bitor (value_wo_qubit bitand lower_bits_mask);
          // xxxxx1xxxxxx
          auto const one_index = zero_index bitor qubit_mask;
          auto const zero_iter = first + zero_index;
          auto const one_iter = first + one_index;
          auto const zero_iter_value = *zero_iter;

          *zero_iter *= cosine;
          *zero_iter += sine_phase_coefficient2 * *one_iter;
          *one_iter *= cosine_phase_coefficient2;
          *one_iter -= sine * zero_iter_value;
          *one_iter *= phase_coefficient3;
        });
    }

    // CU3+_{tc}(theta, theta', theta''), or C1U3+_{tc}(theta, theta', theta'')
    // CU3+_{1,2}(theta, theta', theta'') (a_{00} |00> + a_{01} |01> + a_{10} |10> + a{11} |11>)
    //   = a_{00} |00> + a_{01} |01> + (cos(theta/2) a_{10} + e^{-i theta'} sin(theta/2) a_{11}) |10>
    //     + (-e^{-i theta''} sin(theta/2) a_{10} + e^{-i(theta' + theta'')} cos(theta/2) a_{11}) |11>
    template <
      typename ParallelPolicy, typename RandomAccessIterator, typename Real,
      typename StateInteger, typename BitInteger>
    inline auto adj_phase_shift3(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      Real const phase1, Real const phase2, Real const phase3,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit)
    -> void
    {
      static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
      static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");

      using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
      static_assert(
        std::is_same<Real, ::ket::utility::meta::real_t<complex_type>>::value,
        "Real must be the same to \"real part\" of value_type of RandomAccessIterator");

      assert(::ket::utility::integer_exp2<StateInteger>(target_qubit) < static_cast<StateInteger>(last - first));
      assert(::ket::utility::integer_exp2<StateInteger>(control_qubit) < static_cast<StateInteger>(last - first));
      assert(target_qubit != control_qubit);
      assert(
        ::ket::utility::integer_exp2<StateInteger>(::ket::utility::integer_log2<BitInteger>(last - first))
        == static_cast<StateInteger>(last - first));

      using std::cos;
      using std::sin;
      using boost::math::constants::half;
      auto const sine = sin(half<Real>() * phase1);
      auto const cosine = cos(half<Real>() * phase1);

      auto const phase_coefficient2 = ::ket::utility::exp_i<complex_type>(-phase2);
      auto const phase_coefficient3 = ::ket::utility::exp_i<complex_type>(-phase3);

      auto const sine_phase_coefficient2 = sine * phase_coefficient2;
      auto const cosine_phase_coefficient2 = cosine * phase_coefficient2;

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
        [first, sine, cosine, &sine_phase_coefficient2, &cosine_phase_coefficient2, &phase_coefficient3,
         target_qubit_mask, control_qubit_mask, lower_bits_mask, middle_bits_mask, upper_bits_mask](
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

          *control_on_iter *= cosine;
          *control_on_iter += sine_phase_coefficient2 * *target_control_on_iter;
          *target_control_on_iter *= cosine_phase_coefficient2;
          *target_control_on_iter -= sine * control_on_iter_value;
          *target_control_on_iter *= phase_coefficient3;
        });
    }

    // C...CU3+_{tc...c'}(theta, theta', theta''), or CnU3+_{tc...c'}(theta, theta', theta'')
    template <typename ParallelPolicy, typename RandomAccessIterator, typename Real, typename StateInteger, typename BitInteger, typename... ControlQubits>
    inline auto adj_phase_shift3(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      Real const phase1, Real const phase2, Real const phase3,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2, ControlQubits const... control_qubits)
    -> void
    {
      static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
      static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");

      using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
      static_assert(
        std::is_same<Real, ::ket::utility::meta::real_t<complex_type>>::value,
        "Real must be the same to \"real part\" of value_type of RandomAccessIterator");

      assert(::ket::utility::integer_exp2<StateInteger>(target_qubit) < static_cast<StateInteger>(last - first));
      assert(::ket::utility::integer_exp2<StateInteger>(control_qubit1) < static_cast<StateInteger>(last - first));
      assert(::ket::utility::integer_exp2<StateInteger>(control_qubit2) < static_cast<StateInteger>(last - first));
      assert(target_qubit != control_qubit1);
      assert(target_qubit != control_qubit2);
      assert(
        ::ket::utility::integer_exp2<StateInteger>(::ket::utility::integer_log2<BitInteger>(last - first))
        == static_cast<StateInteger>(last - first));

# ifndef KET_USE_BIT_MASKS_EXPLICITLY
      using std::cos;
      using std::sin;
      using boost::math::constants::half;
      auto const sine = sin(half<Real>() * phase1);
      auto const cosine = cos(half<Real>() * phase1);

      auto const phase_coefficient2 = ::ket::utility::exp_i<complex_type>(-phase2);
      auto const phase_coefficient3 = ::ket::utility::exp_i<complex_type>(-phase3);

      auto const sine_phase_coefficient2 = sine * phase_coefficient2;
      auto const cosine_phase_coefficient2 = cosine * phase_coefficient2;

      using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
      constexpr auto num_control_qubits = static_cast<BitInteger>(sizeof...(ControlQubits) + 2u);
      constexpr auto num_operated_qubits = num_control_qubits + BitInteger{1u};

      ::ket::gate::nocache::gate(
        parallel_policy, first, last,
        [sine, cosine, &sine_phase_coefficient2, &cosine_phase_coefficient2, &phase_coefficient3](
          auto const first, StateInteger const index_wo_qubits,
          std::array<qubit_type, num_operated_qubits> const& unsorted_qubits,
          std::array<qubit_type, num_operated_qubits + BitInteger{1u}> const& sorted_qubits_with_sentinel,
          int const)
        {
          // 0b11...10u
          constexpr auto index0 = ((std::size_t{1u} << num_control_qubits) - std::size_t{1u}) << std::size_t{1u};
          // 0b11...11u
          constexpr auto index1 = index0 bitor std::size_t{1u};

          using std::begin;
          using std::end;
          auto const iter0
            = first
              + ::ket::gate::utility::index_with_qubits(
                  index_wo_qubits, index0,
                  begin(unsorted_qubits), end(unsorted_qubits),
                  begin(sorted_qubits_with_sentinel), end(sorted_qubits_with_sentinel));
          auto const iter1
            = first
              + ::ket::gate::utility::index_with_qubits(
                  index_wo_qubits, index1,
                  begin(unsorted_qubits), end(unsorted_qubits),
                  begin(sorted_qubits_with_sentinel), end(sorted_qubits_with_sentinel));
          auto const value0 = *iter0;

          *iter0 *= cosine;
          *iter0 += sine_phase_coefficient2 * *iter1;
          *iter1 *= cosine_phase_coefficient2;
          *iter1 -= sine * value0;
          *iter1 *= phase_coefficient3;
        },
        target_qubit, control_qubit1, control_qubit2, control_qubits...);
# else // KET_USE_BIT_MASKS_EXPLICITLY
      using std::cos;
      using std::sin;
      using boost::math::constants::half;
      auto const sine = sin(half<Real>() * phase1);
      auto const cosine = cos(half<Real>() * phase1);

      auto const phase_coefficient2 = ::ket::utility::exp_i<complex_type>(-phase2);
      auto const phase_coefficient3 = ::ket::utility::exp_i<complex_type>(-phase3);

      auto const sine_phase_coefficient2 = sine * phase_coefficient2;
      auto const cosine_phase_coefficient2 = cosine * phase_coefficient2;

      constexpr auto num_control_qubits = static_cast<BitInteger>(sizeof...(ControlQubits) + 2u);
      constexpr auto num_operated_qubits = num_control_qubits + BitInteger{1u};

      ::ket::gate::nocache::gate(
        parallel_policy, first, last,
        [sine, cosine, &sine_phase_coefficient2, &cosine_phase_coefficient2, &phase_coefficient3](
          auto const first, StateInteger const index_wo_qubits,
          std::array<StateInteger, num_operated_qubits> const& qubit_masks,
          std::array<StateInteger, num_operated_qubits + 1u> const& index_masks,
          int const)
        {
          // 0b11...10u
          constexpr auto index0 = ((std::size_t{1u} << num_control_qubits) - std::size_t{1u}) << std::size_t{1u};
          // 0b11...11u
          constexpr auto index1 = index0 bitor std::size_t{1u};

          using std::begin;
          using std::end;
          auto const iter0
            = first
              + ::ket::gate::utility::index_with_qubits(
                  index_wo_qubits, index0,
                  begin(qubit_masks), end(qubit_masks), begin(index_masks), end(index_masks));
          auto const iter1
            = first
              + ::ket::gate::utility::index_with_qubits(
                  index_wo_qubits, index1,
                  begin(qubit_masks), end(qubit_masks), begin(index_masks), end(index_masks));
          auto const value0 = *iter0;

          *iter0 *= cosine;
          *iter0 += sine_phase_coefficient2 * *iter1;
          *iter1 *= cosine_phase_coefficient2;
          *iter1 -= sine * value0;
          *iter1 *= phase_coefficient3;
        },
        target_qubit, control_qubit1, control_qubit2, control_qubits...);
# endif // KET_USE_BIT_MASKS_EXPLICITLY
    }

    template <typename RandomAccessIterator, typename Real, typename StateInteger, typename BitInteger, typename... ControlQubits>
    inline auto adj_phase_shift3(
      RandomAccessIterator const first, RandomAccessIterator const last,
      Real const phase1, Real const phase2, Real const phase3,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
    -> void
    { ::ket::gate::adj_phase_shift3(::ket::utility::policy::make_sequential(), first, last, phase1, phase2, phase3, target_qubit, control_qubits...); }

    namespace ranges
    {
      template <typename ParallelPolicy, typename RandomAccessRange, typename Real, typename StateInteger, typename BitInteger, typename... ControlQubits>
      inline auto adj_phase_shift3(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& state, Real const phase1, Real const phase2, Real const phase3,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
      -> RandomAccessRange&
      {
        using std::begin;
        using std::end;
        ::ket::gate::adj_phase_shift3(parallel_policy, begin(state), end(state), phase1, phase2, phase3, target_qubit, control_qubits...);
        return state;
      }

      template <typename RandomAccessRange, typename Real, typename StateInteger, typename BitInteger, typename... ControlQubits>
      inline auto adj_phase_shift3(
        RandomAccessRange& state, Real const phase1, Real const phase2, Real const phase3,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
      -> RandomAccessRange&
      { return ::ket::gate::ranges::adj_phase_shift3(::ket::utility::policy::make_sequential(), state, phase1, phase2, phase3, target_qubit, control_qubits...); }
    } // namespace ranges
  } // namespace gate
} // namespace ket


#endif // KET_GATE_PHASE_SHIFT_HPP
