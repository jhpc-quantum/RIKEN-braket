#ifndef KET_GATE_CONTROLLED_V_HPP
# define KET_GATE_CONTROLLED_V_HPP

# include <cassert>
# include <complex>
# include <array>
# include <iterator>
# include <algorithm>
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
    // controlled_v_coeff
    // V_{tc}(theta) or CV_{tc}(theta)
    // V_{1,2}(theta) (a_{00} |00> + a_{01} |01> + a_{10} |10> + a_{11} |11>)
    //   = a_{00} |00> + a_{01} |01> + [a_{10} (1+e^{i theta})/2 + a_{11} (1-e^{i theta})/2] |10> + [a_{10} (1-e^{i theta})/2 + a_{11} (1+e^{i theta})/2] |11>
    template <typename ParallelPolicy, typename RandomAccessIterator, typename Complex, typename StateInteger, typename BitInteger>
    inline auto controlled_v_coeff(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      Complex const& phase_coefficient,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit)
    -> void
    {
      static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
      static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");
      static_assert(
        std::is_same<Complex, typename std::iterator_traits<RandomAccessIterator>::value_type>::value,
        "Complex must be the same to value_type of RandomAccessIterator");

      assert(
        ::ket::utility::integer_exp2<StateInteger>(target_qubit) < static_cast<StateInteger>(last - first)
        and ::ket::utility::integer_exp2<StateInteger>(control_qubit) < static_cast<StateInteger>(last - first)
        and target_qubit != control_qubit);
      assert(
        ::ket::utility::integer_exp2<StateInteger>(::ket::utility::integer_log2<BitInteger>(last - first))
        == static_cast<StateInteger>(last - first));

      using real_type = ::ket::utility::meta::real_t<Complex>;
      auto const one_plus_phase_coefficient = Complex{real_type{1}} + phase_coefficient;
      auto const one_minus_phase_coefficient = Complex{real_type{1}} - phase_coefficient;

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
        [first, &one_plus_phase_coefficient, &one_minus_phase_coefficient,
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

          using boost::math::constants::half;
          *control_on_iter
            = half<real_type>()
              * (one_plus_phase_coefficient * control_on_iter_value
                 + one_minus_phase_coefficient * (*target_control_on_iter));
          *target_control_on_iter
            = half<real_type>()
              * (one_minus_phase_coefficient * control_on_iter_value
                 + one_plus_phase_coefficient * (*target_control_on_iter));
        });
    }

    // C...CV_{tc...c'}(theta) or CnV_{tc...c'}(theta)
    template <typename ParallelPolicy, typename RandomAccessIterator, typename Complex, typename StateInteger, typename BitInteger, typename... ControlQubits>
    inline auto controlled_v_coeff(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      Complex const& phase_coefficient,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2, ControlQubits const... control_qubits)
    -> void
    {
      using real_type = ::ket::utility::meta::real_t<Complex>;
      auto const one_plus_phase_coefficient = Complex{real_type{1}} + phase_coefficient;
      auto const one_minus_phase_coefficient = Complex{real_type{1}} - phase_coefficient;

# ifndef KET_USE_BIT_MASKS_EXPLICITLY
      using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
      constexpr auto num_control_qubits = static_cast<BitInteger>(sizeof...(ControlQubits) + 2u);
      constexpr auto num_operated_qubits = num_control_qubits + BitInteger{1u};

      ::ket::gate::gate(
        parallel_policy, first, last,
        [&one_plus_phase_coefficient, &one_minus_phase_coefficient](
          auto const first, StateInteger const index_wo_qubits,
          std::array<qubit_type, num_operated_qubits> const& unsorted_qubits,
          std::array<qubit_type, num_operated_qubits + BitInteger{1u}> const& sorted_qubits_with_sentinel,
          int const)
        {
          // 0b11...10u
          constexpr auto index0 = ((std::size_t{1u} << num_control_qubits) - std::size_t{1u}) << std::size_t{1u};
          // 0b11...11u
          constexpr auto index1 = index0 bitor std::size_t{1u};

          auto const control_on_iter = first + ::ket::gate::utility::index_with_qubits(index_wo_qubits, index0, unsorted_qubits, sorted_qubits_with_sentinel);
          auto const target_control_on_iter = first + ::ket::gate::utility::index_with_qubits(index_wo_qubits, index1, unsorted_qubits, sorted_qubits_with_sentinel);
          auto const control_on_iter_value = *control_on_iter;

          using boost::math::constants::half;
          *control_on_iter
            = half<real_type>()
              * (one_plus_phase_coefficient * control_on_iter_value
                 + one_minus_phase_coefficient * (*target_control_on_iter));
          *target_control_on_iter
            = half<real_type>()
              * (one_minus_phase_coefficient * control_on_iter_value
                 + one_plus_phase_coefficient * (*target_control_on_iter));
        },
        target_qubit, control_qubit1, control_qubit2, control_qubits...);
# else // KET_USE_BIT_MASKS_EXPLICITLY
      constexpr auto num_control_qubits = static_cast<BitInteger>(sizeof...(ControlQubits) + 2u);
      constexpr auto num_operated_qubits = num_control_qubits + BitInteger{1u};

      ::ket::gate::gate(
        parallel_policy, first, last,
        [&one_plus_phase_coefficient, &one_minus_phase_coefficient](
          auto const first, StateInteger const index_wo_qubits,
          std::array<StateInteger, num_operated_qubits> const& qubit_masks,
          std::array<StateInteger, num_operated_qubits + 1u> const& index_masks,
          int const)
        {
          // 0b11...10u
          constexpr auto index0 = ((std::size_t{1u} << num_control_qubits) - std::size_t{1u}) << std::size_t{1u};
          // 0b11...11u
          constexpr auto index1 = index0 bitor std::size_t{1u};

          auto const control_on_iter = first + ::ket::gate::utility::index_with_qubits(index_wo_qubits, index0, qubit_masks, index_masks);
          auto const target_control_on_iter = first + ::ket::gate::utility::index_with_qubits(index_wo_qubits, index1, qubit_masks, index_masks);
          auto const control_on_iter_value = *control_on_iter;

          using boost::math::constants::half;
          *control_on_iter
            = half<real_type>()
              * (one_plus_phase_coefficient * control_on_iter_value
                 + one_minus_phase_coefficient * (*target_control_on_iter));
          *target_control_on_iter
            = half<real_type>()
              * (one_minus_phase_coefficient * control_on_iter_value
                 + one_plus_phase_coefficient * (*target_control_on_iter));
        },
        target_qubit, control_qubit1, control_qubit2, control_qubits...);
# endif // KET_USE_BIT_MASKS_EXPLICITLY
    }

    template <typename RandomAccessIterator, typename Complex, typename StateInteger, typename BitInteger, typename... ControlQubits>
    inline auto controlled_v_coeff(
      RandomAccessIterator const first, RandomAccessIterator const last,
      Complex const& phase_coefficient,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit, ControlQubits const... control_qubits)
    -> void
    { ::ket::gate::controlled_v_coeff(::ket::utility::policy::make_sequential(), first, last, phase_coefficient, target_qubit, control_qubit, control_qubits...); }

    namespace ranges
    {
      template <typename ParallelPolicy, typename RandomAccessRange, typename Complex, typename StateInteger, typename BitInteger, typename... ControlQubits>
      inline auto controlled_v_coeff(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& state, Complex const& phase_coefficient,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit, ControlQubits const... control_qubits)
      -> RandomAccessRange&
      {
        using std::begin;
        using std::end;
        ::ket::gate::controlled_v_coeff(parallel_policy, begin(state), end(state), phase_coefficient, target_qubit, control_qubit, control_qubits...);
        return state;
      }

      template <typename RandomAccessRange, typename Complex, typename StateInteger, typename BitInteger, typename... ControlQubits>
      inline auto controlled_v_coeff(
        RandomAccessRange& state, Complex const& phase_coefficient,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit, ControlQubits const... control_qubits)
      -> RandomAccessRange&
      { return ::ket::gate::ranges::controlled_v_coeff(::ket::utility::policy::make_sequential(), state, phase_coefficient, target_qubit, control_qubit, control_qubits...); }
    } // namespace ranges

    template <typename ParallelPolicy, typename RandomAccessIterator, typename Complex, typename StateInteger, typename BitInteger, typename... ControlQubits>
    inline auto adj_controlled_v_coeff(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      Complex const& phase_coefficient,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit, ControlQubits const... control_qubits)
    -> void
    { using std::conj; ::ket::gate::controlled_v_coeff(parallel_policy, first, last, conj(phase_coefficient), target_qubit, control_qubit, control_qubits...); }

    template <typename RandomAccessIterator, typename Complex, typename StateInteger, typename BitInteger, typename... ControlQubits>
    inline auto adj_controlled_v_coeff(
      RandomAccessIterator const first, RandomAccessIterator const last,
      Complex const& phase_coefficient,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit, ControlQubits const... control_qubits)
    -> void
    { using std::conj; ::ket::gate::controlled_v_coeff(first, last, conj(phase_coefficient), target_qubit, control_qubit, control_qubits...); }

    namespace ranges
    {
      template <typename ParallelPolicy, typename RandomAccessRange, typename Complex, typename StateInteger, typename BitInteger, typename... ControlQubits>
      inline auto adj_controlled_v_coeff(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& state, Complex const& phase_coefficient,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit, ControlQubits const... control_qubits)
      -> RandomAccessRange&
      { using std::conj; return ::ket::gate::ranges::controlled_v_coeff(parallel_policy, state, conj(phase_coefficient), target_qubit, control_qubit, control_qubits...); }

      template <typename RandomAccessRange, typename Complex, typename StateInteger, typename BitInteger, typename... ControlQubits>
      inline auto adj_ontrolled_v_coeff(
        RandomAccessRange& state, Complex const& phase_coefficient,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit, ControlQubits const... control_qubits)
      -> RandomAccessRange&
      { using std::conj; return ::ket::gate::ranges::controlled_v_coeff(state, conj(phase_coefficient), target_qubit, control_qubit, control_qubits...); }
    } // namespace ranges


    // controlled_v
    template <typename ParallelPolicy, typename RandomAccessIterator, typename Real, typename StateInteger, typename BitInteger, typename... ControlQubits>
    inline auto controlled_v(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      Real const phase,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit, ControlQubits const... control_qubits)
    -> void
    {
      using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
      ::ket::gate::controlled_v_coeff(parallel_policy, first, last, ::ket::utility::exp_i<complex_type>(phase), target_qubit, control_qubit, control_qubits...);
    }

    template <typename RandomAccessIterator, typename Real, typename StateInteger, typename BitInteger, typename... ControlQubits>
    inline auto controlled_v(
      RandomAccessIterator const first, RandomAccessIterator const last,
      Real const phase,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit, ControlQubits const... control_qubits)
    -> void
    {
      using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
      ::ket::gate::controlled_v_coeff(first, last, ::ket::utility::exp_i<complex_type>(phase), target_qubit, control_qubit, control_qubits...);
    }

    namespace ranges
    {
      template <typename ParallelPolicy, typename RandomAccessRange, typename Real, typename StateInteger, typename BitInteger, typename... ControlQubits>
      inline auto controlled_v(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& state, Real const& phase,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit, ControlQubits const... control_qubits)
      -> RandomAccessRange&
      {
        using complex_type = ::ket::utility::meta::range_value_t<RandomAccessRange>;
        return ::ket::gate::ranges::controlled_v_coeff(parallel_policy, state, ::ket::utility::exp_i<complex_type>(phase), target_qubit, control_qubit, control_qubits...);
      }

      template <typename RandomAccessRange, typename Real, typename StateInteger, typename BitInteger, typename... ControlQubits>
      inline auto controlled_v(
        RandomAccessRange& state, Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit, ControlQubits const... control_qubits)
      -> RandomAccessRange&
      {
        using complex_type = ::ket::utility::meta::range_value_t<RandomAccessRange>;
        return ::ket::gate::ranges::controlled_v_coeff(state, ::ket::utility::exp_i<complex_type>(phase), target_qubit, control_qubit, control_qubits...);
      }
    } // namespace ranges

    template <typename ParallelPolicy, typename RandomAccessIterator, typename Real, typename StateInteger, typename BitInteger, typename... ControlQubits>
    inline auto adj_controlled_v(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      Real const phase,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit, ControlQubits const... control_qubits)
    -> void
    { ::ket::gate::controlled_v(parallel_policy, first, last, -phase, target_qubit, control_qubit, control_qubits...); }

    template <typename RandomAccessIterator, typename Real, typename StateInteger, typename BitInteger, typename... ControlQubits>
    inline auto adj_controlled_v(
      RandomAccessIterator const first, RandomAccessIterator const last,
      Real const phase,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit, ControlQubits const... control_qubits)
    -> void
    { ::ket::gate::controlled_v(first, last, -phase, target_qubit, control_qubit, control_qubits...); }

    namespace ranges
    {
      template <typename ParallelPolicy, typename RandomAccessRange, typename Real, typename StateInteger, typename BitInteger, typename... ControlQubits>
      inline auto adj_controlled_v(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& state, Real const& phase,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit, ControlQubits const... control_qubits)
      -> RandomAccessRange&
      { return ::ket::gate::ranges::controlled_v(parallel_policy, state, -phase, target_qubit, control_qubit, control_qubits...); }

      template <typename RandomAccessRange, typename Real, typename StateInteger, typename BitInteger, typename... ControlQubits>
      inline auto adj_controlled_v(
        RandomAccessRange& state, Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit, ControlQubits const... control_qubits)
      -> RandomAccessRange&
      { return ::ket::gate::ranges::controlled_v(state, -phase, target_qubit, control_qubit, control_qubits...); }
    } // namespace ranges
  } // namespace gate
} // namespace ket


#endif // KET_GATE_CONTROLLED_V_HPP
