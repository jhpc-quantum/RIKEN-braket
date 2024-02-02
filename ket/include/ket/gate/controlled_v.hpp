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
# include <ket/utility/loop_n.hpp>
# include <ket/utility/integer_exp2.hpp>
# ifndef NDEBUG
#   include <ket/utility/integer_log2.hpp>
# endif
# include <ket/utility/exp_i.hpp>
# include <ket/utility/meta/real_of.hpp>


namespace ket
{
  namespace gate
  {
    // controlled_v_coeff
    // V_{tc}(s)
    // V_{1,2}(s) (a_{00} |00> + a_{01} |01> + a_{10} |10> + a_{11} |11>)
    //   = a_{00} |00> + a_{01} |01> + [a_{10} (1+e^{is})/2 + a_{11} (1-e^{is})/2] |10> + [a_{10} (1-e^{is})/2 + a_{11} (1+e^{is})/2] |11>
    template <typename ParallelPolicy, typename RandomAccessIterator, typename Complex, typename StateInteger, typename BitInteger>
    inline void controlled_v_coeff(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      Complex const& phase_coefficient,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit)
    {
      static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
      static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");
      static_assert(
        (std::is_same<Complex, typename std::iterator_traits<RandomAccessIterator>::value_type>::value),
        "Complex must be the same to value_type of RandomAccessIterator");

      assert(
        ::ket::utility::integer_exp2<StateInteger>(target_qubit) < static_cast<StateInteger>(last - first)
        and ::ket::utility::integer_exp2<StateInteger>(control_qubit) < static_cast<StateInteger>(last - first)
        and target_qubit != control_qubit);
      assert(
        ::ket::utility::integer_exp2<StateInteger>(::ket::utility::integer_log2<BitInteger>(last - first))
        == static_cast<StateInteger>(last - first));

      using real_type = typename ::ket::utility::meta::real_of<Complex>::type;
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

      using ::ket::utility::loop_n;
      loop_n(
        parallel_policy,
        static_cast<StateInteger>(last - first) >> 2u,
        [first, &one_plus_phase_coefficient, &one_minus_phase_coefficient,
         target_qubit_mask, control_qubit_mask,
         lower_bits_mask, middle_bits_mask, upper_bits_mask](
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

    // C...V_{t,c,...,c'}(s)
    template <typename ParallelPolicy, typename RandomAccessIterator, typename Complex, typename StateInteger, typename BitInteger, typename... ControlQubits>
    inline void controlled_v_coeff(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      Complex const& phase_coefficient,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2, ControlQubits const... control_qubits)
    {
      constexpr auto num_control_qubits = static_cast<BitInteger>(sizeof...(ControlQubits) + 2u);
      constexpr auto num_indices = ::ket::utility::integer_exp2<std::size_t>(num_control_qubits + BitInteger{1u});

      using real_type = typename ::ket::utility::meta::real_of<Complex>::type;
      auto const one_plus_phase_coefficient = Complex{real_type{1}} + phase_coefficient;
      auto const one_minus_phase_coefficient = Complex{real_type{1}} - phase_coefficient;

      ::ket::gate::gate(
        parallel_policy, first, last,
        [&one_plus_phase_coefficient, &one_minus_phase_coefficient](
          RandomAccessIterator const first, std::array<StateInteger, num_indices> const& indices, int const)
        {
          // 0b11...10u
          auto const index0 = ((StateInteger{1u} << num_control_qubits) - StateInteger{1u}) << BitInteger{1u};
          // 0b11...11u
          auto const index1 = index0 bitor StateInteger{1u};

          auto const iter0 = first + indices[index0];
          auto const iter1 = first + indices[index1];
          auto const iter0_value = *iter0;

          using boost::math::constants::half;
          *iter0
            = half<real_type>()
              * (one_plus_phase_coefficient * iter0_value
                 + one_minus_phase_coefficient * (*iter1));
          *iter1
            = half<real_type>()
              * (one_minus_phase_coefficient * iter0_value
                 + one_plus_phase_coefficient * (*iter1));
        },
        target_qubit, control_qubit1, control_qubit2, control_qubits...);
    }

    template <typename RandomAccessIterator, typename Complex, typename StateInteger, typename BitInteger, typename... ControlQubits>
    inline void controlled_v_coeff(
      RandomAccessIterator const first, RandomAccessIterator const last,
      Complex const& phase_coefficient,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit, ControlQubits const... control_qubits)
    { ::ket::gate::controlled_v_coeff(::ket::utility::policy::make_sequential(), first, last, phase_coefficient, target_qubit, control_qubit, control_qubits...); }

    namespace ranges
    {
      template <typename ParallelPolicy, typename RandomAccessRange, typename Complex, typename StateInteger, typename BitInteger, typename... ControlQubits>
      inline RandomAccessRange& controlled_v_coeff(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& state, Complex const& phase_coefficient,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit, ControlQubits const... control_qubits)
      {
        ::ket::gate::controlled_v_coeff(parallel_policy, std::begin(state), std::end(state), phase_coefficient, target_qubit, control_qubit, control_qubits...);
        return state;
      }

      template <typename RandomAccessRange, typename Complex, typename StateInteger, typename BitInteger, typename... ControlQubits>
      inline RandomAccessRange& controlled_v_coeff(
        RandomAccessRange& state, Complex const& phase_coefficient,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit, ControlQubits const... control_qubits)
      { return ::ket::gate::ranges::controlled_v_coeff(::ket::utility::policy::make_sequential(), state, phase_coefficient, target_qubit, control_qubit, control_qubits...); }
    } // namespace ranges

    template <typename ParallelPolicy, typename RandomAccessIterator, typename Complex, typename StateInteger, typename BitInteger, typename... ControlQubits>
    inline void adj_controlled_v_coeff(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      Complex const& phase_coefficient,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit, ControlQubits const... control_qubits)
    {
      using std::conj;
      ::ket::gate::controlled_v_coeff(parallel_policy, first, last, conj(phase_coefficient), target_qubit, control_qubit, control_qubits...);
    }

    template <typename RandomAccessIterator, typename Complex, typename StateInteger, typename BitInteger, typename... ControlQubits>
    inline void adj_controlled_v_coeff(
      RandomAccessIterator const first, RandomAccessIterator const last,
      Complex const& phase_coefficient,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit, ControlQubits const... control_qubits)
    {
      using std::conj;
      ::ket::gate::controlled_v_coeff(first, last, conj(phase_coefficient), target_qubit, control_qubit, control_qubits...);
    }

    namespace ranges
    {
      template <typename ParallelPolicy, typename RandomAccessRange, typename Complex, typename StateInteger, typename BitInteger, typename... ControlQubits>
      inline RandomAccessRange& adj_controlled_v_coeff(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& state, Complex const& phase_coefficient,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit, ControlQubits const... control_qubits)
      {
        using std::conj;
        return ::ket::gate::ranges::controlled_v_coeff(parallel_policy, state, conj(phase_coefficient), target_qubit, control_qubit, control_qubits...);
      }

      template <typename RandomAccessRange, typename Complex, typename StateInteger, typename BitInteger, typename... ControlQubits>
      inline RandomAccessRange& adj_ontrolled_v_coeff(
        RandomAccessRange& state, Complex const& phase_coefficient,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit, ControlQubits const... control_qubits)
      {
        using std::conj;
        return ::ket::gate::ranges::controlled_v_coeff(state, conj(phase_coefficient), target_qubit, control_qubit, control_qubits...);
      }
    } // namespace ranges


    // controlled_v
    template <typename ParallelPolicy, typename RandomAccessIterator, typename Real, typename StateInteger, typename BitInteger, typename... ControlQubits>
    inline void controlled_v(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      Real const phase,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit, ControlQubits const... control_qubits)
    {
      using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
      ::ket::gate::controlled_v_coeff(parallel_policy, first, last, ::ket::utility::exp_i<complex_type>(phase), target_qubit, control_qubit, control_qubits...);
    }

    template <typename RandomAccessIterator, typename Real, typename StateInteger, typename BitInteger, typename... ControlQubits>
    inline void controlled_v(
      RandomAccessIterator const first, RandomAccessIterator const last,
      Real const phase,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit, ControlQubits const... control_qubits)
    {
      using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
      ::ket::gate::controlled_v_coeff(first, last, ::ket::utility::exp_i<complex_type>(phase), target_qubit, control_qubit, control_qubits...);
    }

    namespace ranges
    {
      template <typename ParallelPolicy, typename RandomAccessRange, typename Real, typename StateInteger, typename BitInteger, typename... ControlQubits>
      inline RandomAccessRange& controlled_v(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& state, Real const& phase,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit, ControlQubits const... control_qubits)
      {
        using complex_type = typename boost::range_value<RandomAccessRange>::type;
        return ::ket::gate::ranges::controlled_v_coeff(parallel_policy, state, ::ket::utility::exp_i<complex_type>(phase), target_qubit, control_qubit, control_qubits...);
      }

      template <typename RandomAccessRange, typename Real, typename StateInteger, typename BitInteger, typename... ControlQubits>
      inline RandomAccessRange& controlled_v(
        RandomAccessRange& state, Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit, ControlQubits const... control_qubits)
      {
        using complex_type = typename boost::range_value<RandomAccessRange>::type;
        return ::ket::gate::ranges::controlled_v_coeff(state, ::ket::utility::exp_i<complex_type>(phase), target_qubit, control_qubit, control_qubits...);
      }
    } // namespace ranges

    template <typename ParallelPolicy, typename RandomAccessIterator, typename Real, typename StateInteger, typename BitInteger, typename... ControlQubits>
    inline void adj_controlled_v(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      Real const phase,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit, ControlQubits const... control_qubits)
    { ::ket::gate::controlled_v(parallel_policy, first, last, -phase, target_qubit, control_qubit, control_qubits...); }

    template <typename RandomAccessIterator, typename Real, typename StateInteger, typename BitInteger, typename... ControlQubits>
    inline void adj_controlled_v(
      RandomAccessIterator const first, RandomAccessIterator const last,
      Real const phase,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit, ControlQubits const... control_qubits)
    { ::ket::gate::controlled_v(first, last, -phase, target_qubit, control_qubit, control_qubits...); }

    namespace ranges
    {
      template <typename ParallelPolicy, typename RandomAccessRange, typename Real, typename StateInteger, typename BitInteger, typename... ControlQubits>
      inline RandomAccessRange& adj_controlled_v(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& state, Real const& phase,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit, ControlQubits const... control_qubits)
      { return ::ket::gate::ranges::controlled_v(parallel_policy, state, -phase, target_qubit, control_qubit, control_qubits...); }

      template <typename RandomAccessRange, typename Real, typename StateInteger, typename BitInteger, typename... ControlQubits>
      inline RandomAccessRange& adj_controlled_v(
        RandomAccessRange& state, Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit, ControlQubits const... control_qubits)
      { return ::ket::gate::ranges::controlled_v(state, -phase, target_qubit, control_qubit, control_qubits...); }
    } // namespace ranges
  } // namespace gate
} // namespace ket


#endif // KET_GATE_CONTROLLED_V_HPP
