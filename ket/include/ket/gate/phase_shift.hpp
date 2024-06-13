#ifndef KET_GATE_PHASE_SHIFT_HPP
# define KET_GATE_PHASE_SHIFT_HPP

# include <cassert>
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
    // phase_shift_coeff
    // U1_i(theta)
    // U1_1(theta) (a_0 |0> + a_1 |1>) = a_0 |0> + e^{i theta} a_1 |1>
    template <typename ParallelPolicy, typename RandomAccessIterator, typename Complex, typename StateInteger, typename BitInteger>
    inline void phase_shift_coeff(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
      ::ket::qubit<StateInteger, BitInteger> const qubit)
    {
      static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
      static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");
      static_assert(
        (std::is_same<Complex, typename std::iterator_traits<RandomAccessIterator>::value_type>::value),
        "Complex must be the same to value_type of RandomAccessIterator");

      assert(::ket::utility::integer_exp2<StateInteger>(qubit) < static_cast<StateInteger>(last - first));
      assert(
        ::ket::utility::integer_exp2<StateInteger>(::ket::utility::integer_log2<BitInteger>(last - first))
        == static_cast<StateInteger>(last - first));

      auto const qubit_mask = ::ket::utility::integer_exp2<StateInteger>(qubit);
      auto const lower_bits_mask = qubit_mask - StateInteger{1u};
      auto const upper_bits_mask = compl lower_bits_mask;

      using ::ket::utility::loop_n;
      loop_n(
        parallel_policy,
        static_cast<StateInteger>(last - first) >> 1u,
        [first, &phase_coefficient, qubit_mask, lower_bits_mask, upper_bits_mask](StateInteger const value_wo_qubit, int const)
        {
          // xxxxx1xxxxxx
          auto const one_index
            = ((value_wo_qubit bitand upper_bits_mask) << 1u)
              bitor (value_wo_qubit bitand lower_bits_mask) bitor qubit_mask;
          *(first + one_index) *= phase_coefficient;
        });
    }

    // CU1_{tc}(theta) or C1U1_{tc}(theta)
    // CU1_{1,2}(theta) (a_{00} |00> + a_{01} |01> + a_{10} |10> + a{11} |11>)
    //   = a_{00} |00> + a_{01} |01> + a_{10} |10> + e^{i thta} a_{11} |11>
    template <typename ParallelPolicy, typename RandomAccessIterator, typename Complex, typename StateInteger, typename BitInteger>
    inline void phase_shift_coeff(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
      ::ket::qubit<StateInteger, BitInteger> const target_qubit,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit)
    {
      static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
      static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");
      static_assert(
        (std::is_same<Complex, typename std::iterator_traits<RandomAccessIterator>::value_type>::value),
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

      using ::ket::utility::loop_n;
      loop_n(
        parallel_policy,
        static_cast<StateInteger>(last - first) >> 2u,
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
    inline void phase_shift_coeff(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
      ::ket::qubit<StateInteger, BitInteger> const target_qubit,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2, ControlQubits const... control_qubits)
    {
      static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
      static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");
      static_assert(
        (std::is_same<Complex, typename std::iterator_traits<RandomAccessIterator>::value_type>::value),
        "Complex must be the same to value_type of RandomAccessIterator");
      assert(
        ::ket::utility::integer_exp2<StateInteger>(::ket::utility::integer_log2<BitInteger>(last - first))
        == static_cast<StateInteger>(last - first));

      constexpr auto num_control_qubits = static_cast<BitInteger>(sizeof...(ControlQubits) + 2u);
      constexpr auto num_qubits = num_control_qubits + BitInteger{1u};
      constexpr auto num_indices = ::ket::utility::integer_exp2<std::size_t>(num_qubits);

      // 0b11...11u
      constexpr auto indices_index = ((std::size_t{1u} << num_qubits) - std::size_t{1u});

      ::ket::gate::gate(
        parallel_policy, first, last,
        [&phase_coefficient](RandomAccessIterator const first, std::array<StateInteger, num_indices> const& indices, int const)
        { *(first + indices[indices_index]) *= phase_coefficient; },
        target_qubit, control_qubit1, control_qubit2, control_qubits...);
    }

    template <typename RandomAccessIterator, typename Complex, typename StateInteger, typename BitInteger, typename... ControlQubits>
    inline void phase_shift_coeff(
      RandomAccessIterator const first, RandomAccessIterator const last,
      Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
      ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
    { ::ket::gate::phase_shift_coeff(::ket::utility::policy::make_sequential(), first, last, phase_coefficient, target_qubit, control_qubits...); }

    namespace ranges
    {
      template <typename ParallelPolicy, typename RandomAccessRange, typename Complex, typename StateInteger, typename BitInteger, typename... ControlQubits>
      inline RandomAccessRange& phase_shift_coeff(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& state,
        Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
        ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
      {
        ::ket::gate::phase_shift_coeff(parallel_policy, std::begin(state), std::end(state), phase_coefficient, target_qubit, control_qubits...);
        return state;
      }

      template <typename RandomAccessRange, typename Complex, typename StateInteger, typename BitInteger, typename... ControlQubits>
      inline RandomAccessRange& phase_shift_coeff(
        RandomAccessRange& state,
        Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
        ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
      { return ::ket::gate::ranges::phase_shift_coeff(::ket::utility::policy::make_sequential(), state, phase_coefficient, target_qubit, control_qubits...); }
    } // namespace ranges

    template <typename ParallelPolicy, typename RandomAccessIterator, typename Complex, typename StateInteger, typename BitInteger, typename... ControlQubits>
    inline void adj_phase_shift_coeff(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
      ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
    {
      using std::conj;
      ::ket::gate::phase_shift_coeff(parallel_policy, first, last, conj(phase_coefficient), target_qubit, control_qubits...);
    }

    template <typename RandomAccessIterator, typename Complex, typename StateInteger, typename BitInteger, typename... ControlQubits>
    inline void adj_phase_shift_coeff(
      RandomAccessIterator const first, RandomAccessIterator const last,
      Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
      ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
    { using std::conj; ::ket::gate::phase_shift_coeff(first, last, conj(phase_coefficient), target_qubit, control_qubits...); }

    namespace ranges
    {
      template <typename ParallelPolicy, typename RandomAccessRange, typename Complex, typename StateInteger, typename BitInteger, typename... ControlQubits>
      inline RandomAccessRange& adj_phase_shift_coeff(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& state,
        Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
        ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
      { using std::conj; return ::ket::gate::ranges::phase_shift_coeff(parallel_policy, state, conj(phase_coefficient), target_qubit, control_qubits...); }

      template <typename RandomAccessRange, typename Complex, typename StateInteger, typename BitInteger, typename... ControlQubits>
      inline RandomAccessRange& adj_phase_shift_coeff(
        RandomAccessRange& state,
        Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
        ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
      { using std::conj; return ::ket::gate::ranges::phase_shift_coeff(state, conj(phase_coefficient), target_qubit, control_qubits...); }
    } // namespace ranges

    // phase_shift
    template <typename ParallelPolicy, typename RandomAccessIterator, typename Real, typename StateInteger, typename BitInteger, typename... ControlQubits>
    inline void phase_shift(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      Real const phase, ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
    {
      using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
      ::ket::gate::phase_shift_coeff(parallel_policy, first, last, ::ket::utility::exp_i<complex_type>(phase), target_qubit, control_qubits...);
    }

    template <typename RandomAccessIterator, typename Real, typename StateInteger, typename BitInteger, typename... ControlQubits>
    inline void phase_shift(
      RandomAccessIterator const first, RandomAccessIterator const last,
      Real const phase, ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
    {
      using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
      ::ket::gate::phase_shift_coeff(first, last, ::ket::utility::exp_i<complex_type>(phase), target_qubit, control_qubits...);
    }

    namespace ranges
    {
      template <typename ParallelPolicy, typename RandomAccessRange, typename Real, typename StateInteger, typename BitInteger, typename... ControlQubits>
      inline RandomAccessRange& phase_shift(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& state, Real const phase, ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
      {
        using complex_type = typename boost::range_value<RandomAccessRange>::type;
        return ::ket::gate::ranges::phase_shift_coeff(parallel_policy, state, ::ket::utility::exp_i<complex_type>(phase), target_qubit, control_qubits...);
      }

      template <typename RandomAccessRange, typename Real, typename StateInteger, typename BitInteger, typename... ControlQubits>
      inline RandomAccessRange& phase_shift(
        RandomAccessRange& state, Real const phase, ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
      {
        using complex_type = typename boost::range_value<RandomAccessRange>::type;
        return ::ket::gate::ranges::phase_shift_coeff(state, ::ket::utility::exp_i<complex_type>(phase), target_qubit, control_qubits...);
      }
    } // namespace ranges

    template <typename ParallelPolicy, typename RandomAccessIterator, typename Real, typename StateInteger, typename BitInteger, typename... ControlQubits>
    inline void adj_phase_shift(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      Real const phase, ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
    { ::ket::gate::phase_shift(parallel_policy, first, last, -phase, target_qubit, control_qubits...); }

    template <typename RandomAccessIterator, typename Real, typename StateInteger, typename BitInteger, typename... ControlQubits>
    inline void adj_phase_shift(
      RandomAccessIterator const first, RandomAccessIterator const last,
      Real const phase, ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
    { ::ket::gate::phase_shift(first, last, -phase, target_qubit, control_qubits...); }

    namespace ranges
    {
      template <typename ParallelPolicy, typename RandomAccessRange, typename Real, typename StateInteger, typename BitInteger, typename... ControlQubits>
      inline RandomAccessRange& adj_phase_shift(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& state, Real const phase, ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
      { return ::ket::gate::ranges::phase_shift(parallel_policy, state, -phase, target_qubit, control_qubits...); }

      template <typename RandomAccessRange, typename Real, typename StateInteger, typename BitInteger, typename... ControlQubits>
      inline RandomAccessRange& adj_phase_shift(
        RandomAccessRange& state, Real const phase, ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
      { return ::ket::gate::ranges::phase_shift(state, -phase, target_qubit, control_qubits...); }
    } // namespace ranges

    // generalized phase_shift
    // U2_i(theta, theta')
    // U2_1(theta, theta') (a_0 |0> + a_1 |1>)
    //   = (a_0 - e^{i theta'} a_1)/sqrt(2) |0> + (e^{i theta} a_0 + e^{i(theta + theta')} a_1)/sqrt(2) |1>
    template <
      typename ParallelPolicy, typename RandomAccessIterator, typename Real,
      typename StateInteger, typename BitInteger>
    inline void phase_shift2(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      Real const phase1, Real const phase2,
      ::ket::qubit<StateInteger, BitInteger> const qubit)
    {
      static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
      static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");

      using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
      static_assert(
        (std::is_same<Real, typename ::ket::utility::meta::real_of<complex_type>::type>::value),
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

      using ::ket::utility::loop_n;
      loop_n(
        parallel_policy,
        static_cast<StateInteger>(last - first) >> 1u,
        [first, &modified_phase_coefficient1, &phase_coefficient2, qubit_mask, lower_bits_mask, upper_bits_mask](
          StateInteger const value_wo_qubit, int const)
        {
          // xxxxx0xxxxxx
          auto const zero_index
            = ((value_wo_qubit bitand upper_bits_mask) << 1u)
              bitor (value_wo_qubit bitand lower_bits_mask);
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
    inline void phase_shift2(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      Real const phase1, Real const phase2,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit)
    {
      static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
      static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");

      using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
      static_assert(
        (std::is_same<Real, typename ::ket::utility::meta::real_of<complex_type>::type>::value),
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

      using ::ket::utility::loop_n;
      loop_n(
        parallel_policy,
        static_cast<StateInteger>(last - first) >> 2u,
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
    inline void phase_shift2(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      Real const phase1, Real const phase2,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2, ControlQubits const... control_qubits)
    {
      static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
      static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");

      using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
      static_assert(
        (std::is_same<Real, typename ::ket::utility::meta::real_of<complex_type>::type>::value),
        "Real must be the same to \"real part\" of value_type of RandomAccessIterator");

      assert(::ket::utility::integer_exp2<StateInteger>(target_qubit) < static_cast<StateInteger>(last - first));
      assert(::ket::utility::integer_exp2<StateInteger>(control_qubit1) < static_cast<StateInteger>(last - first));
      assert(::ket::utility::integer_exp2<StateInteger>(control_qubit2) < static_cast<StateInteger>(last - first));
      assert(target_qubit != control_qubit1);
      assert(target_qubit != control_qubit2);
      assert(
        ::ket::utility::integer_exp2<StateInteger>(::ket::utility::integer_log2<BitInteger>(last - first))
        == static_cast<StateInteger>(last - first));

      auto const phase_coefficient1 = ::ket::utility::exp_i<complex_type>(phase1);
      auto const phase_coefficient2 = ::ket::utility::exp_i<complex_type>(phase2);

      using boost::math::constants::one_div_root_two;
      auto const modified_phase_coefficient1 = one_div_root_two<Real>() * phase_coefficient1;

      constexpr auto num_control_qubits = static_cast<BitInteger>(sizeof...(ControlQubits) + 2u);
      constexpr auto num_qubits = num_control_qubits + BitInteger{1u};
      constexpr auto num_indices = ::ket::utility::integer_exp2<std::size_t>(num_qubits);

      // 0b11...10u
      constexpr auto indices_index0 = ((std::size_t{1u} << num_control_qubits) - std::size_t{1u}) << std::size_t{1u};
      // 0b11...11u
      constexpr auto indices_index1 = indices_index0 bitor std::size_t{1u};

      ::ket::gate::gate(
        parallel_policy, first, last,
        [&modified_phase_coefficient1, &phase_coefficient2](
          RandomAccessIterator const first, std::array<StateInteger, num_indices> const& indices, int const)
        {
          auto const control_on_iter = first + indices[indices_index0];
          auto const target_control_on_iter = first + indices[indices_index1];
          auto const control_on_iter_value = *control_on_iter;

          *control_on_iter -= phase_coefficient2 * *target_control_on_iter;
          *control_on_iter *= one_div_root_two<Real>();
          *target_control_on_iter *= phase_coefficient2;
          *target_control_on_iter += control_on_iter_value;
          *target_control_on_iter *= modified_phase_coefficient1;
        },
        target_qubit, control_qubit1, control_qubit2, control_qubits...);
    }

    template <typename RandomAccessIterator, typename Real, typename StateInteger, typename BitInteger, typename... ControlQubits>
    inline void phase_shift2(
      RandomAccessIterator const first, RandomAccessIterator const last,
      Real const phase1, Real const phase2, ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
    { ::ket::gate::phase_shift2(::ket::utility::policy::make_sequential(), first, last, phase1, phase2, target_qubit, control_qubits...); }

    namespace ranges
    {
      template <typename ParallelPolicy, typename RandomAccessRange, typename Real, typename StateInteger, typename BitInteger, typename... ControlQubits>
      inline RandomAccessRange& phase_shift2(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& state, Real const phase1, Real const phase2,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
      {
        ::ket::gate::phase_shift2(parallel_policy, std::begin(state), std::end(state), phase1, phase2, target_qubit, control_qubits...);
        return state;
      }

      template <typename RandomAccessRange, typename Real, typename StateInteger, typename BitInteger, typename... ControlQubits>
      inline RandomAccessRange& phase_shift2(
        RandomAccessRange& state, Real const phase1, Real const phase2,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
      { return ::ket::gate::ranges::phase_shift2(::ket::utility::policy::make_sequential(), state, phase1, phase2, target_qubit, control_qubits...); }
    } // namespace ranges

    // U2+_i(theta, theta')
    // U2+_1(theta, theta') (a_0 |0> + a_1 |1>)
    //   = (a_0 + e^{-i theta} a_1)/sqrt(2) |0>
    //     + (-e^{-i theta'} a_0 + e^{-i(theta + theta')} a_1)/sqrt(2) |1>
    template <
      typename ParallelPolicy, typename RandomAccessIterator, typename Real,
      typename StateInteger, typename BitInteger>
    inline void adj_phase_shift2(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      Real const phase1, Real const phase2,
      ::ket::qubit<StateInteger, BitInteger> const qubit)
    {
      static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
      static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");

      using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
      static_assert(
        (std::is_same<Real, typename ::ket::utility::meta::real_of<complex_type>::type>::value),
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

      using ::ket::utility::loop_n;
      loop_n(
        parallel_policy,
        static_cast<StateInteger>(last - first) >> 1u,
        [first, &phase_coefficient1, &modified_phase_coefficient2, qubit_mask, lower_bits_mask, upper_bits_mask](
          StateInteger const value_wo_qubit, int const)
        {
          // xxxxx0xxxxxx
          auto const zero_index
            = ((value_wo_qubit bitand upper_bits_mask) << 1u)
              bitor (value_wo_qubit bitand lower_bits_mask);
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
    inline void adj_phase_shift2(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      Real const phase1, Real const phase2,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit)
    {
      static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
      static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");

      using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
      static_assert(
        (std::is_same<Real, typename ::ket::utility::meta::real_of<complex_type>::type>::value),
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

      using ::ket::utility::loop_n;
      loop_n(
        parallel_policy,
        static_cast<StateInteger>(last - first) >> 2u,
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
    inline void adj_phase_shift2(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      Real const phase1, Real const phase2,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2, ControlQubits const... control_qubits)
    {
      static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
      static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");

      using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
      static_assert(
        (std::is_same<Real, typename ::ket::utility::meta::real_of<complex_type>::type>::value),
        "Real must be the same to \"real part\" of value_type of RandomAccessIterator");

      assert(::ket::utility::integer_exp2<StateInteger>(target_qubit) < static_cast<StateInteger>(last - first));
      assert(::ket::utility::integer_exp2<StateInteger>(control_qubit1) < static_cast<StateInteger>(last - first));
      assert(::ket::utility::integer_exp2<StateInteger>(control_qubit2) < static_cast<StateInteger>(last - first));
      assert(target_qubit != control_qubit1);
      assert(target_qubit != control_qubit2);
      assert(
        ::ket::utility::integer_exp2<StateInteger>(::ket::utility::integer_log2<BitInteger>(last - first))
        == static_cast<StateInteger>(last - first));

      auto const phase_coefficient1 = ::ket::utility::exp_i<complex_type>(-phase1);
      auto const phase_coefficient2 = ::ket::utility::exp_i<complex_type>(-phase2);

      using boost::math::constants::one_div_root_two;
      auto const modified_phase_coefficient2 = one_div_root_two<Real>() * phase_coefficient2;

      constexpr auto num_control_qubits = static_cast<BitInteger>(sizeof...(ControlQubits) + 2u);
      constexpr auto num_qubits = num_control_qubits + BitInteger{1u};
      constexpr auto num_indices = ::ket::utility::integer_exp2<std::size_t>(num_qubits);

      // 0b11...10u
      constexpr auto indices_index0 = ((std::size_t{1u} << num_control_qubits) - std::size_t{1u}) << std::size_t{1u};
      // 0b11...11u
      constexpr auto indices_index1 = indices_index0 bitor std::size_t{1u};

      ::ket::gate::gate(
        parallel_policy, first, last,
        [&phase_coefficient1, &modified_phase_coefficient2](
          RandomAccessIterator const first, std::array<StateInteger, num_indices> const& indices, int const)
        {
          auto const control_on_iter = first + indices[indices_index0];
          auto const target_control_on_iter = first + indices[indices_index1];
          auto const control_on_iter_value = *control_on_iter;

          *control_on_iter += phase_coefficient1 * *target_control_on_iter;
          *control_on_iter *= one_div_root_two<Real>();
          *target_control_on_iter *= phase_coefficient1;
          *target_control_on_iter -= control_on_iter_value;
          *target_control_on_iter *= modified_phase_coefficient2;
        },
        target_qubit, control_qubit1, control_qubit2, control_qubits...);
    }

    template <typename RandomAccessIterator, typename Real, typename StateInteger, typename BitInteger, typename... ControlQubits>
    inline void adj_phase_shift2(
      RandomAccessIterator const first, RandomAccessIterator const last,
      Real const phase1, Real const phase2, ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
    { ::ket::gate::adj_phase_shift2(::ket::utility::policy::make_sequential(), first, last, phase1, phase2, target_qubit, control_qubits...); }

    namespace ranges
    {
      template <typename ParallelPolicy, typename RandomAccessRange, typename Real, typename StateInteger, typename BitInteger, typename... ControlQubits>
      inline RandomAccessRange& adj_phase_shift2(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& state, Real const phase1, Real const phase2,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
      {
        ::ket::gate::adj_phase_shift2(parallel_policy, std::begin(state), std::end(state), phase1, phase2, target_qubit, control_qubits...);
        return state;
      }

      template <typename RandomAccessRange, typename Real, typename StateInteger, typename BitInteger, typename... ControlQubits>
      inline RandomAccessRange& adj_phase_shift2(
        RandomAccessRange& state, Real const phase1, Real const phase2,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
      { return ::ket::gate::ranges::adj_phase_shift2(::ket::utility::policy::make_sequential(), state, phase1, phase2, target_qubit, control_qubits...); }
    } // namespace ranges

    // U3_i(theta, theta', theta'')
    // U3_1(theta, theta', theta'') (a_0 |0> + a_1 |1>)
    //   = (cos(theta/2) a_0 - e^{i theta''} sin(theta/2) a_1) |0>
    //     + (e^{i theta'} sin(theta/2) a_0 + e^{i(theta' + theta'')} cos(theta/2) a_1) |1>
    template <typename ParallelPolicy, typename RandomAccessIterator, typename Real, typename StateInteger, typename BitInteger>
    inline void phase_shift3(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      Real const phase1, Real const phase2, Real const phase3, ::ket::qubit<StateInteger, BitInteger> const qubit)
    {
      static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
      static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");

      using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
      static_assert(
        (std::is_same<Real, typename ::ket::utility::meta::real_of<complex_type>::type>::value),
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

      using ::ket::utility::loop_n;
      loop_n(
        parallel_policy,
        static_cast<StateInteger>(last - first) >> 1u,
        [first, sine, cosine, &phase_coefficient2, &sine_phase_coefficient3, &cosine_phase_coefficient3, qubit_mask, lower_bits_mask, upper_bits_mask](
          StateInteger const value_wo_qubit, int const)
        {
          // xxxxx0xxxxxx
          auto const zero_index
            = ((value_wo_qubit bitand upper_bits_mask) << 1u)
              bitor (value_wo_qubit bitand lower_bits_mask);
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
    inline void phase_shift3(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      Real const phase1, Real const phase2, Real const phase3,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit)
    {
      static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
      static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");

      using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
      static_assert(
        (std::is_same<Real, typename ::ket::utility::meta::real_of<complex_type>::type>::value),
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

      using ::ket::utility::loop_n;
      loop_n(
        parallel_policy,
        static_cast<StateInteger>(last - first) >> 2u,
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
    inline void phase_shift3(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      Real const phase1, Real const phase2, Real const phase3,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2, ControlQubits const... control_qubits)
    {
      static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
      static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");

      using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
      static_assert(
        (std::is_same<Real, typename ::ket::utility::meta::real_of<complex_type>::type>::value),
        "Real must be the same to \"real part\" of value_type of RandomAccessIterator");

      assert(::ket::utility::integer_exp2<StateInteger>(target_qubit) < static_cast<StateInteger>(last - first));
      assert(::ket::utility::integer_exp2<StateInteger>(control_qubit1) < static_cast<StateInteger>(last - first));
      assert(::ket::utility::integer_exp2<StateInteger>(control_qubit2) < static_cast<StateInteger>(last - first));
      assert(target_qubit != control_qubit1);
      assert(target_qubit != control_qubit2);
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

      constexpr auto num_control_qubits = static_cast<BitInteger>(sizeof...(ControlQubits) + 2u);
      constexpr auto num_qubits = num_control_qubits + BitInteger{1u};
      constexpr auto num_indices = ::ket::utility::integer_exp2<std::size_t>(num_qubits);

      // 0b11...10u
      constexpr auto indices_index0 = ((std::size_t{1u} << num_control_qubits) - std::size_t{1u}) << std::size_t{1u};
      // 0b11...11u
      constexpr auto indices_index1 = indices_index0 bitor std::size_t{1u};

      ::ket::gate::gate(
        parallel_policy, first, last,
        [sine, cosine, &phase_coefficient2, &sine_phase_coefficient3, &cosine_phase_coefficient3](
          RandomAccessIterator const first, std::array<StateInteger, num_indices> const& indices, int const)
        {
          auto const control_on_iter = first + indices[indices_index0];
          auto const target_control_on_iter = first + indices[indices_index1];
          auto const control_on_iter_value = *control_on_iter;

          *control_on_iter *= cosine;
          *control_on_iter -= sine_phase_coefficient3 * *target_control_on_iter;
          *target_control_on_iter *= cosine_phase_coefficient3;
          *target_control_on_iter += sine * control_on_iter_value;
          *target_control_on_iter *= phase_coefficient2;
        },
        target_qubit, control_qubit1, control_qubit2, control_qubits...);
    }

    template <typename RandomAccessIterator, typename Real, typename StateInteger, typename BitInteger, typename... ControlQubits>
    inline void phase_shift3(
      RandomAccessIterator const first, RandomAccessIterator const last,
      Real const phase1, Real const phase2, Real const phase3,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
    { ::ket::gate::phase_shift3(::ket::utility::policy::make_sequential(), first, last, phase1, phase2, phase3, target_qubit, control_qubits...); }

    namespace ranges
    {
      template <typename ParallelPolicy, typename RandomAccessRange, typename Real, typename StateInteger, typename BitInteger, typename... ControlQubits>
      inline RandomAccessRange& phase_shift3(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& state, Real const phase1, Real const phase2, Real const phase3,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
      {
        ::ket::gate::phase_shift3(parallel_policy, std::begin(state), std::end(state), phase1, phase2, phase3, target_qubit, control_qubits...);
        return state;
      }

      template <typename RandomAccessRange, typename Real, typename StateInteger, typename BitInteger, typename... ControlQubits>
      inline RandomAccessRange& phase_shift3(
        RandomAccessRange& state, Real const phase1, Real const phase2, Real const phase3,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
      { return ::ket::gate::ranges::phase_shift3(::ket::utility::policy::make_sequential(), state, phase1, phase2, phase3, target_qubit, control_qubits...); }
    } // namespace ranges

    // U3+_i(theta, theta', theta'')
    // U3+_1(theta, theta', theta'') (a_0 |0> + a_1 |1>)
    //   = (cos(theta/2) a_0 + e^{-i theta'} sin(theta/2) a_1) |0>
    //     + (-e^{-i theta''} sin(theta/2) a_0 + e^{-i(theta' + theta'')} cos(theta/2) a_1) |1>
    template <typename ParallelPolicy, typename RandomAccessIterator, typename Real, typename StateInteger, typename BitInteger>
    inline void adj_phase_shift3(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      Real const phase1, Real const phase2, Real const phase3, ::ket::qubit<StateInteger, BitInteger> const qubit)
    {
      static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
      static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");

      using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
      static_assert(
        (std::is_same<Real, typename ::ket::utility::meta::real_of<complex_type>::type>::value),
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

      using ::ket::utility::loop_n;
      loop_n(
        parallel_policy,
        static_cast<StateInteger>(last - first) >> 1u,
        [first, sine, cosine, &sine_phase_coefficient2, &cosine_phase_coefficient2, &phase_coefficient3, qubit_mask, lower_bits_mask, upper_bits_mask](
          StateInteger const value_wo_qubit, int const)
        {
          // xxxxx0xxxxxx
          auto const zero_index
            = ((value_wo_qubit bitand upper_bits_mask) << 1u)
              bitor (value_wo_qubit bitand lower_bits_mask);
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
    inline void adj_phase_shift3(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      Real const phase1, Real const phase2, Real const phase3,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit)
    {
      static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
      static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");

      using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
      static_assert(
        (std::is_same<Real, typename ::ket::utility::meta::real_of<complex_type>::type>::value),
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

      using ::ket::utility::loop_n;
      loop_n(
        parallel_policy,
        static_cast<StateInteger>(last - first) >> 2u,
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
    inline void adj_phase_shift3(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      Real const phase1, Real const phase2, Real const phase3,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2, ControlQubits const... control_qubits)
    {
      static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
      static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");

      using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
      static_assert(
        (std::is_same<Real, typename ::ket::utility::meta::real_of<complex_type>::type>::value),
        "Real must be the same to \"real part\" of value_type of RandomAccessIterator");

      assert(::ket::utility::integer_exp2<StateInteger>(target_qubit) < static_cast<StateInteger>(last - first));
      assert(::ket::utility::integer_exp2<StateInteger>(control_qubit1) < static_cast<StateInteger>(last - first));
      assert(::ket::utility::integer_exp2<StateInteger>(control_qubit2) < static_cast<StateInteger>(last - first));
      assert(target_qubit != control_qubit1);
      assert(target_qubit != control_qubit2);
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

      constexpr auto num_control_qubits = static_cast<BitInteger>(sizeof...(ControlQubits) + 2u);
      constexpr auto num_qubits = num_control_qubits + BitInteger{1u};
      constexpr auto num_indices = ::ket::utility::integer_exp2<std::size_t>(num_qubits);

      // 0b11...10u
      constexpr auto indices_index0 = ((std::size_t{1u} << num_control_qubits) - std::size_t{1u}) << std::size_t{1u};
      // 0b11...11u
      constexpr auto indices_index1 = indices_index0 bitor std::size_t{1u};

      ::ket::gate::gate(
        parallel_policy, first, last,
        [sine, cosine, &sine_phase_coefficient2, &cosine_phase_coefficient2, &phase_coefficient3](
          RandomAccessIterator const first, std::array<StateInteger, num_indices> const& indices, int const)
        {
          auto const control_on_iter = first + indices[indices_index0];
          auto const target_control_on_iter = first + indices[indices_index1];
          auto const control_on_iter_value = *control_on_iter;

          *control_on_iter *= cosine;
          *control_on_iter += sine_phase_coefficient2 * *target_control_on_iter;
          *target_control_on_iter *= cosine_phase_coefficient2;
          *target_control_on_iter -= sine * control_on_iter_value;
          *target_control_on_iter *= phase_coefficient3;
        },
        target_qubit, control_qubit1, control_qubit2, control_qubits...);
    }

    template <typename RandomAccessIterator, typename Real, typename StateInteger, typename BitInteger, typename... ControlQubits>
    inline void adj_phase_shift3(
      RandomAccessIterator const first, RandomAccessIterator const last,
      Real const phase1, Real const phase2, Real const phase3,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
    { ::ket::gate::adj_phase_shift3(::ket::utility::policy::make_sequential(), first, last, phase1, phase2, phase3, target_qubit, control_qubits...); }

    namespace ranges
    {
      template <typename ParallelPolicy, typename RandomAccessRange, typename Real, typename StateInteger, typename BitInteger, typename... ControlQubits>
      inline RandomAccessRange& adj_phase_shift3(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& state, Real const phase1, Real const phase2, Real const phase3,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
      {
        ::ket::gate::adj_phase_shift3(parallel_policy, std::begin(state), std::end(state), phase1, phase2, phase3, target_qubit, control_qubits...);
        return state;
      }

      template <typename RandomAccessRange, typename Real, typename StateInteger, typename BitInteger, typename... ControlQubits>
      inline RandomAccessRange& adj_phase_shift3(
        RandomAccessRange& state, Real const phase1, Real const phase2, Real const phase3,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
      { return ::ket::gate::ranges::adj_phase_shift3(::ket::utility::policy::make_sequential(), state, phase1, phase2, phase3, target_qubit, control_qubits...); }
    } // namespace ranges
  } // namespace gate
} // namespace ket


#endif // KET_GATE_PHASE_SHIFT_HPP
