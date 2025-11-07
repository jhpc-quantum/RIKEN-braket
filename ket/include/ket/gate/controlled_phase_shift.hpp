#ifndef KET_GATE_CONTROLLED_PHASE_SHIFT_HPP
# define KET_GATE_CONTROLLED_PHASE_SHIFT_HPP

# include <cassert>
# include <complex>
# include <array>
# include <iterator>
# include <algorithm>
# include <utility>
# include <type_traits>

# include <ket/qubit.hpp>
# include <ket/control.hpp>
# include <ket/gate/phase_shift.hpp>
# include <ket/gate/gate.hpp>
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
    // controlled_phase_shift_coeff
    // Case 1: the first argument of qubits is ket::control<ket::qubit<S, B>>
    template <typename ParallelPolicy, typename RandomAccessIterator, typename Complex>
    inline auto controlled_phase_shift_coeff(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      Complex const& phase_coefficient) // exp(i theta) = cos(theta) + i sin(theta)
    -> void
    { ::ket::gate::phase_shift_coeff(parallel_policy, first, last, phase_coefficient); }

    template <typename ParallelPolicy, typename RandomAccessIterator, typename Complex, typename StateInteger, typename BitInteger>
    inline auto controlled_phase_shift_coeff(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit)
    -> void
    { ::ket::gate::phase_shift_coeff(parallel_policy, first, last, phase_coefficient, control_qubit); }

    // U_{cc'}(theta), CU_{cc'}(theta), or C1U_{cc'}(theta)
    // U_{1,2}(theta) (a_{00} |00> + a_{01} |01> + a_{10} |10> + a_{11} |11>)
    //   = a_{00} |00> + a_{01} |01> + a_{10} |10> + e^{i theta} a_{11} |11>
    template <typename ParallelPolicy, typename RandomAccessIterator, typename Complex, typename StateInteger, typename BitInteger>
    inline auto controlled_phase_shift_coeff(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2)
    -> void
    { ::ket::gate::phase_shift_coeff(parallel_policy, first, last, phase_coefficient, control_qubit1, control_qubit2); }

    // C...CU_{c0,c...c'}(theta) or CnU_{c0,c...c'}(theta)
    template <typename ParallelPolicy, typename RandomAccessIterator, typename Complex, typename StateInteger, typename BitInteger, typename... Qubits>
    inline auto controlled_phase_shift_coeff(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      Complex const& phase_coefficient,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit3,
      ::ket::control<Qubits> const... control_qubits)
    -> void
    { ::ket::gate::phase_shift_coeff(parallel_policy, first, last, phase_coefficient, control_qubit1, control_qubit2, control_qubit3, control_qubits...); }

    template <typename RandomAccessIterator, typename Complex, typename StateInteger, typename BitInteger, typename... Qubits>
    inline auto controlled_phase_shift_coeff(
      RandomAccessIterator const first, RandomAccessIterator const last,
      Complex const& phase_coefficient,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2,
      ::ket::control<Qubits> const... control_qubits)
    -> void
    { ::ket::gate::controlled_phase_shift_coeff(::ket::utility::policy::make_sequential(), first, last, phase_coefficient, control_qubit1, control_qubit2, control_qubits...); }

    namespace ranges
    {
      template <typename ParallelPolicy, typename RandomAccessRange, typename Complex, typename... Qubits>
      inline auto controlled_phase_shift_coeff(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& state, Complex const& phase_coefficient,
        ::ket::control<Qubits> const... control_qubits)
      -> RandomAccessRange&
      {
        using std::begin;
        using std::end;
        ::ket::gate::controlled_phase_shift_coeff(parallel_policy, begin(state), end(state), phase_coefficient, control_qubits...);
        return state;
      }

      template <typename RandomAccessRange, typename Complex, typename... Qubits>
      inline auto controlled_phase_shift_coeff(
        RandomAccessRange& state, Complex const& phase_coefficient,
        ::ket::control<Qubits> const... control_qubits)
      -> RandomAccessRange&
      { return ::ket::gate::ranges::controlled_phase_shift_coeff(::ket::utility::policy::make_sequential(), state, phase_coefficient, control_qubits...); }
    } // namespace ranges

    template <typename ParallelPolicy, typename RandomAccessIterator, typename Complex, typename... Qubits>
    inline auto adj_controlled_phase_shift_coeff(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      Complex const& phase_coefficient,
      ::ket::control<Qubits> const... control_qubits)
    -> void
    { using std::conj; ::ket::gate::controlled_phase_shift_coeff(parallel_policy, first, last, conj(phase_coefficient), control_qubits...); }

    template <typename RandomAccessIterator, typename Complex, typename... Qubits>
    inline auto adj_controlled_phase_shift_coeff(
      RandomAccessIterator const first, RandomAccessIterator const last,
      Complex const& phase_coefficient,
      ::ket::control<Qubits> const... control_qubits)
    -> void
    { using std::conj; ::ket::gate::controlled_phase_shift_coeff(first, last, conj(phase_coefficient), control_qubits...); }

    namespace ranges
    {
      template <typename ParallelPolicy, typename RandomAccessRange, typename Complex, typename... Qubits>
      inline auto adj_controlled_phase_shift_coeff(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& state, Complex const& phase_coefficient,
        ::ket::control<Qubits> const... control_qubits)
      -> RandomAccessRange&
      { using std::conj; return ::ket::gate::ranges::controlled_phase_shift_coeff(parallel_policy, state, conj(phase_coefficient), control_qubits...); }

      template <typename RandomAccessRange, typename Complex, typename... Qubits>
      inline auto adj_controlled_phase_shift_coeff(
        RandomAccessRange& state, Complex const& phase_coefficient,
        ::ket::control<Qubits> const... control_qubits)
      -> RandomAccessRange&
      { using std::conj; return ::ket::gate::ranges::controlled_phase_shift_coeff(state, conj(phase_coefficient), control_qubits...); }
    } // namespace ranges

    // Case 2: the first argument of qubits is ket::qubit<S, B>
    // U_{tc}(theta), CU_{tc}(theta), or C1U_{tc}(theta)
    // U_{1,2}(theta) (a_{00} |00> + a_{01} |01> + a_{10} |10> + a_{11} |11>)
    //   = a_{00} |00> + a_{01} |01> + a_{10} |10> + e^{i theta} a_{11} |11>
    template <typename ParallelPolicy, typename RandomAccessIterator, typename Complex, typename StateInteger, typename BitInteger>
    inline auto controlled_phase_shift_coeff(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
      ::ket::qubit<StateInteger, BitInteger> const target_qubit,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit)
    -> void
    { ::ket::gate::phase_shift_coeff(parallel_policy, first, last, phase_coefficient, target_qubit, control_qubit); }

    // C...CU_{tc...c'}(theta) or CnU_{tc...c'}(theta)
    template <typename ParallelPolicy, typename RandomAccessIterator, typename Complex, typename StateInteger, typename BitInteger, typename... ControlQubits>
    inline auto controlled_phase_shift_coeff(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      Complex const& phase_coefficient,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2, ControlQubits const... control_qubits)
    -> void
    { ::ket::gate::phase_shift_coeff(parallel_policy, first, last, phase_coefficient, target_qubit, control_qubit1, control_qubit2, control_qubits...); }

    template <typename RandomAccessIterator, typename Complex, typename StateInteger, typename BitInteger, typename... ControlQubits>
    inline auto controlled_phase_shift_coeff(
      RandomAccessIterator const first, RandomAccessIterator const last,
      Complex const& phase_coefficient,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit, ControlQubits const... control_qubits)
    -> void
    { ::ket::gate::controlled_phase_shift_coeff(::ket::utility::policy::make_sequential(), first, last, phase_coefficient, target_qubit, control_qubit, control_qubits...); }

    namespace ranges
    {
      template <typename ParallelPolicy, typename RandomAccessRange, typename Complex, typename StateInteger, typename BitInteger, typename... ControlQubits>
      inline auto controlled_phase_shift_coeff(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& state, Complex const& phase_coefficient,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit, ControlQubits const... control_qubits)
      -> RandomAccessRange&
      {
        using std::begin;
        using std::end;
        ::ket::gate::controlled_phase_shift_coeff(parallel_policy, begin(state), end(state), phase_coefficient, target_qubit, control_qubit, control_qubits...);
        return state;
      }

      template <typename RandomAccessRange, typename Complex, typename StateInteger, typename BitInteger, typename... ControlQubits>
      inline auto controlled_phase_shift_coeff(
        RandomAccessRange& state, Complex const& phase_coefficient,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit, ControlQubits const... control_qubits)
      -> RandomAccessRange&
      { return ::ket::gate::ranges::controlled_phase_shift_coeff(::ket::utility::policy::make_sequential(), state, phase_coefficient, target_qubit, control_qubit, control_qubits...); }
    } // namespace ranges

    template <typename ParallelPolicy, typename RandomAccessIterator, typename Complex, typename StateInteger, typename BitInteger, typename... ControlQubits>
    inline auto adj_controlled_phase_shift_coeff(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      Complex const& phase_coefficient,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit, ControlQubits const... control_qubits)
    -> void
    { using std::conj; ::ket::gate::controlled_phase_shift_coeff(parallel_policy, first, last, conj(phase_coefficient), target_qubit, control_qubit, control_qubits...); }

    template <typename RandomAccessIterator, typename Complex, typename StateInteger, typename BitInteger, typename... ControlQubits>
    inline auto adj_controlled_phase_shift_coeff(
      RandomAccessIterator const first, RandomAccessIterator const last,
      Complex const& phase_coefficient,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit, ControlQubits const... control_qubits)
    -> void
    { using std::conj; ::ket::gate::controlled_phase_shift_coeff(first, last, conj(phase_coefficient), target_qubit, control_qubit, control_qubits...); }

    namespace ranges
    {
      template <typename ParallelPolicy, typename RandomAccessRange, typename Complex, typename StateInteger, typename BitInteger, typename... ControlQubits>
      inline auto adj_controlled_phase_shift_coeff(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& state, Complex const& phase_coefficient,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit, ControlQubits const... control_qubits)
      -> RandomAccessRange&
      { using std::conj; return ::ket::gate::ranges::controlled_phase_shift_coeff(parallel_policy, state, conj(phase_coefficient), target_qubit, control_qubit, control_qubits...); }

      template <typename RandomAccessRange, typename Complex, typename StateInteger, typename BitInteger, typename... ControlQubits>
      inline auto adj_controlled_phase_shift_coeff(
        RandomAccessRange& state, Complex const& phase_coefficient,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit, ControlQubits const... control_qubits)
      -> RandomAccessRange&
      { using std::conj; return ::ket::gate::ranges::controlled_phase_shift_coeff(state, conj(phase_coefficient), target_qubit, control_qubit, control_qubits...); }
    } // namespace ranges

    // controlled_phase_shift
    // Case 1: the first argument of qubits is ket::control<ket::qubit<S, B>>
    template <typename ParallelPolicy, typename RandomAccessIterator, typename Real, typename... Qubits>
    inline auto controlled_phase_shift(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      Real const phase,
      ::ket::control<Qubits> const... control_qubits)
    -> void
    {
      using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
      ::ket::gate::controlled_phase_shift_coeff(parallel_policy, first, last, ::ket::utility::exp_i<complex_type>(phase), control_qubits...);
    }

    template <typename RandomAccessIterator, typename Real, typename... Qubits>
    inline auto controlled_phase_shift(
      RandomAccessIterator const first, RandomAccessIterator const last,
      Real const phase,
      ::ket::control<Qubits> const... control_qubits)
    -> void
    {
      using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
      ::ket::gate::controlled_phase_shift_coeff(first, last, ::ket::utility::exp_i<complex_type>(phase), control_qubits...);
    }

    namespace ranges
    {
      template <typename ParallelPolicy, typename RandomAccessRange, typename Real, typename... Qubits>
      inline auto controlled_phase_shift(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& state, Real const phase,
        ::ket::control<Qubits> const... control_qubits)
      -> RandomAccessRange&
      {
        using complex_type = ::ket::utility::meta::range_value_t<RandomAccessRange>;
        return ::ket::gate::ranges::controlled_phase_shift_coeff(parallel_policy, state, ::ket::utility::exp_i<complex_type>(phase), control_qubits...);
      }

      template <typename RandomAccessRange, typename Real, typename... Qubits>
      inline auto controlled_phase_shift(
        RandomAccessRange& state, Real const phase,
        ::ket::control<Qubits> const... control_qubits)
      -> RandomAccessRange&
      {
        using complex_type = ::ket::utility::meta::range_value_t<RandomAccessRange>;
        return ::ket::gate::ranges::controlled_phase_shift_coeff(state, ::ket::utility::exp_i<complex_type>(phase), control_qubits...);
      }
    } // namespace ranges

    template <typename ParallelPolicy, typename RandomAccessIterator, typename Real, typename... Qubits>
    inline auto adj_controlled_phase_shift(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      Real const phase,
      ::ket::control<Qubits> const... control_qubits)
    -> void
    { ::ket::gate::controlled_phase_shift(parallel_policy, first, last, -phase, control_qubits...); }

    template <typename RandomAccessIterator, typename Real, typename... Qubits>
    inline auto adj_controlled_phase_shift(
      RandomAccessIterator const first, RandomAccessIterator const last,
      Real const phase,
      ::ket::control<Qubits> const... control_qubits)
    -> void
    { ::ket::gate::controlled_phase_shift(first, last, -phase, control_qubits...); }

    namespace ranges
    {
      template <typename ParallelPolicy, typename RandomAccessRange, typename Real, typename... Qubits>
      inline auto adj_controlled_phase_shift(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& state, Real const phase,
        ::ket::control<Qubits> const... control_qubits)
      -> RandomAccessRange&
      { return ::ket::gate::ranges::controlled_phase_shift(parallel_policy, state, -phase, control_qubits...); }

      template <typename RandomAccessRange, typename Real, typename... Qubits>
      inline auto adj_controlled_phase_shift(
        RandomAccessRange& state, Real const phase,
        ::ket::control<Qubits> const... control_qubits)
      -> RandomAccessRange&
      { return ::ket::gate::ranges::controlled_phase_shift(state, -phase, control_qubits...); }
    } // namespace ranges

    // Case 2: the first argument of qubits is ket::qubit<S, B>
    template <typename ParallelPolicy, typename RandomAccessIterator, typename Real, typename StateInteger, typename BitInteger, typename... ControlQubits>
    inline auto controlled_phase_shift(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      Real const phase,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit, ControlQubits const... control_qubits)
    -> void
    {
      using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
      ::ket::gate::controlled_phase_shift_coeff(parallel_policy, first, last, ::ket::utility::exp_i<complex_type>(phase), target_qubit, control_qubit, control_qubits...);
    }

    template <typename RandomAccessIterator, typename Real, typename StateInteger, typename BitInteger, typename... ControlQubits>
    inline auto controlled_phase_shift(
      RandomAccessIterator const first, RandomAccessIterator const last,
      Real const phase,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit, ControlQubits const... control_qubits)
    -> void
    {
      using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
      ::ket::gate::controlled_phase_shift_coeff(first, last, ::ket::utility::exp_i<complex_type>(phase), target_qubit, control_qubit, control_qubits...);
    }

    namespace ranges
    {
      template <typename ParallelPolicy, typename RandomAccessRange, typename Real, typename StateInteger, typename BitInteger, typename... ControlQubits>
      inline auto controlled_phase_shift(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& state, Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit, ControlQubits const... control_qubits)
      -> RandomAccessRange&
      {
        using complex_type = ::ket::utility::meta::range_value_t<RandomAccessRange>;
        return ::ket::gate::ranges::controlled_phase_shift_coeff(parallel_policy, state, ::ket::utility::exp_i<complex_type>(phase), target_qubit, control_qubit, control_qubits...);
      }

      template <typename RandomAccessRange, typename Real, typename StateInteger, typename BitInteger, typename... ControlQubits>
      inline auto controlled_phase_shift(
        RandomAccessRange& state, Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit, ControlQubits const... control_qubits)
      -> RandomAccessRange&
      {
        using complex_type = ::ket::utility::meta::range_value_t<RandomAccessRange>;
        return ::ket::gate::ranges::controlled_phase_shift_coeff(state, ::ket::utility::exp_i<complex_type>(phase), target_qubit, control_qubit, control_qubits...);
      }
    } // namespace ranges

    template <typename ParallelPolicy, typename RandomAccessIterator, typename Real, typename StateInteger, typename BitInteger, typename... ControlQubits>
    inline auto adj_controlled_phase_shift(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      Real const phase,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit, ControlQubits const... control_qubits)
    -> void
    { ::ket::gate::controlled_phase_shift(parallel_policy, first, last, -phase, target_qubit, control_qubit, control_qubits...); }

    template <typename RandomAccessIterator, typename Real, typename StateInteger, typename BitInteger, typename... ControlQubits>
    inline auto adj_controlled_phase_shift(
      RandomAccessIterator const first, RandomAccessIterator const last,
      Real const phase,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit, ControlQubits const... control_qubits)
    -> void
    { ::ket::gate::controlled_phase_shift(first, last, -phase, target_qubit, control_qubit, control_qubits...); }

    namespace ranges
    {
      template <typename ParallelPolicy, typename RandomAccessRange, typename Real, typename StateInteger, typename BitInteger, typename... ControlQubits>
      inline auto adj_controlled_phase_shift(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& state, Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit, ControlQubits const... control_qubits)
      -> RandomAccessRange&
      { return ::ket::gate::ranges::controlled_phase_shift(parallel_policy, state, -phase, target_qubit, control_qubit, control_qubits...); }

      template <typename RandomAccessRange, typename Real, typename StateInteger, typename BitInteger, typename... ControlQubits>
      inline auto adj_controlled_phase_shift(
        RandomAccessRange& state, Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit, ControlQubits const... control_qubits)
      -> RandomAccessRange&
      { return ::ket::gate::ranges::controlled_phase_shift(state, -phase, target_qubit, control_qubit, control_qubits...); }
    } // namespace ranges
  } // namespace gate
} // namespace ket


#endif // KET_GATE_CONTROLLED_PHASE_SHIFT_HPP
