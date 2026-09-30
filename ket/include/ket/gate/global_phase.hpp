#ifndef KET_GATE_GLOBAL_PHASE_HPP
# define KET_GATE_GLOBAL_PHASE_HPP

# include <cassert>
# include <complex>
# include <cstdint>
# include <iterator>
# include <type_traits>

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
    // global_phase_coeff
    template <typename ParallelPolicy, typename RandomAccessIterator, typename Complex>
    inline auto global_phase_coeff(
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

    template <typename RandomAccessIterator, typename Complex>
    inline auto global_phase_coeff(
      RandomAccessIterator const first, RandomAccessIterator const last,
      Complex const& phase_coefficient) // exp(i theta) = cos(theta) + i sin(theta)
    -> void
    { ::ket::gate::global_phase_coeff(::ket::utility::policy::make_sequential(), first, last, phase_coefficient); }

    namespace ranges
    {
      template <typename ParallelPolicy, typename RandomAccessRange, typename Complex>
      inline auto global_phase_coeff(
        ParallelPolicy const parallel_policy, RandomAccessRange& state,
        Complex const& phase_coefficient) // exp(i theta) = cos(theta) + i sin(theta)
      -> std::enable_if_t< ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value, RandomAccessRange& >
      {
        using std::begin;
        using std::end;
        ::ket::gate::global_phase_coeff(parallel_policy, begin(state), end(state), phase_coefficient);
        return state;
      }

      template <typename RandomAccessRange, typename Complex>
      inline auto global_phase_coeff(
        RandomAccessRange& state,
        Complex const& phase_coefficient) // exp(i theta) = cos(theta) + i sin(theta)
      -> RandomAccessRange&
      { return ::ket::gate::ranges::global_phase_coeff(::ket::utility::policy::make_sequential(), state, phase_coefficient); }
    } // namespace ranges

    template <typename ParallelPolicy, typename RandomAccessIterator, typename Complex>
    inline auto adj_global_phase_coeff(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      Complex const& phase_coefficient) // exp(i theta) = cos(theta) + i sin(theta)
    -> void
    { using std::conj; ::ket::gate::global_phase_coeff(parallel_policy, first, last, conj(phase_coefficient)); }

    template <typename RandomAccessIterator, typename Complex>
    inline auto adj_global_phase_coeff(
      RandomAccessIterator const first, RandomAccessIterator const last,
      Complex const& phase_coefficient) // exp(i theta) = cos(theta) + i sin(theta)
    -> void
    { using std::conj; ::ket::gate::global_phase_coeff(first, last, conj(phase_coefficient)); }

    namespace ranges
    {
      template <typename ParallelPolicy, typename RandomAccessRange, typename Complex>
      inline auto adj_global_phase_coeff(
        ParallelPolicy const parallel_policy, RandomAccessRange& state,
        Complex const& phase_coefficient) // exp(i theta) = cos(theta) + i sin(theta)
      -> std::enable_if_t< ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value, RandomAccessRange& >
      { using std::conj; return ::ket::gate::ranges::global_phase_coeff(parallel_policy, state, conj(phase_coefficient)); }

      template <typename RandomAccessRange, typename Complex>
      inline auto adj_global_phase_coeff(
        RandomAccessRange& state,
        Complex const& phase_coefficient) // exp(i theta) = cos(theta) + i sin(theta)
      -> RandomAccessRange&
      { using std::conj; return ::ket::gate::ranges::global_phase_coeff(state, conj(phase_coefficient)); }
    } // namespace ranges

    template <typename ParallelPolicy, typename RandomAccessIterator, typename Real>
    inline auto global_phase(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last, Real const phase)
    -> void
    {
      using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
      ::ket::gate::global_phase_coeff(parallel_policy, first, last, ::ket::utility::exp_i<complex_type>(phase));
    }

    template <typename RandomAccessIterator, typename Real>
    inline auto global_phase(RandomAccessIterator const first, RandomAccessIterator const last, Real const phase)
    -> void
    { ::ket::gate::global_phase(::ket::utility::policy::make_sequential(), first, last, phase); }

    namespace ranges
    {
      template <typename ParallelPolicy, typename RandomAccessRange, typename Real>
      inline auto global_phase(ParallelPolicy const parallel_policy, RandomAccessRange& state, Real const phase)
      -> std::enable_if_t< ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value, RandomAccessRange& >
      {
        using std::begin;
        using std::end;
        ::ket::gate::global_phase(parallel_policy, begin(state), end(state), phase);
        return state;
      }

      template <typename RandomAccessRange, typename Real>
      inline auto global_phase(RandomAccessRange& state, Real const phase) -> RandomAccessRange&
      { return ::ket::gate::ranges::global_phase(::ket::utility::policy::make_sequential(), state, phase); }
    } // namespace ranges

    template <typename ParallelPolicy, typename RandomAccessIterator, typename Real>
    inline auto adj_global_phase(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last, Real const phase)
    -> void
    { ::ket::gate::global_phase(parallel_policy, first, last, -phase); }

    template <typename RandomAccessIterator, typename Real>
    inline auto adj_global_phase(RandomAccessIterator const first, RandomAccessIterator const last, Real const phase)
    -> void
    { ::ket::gate::global_phase(first, last, -phase); }

    namespace ranges
    {
      template <typename ParallelPolicy, typename RandomAccessRange, typename Real>
      inline auto adj_global_phase(ParallelPolicy const parallel_policy, RandomAccessRange& state, Real const phase)
      -> std::enable_if_t< ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value, RandomAccessRange& >
      { return ::ket::gate::ranges::global_phase(parallel_policy, state, -phase); }

      template <typename RandomAccessRange, typename Real>
      inline auto adj_global_phase(RandomAccessRange& state, Real const phase) -> RandomAccessRange&
      { return ::ket::gate::ranges::global_phase(state, -phase); }
    } // namespace ranges

    namespace runtime
    {
      template <typename ParallelPolicy, typename RandomAccessIterator, typename Complex>
      inline auto global_phase_coeff(
        ParallelPolicy const parallel_policy,
        RandomAccessIterator const first, RandomAccessIterator const last,
        Complex const& phase_coefficient) // exp(i theta) = cos(theta) + i sin(theta)
      -> void
      { ::ket::gate::global_phase_coeff(parallel_policy, first, last, phase_coefficient); }

      template <typename RandomAccessIterator, typename Complex>
      inline auto global_phase_coeff(
        RandomAccessIterator const first, RandomAccessIterator const last,
        Complex const& phase_coefficient) // exp(i theta) = cos(theta) + i sin(theta)
      -> void
      { ::ket::gate::runtime::global_phase_coeff(::ket::utility::policy::make_sequential(), first, last, phase_coefficient); }

      namespace ranges
      {
        template <typename ParallelPolicy, typename RandomAccessRange, typename Complex>
        inline auto global_phase_coeff(
          ParallelPolicy const parallel_policy, RandomAccessRange& state,
          Complex const& phase_coefficient) // exp(i theta) = cos(theta) + i sin(theta)
        -> std::enable_if_t< ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value, RandomAccessRange& >
        {
          using std::begin;
          using std::end;
          ::ket::gate::runtime::global_phase_coeff(parallel_policy, begin(state), end(state), phase_coefficient);
          return state;
        }

        template <typename RandomAccessRange, typename Complex>
        inline auto global_phase_coeff(
          RandomAccessRange& state,
          Complex const& phase_coefficient) // exp(i theta) = cos(theta) + i sin(theta)
        -> RandomAccessRange&
        { return ::ket::gate::runtime::ranges::global_phase_coeff(::ket::utility::policy::make_sequential(), state, phase_coefficient); }
      } // namespace ranges

      template <typename ParallelPolicy, typename RandomAccessIterator, typename Complex>
      inline auto adj_global_phase_coeff(
        ParallelPolicy const parallel_policy,
        RandomAccessIterator const first, RandomAccessIterator const last,
        Complex const& phase_coefficient) // exp(i theta) = cos(theta) + i sin(theta)
      -> void
      { using std::conj; ::ket::gate::runtime::global_phase_coeff(parallel_policy, first, last, conj(phase_coefficient)); }

      template <typename RandomAccessIterator, typename Complex>
      inline auto adj_global_phase_coeff(
        RandomAccessIterator const first, RandomAccessIterator const last,
        Complex const& phase_coefficient) // exp(i theta) = cos(theta) + i sin(theta)
      -> void
      { using std::conj; ::ket::gate::runtime::global_phase_coeff(first, last, conj(phase_coefficient)); }

      namespace ranges
      {
        template <typename ParallelPolicy, typename RandomAccessRange, typename Complex>
        inline auto adj_global_phase_coeff(
          ParallelPolicy const parallel_policy, RandomAccessRange& state,
          Complex const& phase_coefficient) // exp(i theta) = cos(theta) + i sin(theta)
        -> std::enable_if_t< ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value, RandomAccessRange& >
        { using std::conj; return ::ket::gate::runtime::ranges::global_phase_coeff(parallel_policy, state, conj(phase_coefficient)); }

        template <typename RandomAccessRange, typename Complex>
        inline auto adj_global_phase_coeff(
          RandomAccessRange& state,
          Complex const& phase_coefficient) // exp(i theta) = cos(theta) + i sin(theta)
        -> RandomAccessRange&
        { using std::conj; return ::ket::gate::runtime::ranges::global_phase_coeff(state, conj(phase_coefficient)); }
      } // namespace ranges

      template <typename ParallelPolicy, typename RandomAccessIterator, typename Real>
      inline auto global_phase(
        ParallelPolicy const parallel_policy,
        RandomAccessIterator const first, RandomAccessIterator const last, Real const phase)
      -> void
      {
        using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
        ::ket::gate::runtime::global_phase_coeff(parallel_policy, first, last, ::ket::utility::exp_i<complex_type>(phase));
      }

      template <typename RandomAccessIterator, typename Real>
      inline auto global_phase(RandomAccessIterator const first, RandomAccessIterator const last, Real const phase)
      -> void
      { ::ket::gate::runtime::global_phase(::ket::utility::policy::make_sequential(), first, last, phase); }

      namespace ranges
      {
        template <typename ParallelPolicy, typename RandomAccessRange, typename Real>
        inline auto global_phase(ParallelPolicy const parallel_policy, RandomAccessRange& state, Real const phase)
        -> std::enable_if_t< ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value, RandomAccessRange& >
        {
          using std::begin;
          using std::end;
          ::ket::gate::runtime::global_phase(parallel_policy, begin(state), end(state), phase);
          return state;
        }

        template <typename RandomAccessRange, typename Real>
        inline auto global_phase(RandomAccessRange& state, Real const phase) -> RandomAccessRange&
        { return ::ket::gate::runtime::ranges::global_phase(::ket::utility::policy::make_sequential(), state, phase); }
      } // namespace ranges

      template <typename ParallelPolicy, typename RandomAccessIterator, typename Real>
      inline auto adj_global_phase(
        ParallelPolicy const parallel_policy,
        RandomAccessIterator const first, RandomAccessIterator const last, Real const phase)
      -> void
      { ::ket::gate::runtime::global_phase(parallel_policy, first, last, -phase); }

      template <typename RandomAccessIterator, typename Real>
      inline auto adj_global_phase(RandomAccessIterator const first, RandomAccessIterator const last, Real const phase)
      -> void
      { ::ket::gate::runtime::global_phase(first, last, -phase); }

      namespace ranges
      {
        template <typename ParallelPolicy, typename RandomAccessRange, typename Real>
        inline auto adj_global_phase(ParallelPolicy const parallel_policy, RandomAccessRange& state, Real const phase)
        -> std::enable_if_t< ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value, RandomAccessRange& >
        { return ::ket::gate::runtime::ranges::global_phase(parallel_policy, state, -phase); }

        template <typename RandomAccessRange, typename Real>
        inline auto adj_global_phase(RandomAccessRange& state, Real const phase) -> RandomAccessRange&
        { return ::ket::gate::runtime::ranges::global_phase(state, -phase); }
      } // namespace ranges
    } // namespace runtime
  } // namespace gate
} // namespace ket

#endif // KET_GATE_GLOBAL_PHASE_HPP
