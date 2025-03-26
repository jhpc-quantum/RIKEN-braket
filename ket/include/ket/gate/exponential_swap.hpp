#ifndef KET_GATE_EXPONENTIAL_SWAP_HPP
# define KET_GATE_EXPONENTIAL_SWAP_HPP

# include <cassert>
# include <cmath>
# include <complex>
# include <algorithm>
# include <iterator>
# include <utility>
# include <type_traits>

# include <ket/qubit.hpp>
# include <ket/control.hpp>
# include <ket/gate/gate.hpp>
# include <ket/gate/utility/index_with_qubits.hpp>
# include <ket/gate/meta/num_control_qubits.hpp>
# include <ket/utility/loop_n.hpp>
# include <ket/utility/integer_exp2.hpp>
# ifndef NDEBUG
#   include <ket/utility/integer_log2.hpp>
# endif
# include <ket/utility/imaginary_unit.hpp>
# include <ket/utility/exp_i.hpp>
# include <ket/utility/meta/ranges.hpp>


namespace ket
{
  namespace gate
  {
    // exponential_swap_coeff
    // eSWAP_{ij}(s) = exp(is SWAP_{ij}) = I cos s + i SWAP_{ij} sin s
    // eSWAP_{1,2}(s) (a_{00} |00> + a_{01} |01> + a_{10} |10> + a_{11} |11>)
    //   = e^{is} a_{00} |00> + (cos s a_{01} + i sin s a_{10}) |01> + (i sin s a_{01} + cos s a_{10}) |10> + e^{is} a_{11} |11>
    template <typename ParallelPolicy, typename RandomAccessIterator, typename Complex, typename StateInteger, typename BitInteger>
    inline auto exponential_swap_coeff(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
      ::ket::qubit<StateInteger, BitInteger> const qubit1, ::ket::qubit<StateInteger, BitInteger> const qubit2)
    -> void
    {
      static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
      static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");
      static_assert(
        std::is_same<Complex, typename std::iterator_traits<RandomAccessIterator>::value_type>::value,
        "Complex must be the same to value_type of RandomAccessIterator");

      assert(::ket::utility::integer_exp2<StateInteger>(qubit1) < static_cast<StateInteger>(last - first));
      assert(::ket::utility::integer_exp2<StateInteger>(qubit2) < static_cast<StateInteger>(last - first));
      assert(qubit1 != qubit2);
      assert(
        ::ket::utility::integer_exp2<StateInteger>(::ket::utility::integer_log2<BitInteger>(last - first))
        == static_cast<StateInteger>(last - first));

      auto const minmax_qubits = std::minmax(qubit1, qubit2);
      auto const qubit1_mask = ::ket::utility::integer_exp2<StateInteger>(qubit1);
      auto const qubit2_mask = ::ket::utility::integer_exp2<StateInteger>(qubit2);
      auto const lower_bits_mask = ::ket::utility::integer_exp2<StateInteger>(minmax_qubits.first) - StateInteger{1u};
      auto const middle_bits_mask
        = (::ket::utility::integer_exp2<StateInteger>(minmax_qubits.second - BitInteger{1u}) - StateInteger{1u})
          xor lower_bits_mask;
      auto const upper_bits_mask = compl (lower_bits_mask bitor middle_bits_mask);

      using std::imag;
      auto const i_sin_theta = ::ket::utility::imaginary_unit<Complex>() * imag(phase_coefficient);

      ::ket::utility::loop_n(
        parallel_policy, static_cast<StateInteger>(last - first) >> 2u,
        [first, &phase_coefficient, &i_sin_theta, qubit1_mask, qubit2_mask, lower_bits_mask, middle_bits_mask, upper_bits_mask](
          StateInteger const value_wo_qubits, int const)
        {
          // xxx0_1xxx0_2xxx
          auto const base_index
            = ((value_wo_qubits bitand upper_bits_mask) << 2u)
              bitor ((value_wo_qubits bitand middle_bits_mask) << 1u)
              bitor (value_wo_qubits bitand lower_bits_mask);
          // xxx1_1xxx0_2xxx
          auto const qubit1_on_index = base_index bitor qubit1_mask;
          auto const qubit1_on_iter = first + qubit1_on_index;
          // xxx0_1xxx1_2xxx
          auto const qubit2_on_iter = first + (base_index bitor qubit2_mask);
          // xxx1_1xxx1_2xxx
          auto const qubit12_on_iter = first + (qubit1_on_index bitor qubit2_mask);

          *(first + base_index) *= phase_coefficient;
          *qubit12_on_iter *= phase_coefficient;

          auto const qubit1_on_iter_value = *qubit1_on_iter;
          using std::real;
          using std::imag;
          *qubit1_on_iter *= real(phase_coefficient);
          *qubit1_on_iter += *qubit2_on_iter * i_sin_theta;
          *qubit2_on_iter *= real(phase_coefficient);
          *qubit2_on_iter += qubit1_on_iter_value * i_sin_theta;
        });
    }

    // C...CeSWAP_{tt'c...c'}(s) = C...C[exp(is SWAP_{tt'})]_{c...c'} = C...C[I cos s + i SWAP_{tt'} sin s]_{c...c'}
    template <typename ParallelPolicy, typename RandomAccessIterator, typename Complex, typename StateInteger, typename BitInteger, typename... ControlQubits>
    inline auto exponential_swap_coeff(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
      ::ket::qubit<StateInteger, BitInteger> const target_qubit1, ::ket::qubit<StateInteger, BitInteger> const target_qubit2,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit, ControlQubits const... control_qubits)
    -> void
    {
      static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
      static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");
      static_assert(
        std::is_same<Complex, typename std::iterator_traits<RandomAccessIterator>::value_type>::value,
        "Complex must be the same to value_type of RandomAccessIterator");

      assert(::ket::utility::integer_exp2<StateInteger>(target_qubit1) < static_cast<StateInteger>(last - first));
      assert(::ket::utility::integer_exp2<StateInteger>(target_qubit2) < static_cast<StateInteger>(last - first));
      assert(target_qubit1 != target_qubit2);
      assert(
        ::ket::utility::integer_exp2<StateInteger>(::ket::utility::integer_log2<BitInteger>(last - first))
        == static_cast<StateInteger>(last - first));

# ifndef KET_USE_BIT_MASKS_EXPLICITLY
      using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
      constexpr auto num_control_qubits = static_cast<BitInteger>(sizeof...(ControlQubits) + 1u);
      constexpr auto num_operated_qubits = num_control_qubits + BitInteger{2u};

      using std::imag;
      auto const i_sin_theta = ::ket::utility::imaginary_unit<Complex>() * imag(phase_coefficient);

      ::ket::gate::nocache::gate(
        parallel_policy, first, last,
        [&phase_coefficient, &i_sin_theta](
          auto const first, StateInteger const index_wo_qubits,
          std::array<qubit_type, num_operated_qubits> const& unsorted_qubits,
          std::array<qubit_type, num_operated_qubits + BitInteger{1u}> const& sorted_qubits_with_sentinel,
          int const)
        {
          // 0b11...100u
          constexpr auto index00 = ((std::size_t{1u} << num_control_qubits) - std::size_t{1u}) << BitInteger{2u};
          // 0b11...101u
          constexpr auto index01 = index00 bitor std::size_t{1u};
          // 0b11...110u
          constexpr auto index10 = index00 bitor (std::size_t{1u} << BitInteger{1u});
          // 0b11...111u
          constexpr auto index11 = index10 bitor std::size_t{1u};

          using std::begin;
          using std::end;
          auto const iter00
            = first
              + ::ket::gate::utility::index_with_qubits(
                  index_wo_qubits, index00,
                  begin(unsorted_qubits), end(unsorted_qubits),
                  begin(sorted_qubits_with_sentinel), end(sorted_qubits_with_sentinel));
          auto const iter11
            = first
              + ::ket::gate::utility::index_with_qubits(
                  index_wo_qubits, index11,
                  begin(unsorted_qubits), end(unsorted_qubits),
                  begin(sorted_qubits_with_sentinel), end(sorted_qubits_with_sentinel));
          *iter00 *= phase_coefficient;
          *iter11 *= phase_coefficient;

          auto const iter01
            = first
              + ::ket::gate::utility::index_with_qubits(
                  index_wo_qubits, index01,
                  begin(unsorted_qubits), end(unsorted_qubits),
                  begin(sorted_qubits_with_sentinel), end(sorted_qubits_with_sentinel));
          auto const iter10
            = first
              + ::ket::gate::utility::index_with_qubits(
                  index_wo_qubits, index10,
                  begin(unsorted_qubits), end(unsorted_qubits),
                  begin(sorted_qubits_with_sentinel), end(sorted_qubits_with_sentinel));
          auto const value01 = *iter01;
          using std::real;
          using std::imag;
          *iter01 *= real(phase_coefficient);
          *iter01 += *iter10 * i_sin_theta;
          *iter10 *= real(phase_coefficient);
          *iter10 += value01 * i_sin_theta;
        },
        target_qubit1, target_qubit2, control_qubit, control_qubits...);
# else // KET_USE_BIT_MASKS_EXPLICITLY
      constexpr auto num_control_qubits = static_cast<BitInteger>(sizeof...(ControlQubits) + 1u);
      constexpr auto num_operated_qubits = num_control_qubits + BitInteger{2u};

      using std::imag;
      auto const i_sin_theta = ::ket::utility::imaginary_unit<Complex>() * imag(phase_coefficient);

      ::ket::gate::nocache::gate(
        parallel_policy, first, last,
        [&phase_coefficient, &i_sin_theta](
          auto const first, StateInteger const index_wo_qubits,
          std::array<StateInteger, num_operated_qubits> const& qubit_masks,
          std::array<StateInteger, num_operated_qubits + 1u> const& index_masks,
          int const)
        {
          // 0b11...100u
          constexpr auto index00 = ((std::size_t{1u} << num_control_qubits) - std::size_t{1u}) << BitInteger{2u};
          // 0b11...101u
          constexpr auto index01 = index00 bitor std::size_t{1u};
          // 0b11...110u
          constexpr auto index10 = index00 bitor (std::size_t{1u} << BitInteger{1u});
          // 0b11...111u
          constexpr auto index11 = index10 bitor std::size_t{1u};

          using std::begin;
          using std::end;
          auto const iter00
            = first
              + ::ket::gate::utility::index_with_qubits(
                  index_wo_qubits, index00,
                  begin(qubit_masks), end(qubit_masks), begin(index_masks), end(index_masks));
          auto const iter11
            = first
              + ::ket::gate::utility::index_with_qubits(
                  index_wo_qubits, index11,
                  begin(qubit_masks), end(qubit_masks), begin(index_masks), end(index_masks));
          *iter00 *= phase_coefficient;
          *iter11 *= phase_coefficient;

          auto const iter01
            = first
              + ::ket::gate::utility::index_with_qubits(
                  index_wo_qubits, index01,
                  begin(qubit_masks), end(qubit_masks), begin(index_masks), end(index_masks));
          auto const iter10
            = first
              + ::ket::gate::utility::index_with_qubits(
                  index_wo_qubits, index10,
                  begin(qubit_masks), end(qubit_masks), begin(index_masks), end(index_masks));
          auto const value01 = *iter01;
          using std::real;
          using std::imag;
          *iter01 *= real(phase_coefficient);
          *iter01 += *iter10 * i_sin_theta;
          *iter10 *= real(phase_coefficient);
          *iter10 += value01 * i_sin_theta;
        },
        target_qubit1, target_qubit2, control_qubit, control_qubits...);
# endif // KET_USE_BIT_MASKS_EXPLICITLY
    }

    template <typename RandomAccessIterator, typename Complex, typename StateInteger, typename BitInteger, typename... ControlQubits>
    inline auto exponential_swap_coeff(
      RandomAccessIterator const first, RandomAccessIterator const last,
      Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
      ::ket::qubit<StateInteger, BitInteger> const target_qubit1, ::ket::qubit<StateInteger, BitInteger> const target_qubit2,
      ControlQubits const... control_qubits)
    -> void
    { ::ket::gate::exponential_swap_coeff(::ket::utility::policy::make_sequential(), first, last, phase_coefficient, target_qubit1, target_qubit2, control_qubits...); }

    namespace ranges
    {
      template <typename ParallelPolicy, typename RandomAccessRange, typename Complex, typename StateInteger, typename BitInteger, typename... ControlQubits>
      inline auto exponential_swap_coeff(
        ParallelPolicy const parallel_policy, RandomAccessRange& state,
        Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
        ::ket::qubit<StateInteger, BitInteger> const target_qubit1, ::ket::qubit<StateInteger, BitInteger> const target_qubit2,
        ControlQubits const... control_qubits)
      -> RandomAccessRange&
      {
        using std::begin;
        using std::end;
        ::ket::gate::exponential_swap_coeff(parallel_policy, begin(state), end(state), phase_coefficient, target_qubit1, target_qubit2, control_qubits...);
        return state;
      }

      template <typename RandomAccessRange, typename Complex, typename StateInteger, typename BitInteger, typename... ControlQubits>
      inline auto exponential_swap_coeff(
        RandomAccessRange& state,
        Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
        ::ket::qubit<StateInteger, BitInteger> const target_qubit1, ::ket::qubit<StateInteger, BitInteger> const target_qubit2,
        ControlQubits const... control_qubits)
      -> RandomAccessRange&
      { return ::ket::gate::ranges::exponential_swap_coeff(::ket::utility::policy::make_sequential(), state, phase_coefficient, target_qubit1, target_qubit2, control_qubits...); }
    } // namespace ranges

    template <typename ParallelPolicy, typename RandomAccessIterator, typename Complex, typename StateInteger, typename BitInteger, typename... ControlQubits>
    inline auto adj_exponential_swap_coeff(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
      ::ket::qubit<StateInteger, BitInteger> const target_qubit1, ::ket::qubit<StateInteger, BitInteger> const target_qubit2,
      ControlQubits const... control_qubits)
    -> void
    { using std::conj; ::ket::gate::exponential_swap_coeff(parallel_policy, first, last, conj(phase_coefficient), target_qubit1, target_qubit2, control_qubits...); }

    template <typename RandomAccessIterator, typename Complex, typename StateInteger, typename BitInteger, typename... ControlQubits>
    inline auto adj_exponential_swap_coeff(
      RandomAccessIterator const first, RandomAccessIterator const last,
      Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
      ::ket::qubit<StateInteger, BitInteger> const target_qubit1, ::ket::qubit<StateInteger, BitInteger> const target_qubit2,
      ControlQubits const... control_qubits)
    -> void
    { using std::conj; ::ket::gate::exponential_swap_coeff(first, last, conj(phase_coefficient), target_qubit1, target_qubit2, control_qubits...); }

    namespace ranges
    {
      template <typename ParallelPolicy, typename RandomAccessRange, typename Complex, typename StateInteger, typename BitInteger, typename... ControlQubits>
      inline auto adj_exponential_swap_coeff(
        ParallelPolicy const parallel_policy, RandomAccessRange& state,
        Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
        ::ket::qubit<StateInteger, BitInteger> const target_qubit1, ::ket::qubit<StateInteger, BitInteger> const target_qubit2,
        ControlQubits const... control_qubits)
      -> RandomAccessRange&
      { using std::conj; return ::ket::gate::ranges::exponential_swap_coeff(parallel_policy, state, conj(phase_coefficient), target_qubit1, target_qubit2, control_qubits...); }

      template <typename RandomAccessRange, typename Complex, typename StateInteger, typename BitInteger, typename... ControlQubits>
      inline auto adj_exponential_swap_coeff(
        RandomAccessRange& state,
        Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
        ::ket::qubit<StateInteger, BitInteger> const target_qubit1, ::ket::qubit<StateInteger, BitInteger> const target_qubit2,
        ControlQubits const... control_qubits)
      -> RandomAccessRange&
      { using std::conj; return ::ket::gate::ranges::exponential_swap_coeff(state, conj(phase_coefficient), target_qubit1, target_qubit2, control_qubits...); }
    } // namespace ranges

    // exponential_swap
    template <typename ParallelPolicy, typename RandomAccessIterator, typename Real, typename StateInteger, typename BitInteger, typename... ControlQubits>
    inline auto exponential_swap(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last, Real const phase, // theta
      ::ket::qubit<StateInteger, BitInteger> const target_qubit1, ::ket::qubit<StateInteger, BitInteger> const target_qubit2,
      ControlQubits const... control_qubits)
    -> void
    {
      using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
      ::ket::gate::exponential_swap_coeff(parallel_policy, first, last, ::ket::utility::exp_i<complex_type>(phase), target_qubit1, target_qubit2, control_qubits...);
    }

    template <typename RandomAccessIterator, typename Real, typename StateInteger, typename BitInteger, typename... ControlQubits>
    inline auto exponential_swap(
      RandomAccessIterator const first, RandomAccessIterator const last, Real const phase, // theta
      ::ket::qubit<StateInteger, BitInteger> const target_qubit1, ::ket::qubit<StateInteger, BitInteger> const target_qubit2,
      ControlQubits const... control_qubits)
    -> void
    {
      using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
      ::ket::gate::exponential_swap_coeff(first, last, ::ket::utility::exp_i<complex_type>(phase), target_qubit1, target_qubit2, control_qubits...);
    }

    namespace ranges
    {
      template <typename ParallelPolicy, typename RandomAccessRange, typename Real, typename StateInteger, typename BitInteger, typename... ControlQubits>
      inline auto exponential_swap(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& state, Real const phase, // theta
        ::ket::qubit<StateInteger, BitInteger> const target_qubit1, ::ket::qubit<StateInteger, BitInteger> const target_qubit2,
        ControlQubits const... control_qubits)
      -> RandomAccessRange&
      {
        using complex_type = ::ket::utility::meta::range_value_t<RandomAccessRange>;
        return ::ket::gate::ranges::exponential_swap_coeff(parallel_policy, state, ::ket::utility::exp_i<complex_type>(phase), target_qubit1, target_qubit2, control_qubits...);
      }

      template <typename RandomAccessRange, typename Real, typename StateInteger, typename BitInteger, typename... ControlQubits>
      inline auto exponential_swap(
        RandomAccessRange& state, Real const phase, // theta
        ::ket::qubit<StateInteger, BitInteger> const target_qubit1, ::ket::qubit<StateInteger, BitInteger> const target_qubit2,
        ControlQubits const... control_qubits)
      -> RandomAccessRange&
      {
        using complex_type = ::ket::utility::meta::range_value_t<RandomAccessRange>;
        return ::ket::gate::ranges::exponential_swap_coeff(state, ::ket::utility::exp_i<complex_type>(phase), target_qubit1, target_qubit2, control_qubits...);
      }
    } // namespace ranges

    template <typename ParallelPolicy, typename RandomAccessIterator, typename Real, typename StateInteger, typename BitInteger, typename... ControlQubits>
    inline auto adj_exponential_swap(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last, Real const phase, // theta
      ::ket::qubit<StateInteger, BitInteger> const target_qubit1, ::ket::qubit<StateInteger, BitInteger> const target_qubit2,
      ControlQubits const... control_qubits)
    -> void
    { ::ket::gate::exponential_swap(parallel_policy, first, last, -phase, target_qubit1, target_qubit2, control_qubits...); }

    template <typename RandomAccessIterator, typename Real, typename StateInteger, typename BitInteger, typename... ControlQubits>
    inline auto adj_exponential_swap(
      RandomAccessIterator const first, RandomAccessIterator const last, Real const phase, // theta
      ::ket::qubit<StateInteger, BitInteger> const target_qubit1, ::ket::qubit<StateInteger, BitInteger> const target_qubit2,
      ControlQubits const... control_qubits)
    -> void
    { ::ket::gate::exponential_swap(first, last, -phase, target_qubit1, target_qubit2, control_qubits...); }

    namespace ranges
    {
      template <typename ParallelPolicy, typename RandomAccessRange, typename Real, typename StateInteger, typename BitInteger, typename... ControlQubits>
      inline auto adj_exponential_swap(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& state, Real const phase, // theta
        ::ket::qubit<StateInteger, BitInteger> const target_qubit1, ::ket::qubit<StateInteger, BitInteger> const target_qubit2,
        ControlQubits const... control_qubits)
      -> RandomAccessRange&
      { return ::ket::gate::ranges::exponential_swap(parallel_policy, state, -phase, target_qubit1, target_qubit2, control_qubits...); }

      template <typename RandomAccessRange, typename Real, typename StateInteger, typename BitInteger, typename... ControlQubits>
      inline auto adj_exponential_swap(
        RandomAccessRange& state, Real const phase, // theta
        ::ket::qubit<StateInteger, BitInteger> const target_qubit1, ::ket::qubit<StateInteger, BitInteger> const target_qubit2,
        ControlQubits const... control_qubits)
      -> RandomAccessRange&
      { return ::ket::gate::ranges::exponential_swap(state, -phase, target_qubit1, target_qubit2, control_qubits...); }
    } // namespace ranges
  } // namespace gate
} // namespace ket


#endif // KET_GATE_EXPONENTIAL_SWAP_HPP
