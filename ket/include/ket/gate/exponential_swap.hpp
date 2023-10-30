#ifndef KET_GATE_EXPONENTIAL_SWAP_HPP
# define KET_GATE_EXPONENTIAL_SWAP_HPP

# include <cassert>
# include <cmath>
# include <complex>
# include <iterator>
# include <algorithm>
# include <utility>
# include <type_traits>

# include <ket/qubit.hpp>
# include <ket/utility/loop_n.hpp>
# include <ket/utility/integer_exp2.hpp>
# ifndef NDEBUG
#   include <ket/utility/integer_log2.hpp>
# endif
# include <ket/utility/imaginary_unit.hpp>
# include <ket/utility/exp_i.hpp>


namespace ket
{
  namespace gate
  {
    // exponential_swap_coeff
    // eSWAP_{ij}(s) = exp(is SWAP_{ij}) = I cos s + i SWAP_{ij} sin s
    // eSWAP_{1,2}(s) (a_{00} |00> + a_{01} |01> + a_{10} |10> + a_{11} |11>)
    //   = e^{is} a_{00} |00> + (cos s a_{01} + i sin s a_{10}) |01> + (i sin s a_{01} + cos s a_{10}) |10> + e^{is} a_{11} |11>
    template <typename ParallelPolicy, typename RandomAccessIterator, typename Complex, typename StateInteger, typename BitInteger>
    inline void exponential_swap_coeff(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
      ::ket::qubit<StateInteger, BitInteger> const qubit1, ::ket::qubit<StateInteger, BitInteger> const qubit2)
    {
      static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
      static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");
      static_assert(
        (std::is_same<Complex, typename std::iterator_traits<RandomAccessIterator>::value_type>::value),
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

      using ::ket::utility::loop_n;
      loop_n(
        parallel_policy,
        static_cast<StateInteger>(last - first) >> 2u,
        [first, &phase_coefficient, qubit1_mask, qubit2_mask, lower_bits_mask, middle_bits_mask, upper_bits_mask](
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
          *qubit1_on_iter += *qubit2_on_iter * (::ket::utility::imaginary_unit<Complex>() * imag(phase_coefficient));
          *qubit2_on_iter *= real(phase_coefficient);
          *qubit2_on_iter += qubit1_on_iter_value * (::ket::utility::imaginary_unit<Complex>() * imag(phase_coefficient));
        });
    }

    template <typename RandomAccessIterator, typename Complex, typename StateInteger, typename BitInteger>
    inline void exponential_swap_coeff(
      RandomAccessIterator const first, RandomAccessIterator const last,
      Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
      ::ket::qubit<StateInteger, BitInteger> const qubit1, ::ket::qubit<StateInteger, BitInteger> const qubit2)
    { ::ket::gate::exponential_swap_coeff(::ket::utility::policy::make_sequential(), first, last, phase_coefficient, qubit1, qubit2); }

    namespace ranges
    {
      template <typename ParallelPolicy, typename RandomAccessRange, typename Complex, typename StateInteger, typename BitInteger>
      inline RandomAccessRange& exponential_swap_coeff(
        ParallelPolicy const parallel_policy, RandomAccessRange& state,
        Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
        ::ket::qubit<StateInteger, BitInteger> const qubit1, ::ket::qubit<StateInteger, BitInteger> const qubit2)
      {
        ::ket::gate::exponential_swap_coeff(parallel_policy, std::begin(state), std::end(state), phase_coefficient, qubit1, qubit2);
        return state;
      }

      template <typename RandomAccessRange, typename Complex, typename StateInteger, typename BitInteger>
      inline RandomAccessRange& exponential_swap_coeff(
        RandomAccessRange& state,
        Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
        ::ket::qubit<StateInteger, BitInteger> const qubit1, ::ket::qubit<StateInteger, BitInteger> const qubit2)
      { return ::ket::gate::ranges::exponential_swap_coeff(::ket::utility::policy::make_sequential(), state, phase_coefficient, qubit1, qubit2); }
    } // namespace ranges

    template <typename ParallelPolicy, typename RandomAccessIterator, typename Complex, typename StateInteger, typename BitInteger>
    inline void adj_exponential_swap_coeff(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
      ::ket::qubit<StateInteger, BitInteger> const qubit1, ::ket::qubit<StateInteger, BitInteger> const qubit2)
    {
      using std::conj;
      ::ket::gate::exponential_swap_coeff(parallel_policy, first, last, conj(phase_coefficient), qubit1, qubit2);
    }

    template <typename RandomAccessIterator, typename Complex, typename StateInteger, typename BitInteger>
    inline void adj_exponential_swap_coeff(
      RandomAccessIterator const first, RandomAccessIterator const last,
      Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
      ::ket::qubit<StateInteger, BitInteger> const qubit1, ::ket::qubit<StateInteger, BitInteger> const qubit2)
    {
      using std::conj;
      ::ket::gate::exponential_swap_coeff(first, last, conj(phase_coefficient), qubit1, qubit2);
    }

    namespace ranges
    {
      template <typename ParallelPolicy, typename RandomAccessRange, typename Complex, typename StateInteger, typename BitInteger>
      inline RandomAccessRange& adj_exponential_swap_coeff(
        ParallelPolicy const parallel_policy, RandomAccessRange& state,
        Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
        ::ket::qubit<StateInteger, BitInteger> const qubit1, ::ket::qubit<StateInteger, BitInteger> const qubit2)
      {
        using std::conj;
        return ::ket::gate::ranges::exponential_swap_coeff(parallel_policy, state, conj(phase_coefficient), qubit1, qubit2);
      }

      template <typename RandomAccessRange, typename Complex, typename StateInteger, typename BitInteger>
      inline RandomAccessRange& adj_exponential_swap_coeff(
        RandomAccessRange& state,
        Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
        ::ket::qubit<StateInteger, BitInteger> const qubit1, ::ket::qubit<StateInteger, BitInteger> const qubit2)
      {
        using std::conj;
        return ::ket::gate::ranges::exponential_swap_coeff(state, conj(phase_coefficient), qubit1, qubit2);
      }
    } // namespace ranges

    // exponential_swap
    template <typename ParallelPolicy, typename RandomAccessIterator, typename Real, typename StateInteger, typename BitInteger>
    inline void exponential_swap(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last, Real const phase, // theta
      ::ket::qubit<StateInteger, BitInteger> const qubit1, ::ket::qubit<StateInteger, BitInteger> const qubit2)
    {
      using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
      ::ket::gate::exponential_swap_coeff(parallel_policy, first, last, ::ket::utility::exp_i<complex_type>(phase), qubit1, qubit2);
    }

    template <typename RandomAccessIterator, typename Real, typename StateInteger, typename BitInteger>
    inline void exponential_swap(
      RandomAccessIterator const first, RandomAccessIterator const last, Real const phase, // theta
      ::ket::qubit<StateInteger, BitInteger> const qubit1, ::ket::qubit<StateInteger, BitInteger> const qubit2)
    {
      using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
      ::ket::gate::exponential_swap_coeff(first, last, ::ket::utility::exp_i<complex_type>(phase), qubit1, qubit2);
    }

    namespace ranges
    {
      template <typename ParallelPolicy, typename RandomAccessRange, typename Real, typename StateInteger, typename BitInteger>
      inline RandomAccessRange& exponential_swap(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& state, Real const phase, // theta
        ::ket::qubit<StateInteger, BitInteger> const qubit1, ::ket::qubit<StateInteger, BitInteger> const qubit2)
      {
        using complex_type = typename boost::range_value<RandomAccessRange>::type;
        return ::ket::gate::ranges::exponential_swap_coeff(parallel_policy, state, ::ket::utility::exp_i<complex_type>(phase), qubit1, qubit2);
      }

      template <typename RandomAccessRange, typename Real, typename StateInteger, typename BitInteger>
      inline RandomAccessRange& exponential_swap(
        RandomAccessRange& state, Real const phase, // theta
        ::ket::qubit<StateInteger, BitInteger> const qubit1, ::ket::qubit<StateInteger, BitInteger> const qubit2)
      {
        using complex_type = typename boost::range_value<RandomAccessRange>::type;
        return ::ket::gate::ranges::exponential_swap_coeff(state, ::ket::utility::exp_i<complex_type>(phase), qubit1, qubit2);
      }
    } // namespace ranges

    template <typename ParallelPolicy, typename RandomAccessIterator, typename Real, typename StateInteger, typename BitInteger>
    inline void adj_exponential_swap(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last, Real const phase, // theta
      ::ket::qubit<StateInteger, BitInteger> const qubit1, ::ket::qubit<StateInteger, BitInteger> const qubit2)
    { ::ket::gate::exponential_swap(parallel_policy, first, last, -phase, qubit1, qubit2); }

    template <typename RandomAccessIterator, typename Real, typename StateInteger, typename BitInteger>
    inline void adj_exponential_swap(
      RandomAccessIterator const first, RandomAccessIterator const last, Real const phase, // theta
      ::ket::qubit<StateInteger, BitInteger> const qubit1, ::ket::qubit<StateInteger, BitInteger> const qubit2)
    { ::ket::gate::exponential_swap(first, last, -phase, qubit1, qubit2); }

    namespace ranges
    {
      template <typename ParallelPolicy, typename RandomAccessRange, typename Real, typename StateInteger, typename BitInteger>
      inline RandomAccessRange& adj_exponential_swap(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& state, Real const phase, // theta
        ::ket::qubit<StateInteger, BitInteger> const qubit1, ::ket::qubit<StateInteger, BitInteger> const qubit2)
      { return ::ket::gate::ranges::exponential_swap(parallel_policy, state, -phase, qubit1, qubit2); }

      template <typename RandomAccessRange, typename Real, typename StateInteger, typename BitInteger>
      inline RandomAccessRange& adj_exponential_swap(
        RandomAccessRange& state, Real const phase, // theta
        ::ket::qubit<StateInteger, BitInteger> const qubit1, ::ket::qubit<StateInteger, BitInteger> const qubit2)
      { return ::ket::gate::ranges::exponential_swap(state, -phase, qubit1, qubit2); }
    } // namespace ranges
  } // namespace gate
} // namespace ket


#endif // KET_GATE_EXPONENTIAL_SWAP_HPP
