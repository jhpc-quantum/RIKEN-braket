#ifndef KET_GATE_EXPONENTIAL_PAULI_Z_HPP
# define KET_GATE_EXPONENTIAL_PAULI_Z_HPP

# include <cassert>
# include <cmath>
# include <complex>
# include <array>
# include <iterator>
# include <utility>
# include <type_traits>

# include <ket/qubit.hpp>
# include <ket/gate/gate.hpp>
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
    // exponential_pauli_z_coeff
    // eZ_i(s) = exp(is Z_i) = I cos s + i Z_i sin s
    // eZ_1(s) (a_0 |0> + a_1 |1>) = e^{is} a_0 |0> + e^{-is} a_1 |1>
    template <typename ParallelPolicy, typename RandomAccessIterator, typename Complex, typename StateInteger, typename BitInteger>
    inline void exponential_pauli_z_coeff(
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

      using std::conj;
      auto const conj_phase_coefficient = conj(phase_coefficient);

      using ::ket::utility::loop_n;
      loop_n(
        parallel_policy,
        static_cast<StateInteger>(last - first) >> 1u,
        [first, &phase_coefficient, &conj_phase_coefficient, qubit_mask, lower_bits_mask, upper_bits_mask](
          StateInteger const value_wo_qubits, int const)
        {
          // xxxxx0xxxxxx
          auto const zero_index
            = ((value_wo_qubit bitand upper_bits_mask) << 1u)
              bitor (value_wo_qubit bitand lower_bits_mask);
          // xxxxx1xxxxxx
          auto const one_index = zero_index bitor qubit_mask;

          *(first + zero_index) *= phase_coefficient;
          *(first + one_index) *= conj_phase_coefficient;
        });
    }

    // eZZ_{ij}(s) = exp(is Z_i Z_j) = I cos s + i Z_i Z_j sin s
    // eZZ_{1,2}(s) (a_{00} |00> + a_{01} |01> + a_{10} |10> + a{11} |11>)
    //   = e^{is} |00> + e^{-is} |01> + e^{-is} |10> + e^{is} |11>
    template <typename ParallelPolicy, typename RandomAccessIterator, typename Complex, typename StateInteger, typename BitInteger>
    inline void exponential_pauli_z_coeff(
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

      using std::conj;
      auto const conj_phase_coefficient = conj(phase_coefficient);

      using ::ket::utility::loop_n;
      loop_n(
        parallel_policy,
        static_cast<StateInteger>(last - first) >> 2u,
        [first, &phase_coefficient, &conj_phase_coefficient, qubit1_mask, qubit2_mask, lower_bits_mask, middle_bits_mask, upper_bits_mask](
          StateInteger const value_wo_qubits, int const)
        {
          // xxx0_1xxx0_2xxx
          auto const base_index
            = ((value_wo_qubits bitand upper_bits_mask) << 2u)
              bitor ((value_wo_qubits bitand middle_bits_mask) << 1u)
              bitor (value_wo_qubits bitand lower_bits_mask);
          // xxx1_1xxx0_2xxx
          auto const qubit1_on_index = base_index bitor qubit1_mask;
          // xxx0_1xxx1_2xxx
          auto const qubit2_on_index = base_index bitor qubit2_mask;
          // xxx1_1xxx1_2xxx
          auto const qubit12_on_index = qubit1_on_index bitor qubit2_mask;

          *(first + base_index) *= phase_coefficient;
          *(first + qubit1_on_index) *= conj_phase_coefficient;
          *(first + qubit2_on_index) *= conj_phase_coefficient;
          *(first + qubit12_on_index) *= phase_coefficient;
        });
    }

    // eZ...Z_{i...j}(s) = exp(is Z_i ... Z_j) = I cos s + i Z_i ... Z_j sin s
    //   (Z_1...Z_N)_{nn} = (-1)^f(n-1) for 1<=n<=2^N, where f(n): num. of "1" bits in n
    template <typename ParallelPolicy, typename RandomAccessIterator, typename Complex, typename StateInteger, typename BitInteger, typename... Qubits>
    inline void exponential_pauli_z_coeff(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
      ::ket::qubit<StateInteger, BitInteger> const qubit1, ::ket::qubit<StateInteger, BitInteger> const qubit2,
      ::ket::qubit<StateInteger, BitInteger> const qubit3, Qubits const... qubits)
    {
      constexpr auto num_qubits = static_cast<BitInteger>(sizeof...(Qubits) + 3u);
      constexpr auto num_indices = ::ket::utility::integer_exp2<std::size_t>(num_qubits);

      using std::conj;
      auto const conj_phase_coefficient = conj(phase_coefficient);

      ::ket::gate::gate(
        parallel_policy, first, last,
        [&phase_coefficient, &conj_phase_coefficient](
          RandomAccessIterator const first, std::array<StateInteger, num_indices> const& indices, int const)
        {
          auto const num_indices = static_cast<StateInteger>(boost::size(indices));
          for (auto i = StateInteger{0u}; i < num_indices; ++i)
          {
            auto num_ones_in_i = BitInteger{0u};
            auto i_tmp = i;
            for (auto count = BitInteger{0u}; count < num_qubits; ++count)
            {
              if (i_tmp bitand StateInteger{1u} == StateInteger{1u})
                ++num_ones_in_i;

              i_tmp >>= BitInteger{1u};
            }

            *(first + indices[i]) *= num_ones_in_i % BitInteger{2u} == BitInteger{0u} ? phase_coefficient : conj_phase_coefficient;
          }
        },
        qubit1, qubit2, qubit3, qubits...);
    }

    template <typename RandomAccessIterator, typename Complex, typename StateInteger, typename BitInteger, typename... Qubits>
    inline void exponential_pauli_z_coeff(
      RandomAccessIterator const first, RandomAccessIterator const last,
      Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
      ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
    { ::ket::gate::exponential_pauli_z_coeff(::ket::utility::policy::make_sequential(), first, last, phase_coefficient, qubit, qubits...); }

    namespace ranges
    {
      template <typename ParallelPolicy, typename RandomAccessRange, typename Complex, typename StateInteger, typename BitInteger, typename... Qubits>
      inline RandomAccessRange& exponential_pauli_z_coeff(
        ParallelPolicy const parallel_policy, RandomAccessRange& state,
        Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
        ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
      {
        ::ket::gate::exponential_pauli_z_coeff(parallel_policy, std::begin(state), std::end(state), phase_coefficient, qubit, qubits...);
        return state;
      }

      template <typename RandomAccessRange, typename Complex, typename StateInteger, typename BitInteger, typename... Qubits>
      inline RandomAccessRange& exponential_pauli_z_coeff(
        RandomAccessRange& state,
        Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
        ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
      { return ::ket::gate::ranges::exponential_pauli_z_coeff(::ket::utility::policy::make_sequential(), state, phase_coefficient, qubit, qubits...); }
    } // namespace ranges

    template <typename ParallelPolicy, typename RandomAccessIterator, typename Complex, typename StateInteger, typename BitInteger, typename... Qubits>
    inline void adj_exponential_pauli_z_coeff(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
      ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
    {
      using std::conj;
      ::ket::gate::exponential_pauli_z_coeff(parallel_policy, first, last, conj(phase_coefficient), qubit, qubits...);
    }

    template <typename RandomAccessIterator, typename Complex, typename StateInteger, typename BitInteger, typename... Qubits>
    inline void adj_exponential_pauli_z_coeff(
      RandomAccessIterator const first, RandomAccessIterator const last,
      Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
      ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
    {
      using std::conj;
      ::ket::gate::exponential_pauli_z_coeff(first, last, conj(phase_coefficient), qubit, qubits...);
    }

    namespace ranges
    {
      template <typename ParallelPolicy, typename RandomAccessRange, typename Complex, typename StateInteger, typename BitInteger, typename... Qubits>
      inline RandomAccessRange& adj_exponential_pauli_z_coeff(
        ParallelPolicy const parallel_policy, RandomAccessRange& state,
        Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
        ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
      {
        using std::conj;
        return ::ket::gate::ranges::exponential_pauli_z_coeff(parallel_policy, state, conj(phase_coefficient), qubit, qubits...);
      }

      template <typename RandomAccessRange, typename Complex, typename StateInteger, typename BitInteger, typename... Qubits>
      inline RandomAccessRange& adj_exponential_pauli_z_coeff(
        RandomAccessRange& state,
        Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
        ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
      {
        using std::conj;
        return ::ket::gate::ranges::exponential_pauli_z_coeff(state, conj(phase_coefficient), qubit, qubits...);
      }
    } // namespace ranges

    // exponential_pauli_z
    template <typename ParallelPolicy, typename RandomAccessIterator, typename Real, typename StateInteger, typename BitInteger, typename... Qubits>
    inline void exponential_pauli_z(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last, Real const phase, // theta
      ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
    {
      using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
      ::ket::gate::exponential_pauli_z_coeff(parallel_policy, first, last, ::ket::utility::exp_i<complex_type>(phase), qubit, qubits...);
    }

    template <typename RandomAccessIterator, typename Real, typename StateInteger, typename BitInteger, typename... Qubits>
    inline void exponential_pauli_z(
      RandomAccessIterator const first, RandomAccessIterator const last, Real const phase, // theta
      ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const.. qubits)
    {
      using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
      ::ket::gate::exponential_pauli_z_coeff(first, last, ::ket::utility::exp_i<complex_type>(phase), qubit, qubits...);
    }

    namespace ranges
    {
      template <typename ParallelPolicy, typename RandomAccessRange, typename Real, typename StateInteger, typename BitInteger, typename... Qubits>
      inline RandomAccessRange& exponential_pauli_z(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& state, Real const phase, // theta
        ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
      {
        using complex_type = typename boost::range_value<RandomAccessRange>::type;
        return ::ket::gate::ranges::exponential_pauli_z_coeff(parallel_policy, state, ::ket::utility::exp_i<complex_type>(phase), qubit, qubits...);
      }

      template <typename RandomAccessRange, typename Real, typename StateInteger, typename BitInteger, typename... Qubits>
      inline RandomAccessRange& exponential_pauli_z(
        RandomAccessRange& state, Real const phase, // theta
        ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
      {
        using complex_type = typename boost::range_value<RandomAccessRange>::type;
        return ::ket::gate::ranges::exponential_pauli_z_coeff(state, ::ket::utility::exp_i<complex_type>(phase), qubit, qubits...);
      }
    } // namespace ranges

    template <typename ParallelPolicy, typename RandomAccessIterator, typename Real, typename StateInteger, typename BitInteger, typename... Qubits>
    inline void adj_exponential_pauli_z(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last, Real const phase, // theta
      ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
    { ::ket::gate::exponential_pauli_z(parallel_policy, first, last, -phase, qubit, qubits...); }

    template <typename RandomAccessIterator, typename Real, typename StateInteger, typename BitInteger, typename... Qubits>
    inline void adj_exponential_pauli_z(
      RandomAccessIterator const first, RandomAccessIterator const last, Real const phase, // theta
      ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
    { ::ket::gate::exponential_pauli_z(first, last, -phase, qubit, qubits...); }

    namespace ranges
    {
      template <typename ParallelPolicy, typename RandomAccessRange, typename Real, typename StateInteger, typename BitInteger, typename... Qubits>
      inline RandomAccessRange& adj_exponential_pauli_z(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& state, Real const phase, // theta
        ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
      { return ::ket::gate::ranges::exponential_pauli_z(parallel_policy, state, -phase, qubit, qubits...); }

      template <typename RandomAccessRange, typename Real, typename StateInteger, typename BitInteger, typename... Qubits>
      inline RandomAccessRange& adj_exponential_pauli_z(
        RandomAccessRange& state, Real const phase, // theta
        ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
      { return ::ket::gate::ranges::exponential_pauli_z(state, -phase, qubit, qubits...); }
    } // namespace ranges
  } // namespace gate
} // namespace ket


#endif // KET_GATE_EXPONENTIAL_PAULI_Z_HPP
