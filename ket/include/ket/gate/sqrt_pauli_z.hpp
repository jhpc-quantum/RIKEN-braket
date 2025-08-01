#ifndef KET_GATE_SQRT_PAULI_Z_HPP
# define KET_GATE_SQRT_PAULI_Z_HPP

# include <cassert>
# include <cstddef>
# include <cstdint>
# include <array>
# include <algorithm>
# include <iterator>
# include <utility>
# include <type_traits>

# include <ket/qubit.hpp>
# include <ket/control.hpp>
# include <ket/gate/gate.hpp>
# include <ket/gate/utility/index_with_qubits.hpp>
# include <ket/gate/meta/num_control_qubits.hpp>
# include <ket/utility/imaginary_unit.hpp>
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
    // Case 1: the first argument of qubits is ket::control<ket::qubit<S, B>>
    template <typename ParallelPolicy, typename RandomAccessIterator>
    inline auto sqrt_pauli_z(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last)
    -> void
    {
      assert(
        ::ket::utility::integer_exp2<std::uint64_t>(::ket::utility::integer_log2<std::size_t>(last - first))
        == static_cast<std::uint64_t>(last - first));

      ::ket::utility::loop_n(
        parallel_policy, static_cast<std::uint64_t>(last - first),
        [first](std::uint64_t const index, int const)
        {
          using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
          *(first + index) *= ::ket::utility::imaginary_unit<complex_type>();
        });
    }

    // sZ_i
    // sZ_1 (a_0 |0> + a_1 |1>) = a_0 |0> + i a_1 |1>
    template <typename ParallelPolicy, typename RandomAccessIterator, typename StateInteger, typename BitInteger>
    inline auto sqrt_pauli_z(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit)
    -> void
    {
      static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
      static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");
      assert(::ket::utility::integer_exp2<StateInteger>(control_qubit) < static_cast<StateInteger>(last - first));
      assert(
        ::ket::utility::integer_exp2<StateInteger>(::ket::utility::integer_log2<BitInteger>(last - first))
        == static_cast<StateInteger>(last - first));

      auto const qubit_mask = ::ket::utility::integer_exp2<StateInteger>(control_qubit);
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

          using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
          *(first + one_index) *= ::ket::utility::imaginary_unit<complex_type>();
        });
    }

    // CsZ_{cc'} or C1sZ_{cc'}
    // CsZ_{1,2} (a_{00} |00> + a_{01} |01> + a_{10} |10> + a_{11} |11>)
    //   = a_{00} |00> + a_{01} |01> + a_{10} |10> + i a_{11} |11>
    template <typename ParallelPolicy, typename RandomAccessIterator, typename StateInteger, typename BitInteger>
    inline auto sqrt_pauli_z(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2)
    -> void
    {
      static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
      static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");

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
        [first, control_qubit1_mask, control_qubit2_mask, lower_bits_mask, middle_bits_mask, upper_bits_mask](
          StateInteger const value_wo_qubits, int const)
        {
          // xxx0_c1xxx0_c2xxx
          auto const index00
            = ((value_wo_qubits bitand upper_bits_mask) << 2u)
              bitor ((value_wo_qubits bitand middle_bits_mask) << 1u)
              bitor (value_wo_qubits bitand lower_bits_mask);
          // xxx1_c1xxx1_c2xxx
          auto const index11 = index00 bitor control_qubit1_mask bitor control_qubit2_mask;

          using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
          *(first + index11) *= ::ket::utility::imaginary_unit<complex_type>();
        });
    }

    // C...CsZ_{c0,c...c'} or CnsZ_{c0,c...c'}
    template <typename ParallelPolicy, typename RandomAccessIterator, typename StateInteger, typename BitInteger, typename... Qubits>
    inline auto sqrt_pauli_z(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit3,
      ::ket::control<Qubits> const... control_qubits)
    -> void
    {
      static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
      static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");
      assert(
        ::ket::utility::integer_exp2<StateInteger>(::ket::utility::integer_log2<BitInteger>(last - first))
        == static_cast<StateInteger>(last - first));

# ifndef KET_USE_BIT_MASKS_EXPLICITLY
      using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
      constexpr auto num_control_qubits = BitInteger{sizeof...(Qubits) + 3u};
      constexpr auto num_operated_qubits = num_control_qubits;

      ::ket::gate::nocache::gate(
        parallel_policy, first, last,
        [](
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
          using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
          *iter *= ::ket::utility::imaginary_unit<complex_type>();
        },
        control_qubit1, control_qubit2, control_qubit3, control_qubits...);
# else // KET_USE_BIT_MASKS_EXPLICITLY
      constexpr auto num_control_qubits = BitInteger{sizeof...(Qubits) + 3u};
      constexpr auto num_operated_qubits = num_control_qubits;

      ::ket::gate::nocache::gate(
        parallel_policy, first, last,
        [](
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
          using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
          *iter *= ::ket::utility::imaginary_unit<complex_type>();
        },
        control_qubit1, control_qubit2, control_qubit3, control_qubits...);
# endif // KET_USE_BIT_MASKS_EXPLICITLY
    }

    template <typename RandomAccessIterator, typename... Qubits>
    inline auto sqrt_pauli_z(
      RandomAccessIterator const first, RandomAccessIterator const last,
      ::ket::control<Qubits> const... control_qubits)
    -> void
    { ::ket::gate::sqrt_pauli_z(::ket::utility::policy::make_sequential(), first, last, control_qubits...); }

    namespace ranges
    {
      template <typename ParallelPolicy, typename RandomAccessRange, typename... Qubits>
      inline auto sqrt_pauli_z(
        ParallelPolicy const parallel_policy, RandomAccessRange& state,
        ::ket::control<Qubits> const... control_qubits)
      -> RandomAccessRange&
      {
        using std::begin;
        using std::end;
        ::ket::gate::sqrt_pauli_z(parallel_policy, begin(state), end(state), control_qubits...);
        return state;
      }

      template <typename RandomAccessRange, typename... Qubits>
      inline auto sqrt_pauli_z(RandomAccessRange& state, ::ket::control<Qubits> const... control_qubits) -> RandomAccessRange&
      { return ::ket::gate::ranges::sqrt_pauli_z(::ket::utility::policy::make_sequential(), state, control_qubits...); }
    } // namespace ranges

    template <typename ParallelPolicy, typename RandomAccessIterator>
    inline auto adj_sqrt_pauli_z(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last)
    -> void
    {
      assert(
        ::ket::utility::integer_exp2<std::uint64_t>(::ket::utility::integer_log2<std::size_t>(last - first))
        == static_cast<std::uint64_t>(last - first));

      ::ket::utility::loop_n(
        parallel_policy, static_cast<std::uint64_t>(last - first),
        [first](std::uint64_t const index, int const)
        {
          using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
          *(first + index) *= ::ket::utility::minus_imaginary_unit<complex_type>();
        });
    }

    // sZ+_i
    // sZ+_1 (a_0 |0> + a_1 |1>) = a_0 |0> - i a_1 |1>
    template <typename ParallelPolicy, typename RandomAccessIterator, typename StateInteger, typename BitInteger>
    inline auto adj_sqrt_pauli_z(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit)
    -> void
    {
      static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
      static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");
      assert(::ket::utility::integer_exp2<StateInteger>(control_qubit) < static_cast<StateInteger>(last - first));
      assert(
        ::ket::utility::integer_exp2<StateInteger>(::ket::utility::integer_log2<BitInteger>(last - first))
        == static_cast<StateInteger>(last - first));

      auto const qubit_mask = ::ket::utility::integer_exp2<StateInteger>(control_qubit);
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

          using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
          *(first + one_index) *= ::ket::utility::minus_imaginary_unit<complex_type>();
        });
    }

    // CsZ+_{cc'} or C1sZ+_{cc'}
    // CsZ+_{1,2} (a_{00} |00> + a_{01} |01> + a_{10} |10> + a_{11} |11>)
    //   = a_{00} |00> + a_{01} |01> + a_{10} |10> - i a_{11} |11>
    template <typename ParallelPolicy, typename RandomAccessIterator, typename StateInteger, typename BitInteger>
    inline auto adj_sqrt_pauli_z(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2)
    -> void
    {
      static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
      static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");

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
        [first, control_qubit1_mask, control_qubit2_mask, lower_bits_mask, middle_bits_mask, upper_bits_mask](
          StateInteger const value_wo_qubits, int const)
        {
          // xxx0_c1xxx0_c2xxx
          auto const index00
            = ((value_wo_qubits bitand upper_bits_mask) << 2u)
              bitor ((value_wo_qubits bitand middle_bits_mask) << 1u)
              bitor (value_wo_qubits bitand lower_bits_mask);
          // xxx1_c1xxx1_c2xxx
          auto const index11 = index00 bitor control_qubit1_mask bitor control_qubit2_mask;

          using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
          *(first + index11) *= ::ket::utility::minus_imaginary_unit<complex_type>();
        });
    }

    // C...CsZ+_{c0,c...c'} or CnsZ+_{c0,c...c'}
    template <typename ParallelPolicy, typename RandomAccessIterator, typename StateInteger, typename BitInteger, typename... Qubits>
    inline auto adj_sqrt_pauli_z(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit3,
      ::ket::control<Qubits> const... control_qubits)
    -> void
    {
      static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
      static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");
      assert(
        ::ket::utility::integer_exp2<StateInteger>(::ket::utility::integer_log2<BitInteger>(last - first))
        == static_cast<StateInteger>(last - first));

# ifndef KET_USE_BIT_MASKS_EXPLICITLY
      using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
      constexpr auto num_control_qubits = BitInteger{sizeof...(Qubits) + 3u};
      constexpr auto num_operated_qubits = num_control_qubits;

      ::ket::gate::nocache::gate(
        parallel_policy, first, last,
        [](
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
          using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
          *iter *= ::ket::utility::minus_imaginary_unit<complex_type>();
        },
        control_qubit1, control_qubit2, control_qubit3, control_qubits...);
# else // KET_USE_BIT_MASKS_EXPLICITLY
      constexpr auto num_control_qubits = BitInteger{sizeof...(Qubits) + 3u};
      constexpr auto num_operated_qubits = num_control_qubits;

      ::ket::gate::nocache::gate(
        parallel_policy, first, last,
        [](
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
          using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
          *iter *= ::ket::utility::minus_imaginary_unit<complex_type>();
        },
        control_qubit1, control_qubit2, control_qubit3, control_qubits...);
# endif // KET_USE_BIT_MASKS_EXPLICITLY
    }

    template <typename RandomAccessIterator, typename... Qubits>
    inline auto adj_sqrt_pauli_z(
      RandomAccessIterator const first, RandomAccessIterator const last,
      ::ket::control<Qubits> const... control_qubits)
    -> void
    { ::ket::gate::adj_sqrt_pauli_z(::ket::utility::policy::make_sequential(), first, last, control_qubits...); }

    namespace ranges
    {
      template <typename ParallelPolicy, typename RandomAccessRange, typename... Qubits>
      inline auto adj_sqrt_pauli_z(
        ParallelPolicy const parallel_policy, RandomAccessRange& state,
        ::ket::control<Qubits> const... control_qubits)
      -> RandomAccessRange&
      {
        using std::begin;
        using std::end;
        ::ket::gate::adj_sqrt_pauli_z(parallel_policy, begin(state), end(state), control_qubits...);
        return state;
      }

      template <typename RandomAccessRange, typename... Qubits>
      inline auto adj_sqrt_pauli_z(RandomAccessRange& state, ::ket::control<Qubits> const... control_qubits) -> RandomAccessRange&
      { return ::ket::gate::ranges::adj_sqrt_pauli_z(::ket::utility::policy::make_sequential(), state, control_qubits...); }
    } // namespace ranges

    // Case 2: the first argument of qubits is ket::qubit<S, B>
    // sZ_i
    // sZ_1 (a_0 |0> + a_1 |1>) = a_0 |0> + i a_1 |1>
    template <typename ParallelPolicy, typename RandomAccessIterator, typename StateInteger, typename BitInteger>
    inline auto sqrt_pauli_z(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last, ::ket::qubit<StateInteger, BitInteger> const qubit)
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

          using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
          *(first + one_index) *= ::ket::utility::imaginary_unit<complex_type>();
        });
    }

    // sZZ_i = sZ_i sZ_j or sZ2_{ij}
    // sZZ_{1,2} (a_{00} |00> + a_{01} |01> + a_{10} |10> + a{11} |11>)
    //   = a_{00} |00> + i a_{01} |01> + i a_{10} |10> - a{11} |11>
    template <typename ParallelPolicy, typename RandomAccessIterator, typename StateInteger, typename BitInteger>
    inline auto sqrt_pauli_z(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      ::ket::qubit<StateInteger, BitInteger> const qubit1, ::ket::qubit<StateInteger, BitInteger> const qubit2)
    -> void
    {
      static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
      static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");

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

      ::ket::utility::loop_n(
        parallel_policy, static_cast<StateInteger>(last - first) >> 2u,
        [first, qubit1_mask, qubit2_mask, lower_bits_mask, middle_bits_mask, upper_bits_mask](
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

          using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
          using real_type = ::ket::utility::meta::real_t<complex_type>;
          *qubit1_on_iter *= ::ket::utility::imaginary_unit<complex_type>();
          *qubit2_on_iter *= ::ket::utility::imaginary_unit<complex_type>();
          *qubit12_on_iter *= real_type{-1};
        });
    }

    // CsZ_{tc} or C1sZ_{tc}
    // CsZ_{1,2} (a_{00} |00> + a_{01} |01> + a_{10} |10> + a_{11} |11>)
    //   = a_{00} |00> + a_{01} |01> + a_{10} |10> + i a_{11} |11>
    template <typename ParallelPolicy, typename RandomAccessIterator, typename StateInteger, typename BitInteger>
    inline auto sqrt_pauli_z(
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
          // xxx1_txxx1_cxxx
          auto const target_control_on_index = base_index bitor control_qubit_mask bitor target_qubit_mask;

          using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
          *(first + target_control_on_index) *= ::ket::utility::imaginary_unit<complex_type>();
        });
    }

    // C...CsZ...Z_{t...t'c...c'} = C...C(sZ_t ... sZ_t')_{c...c'}, CnsZ...Z_{...}, C...CsZm_{...}, or CnsZm_{...}
    //   (sZ_1...sZ_N)_{nn} = i^f(n-1) for 1<=n<=2^N, where f(n): the number of "1" bits in n
    template <typename ParallelPolicy, typename RandomAccessIterator, typename StateInteger, typename BitInteger, typename Qubit2, typename Qubit3, typename... Qubits>
    inline auto sqrt_pauli_z(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      ::ket::qubit<StateInteger, BitInteger> const qubit1, Qubit2 const qubit2, Qubit3 const qubit3, Qubits const... qubits)
    -> void
    {
      static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
      static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");
      assert(
        ::ket::utility::integer_exp2<StateInteger>(::ket::utility::integer_log2<BitInteger>(last - first))
        == static_cast<StateInteger>(last - first));

# ifndef KET_USE_BIT_MASKS_EXPLICITLY
      using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
      constexpr auto num_operated_qubits = static_cast<BitInteger>(sizeof...(Qubits) + 3u);
      constexpr auto num_control_qubits = ::ket::gate::meta::num_control_qubits<BitInteger, Qubit2, Qubit3, Qubits...>::value;
      constexpr auto num_target_qubits = num_operated_qubits - num_control_qubits;
      constexpr auto num_target_indices = ::ket::utility::integer_exp2<std::size_t>(num_target_qubits);

      ::ket::gate::nocache::gate(
        parallel_policy, first, last,
        [](
          auto const first, StateInteger const index_wo_qubits,
          std::array<qubit_type, num_operated_qubits> const& unsorted_qubits,
          std::array<qubit_type, num_operated_qubits + BitInteger{1u}> const& sorted_qubits_with_sentinel,
          int const)
        {
          // 0b1...10...0u
          constexpr auto base_index = ((std::size_t{1u} << num_control_qubits) - std::size_t{1u}) << num_target_qubits;

          for (auto i = std::size_t{0u}; i < num_target_indices; ++i)
          {
            auto num_ones_in_i = BitInteger{0u};
            auto i_tmp = i;
            for (auto count = BitInteger{0u}; count < num_target_qubits; ++count)
            {
              if ((i_tmp bitand StateInteger{1u}) == StateInteger{1u})
                ++num_ones_in_i;

              i_tmp >>= BitInteger{1u};
            }

            using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
            using real_type = ::ket::utility::meta::real_t<complex_type>;
            auto const remainder = num_ones_in_i % BitInteger{4u};
            auto const coefficient
              = remainder == BitInteger{0u}
                ? complex_type{real_type{1}}
                : remainder == BitInteger{1u}
                  ? ::ket::utility::imaginary_unit<complex_type>()
                  : remainder == BitInteger{2u}
                    ? complex_type{real_type{-1}}
                    : ::ket::utility::minus_imaginary_unit<complex_type>();

            using std::begin;
            using std::end;
            auto const iter
              = first
                + ::ket::gate::utility::index_with_qubits(
                    index_wo_qubits, base_index + i,
                    begin(unsorted_qubits), end(unsorted_qubits),
                    begin(sorted_qubits_with_sentinel), end(sorted_qubits_with_sentinel));
            *iter *= coefficient;
          }
        },
        qubit1, qubit2, qubit3, qubits...);
# else // KET_USE_BIT_MASKS_EXPLICITLY
      constexpr auto num_operated_qubits = static_cast<BitInteger>(sizeof...(Qubits) + 3u);
      constexpr auto num_control_qubits = ::ket::gate::meta::num_control_qubits<BitInteger, Qubit2, Qubit3, Qubits...>::value;
      constexpr auto num_target_qubits = num_operated_qubits - num_control_qubits;
      constexpr auto num_target_indices = ::ket::utility::integer_exp2<std::size_t>(num_target_qubits);

      ::ket::gate::nocache::gate(
        parallel_policy, first, last,
        [](
          auto const first, StateInteger const index_wo_qubits,
          std::array<StateInteger, num_operated_qubits> const& qubit_masks,
          std::array<StateInteger, num_operated_qubits + 1u> const& index_masks,
          int const)
        {
          // 0b1...10...0u
          constexpr auto base_index = ((std::size_t{1u} << num_control_qubits) - std::size_t{1u}) << num_target_qubits;

          for (auto i = std::size_t{0u}; i < num_target_indices; ++i)
          {
            auto num_ones_in_i = BitInteger{0u};
            auto i_tmp = i;
            for (auto count = BitInteger{0u}; count < num_target_qubits; ++count)
            {
              if ((i_tmp bitand StateInteger{1u}) == StateInteger{1u})
                ++num_ones_in_i;

              i_tmp >>= BitInteger{1u};
            }

            using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
            using real_type = ::ket::utility::meta::real_t<complex_type>;
            auto const remainder = num_ones_in_i % BitInteger{4u};
            auto const coefficient
              = remainder == BitInteger{0u}
                ? complex_type{real_type{1}}
                : remainder == BitInteger{1u}
                  ? ::ket::utility::imaginary_unit<complex_type>()
                  : remainder == BitInteger{2u}
                    ? complex_type{real_type{-1}}
                    : ::ket::utility::minus_imaginary_unit<complex_type>();

            using std::begin;
            using std::end;
            auto const iter
              = first
                + ::ket::gate::utility::index_with_qubits(
                    index_wo_qubits, base_index + i,
                    begin(qubit_masks), end(qubit_masks), begin(index_masks), end(index_masks));
            *iter *= coefficient;
          }
        },
        qubit1, qubit2, qubit3, qubits...);
# endif // KET_USE_BIT_MASKS_EXPLICITLY
    }

    template <typename RandomAccessIterator, typename StateInteger, typename BitInteger, typename... Qubits>
    inline auto sqrt_pauli_z(
      RandomAccessIterator const first, RandomAccessIterator const last,
      ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
    -> void
    { ::ket::gate::sqrt_pauli_z(::ket::utility::policy::make_sequential(), first, last, qubit, qubits...); }

    namespace ranges
    {
      template <typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename... Qubits>
      inline auto sqrt_pauli_z(
        ParallelPolicy const parallel_policy, RandomAccessRange& state,
        ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
      -> RandomAccessRange&
      {
        using std::begin;
        using std::end;
        ::ket::gate::sqrt_pauli_z(parallel_policy, begin(state), end(state), qubit, qubits...);
        return state;
      }

      template <typename RandomAccessRange, typename StateInteger, typename BitInteger, typename... Qubits>
      inline auto sqrt_pauli_z(RandomAccessRange& state, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits) -> RandomAccessRange&
      { return ::ket::gate::ranges::sqrt_pauli_z(::ket::utility::policy::make_sequential(), state, qubit, qubits...); }
    } // namespace ranges

    // sZ+_i
    // sZ+_1 (a_0 |0> + a_1 |1>) = a_0 |0> - i a_1 |1>
    template <typename ParallelPolicy, typename RandomAccessIterator, typename StateInteger, typename BitInteger>
    inline auto adj_sqrt_pauli_z(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last, ::ket::qubit<StateInteger, BitInteger> const qubit)
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

          using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
          *(first + one_index) *= ::ket::utility::minus_imaginary_unit<complex_type>();
        });
    }

    // sZZ+_{ij} = sZ+_i sZ+_j or sZ2+_{ij}
    // sZZ+_{1,2} (a_{00} |00> + a_{01} |01> + a_{10} |10> + a{11} |11>)
    //   = a_{00} |00> - i a_{01} |01> - i a_{10} |10> - a{11} |11>
    template <typename ParallelPolicy, typename RandomAccessIterator, typename StateInteger, typename BitInteger>
    inline auto adj_sqrt_pauli_z(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      ::ket::qubit<StateInteger, BitInteger> const qubit1, ::ket::qubit<StateInteger, BitInteger> const qubit2)
    -> void
    {
      static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
      static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");

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

      ::ket::utility::loop_n(
        parallel_policy, static_cast<StateInteger>(last - first) >> 2u,
        [first, qubit1_mask, qubit2_mask, lower_bits_mask, middle_bits_mask, upper_bits_mask](
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

          using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
          using real_type = ::ket::utility::meta::real_t<complex_type>;
          *qubit1_on_iter *= ::ket::utility::minus_imaginary_unit<complex_type>();
          *qubit2_on_iter *= ::ket::utility::minus_imaginary_unit<complex_type>();
          *qubit12_on_iter *= real_type{-1};
        });
    }

    // CsZ+_{tc} or C1sZ+_{tc}
    // CsZ+_{1,2} (a_{00} |00> + a_{01} |01> + a_{10} |10> + a_{11} |11>)
    //   = a_{00} |00> + a_{01} |01> + a_{10} |10> - i a_{11} |11>
    template <typename ParallelPolicy, typename RandomAccessIterator, typename StateInteger, typename BitInteger>
    inline auto adj_sqrt_pauli_z(
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
          // xxx1_txxx1_cxxx
          auto const target_control_on_index = base_index bitor control_qubit_mask bitor target_qubit_mask;

          using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
          *(first + target_control_on_index) *= ::ket::utility::minus_imaginary_unit<complex_type>();
        });
    }

    // C...CsZ...Z+_{t...t'c...c'} = C...C(sZ+_t ... sZ+_t')_{c...c'}, CnsZ...Z+_{...}, C...CsZm+_{...}, or CnsZm+_{...}
    //   (sZ+_1...sZ+_N)_{nn} = (-i)^f(n-1) for 1<=n<=2^N, where f(n): the number of "1" bits in n
    template <typename ParallelPolicy, typename RandomAccessIterator, typename StateInteger, typename BitInteger, typename Qubit2, typename Qubit3, typename... Qubits>
    inline auto adj_sqrt_pauli_z(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      ::ket::qubit<StateInteger, BitInteger> const qubit1, Qubit2 const qubit2, Qubit3 const qubit3, Qubits const... qubits)
    -> void
    {
      static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
      static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");
      assert(
        ::ket::utility::integer_exp2<StateInteger>(::ket::utility::integer_log2<BitInteger>(last - first))
        == static_cast<StateInteger>(last - first));

# ifndef KET_USE_BIT_MASKS_EXPLICITLY
      using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
      constexpr auto num_operated_qubits = static_cast<BitInteger>(sizeof...(Qubits) + 3u);
      constexpr auto num_control_qubits = ::ket::gate::meta::num_control_qubits<BitInteger, Qubit2, Qubit3, Qubits...>::value;
      constexpr auto num_target_qubits = num_operated_qubits - num_control_qubits;
      constexpr auto num_target_indices = ::ket::utility::integer_exp2<std::size_t>(num_target_qubits);

      ::ket::gate::nocache::gate(
        parallel_policy, first, last,
        [](
          auto const first, StateInteger const index_wo_qubits,
          std::array<qubit_type, num_operated_qubits> const& unsorted_qubits,
          std::array<qubit_type, num_operated_qubits + BitInteger{1u}> const& sorted_qubits_with_sentinel,
          int const)
        {
          // 0b1...10...0u
          constexpr auto base_index = ((std::size_t{1u} << num_control_qubits) - std::size_t{1u}) << num_target_qubits;

          for (auto i = std::size_t{0u}; i < num_target_indices; ++i)
          {
            auto num_ones_in_i = BitInteger{0u};
            auto i_tmp = i;
            for (auto count = BitInteger{0u}; count < num_target_qubits; ++count)
            {
              if ((i_tmp bitand StateInteger{1u}) == StateInteger{1u})
                ++num_ones_in_i;

              i_tmp >>= BitInteger{1u};
            }

            using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
            using real_type = ::ket::utility::meta::real_t<complex_type>;
            auto const remainder = num_ones_in_i % BitInteger{4u};
            auto const coefficient
              = remainder == BitInteger{0u}
                ? complex_type{real_type{1}}
                : remainder == BitInteger{1u}
                  ? ::ket::utility::minus_imaginary_unit<complex_type>()
                  : remainder == BitInteger{2u}
                    ? complex_type{real_type{-1}}
                    : ::ket::utility::imaginary_unit<complex_type>();

            using std::begin;
            using std::end;
            auto const iter
              = first
                + ::ket::gate::utility::index_with_qubits(
                    index_wo_qubits, base_index + i,
                    begin(unsorted_qubits), end(unsorted_qubits),
                    begin(sorted_qubits_with_sentinel), end(sorted_qubits_with_sentinel));
            *iter *= coefficient;
          }
        },
        qubit1, qubit2, qubit3, qubits...);
# else // KET_USE_BIT_MASKS_EXPLICITLY
      constexpr auto num_operated_qubits = static_cast<BitInteger>(sizeof...(Qubits) + 3u);
      constexpr auto num_control_qubits = ::ket::gate::meta::num_control_qubits<BitInteger, Qubit2, Qubit3, Qubits...>::value;
      constexpr auto num_target_qubits = num_operated_qubits - num_control_qubits;
      constexpr auto num_target_indices = ::ket::utility::integer_exp2<std::size_t>(num_target_qubits);

      ::ket::gate::nocache::gate(
        parallel_policy, first, last,
        [](
          auto const first, StateInteger const index_wo_qubits,
          std::array<StateInteger, num_operated_qubits> const& qubit_masks,
          std::array<StateInteger, num_operated_qubits + 1u> const& index_masks,
          int const)
        {
          // 0b1...10...0u
          constexpr auto base_index = ((std::size_t{1u} << num_control_qubits) - std::size_t{1u}) << num_target_qubits;

          for (auto i = std::size_t{0u}; i < num_target_indices; ++i)
          {
            auto num_ones_in_i = BitInteger{0u};
            auto i_tmp = i;
            for (auto count = BitInteger{0u}; count < num_target_qubits; ++count)
            {
              if ((i_tmp bitand StateInteger{1u}) == StateInteger{1u})
                ++num_ones_in_i;

              i_tmp >>= BitInteger{1u};
            }

            using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
            using real_type = ::ket::utility::meta::real_t<complex_type>;
            auto const remainder = num_ones_in_i % BitInteger{4u};
            auto const coefficient
              = remainder == BitInteger{0u}
                ? complex_type{real_type{1}}
                : remainder == BitInteger{1u}
                  ? ::ket::utility::minus_imaginary_unit<complex_type>()
                  : remainder == BitInteger{2u}
                    ? complex_type{real_type{-1}}
                    : ::ket::utility::imaginary_unit<complex_type>();

            using std::begin;
            using std::end;
            auto const iter
              = first
                + ::ket::gate::utility::index_with_qubits(
                    index_wo_qubits, base_index + i,
                    begin(qubit_masks), end(qubit_masks), begin(index_masks), end(index_masks));
            *iter *= coefficient;
          }
        },
        qubit1, qubit2, qubit3, qubits...);
# endif // KET_USE_BIT_MASKS_EXPLICITLY
    }

    template <typename RandomAccessIterator, typename StateInteger, typename BitInteger, typename... Qubits>
    inline auto adj_sqrt_pauli_z(
      RandomAccessIterator const first, RandomAccessIterator const last,
      ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
    -> void
    { ::ket::gate::adj_sqrt_pauli_z(::ket::utility::policy::make_sequential(), first, last, qubit, qubits...); }

    namespace ranges
    {
      template <typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename... Qubits>
      inline auto adj_sqrt_pauli_z(
        ParallelPolicy const parallel_policy, RandomAccessRange& state,
        ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
      -> RandomAccessRange&
      {
        using std::begin;
        using std::end;
        ::ket::gate::adj_sqrt_pauli_z(parallel_policy, begin(state), end(state), qubit, qubits...);
        return state;
      }

      template <typename RandomAccessRange, typename StateInteger, typename BitInteger, typename... Qubits>
      inline auto adj_sqrt_pauli_z(RandomAccessRange& state, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits) -> RandomAccessRange&
      { return ::ket::gate::ranges::adj_sqrt_pauli_z(::ket::utility::policy::make_sequential(), state, qubit, qubits...); }
    } // namespace ranges
  } // namespace gate
} // namespace ket


#endif // KET_GATE_SQRT_PAULI_Z_HPP
