#ifndef KET_GATE_FUSED_PAULI_Z_HPP
# define KET_GATE_FUSED_PAULI_Z_HPP

# include <cassert>
# include <cstddef>
# include <array>
# include <algorithm>
# include <iterator>
# include <utility>
# include <type_traits>

# include <ket/qubit.hpp>
# include <ket/control.hpp>
# include <ket/gate/fused/gate.hpp>
# include <ket/gate/utility/index_with_qubits.hpp>
# include <ket/gate/meta/num_control_qubits.hpp>
# include <ket/utility/integer_exp2.hpp>
# include <ket/utility/integer_log2.hpp>
# include <ket/utility/meta/real_of.hpp>


namespace ket
{
  namespace gate
  {
    namespace fused
    {
# ifndef KET_USE_BIT_MASKS_EXPLICITLY
      // Case 1: the first argument of qubits is ket::control<ket::qubit<S, B>>
      template <typename RandomAccessIterator, typename StateInteger, typename BitInteger, std::size_t num_fused_qubits>
      inline auto pauli_z(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_fused_qubits > const& unsorted_fused_qubits,
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_fused_qubits + 1u> const& sorted_fused_qubits_with_sentinel)
      -> void
      {
        static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
        static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");

        constexpr auto count = ::ket::utility::integer_exp2<StateInteger>(num_fused_qubits);
        for (auto index = std::size_t{0u}; index < count; ++index)
        {
          using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
          using real_type = ::ket::utility::meta::real_t<complex_type>;
          using std::begin;
          using std::end;
          auto const iter
            = first
              + ::ket::gate::utility::index_with_qubits(
                  fused_index_wo_qubits, index,
                  begin(unsorted_fused_qubits), end(unsorted_fused_qubits),
                  begin(sorted_fused_qubits_with_sentinel), end(sorted_fused_qubits_with_sentinel));
          *iter *= static_cast<real_type>(-1);
        }
      }

      // Z_i or Z1_i
      // Z_1 (a_0 |0> + a_1 |1>) = a_0 |0> - a_1 |1>
      template <typename RandomAccessIterator, typename StateInteger, typename BitInteger, std::size_t num_fused_qubits>
      inline auto pauli_z(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_fused_qubits > const& unsorted_fused_qubits,
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_fused_qubits + 1u> const& sorted_fused_qubits_with_sentinel,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit)
      -> void
      {
        static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
        static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");
        assert(control_qubit < ::ket::make_qubit<StateInteger>(static_cast<BitInteger>(num_fused_qubits)));

        constexpr auto num_operated_qubits = BitInteger{1u};

        auto const qubit_mask = ::ket::utility::integer_exp2<StateInteger>(control_qubit);
        auto const lower_bits_mask = qubit_mask - StateInteger{1u};
        auto const upper_bits_mask = compl lower_bits_mask;

        constexpr auto count = ::ket::utility::integer_exp2<StateInteger>(num_fused_qubits) >> num_operated_qubits;
        for (auto index_wo_qubit = std::size_t{0u}; index_wo_qubit < count; ++index_wo_qubit)
        {
          // xxxxx0xxxxxx
          auto const zero_index = ((index_wo_qubit bitand upper_bits_mask) << 1u) bitor (index_wo_qubit bitand lower_bits_mask);
          // xxxxx1xxxxxx
          auto const one_index = zero_index bitor qubit_mask;

          using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
          using real_type = ::ket::utility::meta::real_t<complex_type>;
          using std::begin;
          using std::end;
          auto const one_iter
            = first
              + ::ket::gate::utility::index_with_qubits(
                  fused_index_wo_qubits, one_index,
                  begin(unsorted_fused_qubits), end(unsorted_fused_qubits),
                  begin(sorted_fused_qubits_with_sentinel), end(sorted_fused_qubits_with_sentinel));
          *one_iter *= static_cast<real_type>(-1);
        }
      }

      // CZ_{cc'}, CZ1_{cc'}, C1Z_{cc'}, or C1Z1_{cc'}
      // CZ_{1,2} (a_{00} |00> + a_{01} |01> + a_{10} |10> + a{11} |11>)
      //   = a_{00} |00> + a_{01} |01> + a_{10} |10> - a{11} |11>
      template <typename RandomAccessIterator, typename StateInteger, typename BitInteger, std::size_t num_fused_qubits>
      inline auto pauli_z(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_fused_qubits > const& unsorted_fused_qubits,
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_fused_qubits + 1u> const& sorted_fused_qubits_with_sentinel,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2)
      -> void
      {
        static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
        static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");
        assert(control_qubit1 < ::ket::make_qubit<StateInteger>(static_cast<BitInteger>(num_fused_qubits)));
        assert(control_qubit2 < ::ket::make_qubit<StateInteger>(static_cast<BitInteger>(num_fused_qubits)));
        assert(control_qubit1 != control_qubit2);

        constexpr auto num_operated_qubits = BitInteger{2u};

        auto const minmax_qubits = std::minmax(control_qubit1, control_qubit2);
        auto const control_qubit1_mask = ::ket::utility::integer_exp2<StateInteger>(control_qubit1);
        auto const control_qubit2_mask = ::ket::utility::integer_exp2<StateInteger>(control_qubit2);
        auto const lower_bits_mask = ::ket::utility::integer_exp2<StateInteger>(minmax_qubits.first) - StateInteger{1u};
        auto const middle_bits_mask
          = (::ket::utility::integer_exp2<StateInteger>(minmax_qubits.second - BitInteger{1u}) - StateInteger{1u})
            xor lower_bits_mask;
        auto const upper_bits_mask = compl (lower_bits_mask bitor middle_bits_mask);

        constexpr auto count = ::ket::utility::integer_exp2<StateInteger>(num_fused_qubits) >> num_operated_qubits;
        for (auto index_wo_qubits = std::size_t{0u}; index_wo_qubits < count; ++index_wo_qubits)
        {
          // xxx0_c1xxx0_c2xxx
          auto const index00
            = ((index_wo_qubits bitand upper_bits_mask) << 2u)
              bitor ((index_wo_qubits bitand middle_bits_mask) << 1u)
              bitor (index_wo_qubits bitand lower_bits_mask);
          // xxx1_c1xxx1_c2xxx
          auto const index11 = index00 bitor control_qubit1_mask bitor control_qubit2_mask;
          using std::begin;
          using std::end;
          auto const iter11
            = first
              + ::ket::gate::utility::index_with_qubits(
                  fused_index_wo_qubits, index11,
                  begin(unsorted_fused_qubits), end(unsorted_fused_qubits),
                  begin(sorted_fused_qubits_with_sentinel), end(sorted_fused_qubits_with_sentinel));

          using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
          using real_type = ::ket::utility::meta::real_t<complex_type>;
          *iter11 *= real_type{-1.0};
        }
      }

      // C...CZ_{c0,c...c'} or CnZ_{c0,c...c'}
      template <typename RandomAccessIterator, typename StateInteger, typename BitInteger, std::size_t num_fused_qubits, typename... Qubits>
      inline auto pauli_z(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_fused_qubits > const& unsorted_fused_qubits,
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_fused_qubits + 1u> const& sorted_fused_qubits_with_sentinel,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit3,
        ::ket::control<Qubits> const... control_qubits)
      -> void
      {
        using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
        constexpr auto num_control_qubits = BitInteger{sizeof...(Qubits) + 3u};
        constexpr auto num_operated_qubits = num_control_qubits;

        ::ket::gate::fused::gate<num_fused_qubits>(
          first,
          [fused_index_wo_qubits, &unsorted_fused_qubits, &sorted_fused_qubits_with_sentinel](
            auto const first, StateInteger const operated_index_wo_qubits,
            std::array<qubit_type, num_operated_qubits> const& unsorted_operated_qubits,
            std::array<qubit_type, num_operated_qubits + 1u> const& sorted_operated_qubits_with_sentinel)
          {
            // 0b11...11u
            constexpr auto index = ((std::size_t{1u} << num_operated_qubits) - std::size_t{1u});

            using std::begin;
            using std::end;
            auto const iter
              = first
                + ::ket::gate::utility::index_with_qubits(
                    fused_index_wo_qubits,
                    ::ket::gate::utility::index_with_qubits(
                      operated_index_wo_qubits, index,
                      begin(unsorted_operated_qubits), end(unsorted_operated_qubits),
                      begin(sorted_operated_qubits_with_sentinel), end(sorted_operated_qubits_with_sentinel)),
                    begin(unsorted_fused_qubits), end(unsorted_fused_qubits),
                    begin(sorted_fused_qubits_with_sentinel), end(sorted_fused_qubits_with_sentinel));
            using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
            using real_type = ::ket::utility::meta::real_t<complex_type>;
            *iter *= real_type{-1};
          },
          control_qubit1, control_qubit2, control_qubit3, control_qubits...);
      }

      template <typename RandomAccessIterator, typename StateInteger, typename BitInteger, std::size_t num_fused_qubits, typename... Qubits>
      inline auto adj_pauli_z(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_fused_qubits > const& unsorted_fused_qubits,
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_fused_qubits + 1u> const& sorted_fused_qubits_with_sentinel,
        ::ket::control<Qubits> const... control_qubits)
      -> void
      { ::ket::gate::fused::pauli_z(first, fused_index_wo_qubits, unsorted_fused_qubits, sorted_fused_qubits_with_sentinel, control_qubits...); }

      // Case 2: the first argument of qubits is ket::qubit<S, B>
      // Z_i or Z1_i
      // Z_1 (a_0 |0> + a_1 |1>) = a_0 |0> - a_1 |1>
      template <typename RandomAccessIterator, typename StateInteger, typename BitInteger, std::size_t num_fused_qubits>
      inline auto pauli_z(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_fused_qubits > const& unsorted_fused_qubits,
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_fused_qubits + 1u> const& sorted_fused_qubits_with_sentinel,
        ::ket::qubit<StateInteger, BitInteger> const qubit)
      -> void
      {
        static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
        static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");
        assert(qubit < ::ket::make_qubit<StateInteger>(static_cast<BitInteger>(num_fused_qubits)));

        constexpr auto num_operated_qubits = BitInteger{1u};

        auto const qubit_mask = ::ket::utility::integer_exp2<StateInteger>(qubit);
        auto const lower_bits_mask = qubit_mask - StateInteger{1u};
        auto const upper_bits_mask = compl lower_bits_mask;

        constexpr auto count = ::ket::utility::integer_exp2<StateInteger>(num_fused_qubits) >> num_operated_qubits;
        for (auto index_wo_qubit = std::size_t{0u}; index_wo_qubit < count; ++index_wo_qubit)
        {
          // xxxxx0xxxxxx
          auto const zero_index = ((index_wo_qubit bitand upper_bits_mask) << 1u) bitor (index_wo_qubit bitand lower_bits_mask);
          // xxxxx1xxxxxx
          auto const one_index = zero_index bitor qubit_mask;

          using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
          using real_type = ::ket::utility::meta::real_t<complex_type>;
          using std::begin;
          using std::end;
          auto const one_iter
            = first
              + ::ket::gate::utility::index_with_qubits(
                  fused_index_wo_qubits, one_index,
                  begin(unsorted_fused_qubits), end(unsorted_fused_qubits),
                  begin(sorted_fused_qubits_with_sentinel), end(sorted_fused_qubits_with_sentinel));
          *one_iter *= static_cast<real_type>(-1);
        }
      }

      // ZZ_i = Z_i Z_j or Z2_{ij}
      // ZZ_{1,2} (a_{00} |00> + a_{01} |01> + a_{10} |10> + a{11} |11>)
      //   = a_{00} |00> - a_{01} |01> - a_{10} |10> + a{11} |11>
      template <typename RandomAccessIterator, typename StateInteger, typename BitInteger, std::size_t num_fused_qubits>
      inline auto pauli_z(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_fused_qubits > const& unsorted_fused_qubits,
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_fused_qubits + 1u> const& sorted_fused_qubits_with_sentinel,
        ::ket::qubit<StateInteger, BitInteger> const qubit1, ::ket::qubit<StateInteger, BitInteger> const qubit2)
      -> void
      {
        static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
        static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");
        assert(qubit1 < ::ket::make_qubit<StateInteger>(static_cast<BitInteger>(num_fused_qubits)));
        assert(qubit2 < ::ket::make_qubit<StateInteger>(static_cast<BitInteger>(num_fused_qubits)));
        assert(qubit1 != qubit2);

        constexpr auto num_operated_qubits = BitInteger{2u};

        auto const minmax_qubits = std::minmax(qubit1, qubit2);
        auto const qubit1_mask = ::ket::utility::integer_exp2<StateInteger>(qubit1);
        auto const qubit2_mask = ::ket::utility::integer_exp2<StateInteger>(qubit2);
        auto const lower_bits_mask = ::ket::utility::integer_exp2<StateInteger>(minmax_qubits.first) - StateInteger{1u};
        auto const middle_bits_mask
          = (::ket::utility::integer_exp2<StateInteger>(minmax_qubits.second - BitInteger{1u}) - StateInteger{1u})
            xor lower_bits_mask;
        auto const upper_bits_mask = compl (lower_bits_mask bitor middle_bits_mask);

        constexpr auto count = ::ket::utility::integer_exp2<StateInteger>(num_fused_qubits) >> num_operated_qubits;
        for (auto index_wo_qubits = std::size_t{0u}; index_wo_qubits < count; ++index_wo_qubits)
        {
          // xxx0_1xxx0_2xxx
          auto const base_index
            = ((index_wo_qubits bitand upper_bits_mask) << 2u)
              bitor ((index_wo_qubits bitand middle_bits_mask) << 1u)
              bitor (index_wo_qubits bitand lower_bits_mask);
          // xxx1_1xxx0_2xxx
          using std::begin;
          using std::end;
          auto const qubit1_on_iter
            = first
              + ::ket::gate::utility::index_with_qubits(
                  fused_index_wo_qubits, base_index bitor qubit1_mask,
                  begin(unsorted_fused_qubits), end(unsorted_fused_qubits),
                  begin(sorted_fused_qubits_with_sentinel), end(sorted_fused_qubits_with_sentinel));
          // xxx0_1xxx1_2xxx
          auto const qubit2_on_iter
            = first
              + ::ket::gate::utility::index_with_qubits(
                  fused_index_wo_qubits, base_index bitor qubit2_mask,
                  begin(unsorted_fused_qubits), end(unsorted_fused_qubits),
                  begin(sorted_fused_qubits_with_sentinel), end(sorted_fused_qubits_with_sentinel));

          using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
          using real_type = ::ket::utility::meta::real_t<complex_type>;
          *qubit1_on_iter *= real_type{-1.0};
          *qubit2_on_iter *= real_type{-1.0};
        }
      }

      // CZ_{tc}, CZ1_{tc}, C1Z_{tc}, or C1Z1_{tc}
      // CZ_{1,2} (a_{00} |00> + a_{01} |01> + a_{10} |10> + a{11} |11>)
      //   = a_{00} |00> + a_{01} |01> + a_{10} |10> - a{11} |11>
      template <typename RandomAccessIterator, typename StateInteger, typename BitInteger, std::size_t num_fused_qubits>
      inline auto pauli_z(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_fused_qubits > const& unsorted_fused_qubits,
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_fused_qubits + 1u> const& sorted_fused_qubits_with_sentinel,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit)
      -> void
      {
        static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
        static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");
        assert(target_qubit < ::ket::make_qubit<StateInteger>(static_cast<BitInteger>(num_fused_qubits)));
        assert(control_qubit < ::ket::make_qubit<StateInteger>(static_cast<BitInteger>(num_fused_qubits)));
        assert(target_qubit != control_qubit);

        constexpr auto num_operated_qubits = BitInteger{2u};

        auto const minmax_qubits = std::minmax(target_qubit, control_qubit.qubit());
        auto const target_qubit_mask = ::ket::utility::integer_exp2<StateInteger>(target_qubit);
        auto const control_qubit_mask = ::ket::utility::integer_exp2<StateInteger>(control_qubit);
        auto const lower_bits_mask = ::ket::utility::integer_exp2<StateInteger>(minmax_qubits.first) - StateInteger{1u};
        auto const middle_bits_mask
          = (::ket::utility::integer_exp2<StateInteger>(minmax_qubits.second - BitInteger{1u}) - StateInteger{1u})
            xor lower_bits_mask;
        auto const upper_bits_mask = compl (lower_bits_mask bitor middle_bits_mask);

        constexpr auto count = ::ket::utility::integer_exp2<StateInteger>(num_fused_qubits) >> num_operated_qubits;
        for (auto index_wo_qubits = std::size_t{0u}; index_wo_qubits < count; ++index_wo_qubits)
        {
          // xxx0_txxx0_cxxx
          auto const base_index
            = ((index_wo_qubits bitand upper_bits_mask) << 2u)
              bitor ((index_wo_qubits bitand middle_bits_mask) << 1u)
              bitor (index_wo_qubits bitand lower_bits_mask);
          // xxx1_txxx1_cxxx
          auto const target_control_on_index = base_index bitor control_qubit_mask bitor target_qubit_mask;
          using std::begin;
          using std::end;
          auto const target_control_on_iter
            = first
              + ::ket::gate::utility::index_with_qubits(
                  fused_index_wo_qubits, target_control_on_index,
                  begin(unsorted_fused_qubits), end(unsorted_fused_qubits),
                  begin(sorted_fused_qubits_with_sentinel), end(sorted_fused_qubits_with_sentinel));

          using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
          using real_type = ::ket::utility::meta::real_t<complex_type>;
          *target_control_on_iter *= real_type{-1.0};
        }
      }

      // C...CZ...Z_{t...t'c...c'} = C...C(Z_t ... Z_t')_{c...c'}, CnZ...Z_{...}, C...CZm_{...}, or CnZm_{...}
      template <typename RandomAccessIterator, typename StateInteger, typename BitInteger, std::size_t num_fused_qubits, typename Qubit2, typename Qubit3, typename... Qubits>
      inline auto pauli_z(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_fused_qubits > const& unsorted_fused_qubits,
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_fused_qubits + 1u> const& sorted_fused_qubits_with_sentinel,
        ::ket::qubit<StateInteger, BitInteger> const qubit1,
        Qubit2 const qubit2, Qubit3 const qubit3, Qubits const... qubits)
      -> void
      {
        using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
        constexpr auto num_operated_qubits = static_cast<BitInteger>(sizeof...(Qubits) + 3u);
        constexpr auto num_control_qubits = ::ket::gate::meta::num_control_qubits<BitInteger, Qubit2, Qubit3, Qubits...>::value;
        constexpr auto num_target_qubits = num_operated_qubits - num_control_qubits;
        constexpr auto num_target_indices = ::ket::utility::integer_exp2<std::size_t>(num_target_qubits);

        ::ket::gate::fused::gate<num_fused_qubits>(
          first,
          [fused_index_wo_qubits, &unsorted_fused_qubits, &sorted_fused_qubits_with_sentinel](
            auto const first, StateInteger const operated_index_wo_qubits,
            std::array<qubit_type, num_operated_qubits> const& unsorted_operated_qubits,
            std::array<qubit_type, num_operated_qubits + 1u> const& sorted_operated_qubits_with_sentinel)
          {
            // 0b1...10...0u
            constexpr auto base_index = ((std::size_t{1u} << num_control_qubits) - std::size_t{1u}) << num_target_qubits;

            for (auto i = std::size_t{0u}; i < num_target_indices; ++i)
            {
              auto num_ones_in_i = BitInteger{0u};
              auto i_tmp = i;
              for (auto count = BitInteger{0u}; count < num_target_qubits; ++count)
              {
                if ((i_tmp bitand std::size_t{1u}) == std::size_t{1u})
                  ++num_ones_in_i;

                i_tmp >>= BitInteger{1u};
              }

              using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
              using real_type = ::ket::utility::meta::real_t<complex_type>;
              if (num_ones_in_i % BitInteger{2u} == BitInteger{1u})
              {
                using std::begin;
                using std::end;
                auto const iter
                  = first
                    + ::ket::gate::utility::index_with_qubits(
                        fused_index_wo_qubits,
                        ::ket::gate::utility::index_with_qubits(
                          operated_index_wo_qubits, base_index + i,
                          begin(unsorted_operated_qubits), end(unsorted_operated_qubits),
                          begin(sorted_operated_qubits_with_sentinel), end(sorted_operated_qubits_with_sentinel)),
                        begin(unsorted_fused_qubits), end(unsorted_fused_qubits),
                        begin(sorted_fused_qubits_with_sentinel), end(sorted_fused_qubits_with_sentinel));
                *iter *= real_type{-1.0};
              }
            }
          },
          qubit1, qubit2, qubit3, qubits...);
      }

      template <typename RandomAccessIterator, typename StateInteger, typename BitInteger, std::size_t num_fused_qubits, typename... Qubits>
      inline auto adj_pauli_z(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_fused_qubits > const& unsorted_fused_qubits,
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_fused_qubits + 1u> const& sorted_fused_qubits_with_sentinel,
        ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
      -> void
      { ::ket::gate::fused::pauli_z(first, fused_index_wo_qubits, unsorted_fused_qubits, sorted_fused_qubits_with_sentinel, qubit, qubits...); }
# else // KET_USE_BIT_MASKS_EXPLICITLY
      // Case 1: the first argument of qubits is ket::control<ket::qubit<S, B>>
      template <typename RandomAccessIterator, typename StateInteger, std::size_t num_fused_qubits>
      inline auto pauli_z(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array<StateInteger, num_fused_qubits> const& fused_qubit_masks, std::array<StateInteger, num_fused_qubits + 1u> const& fused_index_masks)
      -> void
      {
        static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");

        constexpr auto count = ::ket::utility::integer_exp2<StateInteger>(num_fused_qubits);
        for (auto index = std::size_t{0u}; index < count; ++index)
        {
          using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
          using real_type = ::ket::utility::meta::real_t<complex_type>;
          using std::begin;
          using std::end;
          auto const iter
            = first
              + ::ket::gate::utility::index_with_qubits(
                  fused_index_wo_qubits, index,
                  begin(fused_qubit_masks), end(fused_qubit_masks), begin(fused_index_masks), end(fused_index_masks));
          *iter *= static_cast<real_type>(-1);
        }
      }

      // Z_i or Z1_i
      // Z_1 (a_0 |0> + a_1 |1>) = a_0 |0> - a_1 |1>
      template <typename RandomAccessIterator, typename StateInteger, std::size_t num_fused_qubits, typename BitInteger>
      inline auto pauli_z(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array<StateInteger, num_fused_qubits> const& fused_qubit_masks, std::array<StateInteger, num_fused_qubits + 1u> const& fused_index_masks,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit)
      -> void
      {
        static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
        static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");
        assert(control_qubit < ::ket::make_qubit<StateInteger>(static_cast<BitInteger>(num_fused_qubits)));

        constexpr auto num_operated_qubits = BitInteger{1u};

        auto const qubit_mask = ::ket::utility::integer_exp2<StateInteger>(control_qubit);
        auto const lower_bits_mask = qubit_mask - StateInteger{1u};
        auto const upper_bits_mask = compl lower_bits_mask;

        constexpr auto count = ::ket::utility::integer_exp2<StateInteger>(num_fused_qubits) >> num_operated_qubits;
        for (auto index_wo_qubit = std::size_t{0u}; index_wo_qubit < count; ++index_wo_qubit)
        {
          // xxxxx0xxxxxx
          auto const zero_index = ((index_wo_qubit bitand upper_bits_mask) << 1u) bitor (index_wo_qubit bitand lower_bits_mask);
          // xxxxx1xxxxxx
          auto const one_index = zero_index bitor qubit_mask;

          using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
          using real_type = ::ket::utility::meta::real_t<complex_type>;
          using std::begin;
          using std::end;
          auto const one_iter
            = first
              + ::ket::gate::utility::index_with_qubits(
                  fused_index_wo_qubits, one_index,
                  begin(fused_qubit_masks), end(fused_qubit_masks), begin(fused_index_masks), end(fused_index_masks));
          *one_iter *= static_cast<real_type>(-1);
        }
      }

      // CZ_{tc}, CZ1_{tc}, C1Z_{tc}, or C1Z1_{tc}
      // CZ_{1,2} (a_{00} |00> + a_{01} |01> + a_{10} |10> + a{11} |11>)
      //   = a_{00} |00> + a_{01} |01> + a_{10} |10> - a{11} |11>
      template <typename RandomAccessIterator, typename StateInteger, std::size_t num_fused_qubits, typename BitInteger>
      inline auto pauli_z(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array<StateInteger, num_fused_qubits> const& fused_qubit_masks, std::array<StateInteger, num_fused_qubits + 1u> const& fused_index_masks,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2)
      -> void
      {
        static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
        static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");
        assert(control_qubit1 < ::ket::make_qubit<StateInteger>(static_cast<BitInteger>(num_fused_qubits)));
        assert(control_qubit2 < ::ket::make_qubit<StateInteger>(static_cast<BitInteger>(num_fused_qubits)));
        assert(control_qubit1 != control_qubit2);

        constexpr auto num_operated_qubits = BitInteger{2u};

        auto const minmax_qubits = std::minmax(control_qubit1, control_qubit2);
        auto const control_qubit1_mask = ::ket::utility::integer_exp2<StateInteger>(control_qubit1);
        auto const control_qubit2_mask = ::ket::utility::integer_exp2<StateInteger>(control_qubit2);
        auto const lower_bits_mask = ::ket::utility::integer_exp2<StateInteger>(minmax_qubits.first) - StateInteger{1u};
        auto const middle_bits_mask
          = (::ket::utility::integer_exp2<StateInteger>(minmax_qubits.second - BitInteger{1u}) - StateInteger{1u})
            xor lower_bits_mask;
        auto const upper_bits_mask = compl (lower_bits_mask bitor middle_bits_mask);

        constexpr auto count = ::ket::utility::integer_exp2<StateInteger>(num_fused_qubits) >> num_operated_qubits;
        for (auto index_wo_qubits = std::size_t{0u}; index_wo_qubits < count; ++index_wo_qubits)
        {
          // xxx0_c1xxx0_c2xxx
          auto const index00
            = ((index_wo_qubits bitand upper_bits_mask) << 2u)
              bitor ((index_wo_qubits bitand middle_bits_mask) << 1u)
              bitor (index_wo_qubits bitand lower_bits_mask);
          // xxx1_c1xxx1_c2xxx
          auto const index11 = index00 bitor control_qubit1_mask bitor control_qubit2_mask;
          using std::begin;
          using std::end;
          auto const iter11
            = first
              + ::ket::gate::utility::index_with_qubits(
                  fused_index_wo_qubits, index11,
                  begin(fused_qubit_masks), end(fused_qubit_masks), begin(fused_index_masks), end(fused_index_masks));

          using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
          using real_type = ::ket::utility::meta::real_t<complex_type>;
          *iter11 *= real_type{-1.0};
        }
      }

      // C...CZ_{c0,c...c'} or CnZ_{c0,c...c'}
      template <typename RandomAccessIterator, typename StateInteger, std::size_t num_fused_qubits, typename BitInteger, typename... Qubits>
      inline auto pauli_z(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array<StateInteger, num_fused_qubits> const& fused_qubit_masks, std::array<StateInteger, num_fused_qubits + 1u> const& fused_index_masks,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit3,
        ::ket::control<Qubits> const... control_qubits)
      -> void
      {
        constexpr auto num_control_qubits = BitInteger{sizeof...(Qubits) + 3u};
        constexpr auto num_operated_qubits = num_control_qubits;

        ::ket::gate::fused::gate<num_fused_qubits>(
          first,
          [fused_index_wo_qubits, &fused_qubit_masks, &fused_index_masks](
            auto const first, StateInteger const operated_index_wo_qubits,
            std::array<StateInteger, num_operated_qubits> const& operated_qubit_masks,
            std::array<StateInteger, num_operated_qubits + 1u> const& operated_index_masks)
          {
            // 0b11...11u
            constexpr auto index = ((std::size_t{1u} << num_operated_qubits) - std::size_t{1u});

            using std::begin;
            using std::end;
            auto const iter
              = first
                + ::ket::gate::utility::index_with_qubits(
                    fused_index_wo_qubits,
                    ::ket::gate::utility::index_with_qubits(
                      operated_index_wo_qubits, index,
                      begin(operated_qubit_masks), end(operated_qubit_masks), begin(operated_index_masks), end(operated_index_masks)),
                    begin(fused_qubit_masks), end(fused_qubit_masks), begin(fused_index_masks), end(fused_index_masks));
            using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
            using real_type = ::ket::utility::meta::real_t<complex_type>;
            *iter *= real_type{-1};
          },
          control_qubit1, control_qubit2, control_qubit3, control_qubits...);
      }

      template <typename RandomAccessIterator, typename StateInteger, std::size_t num_fused_qubits, typename... Qubits>
      inline auto adj_pauli_z(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array<StateInteger, num_fused_qubits> const& fused_qubit_masks, std::array<StateInteger, num_fused_qubits + 1u> const& fused_index_masks,
        ::ket::control<Qubits> const... control_qubits)
      -> void
      { ::ket::gate::fused::pauli_z(first, fused_index_wo_qubits, fused_qubit_masks, fused_index_masks, control_qubits...); }

      // Case 2: the first argument of qubits is ket::qubit<S, B>
      // Z_i or Z1_i
      // Z_1 (a_0 |0> + a_1 |1>) = a_0 |0> - a_1 |1>
      template <typename RandomAccessIterator, typename StateInteger, std::size_t num_fused_qubits, typename BitInteger>
      inline auto pauli_z(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array<StateInteger, num_fused_qubits> const& fused_qubit_masks, std::array<StateInteger, num_fused_qubits + 1u> const& fused_index_masks,
        ::ket::qubit<StateInteger, BitInteger> const qubit)
      -> void
      {
        static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
        static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");
        assert(qubit < ::ket::make_qubit<StateInteger>(static_cast<BitInteger>(num_fused_qubits)));

        constexpr auto num_operated_qubits = BitInteger{1u};

        auto const qubit_mask = ::ket::utility::integer_exp2<StateInteger>(qubit);
        auto const lower_bits_mask = qubit_mask - StateInteger{1u};
        auto const upper_bits_mask = compl lower_bits_mask;

        constexpr auto count = ::ket::utility::integer_exp2<StateInteger>(num_fused_qubits) >> num_operated_qubits;
        for (auto index_wo_qubit = std::size_t{0u}; index_wo_qubit < count; ++index_wo_qubit)
        {
          // xxxxx0xxxxxx
          auto const zero_index = ((index_wo_qubit bitand upper_bits_mask) << 1u) bitor (index_wo_qubit bitand lower_bits_mask);
          // xxxxx1xxxxxx
          auto const one_index = zero_index bitor qubit_mask;

          using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
          using real_type = ::ket::utility::meta::real_t<complex_type>;
          using std::begin;
          using std::end;
          auto const one_iter
            = first
              + ::ket::gate::utility::index_with_qubits(
                  fused_index_wo_qubits, one_index,
                  begin(fused_qubit_masks), end(fused_qubit_masks), begin(fused_index_masks), end(fused_index_masks));
          *one_iter *= static_cast<real_type>(-1);
        }
      }

      // ZZ_i = Z_i Z_j or Z2_{ij}
      // ZZ_{1,2} (a_{00} |00> + a_{01} |01> + a_{10} |10> + a{11} |11>)
      //   = a_{00} |00> - a_{01} |01> - a_{10} |10> + a{11} |11>
      template <typename RandomAccessIterator, typename StateInteger, std::size_t num_fused_qubits, typename BitInteger>
      inline auto pauli_z(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array<StateInteger, num_fused_qubits> const& fused_qubit_masks, std::array<StateInteger, num_fused_qubits + 1u> const& fused_index_masks,
        ::ket::qubit<StateInteger, BitInteger> const qubit1, ::ket::qubit<StateInteger, BitInteger> const qubit2)
      -> void
      {
        static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
        static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");
        assert(qubit1 < ::ket::make_qubit<StateInteger>(static_cast<BitInteger>(num_fused_qubits)));
        assert(qubit2 < ::ket::make_qubit<StateInteger>(static_cast<BitInteger>(num_fused_qubits)));
        assert(qubit1 != qubit2);

        constexpr auto num_operated_qubits = BitInteger{2u};

        auto const minmax_qubits = std::minmax(qubit1, qubit2);
        auto const qubit1_mask = ::ket::utility::integer_exp2<StateInteger>(qubit1);
        auto const qubit2_mask = ::ket::utility::integer_exp2<StateInteger>(qubit2);
        auto const lower_bits_mask = ::ket::utility::integer_exp2<StateInteger>(minmax_qubits.first) - StateInteger{1u};
        auto const middle_bits_mask
          = (::ket::utility::integer_exp2<StateInteger>(minmax_qubits.second - BitInteger{1u}) - StateInteger{1u})
            xor lower_bits_mask;
        auto const upper_bits_mask = compl (lower_bits_mask bitor middle_bits_mask);

        constexpr auto count = ::ket::utility::integer_exp2<StateInteger>(num_fused_qubits) >> num_operated_qubits;
        for (auto index_wo_qubits = std::size_t{0u}; index_wo_qubits < count; ++index_wo_qubits)
        {
          // xxx0_1xxx0_2xxx
          auto const base_index
            = ((index_wo_qubits bitand upper_bits_mask) << 2u)
              bitor ((index_wo_qubits bitand middle_bits_mask) << 1u)
              bitor (index_wo_qubits bitand lower_bits_mask);
          // xxx1_1xxx0_2xxx
          using std::begin;
          using std::end;
          auto const qubit1_on_iter
            = first
              + ::ket::gate::utility::index_with_qubits(
                  fused_index_wo_qubits, base_index bitor qubit1_mask,
                  begin(fused_qubit_masks), end(fused_qubit_masks), begin(fused_index_masks), end(fused_index_masks));
          // xxx0_1xxx1_2xxx
          auto const qubit2_on_iter
            = first
              + ::ket::gate::utility::index_with_qubits(
                  fused_index_wo_qubits, base_index bitor qubit2_mask,
                  begin(fused_qubit_masks), end(fused_qubit_masks), begin(fused_index_masks), end(fused_index_masks));

          using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
          using real_type = ::ket::utility::meta::real_t<complex_type>;
          *qubit1_on_iter *= real_type{-1.0};
          *qubit2_on_iter *= real_type{-1.0};
        }
      }

      // CZ_{tc}, CZ1_{tc}, C1Z_{tc}, or C1Z1_{tc}
      // CZ_{1,2} (a_{00} |00> + a_{01} |01> + a_{10} |10> + a{11} |11>)
      //   = a_{00} |00> + a_{01} |01> + a_{10} |10> - a{11} |11>
      template <typename RandomAccessIterator, typename StateInteger, std::size_t num_fused_qubits, typename BitInteger>
      inline auto pauli_z(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array<StateInteger, num_fused_qubits> const& fused_qubit_masks, std::array<StateInteger, num_fused_qubits + 1u> const& fused_index_masks,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit)
      -> void
      {
        static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
        static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");
        assert(target_qubit < ::ket::make_qubit<StateInteger>(static_cast<BitInteger>(num_fused_qubits)));
        assert(control_qubit < ::ket::make_qubit<StateInteger>(static_cast<BitInteger>(num_fused_qubits)));
        assert(target_qubit != control_qubit);

        constexpr auto num_operated_qubits = BitInteger{2u};

        auto const minmax_qubits = std::minmax(target_qubit, control_qubit.qubit());
        auto const target_qubit_mask = ::ket::utility::integer_exp2<StateInteger>(target_qubit);
        auto const control_qubit_mask = ::ket::utility::integer_exp2<StateInteger>(control_qubit);
        auto const lower_bits_mask = ::ket::utility::integer_exp2<StateInteger>(minmax_qubits.first) - StateInteger{1u};
        auto const middle_bits_mask
          = (::ket::utility::integer_exp2<StateInteger>(minmax_qubits.second - BitInteger{1u}) - StateInteger{1u})
            xor lower_bits_mask;
        auto const upper_bits_mask = compl (lower_bits_mask bitor middle_bits_mask);

        constexpr auto count = ::ket::utility::integer_exp2<StateInteger>(num_fused_qubits) >> num_operated_qubits;
        for (auto index_wo_qubits = std::size_t{0u}; index_wo_qubits < count; ++index_wo_qubits)
        {
          // xxx0_txxx0_cxxx
          auto const base_index
            = ((index_wo_qubits bitand upper_bits_mask) << 2u)
              bitor ((index_wo_qubits bitand middle_bits_mask) << 1u)
              bitor (index_wo_qubits bitand lower_bits_mask);
          // xxx1_txxx1_cxxx
          auto const target_control_on_index = base_index bitor control_qubit_mask bitor target_qubit_mask;
          using std::begin;
          using std::end;
          auto const target_control_on_iter
            = first
              + ::ket::gate::utility::index_with_qubits(
                  fused_index_wo_qubits, target_control_on_index,
                  begin(fused_qubit_masks), end(fused_qubit_masks), begin(fused_index_masks), end(fused_index_masks));

          using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
          using real_type = ::ket::utility::meta::real_t<complex_type>;
          *target_control_on_iter *= real_type{-1.0};
        }
      }

      // C...CZ...Z_{t...t'c...c'} = C...C(Z_t ... Z_t')_{c...c'}, CnZ...Z_{...}, C...CZm_{...}, or CnZm_{...}
      template <typename RandomAccessIterator, typename StateInteger, std::size_t num_fused_qubits, typename BitInteger, typename Qubit2, typename Qubit3, typename... Qubits>
      inline auto pauli_z(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array<StateInteger, num_fused_qubits> const& fused_qubit_masks, std::array<StateInteger, num_fused_qubits + 1u> const& fused_index_masks,
        ::ket::qubit<StateInteger, BitInteger> const qubit1,
        Qubit2 const qubit2, Qubit3 const qubit3, Qubits const... qubits)
      -> void
      {
        constexpr auto num_operated_qubits = static_cast<BitInteger>(sizeof...(Qubits) + 3u);
        constexpr auto num_control_qubits = ::ket::gate::meta::num_control_qubits<BitInteger, Qubit2, Qubit3, Qubits...>::value;
        constexpr auto num_target_qubits = num_operated_qubits - num_control_qubits;
        constexpr auto num_target_indices = ::ket::utility::integer_exp2<std::size_t>(num_target_qubits);

        ::ket::gate::fused::gate<num_fused_qubits>(
          first,
          [fused_index_wo_qubits, &fused_qubit_masks, &fused_index_masks](
            auto const first, StateInteger const operated_index_wo_qubits,
            std::array<StateInteger, num_operated_qubits> const& operated_qubit_masks,
            std::array<StateInteger, num_operated_qubits + 1u> const& operated_index_masks)
          {
            // 0b1...10...0u
            constexpr auto base_index = ((std::size_t{1u} << num_control_qubits) - std::size_t{1u}) << num_target_qubits;

            for (auto i = std::size_t{0u}; i < num_target_indices; ++i)
            {
              auto num_ones_in_i = BitInteger{0u};
              auto i_tmp = i;
              for (auto count = BitInteger{0u}; count < num_target_qubits; ++count)
              {
                if ((i_tmp bitand std::size_t{1u}) == std::size_t{1u})
                  ++num_ones_in_i;

                i_tmp >>= BitInteger{1u};
              }

              using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
              using real_type = ::ket::utility::meta::real_t<complex_type>;
              if (num_ones_in_i % BitInteger{2u} == BitInteger{1u})
              {
                using std::begin;
                using std::end;
                auto const iter
                  = first
                    + ::ket::gate::utility::index_with_qubits(
                        fused_index_wo_qubits,
                        ::ket::gate::utility::index_with_qubits(
                          operated_index_wo_qubits, base_index + i,
                          begin(operated_qubit_masks), end(operated_qubit_masks), begin(operated_index_masks), end(operated_index_masks)),
                        begin(fused_qubit_masks), end(fused_qubit_masks), begin(fused_index_masks), end(fused_index_masks));
                *iter *= real_type{-1.0};
              }
            }
          },
          qubit1, qubit2, qubit3, qubits...);
      }

      template <typename RandomAccessIterator, typename StateInteger, std::size_t num_fused_qubits, typename BitInteger, typename... Qubits>
      inline auto adj_pauli_z(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array<StateInteger, num_fused_qubits> const& fused_qubit_masks, std::array<StateInteger, num_fused_qubits + 1u> const& fused_index_masks,
        ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
      -> void
      { ::ket::gate::fused::pauli_z(first, fused_index_wo_qubits, fused_qubit_masks, fused_index_masks, qubit, qubits...); }
# endif // KET_USE_BIT_MASKS_EXPLICITLY
    } // namespace fused
  } // namespace gate
} // namespace ket


#endif // KET_GATE_FUSED_PAULI_Z_HPP
