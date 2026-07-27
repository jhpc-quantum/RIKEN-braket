#ifndef KET_GATE_FUSED_PAULI_Y_HPP
# define KET_GATE_FUSED_PAULI_Y_HPP

# include <cassert>
# include <cstddef>
# include <array>
# include <algorithm>
# include <iterator>
# include <utility>
# include <type_traits>

# include <boost/range/iterator_range.hpp>
# include <boost/range/join.hpp>
# include <boost/range/adaptor/transformed.hpp>

# include <ket/qubit.hpp>
# include <ket/control.hpp>
# include <ket/gate/fused/gate.hpp>
# include <ket/gate/utility/index_with_qubits.hpp>
# include <ket/gate/meta/num_control_qubits.hpp>
# include <ket/utility/integer_exp2.hpp>
# include <ket/utility/integer_log2.hpp>
# include <ket/utility/imaginary_unit.hpp>
# include <ket/utility/meta/real_of.hpp>


namespace ket
{
  namespace gate
  {
    namespace fused
    {
# ifndef KET_USE_BIT_MASKS_EXPLICITLY
      // Y_i or Y1_i
      // Y_1 (a_0 |0> + a_1 |1>) = -i a_1 |0> + i a_0 |1>
      template <typename RandomAccessIterator, typename StateInteger, typename BitInteger, std::size_t num_fused_qubits>
      inline auto pauli_y(
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
          using std::begin;
          using std::end;
          auto const zero_iter
            = first
              + ::ket::gate::utility::index_with_qubits(
                  fused_index_wo_qubits, zero_index,
                  begin(unsorted_fused_qubits), end(unsorted_fused_qubits),
                  begin(sorted_fused_qubits_with_sentinel), end(sorted_fused_qubits_with_sentinel));
          auto const one_iter
            = first
              + ::ket::gate::utility::index_with_qubits(
                  fused_index_wo_qubits, one_index,
                  begin(unsorted_fused_qubits), end(unsorted_fused_qubits),
                  begin(sorted_fused_qubits_with_sentinel), end(sorted_fused_qubits_with_sentinel));

          std::iter_swap(zero_iter, one_iter);

          using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
          *zero_iter *= ::ket::utility::minus_imaginary_unit<complex_type>();
          *one_iter *= ::ket::utility::imaginary_unit<complex_type>();
        }
      }

      // YY_{ij} = Y_i Y_j or Y2_{ij}
      // YY_{1,2} (a_{00} |00> + a_{01} |01> + a_{10} |10> + a{11} |11>)
      //   = -a_{11} |00> + a_{10} |01> + a_{01} |10> - a_{00} |11>
      template <typename RandomAccessIterator, typename StateInteger, typename BitInteger, std::size_t num_fused_qubits>
      inline auto pauli_y(
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
          using std::begin;
          using std::end;
          auto const off_iter
            = first
              + ::ket::gate::utility::index_with_qubits(
                  fused_index_wo_qubits, base_index,
                  begin(unsorted_fused_qubits), end(unsorted_fused_qubits),
                  begin(sorted_fused_qubits_with_sentinel), end(sorted_fused_qubits_with_sentinel));
          // xxx1_1xxx0_2xxx
          auto const qubit1_on_index = base_index bitor qubit1_mask;
          auto const qubit1_on_iter
            = first
              + ::ket::gate::utility::index_with_qubits(
                  fused_index_wo_qubits, qubit1_on_index,
                  begin(unsorted_fused_qubits), end(unsorted_fused_qubits),
                  begin(sorted_fused_qubits_with_sentinel), end(sorted_fused_qubits_with_sentinel));
          // xxx0_1xxx1_2xxx
          auto const qubit2_on_iter
            = first
              + ::ket::gate::utility::index_with_qubits(
                  fused_index_wo_qubits, base_index bitor qubit2_mask,
                  begin(unsorted_fused_qubits), end(unsorted_fused_qubits),
                  begin(sorted_fused_qubits_with_sentinel), end(sorted_fused_qubits_with_sentinel));
          // xxx1_1xxx1_2xxx
          auto const qubit12_on_iter
            = first
              + ::ket::gate::utility::index_with_qubits(
                  fused_index_wo_qubits, qubit1_on_index bitor qubit2_mask,
                  begin(unsorted_fused_qubits), end(unsorted_fused_qubits),
                  begin(sorted_fused_qubits_with_sentinel), end(sorted_fused_qubits_with_sentinel));

          std::iter_swap(off_iter, qubit12_on_iter);
          std::iter_swap(qubit1_on_iter, qubit2_on_iter);

          using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
          using real_type = ::ket::utility::meta::real_t<complex_type>;
          *off_iter *= real_type{-1.0};
          *qubit12_on_iter *= real_type{-1.0};
        }
      }

      // CY_{tc}, CY1_{tc}, C1Y_{tc}, or C1Y1_{tc}
      // CY_{1,2} (a_{00} |00> + a_{01} |01> + a_{10} |10> + a{11} |11>)
      //   = a_{00} |00> + a_{01} |01> - i a_{11} |10> + i a_{10} |11>
      template <typename RandomAccessIterator, typename StateInteger, typename BitInteger, std::size_t num_fused_qubits>
      inline auto pauli_y(
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
          // xxx0_txxx1_cxxx
          auto const control_on_index = base_index bitor control_qubit_mask;
          // xxx1_txxx1_cxxx
          auto const target_control_on_index = control_on_index bitor target_qubit_mask;
          using std::begin;
          using std::end;
          auto const control_on_iter
            = first
              + ::ket::gate::utility::index_with_qubits(
                  fused_index_wo_qubits, control_on_index,
                  begin(unsorted_fused_qubits), end(unsorted_fused_qubits),
                  begin(sorted_fused_qubits_with_sentinel), end(sorted_fused_qubits_with_sentinel));
          auto const target_control_on_iter
            = first
              + ::ket::gate::utility::index_with_qubits(
                  fused_index_wo_qubits, target_control_on_index,
                  begin(unsorted_fused_qubits), end(unsorted_fused_qubits),
                  begin(sorted_fused_qubits_with_sentinel), end(sorted_fused_qubits_with_sentinel));

          std::iter_swap(control_on_iter, target_control_on_iter);

          using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
          *control_on_iter *= ::ket::utility::minus_imaginary_unit<complex_type>();
          *target_control_on_iter *= ::ket::utility::imaginary_unit<complex_type>();
        }
      }

      // C...CY...Y_{t...t'c...c'} = C...C(Y_t ... Y_t')_{c...c'}, CnY...Y_{...}, C...CYm_{...}, or CnYm_{...}
      template <typename RandomAccessIterator, typename StateInteger, typename BitInteger, std::size_t num_fused_qubits, typename Qubit2, typename Qubit3, typename... Qubits>
      inline auto pauli_y(
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
        constexpr auto half_num_target_indices = num_target_indices / std::size_t{2u};

        ::ket::gate::fused::gate<num_fused_qubits>(
          first,
          [fused_index_wo_qubits, &unsorted_fused_qubits, &sorted_fused_qubits_with_sentinel](
            auto const first, StateInteger const operated_index_wo_qubits,
            std::array<qubit_type, num_operated_qubits> const& unsorted_operated_qubits,
            std::array<qubit_type, num_operated_qubits + 1u> const& sorted_operated_qubits_with_sentinel)
          {
            // 0b1...10...0u
            constexpr auto base_index = ((std::size_t{1u} << num_control_qubits) - std::size_t{1u}) << num_target_qubits;

            for (auto i = std::size_t{0u}; i < half_num_target_indices; ++i)
            {
              auto const j = num_target_indices - std::size_t{1u} - i;

              auto num_ones_in_i = BitInteger{0u};
              auto num_ones_in_j = BitInteger{0u};
              auto i_tmp = i;
              auto j_tmp = j;
              for (auto count = BitInteger{0u}; count < num_target_qubits; ++count)
              {
                if ((i_tmp bitand std::size_t{1u}) == std::size_t{1u})
                  ++num_ones_in_i;
                if ((j_tmp bitand std::size_t{1u}) == std::size_t{1u})
                  ++num_ones_in_j;

                i_tmp >>= BitInteger{1u};
                j_tmp >>= BitInteger{1u};
              }

              using std::begin;
              using std::end;
              auto const iter1
                = first
                  + ::ket::gate::utility::index_with_qubits(
                      fused_index_wo_qubits,
                      ::ket::gate::utility::index_with_qubits(
                        operated_index_wo_qubits, base_index + i,
                        begin(unsorted_operated_qubits), end(unsorted_operated_qubits),
                        begin(sorted_operated_qubits_with_sentinel), end(sorted_operated_qubits_with_sentinel)),
                      begin(unsorted_fused_qubits), end(unsorted_fused_qubits),
                      begin(sorted_fused_qubits_with_sentinel), end(sorted_fused_qubits_with_sentinel));
              auto const iter2
                = first
                  + ::ket::gate::utility::index_with_qubits(
                      fused_index_wo_qubits,
                      ::ket::gate::utility::index_with_qubits(
                        operated_index_wo_qubits, base_index + j,
                        begin(unsorted_operated_qubits), end(unsorted_operated_qubits),
                        begin(sorted_operated_qubits_with_sentinel), end(sorted_operated_qubits_with_sentinel)),
                      begin(unsorted_fused_qubits), end(unsorted_fused_qubits),
                      begin(sorted_fused_qubits_with_sentinel), end(sorted_fused_qubits_with_sentinel));
              std::iter_swap(iter1, iter2);

              using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
              constexpr auto residual = num_target_qubits % BitInteger{4u};
              constexpr auto coefficient
                = residual == BitInteger{0u}
                  ? complex_type{1}
                  : residual == BitInteger{1u}
                    ? ::ket::utility::imaginary_unit<complex_type>()
                    : residual == BitInteger{2u}
                      ? complex_type{-1}
                      : ::ket::utility::minus_imaginary_unit<complex_type>();
              *iter1 *= (num_target_qubits - num_ones_in_i) % BitInteger{2u} == BitInteger{0u} ? coefficient : -coefficient;
              *iter2 *= (num_target_qubits - num_ones_in_j) % BitInteger{2u} == BitInteger{0u} ? coefficient : -coefficient;
            }
          },
          qubit1, qubit2, qubit3, qubits...);
      }

      template <typename RandomAccessIterator, typename StateInteger, typename BitInteger, std::size_t num_fused_qubits, typename... Qubits>
      inline auto adj_pauli_y(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_fused_qubits > const& unsorted_fused_qubits,
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_fused_qubits + 1u> const& sorted_fused_qubits_with_sentinel,
        ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
      -> void
      { ::ket::gate::fused::pauli_y(first, fused_index_wo_qubits, unsorted_fused_qubits, sorted_fused_qubits_with_sentinel, qubit, qubits...); }


      namespace runtime
      {
        namespace ranges
        {
          // C...CY...Y_{t...t'c...c'} = C...C(Y_t ... Y_t')_{c...c'}, CnY...Y_{...}, C...CYm_{...}, or CnYm_{...}
          template <typename RandomAccessIterator, typename StateInteger, typename QubitsRange1, typename QubitsRange2, typename QubitsRange3, typename ControlQubitsRange>
          inline auto pauli_y(
            RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
            QubitsRange1 const& unsorted_fused_qubits, QubitsRange2 const& sorted_fused_qubits_with_sentinel,
            QubitsRange3 const& target_qubits, ControlQubitsRange const& control_qubits)
          -> void
          {
            using qubit_type = ::ket::utility::meta::range_value_t<QubitsRange3>;
            using control_qubit_type = ::ket::control<qubit_type>;
            using state_integer_type = ::ket::meta::state_integer_t<qubit_type>;
            using bit_integer_type = ::ket::meta::bit_integer_t<qubit_type>;
            static_assert(std::is_unsigned<state_integer_type>::value, "The state_integer_type of value_type of QubitsRange3 should be unsigned");
            static_assert(std::is_unsigned<bit_integer_type>::value, "The bit_integer_type of value_type of QubitsRange3 should be unsigned");
            static_assert(std::is_same< ::ket::utility::meta::range_value_t<ControlQubitsRange>, control_qubit_type >::value, "The value_type of ControlQubitsRange should be the same as ket::control<ket::qubit<S,B>>");

            using std::begin;
            using std::end;
            auto const num_target_qubits = static_cast<bit_integer_type>(end(target_qubits) - begin(target_qubits));
            auto const num_control_qubits = static_cast<bit_integer_type>(end(control_qubits) - begin(control_qubits));
            auto const num_fused_qubits = static_cast<bit_integer_type>(end(unsorted_fused_qubits) - begin(unsorted_fused_qubits));
            auto const num_target_indices = ::ket::utility::integer_exp2<std::size_t>(num_target_qubits);
            auto const half_num_target_indices = num_target_indices / std::size_t{2u};
            assert(static_cast<bit_integer_type>(end(sorted_fused_qubits_with_sentinel) - begin(sorted_fused_qubits_with_sentinel)) == num_fused_qubits + bit_integer_type{1u});
            assert(num_target_qubits + num_control_qubits <= num_fused_qubits);

            assert(::ket::utility::runtime::ranges::all_in_state_vector(num_fused_qubits, target_qubits));
            assert(::ket::utility::runtime::ranges::all_in_state_vector(num_fused_qubits, control_qubits));

            using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
            using real_type = ::ket::utility::meta::real_t<complex_type>;
            auto const residual = num_target_qubits % bit_integer_type{4u};
            auto const coefficient
              = residual == bit_integer_type{0u}
                ? complex_type{real_type{1}}
                : residual == bit_integer_type{1u}
                  ? ::ket::utility::imaginary_unit<complex_type>()
                  : residual == bit_integer_type{2u}
                    ? complex_type{real_type{-1}}
                    : ::ket::utility::minus_imaginary_unit<complex_type>();

            ::ket::gate::fused::runtime::ranges::gate(
              first, num_fused_qubits,
              [fused_index_wo_qubits, &unsorted_fused_qubits, &sorted_fused_qubits_with_sentinel,
               num_target_qubits, num_control_qubits, num_target_indices, half_num_target_indices, &coefficient](
                auto const first, StateInteger const operated_index_wo_qubits,
                auto const& unsorted_operated_qubits, auto const& sorted_operated_qubits_with_sentinel)
              {
                // 0b1...10...0u
                auto const base_index = ((std::size_t{1u} << num_control_qubits) - std::size_t{1u}) << num_target_qubits;

                for (auto i = std::size_t{0u}; i < half_num_target_indices; ++i)
                {
                  auto const j = num_target_indices - std::size_t{1u} - i;

                  auto num_ones_in_i = bit_integer_type{0u};
                  auto num_ones_in_j = bit_integer_type{0u};
                  auto i_tmp = i;
                  auto j_tmp = j;
                  for (auto count = bit_integer_type{0u}; count < num_target_qubits; ++count)
                  {
                    if ((i_tmp bitand std::size_t{1u}) == std::size_t{1u})
                      ++num_ones_in_i;
                    if ((j_tmp bitand std::size_t{1u}) == std::size_t{1u})
                      ++num_ones_in_j;

                    i_tmp >>= bit_integer_type{1u};
                    j_tmp >>= bit_integer_type{1u};
                  }

                  using std::begin;
                  using std::end;
                  auto const iter1
                    = first
                      + ::ket::gate::utility::ranges::index_with_qubits(
                          fused_index_wo_qubits,
                          ::ket::gate::utility::ranges::index_with_qubits(
                            operated_index_wo_qubits, base_index + i,
                            unsorted_operated_qubits, sorted_operated_qubits_with_sentinel),
                          unsorted_fused_qubits, sorted_fused_qubits_with_sentinel);
                  auto const iter2
                    = first
                      + ::ket::gate::utility::ranges::index_with_qubits(
                          fused_index_wo_qubits,
                          ::ket::gate::utility::ranges::index_with_qubits(
                            operated_index_wo_qubits, base_index + j,
                            unsorted_operated_qubits, sorted_operated_qubits_with_sentinel),
                          unsorted_fused_qubits, sorted_fused_qubits_with_sentinel);
                  std::iter_swap(iter1, iter2);

                  *iter1 *= (num_target_qubits - num_ones_in_i) % bit_integer_type{2u} == bit_integer_type{0u} ? coefficient : -coefficient;
                  *iter2 *= (num_target_qubits - num_ones_in_j) % bit_integer_type{2u} == bit_integer_type{0u} ? coefficient : -coefficient;
                }
              },
              boost::join(
                target_qubits,
                control_qubits | boost::adaptors::transformed(
                  [](control_qubit_type const control_qubit) { return control_qubit.qubit(); })));
          }

          template <typename RandomAccessIterator, typename StateInteger, typename QubitsRange1, typename QubitsRange2, typename QubitsRange3>
          inline auto pauli_y(
            RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
            QubitsRange1 const& unsorted_fused_qubits, QubitsRange2 const& sorted_fused_qubits_with_sentinel,
            QubitsRange3 const& target_qubits)
          -> void
          {
            using qubit_type = ::ket::utility::meta::range_value_t<QubitsRange3>;
            using control_qubit_type = ::ket::control<qubit_type>;
            std::array<control_qubit_type, 0u> const control_qubits{};
            ::ket::gate::fused::runtime::ranges::pauli_y(
              first, fused_index_wo_qubits,
              unsorted_fused_qubits, sorted_fused_qubits_with_sentinel,
              target_qubits, control_qubits);
          }

          template <typename RandomAccessIterator, typename StateInteger, typename QubitsRange1, typename QubitsRange2, typename QubitsRange3, typename ControlQubitsRange>
          inline auto adj_pauli_y(
            RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
            QubitsRange1 const& unsorted_fused_qubits, QubitsRange2 const& sorted_fused_qubits_with_sentinel,
            QubitsRange3 const& target_qubits, ControlQubitsRange const& control_qubits)
          -> void
          {
            ::ket::gate::fused::runtime::ranges::pauli_y(
              first, fused_index_wo_qubits,
              unsorted_fused_qubits, sorted_fused_qubits_with_sentinel,
              target_qubits, control_qubits);
          }

          template <typename RandomAccessIterator, typename StateInteger, typename QubitsRange1, typename QubitsRange2, typename QubitsRange3>
          inline auto adj_pauli_y(
            RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
            QubitsRange1 const& unsorted_fused_qubits, QubitsRange2 const& sorted_fused_qubits_with_sentinel,
            QubitsRange3 const& target_qubits)
          -> void
          {
            ::ket::gate::fused::runtime::ranges::pauli_y(
              first, fused_index_wo_qubits,
              unsorted_fused_qubits, sorted_fused_qubits_with_sentinel,
              target_qubits);
          }
        } // namespace ranges

        template <typename RandomAccessIterator, typename StateInteger, typename QubitIterator1, typename QubitIterator2, typename QubitIterator3, typename ControlQubitIterator>
        inline auto pauli_y(
          RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
          QubitIterator1 const unsorted_fused_qubit_first, QubitIterator1 const unsorted_fused_qubit_last,
          QubitIterator2 const sorted_fused_qubit_with_sentinel_first, QubitIterator2 const sorted_fused_qubit_with_sentinel_last,
          QubitIterator3 const target_qubit_first, QubitIterator3 const target_qubit_last,
          ControlQubitIterator const control_qubit_first, ControlQubitIterator const control_qubit_last)
        -> void
        {
          ::ket::gate::fused::runtime::ranges::pauli_y(
            first, fused_index_wo_qubits,
            boost::make_iterator_range(unsorted_fused_qubit_first, unsorted_fused_qubit_last),
            boost::make_iterator_range(sorted_fused_qubit_with_sentinel_first, sorted_fused_qubit_with_sentinel_last),
            boost::make_iterator_range(target_qubit_first, target_qubit_last),
            boost::make_iterator_range(control_qubit_first, control_qubit_last));
        }

        template <typename RandomAccessIterator, typename StateInteger, typename QubitIterator1, typename QubitIterator2, typename QubitIterator3>
        inline auto pauli_y(
          RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
          QubitIterator1 const unsorted_fused_qubit_first, QubitIterator1 const unsorted_fused_qubit_last,
          QubitIterator2 const sorted_fused_qubit_with_sentinel_first, QubitIterator2 const sorted_fused_qubit_with_sentinel_last,
          QubitIterator3 const target_qubit_first, QubitIterator3 const target_qubit_last)
        -> void
        {
          ::ket::gate::fused::runtime::ranges::pauli_y(
            first, fused_index_wo_qubits,
            boost::make_iterator_range(unsorted_fused_qubit_first, unsorted_fused_qubit_last),
            boost::make_iterator_range(sorted_fused_qubit_with_sentinel_first, sorted_fused_qubit_with_sentinel_last),
            boost::make_iterator_range(target_qubit_first, target_qubit_last));
        }

        template <typename RandomAccessIterator, typename StateInteger, typename QubitIterator1, typename QubitIterator2, typename QubitIterator3, typename ControlQubitIterator>
        inline auto adj_pauli_y(
          RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
          QubitIterator1 const unsorted_fused_qubit_first, QubitIterator1 const unsorted_fused_qubit_last,
          QubitIterator2 const sorted_fused_qubit_with_sentinel_first, QubitIterator2 const sorted_fused_qubit_with_sentinel_last,
          QubitIterator3 const target_qubit_first, QubitIterator3 const target_qubit_last,
          ControlQubitIterator const control_qubit_first, ControlQubitIterator const control_qubit_last)
        -> void
        {
          ::ket::gate::fused::runtime::ranges::adj_pauli_y(
            first, fused_index_wo_qubits,
            boost::make_iterator_range(unsorted_fused_qubit_first, unsorted_fused_qubit_last),
            boost::make_iterator_range(sorted_fused_qubit_with_sentinel_first, sorted_fused_qubit_with_sentinel_last),
            boost::make_iterator_range(target_qubit_first, target_qubit_last),
            boost::make_iterator_range(control_qubit_first, control_qubit_last));
        }

        template <typename RandomAccessIterator, typename StateInteger, typename QubitIterator1, typename QubitIterator2, typename QubitIterator3>
        inline auto adj_pauli_y(
          RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
          QubitIterator1 const unsorted_fused_qubit_first, QubitIterator1 const unsorted_fused_qubit_last,
          QubitIterator2 const sorted_fused_qubit_with_sentinel_first, QubitIterator2 const sorted_fused_qubit_with_sentinel_last,
          QubitIterator3 const target_qubit_first, QubitIterator3 const target_qubit_last)
        -> void
        {
          ::ket::gate::fused::runtime::ranges::adj_pauli_y(
            first, fused_index_wo_qubits,
            boost::make_iterator_range(unsorted_fused_qubit_first, unsorted_fused_qubit_last),
            boost::make_iterator_range(sorted_fused_qubit_with_sentinel_first, sorted_fused_qubit_with_sentinel_last),
            boost::make_iterator_range(target_qubit_first, target_qubit_last));
        }
      } // namespace runtime
# else // KET_USE_BIT_MASKS_EXPLICITLY
      // Y_i or Y1_i
      // Y_1 (a_0 |0> + a_1 |1>) = -i a_1 |0> + i a_0 |1>
      template <typename RandomAccessIterator, typename StateInteger, std::size_t num_fused_qubits, typename BitInteger>
      inline auto pauli_y(
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
          using std::begin;
          using std::end;
          auto const zero_iter
            = first
              + ::ket::gate::utility::index_with_qubits(
                  fused_index_wo_qubits, zero_index,
                  begin(fused_qubit_masks), end(fused_qubit_masks), begin(fused_index_masks), end(fused_index_masks));
          auto const one_iter
            = first
              + ::ket::gate::utility::index_with_qubits(
                  fused_index_wo_qubits, one_index,
                  begin(fused_qubit_masks), end(fused_qubit_masks), begin(fused_index_masks), end(fused_index_masks));

          std::iter_swap(zero_iter, one_iter);

          using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
          *zero_iter *= ::ket::utility::minus_imaginary_unit<complex_type>();
          *one_iter *= ::ket::utility::imaginary_unit<complex_type>();
        }
      }

      // YY_{ij} = Y_i Y_j or Y2_{ij}
      // YY_{1,2} (a_{00} |00> + a_{01} |01> + a_{10} |10> + a{11} |11>)
      //   = -a_{11} |00> + a_{10} |01> + a_{01} |10> - a_{00} |11>
      template <typename RandomAccessIterator, typename StateInteger, std::size_t num_fused_qubits, typename BitInteger>
      inline auto pauli_y(
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
          using std::begin;
          using std::end;
          auto const off_iter
            = first
              + ::ket::gate::utility::index_with_qubits(
                  fused_index_wo_qubits, base_index,
                  begin(fused_qubit_masks), end(fused_qubit_masks), begin(fused_index_masks), end(fused_index_masks));
          // xxx1_1xxx0_2xxx
          auto const qubit1_on_index = base_index bitor qubit1_mask;
          auto const qubit1_on_iter
            = first
              + ::ket::gate::utility::index_with_qubits(
                  fused_index_wo_qubits, qubit1_on_index,
                  begin(fused_qubit_masks), end(fused_qubit_masks), begin(fused_index_masks), end(fused_index_masks));
          // xxx0_1xxx1_2xxx
          auto const qubit2_on_iter
            = first
              + ::ket::gate::utility::index_with_qubits(
                  fused_index_wo_qubits, base_index bitor qubit2_mask,
                  begin(fused_qubit_masks), end(fused_qubit_masks), begin(fused_index_masks), end(fused_index_masks));
          // xxx1_1xxx1_2xxx
          auto const qubit12_on_iter
            = first
              + ::ket::gate::utility::index_with_qubits(
                  fused_index_wo_qubits, qubit1_on_index bitor qubit2_mask,
                  begin(fused_qubit_masks), end(fused_qubit_masks), begin(fused_index_masks), end(fused_index_masks));

          std::iter_swap(off_iter, qubit12_on_iter);
          std::iter_swap(qubit1_on_iter, qubit2_on_iter);

          using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
          using real_type = ::ket::utility::meta::real_t<complex_type>;
          *off_iter *= real_type{-1.0};
          *qubit12_on_iter *= real_type{-1.0};
        }
      }

      // CY_{tc}, CY1_{tc}, C1Y_{tc}, or C1Y1_{tc}
      // CY_{1,2} (a_{00} |00> + a_{01} |01> + a_{10} |10> + a{11} |11>)
      //   = a_{00} |00> + a_{01} |01> - i a_{11} |10> + i a_{10} |11>
      template <typename RandomAccessIterator, typename StateInteger, std::size_t num_fused_qubits, typename BitInteger>
      inline auto pauli_y(
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
          // xxx0_txxx1_cxxx
          auto const control_on_index = base_index bitor control_qubit_mask;
          // xxx1_txxx1_cxxx
          auto const target_control_on_index = control_on_index bitor target_qubit_mask;
          using std::begin;
          using std::end;
          auto const control_on_iter
            = first
              + ::ket::gate::utility::index_with_qubits(
                  fused_index_wo_qubits, control_on_index,
                  begin(fused_qubit_masks), end(fused_qubit_masks), begin(fused_index_masks), end(fused_index_masks));
          auto const target_control_on_iter
            = first
              + ::ket::gate::utility::index_with_qubits(
                  fused_index_wo_qubits, target_control_on_index,
                  begin(fused_qubit_masks), end(fused_qubit_masks), begin(fused_index_masks), end(fused_index_masks));

          std::iter_swap(control_on_iter, target_control_on_iter);

          using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
          *control_on_iter *= ::ket::utility::minus_imaginary_unit<complex_type>();
          *target_control_on_iter *= ::ket::utility::imaginary_unit<complex_type>();
        }
      }

      // C...CY...Y_{t...t'c...c'} = C...C(Y_t ... Y_t')_{c...c'}, CnY...Y_{...}, C...CYm_{...}, or CnYm_{...}
      template <typename RandomAccessIterator, typename StateInteger, std::size_t num_fused_qubits, typename BitInteger, typename Qubit2, typename Qubit3, typename... Qubits>
      inline auto pauli_y(
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
        constexpr auto half_num_target_indices = num_target_indices / std::size_t{2u};

        constexpr auto residual = num_target_qubits % BitInteger{4u};
        ::ket::gate::fused::gate<num_fused_qubits>(
          first,
          [fused_index_wo_qubits, &fused_qubit_masks, &fused_index_masks](
            auto const first, StateInteger const operated_index_wo_qubits,
            std::array<StateInteger, num_operated_qubits> const& operated_qubit_masks,
            std::array<StateInteger, num_operated_qubits + 1u> const& operated_index_masks)
          {
            // 0b1...10...0u
            constexpr auto base_index = ((std::size_t{1u} << num_control_qubits) - std::size_t{1u}) << num_target_qubits;

            for (auto i = std::size_t{0u}; i < half_num_target_indices; ++i)
            {
              auto const j = num_target_indices - std::size_t{1u} - i;

              auto num_ones_in_i = BitInteger{0u};
              auto num_ones_in_j = BitInteger{0u};
              auto i_tmp = i;
              auto j_tmp = j;
              for (auto count = BitInteger{0u}; count < num_target_qubits; ++count)
              {
                if ((i_tmp bitand std::size_t{1u}) == std::size_t{1u})
                  ++num_ones_in_i;
                if ((j_tmp bitand std::size_t{1u}) == std::size_t{1u})
                  ++num_ones_in_j;

                i_tmp >>= BitInteger{1u};
                j_tmp >>= BitInteger{1u};
              }

              using std::begin;
              using std::end;
              auto const iter1
                = first
                  + ::ket::gate::utility::index_with_qubits(
                      fused_index_wo_qubits,
                      ::ket::gate::utility::index_with_qubits(
                        operated_index_wo_qubits, base_index + i,
                        begin(operated_qubit_masks), end(operated_qubit_masks), begin(operated_index_masks), end(operated_index_masks)),
                      begin(fused_qubit_masks), end(fused_qubit_masks), begin(fused_index_masks), end(fused_index_masks));
              auto const iter2
                = first
                  + ::ket::gate::utility::index_with_qubits(
                      fused_index_wo_qubits,
                      ::ket::gate::utility::index_with_qubits(
                        operated_index_wo_qubits, base_index + j,
                        begin(operated_qubit_masks), end(operated_qubit_masks), begin(operated_index_masks), end(operated_index_masks)),
                      begin(fused_qubit_masks), end(fused_qubit_masks), begin(fused_index_masks), end(fused_index_masks));
              std::iter_swap(iter1, iter2);

              using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
              constexpr auto coefficient
                = residual == BitInteger{0u}
                  ? complex_type{1}
                  : residual == BitInteger{1u}
                    ? ::ket::utility::imaginary_unit<complex_type>()
                    : residual == BitInteger{2u}
                      ? complex_type{-1}
                      : ::ket::utility::minus_imaginary_unit<complex_type>();
              *iter1 *= (num_target_qubits - num_ones_in_i) % BitInteger{2u} == BitInteger{0u} ? coefficient : -coefficient;
              *iter2 *= (num_target_qubits - num_ones_in_j) % BitInteger{2u} == BitInteger{0u} ? coefficient : -coefficient;
            }
          },
          qubit1, qubit2, qubit3, qubits...);
      }

      template <typename RandomAccessIterator, typename StateInteger, std::size_t num_fused_qubits, typename BitInteger, typename... Qubits>
      inline auto adj_pauli_y(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array<StateInteger, num_fused_qubits> const& fused_qubit_masks, std::array<StateInteger, num_fused_qubits + 1u> const& fused_index_masks,
        ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
      -> void
      { ::ket::gate::fused::pauli_y(first, fused_index_wo_qubits, fused_qubit_masks, fused_index_masks, qubit, qubits...); }


      namespace runtime
      {
        namespace ranges
        {
          // C...CX...X_{t...t'c...c'} = C...C(X_t ... X_t')_{c...c'}, CnX...X_{...}, C...CXm_{...}, or CnXm_{...}
          template <typename RandomAccessIterator, typename StateInteger, typename StateIntegersRange1, typename StateIntegersRange2, typename QubitsRange, typename ControlQubitsRange>
          inline auto pauli_y(
            RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
            StateIntegersRange1 const& fused_qubit_masks, StateIntegersRange2 const& fused_index_masks,
            QubitsRange const& target_qubits, ControlQubitsRange const& control_qubits)
          -> void
          {
            using qubit_type = ::ket::utility::meta::range_value_t<QubitsRange>;
            using control_qubit_type = ::ket::control<qubit_type>;
            using state_integer_type = ::ket::meta::state_integer_t<qubit_type>;
            using bit_integer_type = ::ket::meta::bit_integer_t<qubit_type>;
            static_assert(std::is_unsigned<state_integer_type>::value, "The state_integer_type of value_type of QubitsRange should be unsigned");
            static_assert(std::is_unsigned<bit_integer_type>::value, "The bit_integer_type of value_type of QubitsRange should be unsigned");
            static_assert(std::is_same< ::ket::utility::meta::range_value_t<ControlQubitsRange>, control_qubit_type >::value, "The value_type of ControlQubitsRange should be the same as ket::control<ket::qubit<S,B>>");

            using std::begin;
            using std::end;
            auto const num_target_qubits = static_cast<bit_integer_type>(end(target_qubits) - begin(target_qubits));
            auto const num_control_qubits = static_cast<bit_integer_type>(end(control_qubits) - begin(control_qubits));
            auto const num_fused_qubits = static_cast<bit_integer_type>(end(fused_qubit_masks) - begin(fused_qubit_masks));
            auto const num_target_indices = ::ket::utility::integer_exp2<std::size_t>(num_target_qubits);
            auto const half_num_target_indices = num_target_indices / std::size_t{2u};
            assert(static_cast<bit_integer_type>(end(fused_index_masks) - begin(fused_index_masks)) == num_fused_qubits + bit_integer_type{1u});
            assert(num_target_qubits + num_control_qubits <= num_fused_qubits);

            assert(::ket::utility::runtime::ranges::all_in_state_vector(num_fused_qubits, target_qubits));
            assert(::ket::utility::runtime::ranges::all_in_state_vector(num_fused_qubits, control_qubits));

            using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
            using real_type = ::ket::utility::meta::real_t<complex_type>;
            auto const residual = num_target_qubits % bit_integer_type{4u};
            auto const coefficient
              = residual == bit_integer_type{0u}
                ? complex_type{real_type{1}}
                : residual == bit_integer_type{1u}
                  ? ::ket::utility::imaginary_unit<complex_type>()
                  : residual == bit_integer_type{2u}
                    ? complex_type{real_type{-1}}
                    : ::ket::utility::minus_imaginary_unit<complex_type>();

            ::ket::gate::fused::runtime::ranges::gate(
              first, num_fused_qubits,
              [fused_index_wo_qubits, &fused_qubit_masks, &fused_index_masks,
               num_target_qubits, num_control_qubits, num_target_indices, half_num_target_indices, &coefficient](
                auto const first, StateInteger const operated_index_wo_qubits,
                auto const& operated_qubit_masks, auto const& operated_index_masks)
              {
                // 0b1...10...0u
                auto const base_index = ((std::size_t{1u} << num_control_qubits) - std::size_t{1u}) << num_target_qubits;

                for (auto i = std::size_t{0u}; i < half_num_target_indices; ++i)
                {
                  auto const j = num_target_indices - std::size_t{1u} - i;

                  auto num_ones_in_i = bit_integer_type{0u};
                  auto num_ones_in_j = bit_integer_type{0u};
                  auto i_tmp = i;
                  auto j_tmp = j;
                  for (auto count = bit_integer_type{0u}; count < num_target_qubits; ++count)
                  {
                    if ((i_tmp bitand std::size_t{1u}) == std::size_t{1u})
                      ++num_ones_in_i;
                    if ((j_tmp bitand std::size_t{1u}) == std::size_t{1u})
                      ++num_ones_in_j;

                    i_tmp >>= bit_integer_type{1u};
                    j_tmp >>= bit_integer_type{1u};
                  }

                  using std::begin;
                  using std::end;
                  auto const iter1
                    = first
                      + ::ket::gate::utility::ranges::index_with_qubits(
                          fused_index_wo_qubits,
                          ::ket::gate::utility::ranges::index_with_qubits(
                            operated_index_wo_qubits, base_index + i,
                            operated_qubit_masks, operated_index_masks),
                          fused_qubit_masks, fused_index_masks);
                  auto const iter2
                    = first
                      + ::ket::gate::utility::ranges::index_with_qubits(
                          fused_index_wo_qubits,
                          ::ket::gate::utility::ranges::index_with_qubits(
                            operated_index_wo_qubits, base_index + j,
                            operated_qubit_masks, operated_index_masks),
                          fused_qubit_masks, fused_index_masks);
                  std::iter_swap(iter1, iter2);

                  *iter1 *= (num_target_qubits - num_ones_in_i) % bit_integer_type{2u} == bit_integer_type{0u} ? coefficient : -coefficient;
                  *iter2 *= (num_target_qubits - num_ones_in_j) % bit_integer_type{2u} == bit_integer_type{0u} ? coefficient : -coefficient;
                }
              },
              boost::join(
                target_qubits,
                control_qubits | boost::adaptors::transformed(
                  [](control_qubit_type const control_qubit) { return control_qubit.qubit(); })));
          }

          template <typename RandomAccessIterator, typename StateInteger, typename StateIntegersRange1, typename StateIntegersRange2, typename QubitsRange>
          inline auto pauli_y(
            RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
            StateIntegersRange1 const& fused_qubit_masks, StateIntegersRange2 const& fused_index_masks,
            QubitsRange const& target_qubits)
          -> void
          {
            using qubit_type = ::ket::utility::meta::range_value_t<QubitsRange>;
            using control_qubit_type = ::ket::control<qubit_type>;
            std::array<control_qubit_type, 0u> const control_qubits{};
            ::ket::gate::fused::runtime::ranges::pauli_y(
              first, fused_index_wo_qubits,
              fused_qubit_masks, fused_index_masks,
              target_qubits, control_qubits);
          }

          template <typename RandomAccessIterator, typename StateInteger, typename StateIntegersRange1, typename StateIntegersRange2, typename QubitsRange, typename ControlQubitsRange>
          inline auto adj_pauli_y(
            RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
            StateIntegersRange1 const& fused_qubit_masks, StateIntegersRange2 const& fused_index_masks,
            QubitsRange const& target_qubits, ControlQubitsRange const& control_qubits)
          -> void
          {
            ::ket::gate::fused::runtime::ranges::pauli_y(
              first, fused_index_wo_qubits,
              fused_qubit_masks, fused_index_masks,
              target_qubits, control_qubits);
          }

          template <typename RandomAccessIterator, typename StateInteger, typename StateIntegersRange1, typename StateIntegersRange2, typename QubitsRange>
          inline auto adj_pauli_y(
            RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
            StateIntegersRange1 const& fused_qubit_masks, StateIntegersRange2 const& fused_index_masks,
            QubitsRange const& target_qubits)
          -> void
          {
            ::ket::gate::fused::runtime::ranges::pauli_y(
              first, fused_index_wo_qubits,
              fused_qubit_masks, fused_index_masks,
              target_qubits);
          }
        } // namespace ranges

        template <typename RandomAccessIterator, typename StateInteger, typename StateIntegerIterator1, typename StateIntegerIterator2, typename QubitIterator, typename ControlQubitIterator>
        inline auto pauli_y(
          RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
          StateIntegerIterator1 const fused_qubit_mask_first, StateIntegerIterator1 const fused_qubit_mask_last,
          StateIntegerIterator2 const fused_index_mask_first, StateIntegerIterator2 const fused_index_mask_last,
          QubitIterator const target_qubit_first, QubitIterator const target_qubit_last,
          ControlQubitIterator const control_qubit_first, ControlQubitIterator const control_qubit_last)
        -> void
        {
          ::ket::gate::fused::runtime::ranges::pauli_y(
            first, fused_index_wo_qubits,
            boost::make_iterator_range(fused_qubit_mask_first, fused_qubit_mask_last),
            boost::make_iterator_range(fused_index_mask_first, fused_index_mask_last),
            boost::make_iterator_range(target_qubit_first, target_qubit_last),
            boost::make_iterator_range(control_qubit_first, control_qubit_last));
        }

        template <typename RandomAccessIterator, typename StateInteger, typename StateIntegerIterator1, typename StateIntegerIterator2, typename QubitIterator>
        inline auto pauli_y(
          RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
          StateIntegerIterator1 const fused_qubit_mask_first, StateIntegerIterator1 const fused_qubit_mask_last,
          StateIntegerIterator2 const fused_index_mask_first, StateIntegerIterator2 const fused_index_mask_last,
          QubitIterator const target_qubit_first, QubitIterator const target_qubit_last)
        -> void
        {
          ::ket::gate::fused::runtime::ranges::pauli_y(
            first, fused_index_wo_qubits,
            boost::make_iterator_range(fused_qubit_mask_first, fused_qubit_mask_last),
            boost::make_iterator_range(fused_index_mask_first, fused_index_mask_last),
            boost::make_iterator_range(target_qubit_first, target_qubit_last));
        }

        template <typename RandomAccessIterator, typename StateInteger, typename StateIntegerIterator1, typename StateIntegerIterator2, typename QubitIterator, typename ControlQubitIterator>
        inline auto adj_pauli_y(
          RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
          StateIntegerIterator1 const fused_qubit_mask_first, StateIntegerIterator1 const fused_qubit_mask_last,
          StateIntegerIterator2 const fused_index_mask_first, StateIntegerIterator2 const fused_index_mask_last,
          QubitIterator const target_qubit_first, QubitIterator const target_qubit_last,
          ControlQubitIterator const control_qubit_first, ControlQubitIterator const control_qubit_last)
        -> void
        {
          ::ket::gate::fused::runtime::ranges::adj_pauli_y(
            first, fused_index_wo_qubits,
            boost::make_iterator_range(fused_qubit_mask_first, fused_qubit_mask_last),
            boost::make_iterator_range(fused_index_mask_first, fused_index_mask_last),
            boost::make_iterator_range(target_qubit_first, target_qubit_last),
            boost::make_iterator_range(control_qubit_first, control_qubit_last));
        }

        template <typename RandomAccessIterator, typename StateInteger, typename StateIntegerIterator1, typename StateIntegerIterator2, typename QubitIterator>
        inline auto adj_pauli_y(
          RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
          StateIntegerIterator1 const fused_qubit_mask_first, StateIntegerIterator1 const fused_qubit_mask_last,
          StateIntegerIterator2 const fused_index_mask_first, StateIntegerIterator2 const fused_index_mask_last,
          QubitIterator const target_qubit_first, QubitIterator const target_qubit_last)
        -> void
        {
          ::ket::gate::fused::runtime::ranges::adj_pauli_y(
            first, fused_index_wo_qubits,
            boost::make_iterator_range(fused_qubit_mask_first, fused_qubit_mask_last),
            boost::make_iterator_range(fused_index_mask_first, fused_index_mask_last),
            boost::make_iterator_range(target_qubit_first, target_qubit_last));
        }
      } // namespace runtime
# endif // KET_USE_BIT_MASKS_EXPLICITLY
    } // namespace fused
  } // namespace gate
} // namespace ket


#endif // KET_GATE_FUSED_PAULI_Y_HPP
