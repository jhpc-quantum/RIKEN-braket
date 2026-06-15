#ifndef KET_GATE_FUSED_Y_ROTATION_HALF_PI_HPP
# define KET_GATE_FUSED_Y_ROTATION_HALF_PI_HPP

# include <cassert>
# include <cstddef>
# include <array>
# include <algorithm>
# include <iterator>
# include <utility>
# include <type_traits>

# include <boost/math/constants/constants.hpp>
# include <boost/range/iterator_range.hpp>
# include <boost/range/join.hpp>
# include <boost/range/adaptor/transformed.hpp>

# include <ket/qubit.hpp>
# include <ket/control.hpp>
# include <ket/gate/utility/index_with_qubits.hpp>
# include <ket/gate/fused/gate.hpp>
# include <ket/utility/integer_exp2.hpp>
# include <ket/utility/integer_log2.hpp>
# include <ket/utility/imaginary_unit.hpp>
# include <ket/utility/meta/ranges.hpp>
# include <ket/utility/meta/real_of.hpp>


namespace ket
{
  namespace gate
  {
    namespace fused
    {
# ifndef KET_USE_BIT_MASKS_EXPLICITLY
      // +Y_i
      // +Y_1 (a_0 |0> + a_1 |1>) = (a_0 + a_1)/sqrt(2) |0> + (-a_0 + a_1)/sqrt(2) |1>
      template <typename RandomAccessIterator, typename StateInteger, typename BitInteger, std::size_t num_fused_qubits>
      inline auto y_rotation_half_pi(
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
          auto const zero_iter_value = *zero_iter;

          using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
          using real_type = ::ket::utility::meta::real_t<complex_type>;
          using boost::math::constants::one_div_root_two;
          *zero_iter += *one_iter;
          *zero_iter *= one_div_root_two<real_type>();
          *one_iter -= zero_iter_value;
          *one_iter *= one_div_root_two<real_type>();
        }
      }

      // C+Y_{tc} or C1+Y_{tc}
      // C+Y_{1,2} (a_{00} |00> + a_{01} |01> + a_{10} |10> + a_{11} |11>)
      //   = a_{00} |00> + a_{01} |01> + (a_{10} + a_{11})/sqrt(2) |10> + (-a_{10} + a_{11})/sqrt(2) |11>
      template <typename RandomAccessIterator, typename StateInteger, typename BitInteger, std::size_t num_fused_qubits>
      inline auto y_rotation_half_pi(
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
          auto const control_on_iter_value = *control_on_iter;

          using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
          using real_type = ::ket::utility::meta::real_t<complex_type>;
          using boost::math::constants::one_div_root_two;
          *control_on_iter += *target_control_on_iter;
          *control_on_iter *= one_div_root_two<real_type>();
          *target_control_on_iter -= control_on_iter_value;
          *target_control_on_iter *= one_div_root_two<real_type>();
        }
      }

      // C...C+Y_{tc...c'} or Cn+Y_{tc...c'}
      template <typename RandomAccessIterator, typename StateInteger, typename BitInteger, std::size_t num_fused_qubits, typename... ControlQubits>
      inline auto y_rotation_half_pi(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_fused_qubits > const& unsorted_fused_qubits,
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_fused_qubits + 1u> const& sorted_fused_qubits_with_sentinel,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2, ControlQubits const... control_qubits)
      -> void
      {
        using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
        constexpr auto num_control_qubits = static_cast<BitInteger>(sizeof...(ControlQubits) + 2u);
        constexpr auto num_operated_qubits = num_control_qubits + BitInteger{1u};
        ::ket::gate::fused::gate<num_fused_qubits>(
          first,
          [fused_index_wo_qubits, &unsorted_fused_qubits, &sorted_fused_qubits_with_sentinel](
            auto const first, StateInteger const operated_index_wo_qubits,
            std::array<qubit_type, num_operated_qubits> const& unsorted_operated_qubits,
            std::array<qubit_type, num_operated_qubits + 1u> const& sorted_operated_qubits_with_sentinel)
          {
            // 0b11...10u
            constexpr auto index0 = ((std::size_t{1u} << num_control_qubits) - std::size_t{1u}) << BitInteger{1u};
            // 0b11...11u
            constexpr auto index1 = index0 bitor std::size_t{1u};

            using std::begin;
            using std::end;
            auto const iter0
              = first
                + ::ket::gate::utility::index_with_qubits(
                    fused_index_wo_qubits,
                    ::ket::gate::utility::index_with_qubits(
                      operated_index_wo_qubits, index0,
                      begin(unsorted_operated_qubits), end(unsorted_operated_qubits),
                      begin(sorted_operated_qubits_with_sentinel), end(sorted_operated_qubits_with_sentinel)),
                    begin(unsorted_fused_qubits), end(unsorted_fused_qubits),
                    begin(sorted_fused_qubits_with_sentinel), end(sorted_fused_qubits_with_sentinel));
            auto const iter1
              = first
                + ::ket::gate::utility::index_with_qubits(
                    fused_index_wo_qubits,
                    ::ket::gate::utility::index_with_qubits(
                      operated_index_wo_qubits, index1,
                      begin(unsorted_operated_qubits), end(unsorted_operated_qubits),
                      begin(sorted_operated_qubits_with_sentinel), end(sorted_operated_qubits_with_sentinel)),
                    begin(unsorted_fused_qubits), end(unsorted_fused_qubits),
                    begin(sorted_fused_qubits_with_sentinel), end(sorted_fused_qubits_with_sentinel));
            auto const value0 = *iter0;

            using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
            using real_type = ::ket::utility::meta::real_t<complex_type>;
            using boost::math::constants::one_div_root_two;
            *iter0 += *iter1;
            *iter0 *= one_div_root_two<real_type>();
            *iter1 -= value0;
            *iter1 *= one_div_root_two<real_type>();
          },
          target_qubit, control_qubit1, control_qubit2, control_qubits...);
      }

      // -Y_i
      // -Y_1 (a_0 |0> + a_1 |1>) = (a_0 - a_1)/sqrt(2) |0> + (a_0 + a_1)/sqrt(2) |1>
      template <typename RandomAccessIterator, typename StateInteger, typename BitInteger, std::size_t num_fused_qubits>
      inline auto adj_y_rotation_half_pi(
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
          auto const zero_iter_value = *zero_iter;

          using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
          using real_type = ::ket::utility::meta::real_t<complex_type>;
          using boost::math::constants::one_div_root_two;
          *zero_iter -= *one_iter;
          *zero_iter *= one_div_root_two<real_type>();
          *one_iter += zero_iter_value;
          *one_iter *= one_div_root_two<real_type>();
        }
      }

      // C-Y_{tc} or C1-Y_{tc}
      // C-Y_{1,2} (a_{00} |00> + a_{01} |01> + a_{10} |10> + a_{11} |11>)
      //   = a_{00} |00> + a_{01} |01> + (a_{10} - a_{11})/sqrt(2) |10> + (a_{10} + a_{11})/sqrt(2) |11>
      template <typename RandomAccessIterator, typename StateInteger, typename BitInteger, std::size_t num_fused_qubits>
      inline auto adj_y_rotation_half_pi(
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
          auto const control_on_iter_value = *control_on_iter;

          using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
          using real_type = ::ket::utility::meta::real_t<complex_type>;
          using boost::math::constants::one_div_root_two;
          *control_on_iter -= *target_control_on_iter;
          *control_on_iter *= one_div_root_two<real_type>();
          *target_control_on_iter += control_on_iter_value;
          *target_control_on_iter *= one_div_root_two<real_type>();
        }
      }

      // C...C-Y_{tc...c'} or Cn-Y_{tc...c'}
      template <typename RandomAccessIterator, typename StateInteger, typename BitInteger, std::size_t num_fused_qubits, typename... ControlQubits>
      inline auto adj_y_rotation_half_pi(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_fused_qubits > const& unsorted_fused_qubits,
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_fused_qubits + 1u> const& sorted_fused_qubits_with_sentinel,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2, ControlQubits const... control_qubits)
      -> void
      {
        using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
        constexpr auto num_control_qubits = static_cast<BitInteger>(sizeof...(ControlQubits) + 2u);
        constexpr auto num_operated_qubits = num_control_qubits + BitInteger{1u};
        ::ket::gate::fused::gate<num_fused_qubits>(
          first,
          [fused_index_wo_qubits, &unsorted_fused_qubits, &sorted_fused_qubits_with_sentinel](
            auto const first, StateInteger const operated_index_wo_qubits,
            std::array<qubit_type, num_operated_qubits> const& unsorted_operated_qubits,
            std::array<qubit_type, num_operated_qubits + 1u> const& sorted_operated_qubits_with_sentinel)
          {
            // 0b11...10u
            constexpr auto index0 = ((std::size_t{1u} << num_control_qubits) - std::size_t{1u}) << BitInteger{1u};
            // 0b11...11u
            constexpr auto index1 = index0 bitor std::size_t{1u};

            using std::begin;
            using std::end;
            auto const iter0
              = first
                + ::ket::gate::utility::index_with_qubits(
                    fused_index_wo_qubits,
                    ::ket::gate::utility::index_with_qubits(
                      operated_index_wo_qubits, index0,
                      begin(unsorted_operated_qubits), end(unsorted_operated_qubits),
                      begin(sorted_operated_qubits_with_sentinel), end(sorted_operated_qubits_with_sentinel)),
                    begin(unsorted_fused_qubits), end(unsorted_fused_qubits),
                    begin(sorted_fused_qubits_with_sentinel), end(sorted_fused_qubits_with_sentinel));
            auto const iter1
              = first
                + ::ket::gate::utility::index_with_qubits(
                    fused_index_wo_qubits,
                    ::ket::gate::utility::index_with_qubits(
                      operated_index_wo_qubits, index1,
                      begin(unsorted_operated_qubits), end(unsorted_operated_qubits),
                      begin(sorted_operated_qubits_with_sentinel), end(sorted_operated_qubits_with_sentinel)),
                    begin(unsorted_fused_qubits), end(unsorted_fused_qubits),
                    begin(sorted_fused_qubits_with_sentinel), end(sorted_fused_qubits_with_sentinel));
            auto const value0 = *iter0;

            using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
            using real_type = ::ket::utility::meta::real_t<complex_type>;
            using boost::math::constants::one_div_root_two;
            *iter0 -= *iter1;
            *iter0 *= one_div_root_two<real_type>();
            *iter1 += value0;
            *iter1 *= one_div_root_two<real_type>();
          },
          target_qubit, control_qubit1, control_qubit2, control_qubits...);
      }


      namespace runtime
      {
        namespace ranges
        {
          // C...C+Y_{tc...c'} or Cn+Y_{tc...c'}
          template <typename RandomAccessIterator, typename StateInteger, typename QubitsRange1, typename QubitsRange2, typename BitInteger, typename ControlQubitsRange>
          inline auto y_rotation_half_pi(
            RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
            QubitsRange1 const& unsorted_fused_qubits, QubitsRange2 const& sorted_fused_qubits_with_sentinel,
            ::ket::qubit<StateInteger, BitInteger> const target_qubit,
            ControlQubitsRange const& control_qubits)
          -> void
          {
            using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
            using control_qubit_type = ::ket::control<qubit_type>;
            static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
            static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");
            static_assert(std::is_same< ::ket::utility::meta::range_value_t<ControlQubitsRange>, control_qubit_type >::value, "The value_type of ControlQubitsRange should be the same as ket::qubit<S,B>");

            using std::begin;
            using std::end;
            auto const num_control_qubits = static_cast<BitInteger>(end(control_qubits) - begin(control_qubits));
            auto const num_fused_qubits = static_cast<BitInteger>(end(unsorted_fused_qubits) - begin(unsorted_fused_qubits));
            assert(static_cast<BitInteger>(end(sorted_fused_qubits_with_sentinel) - begin(sorted_fused_qubits_with_sentinel)) == num_fused_qubits + BitInteger{1u});
            assert(num_control_qubits + BitInteger{1u} <= num_fused_qubits);

            assert(target_qubit < qubit_type{num_fused_qubits});
            assert(::ket::utility::runtime::ranges::all_in_state_vector(num_fused_qubits, control_qubits));

            ::ket::gate::fused::runtime::ranges::gate(
              first, num_fused_qubits,
              [fused_index_wo_qubits, &unsorted_fused_qubits, &sorted_fused_qubits_with_sentinel, num_control_qubits](
                auto const first, StateInteger const operated_index_wo_qubits,
                auto const& unsorted_operated_qubits, auto const& sorted_operated_qubits_with_sentinel)
              {
                // 0b11...10u
                auto const index0 = ((std::size_t{1u} << num_control_qubits) - std::size_t{1u}) << BitInteger{1u};
                // 0b11...11u
                auto const index1 = index0 bitor std::size_t{1u};

                using std::begin;
                using std::end;
                auto const iter0
                  = first
                    + ::ket::gate::utility::ranges::index_with_qubits(
                        fused_index_wo_qubits,
                        ::ket::gate::utility::ranges::index_with_qubits(
                          operated_index_wo_qubits, index0,
                          unsorted_operated_qubits, sorted_operated_qubits_with_sentinel),
                        unsorted_fused_qubits, sorted_fused_qubits_with_sentinel);
                auto const iter1
                  = first
                    + ::ket::gate::utility::ranges::index_with_qubits(
                        fused_index_wo_qubits,
                        ::ket::gate::utility::ranges::index_with_qubits(
                          operated_index_wo_qubits, index1,
                          unsorted_operated_qubits, sorted_operated_qubits_with_sentinel),
                        unsorted_fused_qubits, sorted_fused_qubits_with_sentinel);
                auto const value0 = *iter0;

                using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
                using real_type = ::ket::utility::meta::real_t<complex_type>;
                using boost::math::constants::one_div_root_two;
                *iter0 += *iter1;
                *iter0 *= one_div_root_two<real_type>();
                *iter1 -= value0;
                *iter1 *= one_div_root_two<real_type>();
              },
              boost::join(
                boost::make_iterator_range(std::addressof(target_qubit), std::next(std::addressof(target_qubit))),
                control_qubits | boost::adaptors::transformed(
                  [](control_qubit_type const control_qubit) { return control_qubit.qubit(); })));
          }

          template <typename RandomAccessIterator, typename StateInteger, typename QubitsRange1, typename QubitsRange2, typename BitInteger>
          inline auto y_rotation_half_pi(
            RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
            QubitsRange1 const& unsorted_fused_qubits, QubitsRange2 const& sorted_fused_qubits_with_sentinel,
            ::ket::qubit<StateInteger, BitInteger> const target_qubit)
          -> void
          {
            using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
            using control_qubit_type = ::ket::control<qubit_type>;
            std::array<control_qubit_type, 0u> const control_qubits{};
            ::ket::gate::fused::runtime::ranges::y_rotation_half_pi(
              first, fused_index_wo_qubits,
              unsorted_fused_qubits, sorted_fused_qubits_with_sentinel,
              target_qubit, control_qubits);
          }

          // C...C-Y_{tc...c'} or Cn-Y_{tc...c'}
          template <typename RandomAccessIterator, typename StateInteger, typename QubitsRange1, typename QubitsRange2, typename BitInteger, typename ControlQubitsRange>
          inline auto adj_y_rotation_half_pi(
            RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
            QubitsRange1 const& unsorted_fused_qubits, QubitsRange2 const& sorted_fused_qubits_with_sentinel,
            ::ket::qubit<StateInteger, BitInteger> const target_qubit,
            ControlQubitsRange const& control_qubits)
          -> void
          {
            using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
            using control_qubit_type = ::ket::control<qubit_type>;
            static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
            static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");
            static_assert(std::is_same< ::ket::utility::meta::range_value_t<ControlQubitsRange>, control_qubit_type >::value, "The value_type of ControlQubitsRange should be the same as ket::qubit<S,B>");

            using std::begin;
            using std::end;
            auto const num_control_qubits = static_cast<BitInteger>(end(control_qubits) - begin(control_qubits));
            auto const num_fused_qubits = static_cast<BitInteger>(end(unsorted_fused_qubits) - begin(unsorted_fused_qubits));
            assert(static_cast<BitInteger>(end(sorted_fused_qubits_with_sentinel) - begin(sorted_fused_qubits_with_sentinel)) == num_fused_qubits + BitInteger{1u});
            assert(num_control_qubits + BitInteger{1u} <= num_fused_qubits);

            assert(target_qubit < qubit_type{num_fused_qubits});
            assert(::ket::utility::runtime::ranges::all_in_state_vector(num_fused_qubits, control_qubits));

            ::ket::gate::fused::runtime::ranges::gate(
              first, num_fused_qubits,
              [fused_index_wo_qubits, &unsorted_fused_qubits, &sorted_fused_qubits_with_sentinel, num_control_qubits](
                auto const first, StateInteger const operated_index_wo_qubits,
                auto const& unsorted_operated_qubits, auto const& sorted_operated_qubits_with_sentinel)
              {
                // 0b11...10u
                auto const index0 = ((std::size_t{1u} << num_control_qubits) - std::size_t{1u}) << BitInteger{1u};
                // 0b11...11u
                auto const index1 = index0 bitor std::size_t{1u};

                auto const iter0
                  = first
                    + ::ket::gate::utility::ranges::index_with_qubits(
                        fused_index_wo_qubits,
                        ::ket::gate::utility::ranges::index_with_qubits(
                          operated_index_wo_qubits, index0,
                          unsorted_operated_qubits, sorted_operated_qubits_with_sentinel),
                        unsorted_fused_qubits, sorted_fused_qubits_with_sentinel);
                auto const iter1
                  = first
                    + ::ket::gate::utility::ranges::index_with_qubits(
                        fused_index_wo_qubits,
                        ::ket::gate::utility::ranges::index_with_qubits(
                          operated_index_wo_qubits, index1,
                          unsorted_operated_qubits, sorted_operated_qubits_with_sentinel),
                        unsorted_fused_qubits, sorted_fused_qubits_with_sentinel);
                auto const value0 = *iter0;

                using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
                using real_type = ::ket::utility::meta::real_t<complex_type>;
                using boost::math::constants::one_div_root_two;
                *iter0 -= *iter1;
                *iter0 *= one_div_root_two<real_type>();
                *iter1 += value0;
                *iter1 *= one_div_root_two<real_type>();
              },
              boost::join(
                boost::make_iterator_range(std::addressof(target_qubit), std::next(std::addressof(target_qubit))),
                control_qubits | boost::adaptors::transformed(
                  [](control_qubit_type const control_qubit) { return control_qubit.qubit(); })));
          }

          template <typename RandomAccessIterator, typename StateInteger, typename QubitsRange1, typename QubitsRange2, typename BitInteger>
          inline auto adj_y_rotation_half_pi(
            RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
            QubitsRange1 const& unsorted_fused_qubits, QubitsRange2 const& sorted_fused_qubits_with_sentinel,
            ::ket::qubit<StateInteger, BitInteger> const target_qubit)
          -> void
          {
            using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
            using control_qubit_type = ::ket::control<qubit_type>;
            std::array<control_qubit_type, 0u> const control_qubits{};
            ::ket::gate::fused::runtime::ranges::adj_y_rotation_half_pi(
              first, fused_index_wo_qubits,
              unsorted_fused_qubits, sorted_fused_qubits_with_sentinel,
              target_qubit, control_qubits);
          }
        } // namespace ranges

        template <typename RandomAccessIterator, typename StateInteger, typename QubitIterator1, typename QubitIterator2, typename BitInteger, typename ControlQubitIterator>
        inline auto y_rotation_half_pi(
          RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
          QubitIterator1 const unsorted_fused_qubit_first, QubitIterator1 const unsorted_fused_qubit_last,
          QubitIterator2 const sorted_fused_qubit_with_sentinel_first, QubitIterator2 const sorted_fused_qubit_with_sentinel_last,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ControlQubitIterator const control_qubit_first, ControlQubitIterator const control_qubit_last)
        -> void
        {
          ::ket::gate::fused::runtime::ranges::y_rotation_half_pi(
            first, fused_index_wo_qubits,
            boost::make_iterator_range(unsorted_fused_qubit_first, unsorted_fused_qubit_last),
            boost::make_iterator_range(sorted_fused_qubit_with_sentinel_first, sorted_fused_qubit_with_sentinel_last),
            target_qubit, boost::make_iterator_range(control_qubit_first, control_qubit_last));
        }

        template <typename RandomAccessIterator, typename StateInteger, typename QubitIterator1, typename QubitIterator2, typename BitInteger>
        inline auto y_rotation_half_pi(
          RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
          QubitIterator1 const unsorted_fused_qubit_first, QubitIterator1 const unsorted_fused_qubit_last,
          QubitIterator2 const sorted_fused_qubit_with_sentinel_first, QubitIterator2 const sorted_fused_qubit_with_sentinel_last,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit)
        -> void
        {
          ::ket::gate::fused::runtime::ranges::y_rotation_half_pi(
            first, fused_index_wo_qubits,
            boost::make_iterator_range(unsorted_fused_qubit_first, unsorted_fused_qubit_last),
            boost::make_iterator_range(sorted_fused_qubit_with_sentinel_first, sorted_fused_qubit_with_sentinel_last),
            target_qubit);
        }

        template <typename RandomAccessIterator, typename StateInteger, typename QubitIterator1, typename QubitIterator2, typename BitInteger, typename ControlQubitIterator>
        inline auto adj_y_rotation_half_pi(
          RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
          QubitIterator1 const unsorted_fused_qubit_first, QubitIterator1 const unsorted_fused_qubit_last,
          QubitIterator2 const sorted_fused_qubit_with_sentinel_first, QubitIterator2 const sorted_fused_qubit_with_sentinel_last,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ControlQubitIterator const control_qubit_first, ControlQubitIterator const control_qubit_last)
        -> void
        {
          ::ket::gate::fused::runtime::ranges::adj_y_rotation_half_pi(
            first, fused_index_wo_qubits,
            boost::make_iterator_range(unsorted_fused_qubit_first, unsorted_fused_qubit_last),
            boost::make_iterator_range(sorted_fused_qubit_with_sentinel_first, sorted_fused_qubit_with_sentinel_last),
            target_qubit, boost::make_iterator_range(control_qubit_first, control_qubit_last));
        }

        template <typename RandomAccessIterator, typename StateInteger, typename QubitIterator1, typename QubitIterator2, typename BitInteger>
        inline auto adj_y_rotation_half_pi(
          RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
          QubitIterator1 const unsorted_fused_qubit_first, QubitIterator1 const unsorted_fused_qubit_last,
          QubitIterator2 const sorted_fused_qubit_with_sentinel_first, QubitIterator2 const sorted_fused_qubit_with_sentinel_last,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit)
        -> void
        {
          ::ket::gate::fused::runtime::ranges::adj_y_rotation_half_pi(
            first, fused_index_wo_qubits,
            boost::make_iterator_range(unsorted_fused_qubit_first, unsorted_fused_qubit_last),
            boost::make_iterator_range(sorted_fused_qubit_with_sentinel_first, sorted_fused_qubit_with_sentinel_last),
            target_qubit);
        }
      } // namespace runtime
# else // KET_USE_BIT_MASKS_EXPLICITLY
      // +Y_i
      // +Y_1 (a_0 |0> + a_1 |1>) = (a_0 + a_1)/sqrt(2) |0> + (-a_0 + a_1)/sqrt(2) |1>
      template <typename RandomAccessIterator, typename StateInteger, std::size_t num_fused_qubits, typename BitInteger>
      inline auto y_rotation_half_pi(
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
          auto const zero_iter_value = *zero_iter;

          using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
          using real_type = ::ket::utility::meta::real_t<complex_type>;
          using boost::math::constants::one_div_root_two;
          *zero_iter += *one_iter;
          *zero_iter *= one_div_root_two<real_type>();
          *one_iter -= zero_iter_value;
          *one_iter *= one_div_root_two<real_type>();
        }
      }

      // C+Y_{tc} or C1+Y_{tc}
      // C+Y_{1,2} (a_{00} |00> + a_{01} |01> + a_{10} |10> + a_{11} |11>)
      //   = a_{00} |00> + a_{01} |01> + (a_{10} + a_{11})/sqrt(2) |10> + (-a_{10} + a_{11})/sqrt(2) |11>
      template <typename RandomAccessIterator, typename StateInteger, std::size_t num_fused_qubits, typename BitInteger>
      inline auto y_rotation_half_pi(
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
          auto const control_on_iter_value = *control_on_iter;

          using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
          using real_type = ::ket::utility::meta::real_t<complex_type>;
          using boost::math::constants::one_div_root_two;
          *control_on_iter += *target_control_on_iter;
          *control_on_iter *= one_div_root_two<real_type>();
          *target_control_on_iter -= control_on_iter_value;
          *target_control_on_iter *= one_div_root_two<real_type>();
        }
      }

      // C...C+Y_{tc...c'} or Cn+Y_{tc...c'}
      template <typename RandomAccessIterator, typename StateInteger, std::size_t num_fused_qubits, typename BitInteger, typename... ControlQubits>
      inline auto y_rotation_half_pi(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array<StateInteger, num_fused_qubits> const& fused_qubit_masks, std::array<StateInteger, num_fused_qubits + 1u> const& fused_index_masks,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2, ControlQubits const... control_qubits)
      -> void
      {
        constexpr auto num_control_qubits = static_cast<BitInteger>(sizeof...(ControlQubits) + 2u);
        constexpr auto num_operated_qubits = num_control_qubits + BitInteger{1u};
        ::ket::gate::fused::gate<num_fused_qubits>(
          first,
          [fused_index_wo_qubits, &fused_qubit_masks, &fused_index_masks](
            auto const first, StateInteger const operated_index_wo_qubits,
            std::array<StateInteger, num_operated_qubits> const& operated_qubit_masks,
            std::array<StateInteger, num_operated_qubits + 1u> const& operated_index_masks)
          {
            // 0b11...10u
            constexpr auto index0 = ((std::size_t{1u} << num_control_qubits) - std::size_t{1u}) << BitInteger{1u};
            // 0b11...11u
            constexpr auto index1 = index0 bitor std::size_t{1u};

            using std::begin;
            using std::end;
            auto const iter0
              = first
                + ::ket::gate::utility::index_with_qubits(
                    fused_index_wo_qubits,
                    ::ket::gate::utility::index_with_qubits(
                      operated_index_wo_qubits, index0,
                      begin(operated_qubit_masks), end(operated_qubit_masks), begin(operated_index_masks), end(operated_index_masks)),
                    begin(fused_qubit_masks), end(fused_qubit_masks), begin(fused_index_masks), end(fused_index_masks));
            auto const iter1
              = first
                + ::ket::gate::utility::index_with_qubits(
                    fused_index_wo_qubits,
                    ::ket::gate::utility::index_with_qubits(
                      operated_index_wo_qubits, index1,
                      begin(operated_qubit_masks), end(operated_qubit_masks), begin(operated_index_masks), end(operated_index_masks)),
                    begin(fused_qubit_masks), end(fused_qubit_masks), begin(fused_index_masks), end(fused_index_masks));
            auto const value0 = *iter0;

            using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
            using real_type = ::ket::utility::meta::real_t<complex_type>;
            using boost::math::constants::one_div_root_two;
            *iter0 += *iter1;
            *iter0 *= one_div_root_two<real_type>();
            *iter1 -= value0;
            *iter1 *= one_div_root_two<real_type>();
          },
          target_qubit, control_qubit1, control_qubit2, control_qubits...);
      }

      // -Y_i
      // -Y_1 (a_0 |0> + a_1 |1>) = (a_0 - a_1)/sqrt(2) |0> + (a_0 + a_1)/sqrt(2) |1>
      template <typename RandomAccessIterator, typename StateInteger, std::size_t num_fused_qubits, typename BitInteger>
      inline auto adj_y_rotation_half_pi(
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
          auto const zero_iter_value = *zero_iter;

          using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
          using real_type = ::ket::utility::meta::real_t<complex_type>;
          using boost::math::constants::one_div_root_two;
          *zero_iter -= *one_iter;
          *zero_iter *= one_div_root_two<real_type>();
          *one_iter += zero_iter_value;
          *one_iter *= one_div_root_two<real_type>();
        }
      }

      // C-Y_{tc} or C1-Y_{tc}
      // C-Y_{1,2} (a_{00} |00> + a_{01} |01> + a_{10} |10> + a_{11} |11>)
      //   = a_{00} |00> + a_{01} |01> + (a_{10} - a_{11})/sqrt(2) |10> + (a_{10} + a_{11})/sqrt(2) |11>
      template <typename RandomAccessIterator, typename StateInteger, std::size_t num_fused_qubits, typename BitInteger>
      inline auto adj_y_rotation_half_pi(
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
          auto const control_on_iter_value = *control_on_iter;

          using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
          using real_type = ::ket::utility::meta::real_t<complex_type>;
          using boost::math::constants::one_div_root_two;
          *control_on_iter -= *target_control_on_iter;
          *control_on_iter *= one_div_root_two<real_type>();
          *target_control_on_iter += control_on_iter_value;
          *target_control_on_iter *= one_div_root_two<real_type>();
        }
      }

      // C...C-Y_{tc...c'} or Cn-Y_{tc...c'}
      template <typename RandomAccessIterator, typename StateInteger, std::size_t num_fused_qubits, typename BitInteger, typename... ControlQubits>
      inline auto adj_y_rotation_half_pi(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array<StateInteger, num_fused_qubits> const& fused_qubit_masks, std::array<StateInteger, num_fused_qubits + 1u> const& fused_index_masks,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2, ControlQubits const... control_qubits)
      -> void
      {
        constexpr auto num_control_qubits = static_cast<BitInteger>(sizeof...(ControlQubits) + 2u);
        constexpr auto num_operated_qubits = num_control_qubits + BitInteger{1u};
        ::ket::gate::fused::gate<num_fused_qubits>(
          first,
          [fused_index_wo_qubits, &fused_qubit_masks, &fused_index_masks](
            auto const first, StateInteger const operated_index_wo_qubits,
            std::array<StateInteger, num_operated_qubits> const& operated_qubit_masks,
            std::array<StateInteger, num_operated_qubits + 1u> const& operated_index_masks)
          {
            // 0b11...10u
            constexpr auto index0 = ((std::size_t{1u} << num_control_qubits) - std::size_t{1u}) << BitInteger{1u};
            // 0b11...11u
            constexpr auto index1 = index0 bitor std::size_t{1u};

            using std::begin;
            using std::end;
            auto const iter0
              = first
                + ::ket::gate::utility::index_with_qubits(
                    fused_index_wo_qubits,
                    ::ket::gate::utility::index_with_qubits(
                      operated_index_wo_qubits, index0,
                      begin(operated_qubit_masks), end(operated_qubit_masks), begin(operated_index_masks), end(operated_index_masks)),
                    begin(fused_qubit_masks), end(fused_qubit_masks), begin(fused_index_masks), end(fused_index_masks));
            auto const iter1
              = first
                + ::ket::gate::utility::index_with_qubits(
                    fused_index_wo_qubits,
                    ::ket::gate::utility::index_with_qubits(
                      operated_index_wo_qubits, index1,
                      begin(operated_qubit_masks), end(operated_qubit_masks), begin(operated_index_masks), end(operated_index_masks)),
                    begin(fused_qubit_masks), end(fused_qubit_masks), begin(fused_index_masks), end(fused_index_masks));
            auto const value0 = *iter0;

            using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
            using real_type = ::ket::utility::meta::real_t<complex_type>;
            using boost::math::constants::one_div_root_two;
            *iter0 -= *iter1;
            *iter0 *= one_div_root_two<real_type>();
            *iter1 += value0;
            *iter1 *= one_div_root_two<real_type>();
          },
          target_qubit, control_qubit1, control_qubit2, control_qubits...);
      }


      namespace runtime
      {
        namespace ranges
        {
          // C...C+Y_{tc...c'} or Cn+Y_{tc...c'}
          template <typename RandomAccessIterator, typename StateInteger, typename StateIntegersRange1, typename StateIntegersRange2, typename BitInteger, typename ControlQubitsRange>
          inline auto y_rotation_half_pi(
            RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
            StateIntegersRange1 const& fused_qubit_masks, StateIntegersRange2 const& fused_index_masks,
            ::ket::qubit<StateInteger, BitInteger> const target_qubit,
            ControlQubitsRange const& control_qubits)
          -> void
          {
            using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
            using control_qubit_type = ::ket::control<qubit_type>;
            static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
            static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");
            static_assert(std::is_same< ::ket::utility::meta::range_value_t<ControlQubitsRange>, control_qubit_type >::value, "The value_type of ControlQubitsRange should be the same as ket::qubit<S,B>");

            using std::begin;
            using std::end;
            auto const num_control_qubits = static_cast<BitInteger>(end(control_qubits) - begin(control_qubits));
            auto const num_fused_qubits = static_cast<BitInteger>(end(fused_qubit_masks) - begin(fused_qubit_masks));
            assert(static_cast<BitInteger>(end(fused_index_masks) - begin(fused_index_masks)) == num_fused_qubits + BitInteger{1u});
            assert(num_control_qubits + BitInteger{1u} <= num_fused_qubits);

            assert(target_qubit < qubit_type{num_fused_qubits});
            assert(::ket::utility::runtime::ranges::all_in_state_vector(num_fused_qubits, control_qubits));

            ::ket::gate::fused::runtime::ranges::gate(
              first, num_fused_qubits,
              [fused_index_wo_qubits, &fused_qubit_masks, &fused_index_masks, num_control_qubits](
                auto const first, StateInteger const operated_index_wo_qubits,
                auto const& operated_qubit_masks, auto const& operated_index_masks)
              {
                // 0b11...10u
                auto const index0 = ((std::size_t{1u} << num_control_qubits) - std::size_t{1u}) << BitInteger{1u};
                // 0b11...11u
                auto const index1 = index0 bitor std::size_t{1u};

                auto const iter0
                  = first
                    + ::ket::gate::utility::ranges::index_with_qubits(
                        fused_index_wo_qubits,
                        ::ket::gate::utility::ranges::index_with_qubits(
                          operated_index_wo_qubits, index0,
                          operated_qubit_masks, operated_index_masks),
                        fused_qubit_masks, fused_index_masks);
                auto const iter1
                  = first
                    + ::ket::gate::utility::ranges::index_with_qubits(
                        fused_index_wo_qubits,
                        ::ket::gate::utility::ranges::index_with_qubits(
                          operated_index_wo_qubits, index1,
                          operated_qubit_masks, operated_index_masks),
                        fused_qubit_masks, fused_index_masks);
                auto const value0 = *iter0;

                using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
                using real_type = ::ket::utility::meta::real_t<complex_type>;
                using boost::math::constants::one_div_root_two;
                *iter0 += *iter1;
                *iter0 *= one_div_root_two<real_type>();
                *iter1 -= value0;
                *iter1 *= one_div_root_two<real_type>();
              },
              boost::join(
                boost::make_iterator_range(std::addressof(target_qubit), std::next(std::addressof(target_qubit))),
                control_qubits | boost::adaptors::transformed(
                  [](control_qubit_type const control_qubit) { return control_qubit.qubit(); })));
          }

          template <typename RandomAccessIterator, typename StateInteger, typename StateIntegersRange1, typename StateIntegersRange2, typename BitInteger>
          inline auto y_rotation_half_pi(
            RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
            StateIntegersRange1 const& fused_qubit_masks, StateIntegersRange2 const& fused_index_masks,
            ::ket::qubit<StateInteger, BitInteger> const target_qubit)
          -> void
          {
            using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
            using control_qubit_type = ::ket::control<qubit_type>;
            std::array<control_qubit_type, 0u> const control_qubits{};
            ::ket::gate::fused::runtime::ranges::y_rotation_half_pi(
              first, fused_index_wo_qubits,
              fused_qubit_masks, fused_index_masks,
              target_qubit, control_qubits);
          }

          // C...C-Y_{tc...c'} or Cn-Y_{tc...c'}
          template <typename RandomAccessIterator, typename StateInteger, typename StateIntegersRange1, typename StateIntegersRange2, typename BitInteger, typename ControlQubitsRange>
          inline auto adj_y_rotation_half_pi(
            RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
            StateIntegersRange1 const& fused_qubit_masks, StateIntegersRange2 const& fused_index_masks,
            ::ket::qubit<StateInteger, BitInteger> const target_qubit,
            ControlQubitsRange const& control_qubits)
          -> void
          {
            using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
            using control_qubit_type = ::ket::control<qubit_type>;
            static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
            static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");
            static_assert(std::is_same< ::ket::utility::meta::range_value_t<ControlQubitsRange>, control_qubit_type >::value, "The value_type of ControlQubitsRange should be the same as ket::qubit<S,B>");

            using std::begin;
            using std::end;
            auto const num_control_qubits = static_cast<BitInteger>(end(control_qubits) - begin(control_qubits));
            auto const num_fused_qubits = static_cast<BitInteger>(end(fused_qubit_masks) - begin(fused_qubit_masks));
            assert(static_cast<BitInteger>(end(fused_index_masks) - begin(fused_index_masks)) == num_fused_qubits + BitInteger{1u});
            assert(num_control_qubits + BitInteger{1u} <= num_fused_qubits);

            assert(target_qubit < qubit_type{num_fused_qubits});
            assert(::ket::utility::runtime::ranges::all_in_state_vector(num_fused_qubits, control_qubits));

            ::ket::gate::fused::runtime::ranges::gate(
              first, num_fused_qubits,
              [fused_index_wo_qubits, &fused_qubit_masks, &fused_index_masks, num_control_qubits](
                auto const first, StateInteger const operated_index_wo_qubits,
                auto const& operated_qubit_masks, auto const& operated_index_masks)
              {
                // 0b11...10u
                auto const index0 = ((std::size_t{1u} << num_control_qubits) - std::size_t{1u}) << BitInteger{1u};
                // 0b11...11u
                auto const index1 = index0 bitor std::size_t{1u};

                auto const iter0
                  = first
                    + ::ket::gate::utility::ranges::index_with_qubits(
                        fused_index_wo_qubits,
                        ::ket::gate::utility::ranges::index_with_qubits(
                          operated_index_wo_qubits, index0,
                          operated_qubit_masks, operated_index_masks),
                        fused_qubit_masks, fused_index_masks);
                auto const iter1
                  = first
                    + ::ket::gate::utility::ranges::index_with_qubits(
                        fused_index_wo_qubits,
                        ::ket::gate::utility::ranges::index_with_qubits(
                          operated_index_wo_qubits, index1,
                          operated_qubit_masks, operated_index_masks),
                        fused_qubit_masks, fused_index_masks);
                auto const value0 = *iter0;

                using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
                using real_type = ::ket::utility::meta::real_t<complex_type>;
                using boost::math::constants::one_div_root_two;
                *iter0 -= *iter1;
                *iter0 *= one_div_root_two<real_type>();
                *iter1 += value0;
                *iter1 *= one_div_root_two<real_type>();
              },
              boost::join(
                boost::make_iterator_range(std::addressof(target_qubit), std::next(std::addressof(target_qubit))),
                control_qubits | boost::adaptors::transformed(
                  [](control_qubit_type const control_qubit) { return control_qubit.qubit(); })));
          }

          template <typename RandomAccessIterator, typename StateInteger, typename StateIntegersRange1, typename StateIntegersRange2, typename BitInteger>
          inline auto adj_y_rotation_half_pi(
            RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
            StateIntegersRange1 const& fused_qubit_masks, StateIntegersRange2 const& fused_index_masks,
            ::ket::qubit<StateInteger, BitInteger> const target_qubit)
          -> void
          {
            using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
            using control_qubit_type = ::ket::control<qubit_type>;
            std::array<control_qubit_type, 0u> const control_qubits{};
            ::ket::gate::fused::runtime::ranges::adj_y_rotation_half_pi(
              first, fused_index_wo_qubits,
              fused_qubit_masks, fused_index_masks,
              target_qubit, control_qubits);
          }
        } // namespace ranges

        template <typename RandomAccessIterator, typename StateInteger, typename StateIntegerIterator1, typename StateIntegerIterator2, typename BitInteger, typename ControlQubitIterator>
        inline auto y_rotation_half_pi(
          RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
          StateIntegerIterator1 const fused_qubit_mask_first, StateIntegerIterator1 const fused_qubit_mask_last,
          StateIntegerIterator2 const fused_index_mask_first, StateIntegerIterator2 const fused_index_mask_last,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ControlQubitIterator const control_qubit_first, ControlQubitIterator const control_qubit_last)
        -> void
        {
          ::ket::gate::fused::runtime::ranges::y_rotation_half_pi(
            first, fused_index_wo_qubits,
            boost::make_iterator_range(fused_qubit_mask_first, fused_qubit_mask_last),
            boost::make_iterator_range(fused_index_mask_first, fused_index_mask_last),
            target_qubit, boost::make_iterator_range(control_qubit_first, control_qubit_last));
        }

        template <typename RandomAccessIterator, typename StateInteger, typename StateIntegerIterator1, typename StateIntegerIterator2, typename BitInteger>
        inline auto y_rotation_half_pi(
          RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
          StateIntegerIterator1 const fused_qubit_mask_first, StateIntegerIterator1 const fused_qubit_mask_last,
          StateIntegerIterator2 const fused_index_mask_first, StateIntegerIterator2 const fused_index_mask_last,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit)
        -> void
        {
          ::ket::gate::fused::runtime::ranges::y_rotation_half_pi(
            first, fused_index_wo_qubits,
            boost::make_iterator_range(fused_qubit_mask_first, fused_qubit_mask_last),
            boost::make_iterator_range(fused_index_mask_first, fused_index_mask_last),
            target_qubit);
        }

        template <typename RandomAccessIterator, typename StateInteger, typename StateIntegerIterator1, typename StateIntegerIterator2, typename BitInteger, typename ControlQubitIterator>
        inline auto adj_y_rotation_half_pi(
          RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
          StateIntegerIterator1 const fused_qubit_mask_first, StateIntegerIterator1 const fused_qubit_mask_last,
          StateIntegerIterator2 const fused_index_mask_first, StateIntegerIterator2 const fused_index_mask_last,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ControlQubitIterator const control_qubit_first, ControlQubitIterator const control_qubit_last)
        -> void
        {
          ::ket::gate::fused::runtime::ranges::adj_y_rotation_half_pi(
            first, fused_index_wo_qubits,
            boost::make_iterator_range(fused_qubit_mask_first, fused_qubit_mask_last),
            boost::make_iterator_range(fused_index_mask_first, fused_index_mask_last),
            target_qubit, boost::make_iterator_range(control_qubit_first, control_qubit_last));
        }

        template <typename RandomAccessIterator, typename StateInteger, typename StateIntegerIterator1, typename StateIntegerIterator2, typename BitInteger>
        inline auto adj_y_rotation_half_pi(
          RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
          StateIntegerIterator1 const fused_qubit_mask_first, StateIntegerIterator1 const fused_qubit_mask_last,
          StateIntegerIterator2 const fused_index_mask_first, StateIntegerIterator2 const fused_index_mask_last,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit)
        -> void
        {
          ::ket::gate::fused::runtime::ranges::adj_y_rotation_half_pi(
            first, fused_index_wo_qubits,
            boost::make_iterator_range(fused_qubit_mask_first, fused_qubit_mask_last),
            boost::make_iterator_range(fused_index_mask_first, fused_index_mask_last),
            target_qubit);
        }
      } // namespace runtime
# endif // KET_USE_BIT_MASKS_EXPLICITLY
    } // namespace fused
  } // namespace gate
} // namespace ket


#endif // KET_GATE_FUSED_Y_ROTATION_HALF_PI_HPP
