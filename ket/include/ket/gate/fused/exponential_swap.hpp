#ifndef KET_GATE_FUSED_EXPONENTIAL_SWAP_HPP
# define KET_GATE_FUSED_EXPONENTIAL_SWAP_HPP

# include <cassert>
# include <cstddef>
# include <cmath>
# include <complex>
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
# include <ket/utility/integer_exp2.hpp>
# include <ket/utility/integer_log2.hpp>
# include <ket/utility/imaginary_unit.hpp>
# include <ket/utility/exp_i.hpp>


namespace ket
{
  namespace gate
  {
    namespace fused
    {
# ifndef KET_USE_BIT_MASKS_EXPLICITLY
      // exponential_swap_coeff
      // eSWAP_{ij}(s) = exp(is SWAP_{ij}) = I cos s + i SWAP_{ij} sin s
      // eSWAP_{1,2}(s) (a_{00} |00> + a_{01} |01> + a_{10} |10> + a_{11} |11>)
      //   = e^{is} a_{00} |00> + (cos s a_{01} + i sin s a_{10}) |01> + (i sin s a_{01} + cos s a_{10}) |10> + e^{is} a_{11} |11>
      template <typename RandomAccessIterator, typename StateInteger, typename BitInteger, std::size_t num_fused_qubits, typename Complex>
      inline auto exponential_swap_coeff(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_fused_qubits > const& unsorted_fused_qubits,
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_fused_qubits + 1u> const& sorted_fused_qubits_with_sentinel,
        Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
        ::ket::qubit<StateInteger, BitInteger> const qubit1, ::ket::qubit<StateInteger, BitInteger> const qubit2)
      -> void
      {
        static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
        static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");
        static_assert(
          std::is_same<Complex, typename std::iterator_traits<RandomAccessIterator>::value_type>::value,
          "Complex should be the same to value_type of RandomAccessIterator");
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

        using std::imag;
        auto const i_sin_theta = ::ket::utility::imaginary_unit<Complex>() * imag(phase_coefficient);

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

          *off_iter *= phase_coefficient;
          *qubit12_on_iter *= phase_coefficient;

          auto const qubit1_on_iter_value = *qubit1_on_iter;
          using std::real;
          using std::imag;
          *qubit1_on_iter *= real(phase_coefficient);
          *qubit1_on_iter += *qubit2_on_iter * i_sin_theta;
          *qubit2_on_iter *= real(phase_coefficient);
          *qubit2_on_iter += qubit1_on_iter_value * i_sin_theta;
        }
      }

      // C...CeSWAP_{tt'c...c'}(s) = C...C[exp(is SWAP_{tt'})]_{c...c'} = C...C[I cos s + i SWAP_{tt'} sin s]_{c...c'}
      template <typename RandomAccessIterator, typename StateInteger, typename BitInteger, std::size_t num_fused_qubits, typename Complex, typename... ControlQubits>
      inline auto exponential_swap_coeff(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_fused_qubits > const& unsorted_fused_qubits,
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_fused_qubits + 1u> const& sorted_fused_qubits_with_sentinel,
        Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
        ::ket::qubit<StateInteger, BitInteger> const target_qubit1, ::ket::qubit<StateInteger, BitInteger> const target_qubit2,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit, ControlQubits const... control_qubits)
      -> void
      {
        static_assert(
          std::is_same<Complex, typename std::iterator_traits<RandomAccessIterator>::value_type>::value,
          "Complex should be the same to value_type of RandomAccessIterator");

        using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
        constexpr auto num_control_qubits = static_cast<BitInteger>(sizeof...(ControlQubits) + 1u);
        constexpr auto num_operated_qubits = num_control_qubits + BitInteger{2u};

        using std::imag;
        auto const i_sin_theta = ::ket::utility::imaginary_unit<Complex>() * imag(phase_coefficient);

        ::ket::gate::fused::gate<num_fused_qubits>(
          first,
          [fused_index_wo_qubits, &unsorted_fused_qubits, &sorted_fused_qubits_with_sentinel, &phase_coefficient, &i_sin_theta](
            auto const first, StateInteger const operated_index_wo_qubits,
            std::array<qubit_type, num_operated_qubits> const& unsorted_operated_qubits,
            std::array<qubit_type, num_operated_qubits + 1u> const& sorted_operated_qubits_with_sentinel)
          {
            // 0b11...100u
            constexpr auto index00 = ((std::size_t{1u} << num_control_qubits) - std::size_t{1u}) << BitInteger{2u};
            using std::begin;
            using std::end;
            auto const iter00
              = first
                + ::ket::gate::utility::index_with_qubits(
                    fused_index_wo_qubits,
                    ::ket::gate::utility::index_with_qubits(
                      operated_index_wo_qubits, index00,
                      begin(unsorted_operated_qubits), end(unsorted_operated_qubits),
                      begin(sorted_operated_qubits_with_sentinel), end(sorted_operated_qubits_with_sentinel)),
                    begin(unsorted_fused_qubits), end(unsorted_fused_qubits),
                    begin(sorted_fused_qubits_with_sentinel), end(sorted_fused_qubits_with_sentinel));
            // 0b11...101u
            auto const iter01
              = first
                + ::ket::gate::utility::index_with_qubits(
                    fused_index_wo_qubits,
                    ::ket::gate::utility::index_with_qubits(
                      operated_index_wo_qubits, index00 bitor std::size_t{1u},
                      begin(unsorted_operated_qubits), end(unsorted_operated_qubits),
                      begin(sorted_operated_qubits_with_sentinel), end(sorted_operated_qubits_with_sentinel)),
                    begin(unsorted_fused_qubits), end(unsorted_fused_qubits),
                    begin(sorted_fused_qubits_with_sentinel), end(sorted_fused_qubits_with_sentinel));
            // 0b11...110u
            constexpr auto index10 = index00 bitor (std::size_t{1u} << BitInteger{1u});
            auto const iter10
              = first
                + ::ket::gate::utility::index_with_qubits(
                    fused_index_wo_qubits,
                    ::ket::gate::utility::index_with_qubits(
                      operated_index_wo_qubits, index10,
                      begin(unsorted_operated_qubits), end(unsorted_operated_qubits),
                      begin(sorted_operated_qubits_with_sentinel), end(sorted_operated_qubits_with_sentinel)),
                    begin(unsorted_fused_qubits), end(unsorted_fused_qubits),
                    begin(sorted_fused_qubits_with_sentinel), end(sorted_fused_qubits_with_sentinel));
            // 0b11...111u
            auto const iter11
              = first
                + ::ket::gate::utility::index_with_qubits(
                    fused_index_wo_qubits,
                    ::ket::gate::utility::index_with_qubits(
                      operated_index_wo_qubits, index10 bitor std::size_t{1u},
                      begin(unsorted_operated_qubits), end(unsorted_operated_qubits),
                      begin(sorted_operated_qubits_with_sentinel), end(sorted_operated_qubits_with_sentinel)),
                    begin(unsorted_fused_qubits), end(unsorted_fused_qubits),
                    begin(sorted_fused_qubits_with_sentinel), end(sorted_fused_qubits_with_sentinel));

            *iter00 *= phase_coefficient;
            *iter11 *= phase_coefficient;

            auto const value01 = *iter01;
            using std::real;
            using std::imag;
            *iter01 *= real(phase_coefficient);
            *iter01 += *iter10 * i_sin_theta;
            *iter10 *= real(phase_coefficient);
            *iter10 += value01 * i_sin_theta;
          },
          target_qubit1, target_qubit2, control_qubit, control_qubits...);
      }

      template <typename RandomAccessIterator, typename StateInteger, typename BitInteger, std::size_t num_fused_qubits, typename Complex, typename... ControlQubits>
      inline auto adj_exponential_swap_coeff(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_fused_qubits > const& unsorted_fused_qubits,
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_fused_qubits + 1u> const& sorted_fused_qubits_with_sentinel,
        Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
        ::ket::qubit<StateInteger, BitInteger> const target_qubit1, ::ket::qubit<StateInteger, BitInteger> const target_qubit2,
        ControlQubits const... control_qubits)
      -> void
      { using std::conj; ::ket::gate::fused::exponential_swap_coeff(first, fused_index_wo_qubits, unsorted_fused_qubits, sorted_fused_qubits_with_sentinel, conj(phase_coefficient), target_qubit1, target_qubit2, control_qubits...); }

      // exponential_swap
      template <typename RandomAccessIterator, typename StateInteger, typename BitInteger, std::size_t num_fused_qubits, typename Real, typename... ControlQubits>
      inline auto exponential_swap(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_fused_qubits > const& unsorted_fused_qubits,
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_fused_qubits + 1u> const& sorted_fused_qubits_with_sentinel,
        Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit1, ::ket::qubit<StateInteger, BitInteger> const target_qubit2,
        ControlQubits const... control_qubits)
      -> void
      {
        using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
        ::ket::gate::fused::exponential_swap_coeff(first, fused_index_wo_qubits, unsorted_fused_qubits, sorted_fused_qubits_with_sentinel, ::ket::utility::exp_i<complex_type>(phase), target_qubit1, target_qubit2, control_qubits...);
      }

      template <typename RandomAccessIterator, typename StateInteger, typename BitInteger, std::size_t num_fused_qubits, typename Real, typename... ControlQubits>
      inline auto adj_exponential_swap(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_fused_qubits > const& unsorted_fused_qubits,
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_fused_qubits + 1u> const& sorted_fused_qubits_with_sentinel,
        Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit1, ::ket::qubit<StateInteger, BitInteger> const target_qubit2,
        ControlQubits const... control_qubits)
      -> void
      { ::ket::gate::fused::exponential_swap(first, fused_index_wo_qubits, unsorted_fused_qubits, sorted_fused_qubits_with_sentinel, -phase, target_qubit1, target_qubit2, control_qubits...); }


      namespace runtime
      {
        // exponential_swap_coeff
        namespace ranges
        {
          // C...CeSWAP_{tt'c...c'}(s) = C...C[exp(is SWAP_{tt'})]_{c...c'} = C...C[I cos s + i SWAP_{tt'} sin s]_{c...c'}
          template <typename RandomAccessIterator, typename StateInteger, typename QubitsRange1, typename QubitsRange2, typename Complex, typename BitInteger, typename ControlQubitsRange>
          inline auto exponential_swap_coeff(
            RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
            QubitsRange1 const& unsorted_fused_qubits, QubitsRange2 const& sorted_fused_qubits_with_sentinel,
            Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
            ::ket::qubit<StateInteger, BitInteger> const target_qubit1, ::ket::qubit<StateInteger, BitInteger> const target_qubit2,
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
            assert(num_control_qubits + BitInteger{2u} <= num_fused_qubits);

            assert(target_qubit1 < qubit_type{num_fused_qubits});
            assert(target_qubit2 < qubit_type{num_fused_qubits});
            assert(::ket::utility::runtime::ranges::all_in_state_vector(num_fused_qubits, control_qubits));

            static_assert(
              std::is_same<Complex, typename std::iterator_traits<RandomAccessIterator>::value_type>::value,
              "Complex should be the same to value_type of RandomAccessRange");

            using std::imag;
            auto const i_sin_theta = ::ket::utility::imaginary_unit<Complex>() * imag(phase_coefficient);

            std::array<qubit_type, 2u> target_qubits{{target_qubit1, target_qubit2}};
            ::ket::gate::fused::runtime::ranges::gate(
              first, num_fused_qubits,
              [fused_index_wo_qubits, &unsorted_fused_qubits, &sorted_fused_qubits_with_sentinel,
               num_control_qubits, &phase_coefficient, &i_sin_theta](
                auto const first, StateInteger const operated_index_wo_qubits,
                auto const& unsorted_operated_qubits, auto const& sorted_operated_qubits_with_sentinel)
              {
                // 0b11...100u
                auto const index00 = ((std::size_t{1u} << num_control_qubits) - std::size_t{1u}) << BitInteger{2u};
                auto const iter00
                  = first
                    + ::ket::gate::utility::ranges::index_with_qubits(
                        fused_index_wo_qubits,
                        ::ket::gate::utility::ranges::index_with_qubits(
                          operated_index_wo_qubits, index00,
                          unsorted_operated_qubits, sorted_operated_qubits_with_sentinel),
                        unsorted_fused_qubits, sorted_fused_qubits_with_sentinel);
                // 0b11...101u
                auto const iter01
                  = first
                    + ::ket::gate::utility::ranges::index_with_qubits(
                        fused_index_wo_qubits,
                        ::ket::gate::utility::ranges::index_with_qubits(
                          operated_index_wo_qubits, index00 bitor std::size_t{1u},
                          unsorted_operated_qubits, sorted_operated_qubits_with_sentinel),
                        unsorted_fused_qubits, sorted_fused_qubits_with_sentinel);
                // 0b11...110u
                auto const index10 = index00 bitor (std::size_t{1u} << BitInteger{1u});
                auto const iter10
                  = first
                    + ::ket::gate::utility::ranges::index_with_qubits(
                        fused_index_wo_qubits,
                        ::ket::gate::utility::ranges::index_with_qubits(
                          operated_index_wo_qubits, index10,
                          unsorted_operated_qubits, sorted_operated_qubits_with_sentinel),
                        unsorted_fused_qubits, sorted_fused_qubits_with_sentinel);
                // 0b11...111u
                auto const iter11
                  = first
                    + ::ket::gate::utility::ranges::index_with_qubits(
                        fused_index_wo_qubits,
                        ::ket::gate::utility::ranges::index_with_qubits(
                          operated_index_wo_qubits, index10 bitor std::size_t{1u},
                          unsorted_operated_qubits, sorted_operated_qubits_with_sentinel),
                        unsorted_fused_qubits, sorted_fused_qubits_with_sentinel);

                *iter00 *= phase_coefficient;
                *iter11 *= phase_coefficient;

                auto const value01 = *iter01;
                using std::real;
                using std::imag;
                *iter01 *= real(phase_coefficient);
                *iter01 += *iter10 * i_sin_theta;
                *iter10 *= real(phase_coefficient);
                *iter10 += value01 * i_sin_theta;
              },
              boost::join(
                target_qubits,
                control_qubits | boost::adaptors::transformed(
                  [](control_qubit_type const control_qubit) { return control_qubit.qubit(); })));
          }

          template <typename RandomAccessIterator, typename StateInteger, typename QubitsRange1, typename QubitsRange2, typename Complex, typename BitInteger>
          inline auto exponential_swap_coeff(
            RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
            QubitsRange1 const& unsorted_fused_qubits, QubitsRange2 const& sorted_fused_qubits_with_sentinel,
            Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
            ::ket::qubit<StateInteger, BitInteger> const target_qubit1, ::ket::qubit<StateInteger, BitInteger> const target_qubit2)
          -> void
          {
            using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
            using control_qubit_type = ::ket::control<qubit_type>;
            std::array<control_qubit_type, 0u> const control_qubits{};
            ::ket::gate::fused::runtime::ranges::exponential_swap_coeff(
              first, fused_index_wo_qubits,
              unsorted_fused_qubits, sorted_fused_qubits_with_sentinel,
              phase_coefficient, target_qubit1, target_qubit2, control_qubits);
          }

          template <typename RandomAccessIterator, typename StateInteger, typename QubitsRange1, typename QubitsRange2, typename Complex, typename BitInteger, typename ControlQubitsRange>
          inline auto adj_exponential_swap_coeff(
            RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
            QubitsRange1 const& unsorted_fused_qubits, QubitsRange2 const& sorted_fused_qubits_with_sentinel,
            Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
            ::ket::qubit<StateInteger, BitInteger> const target_qubit1, ::ket::qubit<StateInteger, BitInteger> const target_qubit2,
            ControlQubitsRange const& control_qubits)
          -> void
          {
            using std::conj;
            ::ket::gate::fused::runtime::ranges::exponential_swap_coeff(
              first, fused_index_wo_qubits,
              unsorted_fused_qubits, sorted_fused_qubits_with_sentinel,
              conj(phase_coefficient), target_qubit1, target_qubit2, control_qubits);
          }

          template <typename RandomAccessIterator, typename StateInteger, typename QubitsRange1, typename QubitsRange2, typename Complex, typename BitInteger>
          inline auto adj_exponential_swap_coeff(
            RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
            QubitsRange1 const& unsorted_fused_qubits, QubitsRange2 const& sorted_fused_qubits_with_sentinel,
            Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
            ::ket::qubit<StateInteger, BitInteger> const target_qubit1, ::ket::qubit<StateInteger, BitInteger> const target_qubit2)
          -> void
          {
            using std::conj;
            ::ket::gate::fused::runtime::ranges::exponential_swap_coeff(
              first, fused_index_wo_qubits,
              unsorted_fused_qubits, sorted_fused_qubits_with_sentinel,
              conj(phase_coefficient), target_qubit1, target_qubit2);
          }
        } // namespace ranges

        template <typename RandomAccessIterator, typename StateInteger, typename QubitIterator1, typename QubitIterator2, typename Complex, typename BitInteger, typename ControlQubitIterator>
        inline auto exponential_swap_coeff(
          RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
          QubitIterator1 const unsorted_fused_qubit_first, QubitIterator1 const unsorted_fused_qubit_last,
          QubitIterator2 const sorted_fused_qubit_with_sentinel_first, QubitIterator2 const sorted_fused_qubit_with_sentinel_last,
          Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
          ::ket::qubit<StateInteger, BitInteger> const target_qubit1, ::ket::qubit<StateInteger, BitInteger> const target_qubit2,
          ControlQubitIterator const control_qubit_first, ControlQubitIterator const control_qubit_last)
        -> void
        {
          ::ket::gate::fused::runtime::ranges::exponential_swap_coeff(
            first, fused_index_wo_qubits,
            boost::make_iterator_range(unsorted_fused_qubit_first, unsorted_fused_qubit_last),
            boost::make_iterator_range(sorted_fused_qubit_with_sentinel_first, sorted_fused_qubit_with_sentinel_last),
            phase_coefficient,
            target_qubit1, target_qubit2, boost::make_iterator_range(control_qubit_first, control_qubit_last));
        }

        template <typename RandomAccessIterator, typename StateInteger, typename QubitIterator1, typename QubitIterator2, typename Complex, typename BitInteger>
        inline auto exponential_swap_coeff(
          RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
          QubitIterator1 const unsorted_fused_qubit_first, QubitIterator1 const unsorted_fused_qubit_last,
          QubitIterator2 const sorted_fused_qubit_with_sentinel_first, QubitIterator2 const sorted_fused_qubit_with_sentinel_last,
          Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
          ::ket::qubit<StateInteger, BitInteger> const target_qubit1, ::ket::qubit<StateInteger, BitInteger> const target_qubit2)
        -> void
        {
          ::ket::gate::fused::runtime::ranges::exponential_swap_coeff(
            first, fused_index_wo_qubits,
            boost::make_iterator_range(unsorted_fused_qubit_first, unsorted_fused_qubit_last),
            boost::make_iterator_range(sorted_fused_qubit_with_sentinel_first, sorted_fused_qubit_with_sentinel_last),
            phase_coefficient, target_qubit1, target_qubit2);
        }

        template <typename RandomAccessIterator, typename StateInteger, typename QubitIterator1, typename QubitIterator2, typename Complex, typename BitInteger, typename ControlQubitIterator>
        inline auto adj_exponential_swap_coeff(
          RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
          QubitIterator1 const unsorted_fused_qubit_first, QubitIterator1 const unsorted_fused_qubit_last,
          QubitIterator2 const sorted_fused_qubit_with_sentinel_first, QubitIterator2 const sorted_fused_qubit_with_sentinel_last,
          Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
          ::ket::qubit<StateInteger, BitInteger> const target_qubit1, ::ket::qubit<StateInteger, BitInteger> const target_qubit2,
          ControlQubitIterator const control_qubit_first, ControlQubitIterator const control_qubit_last)
        -> void
        {
          ::ket::gate::fused::runtime::ranges::adj_exponential_swap_coeff(
            first, fused_index_wo_qubits,
            boost::make_iterator_range(unsorted_fused_qubit_first, unsorted_fused_qubit_last),
            boost::make_iterator_range(sorted_fused_qubit_with_sentinel_first, sorted_fused_qubit_with_sentinel_last),
            phase_coefficient,
            target_qubit1, target_qubit2, boost::make_iterator_range(control_qubit_first, control_qubit_last));
        }

        template <typename RandomAccessIterator, typename StateInteger, typename QubitIterator1, typename QubitIterator2, typename Complex, typename BitInteger>
        inline auto adj_exponential_swap_coeff(
          RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
          QubitIterator1 const unsorted_fused_qubit_first, QubitIterator1 const unsorted_fused_qubit_last,
          QubitIterator2 const sorted_fused_qubit_with_sentinel_first, QubitIterator2 const sorted_fused_qubit_with_sentinel_last,
          Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
          ::ket::qubit<StateInteger, BitInteger> const target_qubit1, ::ket::qubit<StateInteger, BitInteger> const target_qubit2)
        -> void
        {
          ::ket::gate::fused::runtime::ranges::adj_exponential_swap_coeff(
            first, fused_index_wo_qubits,
            boost::make_iterator_range(unsorted_fused_qubit_first, unsorted_fused_qubit_last),
            boost::make_iterator_range(sorted_fused_qubit_with_sentinel_first, sorted_fused_qubit_with_sentinel_last),
            phase_coefficient, target_qubit1, target_qubit2);
        }

        // exponential_swap
        template <typename RandomAccessIterator, typename StateInteger, typename QubitIterator1, typename QubitIterator2, typename Real, typename BitInteger, typename ControlQubitIterator>
        inline auto exponential_swap(
          RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
          QubitIterator1 const unsorted_fused_qubit_first, QubitIterator1 const unsorted_fused_qubit_last,
          QubitIterator2 const sorted_fused_qubit_with_sentinel_first, QubitIterator2 const sorted_fused_qubit_with_sentinel_last,
          Real const phase,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit1, ::ket::qubit<StateInteger, BitInteger> const target_qubit2,
          ControlQubitIterator const control_qubit_first, ControlQubitIterator const control_qubit_last)
        -> void
        {
          using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
          ::ket::gate::fused::runtime::exponential_swap_coeff(
            first, fused_index_wo_qubits,
            unsorted_fused_qubit_first, unsorted_fused_qubit_last,
            sorted_fused_qubit_with_sentinel_first, sorted_fused_qubit_with_sentinel_last,
            ::ket::utility::exp_i<complex_type>(phase),
            target_qubit1, target_qubit2, control_qubit_first, control_qubit_last);
        }

        template <typename RandomAccessIterator, typename StateInteger, typename QubitIterator1, typename QubitIterator2, typename Real, typename BitInteger>
        inline auto exponential_swap(
          RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
          QubitIterator1 const unsorted_fused_qubit_first, QubitIterator1 const unsorted_fused_qubit_last,
          QubitIterator2 const sorted_fused_qubit_with_sentinel_first, QubitIterator2 const sorted_fused_qubit_with_sentinel_last,
          Real const phase,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit1, ::ket::qubit<StateInteger, BitInteger> const target_qubit2)
        -> void
        {
          using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
          ::ket::gate::fused::runtime::exponential_swap_coeff(
            first, fused_index_wo_qubits,
            unsorted_fused_qubit_first, unsorted_fused_qubit_last,
            sorted_fused_qubit_with_sentinel_first, sorted_fused_qubit_with_sentinel_last,
            ::ket::utility::exp_i<complex_type>(phase), target_qubit1, target_qubit2);
        }

        template <typename RandomAccessIterator, typename StateInteger, typename QubitIterator1, typename QubitIterator2, typename Real, typename BitInteger, typename ControlQubitIterator>
        inline auto adj_exponential_swap(
          RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
          QubitIterator1 const unsorted_fused_qubit_first, QubitIterator1 const unsorted_fused_qubit_last,
          QubitIterator2 const sorted_fused_qubit_with_sentinel_first, QubitIterator2 const sorted_fused_qubit_with_sentinel_last,
          Real const phase,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit1, ::ket::qubit<StateInteger, BitInteger> const target_qubit2,
          ControlQubitIterator const control_qubit_first, ControlQubitIterator const control_qubit_last)
        -> void
        {
          using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
          ::ket::gate::fused::runtime::adj_exponential_swap_coeff(
            first, fused_index_wo_qubits,
            unsorted_fused_qubit_first, unsorted_fused_qubit_last,
            sorted_fused_qubit_with_sentinel_first, sorted_fused_qubit_with_sentinel_last,
            ::ket::utility::exp_i<complex_type>(phase),
            target_qubit1, target_qubit2, control_qubit_first, control_qubit_last);
        }

        template <typename RandomAccessIterator, typename StateInteger, typename QubitIterator1, typename QubitIterator2, typename Real, typename BitInteger>
        inline auto adj_exponential_swap(
          RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
          QubitIterator1 const unsorted_fused_qubit_first, QubitIterator1 const unsorted_fused_qubit_last,
          QubitIterator2 const sorted_fused_qubit_with_sentinel_first, QubitIterator2 const sorted_fused_qubit_with_sentinel_last,
          Real const phase,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit1, ::ket::qubit<StateInteger, BitInteger> const target_qubit2)
        -> void
        {
          using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
          ::ket::gate::fused::runtime::adj_exponential_swap_coeff(
            first, fused_index_wo_qubits,
            unsorted_fused_qubit_first, unsorted_fused_qubit_last,
            sorted_fused_qubit_with_sentinel_first, sorted_fused_qubit_with_sentinel_last,
            ::ket::utility::exp_i<complex_type>(phase), target_qubit1, target_qubit2);
        }

        namespace ranges
        {
          template <typename RandomAccessIterator, typename StateInteger, typename QubitsRange1, typename QubitsRange2, typename Real, typename BitInteger, typename ControlQubitsRange>
          inline auto exponential_swap(
            RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
            QubitsRange1 const& unsorted_fused_qubits, QubitsRange2 const& sorted_fused_qubits_with_sentinel,
            Real const phase,
            ::ket::qubit<StateInteger, BitInteger> const target_qubit1, ::ket::qubit<StateInteger, BitInteger> const target_qubit2,
            ControlQubitsRange const& control_qubits)
          -> void
          {
            using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
            ::ket::gate::fused::runtime::ranges::exponential_swap_coeff(
              first, fused_index_wo_qubits,
              unsorted_fused_qubits, sorted_fused_qubits_with_sentinel,
              ::ket::utility::exp_i<complex_type>(phase), target_qubit1, target_qubit2, control_qubits);
          }

          template <typename RandomAccessIterator, typename StateInteger, typename QubitsRange1, typename QubitsRange2, typename Real, typename BitInteger>
          inline auto exponential_swap(
            RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
            QubitsRange1 const& unsorted_fused_qubits, QubitsRange2 const& sorted_fused_qubits_with_sentinel,
            Real const phase,
            ::ket::qubit<StateInteger, BitInteger> const target_qubit1, ::ket::qubit<StateInteger, BitInteger> const target_qubit2)
          -> void
          {
            using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
            ::ket::gate::fused::runtime::ranges::exponential_swap_coeff(
              first, fused_index_wo_qubits,
              unsorted_fused_qubits, sorted_fused_qubits_with_sentinel,
              ::ket::utility::exp_i<complex_type>(phase), target_qubit1, target_qubit2);
          }

          template <typename RandomAccessIterator, typename StateInteger, typename QubitsRange1, typename QubitsRange2, typename Real, typename BitInteger, typename ControlQubitsRange>
          inline auto adj_exponential_swap(
            RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
            QubitsRange1 const& unsorted_fused_qubits, QubitsRange2 const& sorted_fused_qubits_with_sentinel,
            Real const phase,
            ::ket::qubit<StateInteger, BitInteger> const target_qubit1, ::ket::qubit<StateInteger, BitInteger> const target_qubit2,
            ControlQubitsRange const& control_qubits)
          -> void
          {
            using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
            ::ket::gate::fused::runtime::ranges::adj_exponential_swap_coeff(
              first, fused_index_wo_qubits,
              unsorted_fused_qubits, sorted_fused_qubits_with_sentinel,
              ::ket::utility::exp_i<complex_type>(phase), target_qubit1, target_qubit2, control_qubits);
          }

          template <typename RandomAccessIterator, typename StateInteger, typename QubitsRange1, typename QubitsRange2, typename Real, typename BitInteger>
          inline auto adj_exponential_swap(
            RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
            QubitsRange1 const& unsorted_fused_qubits, QubitsRange2 const& sorted_fused_qubits_with_sentinel,
            Real const phase,
            ::ket::qubit<StateInteger, BitInteger> const target_qubit1, ::ket::qubit<StateInteger, BitInteger> const target_qubit2)
          -> void
          {
            using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
            ::ket::gate::fused::runtime::ranges::adj_exponential_swap_coeff(
              first, fused_index_wo_qubits,
              unsorted_fused_qubits, sorted_fused_qubits_with_sentinel,
              ::ket::utility::exp_i<complex_type>(phase), target_qubit1, target_qubit2);
          }
        } // namespace ranges
      } // namespace runtime
# else // KET_USE_BIT_MASKS_EXPLICITLY
      // exponential_swap_coeff
      // eSWAP_{ij}(s) = exp(is SWAP_{ij}) = I cos s + i SWAP_{ij} sin s
      // eSWAP_{1,2}(s) (a_{00} |00> + a_{01} |01> + a_{10} |10> + a_{11} |11>)
      //   = e^{is} a_{00} |00> + (cos s a_{01} + i sin s a_{10}) |01> + (i sin s a_{01} + cos s a_{10}) |10> + e^{is} a_{11} |11>
      template <typename RandomAccessIterator, typename StateInteger, std::size_t num_fused_qubits, typename Complex, typename BitInteger>
      inline auto exponential_swap_coeff(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array<StateInteger, num_fused_qubits> const& fused_qubit_masks, std::array<StateInteger, num_fused_qubits + 1u> const& fused_index_masks,
        Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
        ::ket::qubit<StateInteger, BitInteger> const qubit1, ::ket::qubit<StateInteger, BitInteger> const qubit2)
      -> void
      {
        static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
        static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");
        static_assert(
          std::is_same<Complex, typename std::iterator_traits<RandomAccessIterator>::value_type>::value,
          "Complex should be the same to value_type of RandomAccessIterator");
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

        using std::imag;
        auto const i_sin_theta = ::ket::utility::imaginary_unit<Complex>() * imag(phase_coefficient);

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

          *off_iter *= phase_coefficient;
          *qubit12_on_iter *= phase_coefficient;

          auto const qubit1_on_iter_value = *qubit1_on_iter;
          using std::real;
          using std::imag;
          *qubit1_on_iter *= real(phase_coefficient);
          *qubit1_on_iter += *qubit2_on_iter * i_sin_theta;
          *qubit2_on_iter *= real(phase_coefficient);
          *qubit2_on_iter += qubit1_on_iter_value * i_sin_theta;
        }
      }

      // C...CeSWAP_{tt'c...c'}(s) = C...C[exp(is SWAP_{tt'})]_{c...c'} = C...C[I cos s + i SWAP_{tt'} sin s]_{c...c'}
      template <typename RandomAccessIterator, typename StateInteger, std::size_t num_fused_qubits, typename Complex, typename BitInteger, typename... ControlQubits>
      inline auto exponential_swap_coeff(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array<StateInteger, num_fused_qubits> const& fused_qubit_masks, std::array<StateInteger, num_fused_qubits + 1u> const& fused_index_masks,
        Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
        ::ket::qubit<StateInteger, BitInteger> const target_qubit1, ::ket::qubit<StateInteger, BitInteger> const target_qubit2,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit, ControlQubits const... control_qubits)
      -> void
      {
        static_assert(
          std::is_same<Complex, typename std::iterator_traits<RandomAccessIterator>::value_type>::value,
          "Complex should be the same to value_type of RandomAccessIterator");

        constexpr auto num_control_qubits = static_cast<BitInteger>(sizeof...(ControlQubits) + 1u);
        constexpr auto num_operated_qubits = num_control_qubits + BitInteger{2u};

        using std::imag;
        auto const i_sin_theta = ::ket::utility::imaginary_unit<Complex>() * imag(phase_coefficient);

        ::ket::gate::fused::gate<num_fused_qubits>(
          first,
          [fused_index_wo_qubits, &fused_qubit_masks, &fused_index_masks, &phase_coefficient, &i_sin_theta](
            auto const first, StateInteger const operated_index_wo_qubits,
            std::array<StateInteger, num_operated_qubits> const& operated_qubit_masks,
            std::array<StateInteger, num_operated_qubits + 1u> const& operated_index_masks)
          {
            // 0b11...100u
            constexpr auto index00 = ((std::size_t{1u} << num_control_qubits) - std::size_t{1u}) << BitInteger{2u};
            using std::begin;
            using std::end;
            auto const iter00
              = first
                + ::ket::gate::utility::index_with_qubits(
                    fused_index_wo_qubits,
                    ::ket::gate::utility::index_with_qubits(
                      operated_index_wo_qubits, index00,
                      begin(operated_qubit_masks), end(operated_qubit_masks), begin(operated_index_masks), end(operated_index_masks)),
                    begin(fused_qubit_masks), end(fused_qubit_masks), begin(fused_index_masks), end(fused_index_masks));
            // 0b11...101u
            auto const iter01
              = first
                + ::ket::gate::utility::index_with_qubits(
                    fused_index_wo_qubits,
                    ::ket::gate::utility::index_with_qubits(
                      operated_index_wo_qubits, index00 bitor std::size_t{1u},
                      begin(operated_qubit_masks), end(operated_qubit_masks), begin(operated_index_masks), end(operated_index_masks)),
                    begin(fused_qubit_masks), end(fused_qubit_masks), begin(fused_index_masks), end(fused_index_masks));
            // 0b11...110u
            constexpr auto index10 = index00 bitor (std::size_t{1u} << BitInteger{1u});
            auto const iter10
              = first
                + ::ket::gate::utility::index_with_qubits(
                    fused_index_wo_qubits,
                    ::ket::gate::utility::index_with_qubits(
                      operated_index_wo_qubits, index10,
                      begin(operated_qubit_masks), end(operated_qubit_masks), begin(operated_index_masks), end(operated_index_masks)),
                    begin(fused_qubit_masks), end(fused_qubit_masks), begin(fused_index_masks), end(fused_index_masks));
            // 0b11...111u
            auto const iter11
              = first
                + ::ket::gate::utility::index_with_qubits(
                    fused_index_wo_qubits,
                    ::ket::gate::utility::index_with_qubits(
                      operated_index_wo_qubits, index10 bitor std::size_t{1u},
                      begin(operated_qubit_masks), end(operated_qubit_masks), begin(operated_index_masks), end(operated_index_masks)),
                    begin(fused_qubit_masks), end(fused_qubit_masks), begin(fused_index_masks), end(fused_index_masks));

            *iter00 *= phase_coefficient;
            *iter11 *= phase_coefficient;

            auto const value01 = *iter01;
            using std::real;
            using std::imag;
            *iter01 *= real(phase_coefficient);
            *iter01 += *iter10 * i_sin_theta;
            *iter10 *= real(phase_coefficient);
            *iter10 += value01 * i_sin_theta;
          },
          target_qubit1, target_qubit2, control_qubit, control_qubits...);
      }

      template <typename RandomAccessIterator, typename StateInteger, std::size_t num_fused_qubits, typename Complex, typename BitInteger, typename... ControlQubits>
      inline auto adj_exponential_swap_coeff(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array<StateInteger, num_fused_qubits> const& fused_qubit_masks, std::array<StateInteger, num_fused_qubits + 1u> const& fused_index_masks,
        Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
        ::ket::qubit<StateInteger, BitInteger> const target_qubit1, ::ket::qubit<StateInteger, BitInteger> const target_qubit2,
        ControlQubits const... control_qubits)
      -> void
      { using std::conj; ::ket::gate::fused::exponential_swap_coeff(first, fused_index_wo_qubits, fused_qubit_masks, fused_index_masks, conj(phase_coefficient), target_qubit1, target_qubit2, control_qubits...); }

      // exponential_swap
      template <typename RandomAccessIterator, typename StateInteger, std::size_t num_fused_qubits, typename Real, typename BitInteger, typename... ControlQubits>
      inline auto exponential_swap(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array<StateInteger, num_fused_qubits> const& fused_qubit_masks, std::array<StateInteger, num_fused_qubits + 1u> const& fused_index_masks,
        Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit1, ::ket::qubit<StateInteger, BitInteger> const target_qubit2,
        ControlQubits const... control_qubits)
      -> void
      {
        using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
        ::ket::gate::fused::exponential_swap_coeff(first, fused_index_wo_qubits, fused_qubit_masks, fused_index_masks, ::ket::utility::exp_i<complex_type>(phase), target_qubit1, target_qubit2, control_qubits...);
      }

      template <typename RandomAccessIterator, typename StateInteger, std::size_t num_fused_qubits, typename Real, typename BitInteger, typename... ControlQubits>
      inline auto adj_exponential_swap(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array<StateInteger, num_fused_qubits> const& fused_qubit_masks, std::array<StateInteger, num_fused_qubits + 1u> const& fused_index_masks,
        Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit1, ::ket::qubit<StateInteger, BitInteger> const target_qubit2,
        ControlQubits const... control_qubits)
      -> void
      { ::ket::gate::fused::exponential_swap(first, fused_index_wo_qubits, fused_qubit_masks, fused_index_masks, -phase, target_qubit1, target_qubit2, control_qubits...); }


      namespace runtime
      {
        // exponential_swap_coeff
        namespace ranges
        {
          // C...CeSWAP_{tt'c...c'}(s) = C...C[exp(is SWAP_{tt'})]_{c...c'} = C...C[I cos s + i SWAP_{tt'} sin s]_{c...c'}
          template <typename RandomAccessIterator, typename StateInteger, typename StateIntegersRange1, typename StateIntegersRange2, typename Complex, typename BitInteger, typename ControlQubitsRange>
          inline auto exponential_swap_coeff(
            RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
            StateIntegersRange1 const& fused_qubit_masks, StateIntegersRange2 const& fused_index_masks,
            Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
            ::ket::qubit<StateInteger, BitInteger> const target_qubit1, ::ket::qubit<StateInteger, BitInteger> const target_qubit2,
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
            assert(num_control_qubits + BitInteger{2u} <= num_fused_qubits);

            assert(target_qubit1 < qubit_type{num_fused_qubits});
            assert(target_qubit2 < qubit_type{num_fused_qubits});
            assert(::ket::utility::runtime::ranges::all_in_state_vector(num_fused_qubits, control_qubits));

            static_assert(
              std::is_same<Complex, typename std::iterator_traits<RandomAccessIterator>::value_type>::value,
              "Complex should be the same to value_type of RandomAccessRange");

            using std::imag;
            auto const i_sin_theta = ::ket::utility::imaginary_unit<Complex>() * imag(phase_coefficient);

            std::array<qubit_type, 2u> target_qubits{{target_qubit1, target_qubit2}};
            ::ket::gate::fused::runtime::ranges::gate(
              first, num_fused_qubits,
              [fused_index_wo_qubits, &fused_qubit_masks, &fused_index_masks,
               num_control_qubits, &phase_coefficient, &i_sin_theta](
                auto const first, StateInteger const operated_index_wo_qubits,
                auto const& operated_qubit_masks, auto const& operated_index_masks)
              {
                // 0b11...100u
                auto const index00 = ((std::size_t{1u} << num_control_qubits) - std::size_t{1u}) << BitInteger{2u};
                auto const iter00
                  = first
                    + ::ket::gate::utility::ranges::index_with_qubits(
                        fused_index_wo_qubits,
                        ::ket::gate::utility::ranges::index_with_qubits(
                          operated_index_wo_qubits, index00,
                          operated_qubit_masks, operated_index_masks),
                        fused_qubit_masks, fused_index_masks);
                // 0b11...101u
                auto const iter01
                  = first
                    + ::ket::gate::utility::ranges::index_with_qubits(
                        fused_index_wo_qubits,
                        ::ket::gate::utility::ranges::index_with_qubits(
                          operated_index_wo_qubits, index00 bitor std::size_t{1u},
                          operated_qubit_masks, operated_index_masks),
                        fused_qubit_masks, fused_index_masks);
                // 0b11...110u
                auto const index10 = index00 bitor (std::size_t{1u} << BitInteger{1u});
                auto const iter10
                  = first
                    + ::ket::gate::utility::ranges::index_with_qubits(
                        fused_index_wo_qubits,
                        ::ket::gate::utility::ranges::index_with_qubits(
                          operated_index_wo_qubits, index10,
                          operated_qubit_masks, operated_index_masks),
                        fused_qubit_masks, fused_index_masks);
                // 0b11...111u
                auto const iter11
                  = first
                    + ::ket::gate::utility::ranges::index_with_qubits(
                        fused_index_wo_qubits,
                        ::ket::gate::utility::ranges::index_with_qubits(
                          operated_index_wo_qubits, index10 bitor std::size_t{1u},
                          operated_qubit_masks, operated_index_masks),
                        fused_qubit_masks, fused_index_masks);

                *iter00 *= phase_coefficient;
                *iter11 *= phase_coefficient;

                auto const value01 = *iter01;
                using std::real;
                using std::imag;
                *iter01 *= real(phase_coefficient);
                *iter01 += *iter10 * i_sin_theta;
                *iter10 *= real(phase_coefficient);
                *iter10 += value01 * i_sin_theta;
              },
              boost::join(
                target_qubits,
                control_qubits | boost::adaptors::transformed(
                  [](control_qubit_type const control_qubit) { return control_qubit.qubit(); })));
          }

          template <typename RandomAccessIterator, typename StateInteger, typename StateIntegersRange1, typename StateIntegersRange2, typename Complex, typename BitInteger>
          inline auto exponential_swap_coeff(
            RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
            StateIntegersRange1 const& fused_qubit_masks, StateIntegersRange2 const& fused_index_masks,
            Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
            ::ket::qubit<StateInteger, BitInteger> const target_qubit1, ::ket::qubit<StateInteger, BitInteger> const target_qubit2)
          -> void
          {
            using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
            using control_qubit_type = ::ket::control<qubit_type>;
            std::array<control_qubit_type, 0u> const control_qubits{};
            ::ket::gate::fused::runtime::ranges::exponential_swap_coeff(
              first, fused_index_wo_qubits,
              fused_qubit_masks, fused_index_masks,
              phase_coefficient, target_qubit1, target_qubit2, control_qubits);
          }

          template <typename RandomAccessIterator, typename StateInteger, typename StateIntegersRange1, typename StateIntegersRange2, typename Complex, typename BitInteger, typename ControlQubitsRange>
          inline auto adj_exponential_swap_coeff(
            RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
            StateIntegersRange1 const& fused_qubit_masks, StateIntegersRange2 const& fused_index_masks,
            Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
            ::ket::qubit<StateInteger, BitInteger> const target_qubit1, ::ket::qubit<StateInteger, BitInteger> const target_qubit2,
            ControlQubitsRange const& control_qubits)
          -> void
          {
            using std::conj;
            ::ket::gate::fused::runtime::ranges::exponential_swap_coeff(
              first, fused_index_wo_qubits,
              fused_qubit_masks, fused_index_masks,
              conj(phase_coefficient), target_qubit1, target_qubit2, control_qubits);
          }

          template <typename RandomAccessIterator, typename StateInteger, typename StateIntegersRange1, typename StateIntegersRange2, typename Complex, typename BitInteger>
          inline auto adj_exponential_swap_coeff(
            RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
            StateIntegersRange1 const& fused_qubit_masks, StateIntegersRange2 const& fused_index_masks,
            Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
            ::ket::qubit<StateInteger, BitInteger> const target_qubit1, ::ket::qubit<StateInteger, BitInteger> const target_qubit2)
          -> void
          {
            using std::conj;
            ::ket::gate::fused::runtime::ranges::exponential_swap_coeff(
              first, fused_index_wo_qubits,
              fused_qubit_masks, fused_index_masks,
              conj(phase_coefficient), target_qubit1, target_qubit2);
          }
        } // namespace ranges

        template <typename RandomAccessIterator, typename StateInteger, typename StateIntegerIterator1, typename StateIntegerIterator2, typename Complex, typename BitInteger, typename ControlQubitIterator>
        inline auto exponential_swap_coeff(
          RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
          StateIntegerIterator1 const fused_qubit_mask_first, StateIntegerIterator1 const fused_qubit_mask_last,
          StateIntegerIterator2 const fused_index_mask_first, StateIntegerIterator2 const fused_index_mask_last,
          Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
          ::ket::qubit<StateInteger, BitInteger> const target_qubit1, ::ket::qubit<StateInteger, BitInteger> const target_qubit2,
          ControlQubitIterator const control_qubit_first, ControlQubitIterator const control_qubit_last)
        -> void
        {
          ::ket::gate::fused::runtime::ranges::exponential_swap_coeff(
            first, fused_index_wo_qubits,
            boost::make_iterator_range(fused_qubit_mask_first, fused_qubit_mask_last),
            boost::make_iterator_range(fused_index_mask_first, fused_index_mask_last),
            phase_coefficient,
            target_qubit1, target_qubit2, boost::make_iterator_range(control_qubit_first, control_qubit_last));
        }

        template <typename RandomAccessIterator, typename StateInteger, typename StateIntegerIterator1, typename StateIntegerIterator2, typename Complex, typename BitInteger>
        inline auto exponential_swap_coeff(
          RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
          StateIntegerIterator1 const fused_qubit_mask_first, StateIntegerIterator1 const fused_qubit_mask_last,
          StateIntegerIterator2 const fused_index_mask_first, StateIntegerIterator2 const fused_index_mask_last,
          Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
          ::ket::qubit<StateInteger, BitInteger> const target_qubit1, ::ket::qubit<StateInteger, BitInteger> const target_qubit2)
        -> void
        {
          ::ket::gate::fused::runtime::ranges::exponential_swap_coeff(
            first, fused_index_wo_qubits,
            boost::make_iterator_range(fused_qubit_mask_first, fused_qubit_mask_last),
            boost::make_iterator_range(fused_index_mask_first, fused_index_mask_last),
            phase_coefficient, target_qubit1, target_qubit2);
        }

        template <typename RandomAccessIterator, typename StateInteger, typename StateIntegerIterator1, typename StateIntegerIterator2, typename Complex, typename BitInteger, typename ControlQubitIterator>
        inline auto adj_exponential_swap_coeff(
          RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
          StateIntegerIterator1 const fused_qubit_mask_first, StateIntegerIterator1 const fused_qubit_mask_last,
          StateIntegerIterator2 const fused_index_mask_first, StateIntegerIterator2 const fused_index_mask_last,
          Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
          ::ket::qubit<StateInteger, BitInteger> const target_qubit1, ::ket::qubit<StateInteger, BitInteger> const target_qubit2,
          ControlQubitIterator const control_qubit_first, ControlQubitIterator const control_qubit_last)
        -> void
        {
          ::ket::gate::fused::runtime::ranges::adj_exponential_swap_coeff(
            first, fused_index_wo_qubits,
            boost::make_iterator_range(fused_qubit_mask_first, fused_qubit_mask_last),
            boost::make_iterator_range(fused_index_mask_first, fused_index_mask_last),
            phase_coefficient,
            target_qubit1, target_qubit2, boost::make_iterator_range(control_qubit_first, control_qubit_last));
        }

        template <typename RandomAccessIterator, typename StateInteger, typename StateIntegerIterator1, typename StateIntegerIterator2, typename Complex, typename BitInteger>
        inline auto adj_exponential_swap_coeff(
          RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
          StateIntegerIterator1 const fused_qubit_mask_first, StateIntegerIterator1 const fused_qubit_mask_last,
          StateIntegerIterator2 const fused_index_mask_first, StateIntegerIterator2 const fused_index_mask_last,
          Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
          ::ket::qubit<StateInteger, BitInteger> const target_qubit1, ::ket::qubit<StateInteger, BitInteger> const target_qubit2)
        -> void
        {
          ::ket::gate::fused::runtime::ranges::adj_exponential_swap_coeff(
            first, fused_index_wo_qubits,
            boost::make_iterator_range(fused_qubit_mask_first, fused_qubit_mask_last),
            boost::make_iterator_range(fused_index_mask_first, fused_index_mask_last),
            phase_coefficient, target_qubit1, target_qubit2);
        }

        // exponential_swap
        template <typename RandomAccessIterator, typename StateInteger, typename StateIntegerIterator1, typename StateIntegerIterator2, typename Real, typename BitInteger, typename ControlQubitIterator>
        inline auto exponential_swap(
          RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
          StateIntegerIterator1 const fused_qubit_mask_first, StateIntegerIterator1 const fused_qubit_mask_last,
          StateIntegerIterator2 const fused_index_mask_first, StateIntegerIterator2 const fused_index_mask_last,
          Real const phase,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit1, ::ket::qubit<StateInteger, BitInteger> const target_qubit2,
          ControlQubitIterator const control_qubit_first, ControlQubitIterator const control_qubit_last)
        -> void
        {
          using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
          ::ket::gate::fused::runtime::exponential_swap_coeff(
            first, fused_index_wo_qubits,
            fused_qubit_mask_first, fused_qubit_mask_last,
            fused_index_mask_first, fused_index_mask_last,
            ::ket::utility::exp_i<complex_type>(phase),
            target_qubit1, target_qubit2, control_qubit_first, control_qubit_last);
        }

        template <typename RandomAccessIterator, typename StateInteger, typename StateIntegerIterator1, typename StateIntegerIterator2, typename Real, typename BitInteger>
        inline auto exponential_swap(
          RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
          StateIntegerIterator1 const fused_qubit_mask_first, StateIntegerIterator1 const fused_qubit_mask_last,
          StateIntegerIterator2 const fused_index_mask_first, StateIntegerIterator2 const fused_index_mask_last,
          Real const phase,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit1, ::ket::qubit<StateInteger, BitInteger> const target_qubit2)
        -> void
        {
          using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
          ::ket::gate::fused::runtime::exponential_swap_coeff(
            first, fused_index_wo_qubits,
            fused_qubit_mask_first, fused_qubit_mask_last,
            fused_index_mask_first, fused_index_mask_last,
            ::ket::utility::exp_i<complex_type>(phase), target_qubit1, target_qubit2);
        }

        template <typename RandomAccessIterator, typename StateInteger, typename StateIntegerIterator1, typename StateIntegerIterator2, typename Real, typename BitInteger, typename ControlQubitIterator>
        inline auto adj_exponential_swap(
          RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
          StateIntegerIterator1 const fused_qubit_mask_first, StateIntegerIterator1 const fused_qubit_mask_last,
          StateIntegerIterator2 const fused_index_mask_first, StateIntegerIterator2 const fused_index_mask_last,
          Real const phase,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit1, ::ket::qubit<StateInteger, BitInteger> const target_qubit2,
          ControlQubitIterator const control_qubit_first, ControlQubitIterator const control_qubit_last)
        -> void
        {
          using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
          ::ket::gate::fused::runtime::adj_exponential_swap_coeff(
            first, fused_index_wo_qubits,
            fused_qubit_mask_first, fused_qubit_mask_last,
            fused_index_mask_first, fused_index_mask_last,
            ::ket::utility::exp_i<complex_type>(phase),
            target_qubit1, target_qubit2, control_qubit_first, control_qubit_last);
        }

        template <typename RandomAccessIterator, typename StateInteger, typename StateIntegerIterator1, typename StateIntegerIterator2, typename Real, typename BitInteger>
        inline auto adj_exponential_swap(
          RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
          StateIntegerIterator1 const fused_qubit_mask_first, StateIntegerIterator1 const fused_qubit_mask_last,
          StateIntegerIterator2 const fused_index_mask_first, StateIntegerIterator2 const fused_index_mask_last,
          Real const phase,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit1, ::ket::qubit<StateInteger, BitInteger> const target_qubit2)
        -> void
        {
          using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
          ::ket::gate::fused::runtime::adj_exponential_swap_coeff(
            first, fused_index_wo_qubits,
            fused_qubit_mask_first, fused_qubit_mask_last,
            fused_index_mask_first, fused_index_mask_last,
            ::ket::utility::exp_i<complex_type>(phase), target_qubit1, target_qubit2);
        }

        namespace ranges
        {
          template <typename RandomAccessIterator, typename StateInteger, typename StateIntegersRange1, typename StateIntegersRange2, typename Real, typename BitInteger, typename ControlQubitsRange>
          inline auto exponential_swap(
            RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
            StateIntegersRange1 const& fused_qubit_masks, StateIntegersRange2 const& fused_index_masks,
            Real const phase,
            ::ket::qubit<StateInteger, BitInteger> const target_qubit1, ::ket::qubit<StateInteger, BitInteger> const target_qubit2,
            ControlQubitsRange const& control_qubits)
          -> void
          {
            using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
            ::ket::gate::fused::runtime::ranges::exponential_swap_coeff(
              first, fused_index_wo_qubits,
              fused_qubit_masks, fused_index_masks,
              ::ket::utility::exp_i<complex_type>(phase), target_qubit1, target_qubit2, control_qubits);
          }

          template <typename RandomAccessIterator, typename StateInteger, typename StateIntegersRange1, typename StateIntegersRange2, typename Real, typename BitInteger>
          inline auto exponential_swap(
            RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
            StateIntegersRange1 const& fused_qubit_masks, StateIntegersRange2 const& fused_index_masks,
            Real const phase,
            ::ket::qubit<StateInteger, BitInteger> const target_qubit1, ::ket::qubit<StateInteger, BitInteger> const target_qubit2)
          -> void
          {
            using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
            ::ket::gate::fused::runtime::ranges::exponential_swap_coeff(
              first, fused_index_wo_qubits,
              fused_qubit_masks, fused_index_masks,
              ::ket::utility::exp_i<complex_type>(phase), target_qubit1, target_qubit2);
          }

          template <typename RandomAccessIterator, typename StateInteger, typename StateIntegersRange1, typename StateIntegersRange2, typename Real, typename BitInteger, typename ControlQubitsRange>
          inline auto adj_exponential_swap(
            RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
            StateIntegersRange1 const& fused_qubit_masks, StateIntegersRange2 const& fused_index_masks,
            Real const phase,
            ::ket::qubit<StateInteger, BitInteger> const target_qubit1, ::ket::qubit<StateInteger, BitInteger> const target_qubit2,
            ControlQubitsRange const& control_qubits)
          -> void
          {
            using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
            ::ket::gate::fused::runtime::ranges::adj_exponential_swap_coeff(
              first, fused_index_wo_qubits,
              fused_qubit_masks, fused_index_masks,
              ::ket::utility::exp_i<complex_type>(phase), target_qubit1, target_qubit2, control_qubits);
          }

          template <typename RandomAccessIterator, typename StateInteger, typename StateIntegersRange1, typename StateIntegersRange2, typename Real, typename BitInteger>
          inline auto adj_exponential_swap(
            RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
            StateIntegersRange1 const& fused_qubit_masks, StateIntegersRange2 const& fused_index_masks,
            Real const phase,
            ::ket::qubit<StateInteger, BitInteger> const target_qubit1, ::ket::qubit<StateInteger, BitInteger> const target_qubit2)
          -> void
          {
            using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
            ::ket::gate::fused::runtime::ranges::adj_exponential_swap_coeff(
              first, fused_index_wo_qubits,
              fused_qubit_masks, fused_index_masks,
              ::ket::utility::exp_i<complex_type>(phase), target_qubit1, target_qubit2);
          }
        } // namespace ranges
      } // namespace runtime
# endif // KET_USE_BIT_MASKS_EXPLICITLY
    } // namespace fused
  } // namespace gate
} // namespace ket


#endif // KET_GATE_FUSED_EXPONENTIAL_SWAP_HPP
