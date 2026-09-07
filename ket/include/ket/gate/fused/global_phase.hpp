#ifndef KET_GATE_FUSED_GLOBAL_PHASE_HPP
# define KET_GATE_FUSED_GLOBAL_PHASE_HPP

# include <cassert>
# include <complex>
# include <cstddef>
# include <array>
# include <iterator>
# include <type_traits>

# include <boost/range/iterator_range.hpp>

# include <ket/qubit.hpp>
# include <ket/gate/utility/index_with_qubits.hpp>
# include <ket/utility/integer_exp2.hpp>
# include <ket/utility/exp_i.hpp>
# include <ket/utility/meta/real_of.hpp>


namespace ket
{
  namespace gate
  {
    namespace fused
    {
# ifndef KET_USE_BIT_MASKS_EXPLICITLY
      // global_phase_coeff
      template <typename RandomAccessIterator, typename StateInteger, typename BitInteger, std::size_t num_fused_qubits, typename Complex>
      inline auto global_phase_coeff(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_fused_qubits > const& unsorted_fused_qubits,
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_fused_qubits + 1u> const& sorted_fused_qubits_with_sentinel,
        Complex const& phase_coefficient) // exp(i theta) = cos(theta) + i sin(theta)
      -> void
      {
        static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
        static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");
        static_assert(
          std::is_same<Complex, typename std::iterator_traits<RandomAccessIterator>::value_type>::value,
          "Complex should be the same to value_type of RandomAccessIterator");

        constexpr auto count = ::ket::utility::integer_exp2<StateInteger>(num_fused_qubits);
        for (auto index = std::size_t{0u}; index < count; ++index)
        {
          using std::begin;
          using std::end;
          auto const iter
            = first
              + ::ket::gate::utility::index_with_qubits(
                  fused_index_wo_qubits, index,
                  begin(unsorted_fused_qubits), end(unsorted_fused_qubits),
                  begin(sorted_fused_qubits_with_sentinel), end(sorted_fused_qubits_with_sentinel));
          *iter *= phase_coefficient;
        }
      }

      template <typename RandomAccessIterator, typename StateInteger, typename BitInteger, std::size_t num_fused_qubits, typename Complex>
      inline auto adj_global_phase_coeff(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_fused_qubits > const& unsorted_fused_qubits,
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_fused_qubits + 1u> const& sorted_fused_qubits_with_sentinel,
        Complex const& phase_coefficient) // exp(i theta) = cos(theta) + i sin(theta)
      -> void
      {
        using std::conj;
        ::ket::gate::fused::global_phase_coeff(
          first, fused_index_wo_qubits, unsorted_fused_qubits, sorted_fused_qubits_with_sentinel, conj(phase_coefficient));
      }

      template <typename RandomAccessIterator, typename StateInteger, typename BitInteger, std::size_t num_fused_qubits, typename Real>
      inline auto global_phase(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_fused_qubits > const& unsorted_fused_qubits,
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_fused_qubits + 1u> const& sorted_fused_qubits_with_sentinel,
        Real const phase)
      -> void
      {
        using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
        ::ket::gate::fused::global_phase_coeff(
          first, fused_index_wo_qubits, unsorted_fused_qubits, sorted_fused_qubits_with_sentinel,
          ::ket::utility::exp_i<complex_type>(phase));
      }

      template <typename RandomAccessIterator, typename StateInteger, typename BitInteger, std::size_t num_fused_qubits, typename Real>
      inline auto adj_global_phase(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_fused_qubits > const& unsorted_fused_qubits,
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_fused_qubits + 1u> const& sorted_fused_qubits_with_sentinel,
        Real const phase)
      -> void
      { ::ket::gate::fused::global_phase(first, fused_index_wo_qubits, unsorted_fused_qubits, sorted_fused_qubits_with_sentinel, -phase); }

      namespace runtime
      {
        // global_phase_coeff
        namespace ranges
        {
          template <typename RandomAccessIterator, typename StateInteger, typename QubitsRange1, typename QubitsRange2, typename Complex>
          inline auto global_phase_coeff(
            RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
            QubitsRange1 const& unsorted_fused_qubits, QubitsRange2 const& sorted_fused_qubits_with_sentinel,
            Complex const& phase_coefficient) // exp(i theta) = cos(theta) + i sin(theta)
          -> std::enable_if_t<std::is_same<Complex, typename std::iterator_traits<RandomAccessIterator>::value_type>::value, void>
          {
            static_assert(std::is_unsigned<StateInteger>::value, "The StateInteger should be unsigned");

            using std::begin;
            using std::end;
            auto const num_fused_qubits = static_cast<StateInteger>(end(unsorted_fused_qubits) - begin(unsorted_fused_qubits));
            assert(static_cast<StateInteger>(end(sorted_fused_qubits_with_sentinel) - begin(sorted_fused_qubits_with_sentinel)) == num_fused_qubits + StateInteger{1u});

            auto const count = ::ket::utility::integer_exp2<StateInteger>(num_fused_qubits);
            for (auto index = StateInteger{0u}; index < count; ++index)
            {
              auto const iter
                = first
                  + ::ket::gate::utility::ranges::index_with_qubits(
                      fused_index_wo_qubits, index, unsorted_fused_qubits, sorted_fused_qubits_with_sentinel);
              *iter *= phase_coefficient;
            }
          }

          template <typename RandomAccessIterator, typename StateInteger, typename QubitsRange1, typename QubitsRange2, typename Complex>
          inline auto adj_global_phase_coeff(
            RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
            QubitsRange1 const& unsorted_fused_qubits, QubitsRange2 const& sorted_fused_qubits_with_sentinel,
            Complex const& phase_coefficient) // exp(i theta) = cos(theta) + i sin(theta)
          -> std::enable_if_t<std::is_same<Complex, typename std::iterator_traits<RandomAccessIterator>::value_type>::value, void>
          {
            using std::conj;
            ::ket::gate::fused::runtime::ranges::global_phase_coeff(
              first, fused_index_wo_qubits, unsorted_fused_qubits, sorted_fused_qubits_with_sentinel, conj(phase_coefficient));
          }

          template <typename RandomAccessIterator, typename StateInteger, typename QubitsRange1, typename QubitsRange2, typename Real>
          inline auto global_phase(
            RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
            QubitsRange1 const& unsorted_fused_qubits, QubitsRange2 const& sorted_fused_qubits_with_sentinel,
            Real const phase)
          -> std::enable_if_t<std::is_same<Real, ::ket::utility::meta::real_t<typename std::iterator_traits<RandomAccessIterator>::value_type>>::value, void>
          {
            using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
            ::ket::gate::fused::runtime::ranges::global_phase_coeff(
              first, fused_index_wo_qubits,
              unsorted_fused_qubits, sorted_fused_qubits_with_sentinel,
              ::ket::utility::exp_i<complex_type>(phase));
          }

          template <typename RandomAccessIterator, typename StateInteger, typename QubitsRange1, typename QubitsRange2, typename Real>
          inline auto adj_global_phase(
            RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
            QubitsRange1 const& unsorted_fused_qubits, QubitsRange2 const& sorted_fused_qubits_with_sentinel,
            Real const phase)
          -> std::enable_if_t<std::is_same<Real, ::ket::utility::meta::real_t<typename std::iterator_traits<RandomAccessIterator>::value_type>>::value, void>
          { ::ket::gate::fused::runtime::ranges::global_phase(first, fused_index_wo_qubits, unsorted_fused_qubits, sorted_fused_qubits_with_sentinel, -phase); }
        } // namespace ranges

        template <typename RandomAccessIterator, typename StateInteger, typename QubitIterator1, typename QubitIterator2, typename Complex>
        inline auto global_phase_coeff(
          RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
          QubitIterator1 const unsorted_fused_qubit_first, QubitIterator1 const unsorted_fused_qubit_last,
          QubitIterator2 const sorted_fused_qubit_with_sentinel_first, QubitIterator2 const sorted_fused_qubit_with_sentinel_last,
          Complex const& phase_coefficient) // exp(i theta) = cos(theta) + i sin(theta)
        -> void
        {
          ::ket::gate::fused::runtime::ranges::global_phase_coeff(
            first, fused_index_wo_qubits,
            boost::make_iterator_range(unsorted_fused_qubit_first, unsorted_fused_qubit_last),
            boost::make_iterator_range(sorted_fused_qubit_with_sentinel_first, sorted_fused_qubit_with_sentinel_last),
            phase_coefficient);
        }

        template <typename RandomAccessIterator, typename StateInteger, typename QubitIterator1, typename QubitIterator2, typename Complex>
        inline auto adj_global_phase_coeff(
          RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
          QubitIterator1 const unsorted_fused_qubit_first, QubitIterator1 const unsorted_fused_qubit_last,
          QubitIterator2 const sorted_fused_qubit_with_sentinel_first, QubitIterator2 const sorted_fused_qubit_with_sentinel_last,
          Complex const& phase_coefficient) // exp(i theta) = cos(theta) + i sin(theta)
        -> void
        {
          ::ket::gate::fused::runtime::ranges::adj_global_phase_coeff(
            first, fused_index_wo_qubits,
            boost::make_iterator_range(unsorted_fused_qubit_first, unsorted_fused_qubit_last),
            boost::make_iterator_range(sorted_fused_qubit_with_sentinel_first, sorted_fused_qubit_with_sentinel_last),
            phase_coefficient);
        }

        template <typename RandomAccessIterator, typename StateInteger, typename QubitIterator1, typename QubitIterator2, typename Real>
        inline auto global_phase(
          RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
          QubitIterator1 const unsorted_fused_qubit_first, QubitIterator1 const unsorted_fused_qubit_last,
          QubitIterator2 const sorted_fused_qubit_with_sentinel_first, QubitIterator2 const sorted_fused_qubit_with_sentinel_last,
          Real const phase)
        -> void
        {
          using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
          ::ket::gate::fused::runtime::global_phase_coeff(
            first, fused_index_wo_qubits,
            unsorted_fused_qubit_first, unsorted_fused_qubit_last,
            sorted_fused_qubit_with_sentinel_first, sorted_fused_qubit_with_sentinel_last,
            ::ket::utility::exp_i<complex_type>(phase));
        }

        template <typename RandomAccessIterator, typename StateInteger, typename QubitIterator1, typename QubitIterator2, typename Real>
        inline auto adj_global_phase(
          RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
          QubitIterator1 const unsorted_fused_qubit_first, QubitIterator1 const unsorted_fused_qubit_last,
          QubitIterator2 const sorted_fused_qubit_with_sentinel_first, QubitIterator2 const sorted_fused_qubit_with_sentinel_last,
          Real const phase)
        -> void
        { ::ket::gate::fused::runtime::global_phase(first, fused_index_wo_qubits, unsorted_fused_qubit_first, unsorted_fused_qubit_last, sorted_fused_qubit_with_sentinel_first, sorted_fused_qubit_with_sentinel_last, -phase); }
      } // namespace runtime
# else // KET_USE_BIT_MASKS_EXPLICITLY
      // global_phase_coeff
      template <typename RandomAccessIterator, typename StateInteger, std::size_t num_fused_qubits, typename Complex>
      inline auto global_phase_coeff(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array<StateInteger, num_fused_qubits> const& fused_qubit_masks, std::array<StateInteger, num_fused_qubits + 1u> const& fused_index_masks,
        Complex const& phase_coefficient) // exp(i theta) = cos(theta) + i sin(theta)
      -> void
      {
        static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
        static_assert(
          std::is_same<Complex, typename std::iterator_traits<RandomAccessIterator>::value_type>::value,
          "Complex should be the same to value_type of RandomAccessIterator");

        constexpr auto count = ::ket::utility::integer_exp2<StateInteger>(num_fused_qubits);
        for (auto index = std::size_t{0u}; index < count; ++index)
        {
          using std::begin;
          using std::end;
          auto const iter
            = first
              + ::ket::gate::utility::index_with_qubits(
                  fused_index_wo_qubits, index,
                  begin(fused_qubit_masks), end(fused_qubit_masks), begin(fused_index_masks), end(fused_index_masks));
          *iter *= phase_coefficient;
        }
      }

      template <typename RandomAccessIterator, typename StateInteger, std::size_t num_fused_qubits, typename Complex>
      inline auto adj_global_phase_coeff(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array<StateInteger, num_fused_qubits> const& fused_qubit_masks, std::array<StateInteger, num_fused_qubits + 1u> const& fused_index_masks,
        Complex const& phase_coefficient) // exp(i theta) = cos(theta) + i sin(theta)
      -> void
      {
        using std::conj;
        ::ket::gate::fused::global_phase_coeff(first, fused_index_wo_qubits, fused_qubit_masks, fused_index_masks, conj(phase_coefficient));
      }

      template <typename RandomAccessIterator, typename StateInteger, std::size_t num_fused_qubits, typename Real>
      inline auto global_phase(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array<StateInteger, num_fused_qubits> const& fused_qubit_masks, std::array<StateInteger, num_fused_qubits + 1u> const& fused_index_masks,
        Real const phase)
      -> void
      {
        using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
        ::ket::gate::fused::global_phase_coeff(
          first, fused_index_wo_qubits, fused_qubit_masks, fused_index_masks, ::ket::utility::exp_i<complex_type>(phase));
      }

      template <typename RandomAccessIterator, typename StateInteger, std::size_t num_fused_qubits, typename Real>
      inline auto adj_global_phase(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array<StateInteger, num_fused_qubits> const& fused_qubit_masks, std::array<StateInteger, num_fused_qubits + 1u> const& fused_index_masks,
        Real const phase)
      -> void
      { ::ket::gate::fused::global_phase(first, fused_index_wo_qubits, fused_qubit_masks, fused_index_masks, -phase); }

      namespace runtime
      {
        // global_phase_coeff
        namespace ranges
        {
          template <typename RandomAccessIterator, typename StateInteger, typename StateIntegersRange1, typename StateIntegersRange2, typename Complex>
          inline auto global_phase_coeff(
            RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
            StateIntegersRange1 const& fused_qubit_masks, StateIntegersRange2 const& fused_index_masks,
            Complex const& phase_coefficient) // exp(i theta) = cos(theta) + i sin(theta)
          -> std::enable_if_t<std::is_same<Complex, typename std::iterator_traits<RandomAccessIterator>::value_type>::value, void>
          {
            static_assert(std::is_unsigned<StateInteger>::value, "The StateInteger should be unsigned");

            using std::begin;
            using std::end;
            auto const num_fused_qubits = static_cast<std::size_t>(end(fused_qubit_masks) - begin(fused_qubit_masks));
            assert(static_cast<std::size_t>(end(fused_index_masks) - begin(fused_index_masks)) == num_fused_qubits + std::size_t{1u});

            auto const num_fused_indices = ::ket::utility::integer_exp2<std::size_t>(num_fused_qubits);
            for (auto fused_index = std::size_t{0u}; fused_index < num_fused_indices; ++fused_index)
            {
              auto const iter
                = first
                  + ::ket::gate::utility::ranges::index_with_qubits(
                      fused_index_wo_qubits, fused_index, fused_qubit_masks, fused_index_masks);
              *iter *= phase_coefficient;
            }
          }

          template <typename RandomAccessIterator, typename StateInteger, typename StateIntegersRange1, typename StateIntegersRange2, typename Complex>
          inline auto adj_global_phase_coeff(
            RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
            StateIntegersRange1 const& fused_qubit_masks, StateIntegersRange2 const& fused_index_masks,
            Complex const& phase_coefficient) // exp(i theta) = cos(theta) + i sin(theta)
          -> std::enable_if_t<std::is_same<Complex, typename std::iterator_traits<RandomAccessIterator>::value_type>::value, void>
          {
            using std::conj;
            ::ket::gate::fused::runtime::ranges::global_phase_coeff(
              first, fused_index_wo_qubits, fused_qubit_masks, fused_index_masks, conj(phase_coefficient));
          }

          template <typename RandomAccessIterator, typename StateInteger, typename StateIntegersRange1, typename StateIntegersRange2, typename Real>
          inline auto global_phase(
            RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
            StateIntegersRange1 const& fused_qubit_masks, StateIntegersRange2 const& fused_index_masks,
            Real const phase)
          -> std::enable_if_t<std::is_same<Real, ::ket::utility::meta::real_t<typename std::iterator_traits<RandomAccessIterator>::value_type>>::value, void>
          {
            using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
            ::ket::gate::fused::runtime::ranges::global_phase_coeff(
              first, fused_index_wo_qubits, fused_qubit_masks, fused_index_masks, ::ket::utility::exp_i<complex_type>(phase));
          }

          template <typename RandomAccessIterator, typename StateInteger, typename StateIntegersRange1, typename StateIntegersRange2, typename Real>
          inline auto adj_global_phase(
            RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
            StateIntegersRange1 const& fused_qubit_masks, StateIntegersRange2 const& fused_index_masks,
            Real const phase)
          -> std::enable_if_t<std::is_same<Real, ::ket::utility::meta::real_t<typename std::iterator_traits<RandomAccessIterator>::value_type>>::value, void>
          { ::ket::gate::fused::runtime::ranges::global_phase(first, fused_index_wo_qubits, fused_qubit_masks, fused_index_masks, -phase); }
        } // namespace ranges

        template <typename RandomAccessIterator, typename StateInteger, typename StateIntegerIterator1, typename StateIntegerIterator2, typename Complex>
        inline auto global_phase_coeff(
          RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
          StateIntegerIterator1 const fused_qubit_mask_first, StateIntegerIterator1 const fused_qubit_mask_last,
          StateIntegerIterator2 const fused_index_mask_first, StateIntegerIterator2 const fused_index_mask_last,
          Complex const& phase_coefficient) // exp(i theta) = cos(theta) + i sin(theta)
        -> void
        {
          ::ket::gate::fused::runtime::ranges::global_phase_coeff(
            first, fused_index_wo_qubits,
            boost::make_iterator_range(fused_qubit_mask_first, fused_qubit_mask_last),
            boost::make_iterator_range(fused_index_mask_first, fused_index_mask_last),
            phase_coefficient);
        }

        template <typename RandomAccessIterator, typename StateInteger, typename StateIntegerIterator1, typename StateIntegerIterator2, typename Complex>
        inline auto adj_global_phase_coeff(
          RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
          StateIntegerIterator1 const fused_qubit_mask_first, StateIntegerIterator1 const fused_qubit_mask_last,
          StateIntegerIterator2 const fused_index_mask_first, StateIntegerIterator2 const fused_index_mask_last,
          Complex const& phase_coefficient) // exp(i theta) = cos(theta) + i sin(theta)
        -> void
        {
          ::ket::gate::fused::runtime::ranges::adj_global_phase_coeff(
            first, fused_index_wo_qubits,
            boost::make_iterator_range(fused_qubit_mask_first, fused_qubit_mask_last),
            boost::make_iterator_range(fused_index_mask_first, fused_index_mask_last),
            phase_coefficient);
        }

        template <typename RandomAccessIterator, typename StateInteger, typename StateIntegerIterator1, typename StateIntegerIterator2, typename Real>
        inline auto global_phase(
          RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
          StateIntegerIterator1 const fused_qubit_mask_first, StateIntegerIterator1 const fused_qubit_mask_last,
          StateIntegerIterator2 const fused_index_mask_first, StateIntegerIterator2 const fused_index_mask_last,
          Real const phase)
        -> void
        {
          using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
          ::ket::gate::fused::runtime::global_phase_coeff(
            first, fused_index_wo_qubits,
            fused_qubit_mask_first, fused_qubit_mask_last,
            fused_index_mask_first, fused_index_mask_last,
            ::ket::utility::exp_i<complex_type>(phase));
        }

        template <typename RandomAccessIterator, typename StateInteger, typename StateIntegerIterator1, typename StateIntegerIterator2, typename Real>
        inline auto adj_global_phase(
          RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
          StateIntegerIterator1 const fused_qubit_mask_first, StateIntegerIterator1 const fused_qubit_mask_last,
          StateIntegerIterator2 const fused_index_mask_first, StateIntegerIterator2 const fused_index_mask_last,
          Real const phase)
        -> void
        { ::ket::gate::fused::runtime::global_phase(first, fused_index_wo_qubits, fused_qubit_mask_first, fused_qubit_mask_last, fused_index_mask_first, fused_index_mask_last, -phase); }
      } // namespace runtime
# endif // KET_USE_BIT_MASKS_EXPLICITLY
    } // namespace fused
  } // namespace gate
} // namespace ket

#endif // KET_GATE_FUSED_GLOBAL_PHASE_HPP
