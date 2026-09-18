#ifndef BRA_FUSED_GATE_APPLY_PHASE_SHIFT_TERMS_HPP
# define BRA_FUSED_GATE_APPLY_PHASE_SHIFT_TERMS_HPP

# include <cstddef>
# include <iterator>
# include <type_traits>
# include <vector>

# include <boost/range/begin.hpp>
# include <boost/range/end.hpp>
# include <boost/range/size.hpp>

# include <ket/gate/utility/index_with_qubits.hpp>
# include <ket/utility/integer_exp2.hpp>
# include <ket/utility/loop_n.hpp>
# include <ket/utility/meta/ranges.hpp>

# include <bra/types.hpp>
# include <bra/fused_gate/fused_gate.hpp>


namespace bra
{
  namespace fused_gate
  {
    namespace apply_phase_shift_terms_detail
    {
      struct phase_shift_table
      {
        std::vector< ::bra::complex_type > coefficients;
        ::bra::bit_integer_type first_bit;
        ::bra::state_integer_type index_mask;
      };

      inline auto make_phase_shift_table(
        std::vector< ::bra::fused_gate::phase_shift_term > const& phase_shift_terms)
      -> phase_shift_table
      {
        auto operated_mask = ::bra::state_integer_type{0u};
        for (auto const& phase_shift_term: phase_shift_terms)
          operated_mask |= phase_shift_term.control_mask;

        auto first_bit = ::bra::bit_integer_type{0u};
        if (operated_mask != ::bra::state_integer_type{0u})
          while ((operated_mask bitand (::bra::state_integer_type{1u} << first_bit)) == ::bra::state_integer_type{0u})
            ++first_bit;

        auto num_bits = ::bra::bit_integer_type{0u};
        for (auto shifted_mask = operated_mask >> first_bit;
             shifted_mask != ::bra::state_integer_type{0u}; shifted_mask >>= 1u)
          ++num_bits;
        if (num_bits > ::bra::bit_integer_type{16u})
          return phase_shift_table{};

        auto const table_size
          = ::ket::utility::integer_exp2< ::bra::state_integer_type >(num_bits);
        auto result = phase_shift_table{
          std::vector< ::bra::complex_type >(
            static_cast<std::size_t>(table_size), ::bra::complex_type{1.0, 0.0}),
          first_bit, table_size - ::bra::state_integer_type{1u}};
        for (auto const& phase_shift_term: phase_shift_terms)
        {
          auto const control_mask = phase_shift_term.control_mask >> first_bit;
          for (auto index = ::bra::state_integer_type{0u}; index < table_size; ++index)
            if ((index bitand control_mask) == control_mask)
              result.coefficients[static_cast<std::size_t>(index)] *= phase_shift_term.phase_coefficient;
        }
        return result;
      }

      template <typename QubitsRange>
      inline auto is_fused_index_identity(
        ::bra::state_integer_type const fused_index_wo_qubits,
        QubitsRange const& unsorted_fused_qubits)
      -> typename std::enable_if<
           std::is_same<
             typename std::decay< ::ket::utility::meta::range_value_t<QubitsRange> >::type,
             ::bra::qubit_type>::value,
           bool>::type
      {
        if (fused_index_wo_qubits != ::bra::state_integer_type{0u})
          return false;

        auto fused_qubit_index = ::bra::bit_integer_type{0u};
        for (auto const fused_qubit: unsorted_fused_qubits)
          if (fused_qubit != ::bra::qubit_type{fused_qubit_index++})
            return false;
        return true;
      }

      template <typename QubitMasksRange>
      inline auto is_fused_index_identity(
        ::bra::state_integer_type const fused_index_wo_qubits,
        QubitMasksRange const& qubit_masks)
      -> typename std::enable_if<
           std::is_same<
             typename std::decay< ::ket::utility::meta::range_value_t<QubitMasksRange> >::type,
             ::bra::state_integer_type>::value,
           bool>::type
      {
        if (fused_index_wo_qubits != ::bra::state_integer_type{0u})
          return false;

        auto expected_mask = ::bra::state_integer_type{1u};
        for (auto const qubit_mask: qubit_masks)
        {
          if (qubit_mask != expected_mask)
            return false;
          expected_mask <<= 1u;
        }
        return true;
      }

      template <typename Iterator, typename QubitsRange1, typename QubitsRange2>
      inline auto apply_one(
        Iterator const first, ::bra::state_integer_type const fused_index_wo_qubits,
        QubitsRange1 const& unsorted_fused_qubits_or_masks,
        QubitsRange2 const& sorted_fused_qubits_with_sentinel_or_index_masks,
        bool const is_fused_index_identity,
        std::vector< ::bra::fused_gate::phase_shift_term > const& phase_shift_terms,
        ::bra::state_integer_type const fused_index) -> void
      {
        auto phase_coefficient = ::bra::complex_type{1.0, 0.0};
        for (auto const& phase_shift_term: phase_shift_terms)
          if ((fused_index bitand phase_shift_term.control_mask) == phase_shift_term.control_mask)
            phase_coefficient *= phase_shift_term.phase_coefficient;

        if (phase_coefficient == ::bra::complex_type{1.0, 0.0})
          return;

        auto const index
          = is_fused_index_identity
            ? fused_index
            : ::ket::gate::utility::ranges::index_with_qubits(
                fused_index_wo_qubits, fused_index,
                unsorted_fused_qubits_or_masks,
                sorted_fused_qubits_with_sentinel_or_index_masks);
        *(first + index) *= phase_coefficient;
      }

      template <typename Iterator, typename QubitsRange1, typename QubitsRange2>
      inline auto apply_one(
        Iterator const first, ::bra::state_integer_type const fused_index_wo_qubits,
        QubitsRange1 const& unsorted_fused_qubits_or_masks,
        QubitsRange2 const& sorted_fused_qubits_with_sentinel_or_index_masks,
        bool const is_fused_index_identity,
        phase_shift_table const& phase_shift_table,
        ::bra::state_integer_type const fused_index) -> void
      {
        auto const phase_coefficient
          = phase_shift_table.coefficients[static_cast<std::size_t>(
              (fused_index >> phase_shift_table.first_bit) bitand phase_shift_table.index_mask)];
        if (phase_coefficient == ::bra::complex_type{1.0, 0.0})
          return;

        auto const index
          = is_fused_index_identity
            ? fused_index
            : ::ket::gate::utility::ranges::index_with_qubits(
                fused_index_wo_qubits, fused_index,
                unsorted_fused_qubits_or_masks,
                sorted_fused_qubits_with_sentinel_or_index_masks);
        *(first + index) *= phase_coefficient;
      }
    } // namespace apply_phase_shift_terms_detail

    template <typename Iterator, typename QubitsRange1, typename QubitsRange2>
    inline auto apply_phase_shift_terms(
      Iterator const first, ::bra::state_integer_type const fused_index_wo_qubits,
      QubitsRange1 const& unsorted_fused_qubits_or_masks,
      QubitsRange2 const& sorted_fused_qubits_with_sentinel_or_index_masks,
      std::vector< ::bra::fused_gate::phase_shift_term > const& phase_shift_terms) -> void
    {
      auto const is_fused_index_identity
        = ::bra::fused_gate::apply_phase_shift_terms_detail::is_fused_index_identity(
            fused_index_wo_qubits, unsorted_fused_qubits_or_masks);
      auto const count = ::ket::utility::integer_exp2< ::bra::state_integer_type >(
        static_cast< ::bra::bit_integer_type >(boost::size(unsorted_fused_qubits_or_masks)));
      auto const phase_shift_table
        = ::bra::fused_gate::apply_phase_shift_terms_detail::make_phase_shift_table(phase_shift_terms);
      if (not phase_shift_table.coefficients.empty())
      {
        for (auto fused_index = ::bra::state_integer_type{0u}; fused_index < count; ++fused_index)
          ::bra::fused_gate::apply_phase_shift_terms_detail::apply_one(
            first, fused_index_wo_qubits,
            unsorted_fused_qubits_or_masks, sorted_fused_qubits_with_sentinel_or_index_masks,
            is_fused_index_identity, phase_shift_table, fused_index);
        return;
      }

      for (auto fused_index = ::bra::state_integer_type{0u}; fused_index < count; ++fused_index)
        ::bra::fused_gate::apply_phase_shift_terms_detail::apply_one(
          first, fused_index_wo_qubits,
          unsorted_fused_qubits_or_masks, sorted_fused_qubits_with_sentinel_or_index_masks,
          is_fused_index_identity, phase_shift_terms, fused_index);
    }

    template <typename ParallelPolicy, typename Iterator, typename QubitsRange1, typename QubitsRange2>
    inline auto apply_phase_shift_terms(
      ParallelPolicy const parallel_policy, int const thread_index,
      Iterator const first, ::bra::state_integer_type const fused_index_wo_qubits,
      QubitsRange1 const& unsorted_fused_qubits_or_masks,
      QubitsRange2 const& sorted_fused_qubits_with_sentinel_or_index_masks,
      std::vector< ::bra::fused_gate::phase_shift_term > const& phase_shift_terms) -> void
    {
      auto const is_fused_index_identity
        = ::bra::fused_gate::apply_phase_shift_terms_detail::is_fused_index_identity(
            fused_index_wo_qubits, unsorted_fused_qubits_or_masks);
      auto const count = ::ket::utility::integer_exp2< ::bra::state_integer_type >(
        static_cast< ::bra::bit_integer_type >(boost::size(unsorted_fused_qubits_or_masks)));
      auto const phase_shift_table
        = ::bra::fused_gate::apply_phase_shift_terms_detail::make_phase_shift_table(phase_shift_terms);
      if (not phase_shift_table.coefficients.empty())
      {
        ::ket::utility::loop_n_in_execute(
          parallel_policy, count, thread_index,
          [first, fused_index_wo_qubits,
           &unsorted_fused_qubits_or_masks, &sorted_fused_qubits_with_sentinel_or_index_masks,
           is_fused_index_identity, &phase_shift_table](::bra::state_integer_type const fused_index, int const)
          {
            ::bra::fused_gate::apply_phase_shift_terms_detail::apply_one(
              first, fused_index_wo_qubits,
              unsorted_fused_qubits_or_masks, sorted_fused_qubits_with_sentinel_or_index_masks,
              is_fused_index_identity, phase_shift_table, fused_index);
          });
        return;
      }

      ::ket::utility::loop_n_in_execute(
        parallel_policy, count, thread_index,
        [first, fused_index_wo_qubits,
         &unsorted_fused_qubits_or_masks, &sorted_fused_qubits_with_sentinel_or_index_masks,
         is_fused_index_identity, &phase_shift_terms](::bra::state_integer_type const fused_index, int const)
        {
          ::bra::fused_gate::apply_phase_shift_terms_detail::apply_one(
            first, fused_index_wo_qubits,
            unsorted_fused_qubits_or_masks, sorted_fused_qubits_with_sentinel_or_index_masks,
            is_fused_index_identity, phase_shift_terms, fused_index);
        });
    }
  } // namespace fused_gate
} // namespace bra


#endif // BRA_FUSED_GATE_APPLY_PHASE_SHIFT_TERMS_HPP
