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
      constexpr auto phase_shift_table_max_num_bits = ::bra::bit_integer_type{16u};

      struct phase_shift_table
      {
        std::vector< ::bra::complex_type > coefficients;
        ::bra::bit_integer_type first_bit;
        ::bra::state_integer_type index_mask;
      };

      inline auto phase_shift_table_num_bits(::bra::state_integer_type const operated_mask)
      -> ::bra::bit_integer_type
      {
        if (operated_mask == ::bra::state_integer_type{0u})
          return ::bra::bit_integer_type{0u};

        auto first_bit = ::bra::bit_integer_type{0u};
        while ((operated_mask bitand (::bra::state_integer_type{1u} << first_bit)) == ::bra::state_integer_type{0u})
          ++first_bit;

        auto num_bits = ::bra::bit_integer_type{0u};
        for (auto shifted_mask = operated_mask >> first_bit;
             shifted_mask != ::bra::state_integer_type{0u}; shifted_mask >>= 1u)
          ++num_bits;
        return num_bits;
      }

      inline auto phase_shift_table_index(
        phase_shift_table const& phase_shift_table, ::bra::state_integer_type const fused_index)
      -> ::bra::state_integer_type
      {
        return (fused_index >> phase_shift_table.first_bit) bitand phase_shift_table.index_mask;
      }

      inline auto phase_shift_table_num_bits(
        std::vector< ::bra::fused_gate::phase_shift_term > const& phase_shift_terms)
      -> ::bra::bit_integer_type
      {
        auto operated_mask = ::bra::state_integer_type{0u};
        for (auto const& phase_shift_term: phase_shift_terms)
          operated_mask |= phase_shift_term.control_mask;
        return ::bra::fused_gate::apply_phase_shift_terms_detail::phase_shift_table_num_bits(operated_mask);
      }

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

        auto const num_bits
          = ::bra::fused_gate::apply_phase_shift_terms_detail::phase_shift_table_num_bits(operated_mask);
        if (num_bits > ::bra::fused_gate::apply_phase_shift_terms_detail::phase_shift_table_max_num_bits)
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

      inline auto make_phase_shift_tables(
        std::vector< ::bra::fused_gate::phase_shift_term > const& phase_shift_terms)
      -> std::vector<phase_shift_table>
      {
        auto result = std::vector<phase_shift_table>{};
        auto table_terms = std::vector< ::bra::fused_gate::phase_shift_term >{};
        auto operated_mask = ::bra::state_integer_type{0u};
        for (auto const& phase_shift_term: phase_shift_terms)
        {
          auto const new_operated_mask = operated_mask bitor phase_shift_term.control_mask;
          if (not table_terms.empty()
              and ::bra::fused_gate::apply_phase_shift_terms_detail::phase_shift_table_num_bits(new_operated_mask)
                    > ::bra::fused_gate::apply_phase_shift_terms_detail::phase_shift_table_max_num_bits)
          {
            result.push_back(
              ::bra::fused_gate::apply_phase_shift_terms_detail::make_phase_shift_table(table_terms));
            table_terms.clear();
            operated_mask = ::bra::state_integer_type{0u};
          }

          if (::bra::fused_gate::apply_phase_shift_terms_detail::phase_shift_table_num_bits(
                phase_shift_term.control_mask)
              > ::bra::fused_gate::apply_phase_shift_terms_detail::phase_shift_table_max_num_bits)
            return std::vector<phase_shift_table>{};
          table_terms.push_back(phase_shift_term);
          operated_mask |= phase_shift_term.control_mask;
        }
        if (not table_terms.empty())
          result.push_back(
            ::bra::fused_gate::apply_phase_shift_terms_detail::make_phase_shift_table(table_terms));
        return result;
      }

      inline auto phase_shift_table_num_tables(
        std::vector< ::bra::fused_gate::phase_shift_term > const& phase_shift_terms)
      -> std::size_t
      {
        if (phase_shift_terms.empty())
          return std::size_t{0u};

        auto result = std::size_t{1u};
        auto operated_mask = ::bra::state_integer_type{0u};
        for (auto const& phase_shift_term: phase_shift_terms)
        {
          if (::bra::fused_gate::apply_phase_shift_terms_detail::phase_shift_table_num_bits(
                phase_shift_term.control_mask)
              > ::bra::fused_gate::apply_phase_shift_terms_detail::phase_shift_table_max_num_bits)
            return std::size_t{0u};

          auto const new_operated_mask = operated_mask bitor phase_shift_term.control_mask;
          if (operated_mask != ::bra::state_integer_type{0u}
              and ::bra::fused_gate::apply_phase_shift_terms_detail::phase_shift_table_num_bits(new_operated_mask)
                    > ::bra::fused_gate::apply_phase_shift_terms_detail::phase_shift_table_max_num_bits)
          {
            ++result;
            operated_mask = ::bra::state_integer_type{0u};
          }
          operated_mask |= phase_shift_term.control_mask;
        }
        return result;
      }

      inline auto phase_coefficient(
        std::vector<phase_shift_table> const& phase_shift_tables,
        ::bra::state_integer_type const fused_index) -> ::bra::complex_type
      {
        auto result = ::bra::complex_type{1.0, 0.0};
        for (auto const& phase_shift_table: phase_shift_tables)
          result *= phase_shift_table.coefficients[static_cast<std::size_t>(
            ::bra::fused_gate::apply_phase_shift_terms_detail::phase_shift_table_index(
              phase_shift_table, fused_index))];
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
              ::bra::fused_gate::apply_phase_shift_terms_detail::phase_shift_table_index(
                phase_shift_table, fused_index))];
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
      inline auto apply_one_with_common_control(
        Iterator const first, ::bra::state_integer_type const fused_index_wo_qubits,
        QubitsRange1 const& unsorted_fused_qubits_or_masks,
        QubitsRange2 const& sorted_fused_qubits_with_sentinel_or_index_masks,
        bool const is_fused_index_identity,
        ::bra::state_integer_type const common_control_mask,
        ::bra::state_integer_type const lower_bits_mask,
        ::bra::state_integer_type const upper_bits_mask,
        std::vector<phase_shift_table> const& phase_shift_tables,
        ::bra::state_integer_type const index_wo_control) -> void
      {
        auto const fused_index
          = ((index_wo_control bitand upper_bits_mask) << 1u)
            bitor (index_wo_control bitand lower_bits_mask)
            bitor common_control_mask;
        auto const coefficient
          = ::bra::fused_gate::apply_phase_shift_terms_detail::phase_coefficient(
              phase_shift_tables, fused_index);
        if (coefficient == ::bra::complex_type{1.0, 0.0})
          return;

        auto const index
          = is_fused_index_identity
            ? fused_index
            : ::ket::gate::utility::ranges::index_with_qubits(
                fused_index_wo_qubits, fused_index,
                unsorted_fused_qubits_or_masks,
                sorted_fused_qubits_with_sentinel_or_index_masks);
        *(first + index) *= coefficient;
      }
    } // namespace apply_phase_shift_terms_detail

    template <typename Iterator, typename QubitsRange1, typename QubitsRange2>
    inline auto apply_controlled_phase_shift_terms(
      Iterator const first, ::bra::state_integer_type const fused_index_wo_qubits,
      QubitsRange1 const& unsorted_fused_qubits_or_masks,
      QubitsRange2 const& sorted_fused_qubits_with_sentinel_or_index_masks,
      ::bra::bit_integer_type const common_control,
      std::vector< ::bra::fused_gate::phase_shift_term > const& phase_shift_terms) -> void
    {
      auto const num_fused_qubits
        = static_cast< ::bra::bit_integer_type >(boost::size(unsorted_fused_qubits_or_masks));
      auto const is_fused_index_identity
        = ::bra::fused_gate::apply_phase_shift_terms_detail::is_fused_index_identity(
            fused_index_wo_qubits, unsorted_fused_qubits_or_masks);
      auto const common_control_mask
        = ::ket::utility::integer_exp2< ::bra::state_integer_type >(common_control);
      auto const lower_bits_mask = common_control_mask - ::bra::state_integer_type{1u};
      auto const upper_bits_mask = compl lower_bits_mask;
      auto const count
        = ::ket::utility::integer_exp2< ::bra::state_integer_type >(
            num_fused_qubits - ::bra::bit_integer_type{1u});
      auto const phase_shift_tables
        = ::bra::fused_gate::apply_phase_shift_terms_detail::make_phase_shift_tables(phase_shift_terms);
      for (auto index_wo_control = ::bra::state_integer_type{0u}; index_wo_control < count; ++index_wo_control)
        ::bra::fused_gate::apply_phase_shift_terms_detail::apply_one_with_common_control(
          first, fused_index_wo_qubits,
          unsorted_fused_qubits_or_masks, sorted_fused_qubits_with_sentinel_or_index_masks,
          is_fused_index_identity,
          common_control_mask, lower_bits_mask, upper_bits_mask,
          phase_shift_tables, index_wo_control);
    }

    template <typename ParallelPolicy, typename Iterator, typename QubitsRange1, typename QubitsRange2>
    inline auto apply_controlled_phase_shift_terms(
      ParallelPolicy const parallel_policy, int const thread_index,
      Iterator const first, ::bra::state_integer_type const fused_index_wo_qubits,
      QubitsRange1 const& unsorted_fused_qubits_or_masks,
      QubitsRange2 const& sorted_fused_qubits_with_sentinel_or_index_masks,
      ::bra::bit_integer_type const common_control,
      std::vector< ::bra::fused_gate::phase_shift_term > const& phase_shift_terms) -> void
    {
      auto const num_fused_qubits
        = static_cast< ::bra::bit_integer_type >(boost::size(unsorted_fused_qubits_or_masks));
      auto const is_fused_index_identity
        = ::bra::fused_gate::apply_phase_shift_terms_detail::is_fused_index_identity(
            fused_index_wo_qubits, unsorted_fused_qubits_or_masks);
      auto const common_control_mask
        = ::ket::utility::integer_exp2< ::bra::state_integer_type >(common_control);
      auto const lower_bits_mask = common_control_mask - ::bra::state_integer_type{1u};
      auto const upper_bits_mask = compl lower_bits_mask;
      auto const count
        = ::ket::utility::integer_exp2< ::bra::state_integer_type >(
            num_fused_qubits - ::bra::bit_integer_type{1u});
      auto const phase_shift_tables
        = ::bra::fused_gate::apply_phase_shift_terms_detail::make_phase_shift_tables(phase_shift_terms);
      ::ket::utility::loop_n_in_execute(
        parallel_policy, count, thread_index,
        [first, fused_index_wo_qubits,
         &unsorted_fused_qubits_or_masks, &sorted_fused_qubits_with_sentinel_or_index_masks,
         is_fused_index_identity,
         common_control_mask, lower_bits_mask, upper_bits_mask,
         &phase_shift_tables](::bra::state_integer_type const index_wo_control, int const)
        {
          ::bra::fused_gate::apply_phase_shift_terms_detail::apply_one_with_common_control(
            first, fused_index_wo_qubits,
            unsorted_fused_qubits_or_masks, sorted_fused_qubits_with_sentinel_or_index_masks,
            is_fused_index_identity,
            common_control_mask, lower_bits_mask, upper_bits_mask,
            phase_shift_tables, index_wo_control);
        });
    }

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
