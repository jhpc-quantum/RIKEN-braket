#ifndef BRA_FUSED_GATE_APPLY_FUSED_GATES_HPP
# define BRA_FUSED_GATE_APPLY_FUSED_GATES_HPP

# ifdef BRA_PROFILE_PHASE_SHIFT_BATCHES
#   include <chrono>
#   include <iostream>
# endif // BRA_PROFILE_PHASE_SHIFT_BATCHES
# include <cstddef>
# include <iterator>
# include <limits>
# include <utility>
# include <vector>

# include <ket/utility/integer_exp2.hpp>
# include <ket/utility/loop_n.hpp>

# include <bra/types.hpp>
# include <bra/fused_gate/apply_hadamard_and_phase_shift_terms.hpp>
# include <bra/fused_gate/apply_phase_shift_terms.hpp>


namespace bra
{
  namespace fused_gate
  {
    namespace apply_fused_gates_detail
    {
      constexpr auto phase_shift_tables_max_num_tables = std::size_t{4u};

      struct indexed_phase_shift_term
      {
        ::bra::fused_gate::phase_shift_term phase_shift_term;
        std::size_t gate_index;
      };

      struct phase_shift_star
      {
        ::bra::bit_integer_type common_control;
        std::vector< ::bra::fused_gate::phase_shift_term > phase_shift_terms;
        std::vector<std::size_t> gate_indices;
      };

      template <typename FusedGateIterator>
      inline auto append_phase_shift_block(
        FusedGateIterator const first, FusedGateIterator const last,
        std::vector<indexed_phase_shift_term>& indexed_phase_shift_terms,
        std::size_t& num_gates,
        std::vector< ::bra::bit_integer_type > const& to_qubit_index_in_fused_gates,
        ::bra::state_integer_type const unit_qubit_value) -> FusedGateIterator
      {
        auto iter = first;
        num_gates = std::size_t{0u};
        while (iter != last)
        {
          auto phase_shift_terms = std::vector< ::bra::fused_gate::phase_shift_term >{};
          if (not (*iter)->append_phase_shift_term(
                phase_shift_terms, to_qubit_index_in_fused_gates, unit_qubit_value))
            break;
          for (auto const& phase_shift_term: phase_shift_terms)
            indexed_phase_shift_terms.push_back(indexed_phase_shift_term{phase_shift_term, num_gates});
          ++num_gates;
          ++iter;
        }
        return iter;
      }

      inline auto make_phase_shift_stars(
        std::vector<indexed_phase_shift_term> const& indexed_phase_shift_terms,
        std::size_t const num_gates,
        std::vector<bool>& selected_gates) -> std::vector<phase_shift_star>
      {
        selected_gates.assign(num_gates, false);
        auto remaining_terms = std::vector<bool>(indexed_phase_shift_terms.size(), true);
        auto result = std::vector<phase_shift_star>{};
        constexpr auto num_state_integer_bits
          = static_cast< ::bra::bit_integer_type >(
              std::numeric_limits< ::bra::state_integer_type >::digits);

        while (true)
        {
          auto counts = std::vector<std::size_t>(num_state_integer_bits, std::size_t{0u});
          for (auto index = std::size_t{0u}; index < indexed_phase_shift_terms.size(); ++index)
          {
            if (not remaining_terms[index])
              continue;
            auto const control_mask = indexed_phase_shift_terms[index].phase_shift_term.control_mask;
            for (auto bit = ::bra::bit_integer_type{0u}; bit < num_state_integer_bits; ++bit)
              if ((control_mask bitand (::bra::state_integer_type{1u} << bit)) != ::bra::state_integer_type{0u})
                ++counts[bit];
          }

          auto common_control = ::bra::bit_integer_type{0u};
          auto max_count = std::size_t{0u};
          for (auto bit = ::bra::bit_integer_type{0u}; bit < num_state_integer_bits; ++bit)
            if (counts[bit] > max_count)
            {
              common_control = bit;
              max_count = counts[bit];
            }
          if (max_count < std::size_t{3u})
            break;

          auto const common_control_mask
            = ::bra::state_integer_type{1u} << common_control;
          auto star = phase_shift_star{common_control, {}, {}};
          auto term_indices = std::vector<std::size_t>{};
          for (auto index = std::size_t{0u}; index < indexed_phase_shift_terms.size(); ++index)
          {
            if (not remaining_terms[index]
                or (indexed_phase_shift_terms[index].phase_shift_term.control_mask bitand common_control_mask)
                     == ::bra::state_integer_type{0u})
              continue;
            auto phase_shift_term = indexed_phase_shift_terms[index].phase_shift_term;
            phase_shift_term.control_mask &= compl common_control_mask;
            star.phase_shift_terms.push_back(phase_shift_term);
            star.gate_indices.push_back(indexed_phase_shift_terms[index].gate_index);
            term_indices.push_back(index);
          }

          auto const num_tables
            = ::bra::fused_gate::apply_phase_shift_terms_detail::phase_shift_table_num_tables(
                star.phase_shift_terms);
          if (num_tables == std::size_t{0u}
              or num_tables
                   > ::bra::fused_gate::apply_fused_gates_detail::phase_shift_tables_max_num_tables)
            break;

          for (auto const index: term_indices)
            remaining_terms[index] = false;
          for (auto const gate_index: star.gate_indices)
            selected_gates[gate_index] = true;
          result.push_back(std::move(star));
        }
        return result;
      }

      template <typename FusedGateIterator>
      inline auto append_common_target_phase_shift_terms(
        FusedGateIterator const first, FusedGateIterator const last,
        std::vector< ::bra::fused_gate::phase_shift_term >& phase_shift_terms,
        ::bra::bit_integer_type const target_qubit,
        std::vector< ::bra::bit_integer_type > const& to_qubit_index_in_fused_gates,
        ::bra::state_integer_type const unit_qubit_value) -> FusedGateIterator
      {
        auto const target_qubit_mask
          = ::ket::utility::integer_exp2< ::bra::state_integer_type >(target_qubit);
        auto iter = first;
        while (iter != last)
        {
          auto new_terms = std::vector< ::bra::fused_gate::phase_shift_term >{};
          if (not (*iter)->append_phase_shift_term(
                new_terms, to_qubit_index_in_fused_gates, unit_qubit_value))
            break;
          if (new_terms.empty())
          {
            ++iter;
            continue;
          }

          auto is_compatible = true;
          for (auto& phase_shift_term: new_terms)
          {
            if ((phase_shift_term.control_mask bitand target_qubit_mask) == ::bra::state_integer_type{0u})
            {
              is_compatible = false;
              break;
            }
            phase_shift_term.control_mask &= compl target_qubit_mask;
          }
          if (not is_compatible)
            break;

          auto const old_size = phase_shift_terms.size();
          phase_shift_terms.insert(phase_shift_terms.end(), new_terms.begin(), new_terms.end());
          auto const num_tables
            = ::bra::fused_gate::apply_phase_shift_terms_detail::phase_shift_table_num_tables(phase_shift_terms);
          if (num_tables == std::size_t{0u}
              or num_tables
                   > ::bra::fused_gate::apply_fused_gates_detail::phase_shift_tables_max_num_tables)
          {
            phase_shift_terms.resize(old_size);
            break;
          }
          ++iter;
        }
        return iter;
      }

      template <typename FusedGateIterator>
      inline auto append_table_compatible_phase_shift_terms(
        FusedGateIterator const first, FusedGateIterator const last,
        std::vector< ::bra::fused_gate::phase_shift_term >& phase_shift_terms,
        std::vector< ::bra::bit_integer_type > const& to_qubit_index_in_fused_gates,
        ::bra::state_integer_type const unit_qubit_value) -> FusedGateIterator
      {
        auto operated_mask = ::bra::state_integer_type{0u};
        auto iter = first;
        while (iter != last)
        {
          auto const old_size = phase_shift_terms.size();
          if (not (*iter)->append_phase_shift_term(
                phase_shift_terms, to_qubit_index_in_fused_gates, unit_qubit_value))
            break;

          auto new_operated_mask = operated_mask;
          for (auto index = old_size; index < phase_shift_terms.size(); ++index)
            new_operated_mask |= phase_shift_terms[index].control_mask;
          if (::bra::fused_gate::apply_phase_shift_terms_detail::phase_shift_table_num_bits(new_operated_mask)
              > ::bra::fused_gate::apply_phase_shift_terms_detail::phase_shift_table_max_num_bits)
          {
            phase_shift_terms.resize(old_size);
            break;
          }
          operated_mask = new_operated_mask;
          ++iter;
        }
        return iter;
      }

      inline auto should_batch(
        std::vector< ::bra::fused_gate::phase_shift_term > const& phase_shift_terms) -> bool
      {
        if (::bra::fused_gate::apply_phase_shift_terms_detail::phase_shift_table_num_bits(phase_shift_terms)
            > ::bra::fused_gate::apply_phase_shift_terms_detail::phase_shift_table_max_num_bits)
          return false;

        auto separate_work_in_quarters = std::size_t{0u};
        for (auto const& phase_shift_term: phase_shift_terms)
        {
          auto const control_mask = phase_shift_term.control_mask;
          if (control_mask == ::bra::state_integer_type{0u})
            separate_work_in_quarters += 4u;
          else if ((control_mask bitand (control_mask - ::bra::state_integer_type{1u})) == ::bra::state_integer_type{0u})
            separate_work_in_quarters += 2u;
          else
            separate_work_in_quarters += 1u;
        }
        return separate_work_in_quarters > 4u;
      }
    } // namespace apply_fused_gates_detail

    template <typename FusedGates, typename First, typename QubitsRange1, typename QubitsRange2>
    inline auto apply_fused_gates(
      FusedGates const& fused_gates,
      First const first, ::bra::state_integer_type const index_wo_qubits,
      QubitsRange1 const& unsorted_fused_qubits_or_masks,
      QubitsRange2 const& sorted_fused_qubits_with_sentinel_or_index_masks,
      std::vector< ::bra::bit_integer_type > const& to_qubit_index_in_fused_gates,
      ::bra::state_integer_type const unit_qubit_value) -> void
    {
      auto gate_iter = fused_gates.begin();
      while (gate_iter != fused_gates.end())
      {
        auto target_qubit = ::bra::bit_integer_type{0u};
        if ((*gate_iter)->get_hadamard_target(
              target_qubit, to_qubit_index_in_fused_gates, unit_qubit_value))
        {
          auto controlled_phase_shift_terms = std::vector< ::bra::fused_gate::phase_shift_term >{};
          auto const next_controlled_gate_iter
            = ::bra::fused_gate::apply_fused_gates_detail::append_common_target_phase_shift_terms(
                std::next(gate_iter), fused_gates.end(), controlled_phase_shift_terms, target_qubit,
                to_qubit_index_in_fused_gates, unit_qubit_value);
          if (not controlled_phase_shift_terms.empty())
          {
            ::bra::fused_gate::apply_hadamard_and_controlled_phase_shift_terms(
              first, index_wo_qubits,
              unsorted_fused_qubits_or_masks, sorted_fused_qubits_with_sentinel_or_index_masks,
              target_qubit, controlled_phase_shift_terms);
            gate_iter = next_controlled_gate_iter;
            continue;
          }

          auto phase_shift_terms = std::vector< ::bra::fused_gate::phase_shift_term >{};
          auto const next_gate_iter
            = ::bra::fused_gate::apply_fused_gates_detail::append_table_compatible_phase_shift_terms(
                std::next(gate_iter), fused_gates.end(), phase_shift_terms,
                to_qubit_index_in_fused_gates, unit_qubit_value);
          if (not phase_shift_terms.empty())
          {
            ::bra::fused_gate::apply_hadamard_and_phase_shift_terms(
              first, index_wo_qubits,
              unsorted_fused_qubits_or_masks, sorted_fused_qubits_with_sentinel_or_index_masks,
              target_qubit, phase_shift_terms);
            gate_iter = next_gate_iter;
            continue;
          }
        }

        auto phase_shift_terms = std::vector< ::bra::fused_gate::phase_shift_term >{};
        auto const next_gate_iter
          = ::bra::fused_gate::apply_fused_gates_detail::append_table_compatible_phase_shift_terms(
              gate_iter, fused_gates.end(), phase_shift_terms,
              to_qubit_index_in_fused_gates, unit_qubit_value);

        if (next_gate_iter != gate_iter
            and ::bra::fused_gate::apply_fused_gates_detail::should_batch(phase_shift_terms))
        {
          ::bra::fused_gate::apply_phase_shift_terms(
            first, index_wo_qubits,
            unsorted_fused_qubits_or_masks, sorted_fused_qubits_with_sentinel_or_index_masks,
            phase_shift_terms);
          gate_iter = next_gate_iter;
          continue;
        }

        auto indexed_phase_shift_terms
          = std::vector< ::bra::fused_gate::apply_fused_gates_detail::indexed_phase_shift_term >{};
        auto num_phase_shift_gates = std::size_t{0u};
        auto const phase_shift_block_end
          = ::bra::fused_gate::apply_fused_gates_detail::append_phase_shift_block(
              gate_iter, fused_gates.end(), indexed_phase_shift_terms, num_phase_shift_gates,
              to_qubit_index_in_fused_gates, unit_qubit_value);
        auto selected_phase_shift_gates = std::vector<bool>{};
        auto const phase_shift_stars
          = ::bra::fused_gate::apply_fused_gates_detail::make_phase_shift_stars(
              indexed_phase_shift_terms, num_phase_shift_gates, selected_phase_shift_gates);
        if (not phase_shift_stars.empty())
        {
          for (auto const& phase_shift_star: phase_shift_stars)
            ::bra::fused_gate::apply_controlled_phase_shift_terms(
              first, index_wo_qubits,
              unsorted_fused_qubits_or_masks, sorted_fused_qubits_with_sentinel_or_index_masks,
              phase_shift_star.common_control, phase_shift_star.phase_shift_terms);

          auto phase_shift_gate_iter = gate_iter;
          for (auto gate_index = std::size_t{0u}; gate_index < num_phase_shift_gates;
               ++gate_index, ++phase_shift_gate_iter)
            if (not selected_phase_shift_gates[gate_index])
              (*phase_shift_gate_iter)->call(
                first, index_wo_qubits,
                unsorted_fused_qubits_or_masks, sorted_fused_qubits_with_sentinel_or_index_masks,
                to_qubit_index_in_fused_gates, unit_qubit_value);
          gate_iter = phase_shift_block_end;
          continue;
        }

        (*gate_iter)->call(
          first, index_wo_qubits,
          unsorted_fused_qubits_or_masks, sorted_fused_qubits_with_sentinel_or_index_masks,
          to_qubit_index_in_fused_gates, unit_qubit_value);
        ++gate_iter;
      }
    }

    template <typename FusedGates, typename Executor, typename First, typename QubitsRange1, typename QubitsRange2>
    inline auto apply_fused_gates_in_execute(
      FusedGates const& fused_gates,
      ::ket::utility::policy::parallel<unsigned int> const parallel_policy,
      Executor& executor, int const thread_index,
      First const first, ::bra::state_integer_type const index_wo_qubits,
      QubitsRange1 const& unsorted_fused_qubits_or_masks,
      QubitsRange2 const& sorted_fused_qubits_with_sentinel_or_index_masks,
      std::vector< ::bra::bit_integer_type > const& to_qubit_index_in_fused_gates,
      ::bra::state_integer_type const unit_qubit_value) -> void
    {
# ifdef BRA_PROFILE_PHASE_SHIFT_BATCHES
      auto num_batches = std::size_t{0u};
      auto num_table_batches = std::size_t{0u};
      auto num_fallback_batches = std::size_t{0u};
      auto num_batched_terms = std::size_t{0u};
      auto total_batch_time = std::chrono::steady_clock::duration{};
      auto num_hadamard_phase_batches = std::size_t{0u};
      auto num_hadamard_phase_terms = std::size_t{0u};
      auto total_hadamard_phase_batch_time = std::chrono::steady_clock::duration{};
      auto num_phase_fallback_gates = std::size_t{0u};
      auto total_phase_fallback_time = std::chrono::steady_clock::duration{};
      auto num_other_fallback_gates = std::size_t{0u};
      auto total_other_fallback_time = std::chrono::steady_clock::duration{};
      auto num_phase_star_batches = std::size_t{0u};
      auto num_phase_star_terms = std::size_t{0u};
      auto total_phase_star_time = std::chrono::steady_clock::duration{};
# endif // BRA_PROFILE_PHASE_SHIFT_BATCHES
      auto gate_iter = fused_gates.begin();
      while (gate_iter != fused_gates.end())
      {
        auto target_qubit = ::bra::bit_integer_type{0u};
        if ((*gate_iter)->get_hadamard_target(
              target_qubit, to_qubit_index_in_fused_gates, unit_qubit_value))
        {
          auto controlled_phase_shift_terms = std::vector< ::bra::fused_gate::phase_shift_term >{};
          auto const next_controlled_gate_iter
            = ::bra::fused_gate::apply_fused_gates_detail::append_common_target_phase_shift_terms(
                std::next(gate_iter), fused_gates.end(), controlled_phase_shift_terms, target_qubit,
                to_qubit_index_in_fused_gates, unit_qubit_value);
          if (not controlled_phase_shift_terms.empty())
          {
# ifdef BRA_PROFILE_PHASE_SHIFT_BATCHES
            auto const num_tables
              = ::bra::fused_gate::apply_phase_shift_terms_detail::phase_shift_table_num_tables(
                  controlled_phase_shift_terms);
            auto const start_time
              = thread_index == 0 ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
# endif // BRA_PROFILE_PHASE_SHIFT_BATCHES
            ::bra::fused_gate::apply_hadamard_and_controlled_phase_shift_terms(
              parallel_policy, thread_index,
              first, index_wo_qubits,
              unsorted_fused_qubits_or_masks, sorted_fused_qubits_with_sentinel_or_index_masks,
              target_qubit, controlled_phase_shift_terms);
            ::ket::utility::barrier(parallel_policy, executor);
# ifdef BRA_PROFILE_PHASE_SHIFT_BATCHES
            if (thread_index == 0)
            {
              auto const elapsed_time = std::chrono::steady_clock::now() - start_time;
              ++num_hadamard_phase_batches;
              num_hadamard_phase_terms += controlled_phase_shift_terms.size();
              total_hadamard_phase_batch_time += elapsed_time;
              std::clog
                << "[hadamard-controlled-phase-batch] terms=" << controlled_phase_shift_terms.size()
                << " tables=" << num_tables
                << " elapsed=" << std::chrono::duration<double>{elapsed_time}.count()
                << std::endl;
            }
# endif // BRA_PROFILE_PHASE_SHIFT_BATCHES
            gate_iter = next_controlled_gate_iter;
            continue;
          }

          auto phase_shift_terms = std::vector< ::bra::fused_gate::phase_shift_term >{};
          auto const next_gate_iter
            = ::bra::fused_gate::apply_fused_gates_detail::append_table_compatible_phase_shift_terms(
                std::next(gate_iter), fused_gates.end(), phase_shift_terms,
                to_qubit_index_in_fused_gates, unit_qubit_value);
          if (not phase_shift_terms.empty())
          {
# ifdef BRA_PROFILE_PHASE_SHIFT_BATCHES
            auto const table_num_bits
              = ::bra::fused_gate::apply_phase_shift_terms_detail::phase_shift_table_num_bits(phase_shift_terms);
            auto const start_time
              = thread_index == 0 ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
# endif // BRA_PROFILE_PHASE_SHIFT_BATCHES
            ::bra::fused_gate::apply_hadamard_and_phase_shift_terms(
              parallel_policy, thread_index,
              first, index_wo_qubits,
              unsorted_fused_qubits_or_masks, sorted_fused_qubits_with_sentinel_or_index_masks,
              target_qubit, phase_shift_terms);
            ::ket::utility::barrier(parallel_policy, executor);
# ifdef BRA_PROFILE_PHASE_SHIFT_BATCHES
            if (thread_index == 0)
            {
              auto const elapsed_time = std::chrono::steady_clock::now() - start_time;
              ++num_hadamard_phase_batches;
              num_hadamard_phase_terms += phase_shift_terms.size();
              total_hadamard_phase_batch_time += elapsed_time;
              std::clog
                << "[hadamard-phase-batch] terms=" << phase_shift_terms.size()
                << " table_bits=" << table_num_bits
                << " elapsed=" << std::chrono::duration<double>{elapsed_time}.count()
                << std::endl;
            }
# endif // BRA_PROFILE_PHASE_SHIFT_BATCHES
            gate_iter = next_gate_iter;
            continue;
          }
        }

        auto phase_shift_terms = std::vector< ::bra::fused_gate::phase_shift_term >{};
        auto const next_gate_iter
          = ::bra::fused_gate::apply_fused_gates_detail::append_table_compatible_phase_shift_terms(
              gate_iter, fused_gates.end(), phase_shift_terms,
              to_qubit_index_in_fused_gates, unit_qubit_value);

        if (next_gate_iter != gate_iter
            and ::bra::fused_gate::apply_fused_gates_detail::should_batch(phase_shift_terms))
        {
# ifdef BRA_PROFILE_PHASE_SHIFT_BATCHES
          auto const table_num_bits
            = ::bra::fused_gate::apply_phase_shift_terms_detail::phase_shift_table_num_bits(phase_shift_terms);
          auto const uses_table
            = table_num_bits
              <= ::bra::fused_gate::apply_phase_shift_terms_detail::phase_shift_table_max_num_bits;
          auto const start_time
            = thread_index == 0 ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
# endif // BRA_PROFILE_PHASE_SHIFT_BATCHES
          ::bra::fused_gate::apply_phase_shift_terms(
            parallel_policy, thread_index,
            first, index_wo_qubits,
            unsorted_fused_qubits_or_masks, sorted_fused_qubits_with_sentinel_or_index_masks,
            phase_shift_terms);
          ::ket::utility::barrier(parallel_policy, executor);
# ifdef BRA_PROFILE_PHASE_SHIFT_BATCHES
          if (thread_index == 0)
          {
            auto const elapsed_time = std::chrono::steady_clock::now() - start_time;
            ++num_batches;
            num_table_batches += uses_table ? std::size_t{1u} : std::size_t{0u};
            num_fallback_batches += uses_table ? std::size_t{0u} : std::size_t{1u};
            num_batched_terms += phase_shift_terms.size();
            total_batch_time += elapsed_time;
            std::clog
              << "[phase-batch] terms=" << phase_shift_terms.size()
              << " mode=" << (uses_table ? "table" : "fallback")
              << " table_bits=" << table_num_bits
              << " elapsed=" << std::chrono::duration<double>{elapsed_time}.count()
              << std::endl;
          }
# endif // BRA_PROFILE_PHASE_SHIFT_BATCHES
          gate_iter = next_gate_iter;
          continue;
        }

        auto indexed_phase_shift_terms
          = std::vector< ::bra::fused_gate::apply_fused_gates_detail::indexed_phase_shift_term >{};
        auto num_phase_shift_gates = std::size_t{0u};
        auto const phase_shift_block_end
          = ::bra::fused_gate::apply_fused_gates_detail::append_phase_shift_block(
              gate_iter, fused_gates.end(), indexed_phase_shift_terms, num_phase_shift_gates,
              to_qubit_index_in_fused_gates, unit_qubit_value);
        auto selected_phase_shift_gates = std::vector<bool>{};
        auto const phase_shift_stars
          = ::bra::fused_gate::apply_fused_gates_detail::make_phase_shift_stars(
              indexed_phase_shift_terms, num_phase_shift_gates, selected_phase_shift_gates);
        if (not phase_shift_stars.empty())
        {
          for (auto const& phase_shift_star: phase_shift_stars)
          {
# ifdef BRA_PROFILE_PHASE_SHIFT_BATCHES
            auto const num_tables
              = ::bra::fused_gate::apply_phase_shift_terms_detail::phase_shift_table_num_tables(
                  phase_shift_star.phase_shift_terms);
            auto const start_time
              = thread_index == 0 ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
# endif // BRA_PROFILE_PHASE_SHIFT_BATCHES
            ::bra::fused_gate::apply_controlled_phase_shift_terms(
              parallel_policy, thread_index,
              first, index_wo_qubits,
              unsorted_fused_qubits_or_masks, sorted_fused_qubits_with_sentinel_or_index_masks,
              phase_shift_star.common_control, phase_shift_star.phase_shift_terms);
            ::ket::utility::barrier(parallel_policy, executor);
# ifdef BRA_PROFILE_PHASE_SHIFT_BATCHES
            if (thread_index == 0)
            {
              auto const elapsed_time = std::chrono::steady_clock::now() - start_time;
              ++num_phase_star_batches;
              num_phase_star_terms += phase_shift_star.phase_shift_terms.size();
              total_phase_star_time += elapsed_time;
              std::clog
                << "[phase-star-batch] terms=" << phase_shift_star.phase_shift_terms.size()
                << " tables=" << num_tables
                << " elapsed=" << std::chrono::duration<double>{elapsed_time}.count()
                << std::endl;
            }
# endif // BRA_PROFILE_PHASE_SHIFT_BATCHES
          }

          auto phase_shift_gate_iter = gate_iter;
          for (auto gate_index = std::size_t{0u}; gate_index < num_phase_shift_gates;
               ++gate_index, ++phase_shift_gate_iter)
          {
            if (selected_phase_shift_gates[gate_index])
              continue;
# ifdef BRA_PROFILE_PHASE_SHIFT_BATCHES
            auto const start_time
              = thread_index == 0 ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
# endif // BRA_PROFILE_PHASE_SHIFT_BATCHES
            (*phase_shift_gate_iter)->call_in_execute(
              parallel_policy, executor, thread_index,
              first, index_wo_qubits,
              unsorted_fused_qubits_or_masks, sorted_fused_qubits_with_sentinel_or_index_masks,
              to_qubit_index_in_fused_gates, unit_qubit_value);
# ifdef BRA_PROFILE_PHASE_SHIFT_BATCHES
            if (thread_index == 0)
            {
              ++num_phase_fallback_gates;
              total_phase_fallback_time += std::chrono::steady_clock::now() - start_time;
            }
# endif // BRA_PROFILE_PHASE_SHIFT_BATCHES
          }
          gate_iter = phase_shift_block_end;
          continue;
        }

# ifdef BRA_PROFILE_PHASE_SHIFT_BATCHES
        auto const is_phase_fallback = next_gate_iter != gate_iter;
        auto const start_time
          = thread_index == 0 ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
# endif // BRA_PROFILE_PHASE_SHIFT_BATCHES
        (*gate_iter)->call_in_execute(
          parallel_policy, executor, thread_index,
          first, index_wo_qubits,
          unsorted_fused_qubits_or_masks, sorted_fused_qubits_with_sentinel_or_index_masks,
          to_qubit_index_in_fused_gates, unit_qubit_value);
# ifdef BRA_PROFILE_PHASE_SHIFT_BATCHES
        if (thread_index == 0)
        {
          auto const elapsed_time = std::chrono::steady_clock::now() - start_time;
          if (is_phase_fallback)
          {
            ++num_phase_fallback_gates;
            total_phase_fallback_time += elapsed_time;
          }
          else
          {
            ++num_other_fallback_gates;
            total_other_fallback_time += elapsed_time;
          }
        }
# endif // BRA_PROFILE_PHASE_SHIFT_BATCHES
        ++gate_iter;
      }
# ifdef BRA_PROFILE_PHASE_SHIFT_BATCHES
      if (thread_index == 0 and num_batches != std::size_t{0u})
        std::clog
          << "[phase-batch-summary] batches=" << num_batches
          << " table=" << num_table_batches
          << " fallback=" << num_fallback_batches
          << " terms=" << num_batched_terms
          << " elapsed=" << std::chrono::duration<double>{total_batch_time}.count()
          << std::endl;
      if (thread_index == 0 and num_hadamard_phase_batches != std::size_t{0u})
        std::clog
          << "[hadamard-phase-batch-summary] batches=" << num_hadamard_phase_batches
          << " terms=" << num_hadamard_phase_terms
          << " elapsed=" << std::chrono::duration<double>{total_hadamard_phase_batch_time}.count()
          << std::endl;
      if (thread_index == 0 and num_phase_fallback_gates != std::size_t{0u})
        std::clog
          << "[phase-fallback-summary] gates=" << num_phase_fallback_gates
          << " elapsed=" << std::chrono::duration<double>{total_phase_fallback_time}.count()
          << std::endl;
      if (thread_index == 0 and num_other_fallback_gates != std::size_t{0u})
        std::clog
          << "[other-fallback-summary] gates=" << num_other_fallback_gates
          << " elapsed=" << std::chrono::duration<double>{total_other_fallback_time}.count()
          << std::endl;
      if (thread_index == 0 and num_phase_star_batches != std::size_t{0u})
        std::clog
          << "[phase-star-batch-summary] batches=" << num_phase_star_batches
          << " terms=" << num_phase_star_terms
          << " elapsed=" << std::chrono::duration<double>{total_phase_star_time}.count()
          << std::endl;
# endif // BRA_PROFILE_PHASE_SHIFT_BATCHES
    }
  } // namespace fused_gate
} // namespace bra


#endif // BRA_FUSED_GATE_APPLY_FUSED_GATES_HPP
