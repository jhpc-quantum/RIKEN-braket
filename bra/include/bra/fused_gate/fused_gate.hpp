#ifndef BRA_FUSED_GATE_FUSED_GATE_HPP
# define BRA_FUSED_GATE_FUSED_GATE_HPP

# include <cassert>
# include <iterator>
# include <type_traits>
# include <vector>

# include <boost/optional.hpp>
# include <boost/range/begin.hpp>
# include <boost/range/end.hpp>

# include <bra/types.hpp>


namespace bra
{
  namespace fused_gate
  {
    enum class cez_qubit_state : int { not_global, global_zero, global_one };
    enum class control_qubit_state : int { zero, one };

    template <typename Iterator>
    class fused_gate
    {
      bool is_enabled_;
      ::bra::state_integer_type unit_control_qubit_mask_;

     public:
      fused_gate() : is_enabled_{true}, unit_control_qubit_mask_{0u} { }
      virtual ~fused_gate() = default;

      fused_gate(fused_gate const&) = delete;
      fused_gate& operator=(fused_gate const&) = delete;
      fused_gate(fused_gate&&) = delete;
      fused_gate& operator=(fused_gate&&) = delete;

# ifndef KET_USE_BIT_MASKS_EXPLICITLY
      auto call(
        Iterator const first, ::bra::state_integer_type const fused_index_wo_qubits,
        std::vector< ::bra::qubit_type > const& unsorted_fused_qubits,
        std::vector< ::bra::qubit_type > const& sorted_fused_qubits_with_sentinel,
        std::vector< ::bra::bit_integer_type > const& to_qubit_index_in_fused_gates,
        ::bra::state_integer_type const unit_qubit_value = ::bra::state_integer_type{0u}) const -> void
      {
        assert(sorted_fused_qubits_with_sentinel.size() == unsorted_fused_qubits.size() + std::size_t{1u});
        if (not is_enabled_ or (unit_qubit_value bitand unit_control_qubit_mask_) != unit_control_qubit_mask_)
          return;

        do_call(first, fused_index_wo_qubits, unsorted_fused_qubits, sorted_fused_qubits_with_sentinel, to_qubit_index_in_fused_gates);
      }

      template <typename UnsortedFusedQubitsRange, typename SortedFusedQubitsWithSentinelRange>
      auto call(
        Iterator const first, ::bra::state_integer_type const fused_index_wo_qubits,
        UnsortedFusedQubitsRange const& unsorted_fused_qubits,
        SortedFusedQubitsWithSentinelRange const& sorted_fused_qubits_with_sentinel,
        std::vector< ::bra::bit_integer_type > const& to_qubit_index_in_fused_gates,
        ::bra::state_integer_type const unit_qubit_value = ::bra::state_integer_type{0u}) const
      -> typename std::enable_if<
           not std::is_same<typename std::decay<UnsortedFusedQubitsRange>::type, std::vector< ::bra::qubit_type >>::value
           or not std::is_same<typename std::decay<SortedFusedQubitsWithSentinelRange>::type, std::vector< ::bra::qubit_type >>::value>::type
      {
        auto const unsorted_fused_qubits_vector
          = std::vector< ::bra::qubit_type >{boost::begin(unsorted_fused_qubits), boost::end(unsorted_fused_qubits)};
        auto const sorted_fused_qubits_with_sentinel_vector
          = std::vector< ::bra::qubit_type >{
              boost::begin(sorted_fused_qubits_with_sentinel), boost::end(sorted_fused_qubits_with_sentinel)};

        call(
          first, fused_index_wo_qubits,
          unsorted_fused_qubits_vector, sorted_fused_qubits_with_sentinel_vector,
          to_qubit_index_in_fused_gates, unit_qubit_value);
      }
# else // KET_USE_BIT_MASKS_EXPLICITLY
      auto call(
        Iterator const first, ::bra::state_integer_type const fused_index_wo_qubits,
        std::vector< ::bra::state_integer_type > const& qubit_masks,
        std::vector< ::bra::state_integer_type > const& index_masks,
        std::vector< ::bra::bit_integer_type > const& to_qubit_index_in_fused_gates,
        ::bra::state_integer_type const unit_qubit_value = ::bra::state_integer_type{0u}) const -> void
      {
        assert(index_masks.size() == qubit_masks.size() + std::size_t{1u});
        if (not is_enabled_ or (unit_qubit_value bitand unit_control_qubit_mask_) != unit_control_qubit_mask_)
          return;

        do_call(first, fused_index_wo_qubits, qubit_masks, index_masks, to_qubit_index_in_fused_gates);
      }
# endif // KET_USE_BIT_MASKS_EXPLICITLY

      // Masks refer to bit positions within the unit-qubit value passed to call().
      auto disable_unit_control_qubits(
        std::vector< ::bra::control_qubit_type > const& control_qubits,
        std::vector< ::bra::state_integer_type > const& unit_qubit_masks)
      -> void
      {
        assert(control_qubits.size() == unit_qubit_masks.size());
        auto mask_iter = unit_qubit_masks.begin();
        for (auto iter = control_qubits.begin(); iter != control_qubits.end(); ++iter, ++mask_iter)
        {
          assert(*mask_iter != ::bra::state_integer_type{0u});
          assert((*mask_iter bitand (*mask_iter - ::bra::state_integer_type{1u})) == ::bra::state_integer_type{0u});
          if (do_disable_control_qubits(iter, std::next(iter)))
            unit_control_qubit_mask_ |= *mask_iter;
        }
      }

      auto disable_control_qubits(
        typename std::vector< ::bra::qubit_type >::const_iterator const first,
        typename std::vector< ::bra::qubit_type >::const_iterator const last)
      -> bool
      { return do_disable_control_qubits(first, last); }

      auto disable_control_qubits(
        typename std::vector< ::bra::control_qubit_type >::const_iterator const first,
        typename std::vector< ::bra::control_qubit_type >::const_iterator const last)
      -> bool
      { return do_disable_control_qubits(first, last); }

      auto disable_control_qubits(
        typename std::vector< ::bra::control_qubit_type >::const_iterator first,
        typename std::vector< ::bra::control_qubit_type >::const_iterator const last,
        typename std::vector< ::bra::fused_gate::control_qubit_state >::const_iterator control_qubit_state_first)
      -> void
      {
        for (; first != last; ++first, ++control_qubit_state_first)
        {
          auto const next = std::next(first);
          auto const is_control_used = do_disable_control_qubits(first, next);
          if (is_control_used and *control_qubit_state_first == ::bra::fused_gate::control_qubit_state::zero)
            is_enabled_ = false;
        }
      }

      auto disable_cez_global_qubits(
        typename std::vector< ::bra::qubit_type >::const_iterator const first,
        typename std::vector< ::bra::qubit_type >::const_iterator const last)
      -> bool
      { return do_disable_control_qubits(first, last); }

      auto modify_cez(
        typename std::vector< ::bra::qubit_type >::const_iterator const first,
        typename std::vector< ::bra::qubit_type >::const_iterator const last,
        typename std::vector< ::bra::fused_gate::cez_qubit_state >::const_iterator const cez_qubit_state_first)
      -> void
      { do_modify_cez(first, last, cez_qubit_state_first); }

      auto maybe_phase_shiftize_ez(
        typename std::vector< ::bra::qubit_type >::const_iterator const first,
        typename std::vector< ::bra::qubit_type >::const_iterator const last)
      -> boost::optional<std::pair< ::bra::control_qubit_type, ::bra::real_type >>
      { return do_maybe_phase_shiftize_ez(first, last); }

     private:
# ifndef KET_USE_BIT_MASKS_EXPLICITLY
      virtual auto do_call(
        Iterator const first, ::bra::state_integer_type const fused_index_wo_qubits,
        std::vector< ::bra::qubit_type > const& unsorted_fused_qubits,
        std::vector< ::bra::qubit_type > const& sorted_fused_qubits_with_sentinel,
        std::vector< ::bra::bit_integer_type > const& to_qubit_index_in_fused_gates) const -> void = 0;
# else // KET_USE_BIT_MASKS_EXPLICITLY
      virtual auto do_call(
        Iterator const first, ::bra::state_integer_type const fused_index_wo_qubits,
        std::vector< ::bra::state_integer_type > const& qubit_masks,
        std::vector< ::bra::state_integer_type > const& index_masks,
        std::vector< ::bra::bit_integer_type > const& to_qubit_index_in_fused_gates) const -> void = 0;
# endif // KET_USE_BIT_MASKS_EXPLICITLY

      virtual auto do_disable_control_qubits(
        typename std::vector< ::bra::qubit_type >::const_iterator const first,
        typename std::vector< ::bra::qubit_type >::const_iterator const last) -> bool;

      virtual auto do_disable_control_qubits(
        typename std::vector< ::bra::control_qubit_type >::const_iterator const first,
        typename std::vector< ::bra::control_qubit_type >::const_iterator const last) -> bool;

      virtual auto do_modify_cez(
        typename std::vector< ::bra::qubit_type >::const_iterator const first,
        typename std::vector< ::bra::qubit_type >::const_iterator const last,
        typename std::vector< ::bra::fused_gate::cez_qubit_state >::const_iterator const cez_qubit_state_first) -> void;

      virtual auto do_maybe_phase_shiftize_ez(
        typename std::vector< ::bra::qubit_type >::const_iterator const first,
        typename std::vector< ::bra::qubit_type >::const_iterator const last)
      -> boost::optional<std::pair< ::bra::control_qubit_type, ::bra::real_type >>;
    }; // class fused_gate<Iterator>
  } // namespace fused_gate
} // namespace bra


#endif // BRA_FUSED_GATE_FUSED_GATE_HPP
