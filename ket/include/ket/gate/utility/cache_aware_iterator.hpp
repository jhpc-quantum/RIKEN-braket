#ifndef KET_GATE_UTILITY_CACHE_AWARE_ITERATOR_HPP
# define KET_GATE_UTILITY_CACHE_AWARE_ITERATOR_HPP

# include <cassert>
# include <cstddef>
# include <array>
# include <iterator>
# include <memory>

# include <ket/gate/utility/index_with_qubits.hpp>
# ifndef KET_USE_BIT_MASKS_EXPLICITLY
#   include <ket/meta/state_integer_of.hpp>
# endif // KET_USE_BIT_MASKS_EXPLICITLY


namespace ket
{
  namespace gate
  {
    namespace utility
    {
# ifndef KET_USE_BIT_MASKS_EXPLICITLY
      template <typename RandomAccessIterator, typename Qubit>
      class cache_aware_iterator
      {
        using state_integer_type = ::ket::meta::state_integer_t<Qubit>;

       public:
        using value_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
        using difference_type = typename std::iterator_traits<RandomAccessIterator>::difference_type;
        using pointer = typename std::iterator_traits<RandomAccessIterator>::pointer;
        using reference = typename std::iterator_traits<RandomAccessIterator>::reference;
        using iterator_category = typename std::iterator_traits<RandomAccessIterator>::iterator_category;

       private:
        RandomAccessIterator first_;
        difference_type index_;
        state_integer_type tag_index_wo_qubits_;
        state_integer_type chunk_size_;
        Qubit const* unsorted_tag_qubits_first_;
        Qubit const* unsorted_tag_qubits_last_;
        Qubit const* sorted_tag_qubits_with_sentinel_first_;
        Qubit const* sorted_tag_qubits_with_sentinel_last_;

       public:
        cache_aware_iterator() = delete;
        cache_aware_iterator(cache_aware_iterator const&) = default;
        cache_aware_iterator(cache_aware_iterator&&) = default;
        cache_aware_iterator& operator=(cache_aware_iterator const&) = default;
        cache_aware_iterator& operator=(cache_aware_iterator&&) = default;

        cache_aware_iterator(
          RandomAccessIterator const first, state_integer_type const tag_index_wo_qubits, state_integer_type const chunk_size,
          Qubit const* unsorted_tag_qubits_first, Qubit const* unsorted_tag_qubits_last,
          Qubit const* sorted_tag_qubits_with_sentinel_first, Qubit const* sorted_tag_qubits_with_sentinel_last) noexcept
          : cache_aware_iterator{
              first, difference_type{0}, tag_index_wo_qubits, chunk_size,
              unsorted_tag_qubits_first, unsorted_tag_qubits_last,
              sorted_tag_qubits_with_sentinel_first, sorted_tag_qubits_with_sentinel_last}
        { }

        cache_aware_iterator(
          RandomAccessIterator const first, difference_type const index,
          state_integer_type const tag_index_wo_qubits, state_integer_type const chunk_size,
          Qubit const* unsorted_tag_qubits_first, Qubit const* unsorted_tag_qubits_last,
          Qubit const* sorted_tag_qubits_with_sentinel_first, Qubit const* sorted_tag_qubits_with_sentinel_last) noexcept
          : first_{first}, index_{index}, tag_index_wo_qubits_{tag_index_wo_qubits}, chunk_size_{chunk_size},
            unsorted_tag_qubits_first_{unsorted_tag_qubits_first},
            unsorted_tag_qubits_last_{unsorted_tag_qubits_last},
            sorted_tag_qubits_with_sentinel_first_{sorted_tag_qubits_with_sentinel_first},
            sorted_tag_qubits_with_sentinel_last_{sorted_tag_qubits_with_sentinel_last}
        { assert(unsorted_tag_qubits_last_ - unsorted_tag_qubits_first_ + 1 == sorted_tag_qubits_with_sentinel_last_ - sorted_tag_qubits_with_sentinel_first_); }

        auto operator==(cache_aware_iterator const& other) const noexcept -> bool
        {
          assert(
            first_ == other.first_ and tag_index_wo_qubits_ == other.tag_index_wo_qubits_ and chunk_size_ == other.chunk_size_
            and unsorted_tag_qubits_first_ == other.unsorted_tag_qubits_first_
            and unsorted_tag_qubits_last_ == other.unsorted_tag_qubits_last_
            and sorted_tag_qubits_with_sentinel_first_ == other.sorted_tag_qubits_with_sentinel_first_
            and sorted_tag_qubits_with_sentinel_last_ == other.sorted_tag_qubits_with_sentinel_last_);
          return index_ == other.index_;
        }

        auto operator<(cache_aware_iterator const& other) const noexcept -> bool
        {
          assert(
            first_ == other.first_ and tag_index_wo_qubits_ == other.tag_index_wo_qubits_ and chunk_size_ == other.chunk_size_
            and unsorted_tag_qubits_first_ == other.unsorted_tag_qubits_first_
            and unsorted_tag_qubits_last_ == other.unsorted_tag_qubits_last_
            and sorted_tag_qubits_with_sentinel_first_ == other.sorted_tag_qubits_with_sentinel_first_
            and sorted_tag_qubits_with_sentinel_last_ == other.sorted_tag_qubits_with_sentinel_last_);
          return index_ < other.index_;
        }

        auto operator*() const noexcept -> reference
        {
          auto const chunk_index = static_cast<state_integer_type>(index_) / chunk_size_;
          auto const index_in_chunk = static_cast<state_integer_type>(index_) % chunk_size_;
          return *(first_
                   + ::ket::gate::utility::index_with_qubits(
                       tag_index_wo_qubits_, chunk_index,
                       unsorted_tag_qubits_first_, unsorted_tag_qubits_last_,
                       sorted_tag_qubits_with_sentinel_first_, sorted_tag_qubits_with_sentinel_last_) * chunk_size_
                   + index_in_chunk);
        }

        auto operator[](std::size_t const n) -> reference
        {
          auto const chunk_index = static_cast<state_integer_type>(index_ + static_cast<difference_type>(n)) / chunk_size_;
          auto const index_in_chunk = static_cast<state_integer_type>(index_ + static_cast<difference_type>(n)) % chunk_size_;
          return *(first_
                   + ::ket::gate::utility::index_with_qubits(
                       tag_index_wo_qubits_, chunk_index,
                       unsorted_tag_qubits_first_, unsorted_tag_qubits_last_,
                       sorted_tag_qubits_with_sentinel_first_, sorted_tag_qubits_with_sentinel_last_) * chunk_size_
                   + index_in_chunk);
        }

        auto operator[](std::size_t const n) const -> value_type const&
        {
          auto const chunk_index = static_cast<state_integer_type>(index_ + static_cast<difference_type>(n)) / chunk_size_;
          auto const index_in_chunk = static_cast<state_integer_type>(index_ + static_cast<difference_type>(n)) % chunk_size_;
          return *(first_
                   + ::ket::gate::utility::index_with_qubits(
                       tag_index_wo_qubits_, chunk_index,
                       unsorted_tag_qubits_first_, unsorted_tag_qubits_last_,
                       sorted_tag_qubits_with_sentinel_first_, sorted_tag_qubits_with_sentinel_last_) * chunk_size_
                   + index_in_chunk);
        }

        auto operator++() noexcept -> cache_aware_iterator& { ++index_; return *this; }
        auto operator++(int) noexcept -> cache_aware_iterator { auto result = *this; ++*this; return result; }
        auto operator--() noexcept -> cache_aware_iterator& { --index_; return *this; }
        auto operator--(int) noexcept -> cache_aware_iterator { auto result = *this; --*this; return result; }
        auto operator+=(difference_type const n) noexcept -> cache_aware_iterator& { index_ += n; return *this; }
        auto operator-=(difference_type const n) noexcept -> cache_aware_iterator& { index_ -= n; return *this; }
        auto operator-(cache_aware_iterator const& other) const noexcept -> difference_type { return index_ - other.index_; }

        auto swap(cache_aware_iterator& other) noexcept -> void
        {
          using std::swap;
          swap(first_, other.first_);
          swap(index_, other.index_);
          swap(tag_index_wo_qubits_, other.tag_index_wo_qubits_);
          swap(chunk_size_, other.chunk_size_);
          swap(unsorted_tag_qubits_first_, other.unsorted_tag_qubits_first_);
          swap(unsorted_tag_qubits_last_, other.unsorted_tag_qubits_last_);
          swap(sorted_tag_qubits_with_sentinel_first_, other.sorted_tag_qubits_with_sentinel_first_);
          swap(sorted_tag_qubits_with_sentinel_last_, other.sorted_tag_qubits_with_sentinel_last_);
        }
      }; // class cache_aware_iterator<RandomAccessIterator, Qubit>

      template <typename RandomAccessIterator, typename Qubit>
      inline auto operator!=(
        ::ket::gate::utility::cache_aware_iterator<RandomAccessIterator, Qubit> const& lhs,
        ::ket::gate::utility::cache_aware_iterator<RandomAccessIterator, Qubit> const& rhs)
      -> bool
      { return not (lhs == rhs); }

      template <typename RandomAccessIterator, typename Qubit>
      inline auto operator>(
        ::ket::gate::utility::cache_aware_iterator<RandomAccessIterator, Qubit> const& lhs,
        ::ket::gate::utility::cache_aware_iterator<RandomAccessIterator, Qubit> const& rhs)
      -> bool
      { return rhs < lhs; }

      template <typename RandomAccessIterator, typename Qubit>
      inline auto operator<=(
        ::ket::gate::utility::cache_aware_iterator<RandomAccessIterator, Qubit> const& lhs,
        ::ket::gate::utility::cache_aware_iterator<RandomAccessIterator, Qubit> const& rhs)
      -> bool
      { return not (lhs > rhs); }

      template <typename RandomAccessIterator, typename Qubit>
      inline auto operator>=(
        ::ket::gate::utility::cache_aware_iterator<RandomAccessIterator, Qubit> const& lhs,
        ::ket::gate::utility::cache_aware_iterator<RandomAccessIterator, Qubit> const& rhs)
      -> bool
      { return not (lhs < rhs); }

      template <typename RandomAccessIterator, typename Qubit>
      inline auto operator+(
        ::ket::gate::utility::cache_aware_iterator<RandomAccessIterator, Qubit> iter,
        typename ::ket::gate::utility::cache_aware_iterator<RandomAccessIterator, Qubit>::difference_type const n)
      -> ::ket::gate::utility::cache_aware_iterator<RandomAccessIterator, Qubit>
      { return iter += n; }

      template <typename RandomAccessIterator, typename Qubit>
      inline auto operator+(
        typename ::ket::gate::utility::cache_aware_iterator<RandomAccessIterator, Qubit>::difference_type const n,
        ::ket::gate::utility::cache_aware_iterator<RandomAccessIterator, Qubit> iter)
      -> ::ket::gate::utility::cache_aware_iterator<RandomAccessIterator, Qubit>
      { return iter += n; }

      template <typename RandomAccessIterator, typename Qubit>
      inline auto operator-(
        ::ket::gate::utility::cache_aware_iterator<RandomAccessIterator, Qubit> iter,
        typename ::ket::gate::utility::cache_aware_iterator<RandomAccessIterator, Qubit>::difference_type const n)
      -> ::ket::gate::utility::cache_aware_iterator<RandomAccessIterator, Qubit>
      { return iter -= n; }

      template <typename RandomAccessIterator, typename Qubit>
      inline auto swap(
        ::ket::gate::utility::cache_aware_iterator<RandomAccessIterator, Qubit>& lhs,
        ::ket::gate::utility::cache_aware_iterator<RandomAccessIterator, Qubit>& rhs) noexcept
      -> void
      { lhs.swap(rhs); }

      template <typename RandomAccessIterator, typename Qubit>
      inline auto make_cache_aware_iterator(
        RandomAccessIterator const first,
        ::ket::meta::state_integer_t<Qubit> const tag_index_wo_qubits, ::ket::meta::state_integer_t<Qubit> const chunk_size,
        Qubit const* unsorted_tag_qubits_first, Qubit const* unsorted_tag_qubits_last,
        Qubit const* sorted_tag_qubits_with_sentinel_first, Qubit const* sorted_tag_qubits_with_sentinel_last) noexcept
      -> ::ket::gate::utility::cache_aware_iterator<RandomAccessIterator, Qubit>
      {
        return {first, tag_index_wo_qubits, chunk_size,
                unsorted_tag_qubits_first, unsorted_tag_qubits_last,
                sorted_tag_qubits_with_sentinel_first, sorted_tag_qubits_with_sentinel_last};
      }

      template <typename RandomAccessIterator, typename Qubit>
      inline auto make_cache_aware_iterator(
        RandomAccessIterator const first, typename std::iterator_traits<RandomAccessIterator>::difference_type const index,
        ::ket::meta::state_integer_t<Qubit> const tag_index_wo_qubits, ::ket::meta::state_integer_t<Qubit> const chunk_size,
        Qubit const* unsorted_tag_qubits_first, Qubit const* unsorted_tag_qubits_last,
        Qubit const* sorted_tag_qubits_with_sentinel_first, Qubit const* sorted_tag_qubits_with_sentinel_last) noexcept
      -> ::ket::gate::utility::cache_aware_iterator<RandomAccessIterator, Qubit>
      {
        return {first, index, tag_index_wo_qubits, chunk_size,
                unsorted_tag_qubits_first, unsorted_tag_qubits_last,
                sorted_tag_qubits_with_sentinel_first, sorted_tag_qubits_with_sentinel_last};
      }
# else // KET_USE_BIT_MASKS_EXPLICITLY
      template <typename RandomAccessIterator, typename StateInteger>
      class cache_aware_iterator
      {
       public:
        using value_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
        using difference_type = typename std::iterator_traits<RandomAccessIterator>::difference_type;
        using pointer = typename std::iterator_traits<RandomAccessIterator>::pointer;
        using reference = typename std::iterator_traits<RandomAccessIterator>::reference;
        using iterator_category = typename std::iterator_traits<RandomAccessIterator>::iterator_category;

       private:
        RandomAccessIterator first_;
        difference_type index_;
        StateInteger tag_index_wo_qubits_;
        StateInteger chunk_size_;
        StateInteger const* tag_qubit_masks_first_;
        StateInteger const* tag_qubit_masks_last_;
        StateInteger const* tag_index_masks_first_;
        StateInteger const* tag_index_masks_last_;

       public:
        cache_aware_iterator() = delete;
        cache_aware_iterator(cache_aware_iterator const&) = default;
        cache_aware_iterator(cache_aware_iterator&&) = default;
        cache_aware_iterator& operator=(cache_aware_iterator const&) = default;
        cache_aware_iterator& operator=(cache_aware_iterator&&) = default;

        cache_aware_iterator(
          RandomAccessIterator const first, StateInteger const tag_index_wo_qubits, StateInteger const chunk_size,
          StateInteger const* tag_qubit_masks_first, StateInteger const* tag_qubit_masks_last,
          StateInteger const* tag_index_masks_first, StateInteger const* tag_index_masks_last) noexcept
          : cache_aware_iterator{
              first, 0, tag_index_wo_qubits, chunk_size,
              tag_qubit_masks_first, tag_qubit_masks_last, tag_index_masks_first, tag_index_masks_last}
        { }

        cache_aware_iterator(
          RandomAccessIterator const first, difference_type const index, StateInteger const tag_index_wo_qubits, StateInteger const chunk_size,
          StateInteger const* tag_qubit_masks_first, StateInteger const* tag_qubit_masks_last,
          StateInteger const* tag_index_masks_first, StateInteger const* tag_index_masks_last) noexcept
          : first_{first}, index_{index}, tag_index_wo_qubits_{tag_index_wo_qubits}, chunk_size_{chunk_size},
            tag_qubit_masks_first_{tag_qubit_masks_first}, tag_qubit_masks_last_{tag_qubit_masks_last},
            tag_index_masks_first_{tag_index_masks_first}, tag_index_masks_last_{tag_index_masks_last}
        { assert(tag_qubit_masks_last_ - tag_qubit_masks_first_ + 1 == tag_index_masks_last_ - tag_index_masks_first_); }

        auto operator==(cache_aware_iterator const& other) const noexcept -> bool
        {
          assert(
            first_ == other.first_ and tag_index_wo_qubits_ == other.tag_index_wo_qubits_ and chunk_size_ == other.chunk_size_
            and tag_qubit_masks_first_ == other.tag_qubit_masks_first_
            and tag_qubit_masks_last_ == other.tag_qubit_masks_last_
            and tag_index_masks_first_ == other.tag_index_masks_first_
            and tag_index_masks_last_ == other.tag_index_masks_last_);
          return index_ == other.index_;
        }

        auto operator<(cache_aware_iterator const& other) const noexcept -> bool
        {
          assert(
            first_ == other.first_ and tag_index_wo_qubits_ == other.tag_index_wo_qubits_ and chunk_size_ == other.chunk_size_
            and tag_qubit_masks_first_ == other.tag_qubit_masks_first_
            and tag_qubit_masks_last_ == other.tag_qubit_masks_last_
            and tag_index_masks_first_ == other.tag_index_masks_first_
            and tag_index_masks_last_ == other.tag_index_masks_last_);
          return index_ < other.index_;
        }

        auto operator*() const noexcept -> reference
        {
          auto const chunk_index = static_cast<StateInteger>(index_) / chunk_size_;
          auto const index_in_chunk = static_cast<StateInteger>(index_) % chunk_size_;
          return *(first_
                   + ::ket::gate::utility::index_with_qubits(
                       tag_index_wo_qubits_, chunk_index,
                       tag_qubit_masks_first_, tag_qubit_masks_last_,
                       tag_index_masks_first_, tag_index_masks_last_) * chunk_size_
                   + index_in_chunk);
        }

        auto operator[](std::size_t const n) -> reference
        {
          auto const chunk_index = static_cast<StateInteger>(index_ + static_cast<difference_type>(n)) / chunk_size_;
          auto const index_in_chunk = static_cast<StateInteger>(index_ + static_cast<difference_type>(n)) % chunk_size_;
          return *(first_
                   + ::ket::gate::utility::index_with_qubits(
                       tag_index_wo_qubits_, chunk_index,
                       tag_qubit_masks_first_, tag_qubit_masks_last_,
                       tag_index_masks_first_, tag_index_masks_last_) * chunk_size_
                   + index_in_chunk);
        }

        auto operator[](std::size_t const n) const -> value_type const&
        {
          auto const chunk_index = static_cast<StateInteger>(index_ + static_cast<difference_type>(n)) / chunk_size_;
          auto const index_in_chunk = static_cast<StateInteger>(index_ + static_cast<difference_type>(n)) % chunk_size_;
          return *(first_
                   + ::ket::gate::utility::index_with_qubits(
                       tag_index_wo_qubits_, chunk_index,
                       tag_qubit_masks_first_, tag_qubit_masks_last_,
                       tag_index_masks_first_, tag_index_masks_last_) * chunk_size_
                   + index_in_chunk);
        }

        auto operator++() noexcept -> cache_aware_iterator& { ++index_; return *this; }
        auto operator++(int) noexcept -> cache_aware_iterator { auto result = *this; ++*this; return result; }
        auto operator--() noexcept -> cache_aware_iterator& { --index_; return *this; }
        auto operator--(int) noexcept -> cache_aware_iterator { auto result = *this; --*this; return result; }
        auto operator+=(difference_type const n) noexcept -> cache_aware_iterator& { index_ += n; return *this; }
        auto operator-=(difference_type const n) noexcept -> cache_aware_iterator& { index_ -= n; return *this; }
        auto operator-(cache_aware_iterator const& other) const noexcept -> difference_type { return index_ - other.index_; }

        auto swap(cache_aware_iterator& other) noexcept -> void
        {
          using std::swap;
          swap(first_, other.first_);
          swap(index_, other.index_);
          swap(tag_index_wo_qubits_, other.tag_index_wo_qubits_);
          swap(chunk_size_, other.chunk_size_);
          swap(tag_qubit_masks_first_, other.tag_qubit_masks_first_);
          swap(tag_qubit_masks_last_, other.tag_qubit_masks_last_);
          swap(tag_index_masks_first_, other.tag_index_masks_first_);
          swap(tag_index_masks_last_, other.tag_index_masks_last_);
        }
      }; // class cache_aware_iterator<RandomAccessIterator, StateInteger>

      template <typename RandomAccessIterator, typename StateInteger>
      inline auto operator!=(
        ::ket::gate::utility::cache_aware_iterator<RandomAccessIterator, StateInteger> const& lhs,
        ::ket::gate::utility::cache_aware_iterator<RandomAccessIterator, StateInteger> const& rhs)
      -> bool
      { return not (lhs == rhs); }

      template <typename RandomAccessIterator, typename StateInteger>
      inline auto operator>(
        ::ket::gate::utility::cache_aware_iterator<RandomAccessIterator, StateInteger> const& lhs,
        ::ket::gate::utility::cache_aware_iterator<RandomAccessIterator, StateInteger> const& rhs)
      -> bool
      { return rhs < lhs; }

      template <typename RandomAccessIterator, typename StateInteger>
      inline auto operator<=(
        ::ket::gate::utility::cache_aware_iterator<RandomAccessIterator, StateInteger> const& lhs,
        ::ket::gate::utility::cache_aware_iterator<RandomAccessIterator, StateInteger> const& rhs)
      -> bool
      { return not (lhs > rhs); }

      template <typename RandomAccessIterator, typename StateInteger>
      inline auto operator>=(
        ::ket::gate::utility::cache_aware_iterator<RandomAccessIterator, StateInteger> const& lhs,
        ::ket::gate::utility::cache_aware_iterator<RandomAccessIterator, StateInteger> const& rhs)
      -> bool
      { return not (lhs < rhs); }

      template <typename RandomAccessIterator, typename StateInteger>
      inline auto operator+(
        ::ket::gate::utility::cache_aware_iterator<RandomAccessIterator, StateInteger> iter,
        typename ::ket::gate::utility::cache_aware_iterator<RandomAccessIterator, StateInteger>::difference_type const n)
      -> ::ket::gate::utility::cache_aware_iterator<RandomAccessIterator, StateInteger>
      { return iter += n; }

      template <typename RandomAccessIterator, typename StateInteger>
      inline auto operator+(
        typename ::ket::gate::utility::cache_aware_iterator<RandomAccessIterator, StateInteger>::difference_type const n,
        ::ket::gate::utility::cache_aware_iterator<RandomAccessIterator, StateInteger> iter)
      -> ::ket::gate::utility::cache_aware_iterator<RandomAccessIterator, StateInteger>
      { return iter += n; }

      template <typename RandomAccessIterator, typename StateInteger>
      inline auto operator-(
        ::ket::gate::utility::cache_aware_iterator<RandomAccessIterator, StateInteger> iter,
        typename ::ket::gate::utility::cache_aware_iterator<RandomAccessIterator, StateInteger>::difference_type const n)
      -> ::ket::gate::utility::cache_aware_iterator<RandomAccessIterator, StateInteger>
      { return iter -= n; }

      template <typename RandomAccessIterator, typename StateInteger>
      inline auto swap(
        ::ket::gate::utility::cache_aware_iterator<RandomAccessIterator, StateInteger>& lhs,
        ::ket::gate::utility::cache_aware_iterator<RandomAccessIterator, StateInteger>& rhs) noexcept
      -> void
      { lhs.swap(rhs); }

      template <typename RandomAccessIterator, typename StateInteger>
      inline auto make_cache_aware_iterator(
        RandomAccessIterator const first, StateInteger const tag_index_wo_qubits, StateInteger const chunk_size,
        StateInteger const* tag_qubit_masks_first, StateInteger const* tag_qubit_masks_last,
        StateInteger const* tag_index_masks_first, StateInteger const* tag_index_masks_last) noexcept
      -> ::ket::gate::utility::cache_aware_iterator<RandomAccessIterator, StateInteger>
      { return {first, tag_index_wo_qubits, chunk_size, tag_qubit_masks_first, tag_qubit_masks_last, tag_index_masks_first, tag_index_masks_last}; }

      template <typename RandomAccessIterator, typename StateInteger>
      inline auto make_cache_aware_iterator(
        RandomAccessIterator const first, typename std::iterator_traits<RandomAccessIterator>::difference_type const index,
        StateInteger const tag_index_wo_qubits, StateInteger const chunk_size,
        StateInteger const* tag_qubit_masks_first, StateInteger const* tag_qubit_masks_last,
        StateInteger const* tag_index_masks_first, StateInteger const* tag_index_masks_last) noexcept
      -> ::ket::gate::utility::cache_aware_iterator<RandomAccessIterator, StateInteger>
      { return {first, index, tag_index_wo_qubits, chunk_size, tag_qubit_masks_first, tag_qubit_masks_last, tag_index_masks_first, tag_index_masks_last}; }
# endif // KET_USE_BIT_MASKS_EXPLICITLY
    } // namespace utility
  } // namespace gate
} // namespace ket


#endif // KET_GATE_UTILITY_CACHE_AWARE_ITERATOR_HPP
