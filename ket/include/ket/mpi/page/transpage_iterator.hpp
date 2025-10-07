#ifndef KET_MPI_PAGE_TRANSPAGE_ITERATOR_HPP
# define KET_MPI_PAGE_TRANSPAGE_ITERATOR_HPP

# include <vector>
# include <iterator>
# include <numeric>
# include <functional>
# include <utility>
# include <type_traits>

# include <ket/qubit.hpp>
# include <ket/meta/state_integer_of.hpp>
# include <ket/meta/bit_integer_of.hpp>
# include <ket/utility/meta/ranges.hpp>
# include <ket/mpi/permutated.hpp>


namespace ket
{
  namespace mpi
  {
    namespace page
    {
      template <typename PermutatedQubitIterator, typename StateInteger, typename BitInteger>
      inline auto base_page_index(
        PermutatedQubitIterator const permutated_operated_page_qubit_first,
        PermutatedQubitIterator const permutated_operated_page_qubit_last,
        StateInteger const page_index_wo_qubits, BitInteger const num_nonpage_local_qubits)
      -> StateInteger
      {
        using permutated_qubit_type
          = typename std::iterator_traits<PermutatedQubitIterator>::value_type;
        return std::accumulate(
          permutated_operated_page_qubit_first, permutated_operated_page_qubit_last,
          page_index_wo_qubits,
          [num_nonpage_local_qubits](
            StateInteger const partial_base_page_index,
            permutated_qubit_type const permutated_operated_page_qubit)
          {
            auto const corrected_permutated_operated_page_qubit
              = permutated_operated_page_qubit - num_nonpage_local_qubits;
            auto const lower_mask
              = (StateInteger{1u} << corrected_permutated_operated_page_qubit)
                  - StateInteger{1u};
            auto const upper_mask = compl lower_mask;
            return ((partial_base_page_index bitand upper_mask) << 1) bitor (partial_base_page_index bitand lower_mask);
          });
      }

      template <
        typename StateInteger, typename PermutatedQubitIterator1,
        typename PermutatedQubitIterator2, typename BitInteger>
      inline auto transpage_index_to_page_index(
        StateInteger const transpage_index,
        PermutatedQubitIterator1 const mapped_permutated_nonpage_qubit_first,
        PermutatedQubitIterator1 const mapped_permutated_nonpage_qubit_last,
        PermutatedQubitIterator2 const permutated_operated_page_qubit_first,
        StateInteger const base_page_index, BitInteger const num_nonpage_local_qubits)
      -> StateInteger
      {
        using permutated_qubit_type
          = typename std::iterator_traits<PermutatedQubitIterator1>::value_type;
        static_assert(
          std::is_same<permutated_qubit_type, typename std::iterator_traits<PermutatedQubitIterator2>::value_type>::value,
          "The value_type's of PermutatedQubitIteratot1 and PermutatedQubitIterator2 are the same");
        return std::inner_product(
          mapped_permutated_nonpage_qubit_first,
          mapped_permutated_nonpage_qubit_last,
          permutated_operated_page_qubit_first,
          base_page_index, std::bit_or<StateInteger>{},
          [transpage_index, num_nonpage_local_qubits](
            permutated_qubit_type const mapped_permutated_nonpage_qubit,
            permutated_qubit_type const permutated_operated_page_qubit)
          {
            return
              ((transpage_index bitand (StateInteger{1u} << mapped_permutated_nonpage_qubit))
                 >> mapped_permutated_nonpage_qubit)
                << (permutated_operated_page_qubit - num_nonpage_local_qubits);
          });
      }

      template <typename StateInteger, typename PermutatedQubitIterator, typename BitInteger>
      inline auto nonpage_index(
        PermutatedQubitIterator const mapped_permutated_nonpage_qubit_first,
        PermutatedQubitIterator const mapped_permutated_nonpage_qubit_last,
        StateInteger const transpage_index, StateInteger const mapped_nonpage_qubits_bits,
        BitInteger const num_nonpage_local_qubits)
      -> StateInteger
      {
        auto result = transpage_index;

        for (auto mapped_permutated_nonpage_qubit_iter = mapped_permutated_nonpage_qubit_first;
             mapped_permutated_nonpage_qubit_iter != mapped_permutated_nonpage_qubit_last;
             ++mapped_permutated_nonpage_qubit_iter)
        {
          auto const iter_index
            = mapped_permutated_nonpage_qubit_iter - mapped_permutated_nonpage_qubit_first;
          result
            = (result bitand (compl (StateInteger{1u} << *mapped_permutated_nonpage_qubit_iter)))
                bitor (((mapped_nonpage_qubits_bits bitand (StateInteger{1u} << iter_index)) >> iter_index) << *mapped_permutated_nonpage_qubit_iter);
        }

        return result;
      }

      template <typename PermutatedQubitIterator, typename StateInteger, typename BitInteger>
      inline auto generate_mapped_permutated_nonpage_qubits(
        PermutatedQubitIterator const permutated_operated_nonpage_qubit_first,
        PermutatedQubitIterator const permutated_operated_nonpage_qubit_last,
        ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const least_permutated_page_qubit,
        BitInteger const num_operated_page_qubits)
      -> std::vector< ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > >
      {
        using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
        using permutated_qubit_type = ::ket::mpi::permutated<qubit_type>;
        auto result = std::vector<permutated_qubit_type>(num_operated_page_qubits);

        auto possible_mapped_permutated_nonpage_qubit = least_permutated_page_qubit;
        auto permutated_operated_nonpage_qubit_iter = permutated_operated_nonpage_qubit_last;

        using std::rbegin;
        using std::rend;
        auto const rlast = rend(result);
        for (auto riter = rbegin(result); riter != rlast; ++riter)
        {
          --possible_mapped_permutated_nonpage_qubit;
          if (possible_mapped_permutated_nonpage_qubit < *permutated_operated_nonpage_qubit_first)
          {
            *riter = possible_mapped_permutated_nonpage_qubit;
            continue;
          }

          if (permutated_operated_nonpage_qubit_iter != permutated_operated_nonpage_qubit_first)
            --permutated_operated_nonpage_qubit_iter;

          while (possible_mapped_permutated_nonpage_qubit == *permutated_operated_nonpage_qubit_iter)
          {
            --possible_mapped_permutated_nonpage_qubit;

            if (permutated_operated_nonpage_qubit_iter == permutated_operated_nonpage_qubit_first)
              break;

            --permutated_operated_nonpage_qubit_iter;
          }

          *riter = possible_mapped_permutated_nonpage_qubit;
        }

        return result;
      }

      struct transpage_sentinel { };

      template <typename State, typename PermutatedQubitIterator1, typename PermutatedQubitIterator2>
      class transpage_iterator
      {
        using permutated_qubit_type = typename std::iterator_traits<PermutatedQubitIterator1>::value_type;
        static_assert(
          std::is_same<permutated_qubit_type, typename std::iterator_traits<PermutatedQubitIterator2>::value_type>::value,
          "The value_type's of PermutatedQubitIterator1 and PermutatedQubitIterator2 are the same");
        using state_integer_type = ::ket::meta::state_integer_t<permutated_qubit_type>;
        using bit_integer_type = ::ket::meta::bit_integer_t<permutated_qubit_type>;

       public:
        using value_type = ::ket::utility::meta::range_value_t<State>;
        using difference_type = ::ket::utility::meta::range_difference_t<State>;
        using pointer = ::ket::utility::meta::range_pointer_t<State>;
        using reference = ::ket::utility::meta::range_reference_t<State>;
        using iterator_category = std::random_access_iterator_tag;

       private:
        State* state_ptr_;
        PermutatedQubitIterator1 mapped_permutated_nonpage_qubit_first_;
        PermutatedQubitIterator1 mapped_permutated_nonpage_qubit_last_;
        PermutatedQubitIterator2 permutated_operated_page_qubit_first_;
        state_integer_type data_block_index_;
        state_integer_type base_page_index_;
        state_integer_type mapped_nonpage_qubits_bits_;

        difference_type index_;

       public:
        template <typename StateInteger>
        transpage_iterator(
          State& state,
          PermutatedQubitIterator1 const mapped_permutated_nonpage_qubit_first,
          PermutatedQubitIterator1 const mapped_permutated_nonpage_qubit_last,
          PermutatedQubitIterator2 const permutated_operated_page_qubit_first,
          PermutatedQubitIterator2 const permutated_operated_page_qubit_last,
          StateInteger const data_block_index,
          StateInteger const page_index_wo_qubits,
          StateInteger const mapped_nonpage_qubit_bits,
          difference_type const index = difference_type{0}) noexcept
          : state_ptr_{std::addressof(state)},
            mapped_permutated_nonpage_qubit_first_{mapped_permutated_nonpage_qubit_first},
            mapped_permutated_nonpage_qubit_last_{mapped_permutated_nonpage_qubit_last},
            permutated_operated_page_qubit_first_{permutated_operated_page_qubit_first},
            data_block_index_{data_block_index},
            base_page_index_{
              ::ket::mpi::page::base_page_index(
                permutated_operated_page_qubit_first, permutated_operated_page_qubit_last,
                page_index_wo_qubits,
                static_cast<bit_integer_type>(state.num_local_qubits() - state.num_page_qubits()))},
            mapped_nonpage_qubits_bits_{mapped_nonpage_qubit_bits},
            index_{index}
        {
          static_assert(
            std::is_same<StateInteger, state_integer_type>::value,
            "StateInteger should be the same as state_integer_type of value_type of PermutatedQubitIterator1");
          assert(
            mapped_permutated_nonpage_qubit_last - mapped_permutated_nonpage_qubit_first
            == permutated_operated_page_qubit_last - permutated_operated_page_qubit_first);
        }

        auto operator==(transpage_iterator const& other) const noexcept -> bool
        {
          assert(state_ptr_ == other.state_ptr_);
          return index_ == other.index_;
        }

        auto operator<(transpage_iterator const& other) const noexcept -> bool
        {
          assert(state_ptr_ == other.state_ptr_);
          return index_ < other.index_;
        }

        auto operator==(::ket::mpi::page::transpage_sentinel const& other) const noexcept -> bool
        { return index_ == last_index(*state_ptr_); }

        auto operator<(::ket::mpi::page::transpage_sentinel const& other) const noexcept -> bool
        { return index_ < last_index(*state_ptr_); }

        auto operator*() const noexcept -> reference
        {
          auto const num_nonpage_local_qubits
            = static_cast<bit_integer_type>(state_ptr_->num_local_qubits() - state_ptr_->num_page_qubits());

          auto const page_index
            = ::ket::mpi::page::transpage_index_to_page_index(
                static_cast<state_integer_type>(index_),
                mapped_permutated_nonpage_qubit_first_, mapped_permutated_nonpage_qubit_last_,
                permutated_operated_page_qubit_first_, base_page_index_, num_nonpage_local_qubits);

          auto const nonpage_index
            = ::ket::mpi::page::nonpage_index(
                mapped_permutated_nonpage_qubit_first_, mapped_permutated_nonpage_qubit_last_,
                static_cast<state_integer_type>(index_), mapped_nonpage_qubits_bits_,
                num_nonpage_local_qubits);

          using std::begin;
          return *(begin(state_ptr_->page_range(std::make_pair(data_block_index_, page_index))) + nonpage_index);
        }

        auto operator[](difference_type const n) const -> reference
        {
          auto const index = static_cast<state_integer_type>(index_ + n);
          auto const num_nonpage_local_qubits
            = static_cast<bit_integer_type>(state_ptr_->num_local_qubits() - state_ptr_->num_page_qubits());

          auto const page_index
            = ::ket::mpi::page::transpage_index_to_page_index(
                index, mapped_permutated_nonpage_qubit_first_, mapped_permutated_nonpage_qubit_last_,
                permutated_operated_page_qubit_first_, base_page_index_, num_nonpage_local_qubits);

          auto const nonpage_index
            = ::ket::mpi::page::nonpage_index(
                mapped_permutated_nonpage_qubit_first_, mapped_permutated_nonpage_qubit_last_,
                index, mapped_nonpage_qubits_bits_, num_nonpage_local_qubits);

          using std::begin;
          return *(begin(state_ptr_->page_range(std::make_pair(data_block_index_, page_index))) + nonpage_index);
        }

        auto operator++() noexcept -> transpage_iterator& { ++index_; return *this; }
        auto operator++(int) noexcept -> transpage_iterator { auto result = *this; ++*this; return result; }
        auto operator--() noexcept -> transpage_iterator& { --index_; return *this; }
        auto operator--(int) noexcept -> transpage_iterator { auto result = *this; --*this; return result; }
        auto operator+=(difference_type const n) noexcept -> transpage_iterator& { index_ += n; return *this; }
        auto operator-=(difference_type const n) noexcept -> transpage_iterator& { index_ -= n; return *this; }
        auto operator-(transpage_iterator const& other) const noexcept -> difference_type { return index_ - other.index_; }

        auto swap(transpage_iterator& other) noexcept -> void
        {
          using std::swap;
          swap(state_ptr_, other.state_ptr_);
          swap(mapped_permutated_nonpage_qubit_first_, other.mapped_permutated_nonpage_qubit_first_);
          swap(mapped_permutated_nonpage_qubit_last_, other.mapped_permutated_nonpage_qubit_last_);
          swap(permutated_operated_page_qubit_first_, other.permutated_operated_page_qubit_first_);
          swap(data_block_index_, other.data_block_index_);
          swap(base_page_index_, other.base_page_index_);
          swap(mapped_nonpage_qubits_bits_, other.mapped_nonpage_qubits_bits_);
          swap(index_, other.index_);
        }

       private:
        auto last_index(State& state) const noexcept -> difference_type
        {
          auto const& tmp_page_range
            = state.page_range(std::make_pair(state_integer_type{0u}, state_integer_type{0u}));

          using std::begin;
          using std::end;
          return static_cast<difference_type>(end(tmp_page_range) - begin(tmp_page_range));
        }
      }; // class transpage_iterator<State, PermutatedQubitIterator1, PermutatedQubitIterator2>

      template <typename State, typename PermutatedQubitIterator1, typename PermutatedQubitIterator2>
      inline auto operator!=(
        ::ket::mpi::page::transpage_iterator<State, PermutatedQubitIterator1, PermutatedQubitIterator2> const& lhs,
        ::ket::mpi::page::transpage_iterator<State, PermutatedQubitIterator1, PermutatedQubitIterator2> const& rhs)
      -> bool
      { return not (lhs == rhs); }

      template <typename State, typename PermutatedQubitIterator1, typename PermutatedQubitIterator2>
      inline auto operator>(
        ::ket::mpi::page::transpage_iterator<State, PermutatedQubitIterator1, PermutatedQubitIterator2> const& lhs,
        ::ket::mpi::page::transpage_iterator<State, PermutatedQubitIterator1, PermutatedQubitIterator2> const& rhs)
      -> bool
      { return rhs < lhs; }

      template <typename State, typename PermutatedQubitIterator1, typename PermutatedQubitIterator2>
      inline auto operator<=(
        ::ket::mpi::page::transpage_iterator<State, PermutatedQubitIterator1, PermutatedQubitIterator2> const& lhs,
        ::ket::mpi::page::transpage_iterator<State, PermutatedQubitIterator1, PermutatedQubitIterator2> const& rhs)
      -> bool
      { return not (lhs > rhs); }

      template <typename State, typename PermutatedQubitIterator1, typename PermutatedQubitIterator2>
      inline auto operator>=(
        ::ket::mpi::page::transpage_iterator<State, PermutatedQubitIterator1, PermutatedQubitIterator2> const& lhs,
        ::ket::mpi::page::transpage_iterator<State, PermutatedQubitIterator1, PermutatedQubitIterator2> const& rhs)
      -> bool
      { return not (lhs < rhs); }

      template <typename State, typename PermutatedQubitIterator1, typename PermutatedQubitIterator2>
      inline auto operator+(
        ::ket::mpi::page::transpage_iterator<State, PermutatedQubitIterator1, PermutatedQubitIterator2> iter,
        typename ::ket::mpi::page::transpage_iterator<State, PermutatedQubitIterator1, PermutatedQubitIterator2>::difference_type const n)
      -> ::ket::mpi::page::transpage_iterator<State, PermutatedQubitIterator1, PermutatedQubitIterator2>
      { return iter += n; }

      template <typename State, typename PermutatedQubitIterator1, typename PermutatedQubitIterator2>
      inline auto operator+(
        typename ::ket::mpi::page::transpage_iterator<State, PermutatedQubitIterator1, PermutatedQubitIterator2>::difference_type const n,
        ::ket::mpi::page::transpage_iterator<State, PermutatedQubitIterator1, PermutatedQubitIterator2> iter)
      -> ::ket::mpi::page::transpage_iterator<State, PermutatedQubitIterator1, PermutatedQubitIterator2>
      { return iter += n; }

      template <typename State, typename PermutatedQubitIterator1, typename PermutatedQubitIterator2>
      inline auto operator-(
        ::ket::mpi::page::transpage_iterator<State, PermutatedQubitIterator1, PermutatedQubitIterator2> iter,
        typename ::ket::mpi::page::transpage_iterator<State, PermutatedQubitIterator1, PermutatedQubitIterator2>::difference_type const n)
      -> ::ket::mpi::page::transpage_iterator<State, PermutatedQubitIterator1, PermutatedQubitIterator2>
      { return iter -= n; }

      template <typename State, typename PermutatedQubitIterator1, typename PermutatedQubitIterator2>
      inline auto swap(
        ::ket::mpi::page::transpage_iterator<State, PermutatedQubitIterator1, PermutatedQubitIterator2>& lhs,
        ::ket::mpi::page::transpage_iterator<State, PermutatedQubitIterator1, PermutatedQubitIterator2>& rhs) noexcept
      -> void
      { lhs.swap(rhs); }

      template <typename State, typename PermutatedQubitIterator1, typename PermutatedQubitIterator2>
      inline auto make_transpage_iterator(
        State& state,
        PermutatedQubitIterator1 const mapped_permutated_nonpage_qubit_first,
        PermutatedQubitIterator1 const mapped_permutated_nonpage_qubit_last,
        PermutatedQubitIterator2 const permutated_operated_page_qubit_first,
        PermutatedQubitIterator2 const permutated_operated_page_qubit_last,
        ::ket::meta::state_integer_t<typename std::iterator_traits<PermutatedQubitIterator1>::value_type> const data_block_index,
        ::ket::meta::state_integer_t<typename std::iterator_traits<PermutatedQubitIterator1>::value_type> const page_index_wo_qubits,
        ::ket::meta::state_integer_t<typename std::iterator_traits<PermutatedQubitIterator1>::value_type> const mapped_nonpage_qubit_bits,
        typename std::iterator_traits<PermutatedQubitIterator1>::difference_type const index
          = typename std::iterator_traits<PermutatedQubitIterator1>::difference_type{0u}) noexcept
      -> ::ket::mpi::page::transpage_iterator<State, PermutatedQubitIterator1, PermutatedQubitIterator2>
      {
        return {state,
          mapped_permutated_nonpage_qubit_first, mapped_permutated_nonpage_qubit_last,
          permutated_operated_page_qubit_first, permutated_operated_page_qubit_last,
          data_block_index, page_index_wo_qubits, mapped_nonpage_qubit_bits, index};
      }
    } // namespace page
  } // namespace mpi
} // namespace ket


#endif // KET_MPI_PAGE_TRANSPAGE_ITERATOR_HPP
