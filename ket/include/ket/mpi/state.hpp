#ifndef KET_MPI_STATE_HPP
# define KET_MPI_STATE_HPP

# include <cstddef>
# include <cassert>
# include <iterator>
# include <vector>
# include <algorithm>
# include <memory>
# include <utility>
# include <tuple>
# include <type_traits>
# if __cplusplus < 201703L
#   include <boost/type_traits/is_nothrow_swappable.hpp>
# endif
# include <array>
# include <initializer_list>

# include <boost/range/sub_range.hpp>
# include <boost/range/iterator_range.hpp>

# include <yampi/environment.hpp>
# include <yampi/buffer.hpp>
# include <yampi/datatype_base.hpp>
# include <yampi/communicator.hpp>
# include <yampi/rank.hpp>
# include <yampi/status.hpp>
# include <yampi/algorithm/swap.hpp>

# include <ket/qubit.hpp>
# include <ket/control.hpp>
# include <ket/utility/integer_exp2.hpp>
# include <ket/utility/loop_n.hpp>
# include <ket/utility/meta/ranges.hpp>
# include <ket/mpi/permutated.hpp>
# include <ket/mpi/qubit_permutation.hpp>
# include <ket/mpi/page/is_on_page.hpp>
# include <ket/mpi/page/page_size.hpp>
# include <ket/mpi/utility/simple_mpi.hpp>
# include <ket/mpi/utility/for_each_local_range.hpp>
# include <ket/mpi/utility/buffer_range.hpp>
# include <ket/mpi/utility/transform_inclusive_scan.hpp>
# include <ket/mpi/utility/transform_inclusive_scan_self.hpp>
# include <ket/mpi/utility/upper_bound.hpp>
# include <ket/mpi/utility/resize_buffer_if_empty.hpp>
# include <ket/mpi/utility/detail/swap_permutated_local_qubits.hpp>
# include <ket/mpi/utility/detail/for_each_in_diagonal_loop.hpp>
# include <ket/mpi/utility/detail/swap_local_data.hpp>

# if __cplusplus >= 201703L
#   define KET_is_nothrow_swappable std::is_nothrow_swappable
# else
#   define KET_is_nothrow_swappable boost::is_nothrow_swappable
# endif


namespace ket
{
  namespace mpi
  {
    namespace state_detail
    {
      template <typename State>
      class state_iterator
      {
       public:
        using value_type = typename State::value_type;
        using difference_type = std::ptrdiff_t;
        using pointer = value_type*;
        using reference = value_type&;
        using iterator_category = std::random_access_iterator_tag;

       private:
        State* state_ptr_;
        difference_type index_;

       public:
        constexpr state_iterator() noexcept
          : state_ptr_{nullptr}, index_{}
        { }

        constexpr state_iterator(State& state, difference_type const index) noexcept
          : state_ptr_{std::addressof(state)}, index_{index}
        { }

        auto operator==(state_iterator const& other) const noexcept -> bool
        { assert(state_ptr_ == other.state_ptr_); return index_ == other.index_; }

        auto operator<(state_iterator const& other) const noexcept -> bool
        { assert(state_ptr_ == other.state_ptr_); return index_ < other.index_; }

        auto operator*() const noexcept -> reference
        { return const_cast<reference>((*state_ptr_)[index_]); }

        auto operator[](difference_type const n) const -> reference
        { return const_cast<reference>((*state_ptr_)[index_ + n]); }

        auto operator++() noexcept -> state_iterator& { ++index_; return *this; }
        auto operator++(int) noexcept -> state_iterator { auto result = *this; ++*this; return result; }
        auto operator--() noexcept -> state_iterator& { --index_; return *this; }
        auto operator--(int) noexcept -> state_iterator { auto result = *this; --*this; return result; }
        auto operator+=(difference_type const n) noexcept -> state_iterator& { index_ += n; return *this; }
        auto operator-=(difference_type const n) noexcept -> state_iterator& { index_ -= n; return *this; }
        auto operator-(state_iterator const& other) const noexcept -> difference_type { return index_ - other.index_; }

        auto swap(state_iterator& other) noexcept -> void
        {
          using std::swap;
          swap(state_ptr_, other.state_ptr_);
          swap(index_, other.index_);
        }
      }; // class state_iterator<State>

      template <typename State>
      inline auto operator!=(
        ::ket::mpi::state_detail::state_iterator<State> const& lhs,
        ::ket::mpi::state_detail::state_iterator<State> const& rhs)
      -> bool
      { return not (lhs == rhs); }

      template <typename State>
      inline auto operator>(
        ::ket::mpi::state_detail::state_iterator<State> const& lhs,
        ::ket::mpi::state_detail::state_iterator<State> const& rhs)
      -> bool
      { return rhs < lhs; }

      template <typename State>
      inline auto operator<=(
        ::ket::mpi::state_detail::state_iterator<State> const& lhs,
        ::ket::mpi::state_detail::state_iterator<State> const& rhs)
      -> bool
      { return not (lhs > rhs); }

      template <typename State>
      inline auto operator>=(
        ::ket::mpi::state_detail::state_iterator<State> const& lhs,
        ::ket::mpi::state_detail::state_iterator<State> const& rhs)
      -> bool
      { return not (lhs < rhs); }

      template <typename State>
      inline auto operator+(
        ::ket::mpi::state_detail::state_iterator<State> iter,
        typename ::ket::mpi::state_detail::state_iterator<State>::difference_type const n)
      -> ::ket::mpi::state_detail::state_iterator<State>
      { return iter += n; }

      template <typename State>
      inline auto operator+(
        typename ::ket::mpi::state_detail::state_iterator<State>::difference_type const n,
        ::ket::mpi::state_detail::state_iterator<State> iter)
      -> ::ket::mpi::state_detail::state_iterator<State>
      { return iter += n; }

      template <typename State>
      inline auto operator-(
        ::ket::mpi::state_detail::state_iterator<State> iter,
        typename ::ket::mpi::state_detail::state_iterator<State>::difference_type const n)
      -> ::ket::mpi::state_detail::state_iterator<State>
      { return iter -= n; }

      template <typename State>
      inline auto swap(
        ::ket::mpi::state_detail::state_iterator<State>& lhs,
        ::ket::mpi::state_detail::state_iterator<State>& rhs) noexcept
      -> void
      { lhs.swap(rhs); }
    } // namespace state_detail


    template <typename Complex, bool has_page_qubits = true, typename Allocator = std::allocator<Complex>>
    class state
    {
     public:
      using value_type = Complex;
      using allocator_type = typename Allocator::template rebind<value_type>::other;

     private:
      using data_type = std::vector<value_type, allocator_type>;
      data_type data_;

     public:
      using page_range_type = boost::iterator_range< ::ket::utility::meta::iterator_t<data_type> >;

     private:
      std::size_t num_local_qubits_;
      std::size_t num_page_qubits_;
      std::size_t num_pages_; // 1u << num_page_qubits_
      std::size_t num_data_blocks_;
      std::vector<page_range_type> page_ranges_;
      page_range_type buffer_range_;

     public:
      using size_type = typename data_type::size_type;
      using difference_type = typename data_type::difference_type;
      using reference = typename data_type::reference;
      using const_reference = typename data_type::const_reference;
      using pointer = typename data_type::pointer;
      using const_pointer = typename data_type::const_pointer;
      using iterator = ::ket::mpi::state_detail::state_iterator<state>;
      using const_iterator = ::ket::mpi::state_detail::state_iterator<state const>;
      using reverse_iterator = std::reverse_iterator<iterator>;
      using const_reverse_iterator = std::reverse_iterator<const_iterator>;

      state() = delete;
      ~state() noexcept = default;
      state(state const&) = default;
      state& operator=(state const&) = default;
      state(state&&) = default;
      state& operator=(state&&) = default;

      state(state const& other, allocator_type const& allocator)
        : data_{other.data_, allocator},
          num_local_qubits_{other.num_local_qubits_},
          num_page_qubits_{other.num_page_qubits_},
          num_pages_{other.num_pages_},
          num_data_blocks_{other.num_data_blocks_},
          page_ranges_{other.page_ranges_},
          buffer_range_{other.buffer_range_}
      { }

      state(state&& other, allocator_type const& allocator)
        : data_{std::move(other.data_), allocator},
          num_local_qubits_{std::move(other.num_local_qubits_)},
          num_page_qubits_{std::move(other.num_page_qubits_)},
          num_pages_{std::move(other.num_pages_)},
          num_data_blocks_{std::move(other.num_data_blocks_)},
          page_ranges_{std::move(other.page_ranges_)},
          buffer_range_{std::move(other.buffer_range_)}
      { }

      state(std::initializer_list<value_type> initializer_list, allocator_type const& allocator = allocator_type())
        : data_{generate_initial_data(initializer_list, std::size_t{2u}, std::size_t{1u}, allocator)},
          num_local_qubits_{::ket::utility::integer_log2(initializer_list.size())},
          num_page_qubits_{std::size_t{1u}},
          num_pages_{std::size_t{2u}},
          num_data_blocks_{std::size_t{1u}},
          page_ranges_{generate_initial_page_ranges(data_, num_pages_, num_data_blocks_)},
          buffer_range_{generate_initial_buffer_range(data_, num_pages_, num_data_blocks_)}
      {
        assert(::ket::utility::integer_exp2<std::size_t>(num_local_qubits_) == initializer_list.size());
        assert(num_local_qubits_ > num_page_qubits_);
      }

      template <typename BitInteger>
      state(std::initializer_list<value_type> initializer_list, BitInteger const num_page_qubits, allocator_type const& allocator = allocator_type())
        : data_{generate_initial_data(initializer_list, std::size_t{1u} << num_page_qubits, std::size_t{1u}, allocator)},
          num_local_qubits_{::ket::utility::integer_log2(initializer_list.size())},
          num_page_qubits_{static_cast<std::size_t>(num_page_qubits)},
          num_pages_{std::size_t{1u} << num_page_qubits},
          num_data_blocks_{std::size_t{1u}},
          page_ranges_{generate_initial_page_ranges(data_, num_pages_, num_data_blocks_)},
          buffer_range_{generate_initial_buffer_range(data_, num_pages_, num_data_blocks_)}
      {
        assert(::ket::utility::integer_exp2<std::size_t>(num_local_qubits_) == initializer_list.size());
        assert(num_page_qubits_ >= BitInteger{1u} and num_local_qubits_ > num_page_qubits_);
      }

      template <typename BitInteger, typename StateInteger>
      state(std::initializer_list<value_type> initializer_list, BitInteger const num_page_qubits, StateInteger const num_data_blocks, allocator_type const& allocator = allocator_type())
        : data_{generate_initial_data(initializer_list, std::size_t{1u} << num_page_qubits, static_cast<std::size_t>(num_data_blocks), allocator)},
          num_local_qubits_{::ket::utility::integer_log2(initializer_list.size() / num_data_blocks)},
          num_page_qubits_{static_cast<std::size_t>(num_page_qubits)},
          num_pages_{std::size_t{1u} << num_page_qubits},
          num_data_blocks_{static_cast<std::size_t>(num_data_blocks)},
          page_ranges_{generate_initial_page_ranges(data_, num_pages_, num_data_blocks_)},
          buffer_range_{generate_initial_buffer_range(data_, num_pages_, num_data_blocks_)}
      {
        assert(::ket::utility::integer_exp2<std::size_t>(num_local_qubits_) * num_data_blocks_ == initializer_list.size());
        assert(num_page_qubits_ >= BitInteger{1u} and num_local_qubits_ > num_page_qubits_);
      }

      template <typename BitInteger, typename StateInteger, typename PermutationAllocator>
      state(
        BitInteger const num_local_qubits, BitInteger const num_page_qubits,
        StateInteger const initial_integer,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, PermutationAllocator> const& permutation,
        yampi::communicator const& communicator, yampi::environment const& environment)
        : data_{generate_initial_data(
            ::ket::mpi::utility::policy::make_simple_mpi(),
            num_local_qubits, StateInteger{1u} << num_page_qubits,
            initial_integer, permutation, communicator, environment)},
          num_local_qubits_{static_cast<std::size_t>(num_local_qubits)},
          num_page_qubits_{static_cast<std::size_t>(num_page_qubits)},
          num_pages_{std::size_t{1u} << num_page_qubits},
          num_data_blocks_{std::size_t{1u}},
          page_ranges_{generate_initial_page_ranges(data_, num_pages_, num_data_blocks_)},
          buffer_range_{generate_initial_buffer_range(data_, num_pages_, num_data_blocks_)}
      { assert(num_page_qubits_ >= BitInteger{1u} and num_local_qubits_ > num_page_qubits_); }

      template <typename MpiPolicy, typename BitInteger, typename StateInteger, typename PermutationAllocator>
      state(
        MpiPolicy const& mpi_policy,
        BitInteger const num_local_qubits, BitInteger const num_page_qubits,
        StateInteger const initial_integer,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, PermutationAllocator> const& permutation,
        yampi::communicator const& communicator, yampi::environment const& environment)
        : data_{generate_initial_data(
            mpi_policy, num_local_qubits, StateInteger{1u} << num_page_qubits, initial_integer, permutation, communicator, environment)},
          num_local_qubits_{static_cast<std::size_t>(num_local_qubits)},
          num_page_qubits_{static_cast<std::size_t>(num_page_qubits)},
          num_pages_{std::size_t{1u} << num_page_qubits},
          num_data_blocks_{static_cast<std::size_t>(::ket::mpi::utility::policy::num_data_blocks(mpi_policy, communicator, environment))},
          page_ranges_{generate_initial_page_ranges(data_, num_pages_, num_data_blocks_)},
          buffer_range_{generate_initial_buffer_range(data_, num_pages_, num_data_blocks_)}
      { assert(num_page_qubits_ >= BitInteger{1u} and num_local_qubits_ > num_page_qubits_); }

      auto assign(std::initializer_list<value_type> initializer_list) -> void
      { assign(initializer_list, std::size_t{1u}, std::size_t{1u}); }

      template <typename BitInteger>
      auto assign(std::initializer_list<value_type> initializer_list, BitInteger const num_page_qubits) -> void
      { assign(initializer_list, num_page_qubits, std::size_t{1u}); }

      template <typename BitInteger, typename StateInteger>
      auto assign(std::initializer_list<value_type> initializer_list, BitInteger const num_page_qubits, StateInteger const num_data_blocks) -> void
      {
        initialize_data(data_, initializer_list, std::size_t{1u} << num_page_qubits, static_cast<std::size_t>(num_data_blocks));

        num_local_qubits_ = ::ket::utility::integer_log2(initializer_list.size() / num_data_blocks);
        num_page_qubits_ = static_cast<std::size_t>(num_page_qubits);
        num_pages_ = std::size_t{1u} << num_page_qubits;
        num_data_blocks_ = static_cast<std::size_t>(num_data_blocks);

        assert(::ket::utility::integer_exp2<std::size_t>(num_local_qubits_) * num_data_blocks_ == initializer_list.size());
        assert(num_page_qubits_ >= BitInteger{1u} and num_local_qubits_ > num_page_qubits_);

        page_ranges_ = generate_initial_page_ranges(data_, num_pages_, num_data_blocks_);
        buffer_range_ = generate_initial_buffer_range(data_, num_pages_, num_data_blocks_);
      }

      template <typename BitInteger, typename StateInteger, typename PermutationAllocator>
      auto assign(
        BitInteger const num_local_qubits, BitInteger const num_page_qubits,
        StateInteger const initial_integer,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, PermutationAllocator> const& permutation,
        yampi::communicator const& communicator, yampi::environment const& environment)
      -> void
      { assign(::ket::mpi::utility::policy::make_simple_mpi(), num_local_qubits, num_page_qubits, initial_integer, permutation, communicator, environment); }

      template <typename MpiPolicy, typename BitInteger, typename StateInteger, typename PermutationAllocator>
      auto assign(
        MpiPolicy const& mpi_policy,
        BitInteger const num_local_qubits, BitInteger const num_page_qubits,
        StateInteger const initial_integer,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, PermutationAllocator> const& permutation,
        yampi::communicator const& communicator, yampi::environment const& environment)
      -> void
      {
        initialize_data(data_, mpi_policy, num_local_qubits, StateInteger{1u} << num_page_qubits, initial_integer, permutation, communicator, environment);

        num_local_qubits_ = static_cast<std::size_t>(num_local_qubits);
        num_page_qubits_ = static_cast<std::size_t>(num_page_qubits);
        num_pages_ = std::size_t{1u} << num_page_qubits;
        num_data_blocks_ = static_cast<std::size_t>(::ket::mpi::utility::policy::num_data_blocks(mpi_policy, communicator, environment));

        assert(num_page_qubits_ >= BitInteger{1u} and num_local_qubits_ > num_page_qubits_);

        page_ranges_ = generate_initial_page_ranges(data_, num_pages_, num_data_blocks_);
        buffer_range_ = generate_initial_buffer_range(data_, num_pages_, num_data_blocks_);
      }

      template <typename PairOrTuple>
      auto page_range_index(PairOrTuple const& data_block_page_etc_indices) const -> size_type
      {
        auto const data_block_index = std::get<0u>(data_block_page_etc_indices);
        auto const page_index = std::get<1u>(data_block_page_etc_indices);
        assert(data_block_index >= decltype(data_block_index){0u} and data_block_index < num_data_blocks_);
        assert(page_index >= decltype(page_index){0u} and page_index < num_pages_);
        return static_cast<size_type>(data_block_index * num_pages_ + page_index);
      }

      auto data_block_page_indices(size_type const page_range_index) const -> std::pair<size_type, size_type>
      {
        assert(page_range_index >= size_type{0u} and page_range_index < num_data_blocks_ * num_pages_);
        return {page_range_index / num_pages_, page_range_index % num_pages_};
      }

      template <typename DataBlockIndex, typename PageIndex>
      auto swap_pages(
        std::pair<DataBlockIndex, PageIndex> const& data_block_page_indices1,
        std::pair<DataBlockIndex, PageIndex> const& data_block_page_indices2)
      -> void
      {
        assert(data_block_page_indices1 != data_block_page_indices2);
        using std::swap;
        swap(
          page_ranges_[page_range_index(data_block_page_indices1)],
          page_ranges_[page_range_index(data_block_page_indices2)]);
      }

      template <typename DataBlockIndex, typename PageIndex>
      auto swap_buffer_and_page(std::pair<DataBlockIndex, PageIndex> const& data_block_page_indices) -> void
      {
        using std::swap;
        swap(buffer_range_, page_ranges_[page_range_index(data_block_page_indices)]);
      }

      template <typename DataBlockIndex, typename PageIndex, typename NonpageIndex>
      auto swap_values(
        std::tuple<DataBlockIndex, PageIndex, NonpageIndex> const& data_block_page_nonpage_indices1,
        std::tuple<DataBlockIndex, PageIndex, NonpageIndex> const& data_block_page_nonpage_indices2)
      -> void
      {
        assert(
          std::get<0u>(data_block_page_nonpage_indices1) != std::get<0u>(data_block_page_nonpage_indices2)
          or std::get<1u>(data_block_page_nonpage_indices1) != std::get<1u>(data_block_page_nonpage_indices2));

        auto const nonpage_index1 = std::get<2u>(data_block_page_nonpage_indices1);
        auto const nonpage_index2 = std::get<2u>(data_block_page_nonpage_indices2);
        assert(
          nonpage_index1 >= decltype(nonpage_index1){0u}
          and nonpage_index1 < ::ket::utility::integer_exp2<size_type>(num_local_qubits_ - num_page_qubits_));
        assert(
          nonpage_index2 >= decltype(nonpage_index2){0u}
          and nonpage_index2 < ::ket::utility::integer_exp2<size_type>(num_local_qubits_ - num_page_qubits_));

        using std::swap;
        using std::begin;
        swap(
          begin(page_ranges_[page_range_index(data_block_page_nonpage_indices1)])[nonpage_index1],
          begin(page_ranges_[page_range_index(data_block_page_nonpage_indices2)])[nonpage_index2]);
      }

      template <typename DataBlockIndex, typename PageIndex>
      auto page_range(std::pair<DataBlockIndex, PageIndex> const& data_block_page_indices) const -> page_range_type const&
      { return page_ranges_[page_range_index(data_block_page_indices)]; }

      auto buffer_range() const -> page_range_type const& { return buffer_range_; }

      auto num_local_qubits() const noexcept -> std::size_t { assert(num_local_qubits_ > num_page_qubits_); return num_local_qubits_; }
      auto num_page_qubits() const noexcept -> std::size_t { assert(num_page_qubits_ >= std::size_t{1u}); return num_page_qubits_; }
      auto num_pages() const noexcept -> std::size_t { assert(num_pages_ >= std::size_t{2u}); return num_pages_; }
      auto num_data_blocks() const noexcept -> std::size_t { assert(num_data_blocks_ >= std::size_t{1u}); return num_data_blocks_; }

      auto operator==(state const& other) const noexcept -> bool
      { return num_local_qubits_ == other.num_local_qubits_ and num_data_blocks_ == other.num_data_blocks_ and std::equal(begin(), end(), other.begin()); }
      auto operator<(state const& other) const noexcept -> bool { return std::lexicographical_compare(begin(), end(), other.begin(), other.end()); }

      // Element access
      auto at(size_type const index) -> reference
      {
        using std::begin;
        return data_.at(
          (begin(page_ranges_[page_range_index(get_data_block_page_indices(index))]) - begin(data_))
          + get_nonpage_index(index));
      }

      auto at(size_type const index) const -> const_reference
      {
        using std::begin;
        return data_.at(
          (begin(page_ranges_[page_range_index(get_data_block_page_indices(index))]) - begin(data_))
          + get_nonpage_index(index));
      }

      auto operator[](size_type const index) -> reference
      {
        assert(index < ::ket::utility::integer_exp2<size_type>(num_local_qubits_) * num_data_blocks_);
        using std::begin;
        return begin(page_ranges_[page_range_index(get_data_block_page_indices(index))])[get_nonpage_index(index)];
      }

      auto operator[](size_type const index) const -> const_reference
      {
        assert(index < ::ket::utility::integer_exp2<size_type>(num_local_qubits_) * num_data_blocks_);
        using std::begin;
        return begin(page_ranges_[page_range_index(get_data_block_page_indices(index))])[get_nonpage_index(index)];
      }

      auto front() -> reference { using std::begin; return *begin(page_ranges_[page_range_index(get_data_block_page_indices(0u))]); }
      auto front() const -> const_reference { using std::begin; return *begin(page_ranges_[page_range_index(get_data_block_page_indices(0u))]); }

      auto back() -> reference { using std::end; return *--end(page_ranges_[page_range_index(get_data_block_page_indices((1u << num_local_qubits_) - 1u))]); }
      auto back() const -> const_reference { using std::end; return *--end(page_ranges_[page_range_index(get_data_block_page_indices((1u << num_local_qubits_) - 1u))]); }

      // Iterators
      auto begin() noexcept -> iterator { return iterator{*this, 0}; }
      auto begin() const noexcept -> const_iterator { return const_iterator{*this, 0}; }
      auto cbegin() const noexcept -> const_iterator { return const_iterator{*this, 0}; }
      auto end() noexcept -> iterator { return iterator{*this, ::ket::utility::integer_exp2<int>(num_local_qubits_) * static_cast<int>(num_data_blocks_)}; }
      auto end() const noexcept -> const_iterator { return const_iterator{*this, ::ket::utility::integer_exp2<int>(num_local_qubits_) * static_cast<int>(num_data_blocks_)}; }
      auto cend() const noexcept -> const_iterator { return const_iterator{*this, ::ket::utility::integer_exp2<int>(num_local_qubits_) * static_cast<int>(num_data_blocks_)}; }
      auto rbegin() noexcept -> reverse_iterator { return reverse_iterator{this->end()}; }
      auto rbegin() const noexcept -> const_reverse_iterator { return const_reverse_iterator{this->end()}; }
      auto crbegin() const noexcept -> const_reverse_iterator { return const_reverse_iterator{this->cend()}; }
      auto rend() noexcept -> reverse_iterator { return reverse_iterator{this->begin()}; }
      auto rend() const noexcept -> const_reverse_iterator { return const_reverse_iterator{this->begin()}; }
      auto crend() const noexcept -> const_reverse_iterator { return const_reverse_iterator{this->cbegin()}; }

      // Capacity
      auto size() const noexcept -> size_type { using std::begin; using std::end; return data_.size() - static_cast<size_type>(std::distance(begin(buffer_range_), end(buffer_range_))); }
      auto max_size() const noexcept -> size_type { using std::begin; using std::end; return data_.max_size() - static_cast<size_type>(std::distance(begin(buffer_range_), end(buffer_range_))); }
      auto reserve(size_type const new_capacity) -> void { using std::begin; using std::end; data_.reserve(new_capacity + static_cast<size_type>(std::distance(begin(buffer_range_), end(buffer_range_)))); }
      auto capacity() const noexcept -> size_type { using std::begin; using std::end; return data_.capacity() - static_cast<size_type>(std::distance(begin(buffer_range_), end(buffer_range_))); }
      auto shrink_to_fit() -> void { data_.shrink_to_fit(); }

      // Modifiers
      auto swap(state& other)
      noexcept(
        KET_is_nothrow_swappable<data_type>::value
        and KET_is_nothrow_swappable<std::size_t>::value
        and KET_is_nothrow_swappable<std::vector<page_range_type>>::value
        and KET_is_nothrow_swappable<page_range_type>::value )
      -> void
      {
        using std::swap;
        swap(data_, other.data_);

        swap(num_local_qubits_, other.num_local_qubits_);
        swap(num_page_qubits_, other.num_page_qubits_);
        swap(num_pages_, other.num_pages_);
        swap(num_data_blocks_, other.num_data_blocks_);
        swap(page_ranges_, other.page_ranges_);
        swap(buffer_range_, other.buffer_range_);
      }

     private:
      auto initialize_data(
        data_type& data,
        std::initializer_list<value_type> initializer_list,
        std::size_t const num_pages, std::size_t const num_data_blocks) const
      -> void
      {
        auto const state_size = initializer_list.size();
        auto const data_size = state_size + state_size / num_pages / num_data_blocks;

        assert(state_size % (num_pages * num_data_blocks) == 0);

        data.clear();
        data.reserve(data_size);
        data.assign(initializer_list);
        data.resize(data_size);
      }

      template <
        typename MpiPolicy, typename BitInteger, typename StateInteger,
        typename PermutationAllocator>
      auto initialize_data(
        data_type& data,
        MpiPolicy const& mpi_policy,
        BitInteger const num_local_qubits, StateInteger const num_pages,
        StateInteger const initial_integer,
        ::ket::mpi::qubit_permutation<
          StateInteger, BitInteger, PermutationAllocator> const&
          permutation,
        yampi::communicator const& communicator,
        yampi::environment const& environment) const
      -> void
      {
        auto const data_block_size = ::ket::utility::integer_exp2<std::size_t>(num_local_qubits);
        auto const num_data_blocks = ::ket::mpi::utility::policy::num_data_blocks(mpi_policy, communicator, environment);
        auto const state_size = data_block_size * static_cast<std::size_t>(num_data_blocks);
        auto const data_size = state_size + data_block_size / static_cast<std::size_t>(num_pages);

        assert(state_size % (num_pages * num_data_blocks) == 0);

        data.clear();
        data.reserve(data_size);
        data.assign(state_size, value_type{0});

        auto const rank_index
          = ::ket::mpi::utility::qubit_value_to_rank_index(
              mpi_policy, data, ::ket::mpi::permutate_bits(permutation, initial_integer),
              communicator, environment);

        if (communicator.rank(environment) == rank_index.first)
          data[rank_index.second] = value_type{1};

        data.resize(data_size);
      }

      auto generate_initial_data(
        std::initializer_list<value_type> initializer_list,
        std::size_t const num_pages, std::size_t const num_data_blocks,
        allocator_type const& allocator) const
      -> data_type
      {
        auto result = data_type{allocator};
        initialize_data(result, initializer_list, num_pages, num_data_blocks);
        return result;
      }

      template <
        typename MpiPolicy, typename BitInteger, typename StateInteger,
        typename PermutationAllocator>
      auto generate_initial_data(
        MpiPolicy const& mpi_policy,
        BitInteger const num_local_qubits, StateInteger const num_pages,
        StateInteger const initial_integer,
        ::ket::mpi::qubit_permutation<
          StateInteger, BitInteger, PermutationAllocator> const&
          permutation,
        yampi::communicator const& communicator,
        yampi::environment const& environment) const
      -> data_type
      {
        auto result = data_type{};
        initialize_data(result, mpi_policy, num_local_qubits, num_pages, initial_integer, permutation, communicator, environment);
        return result;
      }

      auto generate_initial_page_ranges(data_type& data, std::size_t const num_pages, std::size_t const num_data_blocks) const
      -> std::vector<page_range_type>
      {
        assert(data.size() % (num_pages * num_data_blocks + 1u) == 0u);
        auto const page_size = static_cast<size_type>(data.size() / (num_pages * num_data_blocks + 1u));

        auto result = std::vector<page_range_type>{};
        result.reserve(num_pages * num_data_blocks);
        using std::begin;
        for (auto page_range_index = std::size_t{0u}; page_range_index < num_pages * num_data_blocks; ++page_range_index)
          result.push_back(
            boost::make_iterator_range(
              begin(data) + page_range_index * page_size,
              begin(data) + (page_range_index + 1u) * page_size));

        return result;
      }

      auto generate_initial_buffer_range(data_type& data, std::size_t const num_pages, std::size_t const num_data_blocks) const -> page_range_type
      {
        assert(data.size() % (num_pages * num_data_blocks + 1u) == 0u);
        auto const page_size = static_cast<size_type>(data.size() / (num_pages * num_data_blocks + 1u));

        using std::begin;
        return boost::make_iterator_range(
          begin(data) + num_pages * num_data_blocks * page_size,
          begin(data) + (num_pages * num_data_blocks + 1u) * page_size);
      }

     public:
      auto get_data_block_page_indices(size_type const index) const -> std::pair<size_type, size_type>
      {
        auto const data_block_size = ::ket::utility::integer_exp2<size_type>(num_local_qubits_);
        assert(index < data_block_size * num_data_blocks_);

        auto const num_nonpage_local_qubits = num_local_qubits_ - num_page_qubits_;
        return std::make_pair(
          index / data_block_size,
          (((num_pages_ - 1u) << num_nonpage_local_qubits) bitand (index % data_block_size)) >> num_nonpage_local_qubits);
      }

      auto get_nonpage_index(size_type const index) const -> size_type
      {
        auto const data_block_size = ::ket::utility::integer_exp2<size_type>(num_local_qubits_);
        assert(index < data_block_size * num_data_blocks_);

        auto const num_nonpage_local_qubits = num_local_qubits_ - num_page_qubits_;
        return (compl ((num_pages_ - 1u) << num_nonpage_local_qubits)) bitand (index % data_block_size);
      }
    }; // class state<Complex, has_page_qubits, Allocator>

    template <typename StateInteger, typename BitInteger, typename Complex, bool has_page_qubits, typename Allocator>
    inline auto is_page_qubit(
      ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_qubit,
      ::ket::mpi::state<Complex, has_page_qubits, Allocator> const& local_state)
    -> bool
    {
      auto const num_local_qubits = static_cast<BitInteger>(local_state.num_local_qubits());
      auto const num_page_qubits = static_cast<BitInteger>(local_state.num_page_qubits());
      return
        permutated_qubit >= ::ket::mpi::make_permutated(::ket::make_qubit<StateInteger>(num_local_qubits - num_page_qubits))
        and permutated_qubit < ::ket::mpi::make_permutated(::ket::make_qubit<StateInteger>(num_local_qubits));
    }

    template <typename StateInteger, typename BitInteger, typename Complex, bool has_page_qubits, typename Allocator>
    inline auto is_page_qubit(
      ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const permutated_control_qubit,
      ::ket::mpi::state<Complex, has_page_qubits, Allocator> const& local_state)
    -> bool
    { return ::ket::mpi::is_page_qubit(::ket::mpi::remove_control(permutated_control_qubit), local_state); }

    namespace state_detail
    {
      template <
        typename Complex, bool has_page_qubits, typename Allocator,
        typename StateInteger, typename BitInteger>
      inline auto interpage_swap(
        ::ket::mpi::state<Complex, has_page_qubits, Allocator>& local_state,
        ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_qubit1,
        ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_qubit2)
      -> void
      {
        assert(local_state.num_page_qubits() >= 2u);
        assert(::ket::mpi::is_page_qubit(permutated_qubit1, local_state) and ::ket::mpi::is_page_qubit(permutated_qubit2, local_state));
        assert(permutated_qubit1 != permutated_qubit2);

        auto const num_nonpage_local_qubits
          = static_cast<BitInteger>(local_state.num_local_qubits() - local_state.num_page_qubits());
        auto const minmax_permutated_qubits = std::minmax(permutated_qubit1, permutated_qubit2);
        auto const lower_bits_mask
          = ::ket::utility::integer_exp2<StateInteger>(minmax_permutated_qubits.first - num_nonpage_local_qubits)
            - StateInteger{1u};
        auto const middle_bits_mask
          = (::ket::utility::integer_exp2<StateInteger>(
               minmax_permutated_qubits.second - (num_nonpage_local_qubits + BitInteger{1u}))
             - StateInteger{1u})
            xor lower_bits_mask;
        auto const upper_bits_mask = compl (lower_bits_mask bitor middle_bits_mask);

        for (auto value_wo_qubits = StateInteger{0u};
             value_wo_qubits < ::ket::utility::integer_exp2<StateInteger>(static_cast<StateInteger>(local_state.num_page_qubits() - 2u));
             ++value_wo_qubits)
        {
          auto const base_page_index
            = ((value_wo_qubits bitand upper_bits_mask) << 2u)
              bitor ((value_wo_qubits bitand middle_bits_mask) << 1u)
              bitor (value_wo_qubits bitand lower_bits_mask);
          auto const page_index1
            = base_page_index bitor (StateInteger{1u} << (permutated_qubit1 - num_nonpage_local_qubits));
          auto const page_index2
            = base_page_index bitor (StateInteger{1u} << (permutated_qubit2 - num_nonpage_local_qubits));

          for (auto data_block_index = StateInteger{0u};
               data_block_index < local_state.num_data_blocks(); ++data_block_index)
            local_state.swap_pages(
              std::make_pair(data_block_index, page_index1),
              std::make_pair(data_block_index, page_index2));
        }
      }

      template <
        typename ParallelPolicy, typename Complex, bool has_page_qubits, typename Allocator,
        typename StateInteger, typename BitInteger>
      inline auto swap_page_and_nonpage_qubits(
        ParallelPolicy const parallel_policy,
        ::ket::mpi::state<Complex, has_page_qubits, Allocator>& local_state,
        ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_qubit1,
        ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_qubit2)
      -> void
      {
        assert(
          (::ket::mpi::is_page_qubit(permutated_qubit1, local_state) and (not ::ket::mpi::is_page_qubit(permutated_qubit2, local_state)))
          or ((not ::ket::mpi::is_page_qubit(permutated_qubit1, local_state)) and ::ket::mpi::is_page_qubit(permutated_qubit2, local_state)));
# ifndef NDEBUG
        using permutated_qubit_type = ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> >;
# endif
        assert(permutated_qubit1 < permutated_qubit_type{local_state.num_local_qubits()});
        assert(permutated_qubit2 < permutated_qubit_type{local_state.num_local_qubits()});

        auto const num_nonpage_local_qubits
          = static_cast<BitInteger>(local_state.num_local_qubits() - local_state.num_page_qubits());
        auto const minmax_permutated_qubits = std::minmax(permutated_qubit1, permutated_qubit2);
        auto const nonpage_lower_bits_mask
          = ::ket::utility::integer_exp2<StateInteger>(minmax_permutated_qubits.first) - StateInteger{1u};
        auto const nonpage_upper_bits_mask
          = (::ket::utility::integer_exp2<StateInteger>(num_nonpage_local_qubits - 1u) - StateInteger{1u})
            xor nonpage_lower_bits_mask;
        auto const page_lower_bits_mask
          = ::ket::utility::integer_exp2<StateInteger>(minmax_permutated_qubits.second - num_nonpage_local_qubits)
            - StateInteger{1u};
        auto const page_upper_bits_mask
          = (::ket::utility::integer_exp2<StateInteger>(local_state.num_page_qubits() - 1u) - StateInteger{1u})
            xor page_lower_bits_mask;

        for (auto page_value_wo_qubits = StateInteger{0u};
             page_value_wo_qubits < ::ket::utility::integer_exp2<StateInteger>(static_cast<StateInteger>(local_state.num_page_qubits() - 1u));
             ++page_value_wo_qubits)
        {
          auto const page_index0
            = ((page_value_wo_qubits bitand page_upper_bits_mask) << 1u)
              bitor (page_value_wo_qubits bitand page_lower_bits_mask);
          auto const page_index1
            = (StateInteger{1u} << (minmax_permutated_qubits.second - num_nonpage_local_qubits))
              bitor page_index0;

          for (auto nonpage_value_wo_qubits = StateInteger{0u};
               nonpage_value_wo_qubits < ::ket::utility::integer_exp2<StateInteger>(static_cast<StateInteger>(num_nonpage_local_qubits - 1u));
               ++nonpage_value_wo_qubits)
          {
            auto const nonpage_index0
              = ((nonpage_value_wo_qubits bitand nonpage_upper_bits_mask) << 1u)
                bitor (nonpage_value_wo_qubits bitand nonpage_lower_bits_mask);
            auto const nonpage_index1
              = nonpage_index0 bitor (StateInteger{1u} << minmax_permutated_qubits.first);

            for (auto data_block_index = StateInteger{0u};
                 data_block_index < local_state.num_data_blocks(); ++data_block_index)
              local_state.swap_values(
                std::make_tuple(data_block_index, page_index0, nonpage_index1),
                std::make_tuple(data_block_index, page_index1, nonpage_index0));
          }
        }
      }

      template <
        typename ParallelPolicy, typename Complex, bool has_page_qubits, typename Allocator,
        typename StateInteger, typename BitInteger>
      inline auto swap_nonpage_qubits(
        ParallelPolicy const parallel_policy,
        ::ket::mpi::state<Complex, has_page_qubits, Allocator>& local_state,
        ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_qubit1,
        ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_qubit2,
        yampi::communicator const& communicator, yampi::environment const& environment)
      -> void
      {
        assert((not ::ket::mpi::is_page_qubit(permutated_qubit1, local_state)) and (not ::ket::mpi::is_page_qubit(permutated_qubit2, local_state)));
# ifndef NDEBUG
        using permutated_qubit_type = ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> >;
# endif
        assert(permutated_qubit1 < permutated_qubit_type{local_state.num_local_qubits()});
        assert(permutated_qubit2 < permutated_qubit_type{local_state.num_local_qubits()});

        auto const num_pages = local_state.num_pages();
        auto const num_data_blocks = local_state.num_data_blocks();
        for (auto data_block_index = std::size_t{0u}; data_block_index < num_data_blocks; ++data_block_index)
          for (auto page_index = std::size_t{0u}; page_index < num_pages; ++page_index)
          {
            auto const data_block_page_indices = std::make_pair(data_block_index, page_index);
            auto const& data_block_page = local_state.page_range(data_block_page_indices);

            using page_range_type
              = std::remove_reference_t<std::remove_cv_t<decltype(local_state.page_range(data_block_page_indices))>>;
            using swap_permutated_local_qubits_type
              = ::ket::mpi::utility::dispatch::swap_permutated_local_qubits<page_range_type>;
            using std::begin;
            using std::end;
            swap_permutated_local_qubits_type::call(
              parallel_policy, data_block_page,
              permutated_qubit1, permutated_qubit2,
              StateInteger{1u}, static_cast<StateInteger>(std::distance(begin(data_block_page), end(data_block_page))),
              communicator, environment);
          }
      }

      template <bool has_page_qubits>
      struct swap_permutated_local_qubits
      {
        template <
          typename ParallelPolicy, typename Complex, typename Allocator,
          typename StateInteger, typename BitInteger>
        static auto call(
          ParallelPolicy const parallel_policy,
          ::ket::mpi::state<Complex, has_page_qubits, Allocator>& local_state,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_qubit1,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_qubit2,
          StateInteger const, StateInteger const,
          yampi::communicator const& communicator, yampi::environment const& environment)
        -> void
        {
# ifndef NDEBUG
          using permutated_qubit_type = ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> >;
# endif
          assert(permutated_qubit1 < permutated_qubit_type{local_state.num_local_qubits()});
          assert(permutated_qubit2 < permutated_qubit_type{local_state.num_local_qubits()});

          if (::ket::mpi::is_page_qubit(permutated_qubit1, local_state))
          {
            if (local_state.num_page_qubits() >= 2u and ::ket::mpi::is_page_qubit(permutated_qubit2, local_state))
              ::ket::mpi::state_detail::interpage_swap(
                local_state, permutated_qubit1, permutated_qubit2);
            else
              ::ket::mpi::state_detail::swap_page_and_nonpage_qubits(
                parallel_policy, local_state, permutated_qubit1, permutated_qubit2);
          }
          else if (::ket::mpi::is_page_qubit(permutated_qubit2, local_state))
            ::ket::mpi::state_detail::swap_page_and_nonpage_qubits(
              parallel_policy, local_state, permutated_qubit2, permutated_qubit1);
          else
            ::ket::mpi::state_detail::swap_nonpage_qubits(
              parallel_policy, local_state, permutated_qubit2, permutated_qubit1,
              communicator, environment);
        }
      }; // struct swap_permutated_local_qubits<has_page_qubits>

      template <bool has_page_qubits>
      struct interchange_qubits
      {
        template <
          typename Allocator, typename Complex, typename Allocator_, typename StateInteger>
        static auto call(
          ::ket::mpi::state<Complex, has_page_qubits, Allocator>& local_state,
          std::vector<Complex, Allocator_>&,
          StateInteger const data_block_index, StateInteger const data_block_size,
          StateInteger const source_local_first_index,
          StateInteger const source_local_last_index,
          yampi::rank const target_rank,
          yampi::communicator const& communicator, yampi::environment const& environment)
        -> void
        {
          using page_range_type
            = typename ::ket::mpi::state<Complex, has_page_qubits, Allocator>::page_range_type;
          using page_iterator = ::ket::utility::meta::iterator_t<page_range_type>;
          do_call(
            local_state, data_block_index, data_block_size,
            source_local_first_index, source_local_last_index,
            [target_rank, &communicator, &environment](
              page_iterator const first, page_iterator const last,
              page_iterator const buffer_first, page_iterator const buffer_last)
            {
              yampi::algorithm::swap(
                yampi::ignore_status,
                yampi::make_buffer(first, last),
                yampi::make_buffer(buffer_first, buffer_last),
                target_rank, communicator, environment);
            });
        }

        template <
          typename Allocator, typename Complex, typename Allocator_, typename StateInteger,
          typename DerivedDatatype>
        static auto call(
          ::ket::mpi::state<Complex, has_page_qubits, Allocator>& local_state,
          std::vector<Complex, Allocator_>&,
          StateInteger const data_block_index, StateInteger const data_block_size,
          StateInteger const source_local_first_index,
          StateInteger const source_local_last_index,
          yampi::datatype_base<DerivedDatatype> const& datatype, yampi::rank const target_rank,
          yampi::communicator const& communicator, yampi::environment const& environment)
        -> void
        {
          using page_range_type
            = typename ::ket::mpi::state<Complex, has_page_qubits, Allocator>::page_range_type;
          using page_iterator = ::ket::utility::meta::iterator_t<page_range_type>;
          do_call(
            local_state, data_block_index, data_block_size,
            source_local_first_index, source_local_last_index,
            [&datatype, target_rank, &communicator, &environment](
              page_iterator const first, page_iterator const last,
              page_iterator const buffer_first, page_iterator const buffer_last)
            {
              yampi::algorithm::swap(
                yampi::ignore_status,
                yampi::make_buffer(first, last, datatype),
                yampi::make_buffer(buffer_first, buffer_last, datatype),
                target_rank, communicator, environment);
            });
        }

       private:
        template <
          typename Allocator, typename Complex, typename StateInteger, typename Function>
        static auto do_call(
          ::ket::mpi::state<Complex, has_page_qubits, Allocator>& local_state,
          StateInteger const data_block_index, StateInteger const data_block_size,
          StateInteger const source_local_first_index, StateInteger const source_local_last_index,
          Function&& yampi_swap)
        -> void
        {
          assert(data_block_index >= StateInteger{0u} and data_block_index < local_state.num_data_blocks());
          assert(data_block_size == ::ket::utility::integer_exp2<std::size_t>(local_state.num_local_qubits()));

          assert(source_local_last_index >= source_local_first_index);

          auto const front_data_block_page_indices
            = local_state.get_data_block_page_indices(data_block_index * data_block_size + source_local_first_index);
          auto const back_data_block_page_indices
            = local_state.get_data_block_page_indices(data_block_index * data_block_size + source_local_last_index - 1u);
          assert(front_data_block_page_indices.first == data_block_index);
          assert(back_data_block_page_indices.first == data_block_index);

          auto const front_page_index = static_cast<StateInteger>(front_data_block_page_indices.second);
          auto const back_page_index = static_cast<StateInteger>(back_data_block_page_indices.second);

          for (auto page_index = front_page_index; page_index <= back_page_index; ++page_index)
          {
            auto const page_range = local_state.page_range(std::make_pair(data_block_index, page_index));
            using std::begin;
            using std::end;
            auto const page_first = begin(page_range);
            auto const page_last = end(page_range);
            auto const page_size = static_cast<StateInteger>(std::distance(page_first, page_last));
            auto const buffer_first = begin(local_state.buffer_range());

            auto const first_index
              = page_index == front_page_index
                ? static_cast<StateInteger>(local_state.get_nonpage_index(data_block_index * data_block_size + source_local_first_index))
                : StateInteger{0u};
            auto const last_index
              = page_index == back_page_index
                ? static_cast<StateInteger>(local_state.get_nonpage_index(data_block_index * data_block_size + source_local_last_index - 1u) + 1u)
                : static_cast<StateInteger>(page_size);

            auto const the_first = page_first + first_index;
            auto const the_last = page_first + last_index;
            auto const the_buffer_first = buffer_first + first_index;
            auto const the_buffer_last = buffer_first + last_index;

            std::copy(page_first, the_first, buffer_first);
            std::copy(the_last, page_last, the_buffer_last);

            yampi_swap(the_first, the_last, the_buffer_first, the_buffer_last);

            local_state.swap_buffer_and_page(std::make_pair(data_block_index, page_index));
          }
        }
      }; // struct interchange_qubits<has_page_qubits>

      template <bool has_page_qubits>
      struct for_each_local_range
      {
        template <typename MpiPolicy, typename LocalState, typename StateInteger, typename Function>
        static auto call(
          MpiPolicy const& mpi_policy, LocalState&& local_state,
          yampi::communicator const& communicator, yampi::environment const& environment,
          StateInteger const unit_control_qubit_mask, Function&& function)
        -> LocalState&&
        {
          auto const rank_in_unit = ::ket::mpi::utility::policy::rank_in_unit(mpi_policy, communicator, environment);
          auto const num_data_blocks = static_cast<StateInteger>(local_state.num_data_blocks());
          assert(num_data_blocks == static_cast<StateInteger>(::ket::mpi::utility::policy::num_data_blocks(mpi_policy, rank_in_unit)));
          auto const num_pages = static_cast<StateInteger>(local_state.num_pages());

          // Gates should not be on page qubits
          for (auto data_block_index = StateInteger{0u}; data_block_index < num_data_blocks; ++data_block_index)
          {
            if ((static_cast<StateInteger>(::ket::mpi::utility::policy::unit_qubit_value(mpi_policy, data_block_index, rank_in_unit)) bitand unit_control_qubit_mask) != unit_control_qubit_mask)
              continue;

            for (auto page_index = StateInteger{0u}; page_index < num_pages; ++page_index)
            {
              using std::begin;
              using std::end;
              function(
                begin(local_state.page_range(std::make_pair(data_block_index, page_index))),
                end(local_state.page_range(std::make_pair(data_block_index, page_index))));
            }
          }

          return std::forward<LocalState>(local_state);
        }

        template <typename MpiPolicy, typename LocalState, typename Function>
        static auto call(
          MpiPolicy const& mpi_policy, LocalState&& local_state,
          yampi::communicator const& communicator, yampi::environment const& environment,
          Function&& function)
        -> LocalState&&
        {
          auto const rank_in_unit = ::ket::mpi::utility::policy::rank_in_unit(mpi_policy, communicator, environment);
          auto const num_data_blocks = local_state.num_data_blocks();
          assert(num_data_blocks == static_cast<decltype(num_data_blocks)>(::ket::mpi::utility::policy::num_data_blocks(mpi_policy, rank_in_unit)));
          auto const num_pages = local_state.num_pages();

          // Gates should not be on page qubits
          for (auto data_block_index = decltype(num_data_blocks){0u}; data_block_index < num_data_blocks; ++data_block_index)
            for (auto page_index = decltype(num_pages){0u}; page_index < num_pages; ++page_index)
            {
              using std::begin;
              using std::end;
              function(
                begin(local_state.page_range(std::make_pair(data_block_index, page_index))),
                end(local_state.page_range(std::make_pair(data_block_index, page_index))));
            }

          return std::forward<LocalState>(local_state);
        }
      }; // struct for_each_local_range<has_page_qubits>

      template <bool has_page_qubits>
      struct swap_local_data
      {
        template <typename Complex, typename Allocator, typename StateInteger>
        static auto call(
          ::ket::mpi::state<Complex, has_page_qubits, Allocator>& local_state,
          StateInteger const data_block_index1, StateInteger const local_first_index1, StateInteger const local_last_index1,
          StateInteger const data_block_index2, StateInteger const local_first_index2,
          StateInteger const data_block_size)
        -> void 
        {
          auto const first_index1 = data_block_index1 * data_block_size + local_first_index1;
          auto const last_index1 = first_index1 + (local_last_index1 - local_first_index1);
          auto const first_index2 = data_block_index2 * data_block_size + local_first_index2;
          auto const last_index2 = first_index2 + (local_last_index1 - local_first_index1);

          auto const front_data_block_page_indices1 = local_state.get_data_block_page_indices(first_index1);
          auto const back_data_block_page_indices1 = local_state.get_data_block_page_indices(last_index1 - StateInteger{1u});
          auto const front_data_block_page_indices2 = local_state.get_data_block_page_indices(first_index2);
          auto const back_data_block_page_indices2 = local_state.get_data_block_page_indices(last_index2 - StateInteger{1u});
          assert(static_cast<StateInteger>(front_data_block_page_indices1.first) == data_block_index1);
          assert(static_cast<StateInteger>(back_data_block_page_indices1.first) == data_block_index1);
          assert(static_cast<StateInteger>(front_data_block_page_indices2.first) == data_block_index2);
          assert(static_cast<StateInteger>(back_data_block_page_indices2.first) == data_block_index2);

          auto const front_page_index1 = static_cast<StateInteger>(front_data_block_page_indices1.second);
          auto const back_page_index1 = static_cast<StateInteger>(back_data_block_page_indices1.second);
          auto const front_page_index2 = static_cast<StateInteger>(front_data_block_page_indices2.second);
          auto const back_page_index2 = static_cast<StateInteger>(back_data_block_page_indices2.second);

          auto const nonpage_first_index1 = static_cast<StateInteger>(local_state.get_nonpage_index(first_index1));
          auto const nonpage_last_index1 = static_cast<StateInteger>(local_state.get_nonpage_index(last_index1 - StateInteger{1u})) + StateInteger{1u};
          auto const nonpage_first_index2 = static_cast<StateInteger>(local_state.get_nonpage_index(first_index2));
          auto const nonpage_last_index2 = static_cast<StateInteger>(local_state.get_nonpage_index(last_index2 - StateInteger{1u})) + StateInteger{1u};

          using std::begin;
          auto const first1 = begin(local_state.page_range(front_data_block_page_indices1)) + nonpage_first_index1;
          auto const last1 = begin(local_state.page_range(back_data_block_page_indices1)) + nonpage_last_index1;
          auto const last2 = begin(local_state.page_range(back_data_block_page_indices2)) + nonpage_last_index2;

          if (nonpage_first_index1 == nonpage_first_index2)
          {
            assert(nonpage_last_index1 == nonpage_last_index2);

            for (auto page_index1 = front_page_index1, page_index2 = front_page_index2;
                 page_index1 <= back_page_index1; ++page_index1, ++page_index2)
            {
              assert(page_index2 <= back_page_index2);

              auto const data_block_page_indices1 = std::make_pair(data_block_index1, page_index1);
              auto const data_block_page_indices2 = std::make_pair(data_block_index2, page_index2);

              auto const page_range1 = local_state.page_range(data_block_page_indices1);
              auto const page_range2 = local_state.page_range(data_block_page_indices2);

              if (page_index1 == front_page_index1)
                std::swap_ranges(begin(page_range1), first1, begin(page_range2));

              if (page_index1 == back_page_index1)
              {
                assert(page_index2 == back_page_index2);
                using std::end;
                std::swap_ranges(last1, end(page_range1), last2);
              }

              local_state.swap_pages(data_block_page_indices1, data_block_page_indices2);
            }
          }
          else // nonpage_first_index1 != nonpage_first_index2
          {
            auto page_index1 = front_page_index1;
            auto page_index2 = front_page_index2;
            auto the_first1 = first1;
            auto the_first2 = begin(local_state.page_range(front_data_block_page_indices2)) + nonpage_first_index2;

            while (true)
            {
              auto const page_range1 = local_state.page_range(std::make_pair(data_block_index1, page_index1));
              auto const page_range2 = local_state.page_range(std::make_pair(data_block_index2, page_index2));

              using std::end;
              auto const the_last1 = page_index1 == back_page_index1 ? last1 : end(page_range1);
              auto const the_last2 = page_index2 == back_page_index2 ? last2 : end(page_range2);

              auto const size1 = the_last1 - the_first1;
              auto const size2 = the_last2 - the_first2;
              if (size1 == size2)
              {
                assert(the_last1 == last1 and the_last2 == last2);
                std::swap_ranges(the_first1, the_last1, the_first2);
                break;
              }

              else if (size1 > size2)
              {
                assert(the_last2 != last2);
                std::swap_ranges(the_first2, the_last2, the_first1);

                std::advance(the_first1, size2);
                the_first2 = the_last2;

                ++page_index2;
                the_first2 = begin(local_state.page_range(std::make_pair(data_block_index2, page_index2)));
              }
              else // size1 < size2
              {
                assert(the_last1 != last1);
                std::swap_ranges(the_first1, the_last1, the_first2);

                std::advance(the_first2, size1);
                the_first1 = the_last1;

                ++page_index1;
                the_first1 = begin(local_state.page_range(std::make_pair(data_block_index1, page_index1)));
              }
            }
          }
        }
      }; // struct swap_local_data<has_page_qubits>

# ifdef KET_USE_DIAGONAL_LOOP
      template <bool has_page_qubits>
      struct for_each_in_diagonal_loop
      {
        template <
          typename ParallelPolicy,
          typename Complex, typename Allocator,
          typename StateInteger, typename BitInteger,
          std::size_t num_local_control_qubits, typename Function>
        static auto call(
          ParallelPolicy const parallel_policy,
          ::ket::mpi::state<Complex, has_page_qubits, Allocator>& local_state,
          StateInteger const data_block_index, StateInteger const,
          StateInteger const last_local_qubit_value,
          std::array< ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > >, num_local_control_qubits> local_permutated_control_qubits,
          Function&& function)
        -> void
        {
          using std::begin;
          using std::end;
          std::sort(begin(local_permutated_control_qubits), end(local_permutated_control_qubits));

          impl(
            parallel_policy, local_state,
            data_block_index, last_local_qubit_value, local_permutated_control_qubits,
            std::forward<Function>(function));
        }

       private:
        template <
          typename ParallelPolicy,
          typename Complex, typename Allocator,
          typename StateInteger, typename BitInteger,
          std::size_t num_local_control_qubits, typename Function>
        static auto impl(
          ParallelPolicy const parallel_policy,
          ::ket::mpi::state<Complex, has_page_qubits, Allocator>& local_state,
          StateInteger const data_block_index,
          StateInteger const last_local_qubit_value,
          std::array< ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > >, num_local_control_qubits> const& sorted_local_permutated_control_qubits,
          Function&& function)
        -> void
        {
          constexpr auto zero_state_integer = StateInteger{0u};

          using permutated_control_qubit_type
            = ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > >;
          using std::begin;
          using std::end;
          // 000101000100
          auto const mask
            = std::accumulate(
                begin(sorted_local_permutated_control_qubits), end(sorted_local_permutated_control_qubits),
                zero_state_integer,
                [](StateInteger const& partial_mask, permutated_control_qubit_type const& permutated_control_qubit)
                {
                  constexpr auto one_state_integer = StateInteger{1u};
                  return partial_mask bitor (one_state_integer << permutated_control_qubit);
                });

          auto const last_integer
            = (last_local_qubit_value >> local_state.num_page_qubits()) >> num_local_control_qubits;

          auto const num_pages = local_state.num_pages();
          for (auto page_index = std::size_t{0u}; page_index < num_pages; ++page_index)
          {
            auto const first = begin(local_state.page_range(std::make_pair(data_block_index, page_index)));
            ::ket::utility::loop_n(
              parallel_policy, last_integer,
              [&function, &sorted_local_permutated_control_qubits, mask, first](StateInteger state_integer, int const)
              {
                constexpr auto one_state_integer = StateInteger{1u};

                // xxx0x0xxx0xx
                for (permutated_control_qubit_type const& permutated_control_qubit: sorted_local_permutated_control_qubits)
                {
                  auto const lower_mask = (one_state_integer << permutated_control_qubit) - one_state_integer;
                  auto const upper_mask = compl lower_mask;
                  state_integer = (state_integer bitand lower_mask) bitor ((state_integer bitand upper_mask) << 1u);
                }

                // xxx1x1xxx1xx
                state_integer |= mask;

                function(first + state_integer, state_integer);
              });
          }
        }
      }; // struct for_each_in_diagonal_loop<has_page_qubits>
# endif // KET_USE_DIAGONAL_LOOP

      template <bool has_page_qubits>
      struct transform_inclusive_scan
      {
# ifdef KET_USE_PARALLEL_EXECUTE_FOR_TRANSFORM_INCLUSIVE_SCAN
        template <
          typename ParallelPolicy,
          typename Complex, typename Allocator, typename ForwardIterator,
          typename BinaryOperation, typename UnaryOperation>
        static auto call(
          ParallelPolicy const parallel_policy,
          ::ket::mpi::state<Complex, has_page_qubits, Allocator> const& local_state,
          ForwardIterator d_first,
          BinaryOperation binary_operation, UnaryOperation unary_operation,
          yampi::environment const&)
        -> Complex
        {
          auto const num_pages = local_state.num_pages();
          auto const num_data_blocks = local_state.num_data_blocks();
          auto const num_threads = static_cast<unsigned int>(::ket::utility::num_threads(parallel_policy));
          auto partial_sums = std::vector<Complex>(num_threads * num_data_blocks * num_pages);

          ::ket::utility::execute(
            parallel_policy,
            [num_pages, num_data_blocks, num_threads, &partial_sums,
             parallel_policy, &local_state, d_first, binary_operation, unary_operation](
              int const thread_index, auto& executor)
            {
              auto d_page_first = d_first;

              for (auto data_block_index = std::size_t{0u}; data_block_index < num_data_blocks; ++data_block_index)
                for (auto page_index = std::size_t{0u}; page_index < num_pages; ++page_index)
                {
                  auto const& page_range = local_state.page_range(std::make_pair(data_block_index, page_index));
                  using std::begin;
                  using std::end;
                  auto const first = begin(page_range);
                  auto const page_range_size = std::distance(first, end(page_range));
                  auto d_iter = d_page_first;
                  auto is_called = false;
                  auto const page_range_index = local_state.page_range_index(std::make_pair(data_block_index, page_index));

                  using difference_type = typename std::iterator_traits<std::remove_cv_t<decltype(first)>>::difference_type;
                  ::ket::utility::loop_n_in_execute(
                    parallel_policy,
                    page_range_size, thread_index,
                    [page_range_index, first, &d_iter, &is_called, num_threads, &partial_sums,
                     binary_operation, unary_operation](
                      difference_type const n, int const thread_index)
                    {
                      auto const partial_sums_index = num_threads * page_range_index + thread_index;
                      if (not is_called)
                      {
                        std::advance(d_iter, n);
                        partial_sums[partial_sums_index] = unary_operation(first[n]);
                        is_called = true;
                      }
                      else
                        partial_sums[partial_sums_index]
                          = binary_operation(partial_sums[partial_sums_index], unary_operation(first[n]));

                      *d_iter++ = partial_sums[partial_sums_index];
                    });

                  std::advance(d_page_first, page_range_size);
                }

              post_process(
                parallel_policy, local_state, d_first, binary_operation,
                partial_sums, thread_index, executor);
            });

          return partial_sums.back();
        }

        template <
          typename ParallelPolicy,
          typename Complex, typename Allocator, typename ForwardIterator,
          typename BinaryOperation, typename UnaryOperation>
        static auto call(
          ParallelPolicy const parallel_policy,
          ::ket::mpi::state<Complex, has_page_qubits, Allocator> const& local_state,
          ForwardIterator d_first,
          BinaryOperation binary_operation, UnaryOperation unary_operation,
          Complex const initial_value, yampi::environment const&)
        -> Complex
        {
          auto const num_pages = local_state.num_pages();
          auto const num_data_blocks = local_state.num_data_blocks();
          auto const num_threads = static_cast<unsigned int>(::ket::utility::num_threads(parallel_policy));
          auto partial_sums = std::vector<Complex>(num_threads * num_data_blocks * num_pages);

          ::ket::utility::execute(
            parallel_policy,
            [num_pages, num_data_blocks, num_threads, &partial_sums,
             parallel_policy, &local_state, d_first, binary_operation, unary_operation,
             initial_value](
              int const thread_index, auto& executor)
            {
              auto d_page_first = d_first;

              for (auto data_block_index = std::size_t{0u}; data_block_index < num_data_blocks; ++data_block_index)
                for (auto page_index = std::size_t{0u}; page_index < num_pages; ++page_index)
                {
                  auto const& page_range = local_state.page_range(std::make_pair(data_block_index, page_index));
                  using std::begin;
                  using std::end;
                  auto const first = begin(page_range);
                  auto const page_range_size = std::distance(first, end(page_range));
                  auto d_iter = d_page_first;
                  auto is_called = false;
                  auto const page_range_index = local_state.page_range_index(std::make_pair(data_block_index, page_index));

                  using difference_type = typename std::iterator_traits<std::remove_cv_t<decltype(first)>>::difference_type;
                  ::ket::utility::loop_n_in_execute(
                    parallel_policy,
                    page_range_size, thread_index,
                    [page_range_index, first, &d_iter, &is_called, num_threads, &partial_sums,
                     binary_operation, unary_operation, initial_value](
                      difference_type const n, int const thread_index)
                    {
                      auto const partial_sums_index = num_threads * page_range_index + thread_index;
                      if (not is_called)
                      {
                        std::advance(d_iter, n);
                        partial_sums[partial_sums_index]
                          = thread_index == 0
                            ? binary_operation(initial_value, unary_operation(first[n]))
                            : unary_operation(first[n]);
                        is_called = true;
                      }
                      else
                        partial_sums[partial_sums_index]
                          = binary_operation(partial_sums[partial_sums_index], unary_operation(first[n]));

                      *d_iter++ = partial_sums[partial_sums_index];
                    });

                  std::advance(d_page_first, page_range_size);
                }

              post_process(
                parallel_policy, local_state, d_first, binary_operation,
                partial_sums, thread_index, executor);
            });

          return partial_sums.back();
        }

       private:
        template <
          typename ParallelPolicy,
          typename Complex, typename Allocator, typename ForwardIterator,
          typename BinaryOperation, typename Executor>
        static auto post_process(
          ParallelPolicy const parallel_policy,
          ::ket::mpi::state<Complex, has_page_qubits, Allocator>& local_state,
          ForwardIterator d_first, BinaryOperation binary_operation,
          std::vector<Complex>& partial_sums, int const thread_index,
          Executor& executor)
        -> void
        {
          ::ket::utility::barrier(parallel_policy, executor);

          ::ket::utility::single_execute(
            parallel_policy, executor,
            [&partial_sums, binary_operation]
            {
              using std::begin;
              using std::end;
              std::partial_sum(begin(partial_sums), end(partial_sums), begin(partial_sums), binary_operation);
            });

          auto d_page_first = d_first;

          auto const num_threads = static_cast<unsigned int>(::ket::utility::num_threads(parallel_policy));
          auto const num_pages = local_state.num_pages();
          auto const num_data_blocks = local_state.num_data_blocks();

          for (auto data_block_index = std::size_t{0u}; data_block_index < num_data_blocks; ++data_block_index)
            for (auto page_index = std::size_t{0u}; page_index < num_pages; ++page_index)
            {
              auto const& page_range = local_state.page_range(std::make_pair(data_block_index, page_index));
              using std::begin;
              using std::end;
              auto const page_range_size = std::distance(begin(page_range), end(page_range));
              auto d_iter = d_page_first;
              auto is_called = false;
              auto const page_range_index = local_state.page_range_index(std::make_pair(data_block_index, page_index));

              using page_range_type = typename ::ket::mpi::state<Complex, has_page_qubits, Allocator>::page_range_type;
              using page_iterator = ::ket::utility::meta::iterator_t<page_range_type>;
              using difference_type = typename std::iterator_traits<page_iterator>::difference_type;
              ::ket::utility::loop_n_in_execute(
                parallel_policy,
                page_range_size, thread_index,
                [page_range_index, &d_iter, &is_called,
                 num_threads, &partial_sums, binary_operation](
                  difference_type const n, int const thread_index)
                {
                  if (thread_index == 0u and page_range_index == 0u)
                    return;

                  auto const partial_sums_index = num_threads * page_range_index + thread_index;

                  if (not is_called)
                  {
                    std::advance(d_iter, n);
                    is_called = true;
                  }

                  *d_iter = binary_operation(partial_sums[partial_sums_index - 1u], *d_iter);
                  ++d_iter;
                });

              std::advance(d_page_first, page_range_size);
            }
        }
# else // KET_USE_PARALLEL_EXECUTE_FOR_TRANSFORM_INCLUSIVE_SCAN
        template <
          typename ParallelPolicy,
          typename Complex, typename Allocator, typename ForwardIterator,
          typename BinaryOperation, typename UnaryOperation>
        static auto call(
          ParallelPolicy const parallel_policy,
          ::ket::mpi::state<Complex, has_page_qubits, Allocator> const& local_state,
          ForwardIterator const d_first,
          BinaryOperation binary_operation, UnaryOperation unary_operation,
          yampi::environment const& environment)
        -> Complex
        {
          return impl(
            typename std::iterator_traits<ForwardIterator>::iterator_category(),
            parallel_policy, local_state, d_first, binary_operation, unary_operation,
            environment);
        }

        template <
          typename ParallelPolicy,
          typename Complex, typename Allocator, typename ForwardIterator,
          typename BinaryOperation, typename UnaryOperation>
        static auto call(
          ParallelPolicy const parallel_policy,
          ::ket::mpi::state<Complex, has_page_qubits, Allocator> const& local_state,
          ForwardIterator const d_first,
          BinaryOperation binary_operation, UnaryOperation unary_operation,
          Complex const initial_value, yampi::environment const& environment)
        -> Complex
        {
          return impl(
            typename std::iterator_traits<ForwardIterator>::iterator_category(),
            parallel_policy, local_state, d_first, binary_operation, unary_operation,
            initial_value, environment);
        }

       private:
        template <
          typename ParallelPolicy,
          typename Complex, typename Allocator, typename ForwardIterator,
          typename BinaryOperation, typename UnaryOperation>
        static auto impl(
          std::forward_iterator_tag const,
          ParallelPolicy const parallel_policy,
          ::ket::mpi::state<Complex, has_page_qubits, Allocator> const& local_state,
          ForwardIterator d_first,
          BinaryOperation binary_operation, UnaryOperation unary_operation,
          yampi::environment const&)
        -> Complex
        {
          auto prev_d_first = d_first;
          auto const& page_range = local_state.page_range(std::make_pair(0u, 0u));
          d_first
            = ::ket::utility::ranges::transform_inclusive_scan(
                parallel_policy, page_range, d_first, binary_operation, unary_operation);
          using std::begin;
          using std::end;
          std::advance(prev_d_first, std::distance(begin(page_range), end(page_range)) - 1);
          auto partial_sum = *prev_d_first;

          auto const num_pages = local_state.num_pages();
          auto const num_data_blocks = local_state.num_data_blocks();
          for (auto page_range_index = std::size_t{1u}; page_range_index < num_pages * num_data_blocks; ++page_range_index)
          {
            auto const data_block_page_indices = local_state.data_block_page_indices(page_range_index);
            auto const& page_range = local_state.page_range(data_block_page_indices);

            prev_d_first = d_first;
            d_first
              = ::ket::utility::ranges::transform_inclusive_scan(
                  parallel_policy, page_range, d_first, binary_operation, unary_operation, partial_sum);
            std::advance(prev_d_first, std::distance(begin(page_range), end(page_range)) - 1);
            partial_sum = *prev_d_first;
          }

          return partial_sum;
        }

        template <
          typename ParallelPolicy,
          typename Complex, typename Allocator, typename ForwardIterator,
          typename BinaryOperation, typename UnaryOperation>
        static auto impl(
          std::forward_iterator_tag const,
          ParallelPolicy const parallel_policy,
          ::ket::mpi::state<Complex, has_page_qubits, Allocator> const& local_state,
          ForwardIterator d_first,
          BinaryOperation binary_operation, UnaryOperation unary_operation,
          Complex const initial_value, yampi::environment const&)
        -> Complex
        {
          auto partial_sum = initial_value;

          auto const num_pages = local_state.num_pages();
          auto const num_data_blocks = local_state.num_data_blocks();
          for (auto data_block_index = std::size_t{0u}; data_block_index < num_data_blocks; ++data_block_index)
            for (auto page_index = std::size_t{0u}; page_index < num_pages; ++page_index)
            {
              auto const& page_range = local_state.page_range(std::make_pair(data_block_index, page_index));

              auto prev_d_first = d_first;
              d_first
                = ::ket::utility::ranges::transform_inclusive_scan(
                    parallel_policy, page_range, d_first, binary_operation, unary_operation, partial_sum);
              using std::begin;
              using std::end;
              std::advance(prev_d_first, std::distance(begin(page_range), end(page_range)) - 1);
              partial_sum = *prev_d_first;
            }

          return partial_sum;
        }

        template <
          typename ParallelPolicy,
          typename Complex, typename Allocator, typename BidirectionalIterator,
          typename BinaryOperation, typename UnaryOperation>
        static auto impl(
          std::bidirectional_iterator_tag const,
          ParallelPolicy const parallel_policy,
          ::ket::mpi::state<Complex, has_page_qubits, Allocator> const& local_state,
          BidirectionalIterator const d_first,
          BinaryOperation binary_operation, UnaryOperation unary_operation,
          yampi::environment const&)
        -> Complex
        {
          d_first
            = ::ket::utility::ranges::transform_inclusive_scan(
                parallel_policy,
                local_state.page_range(std::make_pair(0u, 0u)), d_first, binary_operation, unary_operation);
          auto partial_sum = *std::prev(d_first);

          auto const num_pages = local_state.num_pages();
          auto const num_data_blocks = local_state.num_data_blocks();
          for (auto page_range_index = std::size_t{1u}; page_range_index < num_pages * num_data_blocks; ++page_range_index)
          {
            auto const data_block_page_indices = local_state.data_block_page_indices(page_range_index);

            d_first
              = ::ket::utility::ranges::transform_inclusive_scan(
                  parallel_policy,
                  local_state.page_range(data_block_page_indices), d_first,
                  binary_operation, unary_operation, partial_sum);
            partial_sum = *std::prev(d_first);
          }

          return partial_sum;
        }

        template <
          typename ParallelPolicy,
          typename Complex, typename Allocator, typename BidirectionalIterator,
          typename BinaryOperation, typename UnaryOperation>
        static auto impl(
          std::bidirectional_iterator_tag const,
          ParallelPolicy const parallel_policy,
          ::ket::mpi::state<Complex, has_page_qubits, Allocator> const& local_state,
          BidirectionalIterator const d_first,
          BinaryOperation binary_operation, UnaryOperation unary_operation,
          Complex const initial_value, yampi::environment const&)
        -> Complex
        {
          auto partial_sum = initial_value;

          auto const num_pages = local_state.num_pages();
          auto const num_data_blocks = local_state.num_data_blocks();
          for (auto data_block_index = std::size_t{0u}; data_block_index < num_data_blocks; ++data_block_index)
            for (auto page_index = std::size_t{0u}; page_index < num_pages; ++page_index)
            {
              d_first
                = ::ket::utility::ranges::transform_inclusive_scan(
                    parallel_policy,
                    local_state.page_range(std::make_pair(data_block_index, page_index)), d_first,
                    binary_operation, unary_operation, partial_sum);
              partial_sum = *std::prev(d_first);
            }

          return partial_sum;
        }
# endif // KET_USE_PARALLEL_EXECUTE_FOR_TRANSFORM_INCLUSIVE_SCAN
      }; // struct transform_inclusive_scan<has_page_qubits>

      template <bool has_page_qubits>
      struct transform_inclusive_scan_self
      {
# ifdef KET_USE_PARALLEL_EXECUTE_FOR_TRANSFORM_INCLUSIVE_SCAN
        template <
          typename ParallelPolicy,
          typename Complex, typename Allocator,
          typename BinaryOperation, typename UnaryOperation>
        static auto call(
          ParallelPolicy const parallel_policy,
          ::ket::mpi::state<Complex, has_page_qubits, Allocator>& local_state,
          BinaryOperation binary_operation, UnaryOperation unary_operation,
          yampi::environment const&)
        -> Complex
        {
          auto const num_threads = static_cast<unsigned int>(::ket::utility::num_threads(parallel_policy));
          auto const num_pages = local_state.num_pages();
          auto const num_data_blocks = local_state.num_data_blocks();
          auto partial_sums = std::vector<Complex>(num_threads * num_data_blocks * num_pages);

          ::ket::utility::execute(
            parallel_policy,
            [num_threads, num_pages, num_data_blocks, &partial_sums,
             parallel_policy, &local_state, binary_operation, unary_operation](
              int const thread_index, auto& executor)
            {
              for (auto data_block_index = std::size_t{0u}; data_block_index < num_data_blocks; ++data_block_index)
                for (auto page_index = std::size_t{0u}; page_index < num_pages; ++page_index)
                {
                  auto const& page_range = local_state.page_range(std::make_pair(data_block_index, page_index));
                  using std::begin;
                  auto const first = begin(page_range);
                  bool is_called = false;
                  auto const page_range_index = local_state.page_range_index(std::make_pair(data_block_index, page_index));

                  using difference_type = typename std::iterator_traits<std::remove_cv_t<decltype(first)>>::difference_type;
                  using std::end;
                  ::ket::utility::loop_n_in_execute(
                    parallel_policy,
                    std::distance(first, end(page_range)), thread_index,
                    [page_range_index, first, &is_called, num_threads, &partial_sums,
                     binary_operation, unary_operation](
                      difference_type const n, int const thread_index)
                    {
                      auto const partial_sums_index = num_threads * page_range_index + thread_index;
                      if (not is_called)
                      {
                        partial_sums[partial_sums_index] = unary_operation(first[n]);
                        is_called = true;
                      }
                      else
                        partial_sums[partial_sums_index]
                          = binary_operation(partial_sums[partial_sums_index], unary_operation(first[n]));

                      first[n] = partial_sums[partial_sums_index];
                    });
                }

              post_process(
                parallel_policy, local_state, binary_operation,
                partial_sums, thread_index, executor);
            });

          return partial_sums.back();
        }

        template <
          typename ParallelPolicy,
          typename Complex, typename Allocator,
          typename BinaryOperation, typename UnaryOperation>
        static auto call(
          ParallelPolicy const parallel_policy,
          ::ket::mpi::state<Complex, has_page_qubits, Allocator>& local_state,
          BinaryOperation binary_operation, UnaryOperation unary_operation,
          Complex const initial_value, yampi::environment const&)
        -> Complex
        {
          auto const num_threads = static_cast<unsigned int>(::ket::utility::num_threads(parallel_policy));
          auto const num_pages = local_state.num_pages();
          auto const num_data_blocks = local_state.num_data_blocks();
          auto partial_sums = std::vector<Complex>(num_threads * num_data_blocks * num_pages);

          ::ket::utility::execute(
            parallel_policy,
            [num_threads, num_pages, num_data_blocks, &partial_sums,
             parallel_policy, &local_state, binary_operation, unary_operation,
             initial_value](
              int const thread_index, auto& executor)
            {
              for (auto data_block_index = std::size_t{0u}; data_block_index < num_data_blocks; ++data_block_index)
                for (auto page_index = std::size_t{0u}; page_index < num_pages; ++page_index)
                {
                  auto const& page_range = local_state.page_range(std::make_pair(data_block_index, page_index));
                  using std::begin;
                  auto const first = begin(page_range);
                  auto is_called = false;
                  auto const page_range_index = local_state.page_range_index(std::make_pair(data_block_index, page_index));

                  using difference_type = typename std::iterator_traits<std::remove_cv_t<decltype(first)>>::difference_type;
                  using std::end;
                  ::ket::utility::loop_n_in_execute(
                    parallel_policy,
                    std::distance(first, end(page_range)), thread_index,
                    [page_index, page_range_index, first, &is_called, num_threads, &partial_sums,
                     binary_operation, unary_operation, initial_value](
                      difference_type const n, int const thread_index)
                    {
                      auto const partial_sums_index = num_threads * page_range_index + thread_index;
                      if (not is_called)
                      {
                        partial_sums[partial_sums_index]
                          = page_index == 0 && thread_index == 0
                            ? binary_operation(initial_value, unary_operation(first[n]))
                            : unary_operation(first[n]);
                        is_called = true;
                      }
                      else
                        partial_sums[partial_sums_index]
                          = binary_operation(partial_sums[partial_sums_index], unary_operation(first[n]));

                      first[n] = partial_sums[partial_sums_index];
                    });
                }

              post_process(
                parallel_policy, local_state, binary_operation,
                partial_sums, thread_index, executor);
            });

          return partial_sums.back();
        }

       private:
        template <
          typename ParallelPolicy,
          typename Complex, typename Allocator, typename BinaryOperation,
          typename Executor>
        static auto post_process(
          ParallelPolicy const parallel_policy,
          ::ket::mpi::state<Complex, has_page_qubits, Allocator>& local_state,
          BinaryOperation binary_operation,
          std::vector<Complex>& partial_sums, int const thread_index,
          Executor& executor)
        -> void
        {
          ::ket::utility::barrier(parallel_policy, executor);

          ::ket::utility::single_execute(
            parallel_policy, executor,
            [&partial_sums, binary_operation]
            {
              using std::begin;
              using std::end;
              std::partial_sum(begin(partial_sums), end(partial_sums), begin(partial_sums), binary_operation);
            });

          auto const num_threads = static_cast<unsigned int>(::ket::utility::num_threads(parallel_policy));
          auto const num_pages = local_state.num_pages();
          auto const num_data_blocks = local_state.num_data_blocks();

          for (auto data_block_index = std::size_t{0u}; data_block_index < num_data_blocks; ++data_block_index)
            for (auto page_index = std::size_t{0u}; page_index < num_pages; ++page_index)
            {
              auto const& page_range = local_state.page_range(std::make_pair(data_block_index, page_index));
              using std::begin;
              auto const first = begin(page_range);
              auto const page_range_index = local_state.page_range_index(std::make_pair(data_block_index, page_index));

              using difference_type = typename std::iterator_traits<std::remove_cv_t<decltype(first)>>::difference_type;
              using std::end;
              ::ket::utility::loop_n_in_execute(
                parallel_policy,
                std::distance(first, end(page_range)), thread_index,
                [page_range_index, first, num_threads, &partial_sums, binary_operation](
                  difference_type const n, int const thread_index)
                {
                  if (thread_index == 0u and page_range_index == 0u)
                    return;

                  auto const partial_sums_index = num_threads * page_range_index + thread_index;
                  first[n] = binary_operation(partial_sums[partial_sums_index - 1u], first[n]);
                });
            }
        }
# else // KET_USE_PARALLEL_EXECUTE_FOR_TRANSFORM_INCLUSIVE_SCAN
        template <
          typename ParallelPolicy,
          typename Complex, typename Allocator,
          typename BinaryOperation, typename UnaryOperation>
        static auto call(
          ParallelPolicy const parallel_policy,
          ::ket::mpi::state<Complex, has_page_qubits, Allocator>& local_state,
          BinaryOperation binary_operation, UnaryOperation unary_operation,
          yampi::environment const&)
        -> Complex
        {
          using std::begin;
          ::ket::utility::ranges::transform_inclusive_scan(
            parallel_policy,
            local_state.page_range(std::make_pair(0u, 0u)),
            begin(local_state.page_range(std::make_pair(0u, 0u))),
            binary_operation, unary_operation);
          using std::end;
          auto partial_sum = *std::prev(end(local_state.page_range(std::make_pair(0u, 0u))));

          auto const num_pages = local_state.num_pages();
          auto const num_data_blocks = local_state.num_data_blocks();
          for (auto page_range_index = std::size_t{1u}; page_range_index < num_pages * num_data_blocks; ++page_range_index)
          {
            auto const data_block_page_indices = local_state.data_block_page_indices(page_range_index);

            ::ket::utility::ranges::transform_inclusive_scan(
              parallel_policy,
              local_state.page_range(data_block_page_indices),
              begin(local_state.page_range(data_block_page_indices)),
              binary_operation, unary_operation, partial_sum);
            partial_sum = *std::prev(end(local_state.page_range(data_block_page_indices)));
          }

          return partial_sum;
        }

        template <
          typename ParallelPolicy,
          typename Complex, typename Allocator,
          typename BinaryOperation, typename UnaryOperation>
        static auto call(
          ParallelPolicy const parallel_policy,
          ::ket::mpi::state<Complex, has_page_qubits, Allocator>& local_state,
          BinaryOperation binary_operation, UnaryOperation unary_operation,
          Complex const initial_value, yampi::environment const&)
        -> Complex
        {
          auto partial_sum = initial_value;

          auto const num_pages = local_state.num_pages();
          auto const num_data_blocks = local_state.num_data_blocks();
          for (auto data_block_index = std::size_t{0u}; data_block_index < num_data_blocks; ++data_block_index)
            for (auto page_index = std::size_t{0u}; page_index < num_pages; ++page_index)
            {
              using std::begin;
              ::ket::utility::ranges::transform_inclusive_scan(
                parallel_policy,
                local_state.page_range(std::make_pair(data_block_index, page_index)),
                begin(local_state.page_range(std::make_pair(data_block_index, page_index))),
                binary_operation, unary_operation, partial_sum);
              using std::end;
              partial_sum = *std::prev(end(local_state.page_range(std::make_pair(data_block_index, page_index))));
            }

          return partial_sum;
        }
# endif // KET_USE_PARALLEL_EXECUTE_FOR_TRANSFORM_INCLUSIVE_SCAN
      }; // struct transform_inclusive_scan_self<has_page_qubits>

      // Usually num_page_qubits is small, so linear search for page is probably not bad.
      template <bool has_page_qubits>
      struct upper_bound
      {
        template <typename Complex, typename Allocator, typename Compare>
        static auto call(
          ::ket::mpi::state<Complex, has_page_qubits, Allocator> const& local_state,
          Complex const& value, Compare compare, yampi::environment const&)
        -> typename ::ket::mpi::state<Complex, has_page_qubits, Allocator>::difference_type
        {
          auto const num_pages = local_state.num_pages();
          auto const num_data_blocks = local_state.num_data_blocks();
          using difference_type = typename ::ket::mpi::state<Complex, has_page_qubits, Allocator>::difference_type;

          for (auto data_block_index = std::size_t{0u}; data_block_index < num_data_blocks; ++data_block_index)
            for (auto page_index = std::size_t{0u}; page_index < num_pages; ++page_index)
            {
              auto const page_range = local_state.page_range(std::make_pair(data_block_index, page_index));
              using std::end;
              if (not compare(value, *std::prev(end(page_range))))
                continue;

              using std::begin;
              auto const index_in_page
                = std::upper_bound(begin(page_range), end(page_range), value, compare) - begin(page_range);

              auto const num_nonpage_local_qubits = local_state.num_local_qubits() - local_state.num_page_qubits();
              return static_cast<difference_type>((data_block_index << local_state.num_local_qubits()) bitor (page_index << num_nonpage_local_qubits) bitor index_in_page);
            }

          return static_cast<difference_type>(local_state.size());
        }
      }; // struct upper_bound<has_page_qubits>
    } // namespace state_detail


    template <typename Complex, typename Allocator>
    class state<Complex, false, Allocator>
    {
     public:
      using value_type = Complex;
      using allocator_type = typename Allocator::template rebind<value_type>::other;

     private:
      using data_type = std::vector<value_type, allocator_type>;
      data_type data_;

      std::size_t num_local_qubits_;
      std::size_t num_data_blocks_;

     public:
      using size_type = typename data_type::size_type;
      using difference_type = typename data_type::difference_type;
      using reference = typename data_type::reference;
      using const_reference = typename data_type::const_reference;
      using pointer = typename data_type::pointer;
      using const_pointer = typename data_type::const_pointer;
      using iterator = typename data_type::iterator;
      using const_iterator = typename data_type::const_iterator;
      using reverse_iterator = typename data_type::reverse_iterator;
      using const_reverse_iterator = typename data_type::const_reverse_iterator;

      state() = delete;
      ~state() noexcept = default;
      state(state const&) = default;
      state& operator=(state const&) = default;
      state(state&&) = default;
      state& operator=(state&&) = default;

      state(state const& other, allocator_type const& allocator)
        : data_{other.data_, allocator},
          num_local_qubits_{other.num_local_qubits_},
          num_data_blocks_{other.num_data_blocks_}
      { }

      state(state&& other, allocator_type const& allocator)
        : data_{std::move(other.data_), allocator},
          num_local_qubits_{std::move(other.num_local_qubits_)},
          num_data_blocks_{std::move(other.num_data_blocks_)}
      { }

      state(std::initializer_list<value_type> initializer_list, allocator_type const& allocator = allocator_type())
        : data_{initializer_list, allocator},
          num_local_qubits_{::ket::utility::integer_log2(initializer_list.size())},
          num_data_blocks_{1u}
      { }

      template <typename StateInteger>
      state(std::initializer_list<value_type> initializer_list, StateInteger const num_data_blocks, allocator_type const& allocator = allocator_type())
        : data_{initializer_list, allocator},
          num_local_qubits_{::ket::utility::integer_log2(initializer_list.size() / num_data_blocks)},
          num_data_blocks_{static_cast<std::size_t>(num_data_blocks)}
      { assert(::ket::utility::integer_exp2<std::size_t>(num_local_qubits_) * num_data_blocks_ == initializer_list.size()); }

      template <typename BitInteger, typename StateInteger, typename PermutationAllocator>
      state(
        BitInteger const num_local_qubits, StateInteger const initial_integer,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, PermutationAllocator> const& permutation,
        yampi::communicator const& communicator, yampi::environment const& environment)
        : data_{generate_initial_data(
            ::ket::mpi::utility::policy::make_simple_mpi(),
            num_local_qubits, initial_integer, permutation, communicator, environment)},
          num_local_qubits_{num_local_qubits},
          num_data_blocks_{1u}
      { }

      template <typename MpiPolicy, typename BitInteger, typename StateInteger, typename PermutationAllocator>
      state(
        MpiPolicy const& mpi_policy, BitInteger const num_local_qubits, StateInteger const initial_integer,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, PermutationAllocator> const& permutation,
        yampi::communicator const& communicator, yampi::environment const& environment)
        : data_{generate_initial_data(
            mpi_policy, num_local_qubits, initial_integer, permutation, communicator, environment)},
          num_local_qubits_{num_local_qubits},
          num_data_blocks_{::ket::mpi::utility::policy::num_data_blocks(mpi_policy, communicator, environment)}
      { }

      auto assign(std::initializer_list<value_type> initializer_list) -> void
      { assign(initializer_list, 1u); }

      template <typename StateInteger>
      auto assign(std::initializer_list<value_type> initializer_list, StateInteger const num_data_blocks) -> void
      {
        data_.assign(initializer_list);
        num_local_qubits_ = ::ket::utility::integer_log2(initializer_list.size() / num_data_blocks);
        num_data_blocks_ = static_cast<std::size_t>(num_data_blocks);

        assert(::ket::utility::integer_exp2<std::size_t>(num_local_qubits_) * num_data_blocks_ == initializer_list.size());
      }

      template <typename BitInteger, typename StateInteger, typename PermutationAllocator>
      auto assign(
        BitInteger const num_local_qubits, StateInteger const initial_integer,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, PermutationAllocator> const& permutation,
        yampi::communicator const& communicator, yampi::environment const& environment)
      -> void
      { assign(::ket::mpi::utility::policy::make_simple_mpi(), num_local_qubits, initial_integer, permutation, communicator, environment); }

      template <typename MpiPolicy, typename BitInteger, typename StateInteger, typename PermutationAllocator>
      auto assign(
        MpiPolicy const& mpi_policy, BitInteger const num_local_qubits, StateInteger const initial_integer,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, PermutationAllocator> const& permutation,
        yampi::communicator const& communicator, yampi::environment const& environment)
      -> void
      {
        initialize_data(data_, mpi_policy, num_local_qubits, initial_integer, permutation, communicator, environment);
        num_local_qubits_ = num_local_qubits;
        num_data_blocks_ = ::ket::mpi::utility::policy::num_data_blocks(mpi_policy, communicator, environment);
      }

      auto num_local_qubits() const -> std::size_t { return num_local_qubits_; }
      auto num_data_blocks() const -> std::size_t { return num_data_blocks_; }

      auto operator==(state const& other) const -> bool { return data_ == other.data_; }
      auto operator<(state const& other) const -> bool { return data_ < other.data_; }

      // Element access
      auto at(size_type const position) -> reference { return data_.at(position); }
      auto at(size_type const position) const -> const_reference { return data_.at(position); }
      auto operator[](size_type const position) -> reference { return data_[position]; }
      auto operator[](size_type const position) const -> const_reference { return data_[position]; }
      auto front() -> reference { return data_.front(); }
      auto front() const -> const_reference { return data_.front(); }
      auto back() -> reference { return data_.back(); }
      auto back() const -> const_reference { return data_.back(); }

      // Iterators
      auto begin() noexcept -> iterator { return data_.begin(); }
      auto begin() const noexcept -> const_iterator { return data_.begin(); }
      auto cbegin() const noexcept -> const_iterator { return data_.cbegin(); }
      auto end() noexcept -> iterator { return data_.end(); }
      auto end() const noexcept -> const_iterator { return data_.end(); }
      auto cend() const noexcept -> const_iterator { return data_.cend(); }
      auto rbegin() noexcept -> reverse_iterator { return data_.rbegin(); }
      auto rbegin() const noexcept -> const_reverse_iterator { return data_.rbegin(); }
      auto crbegin() const noexcept -> const_reverse_iterator { return data_.crbegin(); }
      auto rend() noexcept -> reverse_iterator { return data_.rend(); }
      auto rend() const noexcept -> const_reverse_iterator { return data_.rend(); }
      auto crend() const noexcept -> const_reverse_iterator { return data_.crend(); }

      // Capacity
      auto size() const noexcept -> size_type { return data_.size(); }
      auto max_size() const noexcept -> size_type { return data_.max_size(); } 
      auto reserve(size_type const new_capacity) -> void { data_.reserve(new_capacity); }
      auto capacity() const noexcept -> size_type { return data_.capacity(); }
      auto shrink_to_fit() -> void { data_.shrink_to_fit(); }

      // Modifiers
      auto swap(state& other) noexcept(KET_is_nothrow_swappable<data_type>::value) -> void
      {
        using std::swap;
        swap(data_, other.data_);
        swap(num_local_qubits_, other.num_local_qubits_);
        swap(num_data_blocks_, other.num_data_blocks_);
      }

      auto data() -> data_type& { return data_; }
      auto data() const -> data_type const& { return data_; }

     private:
      template <typename MpiPolicy, typename BitInteger, typename StateInteger, typename PermutationAllocator>
      auto initialize_data(
        data_type& data,
        MpiPolicy const& mpi_policy, BitInteger const num_local_qubits, StateInteger const initial_integer,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, PermutationAllocator> const& permutation,
        yampi::communicator const& communicator, yampi::environment const& environment) const
      -> void
      {
        data.assign(
          ::ket::utility::integer_exp2<std::size_t>(num_local_qubits)
            * ::ket::mpi::utility::policy::num_data_blocks(mpi_policy, communicator, environment),
          value_type{0});

        auto const rank_index
          = ::ket::mpi::utility::qubit_value_to_rank_index(
              mpi_policy, data, ::ket::mpi::permutate_bits(permutation, initial_integer), communicator, environment);

        if (communicator.rank(environment) == rank_index.first)
          data[rank_index.second] = value_type{1};
      }

      template <typename MpiPolicy, typename BitInteger, typename StateInteger, typename PermutationAllocator>
      auto generate_initial_data(
        MpiPolicy const& mpi_policy, BitInteger const num_local_qubits, StateInteger const initial_integer,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, PermutationAllocator> const& permutation,
        yampi::communicator const& communicator, yampi::environment const& environment) const
      -> data_type
      {
        auto result = data_type{};
        initialize_data(result, mpi_policy, num_local_qubits, initial_integer, permutation, communicator, environment);
        return result;
      }
    }; // class state<Complex, false, Allocator>

    template <typename Complex, bool has_page_qubits, typename Allocator>
    inline auto operator!=(
      ::ket::mpi::state<Complex, has_page_qubits, Allocator> const& lhs,
      ::ket::mpi::state<Complex, has_page_qubits, Allocator> const& rhs)
    -> bool
    { return not(lhs == rhs); }

    template <typename Complex, bool has_page_qubits, typename Allocator>
    inline auto operator>(
      ::ket::mpi::state<Complex, has_page_qubits, Allocator> const& lhs,
      ::ket::mpi::state<Complex, has_page_qubits, Allocator> const& rhs)
    -> bool
    { return rhs < lhs; }

    template <typename Complex, bool has_page_qubits, typename Allocator>
    inline auto operator<=(
      ::ket::mpi::state<Complex, has_page_qubits, Allocator> const& lhs,
      ::ket::mpi::state<Complex, has_page_qubits, Allocator> const& rhs)
    -> bool
    { return not(lhs > rhs); }

    template <typename Complex, bool has_page_qubits, typename Allocator>
    inline auto operator>=(
      ::ket::mpi::state<Complex, has_page_qubits, Allocator> const& lhs,
      ::ket::mpi::state<Complex, has_page_qubits, Allocator> const& rhs)
    -> bool
    { return not(lhs < rhs); }

    template <typename Complex,  bool has_page_qubits, typename Allocator>
    inline void swap(
      ::ket::mpi::state<Complex, has_page_qubits, Allocator>& lhs,
      ::ket::mpi::state<Complex, has_page_qubits, Allocator>& rhs)
      noexcept(
        KET_is_nothrow_swappable<
          ::ket::mpi::state<Complex, has_page_qubits, Allocator>>::value)
    { lhs.swap(rhs); }

    namespace state_detail
    {
      template <>
      struct swap_permutated_local_qubits<false>
      {
        template <
          typename ParallelPolicy, typename Complex, typename Allocator,
          typename StateInteger, typename BitInteger>
        static auto call(
          ParallelPolicy const parallel_policy,
          ::ket::mpi::state<Complex, false, Allocator>& local_state,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_qubit1,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_qubit2,
          StateInteger const num_data_blocks, StateInteger const data_block_size,
          yampi::communicator const& communicator, yampi::environment const& environment)
        -> void
        {
          ::ket::mpi::utility::detail::swap_permutated_local_qubits(
            parallel_policy, local_state.data(), permutated_qubit1, permutated_qubit2,
            num_data_blocks, data_block_size, communicator, environment);
        }
      }; // struct swap_permutated_local_qubits<false>

      template <>
      struct interchange_qubits<false>
      {
        template <
          typename Allocator, typename Complex, typename Allocator_, typename StateInteger>
        static auto call(
          ::ket::mpi::state<Complex, false, Allocator>& local_state,
          std::vector<Complex, Allocator_>& buffer,
          StateInteger const data_block_index, StateInteger const data_block_size,
          StateInteger const source_local_first_index,
          StateInteger const source_local_last_index,
          yampi::rank const target_rank,
          yampi::communicator const& communicator, yampi::environment const& environment)
        -> void
        {
          assert(data_block_index < local_state.num_data_blocks());
          assert(data_block_size == ::ket::utility::integer_exp2<std::size_t>(local_state.num_local_qubits()));

          ::ket::mpi::utility::detail::interchange_qubits(
            local_state.data(), buffer, data_block_index, data_block_size,
            source_local_first_index, source_local_last_index,
            target_rank, communicator, environment);
        }

        template <
          typename Allocator, typename Complex, typename Allocator_, typename StateInteger,
          typename DerivedDatatype>
        static auto call(
          ::ket::mpi::state<Complex, false, Allocator>& local_state,
          std::vector<Complex, Allocator_>& buffer,
          StateInteger const data_block_index, StateInteger const data_block_size,
          StateInteger const source_local_first_index,
          StateInteger const source_local_last_index,
          yampi::datatype_base<DerivedDatatype> const& datatype, yampi::rank const target_rank,
          yampi::communicator const& communicator, yampi::environment const& environment)
        -> void
        {
          assert(data_block_index < local_state.num_data_blocks());
          assert(data_block_size == ::ket::utility::integer_exp2<std::size_t>(local_state.num_local_qubits()));

          ::ket::mpi::utility::detail::interchange_qubits(
            local_state.data(), buffer, data_block_index, data_block_size,
            source_local_first_index, source_local_last_index,
            datatype, target_rank, communicator, environment);
        }
      }; // struct interchange_qubits<false>

      template <>
      struct for_each_local_range<false>
      {
        template <typename MpiPolicy, typename LocalState, typename StateInteger, typename Function>
        static auto call(
          MpiPolicy const& mpi_policy, LocalState&& local_state,
          yampi::communicator const& communicator, yampi::environment const& environment,
          StateInteger const unit_control_qubit_mask, Function&& function)
        -> LocalState&&
        {
          ::ket::mpi::utility::for_each_local_range(
            mpi_policy, local_state.data(),
            communicator, environment, unit_control_qubit_mask, std::forward<Function>(function));

          return std::forward<LocalState>(local_state);
        }

        template <typename MpiPolicy, typename LocalState, typename Function>
        static auto call(
          MpiPolicy const& mpi_policy, LocalState&& local_state,
          yampi::communicator const& communicator, yampi::environment const& environment,
          Function&& function)
        -> LocalState&&
        {
          ::ket::mpi::utility::for_each_local_range(
            mpi_policy, local_state.data(),
            communicator, environment, std::forward<Function>(function));

          return std::forward<LocalState>(local_state);
        }
      }; // struct for_each_local_range<false>

      template <>
      struct swap_local_data<false>
      {
        template <typename Complex, typename Allocator, typename StateInteger>
        static auto call(
          ::ket::mpi::state<Complex, false, Allocator>& local_state,
          StateInteger const data_block_index1, StateInteger const local_first_index1, StateInteger const local_last_index1,
          StateInteger const data_block_index2, StateInteger const local_first_index2,
          StateInteger const data_block_size)
        -> void
        {
          ::ket::mpi::utility::detail::swap_local_data(
            local_state.data(),
            data_block_index1, local_first_index1, local_last_index1,
            data_block_index2, local_first_index2, data_block_size);
        }
      }; // struct swap_local_data<false>

# ifdef KET_USE_DIAGONAL_LOOP
      template <>
      struct for_each_in_diagonal_loop<false>
      {
        template <
          typename ParallelPolicy,
          typename Complex, typename Allocator,
          typename StateInteger, typename BitInteger,
          std::size_t num_local_control_qubits, typename Function>
        static auto call(
          ParallelPolicy const parallel_policy,
          ::ket::mpi::state<Complex, false, Allocator>& local_state,
          StateInteger const data_block_index, StateInteger const data_block_size,
          StateInteger const last_local_qubit_value,
          std::array< ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > >, num_local_control_qubits > const& local_permutated_control_qubits,
          Function&& function)
        -> void
        {
          using data_type = std::remove_cv_t<std::remove_reference_t<decltype(local_state.data())>>;
          using for_each_in_diagonal_loop_type = ::ket::mpi::utility::dispatch::for_each_in_diagonal_loop<data_type>;
          for_each_in_diagonal_loop_type::call(
            parallel_policy, local_state.data(),
            data_block_index, data_block_size, last_local_qubit_value,
            local_permutated_control_qubits, std::forward<Function>(function));
        }
      }; // struct for_each_in_diagonal_loop<false>
# endif // KET_USE_DIAGONAL_LOOP

      template <>
      struct transform_inclusive_scan<false>
      {
        template <
          typename ParallelPolicy,
          typename Complex, typename Allocator, typename ForwardIterator,
          typename BinaryOperation, typename UnaryOperation>
        static auto call(
          ParallelPolicy const parallel_policy,
          ::ket::mpi::state<Complex, false, Allocator> const& local_state,
          ForwardIterator const d_first,
          BinaryOperation binary_operation, UnaryOperation unary_operation,
          yampi::environment const& environment)
        -> Complex
        {
          return ::ket::mpi::utility::transform_inclusive_scan(
            parallel_policy,
            local_state.data(), d_first, binary_operation, unary_operation,
            environment);
        }

        template <
          typename ParallelPolicy,
          typename Complex, typename Allocator, typename ForwardIterator,
          typename BinaryOperation, typename UnaryOperation>
        static auto call(
          ParallelPolicy const parallel_policy,
          ::ket::mpi::state<Complex, false, Allocator> const& local_state,
          ForwardIterator const d_first,
          BinaryOperation binary_operation, UnaryOperation unary_operation,
          Complex const initial_value, yampi::environment const& environment)
        -> Complex
        {
          return ::ket::mpi::utility::transform_inclusive_scan(
            parallel_policy,
            local_state.data(), d_first, binary_operation, unary_operation,
            initial_value, environment);
        }
      }; // struct transform_inclusive_scan<false>

      template <>
      struct transform_inclusive_scan_self<false>
      {
        template <
          typename ParallelPolicy,
          typename Complex, typename Allocator,
          typename BinaryOperation, typename UnaryOperation>
        static auto call(
          ParallelPolicy const parallel_policy,
          ::ket::mpi::state<Complex, false, Allocator>& local_state,
          BinaryOperation binary_operation, UnaryOperation unary_operation,
          yampi::environment const& environment)
        -> Complex
        {
          return ::ket::mpi::utility::transform_inclusive_scan_self(
            parallel_policy,
            local_state.data(), binary_operation, unary_operation,
            environment);
        }

        template <
          typename ParallelPolicy,
          typename Complex, typename Allocator,
          typename BinaryOperation, typename UnaryOperation>
        static auto call(
          ParallelPolicy const parallel_policy,
          ::ket::mpi::state<Complex, false, Allocator>& local_state,
          BinaryOperation binary_operation, UnaryOperation unary_operation,
          Complex const initial_value, yampi::environment const& environment)
        -> Complex
        {
          return ::ket::mpi::utility::transform_inclusive_scan_self(
            parallel_policy,
            local_state.data(), binary_operation, unary_operation,
            initial_value, environment);
        }
      }; // struct transform_inclusive_scan_self<false>

      template <>
      struct upper_bound<false>
      {
        template <typename Complex, typename Allocator, typename Compare>
        static auto call(
          ::ket::mpi::state<Complex, false, Allocator> const& local_state,
          Complex const& value, Compare compare, yampi::environment const& environment)
        -> typename ::ket::mpi::state<Complex, false, Allocator>::difference_type
        { return ::ket::mpi::utility::upper_bound(local_state.data(), value, compare, environment); }
      }; // struct upper_bound<false>
    } // namespace state_detail

    namespace page
    {
      namespace dispatch
      {
        template <typename LocalState_>
        struct is_on_page;

        template <typename Complex, typename Allocator>
        struct is_on_page< ::ket::mpi::state<Complex, false, Allocator> >
        {
          template <typename Qubit>
          static constexpr auto call(::ket::mpi::permutated<Qubit> const, ::ket::mpi::state<Complex, false, Allocator> const&) -> bool
          { return false; }
        }; // struct is_on_page< ::ket::mpi::state<Complex, false, Allocator> >

        template <typename Complex, typename Allocator>
        struct is_on_page< ::ket::mpi::state<Complex, true, Allocator> >
        {
          template <typename Qubit>
          static constexpr auto call(::ket::mpi::permutated<Qubit> const permutated_qubit, ::ket::mpi::state<Complex, true, Allocator> const& local_state) -> bool
          { return ::ket::mpi::is_page_qubit(permutated_qubit, local_state); }
        }; // struct is_on_page< ::ket::mpi::state<Complex, true, Allocator> >

        template <typename LocalState_>
        struct page_size;

        template <typename Complex, typename Allocator>
        struct page_size< ::ket::mpi::state<Complex, false, Allocator> >
        {
          template <typename MpiPolicy>
          static auto call(MpiPolicy const& mpi_policy, ::ket::mpi::state<Complex, false, Allocator> const& local_state, yampi::communicator const& communicator, yampi::environment const& environment)
          { return ::ket::mpi::utility::policy::data_block_size(mpi_policy, local_state, communicator, environment); }
        }; // struct page_size< ::ket::mpi::state<Complex, false, Allocator> >

        template <typename Complex, typename Allocator>
        struct page_size< ::ket::mpi::state<Complex, true, Allocator> >
        {
          template <typename MpiPolicy>
          static auto call(MpiPolicy const& mpi_policy, ::ket::mpi::state<Complex, true, Allocator> const& local_state, yampi::communicator const& communicator, yampi::environment const& environment)
          { return ::ket::mpi::utility::policy::data_block_size(mpi_policy, local_state, communicator, environment) / local_state.num_pages(); }
        }; // struct page_size< ::ket::mpi::state<Complex, true, Allocator> >
      } // namespace dispatch
    } // namespace page

    namespace utility
    {
      namespace dispatch
      {
        template <typename LocalState_>
        struct swap_permutated_local_qubits;

        template <typename Complex, bool has_page_qubits, typename Allocator>
        struct swap_permutated_local_qubits< ::ket::mpi::state<Complex, has_page_qubits, Allocator> >
        {
          template <typename ParallelPolicy, typename StateInteger, typename BitInteger>
          static auto call(
            ParallelPolicy const parallel_policy,
            ::ket::mpi::state<Complex, has_page_qubits, Allocator>& local_state,
            ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_qubit1,
            ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_qubit2,
            StateInteger const num_data_blocks, StateInteger const data_block_size,
            yampi::communicator const& communicator, yampi::environment const& environment)
          -> void
          {
            ::ket::mpi::state_detail::swap_permutated_local_qubits<has_page_qubits>::call(
              parallel_policy, local_state, permutated_qubit1, permutated_qubit2,
              num_data_blocks, data_block_size, communicator, environment);
          }
        }; // struct swap_permutated_local_qubits< ::ket::mpi::state<Complex, has_page_qubits, Allocator> >

        template <typename LocalState_>
        struct interchange_qubits;

        template <typename Complex, bool has_page_qubits, typename Allocator>
        struct interchange_qubits< ::ket::mpi::state<Complex, has_page_qubits, Allocator> >
        {
          template <typename Allocator_, typename StateInteger>
          static auto call(
            ::ket::mpi::state<Complex, has_page_qubits, Allocator>& local_state,
            std::vector<Complex, Allocator_>& buffer,
            StateInteger const data_block_index, StateInteger const data_block_size,
            StateInteger const source_local_first_index,
            StateInteger const source_local_last_index,
            yampi::rank const target_rank,
            yampi::communicator const& communicator, yampi::environment const& environment)
          -> void
          {
            ::ket::mpi::state_detail::interchange_qubits<has_page_qubits>::call(
              local_state, buffer, data_block_index, data_block_size,
              source_local_first_index, source_local_last_index,
              target_rank, communicator, environment);
          }

          template <typename Allocator_, typename StateInteger, typename DerivedDatatype>
          static auto call(
            ::ket::mpi::state<Complex, has_page_qubits, Allocator>& local_state,
            std::vector<Complex, Allocator_>& buffer,
            StateInteger const data_block_index, StateInteger const data_block_size,
            StateInteger const source_local_first_index,
            StateInteger const source_local_last_index,
            yampi::datatype_base<DerivedDatatype> const& datatype, yampi::rank const target_rank,
            yampi::communicator const& communicator, yampi::environment const& environment)
          -> void
          {
            ::ket::mpi::state_detail::interchange_qubits<has_page_qubits>::call(
              local_state, buffer, data_block_index, data_block_size,
              source_local_first_index, source_local_last_index,
              datatype, target_rank, communicator, environment);
          }
        }; // struct interchange_qubits< ::ket::mpi::state<Complex, has_page_qubits, Allocator> >

        template <typename LocalState_>
        struct for_each_local_range;

        template <typename Complex, bool has_page_qubits, typename Allocator>
        struct for_each_local_range< ::ket::mpi::state<Complex, has_page_qubits, Allocator> >
        {
          template <typename MpiPolicy, typename LocalState, typename StateInteger, typename Function>
          static auto call(
            MpiPolicy const& mpi_policy, LocalState&& local_state,
            yampi::communicator const& communicator, yampi::environment const& environment,
            StateInteger const unit_control_qubit_mask, Function&& function)
          -> LocalState&&
          {
            using for_each_local_range_type = ::ket::mpi::state_detail::for_each_local_range<has_page_qubits>;
            return for_each_local_range_type::call(
              mpi_policy, std::forward<LocalState>(local_state), communicator, environment, unit_control_qubit_mask, std::forward<Function>(function));
          }

          template <typename MpiPolicy, typename LocalState, typename Function>
          static auto call(
            MpiPolicy const& mpi_policy, LocalState&& local_state,
            yampi::communicator const& communicator, yampi::environment const& environment,
            Function&& function)
          -> LocalState&&
          {
            using for_each_local_range_type = ::ket::mpi::state_detail::for_each_local_range<has_page_qubits>;
            return for_each_local_range_type::call(
              mpi_policy, std::forward<LocalState>(local_state), communicator, environment, std::forward<Function>(function));
          }
        }; // struct for_each_local_range< ::ket::mpi::state<Complex, has_page_qubits, Allocator> >

        template <typename LocalState_>
        struct swap_local_data;

        template <typename Complex, bool has_page_qubits, typename Allocator>
        struct swap_local_data< ::ket::mpi::state<Complex, has_page_qubits, Allocator> >
        {
          template <typename StateInteger>
          static auto call(
            ::ket::mpi::state<Complex, has_page_qubits, Allocator>& local_state,
            StateInteger const data_block_index1, StateInteger const local_first_index1, StateInteger const local_last_index1,
            StateInteger const data_block_index2, StateInteger const local_first_index2,
            StateInteger const data_block_size)
          -> void
          {
            using swap_local_data_type = ::ket::mpi::state_detail::swap_local_data<has_page_qubits>;
            swap_local_data_type::call(
              local_state,
              data_block_index1, local_first_index1, local_last_index1,
              data_block_index2, local_first_index2, data_block_size);
          }
        }; // struct swap_local_data< ::ket::mpi::state<Complex, has_page_qubits, Allocator> >

        template <typename Complex, typename Allocator>
        struct buffer_range< ::ket::mpi::state<Complex, true, Allocator> >
        {
          template <typename BufferAllocator>
          static auto call(
            ::ket::mpi::state<Complex, true, Allocator>& local_state,
            std::vector<Complex, BufferAllocator>&)
            -> decltype(local_state.buffer_range())
          { return local_state.buffer_range(); }

          template <typename BufferAllocator>
          static auto call(
            ::ket::mpi::state<Complex, true, Allocator> const& local_state,
            std::vector<Complex, BufferAllocator> const&)
            -> decltype(local_state.buffer_range())
          { return local_state.buffer_range(); }

          template <typename BufferAllocator>
          static auto call_begin(
            ::ket::mpi::state<Complex, true, Allocator>& local_state,
            std::vector<Complex, BufferAllocator>&)
            -> ::ket::utility::meta::iterator_t<decltype(local_state.buffer_range())>
          { using std::begin; return begin(local_state.buffer_range()); }

          template <typename BufferAllocator>
          static auto call_begin(
            ::ket::mpi::state<Complex, true, Allocator> const& local_state,
            std::vector<Complex, BufferAllocator> const&)
            -> ::ket::utility::meta::iterator_t<decltype(local_state.buffer_range())>
          { using std::begin; return begin(local_state.buffer_range()); }

          template <typename BufferAllocator>
          static auto call_end(
            ::ket::mpi::state<Complex, true, Allocator>& local_state,
            std::vector<Complex, BufferAllocator>&)
            -> ::ket::utility::meta::iterator_t<decltype(local_state.buffer_range())>
          { using std::end; return end(local_state.buffer_range()); }

          template <typename BufferAllocator>
          static auto call_end(
            ::ket::mpi::state<Complex, true, Allocator> const& local_state,
            std::vector<Complex, BufferAllocator> const&)
            -> ::ket::utility::meta::iterator_t<decltype(local_state.buffer_range())>
          { using std::end; return end(local_state.buffer_range()); }
        }; // struct buffer_range< ::ket::mpi::state<Complex, true, Allocator> >

        template <typename Complex, typename Allocator>
        struct buffer_range< ::ket::mpi::state<Complex, false, Allocator> >
        {
          template <typename BufferAllocator>
          static auto call(
            ::ket::mpi::state<Complex, false, Allocator>&,
            std::vector<Complex, BufferAllocator>& buffer)
          -> boost::iterator_range<typename std::vector<Complex, BufferAllocator>::iterator>
          { using std::begin; using std::end; return boost::make_iterator_range(begin(buffer), end(buffer)); }

          template <typename BufferAllocator>
          static auto call(
            ::ket::mpi::state<Complex, false, Allocator> const&,
            std::vector<Complex, BufferAllocator> const& buffer)
          -> boost::iterator_range<typename std::vector<Complex, BufferAllocator>::const_iterator>
          { using std::begin; using std::end; return boost::make_iterator_range(begin(buffer), end(buffer)); }

          template <typename BufferAllocator>
          static auto call_begin(
            ::ket::mpi::state<Complex, false, Allocator>&,
            std::vector<Complex, BufferAllocator>& buffer)
          -> typename std::vector<Complex, BufferAllocator>::iterator
          { using std::begin; return begin(buffer); }

          template <typename BufferAllocator>
          static auto call_begin(
            ::ket::mpi::state<Complex, false, Allocator> const&,
            std::vector<Complex, BufferAllocator> const& buffer)
          -> typename std::vector<Complex, BufferAllocator>::const_iterator
          { using std::begin; return begin(buffer); }

          template <typename BufferAllocator>
          static auto call_end(
            ::ket::mpi::state<Complex, false, Allocator>&,
            std::vector<Complex, BufferAllocator>& buffer)
          -> typename std::vector<Complex, BufferAllocator>::iterator
          { using std::end; return end(buffer); }

          template <typename BufferAllocator>
          static auto call_end(
            ::ket::mpi::state<Complex, false, Allocator> const&,
            std::vector<Complex, BufferAllocator> const& buffer)
          -> typename std::vector<Complex, BufferAllocator>::const_iterator
          { using std::end; return end(buffer); }
        }; // struct buffer_range< ::ket::mpi::state<Complex, false, Allocator> >

        template <typename Complex, typename Allocator>
        struct resize_buffer_if_empty< ::ket::mpi::state<Complex, true, Allocator> >
        {
          template <typename BufferAllocator>
          static auto call(
            ::ket::mpi::state<Complex, true, Allocator> const&,
            std::vector<Complex, BufferAllocator>&,
            std::size_t const)
          -> void
          { }
        }; // struct resize_buffer_if_empty< ::ket::mpi::state<Complex, true, Allocator> >

        template <typename Complex, typename Allocator>
        struct resize_buffer_if_empty< ::ket::mpi::state<Complex, false, Allocator> >
        {
          template <typename BufferAllocator>
          static auto call(
            ::ket::mpi::state<Complex, false, Allocator> const&,
            std::vector<Complex, BufferAllocator>& buffer,
            std::size_t const new_size)
          -> void
          {
            if (not buffer.empty())
              return;

            buffer.resize(new_size);
          }
        }; // struct resize_buffer_if_empty< ::ket::mpi::state<Complex, false, Allocator> >

# ifdef KET_USE_DIAGONAL_LOOP
        template <typename LocalState_>
        struct for_each_in_diagonal_loop;

        template <typename Complex, bool has_page_qubits, typename Allocator>
        struct for_each_in_diagonal_loop< ::ket::mpi::state<Complex, has_page_qubits, Allocator> >
        {
          template <
            typename ParallelPolicy,
            typename StateInteger, typename BitInteger,
            std::size_t num_local_control_qubits, typename Function>
          static auto call(
            ParallelPolicy const parallel_policy,
            ::ket::mpi::state<Complex, has_page_qubits, Allocator>& local_state,
            StateInteger const data_block_index, StateInteger const data_block_size,
            StateInteger const last_local_qubit_value,
            std::array< ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > >, num_local_control_qubits> const& local_permutated_control_qubits,
            Function&& function)
          -> void
          {
            ::ket::mpi::state_detail::for_each_in_diagonal_loop<has_page_qubits>::call(
              parallel_policy,
              local_state, data_block_index, data_block_size, last_local_qubit_value,
              local_permutated_control_qubits, std::forward<Function>(function));
          }
        }; // struct for_each_in_diagonal_loop< ::ket::mpi::state<Complex, has_page_qubits, Allocator> >
# endif // KET_USE_DIAGONAL_LOOP

        template <typename LocalState_>
        struct transform_inclusive_scan;

        template <typename Complex, bool has_page_qubits, typename Allocator>
        struct transform_inclusive_scan<
          ::ket::mpi::state<Complex, has_page_qubits, Allocator> >
        {
          template <
            typename ParallelPolicy, typename ForwardIterator,
            typename BinaryOperation, typename UnaryOperation>
          static auto call(
            ParallelPolicy const parallel_policy,
            ::ket::mpi::state<Complex, has_page_qubits, Allocator> const& local_state,
            ForwardIterator const d_first,
            BinaryOperation binary_operation, UnaryOperation unary_operation,
            yampi::environment const& environment)
          -> Complex
          {
            using transform_inclusive_scan_type = ::ket::mpi::state_detail::transform_inclusive_scan<has_page_qubits>;
            return transform_inclusive_scan_type::call(
              parallel_policy,
              local_state, d_first, binary_operation, unary_operation, environment);
          }

          template <
            typename ParallelPolicy, typename ForwardIterator,
            typename BinaryOperation, typename UnaryOperation>
          static auto call(
            ParallelPolicy const parallel_policy,
            ::ket::mpi::state<Complex, has_page_qubits, Allocator> const& local_state,
            ForwardIterator const d_first,
            BinaryOperation binary_operation, UnaryOperation unary_operation,
            Complex const initial_value, yampi::environment const& environment)
          -> Complex
          {
            using transform_inclusive_scan_type = ::ket::mpi::state_detail::transform_inclusive_scan<has_page_qubits>;
            return transform_inclusive_scan_type::call(
              parallel_policy,
              local_state, d_first, binary_operation, unary_operation, initial_value, environment);
          }
        }; // struct transform_inclusive_scan< ::ket::mpi::state<Complex, has_page_qubits, Allocator> >

        template <typename LocalState_>
        struct transform_inclusive_scan_self;

        template <typename Complex, bool has_page_qubits, typename Allocator>
        struct transform_inclusive_scan_self<
          ::ket::mpi::state<Complex, has_page_qubits, Allocator> >
        {
          template <
            typename ParallelPolicy,
            typename BinaryOperation, typename UnaryOperation>
          static auto call(
            ParallelPolicy const parallel_policy,
            ::ket::mpi::state<Complex, has_page_qubits, Allocator>& local_state,
            BinaryOperation binary_operation, UnaryOperation unary_operation,
            yampi::environment const& environment)
          -> Complex
          {
            using transform_inclusive_scan_self_type = ::ket::mpi::state_detail::transform_inclusive_scan_self<has_page_qubits>;
            return transform_inclusive_scan_self_type::call(
              parallel_policy,
              local_state, binary_operation, unary_operation, environment);
          }

          template <
            typename ParallelPolicy,
            typename BinaryOperation, typename UnaryOperation>
          static auto call(
            ParallelPolicy const parallel_policy,
            ::ket::mpi::state<Complex, has_page_qubits, Allocator>& local_state,
            BinaryOperation binary_operation, UnaryOperation unary_operation,
            Complex const initial_value, yampi::environment const& environment)
          -> Complex
          {
            using transform_inclusive_scan_self_type = ::ket::mpi::state_detail::transform_inclusive_scan_self<has_page_qubits>;
            return transform_inclusive_scan_self_type::call(
              parallel_policy,
              local_state, binary_operation, unary_operation, initial_value, environment);
          }
        }; // struct transform_inclusive_scan_self< ::ket::mpi::state<Complex, has_page_qubits, Allocator> >

        template <typename LocalState_>
        struct upper_bound;

        template <typename Complex, bool has_page_qubits, typename Allocator>
        struct upper_bound< ::ket::mpi::state<Complex, has_page_qubits, Allocator> >
        {
          template <typename Compare>
          static auto call(
            ::ket::mpi::state<Complex, has_page_qubits, Allocator> const& local_state,
            Complex const& value, Compare compare, yampi::environment const& environment)
          -> typename ::ket::mpi::state<Complex, has_page_qubits, Allocator>::difference_type
          {
            using upper_bound_type = ::ket::mpi::state_detail::upper_bound<has_page_qubits>;
            return upper_bound_type::call(local_state, value, compare, environment);
          }
        }; // struct upper_bound< ::ket::mpi::state<Complex, has_page_qubits, Allocator> >
      } // namespace dispatch
    } // namespace utility
  } // namespace mpi
} // namespace ket


# undef KET_is_nothrow_swappable

#endif // KET_MPI_STATE_HPP
